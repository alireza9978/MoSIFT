import multiprocessing
import os
import random
import math
import cv2 as cv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import util

# params for ShimTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# params for Lucas Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


def capture_frame(video_path, i):
    cap = cv.VideoCapture(video_path)
    cap.set(1, i)
    ret, frame = cap.read()

    return ret, frame


def count_frames(video_path):
    cap = cv.VideoCapture(video_path)
    frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    return frames


def gen_sift_features(frame):
    sift = cv.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(frame, None)

    return key_points, descriptors


def key_points_to_coordinates(key_points):
    key_points_xy = []

    for kp in key_points:
        key_points_xy.append([kp.pt[0], kp.pt[1]])

    return key_points_xy


def has_sufficient_motion(key_points_xy, key_points, descriptors, p1, lambda_value):
    sm_key_points_xy = []
    sm_key_points = []
    sm_descriptors = []

    for i in range(len(key_points)):
        distance_x = key_points_xy[i].T[0] - p1[i].T[0]
        distance_y = key_points_xy[i].T[1] - p1[i].T[1]

        if distance_x > lambda_value or distance_y > lambda_value:
            sm_key_points_xy.append([int(key_points_xy[i].T[0]), int(key_points_xy[i].T[1])])
            sm_key_points.append(key_points[i])
            sm_descriptors.append(descriptors[i])

    return sm_key_points_xy, sm_key_points, sm_descriptors


def gen_hof(x, y, frame, next_frame):
    hof = []
    histogram = [0, 0, 0, 0, 0, 0, 0, 0]
    neighbors = util.gen_neighbors(x, y)
    neighbors = np.float32(np.array(neighbors)[:, np.newaxis, :])
    p1, st, err = cv.calcOpticalFlowPyrLK(frame, next_frame, neighbors, None, **lk_params)

    for i in range(len(p1)):
        if st[i]:
            dx = (p1[i].T[0] - neighbors[i].T[0])
            dy = -(p1[i].T[1] - neighbors[i].T[1])
            direction = np.arctan(dy / dx)
            if dx < 0 < dy:
                direction = math.pi + direction
            if dx < 0 and dy < 0:
                direction = math.pi + direction
            if dx > 0 > dy:
                direction = (2 * math.pi) + direction
            direction = direction * 180 / math.pi
            histogram = util.gen_arc_histogram(direction, histogram)

        if i in util.cells:
            hof += histogram
            histogram = [0, 0, 0, 0, 0, 0, 0, 0]

    return hof


def gen_mosift_features(video_path, lambda_value, interval, sample_size):
    frames = count_frames(video_path)
    mosift_descriptors = []

    for i in range(1, frames - 1, interval):
        ret, frame = capture_frame(video_path, i)
        ret, next_frame = capture_frame(video_path, i + 1)
        key_points, descriptors = gen_sift_features(frame)
        if len(key_points) == 0:
            continue
        key_points_xy = key_points_to_coordinates(key_points)
        key_points_xy = np.float32(np.array(key_points_xy)[:, np.newaxis, :])
        p1, st, err = cv.calcOpticalFlowPyrLK(frame, next_frame, key_points_xy, None, **lk_params)
        sm_key_points_xy, sm_key_points, sm_descriptors = has_sufficient_motion(key_points_xy, key_points, descriptors,
                                                                                p1, lambda_value)

        for j in range(len(sm_key_points)):
            hof = np.array(gen_hof(sm_key_points_xy[j][0], sm_key_points_xy[j][1], frame, next_frame))
            mosift_descriptors.append(list(np.concatenate((sm_descriptors[j], hof))))

    random_mosift_descriptors = random.sample(mosift_descriptors, k=int(len(mosift_descriptors) * sample_size))

    return random_mosift_descriptors


def run_feature_extractor(input_path, output_path, lambda_value, interval, dict_directory, sample_size):
    listing = os.listdir(input_path)
    progress_count = 0
    for video_name in listing:
        progress_count += 1
        print("# progress: " + str(progress_count) + '/' + str(len(listing)))
        video_path = input_path + video_name

        if dict_directory:
            df_dict = pd.DataFrame(gen_mosift_features(video_path, lambda_value, interval, sample_size))
            df_dict.to_csv(output_path + "dict.csv", mode='a', header=False, index=False)
        else:
            df_mosift_features = pd.DataFrame(gen_mosift_features(video_path, lambda_value, interval, sample_size))
            df_mosift_features.to_csv(output_path + video_name[:-4] + ".csv", mode='a', header=False, index=False)


if __name__ == '__main__':
    paths = [(r"/home/alireza/projects/python/MoSIFT/dataset/KTH/running/",
              r"/home/alireza/projects/python/MoSIFT/dataset/csv/running/"),
             (r"/home/alireza/projects/python/MoSIFT/dataset/KTH/boxing/",
              r"/home/alireza/projects/python/MoSIFT/dataset/csv/boxing/"),
             (r"/home/alireza/projects/python/MoSIFT/dataset/KTH/handclapping/",
              r"/home/alireza/projects/python/MoSIFT/dataset/csv/handclapping/"),
             (r"/home/alireza/projects/python/MoSIFT/dataset/KTH/handwaving/",
              r"/home/alireza/projects/python/MoSIFT/dataset/csv/handwaving/"),
             (r"/home/alireza/projects/python/MoSIFT/dataset/KTH/jogging/",
              r"/home/alireza/projects/python/MoSIFT/dataset/csv/jogging/"),
             (r"/home/alireza/projects/python/MoSIFT/dataset/KTH/walking/",
              r"/home/alireza/projects/python/MoSIFT/dataset/csv/walking/")]

    Parallel(n_jobs=int(multiprocessing.cpu_count()))(
        delayed(run_feature_extractor)(path, target, 0.7, 1, False, 0.2) for path, target in paths)
