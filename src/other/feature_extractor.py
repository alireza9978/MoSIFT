import math
import multiprocessing
import random

import cv2 as cv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import util
from src.load_dataset import load_all, get_saving_file_name

# params for Lucas Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


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
            if dx == 0:
                dx = 0.00001
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


def gen_mosift_features(video_data, lambda_value, interval, sample_size):
    frames = video_data.shape[0]
    mosift_descriptors = []

    for i in range(1, frames - 1, interval):
        frame = video_data[i]
        next_frame = video_data[i + 1]

        key_points, descriptors = gen_sift_features(frame)
        if len(key_points) == 0:
            continue
        key_points_xy = cv.KeyPoint_convert(key_points)
        p1, st, err = cv.calcOpticalFlowPyrLK(frame, next_frame, key_points_xy, None, **lk_params)
        sm_key_points_xy, sm_key_points, sm_descriptors = has_sufficient_motion(key_points_xy, key_points, descriptors,
                                                                                p1, lambda_value)

        for j in range(len(sm_key_points)):
            hof = np.array(gen_hof(sm_key_points_xy[j][0], sm_key_points_xy[j][1], frame, next_frame))
            spatio_temporal_info = [sm_key_points_xy[j][0], sm_key_points_xy[j][1], i]
            mosift_descriptors.append(list(np.concatenate((sm_descriptors[j], hof, spatio_temporal_info))))

    random_mosift_descriptors = random.sample(mosift_descriptors, k=int(len(mosift_descriptors) * sample_size))

    return random_mosift_descriptors


def run_feature_extractor():
    lambda_value, interval, sample_size = 0.7, 1, 0.2
    data, label = load_all()
    total_videos = str(label.shape[0])

    def inner_call(count, video_data, video_label):
        print("# progress: " + str(count) + '/' + total_videos)

        df_dict = pd.DataFrame(gen_mosift_features(video_data, lambda_value, interval, sample_size))
        df_dict.to_csv(get_saving_file_name(count, video_label), header=False, index=False)

    Parallel(n_jobs=int(multiprocessing.cpu_count() - 2))(
        delayed(inner_call)(count, video_data_label[0], video_data_label[1]) for count, video_data_label in
        enumerate(zip(data, label)))


if __name__ == '__main__':
    run_feature_extractor()
