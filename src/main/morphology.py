from sklearn.cluster import estimate_bandwidth, MeanShift

from src.load_dataset import load_all

import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

import multiprocessing

import cv2 as cv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

kernel = np.ones((3, 3), np.uint8)


def plot_image(temp_image):
    plt.close()
    plt.imshow(temp_image)
    plt.show()


def get_speed(temp_df: pd.DataFrame):
    temp_df["speed_x"] = np.abs(temp_df['x'] - temp_df['x'].shift(1))
    temp_df["speed_y"] = np.abs(temp_df['y'] - temp_df['y'].shift(1))
    return temp_df[["speed_x", "speed_y"]].dropna()


def get_speed_mean(temp_df: pd.DataFrame):
    temp_df["mean_speed_x"] = temp_df['speed_x'].rolling(10).sum()
    temp_df["mean_speed_y"] = temp_df['speed_y'].rolling(10).sum()
    return temp_df[["mean_speed_x", "mean_speed_y"]].dropna().mean()


def get_binary_image(temp_image):
    # filter to reduce noise
    temp_image = cv.medianBlur(temp_image, 3)
    flat_image = temp_image.reshape((-1, 1))
    flat_image = np.float32(flat_image)

    # mean shift
    bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled = ms.labels_

    segments = np.unique(labeled)
    total = np.zeros((segments.shape[0], 1), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for segment in segments:
        this_segment_index = labeled == segment
        total[segment] = labeled[this_segment_index].sum()
        count[segment] = this_segment_index.sum()
    avg = total / count
    avg = np.uint8(avg)

    # select human parts
    human = avg < 70
    avg[human] = 255
    avg[~human] = 0
    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    temp_result = res.reshape(temp_image.shape)
    temp_result = cv.dilate(temp_result, kernel, iterations=1)
    return temp_result


def generate_position(video_number, video_data):
    print("# progress: " + str(video_number + 1) + '/' + str(total_videos))
    temp_video, category = video_data
    human_positions_df = pd.DataFrame()
    for temp_image in temp_video:
        temp_image = get_binary_image(temp_image)
        x, y = np.where(temp_image == 255)
        human_points_df = pd.DataFrame({"x": x, "y": y})
        human_position = human_points_df.mean()
        if x.shape[0] != 0:
            height = x.max() - x.min()
            width = y.max() - y.min()
            human_position["height"] = height
            human_position["width"] = width
            human_position["percent"] = x.shape[0] / (temp_image.shape[0] * temp_image.shape[1])
        human_positions_df = human_positions_df.append(human_position, ignore_index=True)
    human_positions_df["video"] = video_number
    human_positions_df["category"] = category
    return human_positions_df


def generate_momentum(temp_df: pd.DataFrame):
    temp_df["speed_x"] = np.abs(temp_df['x'] - temp_df['x'].shift(1)) / 120
    temp_df["speed_y"] = np.abs(temp_df['y'] - temp_df['y'].shift(1)) / 160
    temp_df["speed"] = np.sqrt(np.square(temp_df["speed_y"]) + np.square(temp_df["speed_x"]))
    temp_df["momentum"] = temp_df["speed"] * temp_df["percent"] * 100
    return temp_df["momentum"]


def generate_features_df(humans_positions_df):
    feature_df = pd.DataFrame(index=pd.Series(np.arange(dataset.shape[0]), name="video"))

    moving_human = humans_positions_df.groupby("video").apply(lambda temp: temp[["x", "y"]].isna().any().any())
    moving_human.name = "motion"

    human_speed = humans_positions_df.dropna().groupby(['video', 'category']).apply(get_speed)
    human_speed = human_speed.dropna().groupby(['video', 'category']).apply(get_speed_mean)

    momentum = humans_positions_df[['x', 'y', 'percent'] + ['video', 'category']].groupby(['video', 'category']).apply(
        generate_momentum)
    humans_positions_df = humans_positions_df.reset_index().set_index(['video', 'category', "index"]).join(
        momentum.reset_index().rename(columns={"level_2": "index"}).set_index(['video', 'category', "index"]))
    humans_positions_df = humans_positions_df.reset_index().drop(columns=["index"])

    statistical_features = humans_positions_df.drop(columns=['x', 'y']).groupby(['video', 'category']).agg(
        [np.mean, np.min, np.max, np.std, np.var])
    other_statistical_features = humans_positions_df.drop(columns=['x', 'y']).dropna().groupby(
        ['video', 'category']).agg([skew, kurtosis])

    feature_df = feature_df.join(moving_human, how="outer")
    feature_df = feature_df.join(human_speed, how="outer")
    feature_df = feature_df.join(statistical_features, how="outer")
    feature_df = feature_df.join(other_statistical_features, how="outer")
    return feature_df


if __name__ == '__main__':
    dataset, label = load_all()
    total_videos = dataset.shape[0]
    print("data loaded")

    result = Parallel(n_jobs=int(multiprocessing.cpu_count() - 2))(
        delayed(generate_position)(count, video_data_label) for count, video_data_label in
        enumerate(zip(dataset, label)))
    temp_humans_positions_df = pd.concat(result)

    print("position extracted")

    temp_feature_df = generate_features_df(temp_humans_positions_df)

    print("features generated")

    temp_feature_df.to_csv("../../dataset/main_final_features.csv")
