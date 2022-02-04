import cv2 as cv
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.svm import TimeSeriesSVC
from src.load_dataset import load_all
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew


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


if __name__ == '__main__':
    dataset, label = load_all()
    kernel = np.ones((4, 4), np.uint8)

    humans_positions_df = pd.DataFrame()
    for index, video in enumerate(dataset):
        human_positions_df = pd.DataFrame()
        for img in video:
            img = cv.inRange(img, 0, 50)
            img = cv.dilate(img, kernel, iterations=1)
            x, y = np.where(img == 255)
            human_points_df = pd.DataFrame({"x": x, "y": y})
            human_position = human_points_df.mean()
            if x.shape[0] != 0:
                height = x.max() - x.min()
                width = y.max() - y.min()
                human_position["height"] = height
                human_position["width"] = width
                human_position["percent"] = x.shape[0] / (img.shape[0] * img.shape[1])
            human_positions_df = human_positions_df.append(human_position, ignore_index=True)
        human_positions_df["video"] = index
        human_positions_df["category"] = label[index]
        humans_positions_df = humans_positions_df.append(human_positions_df)

    feature_df = pd.DataFrame(index=pd.Series(np.arange(dataset.shape[0]), name="video"))

    moving_human = humans_positions_df.groupby("video").apply(lambda temp: temp[["x", "y"]].isna().any().any())
    moving_human.name = "motion"

    human_speed = humans_positions_df.dropna().groupby(['video', 'category']).apply(get_speed)
    human_speed = human_speed.dropna().groupby(['video', 'category']).apply(get_speed_mean)

    statistical_features = humans_positions_df.drop(columns=['x', 'y']).groupby(['video', 'category']).agg(
        [np.mean, np.min, np.max, np.std, np.var])
    other_statistical_features = humans_positions_df.drop(columns=['x', 'y']).dropna().groupby(
        ['video', 'category']).agg([skew, kurtosis])

    feature_df = feature_df.join(moving_human, how="outer")
    feature_df = feature_df.join(human_speed, how="outer")
    feature_df = feature_df.join(statistical_features, how="outer")
    feature_df = feature_df.join(other_statistical_features, how="outer")
    feature_df.to_csv("../../dataset/main_final_features.csv")
