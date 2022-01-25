import os

import cv2 as cv
import numpy as np
import pandas as pd

categories = ["boxing", "running"]
root = "/home/alireza/projects/python/MoSIFT/"
base_path = root + "dataset/KTH/"
base_feature_path = root + "dataset/csv/"


def load(category: str):
    dataset = []
    temp_path = base_path + category + "/"
    files = os.listdir(temp_path)
    for file in files[0:5]:
        print(temp_path + file)
        capture = cv.VideoCapture(temp_path + file)
        success, image = capture.read()
        temp_video = []
        while success:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            temp_video.append(image)
            success, image = capture.read()
        temp_video = np.array(temp_video)
        dataset.append(temp_video)

    dataset = np.array(dataset)
    return dataset


def load_feature(category: str):
    dataset = pd.DataFrame()
    temp_path = base_feature_path + category + "/"
    files = os.listdir(temp_path)
    count = 0
    for file in files[0:3]:
        print(temp_path + file)
        temp_df = pd.read_csv(temp_path + file, header=None)
        dataset = dataset.append(temp_df)
        count += 1

    return dataset, count


def load_all():
    dataset = []
    label = []
    for index, temp in enumerate(categories):
        temp_dataset = load(temp)
        dataset.append(temp_dataset)
        label.append(np.array([index] * temp_dataset.shape[0]))

    dataset = np.array(dataset)
    label = np.array(label)
    return dataset, label


def load_all_features():
    dataset = []
    label = []
    for index, temp in enumerate(categories):
        temp_dataset, count = load_feature(temp)
        dataset.append(temp_dataset)
        label.append(np.array([index] * count))

    label = np.array(label)
    return dataset, label


if __name__ == '__main__':
    # load_all()
    print(load_all_features())
