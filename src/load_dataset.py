import os

import cv2 as cv
import numpy as np

categories = ["boxing", "running"]
base_path = "../dataset/KTH/"


def load(category: str):
    dataset = []
    temp_path = base_path + category + "/"
    files = os.listdir(temp_path)
    for file in files:
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


def load_all():
    dataset = []
    label = []
    for index, temp in enumerate(categories):
        temp_dataset = load(temp)
        dataset.append(temp_dataset)
        label.append(np.full_like(temp_dataset.shape[0], index))

    dataset = np.array(dataset)
    label = np.array(label)
    return dataset, label


if __name__ == '__main__':
    load_all()
