import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

from src.load_dataset import load_all


def plot_image(temp_image):
    plt.close()
    plt.imshow(temp_image)
    plt.show()


dataset, label = load_all()
img = dataset[0][45]
plot_image(img)
# filter to reduce noise
img = cv.medianBlur(img, 3)
plot_image(img)
# flatten the image
flat_image = img.reshape((-1, 1))
flat_image = np.float32(flat_image)

# meanshift
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# get number of segments
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# get the average color of each segment
total = np.zeros((segments.shape[0], 1), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total / count
avg = np.uint8(avg)

print(avg)
human = avg < 70
avg[human] = 255
avg[~human] = 0
# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape(img.shape)
kernel = np.ones((3, 3), np.uint8)
result = cv.dilate(result, kernel, iterations=1)

plot_image(result)
