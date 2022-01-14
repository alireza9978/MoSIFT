import cv2 as cv
import numpy as np

from src.load_dataset import load_all

# params for corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

dataset, label = load_all()

old_frame = dataset[0][0][0]
points_to_track = cv.goodFeaturesToTrack(old_frame, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
# Create some random colors for drawing purposes
color = np.random.randint(0, 255, (100, 3))

for index, frame in enumerate(dataset[0][0][1:]):
    # calculate optical flow
    points_to_track_destination, status, err = cv.calcOpticalFlowPyrLK(old_frame,
                                                                       frame,
                                                                       points_to_track,
                                                                       None,
                                                                       **lk_params)

    # draw the tracks
    for i, (new, old) in enumerate(zip(points_to_track_destination, points_to_track)):
        a, b = new.ravel()
        c, d = old.ravel()
        pt1 = (int(np.round(a)), int(np.round(b)))
        pt2 = (int(np.round(c)), int(np.round(d)))
        mask = cv.line(mask, pt1, pt2, color[i].tolist(), 2)

    if index % 10 == 0:
        img = cv.add(old_frame, mask)
        cv.imwrite(f'./op_out/optical_flow_{index}.jpg', img)
        mask = np.zeros_like(old_frame)

    points_to_track = points_to_track_destination[status == 1].reshape(-1, 1, 2)
    old_frame = frame
