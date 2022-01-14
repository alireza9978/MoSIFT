import cv2 as cv

from src.load_dataset import load_all

dataset, label = load_all()

# category --> film --> frame
img = dataset[0][0][0]
gray = img

sift = cv.SIFT_create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img)
cv.imwrite('./sift_keypoints.jpg', img)

# img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('sift_keypoints.jpg', img)
