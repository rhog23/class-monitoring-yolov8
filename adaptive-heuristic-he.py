import cv2
import numpy as np

img = cv2.imread(
    "./data/20240328214035-20240328215300/Camera1_MESIN-04_MESIN-04_20240328214035_20240328215300_742536.jpg"
)


def calc_hist(img):
    hist = cv2.calcHist(
        [img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    return hist


def calc_pdf(img, hist):
    h, w = img.shape[:2]

    return hist / (h * w)


hist = calc_hist(img)
pdf = calc_pdf(img, hist)
print(max_hist, mean_hist)
result = np.hstack([img])

cv2.imshow("result", result)
cv2.waitKey(0)
