from __future__ import print_function
from __future__ import division
import os, logging
import cv2 as cv
import numpy as np

logging.basicConfig(level=logging.INFO)


def loadExposureSeq(path):
    images = []
    times = []

    with open(os.path.join(path, "list.txt")) as f:
        content = f.readlines()  # read contents and return as list
        for line in content:

            # split the line into two parts (image name and time) e.g ['memorial00.png', '0.03125']
            tokens = line.split()

            images.append(
                cv.imread(os.path.join(path, tokens[0]))
            )  # appends image to the images list
            times.append(1 / float(tokens[1]))  # appends exposure time to times list

    return images, np.asarray(times, dtype=np.float32)


images, times = loadExposureSeq("data/tonemapping")

calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)

merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)

tonemap = cv.createTonemap(2.2)
ldr = tonemap.process(hdr)

merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)

results = np.hstack([fusion, ldr, hdr])

cv.imshow("result", results)

cv.waitKey(0)
