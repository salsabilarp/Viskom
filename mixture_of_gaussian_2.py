import cv2 as cv
import numpy as np

capture = cv.VideoCapture('data/mouse.mp4')
if not capture.isOpened():
    exit(0)

subsMog2 = cv.createBackgroundSubtractorMOG2()

while capture.isOpened():
    ret, frame = capture.read()
    scale = 20

    if frame is None:
        break

    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    dim = (width, height)
    image = cv.resize(frame, dim, cv.INTER_AREA)

    blobMog = subsMog2.apply(image)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    morphImage = cv.morphologyEx(blobMog, cv.MORPH_CLOSE, kernel)
    morphImage = cv.morphologyEx(morphImage, cv.MORPH_DILATE, kernel)

    cv.imshow("Original Image", image)
    cv.imshow("MOG2", blobMog)
    cv.imshow("Morphological Image", morphImage)

    keyword = cv.waitKey(30)
    if keyword == ord('q') or keyword == 27:
        break

cv.destroyAllWindows()
exit(0)
