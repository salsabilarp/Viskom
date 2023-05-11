import cv2 as cv

capture = cv.VideoCapture('data/mouse.mp4')
if not capture.isOpened():
    exit(0)
# subsMog2 = cv.createBackgroundSubtractorMOG2(600,125,False)
subsMog2 = cv.createBackgroundSubtractorMOG2()
subsKNN = cv.createBackgroundSubtractorKNN()
i = 0
while (capture.isOpened()):
    re, frame = capture.read()
    scale = 20
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    dim = (width, height)
    image = cv.resize(frame,dim, cv.INTER_AREA)

    if (i==0):
        frame1 = image
        grayscaleframe1 = cv.cvtColor(frame1, cv.COLOR_RGBA2GRAY)

    blobMog = subsMog2.apply(image)
    blobKNN = subsKNN.apply(image)
    cv.imshow("image asli",image)
    cv.imshow("image mog",blobMog)
    cv.imshow("image knn",blobKNN)
    keyword = cv.waitKey(30)
    if keyword=='q' or keyword==27:
        break
cv.destroyAllWindows()