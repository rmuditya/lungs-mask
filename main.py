import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# cv2.imshow("output", thresh)
# cv2.imshow("output", thresh)
# cv2.imshow("unknown", unknown)
# cv2.imshow("sure_fg", sure_fg)
cv2.imshow("original", img)
cv2.imshow("mask", opening)
cv2.waitKey(0)