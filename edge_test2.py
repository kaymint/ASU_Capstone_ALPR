__author__ = 'StreetHustling'

import cv2
import numpy as np
from matplotlib import pyplot as plt


im = cv2.imread('images/test_001.jpg')
hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
COLOR_MIN = np.array([20, 80, 80],np.uint8)
COLOR_MAX = np.array([40, 255, 255],np.uint8)
frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
imgray = frame_threshed
ret,thresh = cv2.threshold(frame_threshed,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

rect = cv2.minAreaRect(cnt)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(im,[box],0,(0,0,255),2)

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Show",im)
cv2.waitKey()
cv2.destroyAllWindows()