__author__ = 'StreetHustling'

from PIL import Image
import pytesseract
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#
# print flags

#Color tracking
cap = cv2.VideoCapture(0);

while(1):

    # Take each frame
    _, frame = cap.read();

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
