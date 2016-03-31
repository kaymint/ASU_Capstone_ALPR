__author__ = 'StreetHustling'


#import necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import os


def fix(image):
    kernel = np.ones((3,3),np.uint8)

    dilation = cv2.dilate(image,kernel,iterations = 1)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((5,5),np.uint8)

    final = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return final


def resize(image, height, width):
    res = cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
    return res


image  = cv2.imread('segments/1.jpg',0)
image2  = cv2.imread('segments/2.jpg',0)
image3  = cv2.imread('segments/3.jpg',0)
image4  = cv2.imread('segments/4.jpg',0)

height, width = image.shape[:2]
height = 50
width = 30
print height
print(width)

image = resize(image, height, width)
image2 = resize(image2, height, width)
image3 = resize(image3, height, width)
image4 = resize(image4, height, width)



final = fix(image)
fix2 = fix(image2)
fix3 = fix(image3)
fix4 = fix(image4)


res1 = np.hstack((final, fix2, fix3, fix4))

cv2.imshow("Stacked", res1)
cv2.waitKey(0)

cv2.imwrite('segments/fix.jpg',res1)

os.system('tesseract segments/fix.jpg testFix nobatch digits_and_letters')