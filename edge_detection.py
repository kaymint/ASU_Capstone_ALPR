__author__ = 'StreetHustling'

#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('images/test_001.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])




ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[4]
cv2.drawContours(imgray, [cnt], 0, (0,255,0), 3)


plt.show()

