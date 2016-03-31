__author__ = 'StreetHustling'

#import necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt

small = 0
image = cv2.imread('images/test6.jpg')

height, width, channels = image.shape

print("Width" + str(width))
print("Height" + str(height))

small2 = cv2.imread('images/test6.jpg', 0)

if width >= 1000 and height >= 1000:
    small = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
    small2 = cv2.resize(small2, (0,0), fx=0.2, fy=0.2)
else:
    small = image



height, width, channels = small.shape

print("Width" + str(width))
print("Height" + str(height))

small2 = cv2.bilateralFilter(small2, 11, 17, 17)


# image binarization
ret,thresh1 = cv2.threshold(small2, 60,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(small2, 60,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret,thresh3 = cv2.threshold(small2, 60,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
# ret,thresh4 = cv2.threshold(dst, 100,255,cv2.THRESH_TOZERO)

thresh4 = cv2.adaptiveThreshold(thresh3,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

# ret,thresh5 = cv2.threshold(dst, 100,255,cv2.THRESH_TOZERO_INV)

kernel = np.ones((6,6),np.float32)/25
dst = cv2.filter2D(small2,-1,kernel)
thresh5 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)


thresh = ['small', 'thresh1', 'thresh2', 'thresh3', 'thresh4', 'thresh5']

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(eval(thresh[i]),'gray')
    plt.title(thresh[i])

plt.show()

# lower = np.array([160,160,160])
# upper = np.array([255,255,255])
#
# shapeMask = cv2.inRange(small, lower, upper)


# find the contours in the mask
# (cnts, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)

(cnts, _) = cv2.findContours(thresh3.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
print "I found %d black shapes" % (len(cnts))
cv2.imshow("Mask", thresh1)

# print cnts

# # loop over the contours
# for c in cnts:
# 	# draw the contour and show it
# 	cv2.drawContours(small, [c], -1, (255, 0, 0), 2)
# 	cv2.imshow("Image", small)
# 	cv2.waitKey(0)

numplate = None
contours = []

# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)

	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		contours.append(approx)
        print("printing approx")
        print(contours)

print("List of Square Contours")
print(contours)

contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

#Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
numplate=contours[max_index]


rect3 = cv2.minAreaRect(numplate)
box3 = cv2.cv.BoxPoints(rect3)
box3 = np.int0(box3)
cv2.drawContours(small,[box3],0,(0,255,255),2)
cv2.imshow("Image", small)
cv2.waitKey(0)

x,y,w,h = cv2.boundingRect(numplate)

cropped_img = small2[y:y+h, x:x+w]
cv2.imshow("Game Boy Screen", cropped_img)
cv2.waitKey(0)

ret,thresh6 = cv2.threshold(cropped_img, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh6 = cv2.bilateralFilter(thresh6, 11, 17, 17)

edges1 = cv2.Canny(thresh6,100,200)

(cnts2, _) = cv2.findContours(edges1.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[:10]

# print(cnts2)

for c in cnts2:
    rect4 = cv2.minAreaRect(c)
    box4 = cv2.cv.BoxPoints(rect4)
    box4 = np.int0(box4)
    cv2.drawContours(cropped_img,[box4],0,(0,255,255),2)

    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(cropped_img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Game Boy Screen", cropped_img)
    cv2.waitKey(0)


cv2.rectangle(thresh6,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Game Boy Screen", cropped_img)
cv2.waitKey(0)

gray = cv2.bilateralFilter(thresh1, 20, 50, 50)
edges = cv2.Canny(gray,100,200)
(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

screenCnt = None

# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)

	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# print(screenCnt)
cv2.drawContours(edges, [screenCnt], -1, (0, 255, 255), 3)
cv2.imshow("Game Boy Screen", edges)
cv2.waitKey(0)




