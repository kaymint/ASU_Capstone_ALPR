__author__ = 'StreetHustling'


__author__ = 'StreetHustling'

from PIL import Image
import pytesseract
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

img = cv2.imread('images/test_001.jpg', 0)
img1 = cv2.imread('images/test_001.jpg')

# img = cv2.imread('images/test7.jpg', 0)
# img1 = cv2.imread('images/test7.jpg')


#image smoothening
kernel = np.ones((2,2),np.float32)/4
dst = cv2.filter2D(img,-1,kernel)

# image binarization
ret,thresh1 = cv2.threshold(dst, 127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(dst, 127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(dst, 100,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(dst, 100,255,cv2.THRESH_TOZERO)

thresh4 = cv2.adaptiveThreshold(thresh1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

# ret,thresh5 = cv2.threshold(dst, 100,255,cv2.THRESH_TOZERO_INV)

kernel = np.ones((6,6),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
thresh5 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)


thresh = ['img', 'thresh1', 'thresh2', 'thresh3', 'thresh4', 'thresh5']

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(eval(thresh[i]),'gray')
    plt.title(thresh[i])

plt.show()

edges = cv2.Canny(thresh3,100,200)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

plt.subplot(111),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

rect = cv2.minAreaRect(cnt)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(img1,[box],0,(0,0,255),2)

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Show",img1)
cv2.waitKey()


cropped_img = img[y:y+h, x:x+w]
# gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("cropped", cropped_img)
# cv2.waitKey(0);


# ret,cropped_thresh = cv2.threshold(cropped_img, 190,255,cv2.THRESH_BINARY)

cropped_thresh = cv2.adaptiveThreshold(cropped_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

#

# kernel = np.ones((2,2),np.float32)/4
# cropped_thresh = cv2.filter2D(cropped_thresh,-1,kernel)

cropped_thresh = cv2.fastNlMeansDenoising(cropped_thresh,12,12,8,22)

ret,cropped_thresh = cv2.threshold(cropped_thresh, 127,255,cv2.THRESH_BINARY)

cv2.pyrUp(cropped_thresh)
cv2.imshow("cropped2", cropped_thresh)
cv2.waitKey(0);

cv2.imwrite('images/plate_test1.jpg',cropped_thresh)
os.system('tesseract images/plate_test1.jpg out2')


img2 = Image.open('images/plate_test1.jpg')
img2.save("images/plate_test2.jpg", dpi=(3000,3000))

print(pytesseract.image_to_string(Image.open('images/plate_test2.jpg')))


