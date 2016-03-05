__author__ = 'Kenneth Mintah Mensah'

from PIL import Image
import pytesseract
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# img = cv2.imread('images/test_001.jpg', 0)
# img1 = cv2.imread('images/test_001.jpg')

img = cv2.imread('images/prince.JPG', 0)
img1 = cv2.imread('images/prince.JPG')

res = cv2.resize(img,None,fx=.2, fy=.2, interpolation = cv2.INTER_CUBIC)

# cv2.imshow("resize", res)


#image smoothening
kernel = np.ones((2,2),np.float32)/4
dst = cv2.filter2D(img,-1,kernel)

# image binarization
ret,thresh1 = cv2.threshold(dst, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(dst, 127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret,thresh3 = cv2.threshold(dst, 100,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
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

edges = cv2.Canny(thresh1,100,200)

contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

plt.subplot(111),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)
    cv2.drawContours(img,contours,0,(0,0,255),1)


    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# cv2.imshow("Show",img)
# cv2.waitKey()

plt.subplot(111),plt.imshow(img)
plt.title('Detected Image'), plt.xticks([]), plt.yticks([])

plt.show()

cropped_img = img[y:y+h, x:x+w]
# gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("cropped", cropped_img)
# cv2.waitKey(0);


ret,cropped_thresh = cv2.threshold(cropped_img, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# cropped_thresh = cv2.adaptiveThreshold(cropped_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

# kernel = np.ones((2,2),np.float32)/4
# cropped_thresh = cv2.filter2D(cropped_thresh,-1,kernel)

cropped_thresh = cv2.fastNlMeansDenoising(cropped_thresh,12,12,8,22)

# ret,cropped_thresh = cv2.threshold(cropped_thresh, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.pyrUp(cropped_thresh)
# cv2.imshow("cropped2", cropped_thresh)
# cv2.waitKey(0)

plt.subplot(111),plt.imshow(cropped_thresh, cmap = 'gray')
plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])

plt.show()

edges2 = cv2.Canny(cropped_thresh,100,200)

contours2, hierarchy2 = cv2.findContours(edges2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours2]
max_index = np.argmax(areas)
cnt3=contours2[max_index]

rect3 = cv2.minAreaRect(cnt3)
box3 = cv2.cv.BoxPoints(rect3)
box3 = np.int0(box)
cv2.drawContours(cropped_thresh,[box3],0,(0,255,255),2)
cv2.drawContours(cropped_thresh,contours2,0,(0,255,255),1)


x2,y2,w2,h2 = cv2.boundingRect(cnt3)
cv2.rectangle(cropped_thresh,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)

for cnt4 in contours2:
    rect4 = cv2.minAreaRect(cnt4)
    box4 = cv2.cv.BoxPoints(rect4)
    box4 = np.int0(box4)
    cv2.drawContours(cropped_thresh,[box4],0,(0,255,255),2)


    x3,y3,w3,h3 = cv2.boundingRect(cnt4)
    cv2.rectangle(cropped_thresh,(x3,y3),(x3+w3,y3+h3),(0,255,255),2)

plt.subplot(111),plt.imshow(cropped_thresh, cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

res = cv2.resize(cropped_thresh,None,fx=.2, fy=.2, interpolation = cv2.INTER_CUBIC)


cv2.imwrite('images/plate_test1.jpg',res)
# cv2.imwrite('images/plate_test1.jpg',thresh1)
os.system('tesseract images/plate_test1.jpg out2 nobatch digits_and_letters')


img2 = Image.open('images/plate_test1.jpg')
img2.save("images/plate_test2.jpg", ppi=(3000,3000))

os.system('tesseract images/plate_test2.jpg out3')

print(pytesseract.image_to_string(Image.open('images/plate_test2.jpg')))

