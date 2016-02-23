__author__ = 'Kenneth Mintah Mensah'

from PIL import Image
import pytesseract
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# img = cv2.imread('images/test_001.jpg', 0)
# img1 = cv2.imread('images/test_001.jpg')

img = cv2.imread('images/bogart.JPG', 0)
img1 = cv2.imread('images/bogart.JPG')

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

rect = cv2.minAreaRect(cnt)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(img1,[box],0,(0,0,255),2)
cv2.drawContours(img1,contours,0,(0,0,255),1)



x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imshow("Show",img1)
# cv2.waitKey()

plt.subplot(111),plt.imshow(img1)
plt.title('Detected Image'), plt.xticks([]), plt.yticks([])

plt.show()


cropped_img = img[y:y+h, x:x+w]
# gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("cropped", cropped_img)
# cv2.waitKey(0);


ret,cropped_thresh = cv2.threshold(cropped_img, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# cropped_thresh = cv2.adaptiveThreshold(cropped_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

#

# kernel = np.ones((2,2),np.float32)/4
# cropped_thresh = cv2.filter2D(cropped_thresh,-1,kernel)

cropped_thresh = cv2.fastNlMeansDenoising(cropped_thresh,12,12,8,22)

# ret,cropped_thresh = cv2.threshold(cropped_thresh, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.pyrUp(cropped_thresh)
# cv2.imshow("cropped2", cropped_thresh)
# cv2.waitKey(0)

plt.subplot(111),plt.imshow(cropped_thresh)
plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])

plt.show()

SZ=20000
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


# cells = [np.hsplit(row,100) for row in np.vsplit(cropped_thresh,50)]
#
# # First half is trainData, remaining is testData
# train_cells = [ i[:50] for i in cells ]
# test_cells = [ i[50:] for i in cells]



def deskew(input_img):
    m = cv2.moments(input_img)
    if abs(m['mu02']) < 1e-2:
        return input_img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(input_img,M,(SZ, SZ),flags=affine_flags)
    return img


# deskewed = [map(deskew,row) for row in train_cells]

# deskewed = deskew(cropped_thresh);
#
# cv2.imshow("deskewed", deskewed)


plt.subplot(111),plt.imshow(cropped_thresh)
plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])

plt.show()

res = cv2.resize(cropped_thresh,None,fx=.2, fy=.2, interpolation = cv2.INTER_CUBIC)

cv2.imwrite('images/plate_test1.jpg',res)
# cv2.imwrite('images/plate_test1.jpg',thresh1)
os.system('tesseract images/plate_test1.jpg out2 nobatch digits_and_letters')


img2 = Image.open('images/plate_test1.jpg')
img2.save("images/plate_test2.jpg", dpi=(3000,3000))

os.system('tesseract images/plate_test2.jpg out3')

print(pytesseract.image_to_string(Image.open('images/plate_test2.jpg')))

