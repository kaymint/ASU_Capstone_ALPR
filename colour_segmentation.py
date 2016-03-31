__author__ = 'StreetHustling'

#import necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import os


#resize image
def resizeInputImg(image):
    height, width, channels = image.shape
    if width >= 1000 and height >= 1000:
        image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
        return image
    else:
        return image

#1. read image
image = cv2.imread('images/md.jpg')

height, width, channels = image.shape

print("Width" + str(width))
print("Height" + str(height))



#2. resize image
smallImg = resizeInputImg(image)

#3. covert to grayscale
smallImgGrey = cv2.cvtColor(smallImg, cv2.COLOR_RGB2GRAY)

#smoothing image
def smoothing(img):
    #image smoothening
    kernel = np.ones((2,2),np.float32)/4
    dst = cv2.filter2D(img,-1,kernel)
    return dst

# plt.hist(smallImgGrey.ravel(),256,[0,256]); plt.show()
# hist = cv2.calcHist([smallImgGrey],[0],None,[256],[0,256])

# equ = cv2.equalizeHist(smallImgGrey)
# res = np.hstack((smallImgGrey,equ)) #stacking images side-by-side
# cv2.imshow("Equalized", res)

# plt.hist(equ.ravel(),256,[0,256]); plt.show()

height, width, channels = smallImg.shape

print("Width" + str(width))
print("Height" + str(height))

#place shape mask on white
lower = np.array([80,90,80])
upper = np.array([255,255,255])

shapeMask = cv2.inRange(smallImg, lower, upper)

cv2.imshow("Shape Mask before thresh", shapeMask)
cv2.waitKey(0)

# ret,shapeMask = cv2.threshold(smallImgGrey, 200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

(cnts, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

# shapeMask = cv2.bilateralFilter(shapeMask, 20, 50, 50)

cv2.imshow("Shape Mask after thresh image", shapeMask)
cv2.waitKey(0)

def getLargestContour(cnts):
    #Find the index of the largest contour
    cnt_areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(cnt_areas)
    max_Cnt=cnts[max_index]
    return max_Cnt


def getMinRectArea(c):
    #find the area of minimum rectangle
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    return box

#Find the index of the largest contour
areas = [cv2.contourArea(c) for c in cnts]
max_index = np.argmax(areas)
numplate=cnts[max_index]

#find the area of minimum rectangle
rect3 = cv2.minAreaRect(numplate)
box3 = cv2.cv.BoxPoints(rect3)
box3 = np.int0(box3)
cv2.drawContours(smallImg,[box3],0,(0,255,255),2)

#find the bounding rectangle of the largest contour
x,y,w,h = cv2.boundingRect(numplate)
cropped_img = smallImgGrey[y:y+h, x:x+w]


cv2.imshow("Image", smallImg)
cv2.waitKey(0)

ret,thresh2 = cv2.threshold(cropped_img, 127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret,thresh3 = cv2.threshold(cropped_img, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((4,4),np.uint8)
erosion = cv2.erode(thresh2,kernel,iterations = 2)

dilation = cv2.dilate(thresh2,kernel,iterations = 1)

opening = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

final = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

res1 = np.hstack((final, closing))

cv2.imshow("Stacked", res1)
cv2.waitKey(0)

edges = cv2.Canny(closing,100,200)
res1 = np.hstack((edges, closing))

(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Edges", res1)
cv2.waitKey(0)



cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]

i = 0

print 'edges' + str(len(cnts))

stack = []

cv2.imwrite('segments/cropped.jpg',final)
os.system('tesseract segments/cropped.jpg out5 nobatch digits_and_letters')


def orderMatrix(x,y, w, h):
    inOrder = []
    inOrder.append(x)
    inOrder.append(y)
    inOrder.append(w)
    inOrder.append(h)
    return inOrder

matrix = []

#filter contours accoring to size and aspect ratio
for c in cnts:
    #find minimum area of bounding rectangle
    rect4 = cv2.minAreaRect(c)
    box4 = cv2.cv.BoxPoints(rect4)
    box4 = np.int0(box4)

    x,y,w,h = cv2.boundingRect(c)
    ratio = float (h) / w
    area = float (w)*h

    cv2.rectangle(cropped_img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Cropped Image", cropped_img)
    cv2.waitKey(0)

    print 'ratio '+str(ratio)

    #filter according to aspect ratio
    if ratio >= 1.5 and ratio <= 1.8 :
        #filter according to area
        if area > 50:
            i = i + 1
            print("contour area "+ str(area))
            stack.append(i)
            cv2.rectangle(cropped_img,(x,y),(x+w,y+h),(0,255,0),2)
            matrix.append(orderMatrix(x,y,w,h))
            cv2.imshow("Cropped Image", cropped_img)
            cv2.waitKey(0)

plate = ''

print(matrix)

def inlineWith(matrix, count):
    if count > 0:
        sameline = 'inline with '+str(count-1) if matrix[count][1] >= matrix[count-1][1] - 4 and matrix[count][1] <= matrix[count-1][1] + 4 else 'nextline'
        return sameline
    elif count == 0:
        return 'first'
    else:
        return 'last'

allIn = []
def allInlineWith(matrix):
    count = 0
    for c in matrix:
        sameStreet = []
        for i in range(0, len(matrix)):
            if c[1] >= matrix[i][1] - 4 and c[1] <= matrix[i][1] + 4:
                sameStreet.append(i)

        # print("for: "+str(c)+" on sameline: "+ str(sameStreet))

# allInlineWith(matrix)


def justInline(matrix, c):

    sameStreet = []
    for i in range(0, len(matrix)):
        if c[1] >= matrix[i][1] - 4 and c[1] <= matrix[i][1] + 4:
            sameStreet.append(i)
    return sameStreet



def streetCount(matrix):
    count = 0
    street = []
    sameStreet = []
    sameStreet2 = []
    for i in range(0, len(matrix)):
        if matrix[0][1] >= matrix[i][1] - 4 and matrix[0][1] <= matrix[i][1] + 4:
            if i not  in sameStreet:
                sameStreet.append(i)
        elif i not in sameStreet2:
            sameStreet2.append(i)
    street.append(sameStreet)
    street.append(sameStreet2)
    return street


streets = streetCount(matrix)

def orderStreet(street, matrix):
    streetOrder = []
    for s in street:
        streetOrder.append(matrix[s])
    streetOrder = sorted(streetOrder, key=lambda x: x[0])
    return streetOrder


def isOnTop(streets, matrix):
    matrix2 = []
    if len(streets) >= 2 and len(streets[1]) > 0:
        index = streets[0][0]
        index2 = streets[1][0]
        if matrix[index][1] < matrix[index2][1]:
            print("Street 1 on top of street 2")
            s1 = orderStreet(streets[0], matrix)
            s2 = orderStreet(streets[1], matrix)
            matrix2.extend(s1)
            matrix2.extend(s2)
            return matrix2
        else:
            print("Street 2 on top of street 1")
            s1 = orderStreet(streets[1], matrix)
            s2 = orderStreet(streets[0], matrix)
            matrix2.extend(s1)
            matrix2.extend(s2)
            return matrix2
    else:
        print("All on one street")
        s1 = orderStreet(streets[0], matrix)
        matrix2.extend(s1)
        return matrix2


#get sorted Matrix
sortedMatrix = isOnTop(streets, matrix)


def saveInOrder(matrix):
    print("test" +str(matrix))
    i = 0
    for m in matrix:
        x = m[0]
        y = m[1]
        h = m[3]
        w = m[2]
        seg = thresh3[y-4:y+h+4, x-4:x+w+4]
        i = i + 1
        cv2.imwrite('segments/'+str(i)+'.jpg',seg)
        img2 = Image.open('segments/'+str(i)+'.jpg')
        img2.save('segments/i'+str(i)+'.jpg', dpi=(600,600))
    return i

def fix(image):
    kernel = np.ones((2,2),np.uint8)

    erosion = cv2.erode(image,kernel,iterations = 2)

    kernel = np.ones((2,2),np.uint8)

    dilation = cv2.dilate(erosion,kernel,iterations = 1)

    kernel = np.ones((3,3),np.uint8)

    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((5,5),np.uint8)

    final = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return final

#resize segmented character
def resize(image, height=50, width=30):
    res = cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
    return res

def readChars(characterStack):
    numStack = []
    for i in range(1, characterStack+1):
        char  = cv2.imread('segments/'+str(i)+'.jpg',0)
        char = resize(char)
        char = fix(char)
        numStack.append(char)
    totalPlate = np.hstack(numStack)
    return totalPlate


characterStack = saveInOrder(sortedMatrix)

totalImage = readChars(characterStack)

cv2.imshow("Stacked", totalImage)
cv2.waitKey(0)

cv2.imwrite('segments/fix.jpg',totalImage)

os.system('tesseract segments/fix.jpg testFix nobatch digits_and_letters')

count = 0
for c in matrix:
    print("x: "+str(c[0])+ " y: "+str(c[1])+ " sum: "+str(c[0]+c[1]) + " inline: "+ str(justInline(matrix,c)))
    count = count + 1


kernel = np.ones((2,2),np.uint8)
testImage = cv2.imread('segments/i3.jpg',0)

ret,testImage = cv2.threshold(testImage, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

testImage = cv2.erode(testImage,kernel,iterations = 3)
cv2.imshow("Test Image", testImage)
cv2.waitKey(0)

kernel = np.ones((3,3),np.uint8)
testImage = cv2.dilate(testImage,kernel,iterations = 1)
cv2.imshow("Dilation Image", testImage)
cv2.waitKey(0)

#image smoothening
kernel = np.ones((2,2),np.float32)/4
testImage = cv2.filter2D(testImage,-1,kernel)

testImage = cv2.morphologyEx(testImage, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Test Image", testImage)
cv2.waitKey(0)

print("number is: " + pytesseract.image_to_string(Image.open("segments/fix.jpg")))

cv2.imwrite('segments/i3.jpg',testImage)

os.system('tesseract segments/i3.jpg out6 nobatch digits_and_letters')