__author__ = 'StreetHustling'

#import necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import Reordering

#resize image
def resizeInputImg(image):
    height, width, channels = image.shape
    if width >= 1000 and height >= 1000:
        image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
        return image
    else:
        return image


#Get the largest contour of a given set
def getLargestContour(cnts):
    #Find the index of the largest contour
    cnt_areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(cnt_areas)
    max_Cnt=cnts[max_index]
    return max_Cnt

#get minimum rect of a contour
def getMinRectArea(c):
    #find the area of minimum rectangle
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    return box

#save characters in sorted order
def saveInOrder(matrix, img):
    print("test" +str(matrix))
    i = 0
    for m in matrix:
        x = m[0]
        y = m[1]
        h = m[3]
        w = m[2]
        seg = img[y-3:y+h+3, x-3:x+w+3]
        i = i + 1
        cv2.imwrite('segments/'+str(i)+'.jpg',seg)
        img2 = Image.open('segments/'+str(i)+'.jpg')
        img2.save('segments/i'+str(i)+'.jpg', dpi=(600,600))
    return i

#resize segmented character
def resize(image, height=50, width=30):
    res = cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
    return res

#clean character with a series of morphological operations
def cleanSegment(image):

    kernel = np.ones((3,3),np.uint8)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((2,2),np.uint8)

    final = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return final

#read characters and stack them together to form one image
def readChars(characterStack):
    numStack = []
    for i in range(1, characterStack+1):
        char  = cv2.imread('segments/'+str(i)+'.jpg',0)
        char = resize(char)
        char = cleanSegment(char)
        numStack.append(char)
    totalPlate = np.hstack(numStack)
    return totalPlate


def getNumPlate(contours, img):
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        aspect_ratio = float (w) / h

        if cv2.contourArea(cnt) < 60000:
            if 1.8 < aspect_ratio < 5.1 :

                numplateImg = img[y:y+h, x:x+w]
                hist = cv2.calcHist([numplateImg],[0],None,[256],[0,256])
                ratio = float (hist[255]) / hist[0]

                if ratio > 1.25:
                    return numplateImg
                else:
                    return False
            else:
                return False
        else:
            return False



image = cv2.imread('images/4_120.jpg')

smallImg = resizeInputImg(image)

smallImgGrey  = cv2.cvtColor(smallImg,cv2.COLOR_BGR2GRAY)

ret,smallImgGrey = cv2.threshold(smallImgGrey,190,255,0)

smallImgGreyCopy = smallImgGrey.copy()

cv2.imshow('IMG',smallImgGreyCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

#use opening to remove unwanted white spaces or noise
opening_kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(smallImgGreyCopy, cv2.MORPH_OPEN, opening_kernel)


contours, hierarchy = cv2.findContours(opening.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
openingCopy = opening.copy()

openingCopy2 = opening.copy()


cropped_img = getNumPlate(contours, openingCopy2)

if cropped_img is not False:
    cv2.imshow('numplate',cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("segments/fix.jpg", cropped_img)
    os.system('tesseract segments/fix.jpg testFix nobatch digits_and_letters')

        #9. binarize cropped image with simple thresholding and otsu
    ret,inv_thresh = cv2.threshold(cropped_img, 127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret,thresh = cv2.threshold(cropped_img, 127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #10.apply morphological operations
    opening_kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(inv_thresh, cv2.MORPH_OPEN, opening_kernel) #open image to remove small white spaces

    closing = cv2.morphologyEx(inv_thresh, cv2.MORPH_CLOSE, opening_kernel)

    open_and_close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, opening_kernel) #close opened image

    #stack opened and closed image side by side
    morphStack = np.hstack((opening,open_and_close))

    #show stacked images
    cv2.imshow("Opening And Closing And Both", morphStack)
    cv2.waitKey(0)

    #11. detect edges in the image
    edges = cv2.Canny(opening,100,200)
    #12. find contours in edges image
    (edge_cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Edges 1", edges.copy())
    cv2.waitKey(0)

    img = cv2.drawContours(edges.copy(), edge_cnts, -1, (0,255,0), 3)
    edge_morph_stack = np.hstack((edges.copy(), opening))

    #12. find contours in edges image
    (edge_cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #show stack
    # cv2.imshow("Edges", edge_morph_stack)
    # cv2.waitKey(0)

    #13. get largest 20 contours
    edge_cnts = sorted(edge_cnts, key = cv2.contourArea, reverse = True)[:20]
    print(len(edge_cnts))
    print("Edges in contour"+str(edge_cnts))


    #probable segment list
    prob_seg_list = []

    #14. get probable segment list
    for c in edge_cnts:
        x,y,w,h = cv2.boundingRect(c)
        if Reordering.isProbableCharacter(x,y,w,h) == True:
            x,y,w,h = cv2.boundingRect(c)
            c_details = Reordering.getContourDetails(x,y,w,h)
            prob_seg_list.append(c_details)

    print(prob_seg_list)



    print 'prob seg list' + str(prob_seg_list)

    if len(prob_seg_list) != 0:
        totalArea = 0
        count = 0
        for p in prob_seg_list:
            count = count + 1
            area = float (p[2]) * (p[3])
            print("Area "+str(area))
            totalArea = area + totalArea
            print("total Area "+str(totalArea))

        totalAverage = 0
        if count > 0:
            totalAverage = float (totalArea) / count
            print("Average Area "+str(totalAverage))

        for p in prob_seg_list:
            area = float (p[2]) * (p[3])
            if (totalAverage/area) > 3.00:
                print("Removing "+str(area))
                prob_seg_list.remove(p)

        #sort segments into rows
        sorted_seg = Reordering.rowSort(prob_seg_list)

        #get sorted Matrix
        sortedMatrix = Reordering.isOnTop(sorted_seg, prob_seg_list)



        #sorted matrix
        for c in sortedMatrix:
            cv2.rectangle(cropped_img,(c[0],c[1]),(c[0]+c[2],c[1]+c[3]),(0,255,0),2)
            cv2.imshow("Cropped Image", cropped_img)
            cv2.waitKey(0)

        #15. save characters in order
        characterStack = saveInOrder(sortedMatrix, thresh)

        #16. read characters and stack them together
        totalImage = readChars(characterStack)

        #show stack characters
        cv2.imshow("Stacked", totalImage)
        cv2.waitKey(0)

        #17. write final image
        cv2.imwrite('segments/fix.jpg',totalImage)

        #18. recognize characters
        #os.system('tesseract segments/fix.jpg testFix nobatch digits_and_letters')

    else:
        print( " Could not recognize")

else:
    print("Cannot recognize exit")
    exit()

