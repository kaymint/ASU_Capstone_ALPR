__author__ = 'StreetHustling'


#this module contains functions for reordering segments

'''
This function filters contours by size and aspect ratio
'''
def isProbableCharacter(x,y,w,h):

        # x,y,w,h = cv2.boundingRect(c)
        ratio = getAspectRatio(w,h)
        area = getBoundingRectArea(w,h)
        # print("Area" + str(area))
        # print("Ratio" + str(ratio))

        #filter according to aspect ratio and area
        if ratio >= 1.5 and ratio <= 1.8:
            # print("Area" + str(area))
            if area > 50:
                return True
        else:
            return False

def getAverageArea(list):

    totalArea = 0
    count = 0
    for x in list:
        totalArea = x + totalArea
        count = count + 1
    if len(list) < 0:
        return totalArea/count
    else:
        return 0


'''
get the aspect ratio of bounding rectangle
'''
def getAspectRatio(w,h):
    return float (h) / w

'''
get the area of bounding rectangle
'''
def getBoundingRectArea(w, h):
    return float (w)*h

'''
This function save the details of the bounding rectangle of segmented characters
'''
def getContourDetails(x,y, w, h):
    contourDetails = []
    contourDetails.append(x)  # x-coordinates of bounding rectangle
    contourDetails.append(y)  # y-coordinates of bounding rectangle
    contourDetails.append(w)  # width of bounding rectangle
    contourDetails.append(h)  # height of bounding rectangle
    return contourDetails

'''
This function identifies characters that are on the same line or
row and groups them
'''
def rowSort(matrix):

    diff = 4
    allRows = []  #all rows
    first_row = []  #first assumed set of characters in the same row
    second_row = []  #second assumed set of characters in the same row
    for i in range(0, len(matrix)):
        if matrix[0][1] >= matrix[i][1] - diff and matrix[0][1] <= matrix[i][1] + diff:
            if i not  in first_row:
                first_row.append(i)
        elif i not in second_row:
            second_row.append(i)
    allRows.append(first_row)
    allRows.append(second_row)
    return allRows


'''
Order contours/segments in the same row
'''
def orderRows(row, all_rows):
    ordered_row = []
    for r in row:
        ordered_row.append(all_rows[r])
    ordered_row = sorted(ordered_row, key=lambda x: x[0]) #order contours by x value
    return ordered_row


'''
determine which segments are on top of each other based on
y coordinates
'''
def isOnTop(all_rows, matrix):
    combined_rows = []
    if len(all_rows) >= 2 and len(all_rows[1]) > 0:
        index = all_rows[0][0]
        index2 = all_rows[1][0]
        if matrix[index][1] < matrix[index2][1]:
            print("Street 1 on top of street 2")
            first_row = orderRows(all_rows[0], matrix)
            second_row = orderRows(all_rows[1], matrix)
            combined_rows.extend(first_row)
            combined_rows.extend(second_row)
            return combined_rows
        else:
            print("Street 2 on top of street 1")
            first_row = orderRows(all_rows[1], matrix)
            second_row = orderRows(all_rows[0], matrix)
            combined_rows.extend(first_row)
            combined_rows.extend(second_row)
            return combined_rows
    else:
        print("All on one street")
        first_row = orderRows(all_rows[0], matrix)
        combined_rows.extend(first_row)
        return combined_rows