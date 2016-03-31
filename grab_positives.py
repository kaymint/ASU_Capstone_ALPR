__author__ = 'StreetHustling'


import urllib
import cv2
import numpy
import os

def stor_pos_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'
    neg_image_urls = urllib.urlopen(neg_images_link).read().decode()

    if not os.path.exists('pos1'):
        os.makedirs('pos1')

    pic_num = 1

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, "pos1/"+str(pic_num)+'.jpg')
            img = cv2.imread("pos1/"+str(pic_num)+'.jpg', cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100,100))
            cv2.imwrite("pos1/"+str(pic_num)+'.jpg', resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))


stor_pos_images()
