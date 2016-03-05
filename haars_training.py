__author__ = 'StreetHustling'

import urllib
import cv2
import numpy
import os

def store_raw_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513'
    neg_image_urls = urllib.urlopen(neg_images_link).read().decode()

    if os.path.exists('neg'):
        os.makedirs('neg')

    pic_num = 1

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.urlretrieve(i, "neg/"+str(pic_num)+'.jpg')
            cv2.imread("neg/"+str(pic_num)+'.jpg')

        except Exception as e:
            print(str(e))


