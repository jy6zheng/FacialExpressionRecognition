#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:07:17 2020

@author: joycezheng
"""

import cv2
import numpy as np
import imutils
from fastai import *
from fastai.vision import *

path = "/Users/joycezheng/.spyder-py3/"

vidcap = cv2.VideoCapture('MomExpression.mov')
success,image = vidcap.read()
count = 0
framecount = []

while success:
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))        
  success,image = vidcap.read()
  cv2.imwrite("frame%d.jpg" % count, image)
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # cv2.imwrite("frame%d.jpg" % count, gray)
  print('Read a new frame: ', success)
  framecount.append(count)
  count += 1
learn = load_learner(path, 'export.pkl')
print(framecount)

for i in framecount[:-1]:
    image = cv2.imread('frame{}.jpg'.format(i))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("frame%d.jpg" % i, gray)

for count in framecount[:-1]:
    print(count)
    img = open_image('frame{}.jpg'.format(count))
    prediction, idx, probability = learn.predict(img)
    print(prediction)
    print(probability)
    