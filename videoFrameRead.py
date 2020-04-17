#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:07:17 2020

@author: joycezheng
"""

import cv2
import numpy as np
from fastai import *
from fastai.vision import *

path = "/Users/joycezheng/FacialRecognitionVideo/"

vidcap = cv2.VideoCapture('MomExpression.mov')
success,image = vidcap.read()
count = 0
framecount = []
learn = load_learner(path, 'export.pkl')
font = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.7
org = (50, 50) 
color = (255, 255, 255)
thickness = 2
while success:
   vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))        
   success,image = vidcap.read()
   if success:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.imwrite("frame%d.jpg" % count, gray)
      img = open_image('frame{}.jpg'.format(count))
      prediction, idx, probability = learn.predict(img)
      text = str(prediction)+" "+str(probability)
      display = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA) 
      cv2.imshow("time %ds" % count, display)
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
      print("time %ds:" % count, prediction)
      framecount.append(count)
   count += 1

vidcap.release()
cv2.destroyAllWindows()

print(framecount)


    