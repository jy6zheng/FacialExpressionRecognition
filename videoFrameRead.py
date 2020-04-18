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
import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video-file", required=True, help="video file in current directory")
ap.add_argument("--time-step", type=int, default = 1, help="time step which video frames are predicted")
ap.add_argument("--save", dest="save", action = "store_true")
ap.add_argument("--no-save", dest="save", action = "store_false")
ap.set_defaults(save_photo = False)
args = vars(ap.parse_args())

path = "/Users/joycezheng/FacialRecognitionVideo/"
vidcap = cv2.VideoCapture(args["video_file"])
success,image = vidcap.read()
count = 0
framecount = []
learn = load_learner(path, 'export.pkl')
data = []

def text_display_results(image, prediction, probability):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (255, 255, 255)
    thickness = 2
    text = str(prediction)+" "+str(probability)
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
    text_offset_x = 10
    text_offset_y = image.shape[0] - 25
    display = cv2.putText(image, text, (text_offset_x, text_offset_y), font, fontScale, color, thickness, cv2.LINE_AA)
    return display


while success:
   vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
   success,image = vidcap.read()

   if success:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.imwrite("frame%d.jpg" % count, gray)
      img = open_image('frame{}.jpg'.format(count))
      prediction, idx, probability = learn.predict(img)
      display = text_display_results(image, prediction, probability)
      cv2.imshow("time %ds" % count, display)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      print("time %ds:" % count, prediction, probability)
      framecount.append(count)
      data.append([count, prediction, probability])
   count += args["time_step"]

if args["save"]:
    df = pd.DataFrame(data, columns = ['Time (seconds)', 'Expression', 'Probability'])
    df.to_csv(path+'/export.csv')
vidcap.release()
cv2.destroyAllWindows()
