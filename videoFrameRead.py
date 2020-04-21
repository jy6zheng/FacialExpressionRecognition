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
ap.add_argument("--savedata", dest="savedata", action = "store_true")
ap.add_argument("--no-savedata", dest="savedata", action = "store_false")
ap.set_defaults(savedata = False)
ap.set_defaults(save = False)
args = vars(ap.parse_args())

path = "/Users/joycezheng/FacialRecognitionVideo/"
vidcap = cv2.VideoCapture(args["video_file"])
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
success,image = vidcap.read()
count = 0
framecount = []
learn = load_learner(path, 'export.pkl')
data = []

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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

if args["save"]:
    out = cv2.VideoWriter(path + "output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while success:
   vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
   success,image = vidcap.read()

   if success:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
      for coords in face_coord:
          X, Y, w, h = coords
          H, W, _ = image.shape
          X_1, X_2 = (max(0, X - int(w * 0.3)), min(X + int(1.3 * w), W))
          Y_1, Y_2 = (max(0, Y - int(0.3 * h)), min(Y + int(1.3 * h), H))
          img_cp = gray[Y_1:Y_2, X_1:X_2].copy()
          prediction, idx, probability = learn.predict(Image(pil2tensor(img_cp, np.float32).div_(225)))
          cv2.rectangle(
                  img=image,
                  pt1=(X_1, Y_1),
                  pt2=(X_2, Y_2),
                  color=(128, 128, 0),
                  thickness=2,
              )
      display = text_display_results(image, prediction, probability)
      cv2.imshow("time %ds" % count, display)
      if args["save"]:
          out.write(display)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      print("time %ds:" % count, prediction, probability)
      framecount.append(count)
      data.append([count, prediction, probability])
   count += args["time_step"]

if args["savedata"]:
    df = pd.DataFrame(data, columns = ['Time (seconds)', 'Expression', 'Probability'])
    df.to_csv(path+'/export.csv')
vidcap.release()
cv2.destroyAllWindows()
