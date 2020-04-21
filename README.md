# FacialExpressionRecognition
Use a deep learning model to predict facial expressions from a videostream

## Detect facial expression from live video
run "python liveVideoFrameRead.py"

Additional tags:
--save to save video with predictions and landmarking
--savedata to save csv file with expression predictions, their probability tensor and eye aspect ratio

## Detect facial expression from video file

run "python videoFrameRead.py --video-file [your video file.mov]" where the video file needs to be in current directory

Additional tags:
--frame-step the frame rate at which predictions are made, default was set to 10 frames
--save to save video with predictions and landmarking
--savedata to save csv file with expression predictions, their probability tensor and eye aspect ratio

