# FacialExpressionRecognition
Use a deep learning model to predict facial expressions from a videostream

## Requirements
- Fast.ai library https://docs.fast.ai/install.html <br/> 
- Dlib https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/ <br/>
- Numpy <br/>
- Scipy <br/>
- IMutils <br/>
- OpenCV <br/>
- Pandas <br/>
- Argparse <br/>
- Python 

## Detect facial expression from live video
run "python liveVideoFrameRead.py"

Additional tags:<br/>
--save to save video with predictions and landmarking <br/>
--savedata to save csv file with expression predictions, their probability tensor and eye aspect ratio

## Detect facial expression from video file

run "python videoFrameRead.py --video-file [your video file.mov]" where the video file needs to be in current directory

Additional tags:<br/>
--frame-step the frame rate at which predictions are made, default was set to 10 frames <br/>
--save to save video with predictions and landmarking <br/>
--savedata to save csv file with expression predictions, their probability tensor and eye aspect ratio

