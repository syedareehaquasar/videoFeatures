## Detect facial expression from live video
run "python liveVideoFrameRead.py"

### Additional tags:<br/>
--save to save the video with predictions and landmarking <br/>
--savedata to save CSV file with expression predictions, their probability tensor, and eye aspect ratio


## Detect facial expression from video file
run "python videoFrameRead.py --video-file [your video file.mov]" where the video file needs to be in current directory

### Additional tags:
--frame-step the frame rate at which predictions are made, default was set to 10 frames <br/>
--save to save video with predictions and landmarking <br/>
--savedata to save csv file with expression predictions, their probability tensor and eye aspect ratio

## environment
virtualenv mirror
mirror\Scripts\activate (windows)
source mirror/bin/activate

py -m pip install -r requirements.txt
pip install -r requirements.txt 
# videoFeatures
