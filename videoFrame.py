#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
non- live video analysis

@author: syedareehaquasar
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
import imutils
import cv2
import numpy as np
from imutils.video import FileVideoStream

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video-file", required=True,
                help="video file in current directory")
ap.add_argument("--frame-step", type=int, default=10,
                help="framecount which video frames are predicted")
ap.add_argument("--save", dest="save", action="store_true")
ap.add_argument("--no-save", dest="save", action="store_false")
ap.add_argument("--savedata", dest="savedata", action="store_true")
ap.add_argument("--no-savedata", dest="savedata", action="store_false")
ap.set_defaults(savedata=False)
ap.set_defaults(save=False)
args = vars(ap.parse_args())

vidcap = FileVideoStream(args["video_file"]).start()
count = 0
framecount = 0
data = []

# parameters for loading data and images
detection_model_path = './haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = './models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = keras.models.load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]


if args["save"]:
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (450, 300))

while vidcap.more():
    frame = vidcap.read()
    if frame is None:
        break
    # reading the frame
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

    else:
        continue

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        # draw the label + probability bar on the canvas
       # emoji_face = feelings_faces[np.argmax(preds)]

        w = int(prob * 300)

        data.append([i, emotion, prob])

        cv2.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    framecount += 1
    if args["save"]:
        out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if args["savedata"]:
    df = pd.DataFrame(
        data, columns=['Framecount', 'Expression', 'Probability'])
    df.to_csv('./export.csv')
    print("data saved to export.csv")
vidcap.stop()
if args["save"]:
    print("done saving")
    out.release()
cv2.destroyAllWindows()
