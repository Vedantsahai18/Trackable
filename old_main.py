import cv2
import numpy as np

from detector import Detector
from centroid_tracker import CentroidTracker

detect = Detector('yolo/tiny/yolov3-tiny.cfg', 'yolo/tiny/yolov3-tiny.weights')
centroid_track = CentroidTracker(100)

cap = cv2.VideoCapture('mp4/001.mp4')

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if count % 2 == 0:
        detect_image, objects = detect.detect_person(frame.copy())
        track_image = centroid_track.track_person(frame.copy(), objects)

    cv2.imshow('Video', frame)
    cv2.imshow('Detected', detect_image)
    cv2.imshow('Track', track_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

cap.release()
cv2.destroyAllWindows()