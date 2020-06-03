import cv2
import numpy as np

from core import Detector

from sort.detection import Detection
from sort.tracker import Tracker

from utils.constants import DETECT_AFTER_N, MAX_DISAPPEARED
from utils.helpers import print_info, is_within_scale, set_width_height

detector = Detector('yolo/y416/yolov3.cfg', 'yolo/y416/yolov3.weights')

print_info('Loading Video...')
cap = cv2.VideoCapture('mp4/001.mp4')
print_info('Video Loaded!')

W = int(cap.get(3))
H = int(cap.get(4))
set_width_height(W, H)
print_info(f"Width: {W} | Height: {H}")
