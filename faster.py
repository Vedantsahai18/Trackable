import cv2
import dlib
import multiprocessing
import numpy as np

from detector import Detector
from centroid_tracker import CentroidTracker

detect = Detector('yolo/y416/yolov3.cfg', 'yolo/y416/yolov3.weights')
centroid_tracker = CentroidTracker(50)

cap = cv2.VideoCapture('mp4/001.mp4')
count = 0
trackers = []
input_queue = []
output_queue = []

def start_tracker(box, rgb_frame, input_queue, output_queue):
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb_frame, rect)

    while True:
        rgb_frame = input_queue.get()

        if rgb_frame is not None:
            t.update(rgb_frame)
            pos = t.get_position()

            start_x = int(pos.left())
            start_y = int(pos.top())
            end_x = int(pos.right())
            end_y = int(pos.bottom())

            output_queue.put(((start_x, start_y, end_x, end_y)))



while True:

    ret, frame = cap.read()

    if frame is None:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(input_queue) == 0:
        
        detect_image, objects = detect.detect_person(frame.copy())

        iq = multiprocessing.Queue()
        oq = multiprocessing.Queue()
        input_queue.append(iq)
        output_queue.append(oq)

        for box in objects:
            p = multiprocessing.Process(
                target=start_tracker,
                args=(box, rgb_frame, iq, oq)
            )
            p.daemon = True
            p.start()
        
        cv2.imshow("Detected", detect_image)
    else:
        rects = []
        for iq in input_queue:
            iq.put(rgb_frame)

        for oq in output_queue:

            (start_x, start_y, end_x, end_y) = oq.get()

            rects.append((start_x, start_y, end_x, end_y))
        
        tracking_image = centroid_tracker.track_person(frame, rects)
        
        for (start_x, start_y, end_x, end_y) in rects:
            cv2.rectangle(tracking_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.imshow("Tracked", tracking_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

cap.release()
cv2.destroyAllWindows()