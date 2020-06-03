import cv2
import dlib
import numpy as np

from core import Detector
from core import Tracker
from core import match_people
from core import Trackable

from utils.constants import DETECT_AFTER_N, MAX_DISAPPEARED
from utils.helpers import print_info, is_within_scale, set_width_height

detect = Detector('yolo/y416/yolov3.cfg', 'yolo/y416/yolov3.weights')
track = Tracker(MAX_DISAPPEARED)

print_info("Loading Video...")
cap = cv2.VideoCapture('mp4/005.mp4')
print_info("Video Loaded!")
W  = int(cap.get(3))
H = int(cap.get(4))
set_width_height(W, H)
print_info(f"Width: {W} | Height: {H}")

count = 0
trackable_objects = {}

while True:

    ret, frame = cap.read()

    if frame is None:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if count % DETECT_AFTER_N == 0:

        next_object_index = len(trackable_objects)
        
        track.reset_trackers()

        detect_image, objects = detect.detect_person(frame.copy())
        for i, box in enumerate(objects):

            if is_within_scale(box):
                found_match, found_index, cropped_image = match_people(frame, box, trackable_objects)
                
                if not found_match:
                    print_info('Creating New Object')
                    # If match is not found register a new object
                    t = track.start_tracker(next_object_index, rgb_frame, box)
                    trackable_objects[next_object_index] = Trackable(next_object_index, t, cropped_image)
                    next_object_index += 1
                
                else:
                    print_info('Updating Object')
                    # If match is found, update the box for existing object
                    t = track.update_tracker(rgb_frame, found_index, box)
                    trackable_objects[found_index].update_detector_image(cropped_image)
                    trackable_objects[found_index].update_tracker(t)
                    trackable_objects[found_index].set_tracking(True)

        frame = cv2.putText(detect_image, f'Detecting...', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        
        tracking_image = track.track_person(frame.copy(), rgb_frame, trackable_objects)
                # Get the updated trackers
        
        trackable_objects = {trkble_id: trkble for trkble_id, trkble in trackable_objects.items() if trkble.is_being_tracked()}
        n_o_p = len(trackable_objects)

        tracking_image = cv2.putText(tracking_image, f'Number of people detected: {n_o_p}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frame = cv2.putText(tracking_image, f'Tracking...', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("Main Window", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

cap.release()
cv2.destroyAllWindows()