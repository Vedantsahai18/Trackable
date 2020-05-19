import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict

from trackable import Trackable

class Tracker:
    '''
    Class to track objects
    '''

    def __init__(self, max_disappeared):
        '''
        Setting the max disappearing threshold for objects
        '''
        self.next_object_id = 0
        self.max_disappeared = max_disappeared
        self.tracking = []
        self.objects = []
        self.trackable_objects = {}

    def remove(self, obj_id):
        '''
        Remove a lost or disappeared object
        '''
        self.tracking.remove(obj_id)
        self.trackable_objects[obj_id].tracked = False
        

    def update_tracker(self, rgb_frame, box, index):
        '''
        Start the tracking of a previously identified object again
        '''
        (start_x, start_y, end_x, end_y) = box
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(start_x, start_y, end_x, end_y)
        t.start_track(rgb_frame, rect)
        
        self.trackable_objects[index].update_tracker(t)
        self.trackable_objects[index].tracked = True
        self.tracking.append(index)


    def start_tracker(self, rgb_frame, box):
        '''
        Add a new object to track
        '''
        (start_x, start_y, end_x, end_y) = box
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(start_x, start_y, end_x, end_y)
        t.start_track(rgb_frame, rect)
        
        self.trackable_objects[self.next_object_id] = Trackable(self.next_object_id, t)
        self.tracking.append(self.next_object_id)
        self.next_object_id += 1

    def reset_trackers(self):
        '''
        Reset trackers if no objects to track
        '''
        self.tracking = []
        for trkble in self.trackable_objects.values():
            trkble.tracked = False

    def get_tracker_length(self):
        '''
        Gives us the number of people being tracked
        '''
        return len(self.tracking)

    def track_person(self, frame, rgb_frame):
        '''
        Use dlib to predict next position of object
        '''
        # Gives us list of boxes of currently tracked objects
        rects = []
        for index in self.tracking:
            t = self.trackable_objects[index].tracker
            t.update(rgb_frame)
            pos = t.get_position()
            start_x = int(pos.left())
            start_y = int(pos.top())
            end_x = int(pos.right())
            end_y = int(pos.bottom())
            rects.append((start_x, start_y, end_x, end_y))
        
        # If there are no objects found
        if len(rects) == 0:

            for obj_id, trkble in self.trackable_objects.items():
                trkble.update_disappeared()

                if trkble.get_disappeared() > self.max_disappeared:
                    self.remove(obj_id)

            return frame

        # Initialzing array based on number of objects detected
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        # Calculating the centroids for each object
        for i, (start_x, start_y, end_x, end_y) in enumerate(rects):
            c_X = (start_x + end_x) // 2
            c_Y = (start_y + end_y) // 2
            input_centroids[i] = (c_X, c_Y)

        # Getting the list ids and values of all the currently tracked objects
        obj_ids = self.tracking
        obj_centroids = [trkble.centroid for trkble in self.trackable_objects.values() if trkble.obj_id in self.tracking and trkble.centroid is not None]

        if len(obj_centroids) == 0:
            for index, centroid in zip(self.tracking, input_centroids):
                self.trackable_objects[index].update_centroid(centroid)

            self.__draw_centroids(frame)
            self.__draw_boxes(frame, rects)
            return frame


        # Calculating the distance between the tracked and new objects
        D = distance.cdist(obj_centroids, input_centroids)


        # Gives the list of rows with the smallest values and sorts it (ascending order)
        rows = D.min(axis=1).argsort()
        # Gives the list of values with the smallest column and sorts according to row
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
    
        for(row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            
            # Row will map existing object to
            # Col of input centroid
            obj_id = obj_ids[row]
            self.trackable_objects[obj_id].update_centroid(input_centroids[col])
            self.trackable_objects[obj_id].disappeared = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        if D.shape[0] >= D.shape[1]:
            # If there are objects that are lost
            for row in unused_rows:
                obj_id = obj_ids[row]
                self.trackable_objects[obj_id].update_disappeared()

                if self.trackable_objects[obj_id].get_disappeared() > self.max_disappeared:
                    self.remove(obj_id)

        self.__draw_centroids(frame)
        self.__draw_boxes(frame, rects)
        return frame

    def __draw_centroids(self, image):
        '''
        Draws the centroids obtained from boxes
        '''
        for obj_id in self.tracking:
            centroid = self.trackable_objects[obj_id].centroid
            text = f'ID: {obj_id}'

            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    def __draw_boxes(self, image, rects):
        '''
        Draws the bounding boxes
        '''
        for (start_x, start_y, end_x, end_y) in rects:
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        return image
        