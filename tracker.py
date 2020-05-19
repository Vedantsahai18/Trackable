import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict


class Tracker:
    '''
    Class to track objects
    '''

    def __init__(self, max_disappeared):
        '''
        Setting the max disappearing threshold for objects
        '''
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappered = OrderedDict()
        self.max_disappeared = max_disappeared
        self.trackers = []

    def register(self, centroid):
        '''
        Register a new object
        '''
        self.objects[self.next_object_id] = centroid
        self.disappered[self.next_object_id] = 0
        self.next_object_id += 1

    def remove(self, obj_id):
        '''
        Remove a lost or disappeared object
        '''
        del self.objects[obj_id]
        del self.disappered[obj_id]
        self.next_object_id -= 1

    def start_tracker(self, rgb_frame, box):
        '''
        Add a new object to track
        '''
        (start_x, start_y, end_x, end_y) = box
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(start_x, start_y, end_x, end_y)
        t.start_track(rgb_frame, rect)
        self.trackers.append(t)

    def reset_trackers(self):
        '''
        Reset trackers if no objects to track
        '''
        self.trackers = []

    def get_tracker_length(self):
        '''
        Gives us the number of people being tracked
        '''
        return len(self.trackers)

    def track_person(self, frame, rgb_frame):
        '''
        Use dlib to predict next position of object
        '''
        rects = []
        for t in self.trackers:
            t.update(rgb_frame)
            pos = t.get_position()
            start_x = int(pos.left())
            start_y = int(pos.top())
            end_x = int(pos.right())
            end_y = int(pos.bottom())
            rects.append((start_x, start_y, end_x, end_y))
        
        # Update the centroids for each box
        centroid_image = self.__update(frame, rects)

        return self.__draw_boxes(centroid_image, rects)
        
    
    def __update(self, image, rects):
        '''
        Update the tracking class
        '''
        # If there are no objects found
        if len(rects) == 0:

            for obj_id in list(self.disappered.keys()):
                self.disappered[obj_id] += 1

                if self.disappered[obj_id] > self.max_disappeared:
                    self.remove(obj_id)

            self.__draw_centroids(image)
            return image

        # Initialzing array based on number of objects detected
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        # Calculating the centroids for each object
        for i, (start_x, start_y, end_x, end_y) in enumerate(rects):
            c_X = (start_x + end_x) // 2
            c_Y = (start_y + end_y) // 2
            input_centroids[i] = (c_X, c_Y)

        # Adding the object to tracker
        if len(self.objects) == 0:
            # For the first new object
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Getting the list ids and values of all the currently tracked objects
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

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
                self.objects[obj_id] = input_centroids[col]
                self.disappered[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                # If there are objects that are lost
                for row in unused_rows:
                    obj_id = obj_ids[row]
                    self.disappered[obj_id] += 1

                    if self.disappered[obj_id] > self.max_disappeared:
                        self.remove(obj_id)
            else:
                # If there are new objects more than tracked objects
                for col in unused_cols:
                    self.register(input_centroids[col])

            self.__draw_centroids(image)

        return image

    def __draw_centroids(self, image):
        '''
        Draws the centroids obtained from boxes
        '''
        for (obj_id, centroid) in self.objects.items():
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
        