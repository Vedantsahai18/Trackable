import cv2
import dlib
import numpy as np
from scipy.spatial import distance


class Tracker:
    '''
    Class to performing tracking functions.
    '''

    def __init__(self, max_disappeared):
        self.max_disappeared = max_disappeared
        self.tracking_indices = []
    
    def reset_trackers(self):
        '''
        Resets the tracking_indices for new detection phase.
        '''
        self.tracking_indices = []

    def start_tracker(self, index, rgb_frame, box):
        '''
        Starts the tracker for a new trackable objects and adds to tracking_indices.
        '''
        (start_x, start_y, end_x, end_y) = box
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(start_x, start_y, end_x, end_y)
        t.start_track(rgb_frame, rect)

        self.tracking_indices.append(index)
        return t

    def update_tracker(self, rgb_frame, index, box):
        '''
        Updates the tracker for an existing trackable object and adds to
        tracking_indices if not tracked previously.
        '''
        (start_x, start_y, end_x, end_y) = box
        t = dlib.correlation_tracker()
        rect = dlib.rectangle(start_x, start_y, end_x, end_y)
        t.start_track(rgb_frame, rect)

        if index not in self.tracking_indices:
            self.tracking_indices.append(index)

        return t

    def track_person(self, frame, rgb_frame, trackables):
        '''
        Performs the main update for the tracking algorithm.
        '''
        rects = []
        for index in self.tracking_indices:
            trackables[index].tracker.update(rgb_frame)
            pos = trackables[index].tracker.get_position()
            start_x, start_y, end_x, end_y = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
            rects.append((start_x, start_y, end_x, end_y))
            trackables[index].update_image(frame.copy(), (start_x, start_y, end_x, end_y))

        if len(rects) == 0:

            for trkble in trackables.values():
                trkble.update_disappeared()

                if trkble.get_disappeared() > self.max_disappeared:
                    trkble.set_tracking(False)

            self.__draw_centroids(frame, trackables)
            return frame

        input_centroids = np.zeros((len(rects), 2), dtype='int')

        for i, (start_x, start_y, end_x, end_y) in enumerate(rects):
            c_X = (start_x + end_x) // 2
            c_Y = (start_y + end_y) // 2
            input_centroids[i] = (c_X, c_Y)

        obj_centroids = [trkble.get_centroid() for trkble in trackables.values() if trkble.is_being_tracked() and trkble.get_centroid() is not None]
        obj_ids = [trkble.get_id() for trkble in trackables.values() if trkble.is_being_tracked()]

        if len(obj_centroids) == 0:
            self.__draw_bounding_boxes(frame, rects)
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

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        if D.shape[0] >= D.shape[1]:
            # If there are objects that are lost
            for row in unused_rows:
                obj_id = obj_ids[row]
                trackables[obj_id].update_disappeared()

                if trackables[obj_id].get_disappeared() > self.max_disappeared:
                    trackables[obj_id].set_tracking(False)

        self.__draw_centroids(frame, trackables)
        self.__draw_bounding_boxes(frame, rects)
        return frame

    def __draw_centroids(self, frame, trackables):
        '''
        Draws the centroids currently tracked objects
        '''
        for index in self.tracking_indices:
            text = f'ID: {index}'

            centroid = trackables[index].get_centroid()
            if centroid is not None:
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    def __draw_bounding_boxes(self, frame, rects):
        '''
        Draws the bounding boxes for the currently tracked objects.
        '''
        for (start_x, start_y, end_x, end_y) in rects:
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
