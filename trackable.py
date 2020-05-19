class Trackable:

    def __init__(self, obj_id, tracker):
        self.obj_id = obj_id
        self.tracker = tracker
        self.centroid = None
        self.disappeared = 0
        self.counted = False

    def update_centroid(self, centroid):
        self.centroid = centroid

    def update_tracker(self, tracker):
        self.tracker = tracker

    def get_tracker(self):
        return self.tracker

    def update_disappeared(self):
        self.disappeared += 1

    def get_disappeared(self):
        return self.disappeared