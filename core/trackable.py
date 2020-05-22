from utils.helpers import is_within_scale, is_within_bounds

class Trackable:
    '''
    Class that stores information about a trackable object.
    '''

    def __init__(self, obj_id, tracker, input_image):
        self.obj_id = obj_id
        self.tracker = tracker
        self.centroid = None
        self.detector_image = input_image
        self.image = input_image
        self.disappeared = 0
        self.counted = False
        self.tracking = True

    def get_id(self):
        '''
        Returns the id of the current trackable object.
        '''
        return self.obj_id

    def set_tracking(self, value):
        '''
        Sets the tracking status of the object.
        '''
        self.tracking = value

    def is_being_tracked(self):
        '''
        Returns bool for if the object is being tracked.
        '''
        return self.tracking

    def update_image(self, input_image, box):
        '''
        Update the image at the current timestep.
        '''
        if not is_within_bounds(box):
            self.image = self.detector_image
            return
        if not is_within_scale(box):
            self.image = self.detector_image
            return

        (start_x, start_y, end_x, end_y) = box

        self.image = input_image[start_y:end_y, start_x:end_x]
        self.__update_centroid(box)

    def update_detector_image(self, input_image):
        '''
        Updates the image taken my the image detector.
        '''
        self.detector_image = input_image

    def update_disappeared(self):
        '''
        Updates the frames passed since it disappeared.
        '''
        self.disappeared += 1

    def get_disappeared(self):
        '''
        Returns the number of frames it has disappeared.
        '''
        return self.disappeared

    def update_tracker(self, t):
        '''
        Updates the tracker object.
        '''
        self.tracker = t

    def get_centroid(self):
        '''
        Returns the centroid given by bounding box.
        '''
        return self.centroid

    def __update_centroid(self, box):
        '''
        Updates the centroid provided by the bounding box.
        '''
        (start_x, start_y, end_x, end_y) = box
        c_X = (start_x + end_x) // 2
        c_Y = (start_y + end_y) // 2
        self.centroid = (c_X, c_Y)