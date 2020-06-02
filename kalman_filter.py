import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTracker:

    def __init__(self, box, image=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.__convert_bbox_to_z(box)
        self.time_since_update = 0


    def __convert_bbox_to_z(self, box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = box[0] + w / 2
        y = box[1] + h / 2
        s = w * h
        r = w / h
        return np.array([x, y, s, r]).reshape(4, 1)

    def __convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = w[2] / 2
        if score is None:
            return np.array([x[0] - w / 2, x[1] - h/2, x[0] + w/2, x[1] + h / 2]).reshape((1,4))
        else:
            return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2, score]).reshape((1,5))