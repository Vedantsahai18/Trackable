import numpy as np

class Detection:

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = confidence
        self.feature = np.asarray(feature, dtype=np.float32)


    def to_tlbr(self):
        '''
        Converting bounding box to format (min x, min y, max x, max y).
        '''

        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        '''
        Converting bounding box to format (x, y, a, h).
        (x, y) is the center of the bounding box.
        a is the aspect ratio.
        h is the height.
        '''

        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret