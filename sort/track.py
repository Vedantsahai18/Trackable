class TrackState:
    '''
    Used to maintain the track state.
    Newly created tracks are classified as tentative until enough evidence is collected.
    Then the state is changed to confirmed.
    Tracks that are no longer alive are classified as deleted.
    '''


    Tentative = 1
    Confirmed = 2
    Deleted = 3



class Track:
    '''
    Track with state space (x, y, a, h) and associated velocities.
    (x, y) is the center of the bounding box.
    a is the aspect ratio.
    h is the height.
    '''
    
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        '''
        Get current position in bounding box format (top left x, top left y)
        '''

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        '''
        Get current position in bounding box format (min x, min y, max x, max y)
        '''
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def predict(self, kf):
        '''
        Propagate the state distribution to the current time step using a Kalman filter prediction step.
        '''

        self.mean, self.covariance  = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1


    def update(self, kf, detection):
        '''
        Perform Kalman filter measurement update step and update the feature cache.
        '''

        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        '''
        Mark state as missed (no association at current time step).
        '''

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentaive(self):
        '''
        True is track is tentative.
        '''

        return self.state == TrackState.Tentative

    def is_confirmed(self):
        '''
        True if track in confirmed.
        '''

        return self.state == TrackState.Confirmed

    def is_deleted(self):
        '''
        True if this track is dead, should be deleted.
        '''

        return self.state == TrackState.Deleted