import numpy as np
import scipy.linalg


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter:
    ''' 
    According to the paper there is an 8 dimensional space (x, y, gamma, h, vx, vy, vgamma, vh).
    Here (x, y) is center of bounding box.
    gamma is aspect ratio.
    h is height.
    Standard Kalman Filter with constant velocity motion and linear observation model was used.
    '''

    def __init__(self):
        ndim, dt = 4, 1

        # Kalman filter model matrices
        # motion_mat is Fk
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Weights that control the amount of uncertainty in model
        self._std_weight_position = 1 / 20
        self._std_weight_velocity = 1 / 160


    def initiate(self, bbox):
        '''
        bbox is the bounding box with the co-ordinates (x, y, a, h).
        (x, y) -> Center of bounding box.
        a -> Aspect Ration.
        h -> Height.
        '''

        mean_pos = bbox
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * bbox[3],
            2 * self._std_weight_position * bbox[3],
            1e-2,
            2 * self._std_weight_position * bbox[3],
            10 * self._std_weight_velocity * bbox[3],
            10 * self._std_weight_velocity * bbox[3],
            1e-5,
            10 * self._std_weight_velocity * bbox[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        '''
        Runs the Kalman Filter prediction step.
        mean and covariance are the 8 dimensional vectors from the previous timestep.
        '''

        std_pos = [
            self._std_weight_position * mean[3], 
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_(std_pos, std_vel)))

        mean = np.dot(self._motion_mat, mean)
        # Pk = Fk . Pk-1 . Transpose(F)k + Qk
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance


    def project(self, mean, covariance):
        '''
        Projects state distribution to measurement space.
        '''

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, bbox):
        '''
        Runs the Kalman Filter correction step.
        '''

        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = bbox - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, bbox, only_position=False):
        '''
        Computes the gating distance between state distribution and measurements.
        '''
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            bbox = bbox[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = bbox - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
