#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# 生成圆形的坐标点
def generate_circular_landmarks(center, num_points, radius):
    """
    Generate landmarks in a circular pattern around a center.

    Parameters:
    - center: tuple, center of the circle (x, y)
    - num_points: int, number of points to generate around the circle
    - radius: float, radius of the circle

    Returns:
    - np.array of shape (num_points, 2), the generated landmark coordinates
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)
    return np.column_stack((x_coords, y_coords))


# 中心位置设定为 (10, 10)
center = (0, 0)

# 设置不同半径和角度
radii = [5, 10, 15, 20]
# radii = [10]# 3个不同的圆环
num_points_per_circle = 20  # 每个圆环上的点数

# 存储所有圆环的坐标
circular_landmarks = []

for radius in radii:
    landmarks = generate_circular_landmarks(center, num_points_per_circle, radius)
    circular_landmarks.append(landmarks)

# 将多个圆环的坐标组合成一个数组
circular_landmarks = np.vstack(circular_landmarks)


class FilterConfiguration:
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2  # Bearing noise variance

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([0.1, 0.1, 0.05]) ** 2


class Map:
    def __init__(self):
        self.landmarks = circular_landmarks
        # self.landmarks = np.array([
        #     [0, 0],
        #     [0, 5],
        #     [0, 10],
        #     [0, 15],
        #     [0, 20],
        #     [5, 0],
        #     [5, 5],
        #     [5, 10],
        #     [5, 15],
        #     [5, 20],
        #     [10, 0],
        #     [10, 5],
        #     [10, 10],
        #     [10, 15],
        #     [10, 20],
        #     [15, 0],
        #     [15, 5],
        #     [15, 10],
        #     [15, 15],
        #     [15, 20],
        #     [20, 0],
        #     [20, 5],
        #     [20, 10],
        #     [20, 15],
        #     [20, 20]
        #     # [10, 0],
        # ])


class RobotEstimator:
    def __init__(self, filter_config, map):
        self._config = filter_config
        self._map = map
        self.start()

    def start(self):
        self._t = 0
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    def set_control_input(self, u):
        self._u = u

    def predict_to(self, time):
        dt = time - self._t
        self._t = time
        self._predict_over_dt(dt)

    def estimate(self):
        return self._x_est, self._Sigma_est

    def _predict_over_dt(self, dt):
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V

        # Predict the new state
        self._x_pred = self._x_est + np.array([
            v_c * np.cos(self._x_est[2]) * dt,
            v_c * np.sin(self._x_est[2]) * dt,
            omega_c * dt
        ])
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]), np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1, v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])
        self._kf_predict_covariance(A, V * dt)

    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    def update_from_landmark_range_bearing(self, y_measurements):
        y_pred = []
        C = []
        x_pred = self._x_pred
        for i, lm in enumerate(self._map.landmarks):
            dx = lm[0] - x_pred[0]
            dy = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx ** 2 + dy ** 2)
            bearing_pred = np.arctan2(dy, dx) - x_pred[2]

            # Expected measurement
            y_pred.extend([range_pred, bearing_pred])

            # Measurement model Jacobian for range and bearing
            C_range = np.array([-(dx) / range_pred, -(dy) / range_pred, 0])
            C_bearing = np.array([dy / (range_pred ** 2), -dx / (range_pred ** 2), -1])
            C.append(C_range)
            C.append(C_bearing)

        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation
        # y_measurements = np.ravel(y_measurements)
        nu = y_measurements - y_pred
        nu[1::2] = (nu[1::2] + np.pi) % (2 * np.pi) - np.pi  # Normalize bearing differences

        # Measurement noise covariance for range and bearing
        num_landmarks = len(self._map.landmarks)
        W = np.diag([self._config.W_range, self._config.W_bearing] * num_landmarks)
        self._do_kf_update(nu, C, W)

        # Normalize angle in the estimate
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]), np.cos(self._x_est[-1]))

    def _do_kf_update(self, nu, C, W):
        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred
