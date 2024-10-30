#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


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
center = (10, 10)

# 设置不同半径和角度
# radii = [5, 10, 15]
radii = [10]# 3个不同的圆环
num_points_per_circle = 8  # 每个圆环上的点数

# 存储所有圆环的坐标
circular_landmarks = []

for radius in radii:
    landmarks = generate_circular_landmarks(center, num_points_per_circle, radius)
    circular_landmarks.append(landmarks)

# 将多个圆环的坐标组合成一个数组
circular_landmarks = np.vstack(circular_landmarks)


class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([2.0, 2.0, 1.0]) ** 2


class Map(object):

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
        # ])
        # self.landmarks = np.array([
        #     [5, 10],
        #     [15, 5],
        #     [10, 15]
        # ])


class ParticleFilter:
    def __init__(self, num_particles, map_landmarks, initial_pose, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.particles = np.tile(initial_pose, (num_particles, 1)) + np.random.normal(0, 0.5, (num_particles, 3))
        self.weights = np.ones(num_particles) / num_particles
        self.map_landmarks = map_landmarks
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_input, dt):
        """Update particle positions based on control inputs (prediction step)"""
        linear_velocity, angular_velocity = control_input
        noise = np.random.normal(0, self.process_noise, (self.num_particles, 3))

        # Update each particle's position
        self.particles[:, 0] += (linear_velocity * np.cos(self.particles[:, 2]) * dt) + noise[:, 0]
        self.particles[:, 1] += (linear_velocity * np.sin(self.particles[:, 2]) * dt) + noise[:, 1]
        self.particles[:, 2] += angular_velocity * dt + noise[:, 2]
        self.particles[:, 2] = np.mod(self.particles[:, 2], 2 * np.pi)  # Angle normalization

    def update(self, measurements):
        """Update particle weights based on sensor measurements"""
        for i, landmark in enumerate(self.map_landmarks):
            dx = landmark[0] - self.particles[:, 0]
            dy = landmark[1] - self.particles[:, 1]
            predicted_range = np.sqrt(dx ** 2 + dy ** 2)
            predicted_bearing = np.arctan2(dy, dx) - self.particles[:, 2]

            # Calculate differences between predicted and actual measurements
            range_diff = measurements[i][0] - predicted_range
            bearing_diff = measurements[i][1] - predicted_bearing
            bearing_diff = (bearing_diff + np.pi) % (2 * np.pi) - np.pi  # Angle normalization

            # Update weights using Gaussian distributions
            range_weight = np.exp(-0.5 * (range_diff ** 2) / self.measurement_noise[0] ** 2)
            bearing_weight = np.exp(-0.5 * (bearing_diff ** 2) / self.measurement_noise[1] ** 2)
            self.weights *= range_weight * bearing_weight

        # Normalize weights
        self.weights += 1.e-300  # Avoid zeros
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on their weights"""
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """Estimate robot's position based on particle distribution"""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        return mean


class RobotEstimator(object):

    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map

    # This method MUST be called to start the filter
    def start(self):
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    # Predict to the time. The time is fed in to
    # allow for variable prediction intervals.
    def predict_to(self, time):
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # Now predict over a duration dT
        self._predict_over_dt(dt)

    # Return the estimate and its covariance
    def estimate(self):
        return self._x_est, self._Sigma_est

    # This method gets called if there are no observations
    def copy_prediction_to_estimate(self):
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    # This method sets the filter to the initial state
    def _set_estimate_to_initial_conditions(self):
        # Initial estimated state and covariance
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    # Predict to the time
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
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]),
                                      np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1, v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])

        self._kf_predict_covariance(A, self._config.V * dt)

    # Predict the EKF covariance; note the mean is
    # totally model specific, so there's nothing we can
    # clearly separate out.
    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    # Implement the Kalman filter update step.
    def _do_kf_update(self, nu, C, W):
        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred

    def update_from_landmark_range_observations(self, y_range):
        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred ** 2 + dy_pred ** 2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are oberving a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))


class RobotSimulator:
    def __init__(self, config, particle_filter):
        self.config = config
        self.pf = particle_filter
        self.dt = 0.1
        self.control_input = [1.0, 0.1]  # Example control input (linear and angular velocities)

    def simulate(self, time_steps):
        trajectory = []
        for t in range(time_steps):
            # Move particles according to control inputs
            self.pf.predict(self.control_input, self.dt)

            # Simulate measurements with noise for range and bearing
            measurements = []
            for lm in self.pf.map_landmarks:
                dx, dy = lm[0] - self.pf.estimate()[0], lm[1] - self.pf.estimate()[1]
                distance = np.sqrt(dx ** 2 + dy ** 2) + np.random.normal(0, self.config.W_range ** 0.5)
                bearing = np.arctan2(dy, dx) - self.pf.estimate()[2] + np.random.normal(0, self.config.W_bearing ** 0.5)
                measurements.append([distance, bearing])

            # Update particles based on measurements and resample
            self.pf.update(measurements)
            self.pf.resample()

            # Record the estimated position for plotting
            trajectory.append(self.pf.estimate()[:2])

        return np.array(trajectory)
