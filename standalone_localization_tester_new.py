#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from robot_localization_system_new import FilterConfiguration, Map, RobotEstimator

class SimulatorConfiguration:
    def __init__(self):
        self.dt = 0.1
        self.total_time = 1000
        self.time_steps = int(self.total_time / self.dt)
        # Control inputs (linear and angular velocities)
        self.v_c = 1.0  # Linear velocity [m/s]
        self.omega_c = 0.1  # Angular velocity [rad/s]

class Controller:
    def __init__(self, config):
        self._config = config

    def next_control_input(self, x_est, Sigma_est):
        return [self._config.v_c, self._config.omega_c]

class Simulator:
    def __init__(self, sim_config, filter_config, map):
        self._config = sim_config
        self._filter_config = filter_config
        self._map = map

    def start(self):
        self._time = 0
        self._x_true = np.random.multivariate_normal(self._filter_config.x0, self._filter_config.Sigma0)
        self._u = [0, 0]

    def set_control_input(self, u):
        self._u = u

    def step(self):
        dt = self._config.dt
        v_c = self._u[0]
        omega_c = self._u[1]
        v = np.random.multivariate_normal(mean=[0, 0, 0], cov=self._filter_config.V * dt)
        self._x_true = self._x_true + np.array([
            v_c * np.cos(self._x_true[2]) * dt,
            v_c * np.sin(self._x_true[2]) * dt,
            omega_c * dt
        ]) + v
        self._x_true[-1] = wrap_angle(self._x_true[-1])  # Normalize angle
        self._time += dt
        return self._time

    def landmark_range_bearing_observations(self):
        """Get the range and bearing observations to landmarks"""
        observations = []
        W_range = self._filter_config.W_range
        W_bearing = self._filter_config.W_bearing
        for lm in self._map.landmarks:
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            bearing_true = np.arctan2(dy, dx) - self._x_true[2]
            range_meas = range_true + np.random.normal(0, np.sqrt(W_range))
            bearing_meas = wrap_angle(bearing_true + np.random.normal(0, np.sqrt(W_bearing)))
            observations.extend([range_meas, bearing_meas])
        return np.array(observations)

    def x_true(self):
        return self._x_true

def wrap_angle(angle):
    """Normalize angle to be within [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))

# Set up the simulator configuration.
sim_config = SimulatorConfiguration()
filter_config = FilterConfiguration()
map = Map()

# Create controller, simulator, and estimator.
controller = Controller(sim_config)
simulator = Simulator(sim_config, filter_config, map)
simulator.start()
estimator = RobotEstimator(filter_config, map)
estimator.start()

# Extract initial estimates and generate first control.
x_est, Sigma_est = estimator.estimate()
u = controller.next_control_input(x_est, Sigma_est)

# Arrays to store data for plotting
x_true_history = []
x_est_history = []
Sigma_est_history = []

# Main loop
for step in range(sim_config.time_steps):
    simulator.set_control_input(u)
    simulation_time = simulator.step()

    estimator.set_control_input(u)
    estimator.predict_to(simulation_time)

    # Get landmark observations (range and bearing) from simulator
    y = simulator.landmark_range_bearing_observations()

    # Update the EKF with the observations
    estimator.update_from_landmark_range_bearing(y)

    # Get the current state estimate
    x_est, Sigma_est = estimator.estimate()

    # Determine next control input
    u = controller.next_control_input(x_est, Sigma_est)

    # Store data for plotting
    x_true_history.append(simulator.x_true())
    x_est_history.append(x_est)
    Sigma_est_history.append(np.diagonal(Sigma_est))

# Convert history lists to arrays.
x_true_history = np.array(x_true_history)
x_est_history = np.array(x_est_history)
Sigma_est_history = np.array(Sigma_est_history)

# Plot true path, estimated path, and landmarks.
plt.figure()
plt.plot(x_true_history[:, 0], x_true_history[:, 1], label='True Path')
plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path')
plt.scatter(map.landmarks[:, 0], map.landmarks[:, 1], marker='x', color='red', label='Landmarks')
plt.legend()
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('Unicycle Robot Localization using EKF with Range-Bearing Sensors')
plt.axis('equal')
plt.grid(True)
plt.savefig("localization_result.png", dpi=300)
plt.show()

# Plot 2-sigma bounds for estimation error in each state.
state_names = ['x', 'y', 'θ']
estimation_error = x_est_history - x_true_history
estimation_error[:, -1] = wrap_angle(estimation_error[:, -1])
for s in range(3):
    plt.figure()
    two_sigma = 2 * np.sqrt(Sigma_est_history[1:, s])
    plt.plot(estimation_error[1:, s], label="Estimation Error")
    plt.plot(two_sigma, linestyle='dashed', color='red', label="2 Sigma Bound")
    plt.plot(-two_sigma, linestyle='dashed', color='red')
    plt.title(f"{state_names[s]} - Estimation Error with 2-Sigma Bound")
    plt.xlabel("Time Step")
    plt.ylabel(f"{state_names[s]} Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{state_names[s]}_error.png", dpi=300)
    plt.show()
