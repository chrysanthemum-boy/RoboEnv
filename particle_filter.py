#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class FilterConfiguration:
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        self.W_range = 0.5 ** 2  # Range measurement noise variance
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2  # Bearing measurement noise variance
        self.x0 = np.array([2.0, 3.0, np.pi / 4])  # Initial state
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2  # Initial covariance


class Map:
    def __init__(self):
        # Landmark positions
        self.landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])


class ParticleFilter:
    def __init__(self, num_particles, map_landmarks, initial_pose, process_noise, measurement_noise):
        self.num_particles = num_particles
        self.particles = np.tile(initial_pose, (num_particles, 1)) + np.random.normal(0, 0.5, (num_particles, 3))
        self.weights = np.ones(num_particles) / num_particles
        self.map_landmarks = map_landmarks
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_input, dt):
        """Predict step of particle filter based on control input."""
        linear_velocity, angular_velocity = control_input
        noise = np.random.normal(0, self.process_noise, (self.num_particles, 3))

        # Update each particle's position with noise
        self.particles[:, 0] += (linear_velocity * np.cos(self.particles[:, 2]) * dt) + noise[:, 0]
        self.particles[:, 1] += (linear_velocity * np.sin(self.particles[:, 2]) * dt) + noise[:, 1]
        self.particles[:, 2] += angular_velocity * dt + noise[:, 2]
        self.particles[:, 2] = np.mod(self.particles[:, 2], 2 * np.pi)  # Normalize angle

    def update(self, measurements):
        """Update particle weights based on sensor measurements."""
        for i, landmark in enumerate(self.map_landmarks):
            dx = landmark[0] - self.particles[:, 0]
            dy = landmark[1] - self.particles[:, 1]
            predicted_range = np.sqrt(dx ** 2 + dy ** 2)
            predicted_bearing = np.arctan2(dy, dx) - self.particles[:, 2]

            # Calculate differences and normalize bearing difference
            range_diff = measurements[i][0] - predicted_range
            bearing_diff = measurements[i][1] - predicted_bearing
            bearing_diff = (bearing_diff + np.pi) % (2 * np.pi) - np.pi

            # Update weights using Gaussian probability density
            range_weight = np.exp(-0.5 * (range_diff ** 2) / self.measurement_noise[0] ** 2)
            bearing_weight = np.exp(-0.5 * (bearing_diff ** 2) / self.measurement_noise[1] ** 2)
            self.weights *= range_weight * bearing_weight

        # Normalize weights to avoid numerical instability
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on weights."""
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """Estimate position based on weighted mean of particles."""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        return mean


class RobotSimulator:
    def __init__(self, config, particle_filter):
        self.config = config
        self.pf = particle_filter
        self.dt = 0.1
        self.control_input = [1.0, 0.1]  # Example control input (linear and angular velocities)

    def simulate(self, time_steps):
        """Simulate robot's movement and perform particle filtering."""
        trajectory = []
        for t in range(time_steps):
            # Predict step for particle filter
            self.pf.predict(self.control_input, self.dt)

            # Generate measurements with noise
            measurements = []
            for lm in self.pf.map_landmarks:
                dx, dy = lm[0] - self.pf.estimate()[0], lm[1] - self.pf.estimate()[1]
                distance = np.sqrt(dx ** 2 + dy ** 2) + np.random.normal(0, self.config.W_range ** 0.5)
                bearing = np.arctan2(dy, dx) - self.pf.estimate()[2] + np.random.normal(0, self.config.W_bearing ** 0.5)
                measurements.append([distance, bearing])

            # Update and resample particles
            self.pf.update(measurements)
            self.pf.resample()

            # Record estimated position for trajectory
            trajectory.append(self.pf.estimate()[:2])

        return np.array(trajectory)


# Initialize configuration, map, and particle filter
config = FilterConfiguration()
landmarks = Map().landmarks
initial_pose = config.x0
pf = ParticleFilter(
    num_particles=500,
    map_landmarks=landmarks,
    initial_pose=initial_pose,
    process_noise=np.array([0.1, 0.1, 0.05]),
    measurement_noise=np.array([config.W_range ** 0.5, config.W_bearing ** 0.5])
)

# Run the simulation
simulator = RobotSimulator(config, pf)
trajectory = simulator.simulate(time_steps=100)

# Plot the estimated trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Estimated Path")
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='x', label="Landmarks")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Particle Filter Estimated Trajectory")
plt.show()
