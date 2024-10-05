import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
from scipy import stats
import statsmodels.api as sm


def compute_adjusted_r_squared(y_true, y_pred, p):
    n = len(y_true)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return r2, adjusted_r2


def compute_f_statistic(y_true, y_pred, p):
    n = len(y_true)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
    return f_stat


def compute_confidence_intervals_ols(regressor_all, tau_mes_all, confidence=0.95):
    # Fit OLS model
    ols_model = sm.OLS(tau_mes_all, regressor_all).fit()

    # Get confidence intervals for the parameters
    ci_params = ols_model.conf_int(alpha=1 - confidence)

    # Get predicted values and confidence intervals for predictions
    predictions = ols_model.get_prediction(regressor_all)
    ci_pred = predictions.conf_int(alpha=1 - confidence)

    return ci_params, ci_pred


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface
    sim.GetMotorTorques
    print(sim)
    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi / 4, np.pi / 6, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4,
                  np.pi / 4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference

    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 2  # seconds

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer:
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # TODO Compute regressor and store it
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        regressor_all.append(cur_regressor)
        tau_mes_all.append(tau_mes)

        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    # Stack all the regressor data and torque measurements
    regressor_all = np.vstack(regressor_all[1::])
    tau_mes_all = np.hstack(tau_mes_all[1::])
    print(regressor_all.shape, tau_mes_all.shape)

    # Compute dynamic parameters using pseudoinverse
    a = np.linalg.pinv(regressor_all) @ tau_mes_all
    print(f"Dynamic parameters: {a}")
    print(a.shape)

    # Compute predicted torques using the learned parameters 'a'
    tau_pred_all = regressor_all @ a

    # TODO compute the metrics for the linear model
    mse = np.mean((tau_mes_all - tau_pred_all) ** 2)
    print(f"Mean Squared Error (MSE) on torque prediction: {mse:.6f}")

    # Compute confidence intervals for parameters and predictions using OLS model
    ci_params, ci_pred = compute_confidence_intervals_ols(regressor_all, tau_mes_all)
    # Print confidence intervals for parameters
    print("Confidence intervals for parameters:")
    for i, ci in enumerate(ci_params):
        print(f"Parameter {i + 1}: {ci[0]:.6f} to {ci[1]:.6f}")

    # TODO plot the  torque prediction error for each joint (optional)
    # Plot the torque prediction error for each joint
    time_values = np.linspace(0, max_time, len(tau_mes_all) // num_joints)

    step = 5
    start_idx = 250

    for i in range(num_joints):
        print(f"Joint {i + 1}")
        # Compute Adjusted R-squared
        r2, adjusted_r2 = compute_adjusted_r_squared(tau_mes_all[i::num_joints], tau_pred_all[i::num_joints],
                                                     len(a[i::num_joints]))
        print(f"R-squared: {r2:.6f}, Adjusted R-squared: {adjusted_r2:.6f}")

        # Compute F-statistics
        f_stat = compute_f_statistic(tau_mes_all[i::num_joints], tau_pred_all[i::num_joints], len(a[i::num_joints]))
        print(f"F-statistic: {f_stat:.6f}")

        # Compute the mean squared error for each joint
        mse = np.mean((tau_mes_all[i::num_joints] - tau_pred_all[i::num_joints]) ** 2)
        print(f"Mean Squared Error (MSE) on torque prediction: {mse:.6f}")

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(time_values[start_idx::step], tau_mes_all[i::num_joints][start_idx::step], label='Measured Torque')
        plt.plot(time_values[start_idx::step], tau_pred_all[i::num_joints][start_idx::step], label='Predicted Torque')
        plt.title(f'Joint {i + 1} Torque', fontsize=16)
        plt.xlabel('Time [s]', fontsize=14)
        plt.ylabel('Torque [Nm]', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=14)

        plt.subplot(2, 1, 2)
        plt.plot(time_values[start_idx::step],
                 tau_pred_all[i::num_joints][start_idx::step] - tau_mes_all[i::num_joints][start_idx::step],
                 label="prediction error")
        plt.title(f'Prediction Error (pre - mes) of Joint {i + 1}', fontsize=16)
        plt.xlabel('Time [s]', fontsize=14)
        plt.ylabel('Error [Nm]', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=14)

        plt.subplots_adjust(hspace=0.3)

        plt.show()


if __name__ == '__main__':
    main()