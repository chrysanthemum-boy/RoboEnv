import os
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, \
    CartesianDiffKin

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")


# single joint tuning
# episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()

    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000] * dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd = np.array([0] * dyn_model.getNumberofActuatedJoints())
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0] * dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

    steps = int(episode_duration / time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)

        # get real time
        current_time += time_step

    # Optional plotting
    if plot:
        num_joints = len(q_mes)
        for i in range(num_joints):
            plt.figure(figsize=(10, 8))

            # Position plot for joint i
            plt.subplot(2, 1, 1)
            plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i + 1}')
            plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i + 1}', linestyle='--')
            plt.title(f'Position Tracking for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Position')
            plt.legend()

            # Velocity plot for joint i
            plt.subplot(2, 1, 2)
            plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i + 1}')
            plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i + 1}', linestyle='--')
            plt.title(f'Velocity Tracking for Joint {i + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend()

            plt.tight_layout()
            plt.show()

        # plt.figure()
        # plt.plot(np.linspace(0, episode_duration, len(q_mes_all)), [q[joints_id] for q in q_mes_all])
        # plt.xlabel("Time [s]")
        # plt.ylabel("Joint Angle [rad]")
        # plt.title(f"Joint {joints_id} Angle with Kp = {kp}")
        # plt.grid(True)
        # plt.show()

    return q_mes_all


def perform_frequency_analysis(data, dt):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n // 2]
    power = 2.0 / n * np.abs(yf[:n // 2])

    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf, power)
    plt.title("FFT of the signal")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return xf, power


def tune_pid_using_ziegler_nichols(sim_, joint_id, initial_kp=1000, gain_step=1.5, max_gain=10000, test_duration=20):
    kp = initial_kp
    dt = sim_.GetTimeStep()

    while kp < max_gain:
        # Simulate with the current Kp value
        q_mes_all = simulate_with_given_pid_values(sim_, kp, joint_id, regulation_displacement=0.1,
                                                   episode_duration=test_duration, plot=False)

        # Perform frequency analysis to check for sustained oscillation
        joint_data = [q[joint_id] for q in q_mes_all]
        xf, power = perform_frequency_analysis(joint_data, dt)

        # Check if there is a clear peak in the frequency spectrum indicating sustained oscillation
        dominant_frequency_index = np.argmax(power)
        dominant_frequency = xf[dominant_frequency_index]

        if power[dominant_frequency_index] > 0.1:  # Threshold to identify sustained oscillation
            print(f"Sustained oscillation detected at Kp = {kp}, Dominant Frequency = {dominant_frequency} Hz")
            Ku = kp
            Tu = 1 / dominant_frequency

            # Calculate PID parameters using Ziegler-Nichols method
            Kp = 0.6 * Ku
            Ki = 1.2 * Ku / Tu
            Kd = 0.075 * Ku * Tu
            print(f"Calculated PID parameters: Kp = {Kp}, Ki = {Ki}, Kd = {Kd}")
            return Kp, Ki, Kd

        # Increase Kp and continue testing
        kp *= gain_step

    print("No sustained oscillation detected within the given gain range.")
    return None, None, None


if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune
    initial_kp = 1000  # Initial gain value
    gain_step = 1.5  # Gain increment factor
    max_gain = 10000  # Maximum gain to test
    test_duration = 10  # Duration for each test in seconds

    # Use Ziegler-Nichols method to tune PID parameters
    Kp, Ki, Kd = tune_pid_using_ziegler_nichols(sim, joint_id, initial_kp, gain_step, max_gain, test_duration)
    if Kp is not None:
        print(f"Final tuned PID parameters: Kp = {Kp}, Ki = {Ki}, Kd = {Kd}")
    else:
        print("Failed to find suitable PID parameters.")