import os
import numpy as np
from numpy.fft import fft, fftfreq
import time
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, \
    CartesianDiffKin

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.getcwd()  # Current directory path
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


# Single joint tuning
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement, episode_duration, plot=False):
    # Reset the simulator each time we start a new test
    sim_.ResetPose()

    # Updating the kp value for the joint we want to tune
    kp_vec = np.array([1000] * dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd = np.array([0] * dyn_model.getNumberofActuatedJoints())
    # Ensure that no side effect happens, we need to copy the initial joint angles
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
    # Testing loop
    for i in range(steps):
        # Measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Control torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)

        # Get real time
        current_time += time_step

    if plot:
        plt.figure(figsize=(10, 8))
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[joints_id] for q in q_mes_all], label=f'Measured Position - Joint {joints_id + 1}')
        plt.plot([q[joints_id] for q in q_d_all], label=f'Desired Position - Joint {joints_id + 1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {joints_id + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()
        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[joints_id] for qd in qd_mes_all], label=f'Measured Velocity - Joint {joints_id + 1}')
        plt.plot([qd[joints_id] for qd in qd_d_all], label=f'Desired Velocity - Joint {joints_id + 1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {joints_id + 1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return q_mes_all, qd_mes_all, q_d_all, qd_d_all


# Function to perform frequency analysis
def perform_frequency_analysis(data, dt, plot):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n // 2]
    power = 2.0 / n * np.abs(yf[:n // 2])

    # Plot the spectrum
    if plot:
        plt.figure()
        plt.plot(xf, power)
        plt.title("FFT of the signal")
        plt.xlabel("Frequency in Hz")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    dominant_frequency = xf[np.argmax(power[1:])]
    return dominant_frequency, power


def find_cycles(data):
    peaks, _ = find_peaks(data)
    troughs, _ = find_peaks(-data)
    cycle_indices = np.sort(np.concatenate([peaks, troughs]))
    return cycle_indices


def calculate_amplitudes(data, cycle_indices):
    amplitudes = []
    for i in range(1, len(cycle_indices)):
        cycle_data = data[cycle_indices[i - 1]:cycle_indices[i]]  # 提取一个周期的数据
        amplitude = np.max(cycle_data) - np.min(cycle_data)  # 振幅 = 最大值 - 最小值
        amplitudes.append(amplitude)
    return np.array(amplitudes)


def is_amplitude_stable(amplitudes, tolerance):
    # 振幅的标准差与平均值之比来衡量振幅变化
    mean_amplitude = np.mean(amplitudes)
    std_amplitude = np.std(amplitudes)

    # 如果标准差相对于平均振幅的比例小于容忍度 tolerance，则认为振幅稳定
    return (std_amplitude / mean_amplitude) < tolerance, (std_amplitude / mean_amplitude)


def is_sustained_oscillation(data):
    # 找到周期的起点和终点（波峰和波谷）
    cycle_indices = find_cycles(data)

    # 计算每个周期的振幅
    amplitudes = calculate_amplitudes(data, cycle_indices)

    # 检查振幅是否稳定
    tolerance = 0.045
    bol, cal = is_amplitude_stable(amplitudes, tolerance)
    if bol:
        print("每个周期的振幅:", amplitudes)
        return True, cal
    else:
        return False, cal


def calculate(joint_id, init_gain, max_gain, gain_step, regulation_displacement, test_duration, dt):
    print(f'joint_id:{joint_id + 1}')
    # current_kp.append(init_gain)
    current_gain = init_gain
    res = []
    while current_gain <= max_gain:
        q_mes_all, qd_mes_all, q_d_all, qd_d_all = simulate_with_given_pid_values(sim, current_gain, joint_id,
                                                                                  regulation_displacement,
                                                                                  test_duration, plot=False)
        data = np.array([q[joint_id] for q in q_mes_all[::]])
        print(data)
        bol, cal = is_sustained_oscillation(data)
        if bol:
            res.append(cal)


            # plt.show()

            Ku = current_gain  # current_gain
            dominant_frequency, power = perform_frequency_analysis(data, dt, False)
            Tu = 1 / dominant_frequency  # dominant_frequency
            Kp_final = 0.6 * Ku
            Ki_final = 2 * Kp_final / Tu
            Kd_final = Kp_final * Tu / 8
            plt.figure(figsize=(10, 8))
            # Position plot for joint i
            plt.subplot(2, 1, 1)
            plt.plot([q[joint_id] for q in q_mes_all[10::]], label=f'Measured Position - Joint {joint_id + 1}')
            plt.plot([q[joint_id] for q in q_d_all[10::]], label=f'Desired Position - Joint {joint_id + 1}',
                     linestyle='--')
            plt.text(0.05, 0.95,
                     f'Ku = {Ku:.3f}\nTu = {Tu:.3f}\nKp = {Kp_final:.3f}\nKi = {Ki_final:.3f}\nKd = {Kd_final:.3f}',
                     transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.title(f'Position Tracking for Joint {joint_id + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Position')
            plt.legend(loc='upper right')
            # Velocity plot for joint i
            plt.subplot(2, 1, 2)
            plt.plot([qd[joint_id] for qd in qd_mes_all[10::]], label=f'Measured Velocity - Joint {joint_id + 1}')
            plt.plot([qd[joint_id] for qd in qd_d_all[10::]], label=f'Desired Velocity - Joint {joint_id + 1}',
                     linestyle='--')
            plt.title(f'Velocity Tracking for Joint {joint_id + 1}')
            plt.xlabel('Time steps')
            plt.ylabel('Velocity')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f"{joint_id + 1}", dpi=300)
            print(f'Computed PID parameters: Ku={Ku}, Tu={Tu}, Kp={Kp_final}, Ki={Ki_final}, Kd={Kd_final}')
            break
        # if is_sustained_oscillation_fft([q[joint_id] for q in q_mes_all[::]], dt):
        #     res.append([diff, current_gain, dominant_frequency])
        # else:
        current_gain += gain_step
        time.sleep(1)

    # if res:
    #     min_diff_record = min(res, key=lambda x: x[0])
    #     Ku = min_diff_record[1]  # current_gain
    #     Tu = 1 / min_diff_record[2]  # dominant_frequency
    #     Kp_final = 0.6 * Ku
    #     Ki_final = 2 * Kp_final / Tu
    #     Kd_final = Kp_final * Tu / 8
    #     print(f'Computed PID parameters: Ku={Ku}, Tu={Tu}, Kp={Kp_final}, Ki={Ki_final}, Kd={Kd_final}')


# Main routine to perform PID tuning using Ziegler-Nichols
def main():
    # joint_id = 0  # Joint ID to tune
    regulation_displacement = 0.1  # Displacement from the initial joint position
    init_gain = 1
    gain_step = 0.25
    max_gain = 18
    test_duration = 10  # in seconds

    dt = sim.GetTimeStep()

    # Loop to find Ku by increasing Kp until sustained oscillation is observed
    for joint_id in range(7):
        calculate(joint_id, init_gain, max_gain, gain_step, regulation_displacement, test_duration, dt)
        time.sleep(1)


if __name__ == "__main__":
    main()
