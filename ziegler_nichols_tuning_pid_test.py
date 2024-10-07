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
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    # Reset the simulator each time we start a new test
    sim_.ResetPose()

    # Updating the kp value for the joint we want to tune
    # kp_vec = np.array([1000] * dyn_model.getNumberofActuatedJoints())
    # kp_vec[joints_id] = kp

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

    # if plot:
    #     plt.plot(np.linspace(0, episode_duration, steps), [q[joints_id] for q in q_mes_all])
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Joint Angle (rad)')
    #     plt.title(f'Joint {joints_id} Response with Kp={kp}')
    #     plt.show()
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

    return q_mes_all


# Function to perform frequency analysis
def perform_frequency_analysis(data, dt, plot):
    n = len(data)
    # print(n)
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
    # 将最大值设为负无穷大，忽略它
    max_index = np.argmax(power)

    power_next = power
    power_next[max_index] = -np.inf

    # 找到第二大值的索引
    second_max_index = np.argmax(power_next)
    dominant_frequency = xf[second_max_index]
    fft_data = power_next[second_max_index]
    return dominant_frequency, power, fft_data


# # Function to check sustained oscillation
# def check_sustained_oscillation(data, threshold=0.05):
#     # Check if the response oscillates around zero with a consistent amplitude
#     mean_value = np.mean(data)
#     oscillation = np.abs(data - mean_value) > threshold
#     return np.any(oscillation)


# Function to check sustained oscillation with periodicity (like sine or cosine waves)
def check_sustained_oscillation(data, threshold=0.05, min_peak_distance=5):
    # 1. 计算数据的平均值
    mean_value = np.mean(data)

    # 2. 去掉平均值，得到振荡信号（去偏移）
    oscillation_signal = data - mean_value

    # 3. 寻找局部峰值，确保有周期性振荡的峰值存在
    peaks, _ = find_peaks(oscillation_signal, height=threshold, distance=min_peak_distance)

    # 4. 如果峰值数量足够多，说明存在持续的周期性振荡
    if len(peaks) > 1:
        return True  # 存在类似正弦/余弦的周期性振荡
    else:
        return False  # 没有足够的周期性振荡


# Check if the dominant frequency corresponds to a sine wave
def is_sine_wave(power, threshold=0.0999):
    # Check if power is highly concentrated at the dominant frequency
    # power = power[1:]
    power_ratio = np.max(power) / np.sum(power)  # Ratio of dominant power to total power

    # If dominant frequency contains most of the signal's energy, it's likely a sine wave
    if power_ratio > threshold:
        return True  # Signal is likely a sine wave
    else:
        return False  # Signal is likely not a pure sine wave


def calculate_sim(data):
    # 计算自相关
    auto_corr = np.correlate(data, data, mode='full')
    lags = np.arange(-len(data) + 1, len(data))

    # 找到自相关的重复峰值
    peaks = np.where((auto_corr > np.mean(auto_corr)) & (np.diff(np.sign(np.diff(auto_corr))) < 0))[0]

    # 判断自相关的重复峰值是否稳定
    if len(peaks) > 3 and np.std(np.diff(peaks)) < 0.1 * len(data):
        return True
    else:
        return False


def is_sustained_oscillation_fft(data, dt, threshold_ratio=9.9):
    # Step 1: 进行傅里叶变换
    N = len(data)
    fft_values = fft(data)
    frequencies = fftfreq(N, dt)

    # 取频谱的幅值
    magnitude = np.abs(fft_values[:N // 2])  # 只取前半部分
    freqs = frequencies[:N // 2]

    # Step 2: 查找最大幅值和背景噪声
    peak_magnitude = np.max(magnitude[1:])  # 最大幅值，跳过第一个直流分量
    mean_background = np.mean(magnitude)  # 频谱平均值，作为噪声基准

    # Step 3: 判断峰值是否显著
    if peak_magnitude / mean_background > threshold_ratio:
        return True  # 存在显著的周期性成分，认为是持续振荡
    else:
        return False  # 没有显著的周期性成分，不是持续振荡


def calculate(joint_id, init_gain, max_gain, gain_step, regulation_displacement, test_duration, dt):
    print(f'joint_id:{joint_id + 1}')
    # current_kp.append(init_gain)
    current_gain = init_gain
    res = []
    while current_gain <= max_gain:
        q_mes_all = simulate_with_given_pid_values(sim, current_gain, joint_id, regulation_displacement,
                                                   test_duration,
                                                   plot=False)

        dominant_frequency, power, fft_data = perform_frequency_analysis([q[joint_id] for q in q_mes_all[::]], dt,
                                                                         False)
        diff = abs(fft_data - regulation_displacement)
        # print(power[1:])
        # if diff < 0.005:
        if is_sustained_oscillation_fft([q[joint_id] for q in q_mes_all[::]], dt):
            res.append([diff, current_gain, dominant_frequency])

        current_gain += gain_step
        # time.sleep(1)

    if res:
        min_diff_record = min(res, key=lambda x: x[0])
        Ku = min_diff_record[1]  # current_gain
        Tu = 1 / min_diff_record[2]  # dominant_frequency
        Kp_final = 0.6 * Ku
        Ki_final = 2 * Kp_final / Tu
        Kd_final = Kp_final * Tu / 8
        print(f'Computed PID parameters: Ku={Ku}, Tu={Tu}, Kp={Kp_final}, Ki={Ki_final}, Kd={Kd_final}')


# Main routine to perform PID tuning using Ziegler-Nichols
def main():
    # joint_id = 0  # Joint ID to tune
    regulation_displacement = 0.1  # Displacement from the initial joint position
    init_gain = 1
    gain_step = 0.1
    max_gain = 18
    test_duration = 10  # in seconds

    dt = sim.GetTimeStep()
    # current_kp = []

    # Loop to find Ku by increasing Kp until sustained oscillation is observed
    for joint_id in range(7):
        calculate(joint_id, init_gain, max_gain, gain_step, regulation_displacement, test_duration, dt)
    # calculate(1, init_gain, max_gain, gain_step, regulation_displacement, test_duration, dt)


if __name__ == "__main__":
    main()
    # [17.5, 13, 10, 4, 17.5, 17.5, 17.5]
    # joint_id:1
    # Computed PID parameters: Ku=17.5, Kp=10.5, Ki=14.700000000000003, Kd=1.8749999999999998
    # joint_id:2
    # Computed PID parameters: Ku=13.0, Kp=7.8, Ki=9.360000000000001, Kd=1.6249999999999998
    # joint_id:3
    # Computed PID parameters: Ku=10.0, Kp=6.0, Ki=6.0, Kd=1.5
    # joint_id:4
    # Computed PID parameters: Ku=4.0, Kp=2.4, Ki=1.4400000000000002, Kd=0.9999999999999999
    # joint_id:5
    # Computed PID parameters: Ku=17.5, Kp=10.5, Ki=14.700000000000003, Kd=1.8749999999999998
    # joint_id:6
    # Computed PID parameters: Ku=17.5, Kp=10.5, Ki=14.700000000000003, Kd=1.8749999999999998
    # joint_id:7
    # Computed PID parameters: Ku=17.5, Kp=10.5, Ki=14.700000000000003, Kd=1.8749999999999998

    # joint_id:1
    # Computed PID parameters: Ku=18.0, Tu=1.4285714285714284, Kp=10.799999999999999, Ki=15.120000000000001, Kd=1.9285714285714282
    # joint_id:2
    # Computed PID parameters: Ku=13.0, Tu=1.6666666666666665, Kp=7.8, Ki=9.360000000000001, Kd=1.6249999999999998
    # joint_id:3
    # Computed PID parameters: Ku=9.0, Tu=2.0, Kp=5.3999999999999995, Ki=5.3999999999999995, Kd=1.3499999999999999
    # joint_id:4
    # Computed PID parameters: Ku=2.0, Tu=5.0, Kp=1.2, Ki=0.48, Kd=0.75
    # joint_id:5
    # Computed PID parameters: Ku=18.0, Tu=1.4285714285714284, Kp=10.799999999999999, Ki=15.120000000000001, Kd=1.9285714285714282
    # joint_id:6
    # Computed PID parameters: Ku=17.5, Tu=1.4285714285714284, Kp=10.5, Ki=14.700000000000003, Kd=1.8749999999999998
    # joint_id:7
    # Computed PID parameters: Ku=17.5, Tu=1.6666666666666665, Kp=10.5, Ki=12.600000000000001, Kd=2.1875

    # joint_id:1
    # Computed PID parameters: Ku=17.899999999999988, Tu=1.4285714285714284, Kp=10.739999999999993, Ki=15.035999999999992, Kd=1.9178571428571414
    # joint_id:2
    # Computed PID parameters: Ku=13.29999999999997, Tu=1.6666666666666665, Kp=7.979999999999982, Ki=9.57599999999998, Kd=1.662499999999996
    # joint_id:3
    # Computed PID parameters: Ku=9.299999999999985, Tu=2.0, Kp=5.57999999999999, Ki=5.57999999999999, Kd=1.3949999999999976
    # joint_id:4
    # Computed PID parameters: Ku=4.8, Tu=2.5, Kp=2.88, Ki=2.304, Kd=0.8999999999999999
    # joint_id:5
    # Computed PID parameters: Ku=17.899999999999988, Tu=1.4285714285714284, Kp=10.739999999999993, Ki=15.035999999999992, Kd=1.9178571428571414
    # joint_id:6
    # Computed PID parameters: Ku=17.799999999999986, Tu=1.4285714285714284, Kp=10.67999999999999, Ki=14.95199999999999, Kd=1.9071428571428553
    # joint_id:7
    # Computed PID parameters: Ku=17.899999999999988, Tu=1.4285714285714284, Kp=10.739999999999993, Ki=15.035999999999992, Kd=1.9178571428571414