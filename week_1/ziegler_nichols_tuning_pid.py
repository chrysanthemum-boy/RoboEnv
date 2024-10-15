import os
import numpy as np
from numpy.fft import fft, fftfreq
import time
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, \
    CartesianDiffKin
from scipy.signal import find_peaks

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
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, = [], [], [], []

    steps = int(episode_duration / time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        # cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        # regressor_all = np.vstack((regressor_all, cur_regressor))

        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print("current time in seconds",current_time)

    # TODO make the plot for the current joint
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


# TODO Implement the table in thi function
# [16.625, 17, 9, 2.5]

def cal_amplitudes_num(data):
    # 找到波峰和波谷
    peaks, _ = find_peaks(data)  # 找到波峰
    troughs, _ = find_peaks(-data)  # 找到波谷

    # 确保波峰和波谷的数量一致，匹配对应的周期
    min_length = min(len(peaks), len(troughs))
    peaks = peaks[:min_length]
    troughs = troughs[:min_length]

    # 计算每个周期的振幅
    amplitudes = data[peaks] - data[troughs]  # 每个周期的振幅是波峰减去波谷
    num = 0
    if amplitudes > 0.19:
        num += 1


# --- 步骤1：找到波峰和波谷 ---

def find_cycles(data):
    # 找到波峰和波谷
    peaks, _ = find_peaks(data)
    troughs, _ = find_peaks(-data)

    # 确保波峰和波谷是交替出现的
    cycle_indices = np.sort(np.concatenate([peaks, troughs]))

    return cycle_indices


# --- 步骤2：计算每个周期的振幅 ---

def calculate_amplitudes(data, cycle_indices):
    amplitudes = []
    for i in range(1, len(cycle_indices)):
        cycle_data = data[cycle_indices[i - 1]:cycle_indices[i]]  # 提取一个周期的数据
        amplitude = np.max(cycle_data) - np.min(cycle_data)  # 振幅 = 最大值 - 最小值
        amplitudes.append(amplitude)
    return np.array(amplitudes)


# --- 步骤3：检查振幅是否稳定 ---

def is_amplitude_stable(amplitudes, tolerance):
    # 振幅的标准差与平均值之比来衡量振幅变化
    mean_amplitude = np.mean(amplitudes)
    std_amplitude = np.std(amplitudes)

    # 如果标准差相对于平均振幅的比例小于容忍度 tolerance，则认为振幅稳定
    return (std_amplitude / mean_amplitude) < tolerance


# --- 运行分析 ---
def is_chixu(data):
    # 找到周期的起点和终点（波峰和波谷）
    cycle_indices = find_cycles(data)

    # 计算每个周期的振幅
    amplitudes = calculate_amplitudes(data, cycle_indices)

    # 检查振幅是否稳定
    tolerance = 0.05  # 设定容忍度，表示允许的振幅变化比例
    if is_amplitude_stable(amplitudes, tolerance):
        print("振幅在多个周期内是稳定的。")
    else:
        print("振幅在多个周期内不稳定。")

    # 输出每个周期的振幅
    print("每个周期的振幅:", amplitudes)


if __name__ == '__main__':
    joint_id = 4  # Joint ID to tune
    regulation_displacement = 0.1  # Displacement from the initial joint position
    init_gain = 16.5  # Kp
    gain_step = 1.5
    max_gain = 1000  # Ku
    test_duration = 10  # in seconds
    q_mes_all = simulate_with_given_pid_values(sim, init_gain, joint_id, regulation_displacement, test_duration, True)

    data = np.array([q[joint_id] for q in q_mes_all[::]])

    data_max = max([q[joint_id] for q in q_mes_all[::]])
    data_min = min([q[joint_id] for q in q_mes_all[::]])
    print((data_max - data_min) / 2)
    data_std = np.std([q[joint_id] for q in q_mes_all[::]])
    print(data_std)
    is_chixu(data)
    # 0.0698
    # print(len(q_mes_all[0]))
    # while init_gain < max_gain:
    #     print(f"Initial gain: {init_gain}")
    #     simulate_with_given_pid_values(sim, init_gain, joint_id, regulation_displacement, test_duration, True)
    #     time.sleep(11)
    #     init_gain += gain_step
    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
    # simulate_with_given_pid_values(sim, init_gain, joint_id, regulation_displacement,test_duration,True)
