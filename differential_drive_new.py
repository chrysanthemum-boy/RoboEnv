import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, \
    CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller, regulation_polar_coordinates, \
    regulation_polar_coordinate_quat, wrap_angle, velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model_new import RegulatorModel

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
    [5, 10],
    [15, 5],
    [10, 15]
])


def landmark_range_observations(base_position):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx ** 2 + dy ** 2)
        y.append(range_meas)
    y = np.array(y)
    return y


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized
    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()
    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]
    # Extract the yaw angle
    bearing_ = base_euler[2]
    return bearing_


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    return sim, dyn_model, num_joints


def main():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim, dyn_model, num_joints = init_simulator(conf_file_name)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    # Initialize data storage
    base_pos_all, base_bearing_all = [], []

    # initializing MPC
    num_states = 3
    num_controls = 2

    # Measuring all the state
    C = np.eye(num_states)

    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    init_pos = np.array([2.0, 3.0])
    init_quat = np.array([0, 0, 0.3827, 0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)

    # Define the cost matrices
    Qcoeff = np.array([310, 310, 80.0])
    Rcoeff = 0.5
    regulator.setCostMatrices(Qcoeff, Rcoeff)

    u_mpc = np.zeros(num_controls)
    wheel_radius = 0.11
    wheel_base_width = 0.46

    v_linear = 0.0
    v_angular = 0.0
    cmd = MotorCommands()
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    state_dim = 3  # x, y, 和 bearing
    control_dim = 2  # v_linear, v_angular

    # 初始状态估计 (x, y, bearing) 和误差协方差矩阵
    x_est = np.array([init_pos[0], init_pos[1], init_base_bearing_])
    P_est = np.eye(state_dim) * 0.1  # 初始误差协方差矩阵

    # 过程噪声协方差矩阵 (根据系统动态调整)
    Q = np.diag([0.1, 0.1, 0.05])

    # 测量噪声协方差矩阵 (基于范围噪声方差)
    R = np.eye(len(landmarks)) * W_range

    while True:
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Kalman filter prediction (implementation placeholder)
        # -- TODO: Implement Kalman update here if needed.
        # 预测步骤
        F = np.eye(state_dim)  # 简单线性模型
        B = np.zeros((state_dim, control_dim))  # 控制输入对位置无直接影响
        B[0, 0] = np.cos(x_est[2]) * time_step  # 线速度对 x 的影响
        B[1, 0] = np.sin(x_est[2]) * time_step  # 线速度对 y 的影响
        B[2, 1] = time_step  # 角速度对航向角的影响

        # 根据当前控制输入 u_mpc 预测下一个状态
        x_pred = F @ x_est + B @ u_mpc
        P_pred = F @ P_est @ F.T + Q

        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        # 更新步骤
        # 根据当前状态预测的观测量，初始化测量矩阵 H
        H = np.zeros((len(landmarks), state_dim))
        y_meas = landmark_range_observations(base_pos)  # 获取真实测量值
        # y_meas = landmark_range_observations(x_pred[:2])  # 获取预测测量值
        y_pred = []  # 预测观测值
        for lm_idx, lm in enumerate(landmarks):
            dx = lm[0] - x_pred[0]
            dy = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx ** 2 + dy ** 2)
            y_pred.append(range_pred)

            # 更新测量矩阵 H 对应项
            H[lm_idx, :] = np.array([-dx / range_pred, -dy / range_pred, 0])

        # 创新/残差
        y_diff = y_meas - y_pred

        # 卡尔曼增益
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 更新状态估计和协方差矩阵
        x_est = x_pred + K @ y_diff
        P_est = (np.eye(state_dim) - K @ H) @ P_pred

        # 将更新后的 x_est 作为当前的状态估计，继续后续的控制计算
        cur_state_x_for_linearization = x_est
        # Measurements without noise for comparison
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1],
                                                    base_ori_no_noise[2])

        # Real measurements with noise
        # base_pos = sim.GetBasePosition()
        # base_ori = sim.GetBaseOrientation()
        # base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        # y = landmark_range_observations(base_pos)

        # cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = np.hstack((base_pos[:2], base_bearing_)).flatten()
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[:num_controls]
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(u_mpc[0], u_mpc[1],
                                                                                       wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array(
            [right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        keys = sim.GetPyBulletClient().getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)

        current_time += time_step

    # Plotting section for visualizing x, y, theta trajectory
    # 绘制真实路径、估计路径和地标位置
    plt.figure()
    plt.plot(np.array(base_pos_all)[:, 0], np.array(base_pos_all)[:, 1], label='Estimated Path')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color='red', label='Landmarks')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('Robot Localization and Trajectory Tracking')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('robot_trajectory.png',dpi=300)
    # plt.show()

    # 定义角度包装函数，处理角度 "环绕" 问题
    def wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    # 如果有真实路径数据（如 x_true_history），可以绘制真实路径和估计路径
    # 计算估计误差和角度包装
    x_est_history = np.array(base_pos_all)
    x_true_history = np.array(
        [[sim.bot[0].base_position[0], sim.bot[0].base_position[1], sim.bot[0].base_position[2]] for _ in base_pos_all])  # 假设仿真中可用真实路径
    estimation_error = x_est_history - x_true_history
    estimation_error[:, -1] = wrap_angle(np.array(base_bearing_all) - base_bearing_no_noise_)

    # 绘制每个状态 (x, y, θ) 的估计误差和 2σ 区间
    state_name = ['x', 'y', 'θ']
    for s in range(3):
        plt.figure()
        # print("jjj")
        two_sigma = 2 * np.sqrt(W_range)  # 用观测噪声方差 W_range 计算 2σ，假设协方差矩阵不可用
        plt.plot(estimation_error[:, s], label='Estimation Error')
        plt.plot([two_sigma] * len(estimation_error), linestyle='dashed', color='red', label='2σ Bound')
        plt.plot([-two_sigma] * len(estimation_error), linestyle='dashed', color='red')
        plt.xlabel('Time Step')
        plt.ylabel(f'{state_name[s]} Error')
        plt.title(f'Estimation Error and 2σ Bound for {state_name[s]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'estimation_error_{state_name[s]}.png',dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
