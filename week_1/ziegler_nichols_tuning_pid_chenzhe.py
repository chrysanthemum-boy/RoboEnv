import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")



# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd = np.array([0]*dyn_model.getNumberofActuatedJoints())
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

   
    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors


    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    

    steps = int(episode_duration/time_step)
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
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print("current time in seconds",current_time)

    
    # TODO make the plot for the current joint
    if plot:
        time_vector = np.linspace(0, episode_duration, steps)
            
        q_mes_joint = [q[joints_id] for q in q_mes_all]
        q_des_joint = [q[joints_id] for q in q_d_all]

        plt.figure()
        plt.plot(time_vector, q_mes_joint, label="Measured Position")
        plt.plot(time_vector, q_des_joint, label="Desired Position", linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Angle (rad)")
        plt.title(f"Joint {joints_id} Position vs. Time")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return q_mes_all
     



def perform_frequency_analysis(data, dt):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    # plt.figure()
    # plt.plot(xf, power)
    # plt.title("FFT of the signal")
    # plt.xlabel("Frequency in Hz")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.show()

    # Find the dominant frequency
    dominant_idx = np.argmax(power)
    dominant_frequency = xf[dominant_idx]

    # Calculate the ultimate period (Tu)
    Tu = 1 / dominant_frequency if dominant_frequency != 0 else float('inf')

    return xf, power, Tu


# TODO Implement the table in thi function

def check_sustained_oscillation(power, threshold=0.1):
    # Check if the maximum power exceeds the threshold, indicating sustained oscillation
    return max(power) > threshold


if __name__ == '__main__':
    # joint_id = 0  # Joint ID to tune
    # regulation_displacement = 1.0  # Displacement from the initial joint position
    # init_gain=15.9
    # gain_step=1.5
    # max_gain=10000
    # test_duration=20 # in seconds
    

    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
    # simulate_with_given_pid_values(sim, init_gain, joint_id, regulation_displacement, test_duration, plot=True)

    joint_id = 0  # Joint ID to tune
    regulation_displacement = 1.0  # Displacement from the initial joint position
    init_gain = 10  # Initial gain value for Kp
    gain_step = 1.5  # Step multiplier for increasing Kp
    max_gain = 20  # Maximum value for Kp
    test_duration = 20  # Duration of each test in seconds

    current_gain = init_gain
    Ku = None
    Tu = None

    while current_gain <= max_gain:
        print(f"Testing Kp = {current_gain}")
        q_mes_all = simulate_with_given_pid_values(sim, current_gain, joint_id, regulation_displacement, test_duration,
                                                   plot=False)

        # Perform frequency analysis to determine if the system is oscillating
        dt = sim.GetTimeStep()
        _, power, Tu_current = perform_frequency_analysis([q[joint_id] for q in q_mes_all], dt)

        # Check if the power spectrum indicates sustained oscillations
        if check_sustained_oscillation(power):
            print(f"Sustained oscillation detected at Kp = {current_gain}")
            print(f"Ultimate period (Tu) = {Tu_current} seconds")
            Ku = current_gain
            Tu = Tu_current
            break

        current_gain += gain_step

    if Ku is not None and Tu is not None:
        print(f"Final Ku: {Ku}, Tu: {Tu}")
        # Compute PID parameters using Ziegler-Nichols method
        Kp = 0.6 * Ku
        Ti = 0.5 * Tu
        Td = 0.125 * Tu
        Ki = Kp / Ti
        Kd = Kp * Td

        print(f"Computed PID parameters using Ziegler-Nichols method:")
        print(f"Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
