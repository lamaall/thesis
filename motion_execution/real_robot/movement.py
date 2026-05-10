import time
from numpy import array, linalg, deg2rad
from nicomotion.Motion import Motion
import pybullet as p
from scipy.interpolate import Rbf
from math import atan2, sqrt

# ----------------------------
# CONFIG
# ----------------------------

RESET_SPEED = 0.02
motorConfig = './robot_config/nico_humanoid_upper_rh7d_ukba.json'

# Initial pose for reset
init_pos = {
    'r_shoulder_z': -1.54,
    'r_shoulder_y': 19.21,
    'r_arm_x': 35.03,
    'r_elbow_y': 98.15,
    'r_wrist_z': -21.67,
    'r_wrist_x': -47.69,
    'r_thumb_z': -140.10,
    'r_thumb_x': -180.0,
    'r_indexfinger_x': -60.71,
    'r_middlefingers_x': -190.81,
}

end_pos = {
    'r_shoulder_z': -5.23,
    'r_shoulder_y': 16.75,
    'r_arm_x': 25.27,
    'r_elbow_y': 108.88,
    'r_wrist_z': -41.10,
    'r_wrist_x': -57.19,
    'r_thumb_z': -148.35,
    'r_thumb_x': -180.00,
    'r_indexfinger_x': -64.57,
    'r_middlefingers_x': -180.00
}

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------

def rad2nicodeg(joint, rad):
    if joint == 'r_wrist_z':
        return rad * 180.0 / 3.14159 * 2
    elif joint == 'r_wrist_x':
        return rad * 180.0 / 3.14159 * 4
    else:
        return rad * 180.0 / 3.14159


def speed_control(initial, target, duration):
    return abs((float(initial) - float(target)) / float(1260 * duration))


def reset_robot(robot, positions):
    for joint, angle in positions.items():
        robot.setAngle(joint, angle, RESET_SPEED)

def yaw_from_xy(x, y, xmin=0.3, xmax=0.475, ymin=-0.25, ymax=0.25):
    x_clamped = max(min(x, xmax), xmin)
    y_clamped = max(min(y, ymax), ymin)
    tx = (x_clamped - xmin) / (xmax - xmin)
    ty = (y_clamped - ymin) / (ymax - ymin)
    yaw_deg = 45.0 * (2.0 - tx) * ty
    return deg2rad(yaw_deg)

def estimate_duration(current_angles, target_angles,
                      max_speed_deg_per_sec=145.0,
                      safety_factor=1.4,
                      min_duration=0.025,
                      max_duration=1.5):
    
    diffs = [abs(t - c) for t, c in zip(target_angles, current_angles)]
    max_diff = max(diffs)

    duration = (max_diff / max_speed_deg_per_sec) * safety_factor

    # clamp duration
    duration = max(min(duration, max_duration), min_duration)

    return duration

# ----------------------------
# MAIN
# ----------------------------

def robot_drawing(target_points):
    try:
        robot = Motion(motorConfig=motorConfig)
        print("Robot initialized.")
    except:
        print("Motors not operational.")
        return

    # Reset pose
    reset_robot(robot, init_pos)
    time.sleep(2)
    print("Robot reset complete.")

    # ----------------------------
    # RBF setup
    # ----------------------------
    RIGHTHAND_RBF_POINTS = {
        'x': [0.133, 0.208, 0.272, 0.338, 0.388, 0.420, 0.458, 0.483, 0.496, 0.480, 0.464, 0.433, 0.392, 0.344, 0.291, 0.227, 0.164, 0.152, 0.139, 0.123, 0.104, 0.111, 0.085, 0.069, 0.085, 0.158, 0.177, 0.193, 0.202, 0.218, 0.234, 0.313, 0.300, 0.278, 0.268, 0.240, 0.227, 0.313, 0.344, 0.366, 0.376, 0.385, 0.407, 0.426, 0.426, 0.272, 0.369],
        'y': [-0.395, -0.437, -0.447, -0.411, -0.358, -0.289, -0.216, -0.142, -0.074, -0.005, 0.068, 0.142, 0.200, 0.247, 0.279, 0.305, 0.321, 0.258, 0.184, 0.100, 0.005, -0.084, -0.163, -0.253, -0.326, -0.289, -0.189, -0.089, 0.016, 0.116, 0.205, 0.158, 0.053, -0.053, -0.153, -0.242, -0.342, -0.316, -0.216, -0.121, -0.026, 0.079, -0.179, -0.089, 0.005, -0.384, -0.274],
        'yaw': [-0.595, -0.728, -0.794, -0.761, -0.694, -0.463, -0.265, -0.132, -0.033, 0.165, 0.331, 0.529, 0.728, 0.893, 1.058, 1.224, 1.455, 1.389, 1.389, 1.389, 1.819, 1.257, 0.496, -0.132, -0.298, 0.0, 0.331, 0.661, 1.29, 1.488, 1.29, 1.157, 1.124, 0.827, 0.496, 0.165, -0.298, -0.198, 0.165, 0.331, 0.43, 0.529, -0.066, 0.198, 0.331, -0.496, -0.331],
        'z': [0.115, 0.123, 0.128, 0.13, 0.132, 0.131, 0.128, 0.129, 0.127, 0.127, 0.126, 0.124, 0.12, 0.114, 0.115, 0.107, 0.091, 0.095, 0.035, 0.038, 0.063, 0.115, 0.105, 0.101, 0.095, 0.092, 0.089, 0.096, 0.102, 0.106, 0.101, 0.098, 0.101, 0.104, 0.112, 0.118, 0.112, 0.109, 0.108, 0.11, 0.12, 0.118, 0.117, 0.122, 0.123]
    }

    righthand_rbf_z = Rbf(
        RIGHTHAND_RBF_POINTS['x'][0:18] + RIGHTHAND_RBF_POINTS['x'][20:],
        RIGHTHAND_RBF_POINTS['y'][0:18] + RIGHTHAND_RBF_POINTS['y'][20:],
        RIGHTHAND_RBF_POINTS['z'],
        function='multiquadric'
    )   

    # ----------------------------
    # PyBullet IK
    # ----------------------------
    physics_client = p.connect(p.DIRECT)
    robot_id = p.loadURDF("./robot_config/urdf/nico_upper_rh6d_r.urdf", [0,0,0], useFixedBase=True)

    joint_indices = []
    joint_names = []
    end_effector_index = None

    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        q_index = info[3]
        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")

        if q_index > -1:
            joint_names.append(joint_name)
            joint_indices.append(j)

        if link_name == "endeffector":
            end_effector_index = j

    test_z_value = 0.07

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    for target_pos in target_points:
        x, y, z = target_pos

        yaw = yaw_from_xy(x, y)
        if test_z_value == z:
            z = righthand_rbf_z(x, y) - 0.0475 + (y+0.2)*0.01
        else:
            z = righthand_rbf_z(x, y) - 0.001

        target_pos = [x, y, z]

        palm_quat = p.getQuaternionFromEuler([3.28, 2.6, yaw])

        ik_solution = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_pos,
            targetOrientation=palm_quat,
            maxNumIterations=2000,
            residualThreshold=1e-6
        )

        target_angles = [rad2nicodeg(j, a) for j, a in zip(joint_names, ik_solution)]
        current_angles = [robot.getAngle(j) for j in joint_names]

        movement_duration = estimate_duration(current_angles, target_angles)

        for j_idx, joint_name in enumerate(joint_names):
            target_angle = target_angles[j_idx]
            current_angle = current_angles[j_idx]
            speed = speed_control(current_angle, target_angle, movement_duration)
            robot.setAngle(joint_name, target_angle, speed)

        time.sleep(movement_duration * 1.1)

        print(f"Reached target: {target_pos} in {movement_duration:.2f}s")

    # Reset again
    reset_robot(robot, init_pos)
    time.sleep(2)

    reset_robot(robot, end_pos)
    time.sleep(2)
    print("Robot reset complete.")

    p.disconnect()

    print("\nDisabling torque in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)

    print("Disabling torque...")
    for joint in robot.getJointNames():
        robot.disableTorque(joint)

    print("Torque disabled.")