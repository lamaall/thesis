import pybullet as p
import time

from numpy import deg2rad

# ----------------------------
# CONFIG
# ----------------------------

RESET_SPEED = 0.02

SIM_USE_GUI = True
SIM_STEP_DELAY = 0.01

TRAIL_LINE_ENABLED = True
TRAIL_LINE_WIDTH = 2.0
TRAIL_Z_TARGET = 0.07
TRAIL_Z_TOL = 0.03

# ----------------------------
# JOINT CONVERSION
# ----------------------------

def rad2nicodeg(joint, rad):

    if joint == 'r_wrist_z':
        return rad * 180.0 / 3.14159 * 2

    elif joint == 'r_wrist_x':
        return rad * 180.0 / 3.14159 * 4

    else:
        return rad * 180.0 / 3.14159


def nicodeg2rad(nicojoints, nicodegrees):

    if isinstance(nicojoints, str):
        nicojoints = [nicojoints]

    if isinstance(nicodegrees, (int, float)):
        nicodegrees = [nicodegrees]

    rads = []

    for nicojoint, nicodegree in zip(nicojoints, nicodegrees):

        if nicojoint == 'r_wrist_z':
            rad = deg2rad(nicodegree / 2)

        elif nicojoint == 'r_wrist_x':
            rad = deg2rad(nicodegree / 4)

        else:
            rad = deg2rad(nicodegree)

        rads.append(rad)

    if len(rads) == 1:
        return rads[0]

    return rads

# ----------------------------
# SPEED / TIMING
# ----------------------------

def speed_control(initial, target, duration):

    return abs(
        (float(initial) - float(target))
        / float(1260 * duration)
    )


def estimate_duration(
    current_angles,
    target_angles,
    max_speed_deg_per_sec=145.0,
    safety_factor=1.4,
    min_duration=0.025,
    max_duration=1.5
):

    diffs = [
        abs(t - c)
        for t, c in zip(target_angles, current_angles)
    ]

    max_diff = max(diffs)

    duration = (
        max_diff / max_speed_deg_per_sec
    ) * safety_factor

    duration = max(
        min(duration, max_duration),
        min_duration
    )

    return duration

# ----------------------------
# JOINT INFO
# ----------------------------

def get_joints_limits(robot_id):

    joints_limits_l = []
    joints_limits_u = []

    joints_ranges = []
    joints_rest_poses = []

    joint_names = []
    link_names = []
    joint_indices = []

    end_effector_index = None

    num_joints = p.getNumJoints(robot_id)

    for jid in range(num_joints):

        joint_info = p.getJointInfo(robot_id, jid)

        q_index = joint_info[3]
        joint_name = joint_info[1].decode("utf-8")
        link_name = joint_info[12].decode("utf-8")

        if q_index > -1:

            joint_names.append(joint_name)
            link_names.append(link_name)
            joint_indices.append(jid)

            joints_limits_l.append(joint_info[8])
            joints_limits_u.append(joint_info[9])

            joints_ranges.append(
                joint_info[9] - joint_info[8]
            )

            joints_rest_poses.append(
                (joint_info[9] + joint_info[8]) / 2
            )

        if link_name == "endeffector":
            end_effector_index = jid

    return (
        [joints_limits_l, joints_limits_u],
        joints_ranges,
        joints_rest_poses,
        end_effector_index,
        joint_names,
        link_names,
        joint_indices
    )

# ----------------------------
# TARGET GENERATION
# ----------------------------

def target_experiment(calibration_matrix, index):

    xs = [p[0] for p in calibration_matrix]
    ys = [p[1] for p in calibration_matrix]

    xmin = min(xs)

    scale = 1.5
    new_min = xmin + 0.01

    y_mean = sum(ys) / len(ys)

    rescaled = [
        [
            (x - xmin) * scale + new_min,
            -(y - y_mean) * scale,
            0.07 if z == 0.1 else 0.11
        ]
        for x, y, z in calibration_matrix
    ]

    first_x, first_y, _ = [0.3, -0.05, 0]

    augmented = (
        [[first_x, first_y, 0.2]]
        + rescaled
        + [[first_x, first_y, 0.2]]
    )

    if index >= len(augmented):
        index = 1

    return augmented[index]

# ----------------------------
# YAW MAPPING
# ----------------------------

def yaw_from_xy(
    x,
    y,
    xmin=0.3,
    xmax=0.475,
    ymin=-0.25,
    ymax=0.25
):

    x_clamped = max(min(x, xmax), xmin)
    y_clamped = max(min(y, ymax), ymin)

    tx = (x_clamped - xmin) / (xmax - xmin)
    ty = (y_clamped - ymin) / (ymax - ymin)

    yaw_deg = 45.0 * (2.0 - tx) * ty

    return deg2rad(yaw_deg)

# ----------------------------
# TRAIL DRAWING
# ----------------------------

trail_last_pos_line = [None]
trail_last_target_on_tablet = [False]


def maybe_add_trail_line(robot_id, ee_index, target_on_tablet):

    if not (target_on_tablet and trail_last_target_on_tablet[0]):
        trail_last_pos_line[0] = None
        trail_last_target_on_tablet[0] = target_on_tablet
        return

    ee_pos = p.getLinkState(robot_id, ee_index)[0]

    if abs(ee_pos[2] - TRAIL_Z_TARGET) > TRAIL_Z_TOL:
        trail_last_pos_line[0] = None
        trail_last_target_on_tablet[0] = target_on_tablet
        return

    if trail_last_pos_line[0] is None:
        trail_last_pos_line[0] = ee_pos
        trail_last_target_on_tablet[0] = target_on_tablet
        return

    p.addUserDebugLine(
        trail_last_pos_line[0],
        ee_pos,
        lineColorRGB=[0, 1, 0],
        lineWidth=TRAIL_LINE_WIDTH,
        lifeTime=0
    )

    trail_last_pos_line[0] = ee_pos
    trail_last_target_on_tablet[0] = target_on_tablet

# ----------------------------
# MAIN
# ----------------------------

def robot_sim(calibration_matrix):

    init_pos = {
        'l_shoulder_z': -24.0,
        'l_shoulder_y': 13.0,
        'l_arm_x': 0.0,
        'l_elbow_y': 104.0,
        'l_wrist_z': -4.0,
        'l_wrist_x': -55.0,
        'l_thumb_z': -62.0,
        'l_thumb_x': -180.0,
        'l_indexfinger_x': -170.0,
        'l_middlefingers_x': -180.0,
        'r_shoulder_z': -25,
        'r_shoulder_y': 84,
        'r_arm_x': 47,
        'r_elbow_y': 94,
        'r_wrist_z': -59,
        'r_wrist_x': 114,
        'r_thumb_z': -1,
        'r_thumb_x': 44,
        'r_indexfinger_x': -90,
        'r_middlefingers_x': 38.0,
        'head_z': 0.0,
        'head_y': 0.0
    }

    # ----------------------------
    # CONNECT
    # ----------------------------

    if SIM_USE_GUI:
        p.connect(p.GUI)

        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI,
            0
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=90,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0]
        )

    else:
        p.connect(p.DIRECT)

    # ----------------------------
    # LOAD ROBOT
    # ----------------------------

    robot_id = p.loadURDF(
        "./robot_config/urdf/nico_upper_rh6d_r.urdf",
        [0, 0, 0],
        useFixedBase=True
    )

    # hide sight link
    sight_link = None

    for j in range(p.getNumJoints(robot_id)):

        link_name = p.getJointInfo(robot_id, j)[12].decode("utf-8")

        if link_name == "sight":
            sight_link = j
            break

    if sight_link is not None:

        p.changeVisualShape(
            robot_id,
            sight_link,
            rgbaColor=[1, 1, 1, 0]
        )

    # ----------------------------
    # TABLE
    # ----------------------------

    p.createMultiBody(
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[.30, .45, 0.025],
            rgbaColor=[0.6, 0.6, 0.6, 1]
        ),

        baseCollisionShapeIndex=p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[.30, .45, 0.025]
        ),

        baseMass=0,
        basePosition=[0.26, 0, 0.029]
    )

    # ----------------------------
    # TABLET
    # ----------------------------

    p.createMultiBody(
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[.165, .267, 0.02],
            rgbaColor=[0, 0, 0, 1]
        ),

        baseCollisionShapeIndex=p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[.165, .267, 0.02]
        ),

        baseMass=0,
        basePosition=[0.395, 0, 0.036]
    )

    (
        joints_limits,
        joints_ranges,
        joints_rest_poses,
        end_effector_index,
        joint_names,
        link_names,
        joint_indices

    ) = get_joints_limits(robot_id)

    # ----------------------------
    # INITIAL POSE
    # ----------------------------

    current_angles_deg = []

    for name in joint_names:

        if name in init_pos:
            current_angles_deg.append(init_pos[name])
        else:
            current_angles_deg.append(0.0)

    for j_idx, j_name in enumerate(joint_names):

        p.resetJointState(
            robot_id,
            joint_indices[j_idx],
            nicodeg2rad(
                j_name,
                current_angles_deg[j_idx]
            )
        )

    # ----------------------------
    # IK SETTINGS
    # ----------------------------

    max_iterations = 2000
    residual_threshold = 1e-6

    # ----------------------------
    # MAIN LOOP
    # ----------------------------

    for i in range(len(calibration_matrix)):

        target_position = calibration_matrix[i]

        target_on_tablet = (
            target_position[2] == TRAIL_Z_TARGET
        )

        x = target_position[0]
        y = target_position[1]

        yaw = yaw_from_xy(x, y)

        palm_down_quat = p.getQuaternionFromEuler(
            [3.28, 2.6, yaw]
        )

        ik_solution = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            target_position,
            targetOrientation=palm_down_quat,
            maxNumIterations=max_iterations,
            residualThreshold=residual_threshold
        )

        target_angles_deg = [
            rad2nicodeg(j_name, angle)
            for j_name, angle in zip(
                joint_names,
                ik_solution
            )
        ]

        movement_duration = estimate_duration(
            current_angles_deg,
            target_angles_deg
        )

        # ----------------------------
        # APPLY JOINT MOTION
        # ----------------------------

        for j_idx, j_name in enumerate(joint_names):

            target_angle = target_angles_deg[j_idx]
            current_angle = current_angles_deg[j_idx]

            speed = speed_control(
                current_angle,
                target_angle,
                movement_duration
            )

            current_angles_deg[j_idx] = target_angle

            p.setJointMotorControl2(
                robot_id,
                joint_indices[j_idx],
                p.POSITION_CONTROL,
                targetPosition=nicodeg2rad(
                    j_name,
                    target_angle
                ),
                force=500,
                maxVelocity=max(speed * 25, 0.1)
            )

        # ----------------------------
        # SIMULATION STEPS
        # ----------------------------

        pos_tol = 0.01
        max_steps = 500

        for step in range(max_steps):

            p.stepSimulation()

            if TRAIL_LINE_ENABLED:
                maybe_add_trail_line(
                    robot_id,
                    end_effector_index,
                    target_on_tablet
                )

            # only check every few frames
            if step % 5 == 0:

                ls = p.getLinkState(
                    robot_id,
                    end_effector_index,
                    computeForwardKinematics=True
                )

                ee_pos = ls[0]

                pos_err = (
                    (ee_pos[0] - target_position[0]) ** 2 +
                    (ee_pos[1] - target_position[1]) ** 2 +
                    (ee_pos[2] - target_position[2]) ** 2
                ) ** 0.5

                if pos_err < pos_tol:
                    break

            # faster-than-realtime GUI
            if SIM_USE_GUI:
                time.sleep(0.001)

        print(
            f"Reached target {i} "
            f"in {movement_duration:.2f}s"
        )

    input("Press Enter to exit...")

    p.disconnect()
