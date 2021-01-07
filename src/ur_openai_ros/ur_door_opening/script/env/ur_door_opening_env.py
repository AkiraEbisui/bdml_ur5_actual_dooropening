#!/usr/bin/env python
'''
    By Akira Ebisui <shrimp.prawn.lobster713@gmail.com>
'''
# Python
import copy
import numpy as np
import math
import sys
import time
from matplotlib import pyplot as plt

# ROS 
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from joint_publisher import JointPub
from joint_traj_publisher import JointTrajPub

# Gazebo
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, GetModelState
from gazebo_msgs.srv import GetWorldProperties
from gazebo_msgs.msg import LinkStates 

# ROS msg
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, WrenchStamped
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Empty
from tactilesensors4.msg import StaticData, Dynamic
from tae_psoc.msg import Sensor_Fast
from tae_psoc.msg import Sensor_Indiv

# Gym
import gym
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register

# For inherit RobotGazeboEnv
from env import robot_gazebo_env_goal

# UR5 Utils
from env.ur_setups import setups
from env import ur_utils

from ur5_interface_for_door import UR5Interface
from robotiq_interface_for_door import RobotiqInterface
from algorithm.ppo_gae import PPOGAEAgent

seed = rospy.get_param("/ML/seed")
obs_dim = rospy.get_param("/ML/obs_dim")
n_act = rospy.get_param("/ML/n_act")
epochs = rospy.get_param("/ML/epochs")
hdim = rospy.get_param("/ML/hdim")
policy_lr = rospy.get_param("/ML/policy_lr")
value_lr = rospy.get_param("/ML/value_lr")
max_std = rospy.get_param("/ML/max_std")
clip_range = rospy.get_param("/ML/clip_range")
n_step = rospy.get_param("/ML/n_step")
sub_step1 = rospy.get_param("/ML/sub_step1")
sub_step2 = rospy.get_param("/ML/sub_step2")
act_add = rospy.get_param("/ML/act_add")

change_sub = rospy.get_param("/act_params/change_sub")
sub_a0 = rospy.get_param("/act_params/sub_a0")
sub_a1 = rospy.get_param("/act_params/sub_a1")
sub_a2 = rospy.get_param("/act_params/sub_a2")
sub_a3 = rospy.get_param("/act_params/sub_a3")
sub_a4 = rospy.get_param("/act_params/sub_a4")
sub_a5 = rospy.get_param("/act_params/sub_a5")

sub2_a0 = rospy.get_param("/act_params/sub2_a0")
sub2_a1 = rospy.get_param("/act_params/sub2_a1")
sub2_a2 = rospy.get_param("/act_params/sub2_a2")
sub2_a3 = rospy.get_param("/act_params/sub2_a3")
sub2_a4 = rospy.get_param("/act_params/sub2_a4")
sub2_a5 = rospy.get_param("/act_params/sub2_a5")

knob_c = rospy.get_param("/reward_params/knob_c")
knob_bonus_c = rospy.get_param("/reward_params/knob_bonus_c")
panel_c = rospy.get_param("/reward_params/panel_c")
panel_b_c = rospy.get_param("/reward_params/panel_b_c")
force_c = rospy.get_param("/reward_params/force_c")
force_c2 = rospy.get_param("/reward_params/force_c2")
taxel_c = rospy.get_param("/reward_params/taxel_c")
act_0_n = rospy.get_param("/reward_params/act_0_n")
act_1_n = rospy.get_param("/reward_params/act_1_n")
act_2_n = rospy.get_param("/reward_params/act_2_n")
act_3_n = rospy.get_param("/reward_params/act_3_n")
act_4_n = rospy.get_param("/reward_params/act_4_n")
act_5_n = rospy.get_param("/reward_params/act_5_n")
act_correct_c = rospy.get_param("/reward_params/act_correct_c")
catesian_xyz_c = rospy.get_param("/reward_params/catesian_xyz_c")
catesian_rpy_c = rospy.get_param("/reward_params/catesian_rpy_c")
cartesian_c = rospy.get_param("/reward_params/cartesian_c")

rospy.loginfo("register...")
#register the training environment in the gym as an available one
reg = gym.envs.register(
    id='URSimDoorOpening-v0',
    entry_point='env.ur_door_opening_env:URSimDoorOpening', # Its directory associated with importing in other sources like from 'ur_reaching.env.ur_sim_env import *' 
    #timestep_limit=100000,
    )
agent = PPOGAEAgent(obs_dim, n_act, epochs, hdim, policy_lr, value_lr, max_std, clip_range, seed)

class URSimDoorOpening(robot_gazebo_env_goal.RobotGazeboEnv):
    def __init__(self):
        rospy.logdebug("Starting URSimDoorOpening Class object...")

        # Subscribe joint state and target pose
        # rospy.Subscriber("/topic", msg, callback)
        rospy.Subscriber("/robotiq_ft_wrench", WrenchStamped, self.wrench_stamped_callback) # FT300
#        rospy.Subscriber("/wrench", WrenchStamped, self.wrench_stamped_callback) # default UR5 sensor
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)
        rospy.Subscriber("/TactileSensor4/StaticData", StaticData, self.tactile_static_callback)
        rospy.Subscriber("/TactileSensor4/Dynamic", Dynamic, self.tactile_dynamic_callback)
        rospy.Subscriber("/Sensor_Fast", Sensor_Fast, self.Sensor_Fast_callback)
        rospy.Subscriber("/Sensor_Indiv", Sensor_Indiv, self.Sensor_Indiv_callback)
        rospy.Subscriber("/imu/data", Imu, self.rt_imu_callback)
        rospy.Subscriber("/imu/data_3dmg", Imu, self.microstrain_imu_callback)

        # Gets training parameters from param server
        self.observations = rospy.get_param("/observations")
        self.init_grp_pose1 = rospy.get_param("/init_grp_pose1")
        self.init_grp_pose2 = rospy.get_param("/init_grp_pose2")
        for obs_name in self.observations:
            if obs_name == "image_cnn":
                self.image_cnn_on = 1
            elif obs_name == "static_taxel":
                self.static_taxel_on = 1
            elif obs_name == "nibs_cnn":
                self.nibs_cnn_on = 1
            elif obs_name == "nibs_indiv":
                self.nibs_indiv_on = 1
            elif obs_name == "nibs_fast":
                self.nibs_fast_on = 1

        # Joint limitation
        shp_max = rospy.get_param("/joint_limits_array/shp_max")
        shp_min = rospy.get_param("/joint_limits_array/shp_min")
        shl_max = rospy.get_param("/joint_limits_array/shl_max")
        shl_min = rospy.get_param("/joint_limits_array/shl_min")
        elb_max = rospy.get_param("/joint_limits_array/elb_max")
        elb_min = rospy.get_param("/joint_limits_array/elb_min")
        wr1_max = rospy.get_param("/joint_limits_array/wr1_max")
        wr1_min = rospy.get_param("/joint_limits_array/wr1_min")
        wr2_max = rospy.get_param("/joint_limits_array/wr2_max")
        wr2_min = rospy.get_param("/joint_limits_array/wr2_min")
        wr3_max = rospy.get_param("/joint_limits_array/wr3_max")
        wr3_min = rospy.get_param("/joint_limits_array/wr3_min")
        self.joint_limits = {"shp_max": shp_max,
                             "shp_min": shp_min,
                             "shl_max": shl_max,
                             "shl_min": shl_min,
                             "elb_max": elb_max,
                             "elb_min": elb_min,
                             "wr1_max": wr1_max,
                             "wr1_min": wr1_min,
                             "wr2_max": wr2_max,
                             "wr2_min": wr2_min,
                             "wr3_max": wr3_max,
                             "wr3_min": wr3_min
                             }

        # cartesian_limits
        self.x_max = rospy.get_param("/cartesian_limits/x_max")
        self.x_min = rospy.get_param("/cartesian_limits/x_min")
        self.y_max = rospy.get_param("/cartesian_limits/y_max")
        self.y_min = rospy.get_param("/cartesian_limits/y_min")
        self.z_max = rospy.get_param("/cartesian_limits/z_max")
        self.z_min = rospy.get_param("/cartesian_limits/z_min")
        self.rpy_x_max = rospy.get_param("/cartesian_limits/rpy_x_max")
        self.rpy_x_min = rospy.get_param("/cartesian_limits/rpy_x_min")
        self.rpy_y_max = rospy.get_param("/cartesian_limits/rpy_y_max")
        self.rpy_y_min = rospy.get_param("/cartesian_limits/rpy_y_min")
        self.rpy_z_max = rospy.get_param("/cartesian_limits/rpy_z_max")
        self.rpy_z_min = rospy.get_param("/cartesian_limits/rpy_z_min")

        shp_init_value1 = rospy.get_param("/init_joint_pose1/shp")
        shl_init_value1 = rospy.get_param("/init_joint_pose1/shl")
        elb_init_value1 = rospy.get_param("/init_joint_pose1/elb")
        wr1_init_value1 = rospy.get_param("/init_joint_pose1/wr1")
        wr2_init_value1 = rospy.get_param("/init_joint_pose1/wr2")
        wr3_init_value1 = rospy.get_param("/init_joint_pose1/wr3")
        self.init_joint_pose1 = [shp_init_value1, shl_init_value1, elb_init_value1, wr1_init_value1, wr2_init_value1, wr3_init_value1]
        self.init_pos1 = self.init_joints_pose(self.init_joint_pose1)
        self.arr_init_pos1 = np.array(self.init_pos1, dtype='float32')

        self.shp_init_value2 = rospy.get_param("/init_joint_pose2/shp")
        self.shl_init_value2 = rospy.get_param("/init_joint_pose2/shl")
        self.elb_init_value2 = rospy.get_param("/init_joint_pose2/elb")
        self.wr1_init_value2 = rospy.get_param("/init_joint_pose2/wr1")
        self.wr2_init_value2 = rospy.get_param("/init_joint_pose2/wr2")
        self.wr3_init_value2 = rospy.get_param("/init_joint_pose2/wr3")
        self.init_joint_pose2 = [self.shp_init_value2, self.shl_init_value2, self.elb_init_value2, self.wr1_init_value2, self.wr2_init_value2, self.wr3_init_value2]
        self.init_pos2 = self.init_joints_pose(self.init_joint_pose2)
        self.arr_init_pos2 = np.array(self.init_pos2, dtype='float32')

        self.shp_after_pull = rospy.get_param("/after_pull_pose/shp")
        self.shl_after_pull = rospy.get_param("/after_pull_pose/shl")
        self.elb_after_pull = rospy.get_param("/after_pull_pose/elb")
        self.wr1_after_pull = rospy.get_param("/after_pull_pose/wr1")
        self.wr2_after_pull = rospy.get_param("/after_pull_pose/wr2")
        self.wr3_after_pull = rospy.get_param("/after_pull_pose/wr3")

        shp_far_pose = rospy.get_param("/far_pose/shp")
        shl_far_pose = rospy.get_param("/far_pose/shl")
        elb_far_pose = rospy.get_param("/far_pose/elb")
        wr1_far_pose = rospy.get_param("/far_pose/wr1")
        wr2_far_pose = rospy.get_param("/far_pose/wr2")
        wr3_far_pose = rospy.get_param("/far_pose/wr3")
        self.far_pose = [shp_far_pose, shl_far_pose, elb_far_pose, wr1_far_pose, wr2_far_pose, wr3_far_pose]
        far_pose = self.init_joints_pose(self.far_pose)
        self.arr_far_pose = np.array(far_pose, dtype='float32')

        shp_before_close_pose = rospy.get_param("/before_close_pose/shp")
        shl_before_close_pose = rospy.get_param("/before_close_pose/shl")
        elb_before_close_pose = rospy.get_param("/before_close_pose/elb")
        wr1_before_close_pose = rospy.get_param("/before_close_pose/wr1")
        wr2_before_close_pose = rospy.get_param("/before_close_pose/wr2")
        wr3_before_close_pose = rospy.get_param("/before_close_pose/wr3")
        self.before_close_pose = [shp_before_close_pose, shl_before_close_pose, elb_before_close_pose, wr1_before_close_pose, wr2_before_close_pose, wr3_before_close_pose]
        before_close_pose = self.init_joints_pose(self.before_close_pose)
        self.arr_before_close_pose = np.array(before_close_pose, dtype='float32')

        shp_close_door_pose = rospy.get_param("/close_door_pose/shp")
        shl_close_door_pose = rospy.get_param("/close_door_pose/shl")
        elb_close_door_pose = rospy.get_param("/close_door_pose/elb")
        wr1_close_door_pose = rospy.get_param("/close_door_pose/wr1")
        wr2_close_door_pose = rospy.get_param("/close_door_pose/wr2")
        wr3_close_door_pose = rospy.get_param("/close_door_pose/wr3")
        self.close_door_pose = [shp_close_door_pose, shl_close_door_pose, elb_close_door_pose, wr1_close_door_pose, wr2_close_door_pose, wr3_close_door_pose]
        close_door_pose = self.init_joints_pose(self.close_door_pose)
        self.arr_close_door_pose = np.array(close_door_pose, dtype='float32')

        # cartesian
        init_pose1_x = rospy.get_param("/init_pose1/x")
        init_pose1_y = rospy.get_param("/init_pose1/y")
        init_pose1_z = rospy.get_param("/init_pose1/z")
        init_pose1_rpy_r = rospy.get_param("/init_pose1/rpy_r")
        init_pose1_rpy_p = rospy.get_param("/init_pose1/rpy_p")
        init_pose1_rpy_y = rospy.get_param("/init_pose1/rpy_y")
        self.init_pose1 = [init_pose1_x, init_pose1_y, init_pose1_z, init_pose1_rpy_r, init_pose1_rpy_p, init_pose1_rpy_y]
        self.arr_init_pose1 = np.array(self.init_pose1, dtype='float32')

        init_pose2_x = rospy.get_param("/init_pose2/x")
        init_pose2_y = rospy.get_param("/init_pose2/y")
        init_pose2_z = rospy.get_param("/init_pose2/z")
        init_pose2_rpy_r = rospy.get_param("/init_pose2/rpy_r")
        init_pose2_rpy_p = rospy.get_param("/init_pose2/rpy_p")
        init_pose2_rpy_y = rospy.get_param("/init_pose2/rpy_y")
        self.init_pose2 = [init_pose2_x, init_pose2_y, init_pose2_z, init_pose2_rpy_r, init_pose2_rpy_p, init_pose2_rpy_y]
        self.arr_init_pose2 = np.array(self.init_pose2, dtype='float32')

        far_xyz_x = rospy.get_param("/far_xyz/x")
        far_xyz_y = rospy.get_param("/far_xyz/y")
        far_xyz_z = rospy.get_param("/far_xyz/z")
        far_xyz_rpy_r = rospy.get_param("/far_xyz/rpy_r")
        far_xyz_rpy_p = rospy.get_param("/far_xyz/rpy_p")
        far_xyz_rpy_y = rospy.get_param("/far_xyz/rpy_y")
        self.far_xyz = [far_xyz_x, far_xyz_y, far_xyz_z, far_xyz_rpy_r, far_xyz_rpy_p, far_xyz_rpy_y]

        before_close_xyz_x = rospy.get_param("/before_close_xyz/x")
        before_close_xyz_y = rospy.get_param("/before_close_xyz/y")
        before_close_xyz_z = rospy.get_param("/before_close_xyz/z")
        before_close_xyz_rpy_r = rospy.get_param("/before_close_xyz/rpy_r")
        before_close_xyz_rpy_p = rospy.get_param("/before_close_xyz/rpy_p")
        before_close_xyz_rpy_y = rospy.get_param("/before_close_xyz/rpy_y")
        self.before_close_xyz = [before_close_xyz_x, before_close_xyz_y, before_close_xyz_z, before_close_xyz_rpy_r, before_close_xyz_rpy_p, before_close_xyz_rpy_y]

        close_door_xyz_x = rospy.get_param("/close_door_xyz/x")
        close_door_xyz_y = rospy.get_param("/close_door_xyz/y")
        close_door_xyz_z = rospy.get_param("/close_door_xyz/z")
        close_door_xyz_rpy_r = rospy.get_param("/close_door_xyz/rpy_r")
        close_door_xyz_rpy_p = rospy.get_param("/close_door_xyz/rpy_p")
        close_door_xyz_rpy_y = rospy.get_param("/close_door_xyz/rpy_y")
        self.close_door_xyz = [close_door_xyz_x, close_door_xyz_y, close_door_xyz_z, close_door_xyz_rpy_r, close_door_xyz_rpy_p, close_door_xyz_rpy_y]

        # Controller type for ros_control
        self._ctrl_type =  rospy.get_param("/control_type")
        self.pre_ctrl_type =  self._ctrl_type

        # Use MoveIt or not
        self.moveit = rospy.get_param("/moveit")

	# Get the force and troque limit
        self.force_limit1 = rospy.get_param("/force_limit1")
        self.torque_limit1 = rospy.get_param("/torque_limit1")
        self.force_limit2 = rospy.get_param("/force_limit2")
        self.torque_limit2 = rospy.get_param("/torque_limit2")
        self.min_static_limit = rospy.get_param("/min_static_limit")
        self.max_static_limit = rospy.get_param("/max_static_limit")

        # Get observation parameters
        self.joint_n = rospy.get_param("/obs_params/joint_n")
        self.eef_n = rospy.get_param("/obs_params/eef_n")
        self.eef_rpy_n = rospy.get_param("/obs_params/eef_rpy_n")
        self.force_n = rospy.get_param("/obs_params/force_n")
        self.torque_n = rospy.get_param("/obs_params/torque_n")
        self.taxel_n = rospy.get_param("/obs_params/taxel_n")
        self.nibs_indiv_n = rospy.get_param("/obs_params/nibs_indiv_n")
        self.nibs_fast_n = rospy.get_param("/obs_params/nibs_fast_n")

        # We init the observations
        self.quat = Quaternion()
        self.door_rpy = Vector3()
        self.door_rotation = Vector3()
        self.door_rpy_ini = Vector3()
        self.knob_rpy = []
        self.knob_rotation = []
        self.knob_rpy_ini = []
        self.link_state = LinkStates()
        self.wrench_stamped = WrenchStamped()

        self.joints_state = JointState()
        self.tactile_static = StaticData()
        self.tactile_static_ini = []
        self.tactile_dynamic = Dynamic()
        self.tactile_dynamic_ini = []

        self.nibssensor_fast = Sensor_Fast()
        self.nibssensor_fast_ini = []
        self.nibssensor_indiv = Sensor_Indiv()
        self.nibssensor_indiv_ini = []

        self.rt_imu = Imu()
        self.microstrain_imu = Imu()
#        self.end_effector = Point() 

        # Arm/Control parameters
        self._ik_params = setups['UR5_6dof']['ik_params']
        
        # ROS msg type
        self._joint_pubisher = JointPub()
        self._joint_traj_pubisher = JointTrajPub()

        # Gym interface and action
        self.action_space = spaces.Discrete(n_act)
        self.observation_space = obs_dim #np.arange(self.get_observations().shape[0])
        self.reward_range = (-np.inf, np.inf)
        self._seed()

        # Gripper interface
        self.gripper = RobotiqInterface()

        # Joint trajectory publisher
        self.jointtrajpub = JointTrajPub()

        self.force = self.wrench_stamped.wrench.force
        self.torque = self.wrench_stamped.wrench.torque
        self.static_taxel = self.tactile_static.taxels

    def check_stop_flg(self):
        if self.stop_flag is False:
            return False
        else:
            return True

    def _start_trainnig(self, req):
        rospy.logdebug("_start_trainnig!!!!")
        self.stop_flag = False
        return SetBoolResponse(True, "_start_trainnig")

    def _stop_trainnig(self, req):
        rospy.logdebug("_stop_trainnig!!!!")
        self.stop_flag = True
        return SetBoolResponse(True, "_stop_trainnig")

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                self._ctrl_conn.start_controllers(controllers_on="joint_state_controller")                
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")

    def check_cartesian_limits(self, sub_action):
        if self.moveit == 0:
            ee_xyz = self.get_xyz(sub_action)
            self.ee_xyz = ee_xyz
#            ur5 = UR5Interface()
#            ee_rpy = ur5.get_rpy()
#            self.ee_xyz = np.append(ee_xyz, ee_rpy)
        elif self.moveit == 1:
            self.ee_xyz = []
            self.ee_xyz = sub_action

#        print("cartesian_ee_xyz", self.ee_xyz)
        if self.x_min > self.ee_xyz[0] or self.ee_xyz[0] > self.x_max:
            print("over the x_cartesian limits", self.x_min, "<", self.ee_xyz[0], "<", self.x_max)
            return False
        elif self.y_min > self.ee_xyz[1] or self.ee_xyz[1] > self.y_max:
            print("over the y_cartesian limits", self.y_min, "<", self.ee_xyz[1], "<", self.y_max)
            return False
        elif self.z_min > self.ee_xyz[2] or self.ee_xyz[2] > self.z_max:
            print("over the z_cartesian limits", self.z_min, "<", self.ee_xyz[2], "<", self.z_max)
            return False
        elif sub_action[5] > self.wr3_init_value2 + 0.9 or sub_action[5] < self.wr3_init_value2:
            print("max_wrist3 over the limit", sub_action[5])
            return False
#        elif self.rpy_x_min > self.ee_xyz[3] or self.ee_xyz[3] > self.rpy_x_max:
#            print("over the rpy_x_cartesian limits", self.rpy_x_min, "<", self.ee_xyz[3], "<", self.rpy_x_max)
#            return False
#        elif self.rpy_y_min > self.ee_xyz[4] or self.ee_xyz[4] > self.rpy_y_max:
#            print("over the rpy_y_cartesian limits", self.rpy_y_min, "<", self.ee_xyz[4], "<", self.rpy_y_max)
#            return False
#        elif self.rpy_z_min > self.ee_xyz[5] or self.ee_xyz[5] > self.rpy_z_max:
#            print("over the rpy_z_cartesian limits", self.rpy_z_min, "<", self.ee_xyz[5], "<", self.rpy_z_max)
#            return False
        else:
            return True


    def get_xyz(self, q):
        """Get x,y,z coordinates 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz

    def get_current_xyz(self):
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[0]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[2]
        wr1_joint_ang = joint_states.position[3]
        wr2_joint_ang = joint_states.position[4]
        wr3_joint_ang = joint_states.position[5]
        
        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz

    def get_joint_value(self):
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[0]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[2]
        wr1_joint_ang = joint_states.position[3]
        wr2_joint_ang = joint_states.position[4]
        wr3_joint_ang = joint_states.position[5]
        
        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        return q
            
    def get_orientation(self, q):
        """Get Euler angles 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        orientation = mat[0:3, 0:3]
        roll = -orientation[1, 2]
        pitch = orientation[0, 2]
        yaw = -orientation[0, 1]
        
        return Vector3(roll, pitch, yaw)


    def cvt_quat_to_euler(self, quat):
        euler_rpy = Vector3()
        euler = euler_from_quaternion([self.quat.x, self.quat.y, self.quat.z, self.quat.w])
        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy

    def init_joints_pose(self, init_pos):
        """
        We initialise the Position variable that saves the desired position where we want our
        joints to be
        :param init_pos:
        :return:
        """
        self.current_joint_pose =[]
        self.current_joint_pose = copy.deepcopy(init_pos)
        return self.current_joint_pose

    def get_euclidean_dist(self, p_in, p_pout):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((p_in.x, p_in.y, p_in.z))
        b = numpy.array((p_pout.x, p_pout.y, p_pout.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def joints_state_callback(self,msg):
        self.joints_state = msg

    def wrench_stamped_callback(self,msg):
        self.wrench_stamped = msg

    def tactile_static_callback(self,msg):
        self.tactile_static = msg

    def tactile_dynamic_callback(self,msg):
        self.tactile_dynamic = msg

    def Sensor_Fast_callback(self,msg):
        self.nibssensor_fast = msg

    def Sensor_Indiv_callback(self,msg):
        self.nibssensor_indiv = msg

    def rt_imu_callback(self,msg):
        self.rt_imu = msg

    def microstrain_imu_callback(self,msg):
        self.microstrain_imu = msg

    def joint_trajectory(self,msg):
        self.jointtrajectory = msg

    def get_observations(self):
        """
        Returns the state of the robot needed for OpenAI QLearn Algorithm
        The state will be defined by an array
        :return: observation
        """
        joint_states = self.joints_state
        self.force = self.wrench_stamped.wrench.force
        self.torque = self.wrench_stamped.wrench.torque
#        dynamic_taxel= tactile_dynamic


#        print("[force]", self.force.x, self.force.y, self.force.z)
#        print("[torque]", self.torque.x, self.torque.y, self.torque.z)
        shp_joint_ang = joint_states.position[0]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[2]
        wr1_joint_ang = joint_states.position[3]
        wr2_joint_ang = joint_states.position[4]
        wr3_joint_ang = joint_states.position[5]

        shp_joint_vel = joint_states.velocity[0]
        shl_joint_vel = joint_states.velocity[1]
        elb_joint_vel = joint_states.velocity[2]
        wr1_joint_vel = joint_states.velocity[3]
        wr2_joint_vel = joint_states.velocity[4]
        wr3_joint_vel = joint_states.velocity[5]

        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
#        print("q(observation):", q)
#        self.end_effector = self.get_xyz(q)
        self.eef_x, self.eef_y, self.eef_z = self.get_xyz(q)
        self.eef_rpy = self.get_orientation(q)

        self.static_taxel = self.tactile_static.taxels
        self.sns_1_Indiv_ini = self.nibssensor_indiv_ini.sns_1_Indiv
        self.sns_2_Indiv_ini = self.nibssensor_indiv_ini.sns_2_Indiv
        self.sns_1_Fast_ini = self.nibssensor_fast_ini.sns_1_Fast
        self.sns_2_Fast_ini = self.nibssensor_fast_ini.sns_2_Fast
        self.sns_1_Indiv = self.nibssensor_indiv.sns_1_Indiv
        self.sns_2_Indiv = self.nibssensor_indiv.sns_2_Indiv
        self.sns_1_Fast = self.nibssensor_fast.sns_1_Fast
        self.sns_2_Fast = self.nibssensor_fast.sns_2_Fast

        observation = []
#        rospy.logdebug("List of Observations==>"+str(self.observations))
        for obs_name in self.observations:
            if obs_name == "shp_joint_ang":
                observation.append((shp_joint_ang - self.init_joint_pose2[0]) * self.joint_n)
            elif obs_name == "shl_joint_ang":
                observation.append((shl_joint_ang - self.init_joint_pose2[1]) * self.joint_n)
            elif obs_name == "elb_joint_ang":
                observation.append((elb_joint_ang - self.init_joint_pose2[2]) * self.joint_n)
            elif obs_name == "wr1_joint_ang":
                observation.append((wr1_joint_ang - self.init_joint_pose2[3]) * self.joint_n)
            elif obs_name == "wr2_joint_ang":
                observation.append((wr2_joint_ang - self.init_joint_pose2[4]) * self.joint_n)
            elif obs_name == "wr3_joint_ang":
                observation.append((wr3_joint_ang - self.init_joint_pose2[5]) * self.joint_n)
            elif obs_name == "shp_joint_vel":
                observation.append(shp_joint_vel)
            elif obs_name == "shl_joint_vel":
                observation.append(shl_joint_vel)
            elif obs_name == "elb_joint_vel":
                observation.append(elb_joint_vel)
            elif obs_name == "wr1_joint_vel":
                observation.append(wr1_joint_vel)
            elif obs_name == "wr2_joint_vel":
                observation.append(wr2_joint_vel)
            elif obs_name == "wr3_joint_vel":
                observation.append(wr3_joint_vel)
            elif obs_name == "eef_x":
                observation.append((self.eef_x - self.eef_x_ini) * self.eef_n)
            elif obs_name == "eef_y":
                observation.append((self.eef_y - self.eef_y_ini) * self.eef_n)
            elif obs_name == "eef_z":
                observation.append((self.eef_z - self.eef_z_ini) * self.eef_n)
            elif obs_name == "eef_rpy_x":
                observation.append((self.eef_rpy.x - self.eef_rpy_ini.x) * self.eef_rpy_n)
            elif obs_name == "eef_rpy_y":
                observation.append((self.eef_rpy.y - self.eef_rpy_ini.y) * self.eef_rpy_n)
            elif obs_name == "eef_rpy_z":
                observation.append((self.eef_rpy.z - self.eef_rpy_ini.z) * self.eef_rpy_n)
            elif obs_name == "force_x":
                observation.append((self.force.x - self.force_ini.x) / self.force_limit1 * self.force_n)
            elif obs_name == "force_y":
                observation.append((self.force.y - self.force_ini.y) / self.force_limit1 * self.force_n)
            elif obs_name == "force_z":
                observation.append((self.force.z - self.force_ini.z) / self.force_limit1 * self.force_n)
            elif obs_name == "torque_x":
                observation.append((self.torque.x - self.torque_ini.x) / self.torque_limit1 * self.torque_n)
            elif obs_name == "torque_y":
                observation.append((self.torque.y - self.torque_ini.y) / self.torque_limit1 * self.torque_n)
            elif obs_name == "torque_z":
                observation.append((self.torque.z - self.torque_ini.z) / self.torque_limit1 * self.torque_n)
            elif obs_name == "image_cnn":
                delta_image_r, delta_image_l = self.get_image()
                self.cnn_image_r = agent.update_cnn(delta_image_r)
                self.cnn_image_l = agent.update_cnn(delta_image_l)
                self.cnn_image_r_list = self.cnn_image_r.tolist()
                self.cnn_image_l_list = self.cnn_image_l.tolist()
                for x in range(0, 10):
                    observation.append(self.cnn_image_r_list[0][x])
                for x in range(0, 10):
                    observation.append(self.cnn_image_l_list[0][x])
            elif obs_name == "static_taxel":
                for x in range(0, 28):
                    observation.append((self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n)
                for x in range(0, 28):
                    observation.append((self.static_taxel[1].values[x] - self.static_taxel_ini[1].values[x]) * self.taxel_n)
#            elif obs_name == "dynamic_taxel":
#                    observation.append(dynamic_taxel[0].values) * self.taxel_n
#                    observation.append(dynamic_taxel[1].values) * self.taxel_n
            elif obs_name == "nibs_cnn":
                delta_nibs_image_r, delta_nibs_image_l = self.get_nibs_image()
                self.cnn_nibs_image_r = agent.update_nibs_cnn(delta_nibs_image_r)
                self.cnn_nibs_image_l = agent.update_nibs_cnn(delta_nibs_image_l)
                self.cnn_nibs_image_r_list = self.cnn_nibs_image_r.tolist()
                self.cnn_nibs_image_l_list = self.cnn_nibs_image_l.tolist()
                for x in range(0, 10):
                    observation.append(self.cnn_nibs_image_r_list[0][x])
                for x in range(0, 10):
                    observation.append(self.cnn_nibs_image_l_list[0][x])
            elif obs_name == "nibs_indiv":
                for x in range(0, 36):
                    observation.append((self.sns_1_Indiv[x] - self.sns_1_Indiv_ini[x]) * self.nibs_indiv_n)
                for x in range(0, 36):
                    observation.append((self.sns_2_Indiv[x] - self.sns_2_Indiv_ini[x]) * self.nibs_indiv_n)
            elif obs_name == "nibs_fast":
                for x in range(0, 4):
                    observation.append((self.sns_1_Fast[x] - self.sns_1_Fast_ini[x]) * self.nibs_fast_n)
                for x in range(0, 4):
                    observation.append((self.sns_2_Fast[x] - self.sns_2_Fast_ini[x]) * self.nibs_fast_n)
            elif obs_name == "nibs_robotiq_cnn":
                delta_image_r = self.get_single_image()
                self.cnn_image_r = agent.update_cnn(delta_image_r)
                self.cnn_image_r_list = self.cnn_image_r.tolist()
                for x in range(0, 10):
                    observation.append(self.cnn_image_r_list[0][x])
                delta_nibs_image_r = self.get_nibs_single_image()
                self.cnn_nibs_image_r = agent.update_nibs_cnn(delta_nibs_image_r)
                self.cnn_nibs_image_r_list = self.cnn_nibs_image_r.tolist()
                for x in range(0, 10):
                    observation.append(self.cnn_nibs_image_r_list[0][x])
            elif obs_name == "nibs_robotiq_fast":
                for x in range(0, 4):
                    observation.append((self.sns_1_Fast[x] - self.sns_1_Fast_ini[x]) * self.nibs_fast_n)
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))
#        print("observation", list(map(round, observation, [3]*len(observation))))

        return observation

    def get_nibs_image(self):
        delta_nibs_image_r = []
        delta_nibs_image_l = []
        for x in range(0, 36):
            delta_nibs_image_r.append((self.sns_1_Indiv[x] - self.sns_1_Indiv_ini[x]) * self.nibs_indiv_n)
        for x in range(0, 36):
            delta_nibs_image_l.append((self.sns_2_Indiv[x] - self.sns_2_Indiv_ini[x]) * self.nibs_indiv_n)
#        print(delta_nibs_image_r)
#        print(delta_nibs_image_l)
        return delta_nibs_image_r, delta_nibs_image_l

    def get_nibs_single_image(self):
        delta_nibs_image_r = []
        for x in range(0, 36):
            delta_nibs_image_r.append((self.sns_1_Indiv[x] - self.sns_1_Indiv_ini[x]) * self.nibs_indiv_n)
#        print(delta_nibs_image_r)
        return delta_nibs_image_r

    def get_image(self):
        delta_image_r = []
        delta_image_l = []
        self.static_taxel = self.tactile_static.taxels
        for x in range(0, 28):
            delta_image_r.append((self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n)
        for x in range(0, 28):
            delta_image_l.append((self.static_taxel[1].values[x] - self.static_taxel_ini[1].values[x]) * self.taxel_n)
        return delta_image_r, delta_image_l

    def get_single_image(self):
        delta_image_r = []
        self.static_taxel = self.tactile_static.taxels
        for x in range(0, 28):
            delta_image_r.append((self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n)
        return delta_image_r

    def clamp_to_joint_limits(self):
        """
        clamps self.current_joint_pose based on the joint limits
        self._joint_limits
        {
         "shp_max": shp_max,
         "shp_min": shp_min,
         ...
         }
        :return:
        """

        rospy.logdebug("Clamping current_joint_pose>>>" + str(self.current_joint_pose))
        shp_joint_value = self.current_joint_pose[0]
        shl_joint_value = self.current_joint_pose[1]
        elb_joint_value = self.current_joint_pose[2]
        wr1_joint_value = self.current_joint_pose[3]
        wr2_joint_value = self.current_joint_pose[4]
        wr3_joint_value = self.current_joint_pose[5]

        self.current_joint_pose[0] = max(min(shp_joint_value, self._joint_limits["shp_max"]), self._joint_limits["shp_min"])
        self.current_joint_pose[1] = max(min(shl_joint_value, self._joint_limits["shl_max"]), self._joint_limits["shl_min"])
        self.current_joint_pose[2] = max(min(elb_joint_value, self._joint_limits["elb_max"]), self._joint_limits["elb_min"])
        self.current_joint_pose[3] = max(min(wr1_joint_value, self._joint_limits["wr1_max"]), self._joint_limits["wr1_min"])
        self.current_joint_pose[4] = max(min(wr2_joint_value, self._joint_limits["wr2_max"]), self._joint_limits["wr2_min"])
        self.current_joint_pose[5] = max(min(wr3_joint_value, self._joint_limits["wr3_max"]), self._joint_limits["wr3_min"])

        rospy.logdebug("DONE Clamping current_joint_pose>>>" + str(self.current_joint_pose))

    def first_reset(self):
	# 1st: Go to initial position
        rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose1))
        self.knob_rpy_ini = copy.deepcopy(self.microstrain_imu.linear_acceleration.y / 9.8 * 1.57)
        self.door_rpy_ini = copy.deepcopy(self.rt_imu.linear_acceleration.z / 9.8 * 1.57)

        if self.moveit ==0:
            self.gripper.goto_gripper_pos(self.init_grp_pose1, False)
            time.sleep(1)
#            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_far_pose)
#            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_before_close_pose)
#            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_close_door_pose)
            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_init_pos1)
        elif self.moveit ==1:
#            self.jointtrajpub.MoveItCommand(self.far_xyz)
#            self.jointtrajpub.MoveItCommand(self.before_close_xyz)
#            self.jointtrajpub.MoveItCommand(self.close_door_xyz)
            self.jointtrajpub.MoveItJointTarget(self.init_pos1)

    # Resets the state of the environment and returns an initial observation.
    def reset(self):
	# 1st: Go to initial position
        rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose1))
        self.max_knob_rotation = 0
        self.max_door_rotation = 0
        self.max_wrist3 = 0
        self.min_wrist3 = 0
        self.max_wrist2 = 0
        self.min_wrist2 = 0
        self.max_wrist1 = 0
        self.min_wrist1 = 0
        self.max_elb = 0
        self.min_elb = 0
        self.max_shl = 0
        self.min_shl = 0
        self.max_shp = 0
        self.min_shp = 0
        self.max_force_x = 0
        self.min_force_x = 0
        self.max_force_y = 0
        self.min_force_y = 0
        self.max_force_z = 0
        self.min_force_z = 0
        self.max_torque_x = 0
        self.min_torque_x = 0
        self.max_torque_y = 0
        self.min_torque_y = 0
        self.max_torque_z = 0
        self.min_torque_z = 0
        self.max_taxel0 = 0
        self.min_taxel0 = 0
        self.max_taxel1 = 0
        self.min_taxel1 = 0
        self.delta_force_x = 0
        self.delta_force_y = 0
        self.delta_force_z = 0
        self.delta_torque_x = 0
        self.delta_torque_y = 0
        self.delta_torque_z = 0
        self.max_act_correct_n = 0
        self.min_act_correct_n = 100
        self.max_eef_x = 0
        self.min_eef_x = 0
        self.max_eef_y = 0
        self.min_eef_y = 0
        self.max_eef_z = 0
        self.min_eef_z = 0
        self.max_eef_rpy_x = 0
        self.min_eef_rpy_x = 0
        self.max_eef_rpy_y = 0
        self.min_eef_rpy_y = 0
        self.max_eef_rpy_z = 0
        self.min_eef_rpy_z = 0
        self.act_correct_n = 0
        self.delta_force_x = 0
        self.delta_force_y = 0
        self.delta_force_z = 0
        self.delta_torque_x = 0
        self.delta_torque_y = 0
        self.delta_torque_z = 0

        if self.moveit ==0:
            self.gripper.goto_gripper_pos(self.init_grp_pose1, False)
            time.sleep(1)
#            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_far_pose)
#            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_before_close_pose)
            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_close_door_pose)
#            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_init_pos1)
        elif self.moveit ==1:
            self.gripper.goto_gripper_pos(self.init_grp_pose1, False)
            time.sleep(1)
#            self.jointtrajpub.MoveItCommand(self.far_xyz)
#            self.jointtrajpub.MoveItCommand(self.before_close_xyz)
            self.jointtrajpub.MoveItCommand(self.close_door_xyz)
#            self.jointtrajpub.MoveItJointTarget(self.init_pos1)

        # 2nd: Check all subscribers work.
        rospy.logdebug("check_all_systems_ready...")
        self.check_all_systems_ready()

        # 3rd: Get the initial state.
        self.force = self.wrench_stamped.wrench.force
        self.torque = self.wrench_stamped.wrench.torque
        self.force_ini = copy.deepcopy(self.force)
        self.torque_ini = copy.deepcopy(self.torque)
        if self.moveit == 0:
            self.previous_action = copy.deepcopy(self.arr_init_pos2)
        elif self.moveit == 1:
            self.previous_action = copy.deepcopy(self.init_pose2)

        # 4th: Go to start position.
        if self.moveit ==0:
            self.jointtrajpub.FollowJointTrajectoryCommand_reset(self.arr_init_pos2)
            time.sleep(1)
            self.gripper.goto_gripper_pos(self.init_grp_pose2, False)
            time.sleep(1)
        elif self.moveit ==1:
            self.jointtrajpub.MoveItJointTarget(self.init_pos2)
#            print(self.get_xyz(self.init_pos2), self.get_orientation(self.init_pos2))
            self.gripper.goto_gripper_pos(self.init_grp_pose2, False)
            time.sleep(1)

        # 5th: Get the State Discrete Stringuified version of the observations
        self.static_taxel = self.tactile_static.taxels
        self.static_taxel_ini = copy.deepcopy(self.static_taxel)
        self.nibssensor_indiv_ini = copy.deepcopy(self.nibssensor_indiv)
        self.nibssensor_fast_ini = copy.deepcopy(self.nibssensor_fast)
#        print("self.nibssensor_indiv_ini", self.nibssensor_indiv_ini)
#        print("self.nibssensor_indiv_ini.sns_1_Indiv", self.nibssensor_indiv_ini.sns_1_Indiv)
#        print("self.nibssensor_fast_ini.sns_1_Fast", self.nibssensor_fast_ini.sns_1_Fast)
        self.eef_x_ini, self.eef_y_ini, self.eef_z_ini = self.get_xyz(self.init_joint_pose2)
        self.eef_rpy_ini = self.get_orientation(self.init_joint_pose2)

#        print("ee_xyz_ini", self.eef_x_ini, self.eef_y_ini, self.eef_z_ini) # ('ee_xyz_ini', -0.08859761113537656, 0.3680231810564474, 0.2769473319312816)
#        print("ee_rpy_ini", self.eef_rpy_ini) # x: -0.022194749153057504 y: 0.9997526460232887 z: -0.0011061239414521327

        rospy.logdebug("get_observations...")
       	observation = self.get_observations()
        return observation

    def _act(self, action):
        if self._ctrl_type == 'traj_pos':
            if self.moveit == 0:
                self.pre_ctrl_type = 'traj_pos'
                self._joint_traj_pubisher.FollowJointTrajectoryCommand(action)
            elif self.moveit == 1:
                self._joint_traj_pubisher.MoveItCommand(action)
        elif self._ctrl_type == 'pos':
            self.pre_ctrl_type = 'pos'
            self._joint_pubisher.move_joints(action)
        elif self._ctrl_type == 'traj_vel':
            self.pre_ctrl_type = 'traj_vel'
            self._joint_traj_pubisher.FollowJointTrajectoryCommand(action)
        elif self._ctrl_type == 'vel':
            self.pre_ctrl_type = 'vel'
            self._joint_pubisher.move_joints(action)
        else:
            self._joint_pubisher.move_joints(action)
        
    def training_ok(self):
        rate = rospy.Rate(1)
        while self.check_stop_flg() is True:                  
            rospy.logdebug("stop_flag is ON!!!!")
            self._gz_conn.unpauseSim()

            if self.check_stop_flg() is False:
                break 
            rate.sleep()
                
    def step(self, action, update):
        '''
        ('action: ', array([ 0.,  0. , -0., -0., -0. , 0. ], dtype=float32))        
        '''
        rospy.logdebug("UR step func")	# define the logger
        self.act_correct_n = 0
        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        # Act

        self.act_end = 0
        mod_action = np.array((0, 0, 0, 0, 0, 0), dtype='float32')
#        print("mod_action", mod_action, type(mod_action), mod_action.shape)
        current_joint_value = self.get_joint_value()
        arr_current_joint_value = np.array(current_joint_value)

        if arr_current_joint_value[5] < self.wr3_init_value2 + change_sub:
            self.sub_step = sub_step1
        else:
            self.sub_step = sub_step2

        for x in range(1, self.sub_step + 1):
            self.cartesian_flag = 0
            self.min_static_taxel0 = 0
            self.min_static_taxel1 = 0
            self.max_static_taxel0 = 0
            self.max_static_taxel1 = 0
            action = np.array(action)

            if self.moveit == 0:
                if arr_current_joint_value[5] < self.wr3_init_value2 + change_sub:
                    mod_action[0] = action[0] / sub_a0
                    mod_action[1] = action[1] / sub_a1
                    mod_action[2] = action[2] / sub_a2
                    mod_action[3] = action[3] / sub_a3
                    mod_action[4] = action[4] / sub_a4
                    mod_action[5] = action[5] / sub_a5
                    print("##### sub1", arr_current_joint_value[5])
                else:
                    mod_action[0] = action[0] / sub2_a0
                    mod_action[1] = action[1] / sub2_a1
                    mod_action[2] = action[2] / sub2_a2
                    mod_action[3] = action[3] / sub2_a3
                    mod_action[4] = action[4] / sub2_a4
                    mod_action[5] = action[5] / sub2_a5
                    print("##### sub2 #####", arr_current_joint_value[5])

                if act_add == 0:
                    goal_action = mod_action + self.arr_init_pos2 # goal
                    delta_action = goal_action - arr_current_joint_value
                    self.sub_action = delta_action / self.sub_step * x + arr_current_joint_value
#                    print("sub_x", x)
                if act_add == 1:
                    delta_action = mod_action
                    self.sub_action = delta_action / self.sub_step * x + arr_current_joint_value
#                    print("add_sub_x", x)
#                print("@self.sub_action", self.sub_action)
#                self.sub_action = self.sub_action + arr_current_joint_value

# after rotate(shp,shl,elb,wr1,wr2,wr3)
#                self.sub_action[0] = 1.491407573528791
#                self.sub_action[1] = -1.434487752926512
#                self.sub_action[2] = 2.413675198293162
#                self.sub_action[3] = 2.177423014918695
#                self.sub_action[4] = -1.4691158467941916
#                self.sub_action[5] = 2.1733145480767723

# after pull
#                if update > 4:
#                    self.sub_action[0] = 1.648087725653139
#                    self.sub_action[1] = -1.4969974700328346
#                    self.sub_action[2] = 2.498128485003836
#                    self.sub_action[3] = 2.1563878359790927
#                    self.sub_action[4] = -1.7477778260118484
#                    self.sub_action[5] = 2.1733145480767723
#                print("self.sub_action", self.sub_action)

            elif self.moveit == 1:
#                self.sub_action[0] = self.sub_action[0] / 42
                self.sub_action[1] = self.sub_action[1] / 42
#                self.sub_action[2] = self.sub_action[2] / 1000
                self.sub_action[3] = self.sub_action[3] * 2
#                self.sub_action[4] = self.sub_action[4] / 1000
#                self.sub_action[5] = self.sub_action[5] / 10
                self.sub_action = self.sub_action + self.arr_init_pose2
                self.sub_action = self.sub_action.tolist()

# after rotate(x,y,z,roll,pitch,yaw)
                self.sub_action[0] = -0.0885606971807
#                self.sub_action[1] = 0.367100554257
                self.sub_action[2] = 0.278060295058
#                self.sub_action[3] = 1.5746781585880325
                self.sub_action[4] = 0.01488937165698871
                self.sub_action[5] = 1.5931206693388063

# after pull
#                if update > 4:
#                    self.sub_action[0] = -0.119503224332
#                    self.sub_action[1] = 0.317118121264
#                    self.sub_action[2] = 0.276059107781
#                    self.sub_action[3] = 2.5706315470591077
#                    self.sub_action[4] = 0.015724591329912007
#                    self.sub_action[5] = 1.4710841122970895
                print("self.sub_action", self.sub_action)

            if self.check_cartesian_limits(self.sub_action) is True:
                self._act(self.sub_action)
                self.wrench_stamped
                self.force = self.wrench_stamped.wrench.force
                self.torque = self.wrench_stamped.wrench.torque
                self.delta_force_x = self.force.x - self.force_ini.x 
                self.delta_force_y = self.force.y - self.force_ini.y
                self.delta_force_z = self.force.z - self.force_ini.z
                self.delta_torque_x = self.torque.x - self.torque_ini.x
                self.delta_torque_y = self.torque.y - self.torque_ini.y
                self.delta_torque_z = self.torque.z - self.torque_ini.z
#                print("delta_force", self.delta_force_x, self.delta_force_y, self.delta_force_z)
#                print("delta_torque", self.delta_torque_x, self.delta_torque_y, self.delta_torque_z)
    
                if self.max_force_x < self.delta_force_x:
                    self.max_force_x = self.delta_force_x
                if self.min_force_x > self.delta_force_x:
                    self.min_force_x = self.delta_force_x
                if self.max_force_y < self.delta_force_y:
                    self.max_force_y = self.delta_force_y
                if self.min_force_y > self.delta_force_y:
                    self.min_force_y = self.delta_force_y
                if self.max_force_z < self.delta_force_z:
                    self.max_force_z = self.delta_force_z
                if self.min_force_z > self.delta_force_z:
                    self.min_force_z = self.delta_force_z
                if self.max_torque_x < self.delta_torque_x:
                    self.max_torque_x = self.delta_torque_x
                if self.min_torque_x > self.delta_torque_x:
                    self.min_torque_x = self.delta_torque_x
                if self.max_torque_y < self.delta_torque_y:
                    self.max_torque_y = self.delta_torque_y
                if self.min_torque_y > self.delta_torque_y:
                    self.min_torque_y = self.delta_torque_y
                if self.max_torque_z < self.delta_torque_z:
                    self.max_torque_z = self.delta_torque_z
                if self.min_torque_z > self.delta_torque_z:
                    self.min_torque_z = self.delta_torque_z

                self.static_taxel = self.tactile_static.taxels
                self.sns_1_Indiv = self.nibssensor_indiv.sns_1_Indiv
                self.sns_2_Indiv = self.nibssensor_indiv.sns_2_Indiv

                for obs_name in self.observations:
                    if obs_name == "image_cnn":
                        for y in range(0, 28):
                            if self.min_static_taxel0 > (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n:
                                self.min_static_taxel0 = (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n
                            if self.min_static_taxel1 > (self.static_taxel[1].values[y] - self.static_taxel_ini[1].values[y]) * self.taxel_n:
                                self.min_static_taxel1 = (self.static_taxel[1].values[y] - self.static_taxel_ini[1].values[y]) * self.taxel_n
                            if self.max_static_taxel0 < (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n:
                                self.max_static_taxel0 = (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n
                            if self.max_static_taxel1 < (self.static_taxel[1].values[y] - self.static_taxel_ini[1].values[y]) * self.taxel_n:
                                self.max_static_taxel1 = (self.static_taxel[1].values[y] - self.static_taxel_ini[1].values[y]) * self.taxel_n
                    elif obs_name == "nibs_cnn":
                        for y in range(0, 36):
                            if self.min_static_taxel0 > (self.sns_1_Indiv[y] - self.sns_1_Indiv_ini[y]) * self.nibs_indiv_n:
                                self.min_static_taxel0 = (self.sns_1_Indiv[y] - self.sns_1_Indiv_ini[y]) * self.nibs_indiv_n
                            if self.min_static_taxel1 > (self.sns_2_Indiv[y] - self.sns_2_Indiv_ini[y]) * self.nibs_indiv_n:
                                self.min_static_taxel1 = (self.sns_2_Indiv[y] - self.sns_2_Indiv_ini[y]) * self.nibs_indiv_n
                            if self.max_static_taxel0 < (self.sns_1_Indiv[y] - self.sns_1_Indiv_ini[y]) * self.nibs_indiv_n:
                                self.max_static_taxel0 = (self.sns_1_Indiv[y] - self.sns_1_Indiv_ini[y]) * self.nibs_indiv_n
                            if self.max_static_taxel1 < (self.sns_2_Indiv[y] - self.sns_2_Indiv_ini[y]) * self.nibs_indiv_n:
                                self.max_static_taxel1 = (self.sns_2_Indiv[y] - self.sns_2_Indiv_ini[y]) * self.nibs_indiv_n
                    elif obs_name == "nibs_robotiq_cnn":
                        for y in range(0, 28):
                            if self.min_static_taxel0 > (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n:
                                self.min_static_taxel0 = (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n
                            if self.max_static_taxel0 < (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n:
                                self.max_static_taxel0 = (self.static_taxel[0].values[y] - self.static_taxel_ini[0].values[y]) * self.taxel_n
                        for z in range(0, 36):
                            if self.min_static_taxel1 > (self.sns_1_Indiv[z] - self.sns_1_Indiv_ini[z]) * self.nibs_indiv_n:
                                self.min_static_taxel1 = (self.sns_1_Indiv[z] - self.sns_1_Indiv_ini[z]) * self.nibs_indiv_n
                            if self.max_static_taxel1 < (self.sns_1_Indiv[z] - self.sns_1_Indiv_ini[z]) * self.nibs_indiv_n:
                                self.max_static_taxel1 = (self.sns_1_Indiv[z] - self.sns_1_Indiv_ini[z]) * self.nibs_indiv_n

                if self.min_taxel0 > self.min_static_taxel0:
                    self.min_taxel0 = self.min_static_taxel0
                if self.min_taxel1 > self.min_static_taxel1:
                    self.min_taxel1 = self.min_static_taxel1
                if self.max_taxel0 < self.max_static_taxel0:
                    self.max_taxel0 = self.max_static_taxel0
                if self.max_taxel1 < self.max_static_taxel1:
                    self.max_taxel1 = self.max_static_taxel1

                if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
                    print(x, "force.x over the limit2", self.delta_force_x)
                    self.act_end = 1
                elif self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
                    print(x, "force.y over the limit2", self.delta_force_y)
                    self.act_end = 1
                elif self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
                    print(x, "force.z over the limit2", self.delta_force_z)
                    self.act_end = 1
                elif self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
                    print(x, "torque.x over the limit2", self.delta_torque_x)
                    self.act_end = 1
                elif self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
                    print(x, "torque.y over the limit2", self.delta_torque_y)
                    self.act_end = 1
                elif self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
                    print(x, "torque.z over the limit2", self.delta_torque_z)
                    self.act_end = 1
                elif self.min_static_taxel0 < self.min_static_limit or self.min_static_taxel1 < self.min_static_limit:
                    print(x, "slipped and break the for loop(min over)", self.min_static_taxel0, self.min_static_taxel1)
                    self.act_end = 1
                elif self.max_static_taxel0 > self.max_static_limit or self.max_static_taxel1 > self.max_static_limit:
                    print(x, "slipped and break the for loop(max over)", self.max_static_taxel0, self.max_static_taxel1)
                    self.act_end = 1
                else:
                    self.act_correct_n += 1
                    print(x, "act correctly", self.act_correct_n)
                    if x == self.sub_step:
                        self.previous_action = copy.deepcopy(self.sub_action)
                        print("copy previous_action")                   

                if self.force_limit1 < self.delta_force_x or self.delta_force_x < -self.force_limit1:
#                    self._act(self.previous_action)
                    print(x, "force.x over the limit1", self.delta_force_x)
                elif self.force_limit1 < self.delta_force_y or self.delta_force_y < -self.force_limit1:
#                    self._act(self.previous_action)
                    print(x, "force.y over the limit1", self.delta_force_y)
                elif self.force_limit1 < self.delta_force_z or self.delta_force_z < -self.force_limit1:
#                    self._act(self.previous_action)
                    print(x, "force.z over the limit1", self.delta_force_z)
                elif self.torque_limit1 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit1:
#                    self._act(self.previous_action)
                    print(x, "torque.x over the limit1", self.delta_torque_x)
                elif self.torque_limit1 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit1:
#                    self._act(self.previous_action)
                    print(x, "torque.y over the limit1", self.delta_torque_y)
                elif self.torque_limit1 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit1:
#                    self._act(self.previous_action)
                    print(x, "torque.z over the limit1", self.delta_torque_x)

            else:
                self.cartesian_flag = 1
                print(x, "over the cartesian limits")
                self.act_end = 1

#            observation = self.get_observations()
#            if observation[3] < -0.1 or observation[3] > 0.1:
#                print(x, "break the for loop(wr1_limit)", observation[3])
#                self.act_end = 1
#            if observation[2] < -0.1 or observation[2] > 0.1:
#                print(x, "break the for loop(elb_limit)", observation[2])
#                self.act_end = 1
#            if observation[1] < -0.1 or observation[1] > 0.1:
#                print(x, "break the for loop(shl_limit)", observation[1])
#                self.act_end = 1

            if self.act_end == 1:
                self._act(self.previous_action)
                print("act previous_action", self.previous_action)
                break
    
        # We now process the latest data saved in the class state to calculate
        # the state and the rewards. This way we guarantee that they work
        # with the same exact data.
        # Generate State based on observations
        observation = self.get_observations()

        if self.max_wrist3 < observation[5]:
            self.max_wrist3 = observation[5]
        if self.min_wrist3 > observation[5]:
            self.min_wrist3 = observation[5]
        if self.max_wrist2 < observation[4]:
            self.max_wrist2 = observation[4]
        if self.min_wrist2 > observation[4]:
            self.min_wrist2 = observation[4]
        if self.max_wrist1 < observation[3]:
            self.max_wrist1 = observation[3]
        if self.min_wrist1 > observation[3]:
            self.min_wrist1 = observation[3]
        if self.max_elb < observation[2]:
            self.max_elb = observation[2]
        if self.min_elb > observation[2]:
            self.min_elb = observation[2]
        if self.max_shl < observation[1]:
            self.max_shl = observation[1]
        if self.min_shl > observation[1]:
            self.min_shl = observation[1]
        if self.max_shp < observation[0]:
            self.max_shp = observation[0]
        if self.min_shp > observation[0]:
            self.min_shp = observation[0]

        # finally we get an evaluation based on what happened in the sim
        reward = self.compute_dist_rewards(action, update)
        done = self.check_done(update)

        return observation, reward, done, {}

    def compute_dist_rewards(self, action, update):
        self.knob_rotation_r = 0
        self.panel_rotation_r = 0
        self.force_limit_r = 0
        self.static_limit_r = 0
        self.action_limit_r = 0
        self.act_correct_r = 0
        self.catesian_xyz_r = 0
        self.catesian_rpy_r = 0
        self.cartesian_bonus_r = 0
        force_x_limit_r = 0
        force_y_limit_r = 0
        force_z_limit_r = 0
        torque_x_limit_r = 0
        torque_y_limit_r = 0
        torque_z_limit_r = 0
        min_static_limit_r = 0
        max_static_limit_r = 0
        action5_limit_r = 0
        action4_limit_r = 0
        action3_limit_r = 0
        action2_limit_r = 0
        action1_limit_r = 0
        action0_limit_r = 0
        catesian_x = 0
        catesian_y = 0
        catesian_z = 0
        catesian_rpy_x = 0
        catesian_rpy_y = 0
        catesian_rpy_z = 0
        compute_rewards = 0.0001

        self.door_rpy = self.rt_imu.linear_acceleration.z / 9.8 * 1.57
        self.door_rotation = self.door_rpy_ini - self.door_rpy
        
        self.knob_rpy = self.microstrain_imu.linear_acceleration.y / 9.8 * 1.57
        self.knob_rotation = self.knob_rpy_ini - self.knob_rpy

        if self.max_knob_rotation < self.knob_rotation:
            self.max_knob_rotation = self.knob_rotation
        if self.max_door_rotation < self.door_rotation:
            self.max_door_rotation = self.door_rotation

        if self.max_act_correct_n < self.act_correct_n:
            self.max_act_correct_n = self.act_correct_n
        if self.min_act_correct_n > self.act_correct_n:
            self.min_act_correct_n = self.act_correct_n

        if self.max_eef_x < self.eef_x - self.eef_x_ini:
            self.max_eef_x = self.eef_x - self.eef_x_ini
        if self.min_eef_x > self.eef_x - self.eef_x_ini:
            self.min_eef_x = self.eef_x - self.eef_x_ini
        if self.max_eef_y < self.eef_y - self.eef_y_ini:
            self.max_eef_y = self.eef_y - self.eef_y_ini
        if self.min_eef_y > self.eef_y - self.eef_y_ini:
            self.min_eef_y = self.eef_y - self.eef_y_ini
        if self.max_eef_z < self.eef_z - self.eef_z_ini:
            self.max_eef_z = self.eef_z - self.eef_z_ini
        if self.min_eef_z > self.eef_z - self.eef_z_ini:
            self.min_eef_z = self.eef_z - self.eef_z_ini

        if self.max_eef_rpy_x < self.eef_rpy.x - self.eef_rpy_ini.x:
            self.max_eef_rpy_x = self.eef_rpy.x - self.eef_rpy_ini.x
        if self.min_eef_rpy_x > self.eef_rpy.x - self.eef_rpy_ini.x:
            self.min_eef_rpy_x = self.eef_rpy.x - self.eef_rpy_ini.x
        if self.max_eef_rpy_y < self.eef_rpy.y - self.eef_rpy_ini.y:
            self.max_eef_rpy_y = self.eef_rpy.y - self.eef_rpy_ini.y
        if self.min_eef_rpy_y > self.eef_rpy.y - self.eef_rpy_ini.y:
            self.min_eef_rpy_y = self.eef_rpy.y - self.eef_rpy_ini.y
        if self.max_eef_rpy_z < self.eef_rpy.z - self.eef_rpy_ini.z:
            self.max_eef_rpy_z = self.eef_rpy.z - self.eef_rpy_ini.z
        if self.min_eef_rpy_z > self.eef_rpy.z - self.eef_rpy_ini.z:
            self.min_eef_rpy_z = self.eef_rpy.z - self.eef_rpy_ini.z

        #1 rotation of knob, bonus of knob rotation(+)
        #2 door panel open(+), 
        knob_rotation_th = 0.6  # 3/4 = 0.56, previously 1.1
        door_rotation_th = 0.9
        if self.knob_rotation < knob_rotation_th / 4:
            self.knob_rotation_r = self.knob_rotation * knob_c                     # 0.12 * 100 = 12 (0-12)
        elif knob_rotation_th / 4 <= self.knob_rotation < knob_rotation_th * 2 / 4:
            self.knob_rotation_r = self.knob_rotation * knob_c + knob_bonus_c      # 0.24 * 100 + 10 = 34 (22-34)
        elif knob_rotation_th * 2 / 4 <= self.knob_rotation < knob_rotation_th * 3 / 4:
            self.knob_rotation_r = self.knob_rotation * knob_c + knob_bonus_c * 2  # 0.36 * 100 + 10 * 2 = 56 (44-56)
        elif knob_rotation_th * 3 / 4 <= self.knob_rotation < knob_rotation_th:
            self.knob_rotation_r = self.knob_rotation * knob_c + knob_bonus_c * 3  # 0.53 * 100 + 10 * 5 = 103 (86-103)
        elif knob_rotation_th <= self.knob_rotation:
            self.knob_rotation_r = knob_rotation_th * knob_c + knob_bonus_c * 10 # 0.53 * 100 + 10 * 5 = 103 (103- )

        if self.door_rotation < 0:
            self.panel_rotation_r =  self.door_rotation            
        elif 0 <= self.door_rotation < door_rotation_th * 1 / 4:
            self.panel_rotation_r =  self.door_rotation * panel_c + panel_b_c      # 0.12 * 100 + 10 (0-22)
        elif door_rotation_th * 1 / 4 <= self.door_rotation < door_rotation_th * 2 / 4:
            self.panel_rotation_r =  self.door_rotation * panel_c + panel_b_c * 2  # 0.24 * 100 + 10 * 2 (32-44)
        elif door_rotation_th * 2 / 4 <= self.door_rotation < door_rotation_th * 3 / 4:
            self.panel_rotation_r =  self.door_rotation * panel_c + panel_b_c * 3  # 0.36 * 100 + 10 * 3 (54-66)
        elif door_rotation_th * 3 / 4 <= self.door_rotation < door_rotation_th:
            self.panel_rotation_r =  self.door_rotation * panel_c + panel_b_c * 5  # 0.48 * 100 + 10 * 5 (86-98)
        elif door_rotation_th <= self.door_rotation:
            self.panel_rotation_r =  door_rotation_th * panel_c + panel_b_c * 10 # 0.89 * 100 + 10 * 10 (189- )

        print("##1 knob_rotation_r", self.knob_rotation_r, self.knob_rotation)
        print("##2 panel_rotation_r", self.panel_rotation_r, self.door_rotation)

        #3 over force limit1(-)
        if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
            force_x_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2 )
            print("# force_x limit2_r", force_x_limit2_r)
        elif self.force_limit1 < self.delta_force_x or self.delta_force_x < -self.force_limit1:
            force_x_limit1_r = - (force_c * abs(abs(self.delta_force_x)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c)
            print("# force_x limit1_r", force_x_limit1_r)
        if self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
            force_y_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
            print("# force_y limit2_r", force_y_limit2_r)
        elif self.force_limit1 < self.delta_force_y or self.delta_force_y < -self.force_limit1:
            force_y_limit1_r = - (force_c * abs(abs(self.delta_force_y)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c)
            print("# force_y limit1_r", force_y_limit1_r)
        if self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
            force_z_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
            print("# force_z limit2_r", force_z_limit2_r)
        elif self.force_limit1 < self.delta_force_z or self.delta_force_z < -self.force_limit1:
            force_z_limit1_r = - (force_c * abs(abs(self.delta_force_z)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c)
            print("# force_z limit1_r", force_z_limit1_r)
        if self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
            torque_x_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
            print("# torque_x limit2_r", torque_x_limit2_r)
        elif self.torque_limit1 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit1:
            torque_x_limit1_r = - (force_c * abs(abs(self.delta_torque_x)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c)
            print("# torque_x limit1_r", torque_x_limit1_r)
        if self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
            torque_y_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
            print("# torque_y limit2_r", torque_y_limit2_r)
        elif self.torque_limit1 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit1:
            torque_y_limit1_r = - (force_c * abs(abs(self.delta_torque_y)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c)
            print("# torque_y limit1_r", torque_y_limit1_r)
        if self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
            torque_z_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
            print("# torque_z limit2_r", torque_z_limit2_r)
        elif self.torque_limit1 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit1:
            torque_z_limit1_r = - (force_c * abs(abs(self.delta_torque_z)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c)
            print("# torque_z limit1_r", torque_z_limit1_r)
        self.force_limit_r = force_x_limit_r + force_y_limit_r + force_z_limit_r + torque_x_limit_r + torque_y_limit_r + torque_z_limit_r
        print("##3 force_limit_r", self.force_limit_r)

        #4 release the knob(-)
        if self.min_static_taxel0 < self.min_static_limit or self.min_static_taxel1 < self.min_static_limit:
            min_static_limit_r = - (taxel_c * (n_step - update) / n_step + taxel_c)
            print("# min_static_limit_r", min_static_limit_r)
        elif self.max_static_taxel0 > self.max_static_limit or self.max_static_taxel1 > self.max_static_limit:
            max_static_limit_r = - (taxel_c * (n_step - update) / n_step + taxel_c)
            print("# max_static_limit_r", max_static_limit_r)
        self.static_limit_r = min_static_limit_r + max_static_limit_r
        print("##4 static_limit_r", self.static_limit_r)

        #5 joint(+)
        act_5_n_limit = self.wr3_init_value2        # 1.5733145480767723
        act_5_p_limit = self.wr3_after_pull + 0.2   # 2.5733145480767723
        act_4_n_limit = self.wr2_after_pull - 0.2   # -1.7477778260118484
        act_4_p_limit = self.wr2_init_value2        # -1.4691158467941916
        act_3_n_limit = self.wr1_after_pull - 0.2   # 2.1563878359790927
        act_3_p_limit = self.wr1_init_value2        # 2.177423014918695
        act_2_n_limit = self.elb_init_value2        # 2.413675198293162
        act_2_p_limit = self.elb_after_pull + 0.2   # 2.498128485003836
        act_1_n_limit = self.shl_after_pull - 0.2   # -1.4969974700328346
        act_1_p_limit = self.shl_init_value2        # -1.434487752926512
        act_0_n_limit = self.shp_init_value2        # 1.491407573528791
        act_0_p_limit = self.shp_after_pull + 0.2   # 1.648087725653139

        current_joint_value = self.get_joint_value()
        if current_joint_value[5] < self.wr3_init_value2 or self.wr3_init_value2 + 0.9 < current_joint_value[5]:
            action5_limit_r = - 100        
        if act_5_n_limit < current_joint_value[5] and current_joint_value[5] < act_5_p_limit:
            action5_limit_r = (current_joint_value[5] - act_5_n_limit) * act_5_n
            print("# action5 limit_r", action5_limit_r)
        if act_4_n_limit < current_joint_value[4] and current_joint_value[4] < act_4_p_limit:
            action4_limit_r = - (current_joint_value[4] - act_4_p_limit) * act_4_n
            print("# action4 limit_r", action4_limit_r)
        if act_3_n_limit < current_joint_value[3] and current_joint_value[3] < act_3_p_limit:
            action3_limit_r = - (current_joint_value[3] - act_3_p_limit) * act_3_n
            print("# action3 limit_r", action3_limit_r)
        if act_2_n_limit < current_joint_value[2] and current_joint_value[2] < act_2_p_limit:
            action2_limit_r = (current_joint_value[2] - act_2_n_limit) * act_2_n
            print("# action2 limit_r", action2_limit_r)
        if act_1_n_limit < current_joint_value[1] and current_joint_value[1] < act_1_p_limit:
            action1_limit_r = - (current_joint_value[1] - act_1_p_limit) * act_1_n
            print("# action1 limit_r", action1_limit_r)
        if act_0_n_limit < current_joint_value[0] and current_joint_value[0] < act_0_p_limit:
            action0_limit_r = (current_joint_value[0] - act_0_n_limit) * act_0_n
            print("# action0 limit_r", action0_limit_r)
        self.action_limit_r = action5_limit_r + action4_limit_r + action3_limit_r + action2_limit_r + action1_limit_r + action0_limit_r
#        print("##5 action_limit_r.", current_joint_value[5], current_joint_value[4], current_joint_value[3], current_joint_value[2], current_joint_value[1], current_joint_value[0])
        print("##5 action_limit_r.", self.action_limit_r)

        #6 act_correct(+)
        self.act_correct_r = self.act_correct_n / self.sub_step * act_correct_c
        print("##6 act_correct_r", self.act_correct_r)

        #7 cartesian(+)
        catesian_x = (1 - abs(self.eef_x_ini - self.eef_x) * 10) * catesian_xyz_c
        catesian_y = (1 - abs(self.eef_y_ini - self.eef_y) * 10) * catesian_xyz_c
        catesian_z = (1 - abs(self.eef_z_ini - self.eef_z) * 10) * catesian_xyz_c
        self.catesian_xyz_r = catesian_x + catesian_y + catesian_z
        print("##7 catesian_xyz_r", catesian_x, catesian_y, catesian_z)

        catesian_rpy_x = (1 - abs(self.eef_rpy_ini.x - self.eef_rpy.x) * 10) * catesian_rpy_c
        catesian_rpy_y = (1 - abs(self.eef_rpy_ini.y - self.eef_rpy.y) * 10) * catesian_rpy_c
        catesian_rpy_z = (self.eef_rpy_ini.z - self.eef_rpy.z) * 10 * catesian_rpy_c
        self.catesian_rpy_r = catesian_rpy_x + catesian_rpy_y + catesian_rpy_z
        print("##7 catesian_rpy_r", catesian_rpy_x, catesian_rpy_y, catesian_rpy_z)

        if self.cartesian_flag == 0:
            compute_rewards += cartesian_c
            print("##7 cartesian_bonus_r", cartesian_c)

        self.negative_r = self.force_limit_r + self.static_limit_r
        self.action_r = self.action_limit_r + self.act_correct_r + self.catesian_xyz_r + self.catesian_rpy_r + self.cartesian_bonus_r
        compute_rewards = self.knob_rotation_r + self.panel_rotation_r + self.negative_r + self.action_r
        print("### action_r", self.action_r)
        print("### total_compute_rewards", compute_rewards)

        return compute_rewards

    def check_done(self, update):
        if update > 1:
            observation = self.get_observations()
            if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
                print("########## force.x over the limit2 ##########", self.delta_force_x)
#                return True
            elif self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
                print("########## force.y over the limit2 ##########", self.delta_force_y)
#                return True
            elif self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
                print("########## force.z over the limit2 ##########", self.delta_force_z)
#                return True
            elif self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
                print("########## torque.x over the limit2 ##########", self.delta_torque_x)
#                return True
            elif self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
                print("########## torque.y over the limit2 ##########", self.delta_torque_y)
#                return True
            elif self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
                print("########## torque.z over the limit2 ##########", self.delta_torque_z)
#                return True
            elif self.min_static_taxel0 < self.min_static_limit or self.min_static_taxel1 < self.min_static_limit:
                print("########## static_taxles over the min limit ##########", update)
#                return True
            elif self.max_static_taxel0 > self.max_static_limit or self.max_static_taxel1 > self.max_static_limit:
                print("########## static_taxles over the max limit ##########", update)
#                return True
            elif self.sub_action[5] < self.wr3_init_value2 or self.sub_action[5] > self.wr3_init_value2 + 0.9:
                print("########## action_limit ##########", self.sub_action[5])
#                return True
            elif observation[5] < -0.05 or observation[5] > 0.9:
                print("########## wr3_limit ##########", observation[5])
#                return True
            elif observation[3] < -0.1 or observation[3] > 0.1:
                print("########## wr1_limit ##########", observation[3])
#                return True
            elif observation[2] < -0.1 or observation[2] > 0.1:
                print("########## elb_limit ##########", observation[2])
#                return True
            elif observation[1] < -0.1 or observation[1] > 0.1:
                print("########## shl_limit ##########", observation[1])
#                return True
            else :
            	return False
