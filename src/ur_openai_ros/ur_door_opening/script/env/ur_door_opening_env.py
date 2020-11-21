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
sub_step = rospy.get_param("/ML/sub_step")

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
        rospy.Subscriber("/robotiq_ft_wrench", WrenchStamped, self.wrench_stamped_callback) # FT300
#        rospy.Subscriber("/wrench", WrenchStamped, self.wrench_stamped_callback) # default UR5 sensor
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)
        rospy.Subscriber("/TactileSensor4/StaticData", StaticData, self.tactile_static_callback)
        rospy.Subscriber("/TactileSensor4/Dynamic", Dynamic, self.tactile_dynamic_callback)
        rospy.Subscriber("/imu/data", Imu, self.rt_imu_callback)
        rospy.Subscriber("/imu/data_3dmg", Imu, self.microstrain_imu_callback)

        # Gets training parameters from param server
        self.observations = rospy.get_param("/observations")
        
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

        shp_init_value1 = rospy.get_param("/init_joint_pose1/shp")
        shl_init_value1 = rospy.get_param("/init_joint_pose1/shl")
        elb_init_value1 = rospy.get_param("/init_joint_pose1/elb")
        wr1_init_value1 = rospy.get_param("/init_joint_pose1/wr1")
        wr2_init_value1 = rospy.get_param("/init_joint_pose1/wr2")
        wr3_init_value1 = rospy.get_param("/init_joint_pose1/wr3")
        self.init_joint_pose1 = [shp_init_value1, shl_init_value1, elb_init_value1, wr1_init_value1, wr2_init_value1, wr3_init_value1]
        init_pos1 = self.init_joints_pose(self.init_joint_pose1)
        self.arr_init_pos1 = np.array(init_pos1, dtype='float32')

        shp_init_value2 = rospy.get_param("/init_joint_pose2/shp")
        shl_init_value2 = rospy.get_param("/init_joint_pose2/shl")
        elb_init_value2 = rospy.get_param("/init_joint_pose2/elb")
        wr1_init_value2 = rospy.get_param("/init_joint_pose2/wr1")
        wr2_init_value2 = rospy.get_param("/init_joint_pose2/wr2")
        wr3_init_value2 = rospy.get_param("/init_joint_pose2/wr3")
        self.init_joint_pose2 = [shp_init_value2, shl_init_value2, elb_init_value2, wr1_init_value2, wr2_init_value2, wr3_init_value2]
        init_pos2 = self.init_joints_pose(self.init_joint_pose2)
        self.arr_init_pos2 = np.array(init_pos2, dtype='float32')

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

        # Controller type for ros_control
        self._ctrl_type =  rospy.get_param("/control_type")
        self.pre_ctrl_type =  self._ctrl_type

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
        self.rt_imu = Imu()
        self.microstrain_imu = Imu()
#        self.end_effector = Point() 
        self.previous_action = copy.deepcopy(self.arr_init_pos2)

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
        self.jointpub = JointTrajPub()

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
        ee_xyz = Point()
        ee_xyz = self.get_xyz(sub_action)
        if self.x_min < ee_xyz[0] and ee_xyz[0] < self.x_max and self.y_min < ee_xyz[1] and ee_xyz[1] < self.y_max and self.z_min < ee_xyz[2] and ee_xyz[2] < self.z_max:
            return True
        elif self.x_min > ee_xyz[0] or ee_xyz[0] > self.x_max:
            print("over the x_cartesian limits", self.x_min, "<", ee_xyz[0], "<", self.x_max)
            return False
        elif self.y_min > ee_xyz[1] or ee_xyz[1] > self.y_max:
            print("over the y_cartesian limits", self.y_min, "<", ee_xyz[1], "<", self.y_max)
            return False
        elif self.z_min > ee_xyz[2] or ee_xyz[2] > self.z_max:
            print("over the z_cartesian limits", self.z_min, "<", ee_xyz[2], "<", self.z_max)
            return False

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
#	print("[current_joint_pose]:", self.current_joint_pose, type(self.current_joint_pose))
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
        self.static_taxel = self.tactile_static.taxels
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

        delta_image_r, delta_image_l = self.get_image()
        self.cnn_image_r = agent.update_cnn(delta_image_r)
        self.cnn_image_l = agent.update_cnn(delta_image_l)
        self.cnn_image_r_list = self.cnn_image_r.tolist()
        self.cnn_image_l_list = self.cnn_image_l.tolist()
#        print("r_list", self.cnn_image_r_list)
#        print("l_list", self.cnn_image_l_list)

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
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

#        print("observation", list(map(round, observation, [3]*len(observation))))

        return observation

    def get_image(self):
        delta_image_r = []
        delta_image_l = []
        self.static_taxel = self.tactile_static.taxels
        for x in range(0, 28):
            delta_image_r.append((self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n)
        for x in range(0, 28):
            delta_image_l.append((self.static_taxel[1].values[x] - self.static_taxel_ini[1].values[x]) * self.taxel_n)
        return delta_image_r, delta_image_l

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

        self.gripper.goto_gripper_pos(0, False)
        time.sleep(1)
        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_far_pose)
        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_before_close_pose)
        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_close_door_pose)
        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_init_pos1)

    # Resets the state of the environment and returns an initial observation.
    def reset(self):
	# 1st: Go to initial position
        rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose1))
        self.max_knob_rotation = 0
        self.max_door_rotation = 0
        self.max_wirst3 = 0
        self.min_wirst3 = 0
        self.max_wirst2 = 0
        self.min_wirst2 = 0
        self.max_wirst1 = 0
        self.min_wirst1 = 0
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

        self.gripper.goto_gripper_pos(0, False)
        time.sleep(1)
#        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_far_pose)
#        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_before_close_pose)
#        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_close_door_pose)
#        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_init_pos1)

        # 2nd: Check all subscribers work.
        rospy.logdebug("check_all_systems_ready...")
        self.check_all_systems_ready()

        # 3rd: Get the initial state.
        self.force = self.wrench_stamped.wrench.force
        self.torque = self.wrench_stamped.wrench.torque

        self.knob_rpy_ini = copy.deepcopy(self.microstrain_imu.linear_acceleration.y / 9.8 * 1.57)
        self.quat = self.rt_imu.orientation
        self.door_rpy_ini = copy.deepcopy(self.cvt_quat_to_euler(self.quat))
        self.force_ini = copy.deepcopy(self.force)
        self.torque_ini = copy.deepcopy(self.torque)

        # 4th: Go to start position.
        self.jointpub.FollowJointTrajectoryCommand_reset(self.arr_init_pos2)
        time.sleep(1)
        self.gripper.goto_gripper_pos(120, False)
        time.sleep(1)

        # 5th: Get the State Discrete Stringuified version of the observations
        self.static_taxel = self.tactile_static.taxels
        self.static_taxel_ini = copy.deepcopy(self.static_taxel)
        self.eef_x_ini, self.eef_y_ini, self.eef_z_ini = self.get_xyz(self.init_joint_pose2)
        self.eef_rpy_ini = self.get_orientation(self.init_joint_pose2)

#        print("ee_xyz_ini", self.eef_x_ini, self.eef_y_ini, self.eef_z_ini) # ('ee_xyz_ini', -0.08859761113537656, 0.3680231810564474, 0.2769473319312816)
#        print("ee_rpy_ini", self.eef_rpy_ini) # x: -0.022194749153057504 y: 0.9997526460232887 z: -0.0011061239414521327

        rospy.logdebug("get_observations...")
       	observation = self.get_observations()
        return observation

    def _act(self, action):
        if self._ctrl_type == 'traj_pos':
            self.pre_ctrl_type = 'traj_pos'
            self._joint_traj_pubisher.FollowJointTrajectoryCommand(action)
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

        if self.max_wirst3 < action[5]:
            self.max_wirst3 = action[5]
        if self.min_wirst3 > action[5]:
            self.min_wirst3 = action[5]
        if self.max_wirst2 < action[4]:
            self.max_wirst2 = action[4]
        if self.min_wirst2 > action[4]:
            self.min_wirst2 = action[4]
        if self.max_wirst1 < action[3]:
            self.max_wirst1 = action[3]
        if self.min_wirst1 > action[3]:
            self.min_wirst1 = action[3]
        if self.max_elb < action[2]:
            self.max_elb = action[2]
        if self.min_elb > action[2]:
            self.min_elb = action[2]
        if self.max_shl < action[1]:
            self.max_shl = action[1]
        if self.min_shl > action[1]:
            self.min_shl = action[1]
        if self.max_shp < action[0]:
            self.max_shp = action[0]
        if self.min_shp > action[0]:
            self.min_shp = action[0]

        for x in range(1, sub_step + 1):
            self.cartesian_flag = 0
            sub_action = np.array(action) / sub_step * x
#            print("sub_action, sub_step", sub_action, sub_step, x)
            sub_action = sub_action + self.arr_init_pos2
            if self.check_cartesian_limits(sub_action) is True:
                self._act(sub_action)
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
    
                if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
                    print("force.x over the limit2", self.delta_force_x)
                    break
                elif self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
                    print("force.y over the limit2", self.delta_force_y)
                    break
                elif self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
                    print("force.z over the limit2", self.delta_force_z)
                    break
                elif self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
                    print("torque.x over the limit2", self.delta_torque_x)
                    break
                elif self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
                    print("torque.y over the limit2", self.delta_torque_y)
                    break
                elif self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
                    print("torque.z over the limit2", self.delta_torque_z)
                    break
                elif self.force_limit1 < self.delta_force_x or self.delta_force_x < -self.force_limit1:
                    self._act(self.previous_action)
                    print("force.x over the limit1", self.delta_force_x)
                elif self.force_limit1 < self.delta_force_y or self.delta_force_y < -self.force_limit1:
                    self._act(self.previous_action)
                    print("force.y over the limit1", self.delta_force_y)
                elif self.force_limit1 < self.delta_force_z or self.delta_force_z < -self.force_limit1:
                    self._act(self.previous_action)
                    print("force.z over the limit1", self.delta_force_z)
                elif self.torque_limit1 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit1:
                    self._act(self.previous_action)
                    print("torque.x over the limit1", self.delta_torque_x)
                elif self.torque_limit1 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit1:
                    self._act(self.previous_action)
                    print("torque.y over the limit1", self.delta_torque_y)
                elif self.torque_limit1 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit1:
                    self._act(self.previous_action)
                    print("torque.z over the limit1", self.delta_torque_x)
                else:
                    self.previous_action = copy.deepcopy(sub_action)
                    self.act_correct_n += 1
                    print("act correctly", self.act_correct_n)
            else:
                self.cartesian_flag = 1
                print("over the cartesian limits")

            self.min_static_taxel0 = 0
            self.min_static_taxel1 = 0
            self.max_static_taxel0 = 0
            self.max_static_taxel1 = 0
            self.static_taxel = self.tactile_static.taxels

            for x in range(0, 28):
                if self.min_static_taxel0 > (self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n:
                    self.min_static_taxel0 = (self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n
                if self.min_static_taxel1 > (self.static_taxel[1].values[x] - self.static_taxel_ini[1].values[x]) * self.taxel_n:
                    self.min_static_taxel1 = (self.static_taxel[1].values[x] - self.static_taxel_ini[1].values[x]) * self.taxel_n
                if self.max_static_taxel0 < (self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n:
                    self.max_static_taxel0 = (self.static_taxel[0].values[x] - self.static_taxel_ini[0].values[x]) * self.taxel_n
                if self.max_static_taxel1 < (self.static_taxel[1].values[x] - self.static_taxel_ini[1].values[x]) * self.taxel_n:
                    self.max_static_taxel1 = (self.static_taxel[1].values[x] - self.static_taxel_ini[1].values[x]) * self.taxel_n
#            print("self.min_static_taxel0, 1", self.min_static_taxel0, self.min_static_taxel1)

            if self.min_taxel0 > self.min_static_taxel0:
                self.min_taxel0 = self.min_static_taxel0
            if self.min_taxel1 > self.min_static_taxel1:
                self.min_taxel1 = self.min_static_taxel1
            if self.max_taxel0 < self.max_static_taxel0:
                self.max_taxel0 = self.max_static_taxel0
            if self.max_taxel1 < self.max_static_taxel1:
                self.max_taxel1 = self.max_static_taxel1

            if self.min_static_taxel0 < self.min_static_limit and self.min_static_taxel1 < self.min_static_limit:
                print("slipped and break the for loop(min over)", self.min_static_taxel0, self.min_static_taxel1)
                break
            if self.max_static_taxel0 > self.max_static_limit and self.max_static_taxel1 > self.max_static_limit:
                print("slipped and break the for loop(max over)", self.max_static_taxel0, self.max_static_taxel1)
                break
    
        # We now process the latest data saved in the class state to calculate
        # the state and the rewards. This way we guarantee that they work
        # with the same exact data.
        # Generate State based on observations
        observation = self.get_observations()

        # finally we get an evaluation based on what happened in the sim
        reward = self.compute_dist_rewards(action, update)
        done = self.check_done(update)

        return observation, reward, done, {}

    def compute_dist_rewards(self, action, update):
        knob_c = 1000         #1 rotation of knob(+)
        knob_bonus_c = 10    #2 bonus of knob rotation(+)
        panel_c = 50         #3 door panel open(+)

        force_c = 1         #6 over force limit1(-)
        force_c2 = 50        #  over force limit2(-)
        taxel_c = 0          #7 release the knob(-)
        act_0_n = 10         #8 action[0] (+)
        act_1_n = 10          #  action[1] (+)
        act_2_n = 10          #  action[2] (+)
        act_3_n = 10          #  action[3] (+)
        act_4_n = 10          #  action[4] (+)
        act_5_n = 10          #  action[5] (+)
        act_correct_c = 100  #9 act_correct (+)
        catesian_xyz_c = 1      #10 cartesian (+)
        catesian_rpy_c = 0.3
        cartesian_c = 3             #   bonus (+)
        compute_rewards = 0.0001

        self.quat = self.rt_imu.orientation
        self.door_rpy = self.cvt_quat_to_euler(self.quat)
        self.door_rotation.x = self.door_rpy.x - self.door_rpy_ini.x
        self.door_rotation.y = self.door_rpy.y - self.door_rpy_ini.y
        self.door_rotation.z = self.door_rpy.z - self.door_rpy_ini.z
        
        self.knob_rpy = self.microstrain_imu.linear_acceleration.y / 9.8 * 1.57
        self.knob_rotation = self.knob_rpy_ini - self.knob_rpy

        if self.max_knob_rotation < self.knob_rotation:
            self.max_knob_rotation = self.knob_rotation
        if self.max_door_rotation < self.door_rotation.z:
            self.max_door_rotation = self.door_rotation.z

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

        #1 rotation of knob(+), #2 bonus of knob rotation(+), #3 door panel open(+), 
        if self.knob_rotation < 0.2:
            compute_rewards = self.knob_rotation * knob_c
            print("##1 reward_knob_rotation_r", compute_rewards)
        elif 0.2 < self.knob_rotation < 0.4:
            compute_rewards = self.knob_rotation * knob_c + knob_bonus_c
            print("##1 reward_knob_rotation_r", compute_rewards)
        elif 0.4 < self.knob_rotation < 0.6:
            compute_rewards = self.knob_rotation * knob_c + knob_bonus_c * 2
            print("##1 reward_knob_rotation_r", compute_rewards)
        elif 0.6 < self.knob_rotation < 0.9:
            compute_rewards = self.knob_rotation * knob_c + knob_bonus_c * 3
            print("##1 reward_knob_rotation_r", compute_rewards)
        elif 0.9 < self.knob_rotation:
            compute_rewards = 0.9 * knob_c + knob_bonus_c * 3 + self.door_rotation.z * panel_c
            print("##1 reward_knob_rotation_r", compute_rewards)

#        print("knob_rotation", self.knob_rotation)
#        print("door_rotation.z", self.door_rotation.z)

        #6 over force limit1(-)
        if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
            compute_rewards -= force_c2 * ( n_step - update ) / n_step + force_c2
            print("##2 force_x limit2_r", -(force_c2 * ( n_step - update ) / n_step + force_c2))
        elif self.force_limit1 < self.delta_force_x or self.delta_force_x < -self.force_limit1:
            compute_rewards -= force_c * abs(abs(self.delta_force_x)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c
            print("##2 force_x limit1_r", -(force_c * abs(abs(self.delta_force_x)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c))
        if self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
            compute_rewards -= force_c2 * ( n_step - update ) / n_step + force_c2
            print("##2 force_y limit2_r", -(force_c2 * ( n_step - update ) / n_step + force_c2))
        elif self.force_limit1 < self.delta_force_y or self.delta_force_y < -self.force_limit1:
            compute_rewards -= force_c * abs(abs(self.delta_force_y)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c
            print("##2 force_y limit1_r", -(force_c* abs(abs(self.delta_force_y)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c))
        if self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
            compute_rewards -= force_c2 * ( n_step - update ) / n_step + force_c2
            print("##2 force_z limit2_r", -(force_c2 * ( n_step - update ) / n_step + force_c2))
        elif self.force_limit1 < self.delta_force_z or self.delta_force_z < -self.force_limit1:
            compute_rewards -= force_c * abs(abs(self.delta_force_z)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c
            print("##2 force_z limit1_r", -(force_c* abs(abs(self.delta_force_z)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c))

        if self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
            compute_rewards -= force_c2 * ( n_step - update ) / n_step + force_c2
            print("##2 torque_x limit2_r", -(force_c2 * ( n_step - update ) / n_step + force_c2))
        elif self.torque_limit1 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit1:
            compute_rewards -= force_c * abs(abs(self.delta_torque_x)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c
            print("##2 torque_x limit1_r", -(force_c* abs(abs(self.delta_torque_x)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c))
        if self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
            compute_rewards -= force_c2 * ( n_step - update ) / n_step + force_c2
            print("##2 torque_y limit2_r", -(force_c2 * ( n_step - update ) / n_step + force_c2))
        elif self.torque_limit1 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit1:
            compute_rewards -= force_c * abs(abs(self.delta_torque_y)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c
            print("##2 torque_y limit1_r", -(force_c* abs(abs(self.delta_torque_y)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c))
        if self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
            compute_rewards -= force_c2 * ( n_step - update ) / n_step + force_c2
            print("##2 torque_z limit2_r", -(force_c2 * ( n_step - update ) / n_step + force_c2))
        elif self.torque_limit1 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit1:
            compute_rewards -= force_c * abs(abs(self.delta_torque_z)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c
            print("##2 torque_z limit1_r", -(force_c* abs(abs(self.delta_torque_z)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c))

        #7 release the knob(-)
        if self.min_static_taxel0 < self.min_static_limit and self.min_static_taxel1 < self.min_static_limit:
            compute_rewards -= taxel_c * (n_step - update) / n_step + taxel_c
            print("##3 min_static_limit_r", -(taxel_c * (n_step - update) / n_step + taxel_c))
        elif self.max_static_taxel0 > self.max_static_limit and self.max_static_taxel1 > self.max_static_limit:
            compute_rewards -= taxel_c * (n_step - update) / n_step + taxel_c
            print("##3 max_static_limit_r", -(taxel_c * (n_step - update) / n_step + taxel_c))

        #8 joint(+)
        act_5_n_limit = 0       # -0.005
        act_5_p_limit = 1       #  1
        act_4_n_limit = 0       # -0.005
        act_4_p_limit = 0.03    #  0.03
        act_3_n_limit = -0.023  # -0.023
        act_3_p_limit = 0       #  0.005
        act_2_n_limit = 0       # -0.005
        act_2_p_limit = 0.14    #  0.14
        act_1_n_limit = -0.11   # -0.11
        act_1_p_limit = 0       #  0.005
        act_0_n_limit = -0.015  # -0.015
        act_0_p_limit = 0       #  0.005

        if act_5_n_limit < action[5] and action[5] < act_5_p_limit:
            compute_rewards += action[5] * act_5_n
            print("##4 action5_r", action[5] * act_5_n)
        if act_4_n_limit < action[4] and action[4] < act_4_p_limit:
            compute_rewards += action[4] * act_4_n
            print("##4 action4_r", action[4] * act_4_n)
        if act_3_n_limit < action[3] and action[3] < act_3_p_limit:
            compute_rewards += -action[3] * act_3_n
            print("##4 action3_r", -action[3] * act_3_n)
        if act_2_n_limit < action[2] and action[2] < act_2_p_limit:
            compute_rewards += action[2] * act_2_n
            print("##4 action2_r", action[2] * act_2_n)
        if act_1_n_limit < action[1] and action[1] < act_1_p_limit:
            compute_rewards += -action[1] * act_1_n
            print("##4 action1_r", -action[1] * act_1_n)
        if act_0_n_limit < action[0] and action[0] < act_0_p_limit:
            compute_rewards += -action[0] * act_0_n
            print("##4 action0_r", -action[0] * act_0_n)

        #9 act_correct(+)
        compute_rewards += self.act_correct_n / sub_step * act_correct_c
        print("##5 act_correct_r", self.act_correct_n / sub_step * act_correct_c)

        #10 cartesian(+)
        catesian_x = (1 - abs(self.eef_x_ini - self.eef_x) * 10) * catesian_xyz_c
        catesian_y = (1 - abs(self.eef_y_ini - self.eef_y) * 10) * catesian_xyz_c
        catesian_z = (1 - abs(self.eef_z_ini - self.eef_z) * 10) * catesian_xyz_c
        compute_rewards += catesian_x + catesian_y + catesian_z
        print("##6 catesian_xyz_r", catesian_x, catesian_y, catesian_z)

        catesian_rpy_x = (1 - abs(self.eef_rpy_ini.x - self.eef_rpy.x) * 10) * catesian_rpy_c
        catesian_rpy_y = (1 - abs(self.eef_rpy_ini.y - self.eef_rpy.y) * 10) * catesian_rpy_c
        catesian_rpy_z = (self.eef_rpy_ini.z - self.eef_rpy.z) * 10 * catesian_rpy_c
        compute_rewards += catesian_rpy_x + catesian_rpy_y + catesian_rpy_z
        print("##6 catesian_rpy_r", catesian_rpy_x, catesian_rpy_y, catesian_rpy_z)

        if self.cartesian_flag == 0:
            compute_rewards += cartesian_c
            print("##6 cartesian_bonus_r", cartesian_c)

        print("### total_rewards", compute_rewards)
        print("action", action)

	return compute_rewards

    def check_done(self, update):
        if update > 1:
            if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
                print("########## force.x over the limit2 ##########", update)
                return True
            elif self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
                print("########## force.y over the limit2 ##########", update)
                return True
            elif self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
                print("########## force.z over the limit2 ##########", update)
                return True
            elif self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
                print("########## torque.x over the limit2 ##########", update)
                return True
            elif self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
                print("########## torque.y over the limit2 ##########", update)
                return True
            elif self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
                print("########## torque.z over the limit2 ##########", update)
                return True
            elif self.min_static_taxel0 < self.min_static_limit and self.min_static_taxel1 < self.min_static_limit:
                print("########## static_taxles over the min limit ##########", update)
                return True
            elif self.max_static_taxel0 > self.max_static_limit and self.max_static_taxel1 > self.max_static_limit:
                print("########## static_taxles over the max limit ##########", update)
                return True
            else :
            	return False
