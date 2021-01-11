#!/usr/bin/env python
import roslib; roslib.load_manifest('ur_driver')
import actionlib
from control_msgs.msg import *

import rospy
import math
import time
import copy
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# MoveIt
import os
import sys
import numpy as np

from ur5_interface_for_door import UR5Interface
from robotiq_interface_for_door import RobotiqInterface

moveit = rospy.get_param("/moveit")
dt_reset = rospy.get_param("/act_params/dt_reset")

class JointTrajPub(object):
    def __init__(self):
        global client
        client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
        print "Waiting for server..."
        client.wait_for_server()
        print "Connected to server"

    def check_publishers_connection(self):
    	"""
    	Checks that all the publishers are working
    	:return:
    	"""
    	rate = rospy.Rate(1)  # 1hz
    	while (self._joint_traj_pub.get_num_connections() == 0):
    	    rospy.logdebug("No subscribers to vel_traj_controller yet so we wait and try again")
    	    try:
    	    	self._ctrl_conn.start_controllers(controllers_on="vel_traj_controller")
    	    	rate.sleep()
    	    except rospy.ROSInterruptException:
    	    	# This is to avoid error when world is rested, time when backwards.
    	    	pass
    	rospy.logdebug("_joint_traj_pub Publisher Connected")
    	rospy.logdebug("All Joint Publishers READY")

    def FollowJointTrajectoryCommand(self, joints_array, dt_act): # dtype=float32), <type 'numpy.ndarray'>
#    	rospy.loginfo("FollowJointTrajectoryCommand")

        try:
#            rospy.loginfo (rospy.get_rostime().to_sec())
#            while rospy.get_rostime().to_sec() == 0.0:
#                time.sleep(0.1)
#                rospy.loginfo (rospy.get_rostime().to_sec())
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names.append("shoulder_pan_joint")
            g.trajectory.joint_names.append("shoulder_lift_joint")
            g.trajectory.joint_names.append("elbow_joint")
            g.trajectory.joint_names.append("wrist_1_joint")
            g.trajectory.joint_names.append("wrist_2_joint")
            g.trajectory.joint_names.append("wrist_3_joint")
    	    	    
            #dt = 1 	#default 3
            g.trajectory.points = []

            Q1 = [joints_array[0], joints_array[1], joints_array[2], joints_array[3], joints_array[4], joints_array[5]]
            print("Q1", Q1)
            g.trajectory.points.append(JointTrajectoryPoint(positions=Q1, velocities=[0]*6, time_from_start=rospy.Duration(dt_act)))

            client.send_goal(g)
        
            client.wait_for_result()

        except rospy.ROSInterruptException: pass

    def FollowJointTrajectoryCommand_reset(self, joints_array): # dtype=float32), <type 'numpy.ndarray'>
#    	rospy.loginfo("FollowJointTrajectoryCommand_reset")

        try:
#            rospy.loginfo (rospy.get_rostime().to_sec())
#            while rospy.get_rostime().to_sec() == 0.0:
#                time.sleep(0.1)
#                rospy.loginfo (rospy.get_rostime().to_sec())
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names.append("shoulder_pan_joint")
            g.trajectory.joint_names.append("shoulder_lift_joint")
            g.trajectory.joint_names.append("elbow_joint")
            g.trajectory.joint_names.append("wrist_1_joint")
            g.trajectory.joint_names.append("wrist_2_joint")
            g.trajectory.joint_names.append("wrist_3_joint")
    	    	    
            #dt_reset = 1 	#default 3
            g.trajectory.points = []

            Q2 = [joints_array[0], joints_array[1], joints_array[2], joints_array[3], joints_array[4], joints_array[5]]
            print("Q2", Q2)
            g.trajectory.points.append(JointTrajectoryPoint(positions=Q2, velocities=[0]*6, time_from_start=rospy.Duration(dt_reset)))

            client.send_goal(g)
        
            client.wait_for_result()

        except rospy.ROSInterruptException: pass

    def MoveItCommand(self, action):
        try:
            self.ur5 = UR5Interface()
            self.ur5.goto_pose_target(action, False)
            print("moveit", action)
        except rospy.ROSInterruptException: pass

    def MoveItJointTarget(self, joint_array):
        try:
            self.ur5 = UR5Interface()
            self.ur5.goto_joint_target(joint_array, False)
            print("moveit_target", joint_array)
        except rospy.ROSInterruptException: pass

#    def MoveItGrpOpen(self):
#        try:
#            self.grp = moveit_commander.MoveGroupCommander("gripper")
#            self.grp.set_named_target('open')
#            self.grp.go(wait=False)
#        except rospy.ROSInterruptException: pass

#    def MoveItGrpClose(self):
#        try:
#            self.grp = moveit_commander.MoveGroupCommander("gripper")
#            self.grp.set_named_target('close0.31')
#            self.grp.go(wait=False)
#        except rospy.ROSInterruptException: pass

