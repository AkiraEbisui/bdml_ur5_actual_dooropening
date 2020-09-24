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
from .controllers_connection import ControllersConnection

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

    def FollowJointTrajectoryCommand(self, joints_array): # dtype=float32), <type 'numpy.ndarray'>
    	rospy.loginfo("FollowJointTrajectoryCommand")

        g = FollowJointTrajectoryGoal()
    	g.trajectory = JointTrajectory()
    	g.trajectory.joint_names.append("shoulder_pan_joint")
    	g.trajectory.joint_names.append("shoulder_lift_joint")
    	g.trajectory.joint_names.append("elbow_joint")
    	g.trajectory.joint_names.append("wrist_1_joint")
    	g.trajectory.joint_names.append("wrist_2_joint")
    	g.trajectory.joint_names.append("wrist_3_joint")
    	    	    
    	dt = 3 	#default 0.01
    	g.trajectory.points = []

### actual home position
#        Q1 = [1.55, -1.81, 2.26, -2.02, -1.57, 3.11]
#        g.trajectory.points.append(JointTrajectoryPoint(positions=Q1, velocities=[0]*6, time_from_start=rospy.Duration(dt)))

        Q2 = [joints_array[0], joints_array[1], joints_array[2], joints_array[3], joints_array[4], joints_array[5]]
        g.trajectory.points.append(JointTrajectoryPoint(positions=Q2, velocities=[0]*6, time_from_start=rospy.Duration(dt)))

        client.send_goal(g)
        try:
            client.wait_for_result()
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def GrpCommand(self, joints_array): # dtype=float32), <type 'numpy.ndarray'>
#    	rospy.loginfo("GrpCommand")
    	try:    
    	    rospy.loginfo (rospy.get_rostime().to_sec())
    	    while rospy.get_rostime().to_sec() == 0.0:
    	    	time.sleep(0.1)
    	    	rospy.loginfo (rospy.get_rostime().to_sec())

    	    jt = JointTrajectory()
    	    jt.header.stamp = rospy.Time.now()
    	    jt.header.frame_id = "grp"
    	    jt.joint_names.append("simple_gripper_right_driver_joint")
    	    jt.joint_names.append("simple_gripper_left_driver_joint")
    	    jt.joint_names.append("simple_gripper_right_follower_joint")
    	    jt.joint_names.append("simple_gripper_left_follower_joint")
    	    jt.joint_names.append("simple_gripper_right_spring_link_joint")
    	    jt.joint_names.append("simple_gripper_left_spring_link_joint")
    	    	    
    	    dt = 0.1 	#default 0.01
    	    p = JointTrajectoryPoint()	
    	    p.positions.append(joints_array[0])
    	    p.positions.append(joints_array[1])
    	    p.positions.append(joints_array[2])
    	    p.positions.append(joints_array[3])
    	    p.positions.append(joints_array[4])
    	    p.positions.append(joints_array[5])
    	    jt.points.append(p)

    	    # set duration
    	    jt.points[0].time_from_start = rospy.Duration.from_sec(dt)

    	    self._grp_pub.publish(jt)

    	except rospy.ROSInterruptException: pass

#if __name__=="__main__":
#    rospy.init_node('joint_publisher_node', log_level=rospy.WARN)
#    joint_publisher = JointTrajPub()
#    rate_value = 8.0

