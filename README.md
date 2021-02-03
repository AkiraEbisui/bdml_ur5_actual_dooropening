# Repository for UR5 + Robotiq Demo and RL for dooropening task in actual machine
Demo code for controlling the UR5 and Robotic 2F gripper in BDML with MoveIt!
RL code for dooropening task with ur_driver

## Requirements:
1. SW
    + ROS melodic (runs well on Ubuntu 18.04)
    + OpenAI ROS
    (+ ROS MoveIt!)
2. HW
    + imu_tools
    + 3DM-GX3-25 imu sensor
      https://www.microstrain.com/inertial/3dm-gx3-25
    + USB Output 9-axis IMU sensor module
      http://wiki.ros.org/rt_usb_9axisimu_driver
    + Robotiq sensors
    + Robotiq force-torque sensor(FT-300)

What's in side the package:
This repository contains a whole catkin workspace. Within it there are eight
ROS packages.
1. Robotiq's ROS-industrial package (cloned on 2019, including force-torque sensor) 
2. Universal Robots's ROS-industrial package (cloned on 2019) 
3. UR5 rospy interface
4. openai_ros
5. imu_tools (madgwick_filter for 3DM-GX3-25 imu sensor)
6. microstrain_3dmgx2_imu (for 3DM-GX3-25 imu sensor)
7. rt_usb_9axismu_driver (for USB Output 9-axis IMU sensor module)
   https://github.com/rt-net/rt_usb_9axisimu_driver
8. tactilesensors4 (for Robotiq sensors)

## Make virtual env:
1. Make Virtual Env in python2
```console 
cd [project dir] 
virtualenv venv 
```

2. Activate/Deactivate 
```console 
source venv/bin/activate 
```
```console 
([newenvname])$ deactivate
```

## Activate the virtual env and install the packages by pip install:
1. numpy==(1.16.6)
2. matplotlib==(2.2.5)
3. sklearn-utils==(0.0.15)
4. tensorflow==1.14.0       *the version is important
5. tensorflow-gpu==1.14.0   *the version is important
6. pyyaml==(5.3.1)
7. rospkg==(1.2.8)
8. gym==(0.16.0)

## Compilation:
1. First time compiling this ROS workspace do:
```console
source /opt/ros/melodic/setup.bash
cd [path/to/demo/project/root]
catkin build
```

## Running this code for RL dooropening task in actual machine:
1. source env
```console 
source [path/to/demo/project/root]/devel/setup.bash
```

2. Launch ur5, robotiq gripper, IMU sensors, robotiq-ft-sensor, and robotiq tactile sensors connection
```console 
roslaunch ur5_demo openai_demo.launch 
```

3. Run RL training script
```console 
python ppo_gae_main.py sys.argv[1]
```
*sys.argv[1] is a folder name for saving the result

4. Run RL test script
```console 
python ppo_gae_test_main.py sys.argv[1]
```
This will try to move the UR5 and the robotiq gripper to open the door.
*sys.argv[1] is a folder name for saving the result
