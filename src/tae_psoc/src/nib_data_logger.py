#!/usr/bin/env python

import rospy
import time
import psocScanner as psoc
import struct
import numpy as np
import os, sys
import math
import scipy.fftpack
import datetime
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt

from tae_psoc.msg import SensorPacket
from tae_psoc.msg import cmdToPsoc
from tae_psoc.msg import Sensor_Fast
from tae_psoc.msg import Sensor_Indiv

from std_msgs.msg import String, Int16
from basic_sensor_interface.msg import tendon_sns, joint_sns

import matplotlib.pyplot as plt



IDLE = 0
LOGGING = 1

currState = IDLE

# Basic sensor data variables
cur_tendon_data = np.zeros(9)
prev_tendon_data = np.zeros(9)
tendon_binary_engagement = np.zeros(9)
tendon_binary_cutoff = np.array([0,0,0,100,100,250,0,0,0])
cur_joint_data = np.zeros(6)
prev_joint_data = np.zeros(6)

# Nib data variables
cur_nib_indiv_data = np.zeros(36)
prev_nib_data = np.zeros(36)

# Alternative method
nib_data_storage_array = np.zeros((36,2))
joint_data_storage_array = np.zeros(6)
tendon_data_storage_array = np.zeros(9)

# Default saving files
dataDirectory = '/home/wilson/ros_ws_1/data_logging_storage/'
defaultFileName = 'most_recent_data'


def logging_cmd_callback(data):
    incomingString = str(data.data)
    global nib_data_storage_array, defaultFileName, dataDirectory
    global joint_data_storage_array, tendon_data_storage_array
    global prev_nib_data
    
    if (incomingString == 'flush_data'):
        # Zero out all data storage arrays
        nib_data_storage_array = np.zeros((36,2))
        tendon_data_storage_array = np.zeros((9,2))
        joint_data_storage_array = np.zeros((6,2))     
        
    elif (incomingString == 'save_nib'):
        np.save(dataDirectory + defaultFileName + '_nib', nib_data_storage_array)
        print(nib_data_storage_array.shape)
        
        # Plot data for sanity check
        fig, axs = plt.subplots(1)
        axs.set_title('X_c and F_grasp')
        x_c_storage_array = np.zeros(nib_data_storage_array.shape[1])
        f_grasp_storage_array = np.zeros(nib_data_storage_array.shape[1])
        x_delta_storage_array = np.zeros(nib_data_storage_array.shape[1])
        
        for i in range(nib_data_storage_array.shape[1]):
            x_c_storage_array[i] = np.average(nib_data_storage_array[0:18, i]) - np.average(nib_data_storage_array[18:36, i])
            f_grasp_storage_array[i] = np.average(nib_data_storage_array[:, i])
            if (i > 0):
                x_delta_storage_array[i] = x_c_storage_array[i] - x_c_storage_array[i-1]
                
        axs.plot(x_c_storage_array, label = 'x_c')
        axs.plot(f_grasp_storage_array, label = 'f_grasp')
#        axs.plot(x_delta_storage_array, label = 'x_d')
        axs.legend();
        
#        for i in range(36):
#            axs.plot(nib_data_storage_array[i,:])
        
        # Show plot
        plt.show()
        
        
    elif (incomingString == 'save_joint'):
        np.save(dataDirectory + defaultFileName + '_joint', joint_data_storage_array)
        
    elif (incomingString == 'save_tendon'):
        np.save(dataDirectory + defaultFileName + '_tendon', tendon_data_storage_array)
        
    elif (incomingString == 'save_all'):
        np.save(dataDirectory + defaultFileName + '_nib', nib_data_storage_array)
        np.save(dataDirectory + defaultFileName + '_joint', joint_data_storage_array)
        np.save(dataDirectory + defaultFileName + '_tendon', tendon_data_storage_array)

    
# Callback function for tendon force sensing data     
def tendon_sns_callback(data):
    # Bump old data to prev variable
    global prev_tendon_data, cur_tendon_data, tendon_data_storage_array
    prev_tendon_data = cur_tendon_data
    # Fill cur_tendon_data with updated data
    # Finger 1
    cur_tendon_data[0] = data.prox1
    cur_tendon_data[1] = data.dist1
    cur_tendon_data[2] = data.hype1
    # Finger 2
    cur_tendon_data[3] = data.prox2
    cur_tendon_data[4] = data.dist2
    cur_tendon_data[5] = data.hype2
    # Thumb
    cur_tendon_data[6] = data.prox3
    cur_tendon_data[7] = data.dist3
    cur_tendon_data[8] = data.hype3
    
    # Go through and set binary values for taught or not
    for i in range(9):
        if (cur_tendon_data[i] > tendon_binary_cutoff[i]):
            tendon_binary_engagement[i] = 1
        else:
            tendon_binary_engagement[i] = 0
    
    
# Callback function for joint position sensing data
def joint_sns_callback(data):
    # Bump old data to prev variable
    global prev_joint_data, cur_joint_data, joint_data_storage_array
    prev_joint_data = cur_joint_data
    # Finger 1
    cur_joint_data[0] = data.prox1
    cur_joint_data[1] = data.dist1
    # Finger 2
    cur_joint_data[2] = data.prox2
    cur_joint_data[3] = data.dist2
    # Thumb
    cur_joint_data[4] = data.prox3
    cur_joint_data[5] = data.dist3
    
    
# Callback function for nib indiv sensor data
def nib_sns_callback(data):
    # Bump old data to prev variable
    global nib_data_storage_array, cur_nib_indiv_data
    
    # Append current data to saved data array
    cur_nib_indiv_data = np.reshape(np.asarray(data.sns_1_Indiv), (-1,1))
    nib_data_storage_array = np.append(nib_data_storage_array, cur_nib_indiv_data, axis = 1)
    
    

def mainLoop():
    global currState

    # Setup node
    rospy.init_node('nib_data_logger')

    # Setup all the subscribers
    rospy.Subscriber('logging_cmd_callback', String, logging_cmd_callback)
    rospy.Subscriber("finger_tendon_sensors", tendon_sns, tendon_sns_callback)
    rospy.Subscriber("finger_joint_sensors", joint_sns, joint_sns_callback)
    rospy.Subscriber("Sensor_Indiv", Sensor_Indiv, nib_sns_callback)
    
    # Save data at 50 hz regardless of sensor speed
    while not rospy.is_shutdown():  
        # TODO: Live plot? 
        x = 1

#    rate.sleep()
            
#            if plotShow:
#                #%% Plot the data
#
#                fig, axs = plt.subplots(SensorNum)
#                axs[0].set_title('Fast Mode')
#                for i in range(0,SensorNum):
#                    axs[i].plot(sensor_data_history_Fast[:,:,i])
#
#                fig1, axs1 = plt.subplots(SensorNum)
#                axs1[0].set_title('Indiv Mode')
#                for i in range(0,SensorNum):
#                    axs1[i].plot(sensor_data_history_Indiv[:,:,i])
#
#
#                plt.show()
#                


# Save output            
            
#            directory = ResultSavingDirectory +'/' + currDateOnlyString
#            if not os.path.exists(directory):
#            	os.makedirs(directory)
#
#            
#            output_file = directory + '/'+ 'result_' +currDateTimeString + SavingFileName + '.mat'
#            	
#            # # currDateString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
#            # output_file = ResultSavingDirectory + '/'+ 'result_' +currDateString + SavingFileName + '.csv'
#            
#
#            # Save as .mat file
#            savemat(output_file, 
#                {'sensor_data_history_Fast':sensor_data_history_Fast,
#                 'sensor_data_history_Indiv':sensor_data_history_Indiv,
#                  'sensingMode' : sensingMode,                  
#                  })
#
#            # np.savetxt(output_file, sensor_1_data_history_second, delimiter=",")
#            print("file Saved")

                
if __name__ == '__main__':
    try:
        print("Started Data Logger Node")
        mainLoop()
    except rospy.ROSInterruptException: pass








