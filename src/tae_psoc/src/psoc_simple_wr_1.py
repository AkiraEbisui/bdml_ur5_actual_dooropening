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
from std_msgs.msg import String

# Wilson variables
# State definitions
IDLE = 0
SETUP_SENSOR = 1
STREAMING = 2
SHUTDOWN_SENSOR = 3
state = SETUP_SENSOR
sensor_select = 36 # Default to MODE_TORSION_AND_INDIV

# Global variables for use later
periodInput = np.zeros(2)
num_sensors_2 = 0
IsTwoModeMerged = False

# Boolean for determining whether already set up or not
is_setup = False

def state_callback(data):
    incomingString = data.data
    global state, sensor_select
    if (incomingString == "manual_input"):
        state = SHUTDOWN_SENSOR
    elif (incomingString == "drive_hand"):
        state = SETUP_SENSOR
        sensor_select = MODE_TORSION_AND_INDIV

# Sensor mode options
MODE_ONE_PAD = 0 
MODE_FOUR_PAD = 1
MODE_NINE_PAD = 2
MODE_INDIVIDUAL = 3
MODE_CAPSENSE = 17
MODE_ONE_AND_FOUR =32   #Problematic  at 1kHz
MODE_NINE_AND_INDIV= 33 
MODE_ONE_AND_NINE = 34  #Problematic
MODE_FOUR_AND_INDIV = 35 # 250
MODE_TORSION_AND_INDIV = 36 #0X24
def set_sensor_mode(sensingMode): 
    global periodInput, num_sensors_2, IsTwoModeMerged
#    slopeCompensateOn = True
    
    if sensingMode == MODE_ONE_PAD:
        SamplingFreq = 1.5e3
    elif sensingMode == MODE_FOUR_PAD:
        SamplingFreq = 300; 
    elif sensingMode == MODE_NINE_PAD:
        SamplingFreq = 150
    elif sensingMode == MODE_INDIVIDUAL:
        SamplingFreq = 60
    elif sensingMode == MODE_ONE_AND_FOUR:
        SamplingFreq = 1e3
        num_sensors_2 = 4
        IsTwoModeMerged = True
    elif sensingMode == MODE_ONE_AND_NINE:
        SamplingFreq = 1e3
        num_sensors_2 = 9
        IsTwoModeMerged = True
    elif sensingMode == MODE_NINE_AND_INDIV:
        SamplingFreq = 110
        num_sensors_2 = 36
        IsTwoModeMerged = True
    elif sensingMode == MODE_FOUR_AND_INDIV or sensingMode == MODE_TORSION_AND_INDIV:        
        SamplingFreq = 250 # For now let's keep it 250, it was 300
        num_sensors_2 = 36
        IsTwoModeMerged = True

    SamplingPeriod = 1.0/SamplingFreq
    KitprogTimerClockFreq = 100e3
    tempPeriodInput = divmod(math.floor(SamplingPeriod * KitprogTimerClockFreq) , 2**8)
    periodInput = np.array([tempPeriodInput[1], tempPeriodInput[0]])

def psoc_pub_sub():
    print("Started psoc_pub_sub")
    global state, num_sensors_2

    # Start node
    rospy.init_node('psoc_pub_sub')
    
    # Subscribe to master_state to know which sensors to be publishing
    rospy.Subscriber('master_state',String, state_callback)
        
    # Setup publishers for use later depending on state
    pub_Fast = rospy.Publisher('Sensor_Fast', Sensor_Fast, queue_size=1)
    pub_Indiv = rospy.Publisher('Sensor_Indiv', Sensor_Indiv, queue_size=1)
    msg_Fast = Sensor_Fast()
    msg_Indiv = Sensor_Indiv() 
    
    # Setup variables: Tae's stuff... don't know exact details
    counter = 0
    SensorExist = 1
    plotShow = 1
    SensorNum = 3
    SensorAddress = np.array([8,9,10])

    while not rospy.is_shutdown():
        if (state == SETUP_SENSOR):
            # Set sensor mode
            set_sensor_mode(sensor_select)
            sensingMode = sensor_select
            
            # Allocate serial ports how we want them with motors and sensors
            ts = psoc.TactileSensor(port="/dev/ttyACM2")
            ts.ser.flushInput()
            
            # Assigning the Number of Sensor and Address
            thisInputArray = np.append(np.append(ord('a'), SensorNum), SensorAddress)
            ts.sendNum(thisInputArray)    
            time.sleep(0.1)
            
            #Set Period of Sampling
            thisInputArray = np.array([ord('p'), periodInput[0], periodInput[1]])
            ts.sendNum(thisInputArray)        
            time.sleep(0.1)
            
            # Select Sensing Mode
            thisInputArray = np.array([ord('m'), sensingMode])    
            ts.sendNum(thisInputArray)
            time.sleep(0.1)
            
            # Query the Data Length
            ts.sendChar("q")
            time.sleep(0.1)
            
#            print(ts.ser.read_all())
            
            ts.packet_size = ord(ts.ser.read(1))-1
            num_sensors_1= int((ts.packet_size - 1) / 2)    
            ts.num_sensors= int((ts.packet_size - 1) / 2)
            ts.unpackFormat = '<'
            for i in range(0,ts.packet_size):
                ts.unpackFormat = ts.unpackFormat + 'B'
                
            if IsTwoModeMerged: #% Deal with Merging Techniq
                num_sensors_1 = int(num_sensors_1/2)
                print(ts.packet_size)
                sensorIndexInData_1 = list(range(int(num_sensors_1),int((ts.packet_size - 1) / 2)) )  # Last Half is the Fast Mode
                sensorIndexInData_2 = list(range(0, int(num_sensors_1)))    # First Half is the Indiv Mod
                
                groupIndexMax = num_sensors_2/ num_sensors_1 -1; #% Follows C convention
                
            else:
                num_sensors_2 = 0   # % Dummy
            
            print("num_sensor 1=")
            print(num_sensors_1)
            print("num_sensor 2=")
            print(num_sensors_2)
            
            # Get Initial samples for measuring Offset Values    
            initialSamplingNum = 16 # it should be an even number
            
            # Buffer
            init_BufferSize = 5000;

            sensor_data_history_Fast = np.zeros((init_BufferSize,num_sensors_1, SensorNum), dtype = 'i')
            sensor_data_history_Indiv = np.zeros((init_BufferSize,num_sensors_2, SensorNum), dtype = 'i')

            sensor_offset_Fast = np.zeros((1, num_sensors_1, SensorNum), dtype = 'i')
            sensor_offset_Indiv = np.zeros((1, num_sensors_2, SensorNum), dtype = 'i')

            readCountArray = np.zeros((SensorNum, 2), dtype = 'i') # 2 is for Fast , Indiv Mode.
            sensor_offsetObtained = False 

            #Start Streaming
            ts.sendChar("s")
            
            global is_setup
            is_setup = True
            print("Sensor setup and streaming")
            state = STREAMING
            
        elif (state == STREAMING):
            while True:
                firstByte = ord(ts.ser.read(1))
                if firstByte % 16 == ts.STX:  # Changed for Address Info coupling
                   thisAddress = math.floor(firstByte / 16)
                   thisSensorIdx = np.where(SensorAddress == thisAddress)[0]
                   if thisSensorIdx.size == 1:
                       thisSensorIdx = thisSensorIdx[0]
                       break

            #Read rest of the data
            tempSampledData = ts.readRestData()       

            if IsTwoModeMerged:
                #Fast Mode
                sensor_data_history_Fast[readCountArray[thisSensorIdx,0], :, thisSensorIdx] = tempSampledData[0,sensorIndexInData_1] - sensor_offset_Fast[0,:,thisSensorIdx]
                readCountArray[thisSensorIdx,0] += 1

                # Indiv Mode
                groupIndex = int(ts.groupIndex)

                sensor_data_history_Indiv[readCountArray[thisSensorIdx,1], groupIndex*int(num_sensors_1):(groupIndex+1)*int(num_sensors_1), thisSensorIdx]= tempSampledData[0,sensorIndexInData_2]
                if groupIndex == groupIndexMax:
                    sensor_data_history_Indiv[readCountArray[thisSensorIdx,1],:,thisSensorIdx] = sensor_data_history_Indiv[readCountArray[thisSensorIdx,1],:,thisSensorIdx] - sensor_offset_Indiv[0,:,thisSensorIdx]
                    readCountArray[thisSensorIdx,1] += 1

                if sensor_offsetObtained and thisSensorIdx == SensorNum-1:
                    #Make the Fast Message
                    msg_Fast.sns_1_Fast = sensor_data_history_Fast[readCountArray[0,0]-1,:,0]
                    if SensorNum >= 2:
                        msg_Fast.sns_2_Fast = sensor_data_history_Fast[readCountArray[1,0]-1,:,1]
                        if SensorNum==3:
                            msg_Fast.sns_3_Fast = sensor_data_history_Fast[readCountArray[2,0]-1,:,2]
                    pub_Fast.publish(msg_Fast)

                    # Make the Indiv Message
                    if groupIndex == groupIndexMax:
                        #Make the Fast Message
                        msg_Indiv.sns_1_Indiv = sensor_data_history_Indiv[readCountArray[0,1]-1,:,0]
                        if SensorNum >= 2:
                            msg_Indiv.sns_2_Indiv = sensor_data_history_Indiv[readCountArray[1,1]-1,:,1]
                            if SensorNum==3:
                                msg_Indiv.sns_3_Indiv = sensor_data_history_Indiv[readCountArray[2,1]-1,:,2]

                        pub_Indiv.publish(msg_Indiv)

                # Obtain Offset from initial few samples
                if not sensor_offsetObtained and  readCountArray[SensorNum-1,1] == initialSamplingNum:
                    for i in range(0,SensorNum):
                        sensor_offset_Fast[0,:,i] = np.mean(sensor_data_history_Fast[readCountArray[i,0]-initialSamplingNum:readCountArray[i,0],:,i],axis=0, dtype = 'i')
                        sensor_offset_Indiv[0,:,i] = np.mean(sensor_data_history_Indiv[5:readCountArray[i,1],:,i],axis=0, dtype = 'i')
                        sensor_offsetObtained = True

                #if the data overflows
                if readCountArray[SensorNum-1,0] > sensor_data_history_Fast.shape[0]-1:
                    print("OverFlow")
                    sensor_data_history_Fast = np.append(sensor_data_history_Fast, np.zeros((init_BufferSize,num_sensors_1, SensorNum), dtype = 'i'), axis = 0 )

                if readCountArray[SensorNum-1,1] > sensor_data_history_Indiv.shape[0]-1:
                    sensor_data_history_Indiv = np.append(sensor_data_history_Indiv, np.zeros((init_BufferSize,num_sensors_2, SensorNum), dtype = 'i'), axis = 0 )

        elif (state == SHUTDOWN_SENSOR):
            global is_setup
            if is_setup:
                ts.sendChar("i")
                if IsTwoModeMerged:
                    sensor_data_history_Fast = sensor_data_history_Fast[0:readCountArray[SensorNum-1,0], :, :]
                    sensor_data_history_Indiv = sensor_data_history_Indiv[0:readCountArray[SensorNum-1,1], :, :]    
                ts.closePort()
                print("Sensor shut down")
                is_setup = False
            state = IDLE    

                
if __name__ == '__main__':
    try:
        psoc_pub_sub()
    except rospy.ROSInterruptException: pass