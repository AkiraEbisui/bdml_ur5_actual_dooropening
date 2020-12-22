#!/usr/bin/env python

import rospy


import time
import psocScanner as psoc
import struct
import numpy as np
import os, sys



from tae_psoc.msg import SensorPacket
from tae_psoc.msg import cmdToPsoc


IDLE = 0
STREAMING = 1


NO_CMD = 0
START_CMD = 2
IDLE_CMD = 3



currState = IDLE
CMD_in = NO_CMD


def callback(data):
    global CMD_in    
    CMD_in = data.cmdInput
    

def mainLoop(savingFileName):
    global currState
    global CMD_in   

    # #Gripper is a C-Model with a TCP connection
    # gripper = robotiq_c_model_control.baseCModel.robotiqBaseCModel()
    # gripper.client = robotiq_modbus_rtu.comModbusRtu.communication()

    # #We connect to the address received as an argument
    # gripper.client.connectToDevice(device)

    rospy.init_node('psocPubSub')

    #Each Sensor Reading is Published to topic 'SensorReading'
    pub = rospy.Publisher('SensorPacket', SensorPacket, queue_size=10)
    rospy.Subscriber('cmdToPsoc',cmdToPsoc, callback)

    msg = SensorPacket()

    # #The Gripper command is received from the topic named 'CModelRobotOutput'
    # rospy.Subscriber('CModelRobotOutput', outputMsg.CModel_robot_output, gripper.refreshCommand)    
    
    counter = 0

    # rospy.spin();

    ResultSavingDirectory = '/home/tae/Data'
    SavingFileName = savingFileName  #'test_box'

    SensorExist = 1
    plotShow = 1





    #We loop
    while not rospy.is_shutdown():
        if currState == IDLE and CMD_in == START_CMD:
            CMD_in = NO_CMD
            currState = STREAMING

            #Get and publish the Each Sensor Read
            msg.sensorRead = [counter, 0 , 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            msg.dataLength = 10;
            counter = counter + 1
            pub.publish(msg)     

            if SensorExist:
                import datetime
                currDateString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
            #    output_file = ResultSavingDirectory + '\\'+ 'result_' +currDateString + SavingFileName + '.html'

                ts = psoc.TactileSensor(port="/dev/ttyACM0")
                ts.ser.flushInput()
                
                ts.sendChar("i")
                time.sleep(0.01)
                ts.sendChar("q")
                time.sleep(0.01)
                
                ts.packet_size = ord(ts.ser.read(1))-1
                ts.num_sensors= (ts.packet_size - 1) / 2
                ts.unpackFormat = '<'
                for i in range(0,ts.packet_size):
                    ts.unpackFormat = ts.unpackFormat + 'B'
                    
                    
                
                tic = time.time()
                
                #Start Streaming
                ts.sendChar("s")
                
                #%% Get Initial samples for measuring Offset Values
                
                initialSamplingNum = 15
                initialData = np.zeros((initialSamplingNum,ts.num_sensors))
                
                for i in range(0,initialSamplingNum):
                #    print(ord(ts.ser.read(1)))    
                    while ord(ts.ser.read(1))!= ts.STX:
                        continue
                                
                    #Read rest of the data
                    initialData[i,:] = ts.readRestData()
                
                #Ignore the first row
                initialData = initialData[5:None,:]
                
                initial_offset = np.mean(initialData,axis=0)
                
                #%% Buffer
                init_BufferSize = 5000;
                dataBuffer = np.zeros((init_BufferSize,ts.num_sensors))
                storingCounter = 0;
                    
                #ts.sendChar("i")
                


                # Loop for getting sensor readings

                while not CMD_in == IDLE_CMD:

                    while ord(ts.ser.read(1))!= ts.STX:
                        continue

                    pub.publish(msg) 
                                
                    #Read rest of the data
                    dataBuffer[storingCounter,:] = ts.readRestData()
                    
                    
                    storingCounter += 1
                    
                    #if the data overflows
                    if storingCounter > dataBuffer.shape[0]-1:
                        dataBuffer = np.append(dataBuffer, np.zeros((init_BufferSize,ts.num_sensors)), axis=0)
            

                
                dataBuffer = dataBuffer[0:storingCounter-1,:]
                ts.sendChar("i")


                #r.stopMotion()
                
                ts.closePort()
                
                #r.demo()
                
                if plotShow:
                    #%% Plot the data
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(dataBuffer)
                    plt.ylabel('Digital Count')
                    plt.xlabel('sample')
                    plt.grid()
                    plt.show()
                
                #%% Save Output
                import datetime
                currDateOnlyString = datetime.datetime.now().strftime("%y%m%d")
                directory = ResultSavingDirectory +'/' + currDateOnlyString
                if not os.path.exists(directory):
                	os.makedirs(directory)

                currDateTimeString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
                output_file = directory + '/'+ 'result_' +currDateTimeString + SavingFileName + '.csv'
                	
                # # currDateString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
                # output_file = ResultSavingDirectory + '/'+ 'result_' +currDateString + SavingFileName + '.csv'
                

                np.savetxt(output_file, dataBuffer, delimiter=",")
                print("file Saved")

                CMD_in = NO_CMD
                currState = IDLE














                
if __name__ == '__main__':
    try:
        print("Started!")
        mainLoop(sys.argv[1])
    except rospy.ROSInterruptException: pass









##########################33



# runTime = 15



# SensorExist = 1
# plotShow = 1

# ResultSavingDirectory = '/home/tae/Data/JooyeunShear'
# SavingFileName = 'test_box'



# if SensorExist:
#     import datetime
#     currDateString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
# #    output_file = ResultSavingDirectory + '\\'+ 'result_' +currDateString + SavingFileName + '.html'

#     ts = psoc.TactileSensor(port="/dev/ttyACM0")
#     ts.ser.flushInput()
    
#     ts.sendChar("i")
#     time.sleep(0.01)
#     ts.sendChar("q")
#     time.sleep(0.01)
    
#     ts.packet_size = ord(ts.ser.read(1))-1
#     ts.num_sensors= (ts.packet_size - 1) / 2
#     ts.unpackFormat = '<'
#     for i in range(0,ts.packet_size):
#         ts.unpackFormat = ts.unpackFormat + 'B'
        
        
    
#     tic = time.time()
    
#     #Start Streaming
#     ts.sendChar("s")
    
#     #%% Get Initial samples for measuring Offset Values
    
#     initialSamplingNum = 15
#     initialData = np.zeros((initialSamplingNum,ts.num_sensors))
    
#     for i in range(0,initialSamplingNum):
#     #    print(ord(ts.ser.read(1)))    
#         while ord(ts.ser.read(1))!= ts.STX:
#             continue
                    
#         #Read rest of the data
#         initialData[i,:] = ts.readRestData()
    
#     #Ignore the first row
#     initialData = initialData[5:None,:]
    
#     initial_offset = np.mean(initialData,axis=0)
    
#     #%% Buffer
#     init_BufferSize = 5000;
#     dataBuffer = np.zeros((init_BufferSize,ts.num_sensors))
#     storingCounter = 0;
        
#     #ts.sendChar("i")
    

# #%%

# tic = time.time()

# stopCMDsent = False


# while time.time() - tic < runTime:
#     if SensorExist:
#         while ord(ts.ser.read(1))!= ts.STX:
#             continue
                    
#         #Read rest of the data
#         dataBuffer[storingCounter,:] = ts.readRestData()
        
        
#         storingCounter += 1
        
#         #if the data overflows
#         if storingCounter > dataBuffer.shape[0]-1:
#             dataBuffer = np.append(dataBuffer, np.zeros((init_BufferSize,ts.num_sensors)), axis=0)
            

# if SensorExist:    
#     dataBuffer = dataBuffer[0:storingCounter-1,:]
#     ts.sendChar("i")

# #r.stopMotion()


# if SensorExist:
#     ts.closePort()
    
#     #r.demo()
    
#     if plotShow:
#         #%% Plot the data
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.plot(dataBuffer)
#         plt.ylabel('Digital Count')
#         plt.xlabel('sample')
#         plt.grid()
#         plt.show()
    
#     #%% Save Output
#     import datetime
#     currDateString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
#     output_file = ResultSavingDirectory + '/'+ 'result_' +currDateString + SavingFileName + '.csv'
    
#     np.savetxt(output_file, dataBuffer, delimiter=",")

