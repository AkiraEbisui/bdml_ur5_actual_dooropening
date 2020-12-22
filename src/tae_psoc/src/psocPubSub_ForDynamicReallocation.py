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



from tae_psoc.msg import SensorPacket
from tae_psoc.msg import cmdToPsoc
from tae_customrobotiqgripper.msg import cmdToGripper


IDLE = 0
STREAMING = 1


NO_CMD = 0
START_CMD = 2
IDLE_CMD = 3



currState = IDLE
CMD_in = NO_CMD

gripperPosInput = -1

UR5_Y_pos = 0 # in mm


def callback(data):
    global CMD_in
    global UR5_Y_pos

    if data.cmdInput < 5:
        CMD_in = data.cmdInput
    else:
        UR5_Y_pos = data.cmdInput


def callback_gripperCMD(data):
    global gripperPosInput
    gripperPosInput = data.position
    print(gripperPosInput)
    

def mainLoop(savingFileName):
    global currState
    global CMD_in   
    global gripperPosInput
    global UR5_Y_pos

    # #Gripper is a C-Model with a TCP connection
    # gripper = robotiq_c_model_control.baseCModel.robotiqBaseCModel()
    # gripper.client = robotiq_modbus_rtu.comModbusRtu.communication()

    # #We connect to the address received as an argument
    # gripper.client.connectToDevice(device)

    rospy.init_node('psocPubSub')

    #Each Sensor Reading is Published to topic 'SensorReading'
    pub = rospy.Publisher('SensorPacket', SensorPacket, queue_size=1)
    rospy.Subscriber('cmdToPsoc',cmdToPsoc, callback)
    rospy.Subscriber('cmdToGripper',cmdToGripper,callback_gripperCMD)

    msg = SensorPacket()

    # #The Gripper command is received from the topic named 'CModelRobotOutput'
    # rospy.Subscriber('CModelRobotOutput', outputMsg.CModel_robot_output, gripper.refreshCommand)    
    
    counter = 0

    # rospy.spin();

    ResultSavingDirectory = '/home/tae/Data'
    SavingFileName = savingFileName  #'test_box'

    SensorExist = 1
    plotShow = 1


    ##################################################
    #%%
    SensorNum = 2
    SensorAddress = np.array([8, 9])


    MODE_ONE_PAD = 0 
    MODE_FOUR_PAD = 1
    MODE_NINE_PAD = 2
    MODE_INDIVIDUAL = 3
    MODE_CAPSENSE = 17
    MODE_ONE_AND_FOUR =32   #Problematic  at 1kHz
    MODE_NINE_AND_INDIV= 33 
    MODE_ONE_AND_NINE = 34  #Problematic
    MODE_FOUR_AND_INDIV = 35; # 250
    MODE_TORSION_AND_INDIV = 36; #0X24




    #################################3# Sensing MODE Select@!!!1 @#####################################
    # sensingMode = MODE_FOUR_AND_INDIV
    sensingMode = MODE_TORSION_AND_INDIV


    slopeCompensateOn = True




    IsTwoModeMerged = False
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
        SamplingFreq = 300
        num_sensors_2 = 36
        IsTwoModeMerged = True

    #%% 
    SamplingPeriod = 1.0/SamplingFreq
    KitprogTimerClockFreq = 100e3
    tempPeriodInput = divmod(math.floor(SamplingPeriod * KitprogTimerClockFreq) , 2**8)
    periodInput = np.array([tempPeriodInput[1], tempPeriodInput[0]])


    ## Read Calibration File
    # Open calibration file
    # !!!!!!!!!!!!!!!!!!!!!!
    MatFile = loadmat('/home/tae/Data/CalibrationMat/SensorA_CalMatrix_2_0_AllArea')
    # CalMat_A = MatFile['A_trainsets']
    CalMat_A2 = MatFile['A_trainsets2']

    MatFile = loadmat('/home/tae/Data/CalibrationMat/SensorB_CalMatrix_2_0_AllArea')
    CalMat_B2 = MatFile['A_trainsets2']
    

    #We loop
    while not rospy.is_shutdown():
        if currState == IDLE and CMD_in == START_CMD:
            CMD_in = NO_CMD
            currState = STREAMING

           
            
            currDateTimeString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
            
        #    output_file = ResultSavingDirectory + '\\'+ 'result_' +currDateString + SavingFileName + '.html'

            ts = psoc.TactileSensor(port="/dev/ttyACM0")
            ts.ser.flushInput()
            
            # ts.sendChar("i")
            # time.sleep(0.01)

            thisInputArray = np.array([ord('a'), SensorNum, SensorAddress[0], SensorAddress[1]])    
            ts.sendNum(thisInputArray)    
            time.sleep(0.1)
            
        #    fwrite(com,['p' periodInput(1) periodInput(2)])
            thisInputArray = np.array([ord('p'), periodInput[0], periodInput[1]])
            ts.sendNum(thisInputArray)  
            # ts.sendChar("p")    
            # ts.sendNum(periodInput[0])
            # ts.sendNum(periodInput[1])
            time.sleep(0.1)
            
            thisInputArray = np.array([ord('m'), sensingMode])    
            ts.sendNum(thisInputArray)
            time.sleep(0.1)
            
            ts.sendChar("q")
            time.sleep(0.1)
            

            ts.packet_size = ord(ts.ser.read(1))-1

            num_sensors_1= (ts.packet_size - 1) / 2    

            ts.num_sensors= (ts.packet_size - 1) / 2
            ts.unpackFormat = '<'
            for i in range(0,ts.packet_size):
                ts.unpackFormat = ts.unpackFormat + 'B'
                
            
            if IsTwoModeMerged: #% Deal with Merging Techniq
                num_sensors_1 = num_sensors_1/2
                
                sensorIndexInData_1 = list(range(0,(ts.packet_size - 1) / 2, 2) ) 
                sensorIndexInData_2 = list(range(1,(ts.packet_size - 1) / 2, 2) ) 
                
                groupIndexMax = num_sensors_2/ num_sensors_1 -1; #% Follows C convention
                
            else:
                num_sensors_2 = 0   # % Dummy
            
            print("num_sensor 1=")
            print(num_sensors_1)
            print("num_sensor 2=")
            print(num_sensors_2)
            
            
        #######################################################

           
            #%% Get Initial samples for measuring Offset Values    
            initialSamplingNum = 16 # it should be an even number
            
            #%% Buffer
            init_BufferSize = 5000;
            sensor_1_data_history_first = np.zeros((init_BufferSize,num_sensors_1))
            sensor_1_data_history_second = np.zeros((init_BufferSize,num_sensors_2))
            sensor_1_offset_first = np.zeros((1, num_sensors_1), dtype = 'i')
            sensor_1_offset_second = np.zeros((1, num_sensors_2), dtype = 'i')
            sensor_1_FT_history_second = np.zeros((init_BufferSize, 6), dtype='f')
            sensor_1_vorticity_history_second = np.zeros((init_BufferSize, 1), dtype='f')
            read_count_sns_1_first = 0
            read_count_sns_1_second = 0
            sensor_1_offsetObtained = False

            sensor_2_data_history_first = np.zeros((init_BufferSize,num_sensors_1))
            sensor_2_data_history_second = np.zeros((init_BufferSize,num_sensors_2))
            sensor_2_offset_first = np.zeros((1, num_sensors_1),dtype = 'i')
            sensor_2_offset_second = np.zeros((1, num_sensors_2),dtype = 'i')
            sensor_2_FT_history_second = np.zeros((init_BufferSize, 6), dtype='f')
            sensor_2_vorticity_history_second = np.zeros((init_BufferSize, 1), dtype='f')
            sensor_2_offsetObtained = False
            read_count_sns_2_first = 0
            read_count_sns_2_second = 0

            FT_movingAverageWindow_N = 5


            GripperPosCMD_History = np.zeros((init_BufferSize,1))-1 # Always -1 if no command in.




            
            # Some initial settings for the FFT.. preliminary here, will move up later
            winSizeT = 500 * 0.001 # in sec
            overlapT = 100e-3

            Fs = SamplingFreq
            T = 1.0 / Fs
            # Window is set to be 100ms
            winSizeN = int(winSizeT / T)
            hammingW = np.hamming(winSizeN)

            overlapN = int(overlapT / T) #sample Count
            isFirst = 1
            
            # Buffer stuff for the window and fft
            waitUntil_sns_1_first = 0
            waitUntil_sns_2_first = 0
            
            fftStoreCounter_sns_1 = 0;
            fftStoreCounter_sns_2 = 0;

                
            fftStorage_sns_1_NS = np.zeros((init_BufferSize, winSizeN//2+1))
            fftStorage_sns_1_WE = np.zeros((init_BufferSize, winSizeN//2+1))

            fftStorage_sns_2_NS = np.zeros((init_BufferSize, winSizeN//2+1))
            fftStorage_sns_2_WE = np.zeros((init_BufferSize, winSizeN//2+1))

            # For Vorticity Calculate
            taxelNum_x = 3
            taxelNum_y = 3
            u = np.zeros([3,3],dtype='f')
            v = np.zeros([3,3],dtype='f')
            duMask = np.array([[-1, -1, -1],[0, 0, 0],[ 1, 1, 1]], dtype = 'f')
            dvMask = np.array([[-1, 0, 1],[-1, 0, 1],[ -1, 0, 1]], dtype = 'f')

                

            #%%

            tic = time.time()

            stopCMDsent = False


            snsIndex = 0   




            #Start Streaming
            ts.sendChar("s")

            currDateOnlyString = datetime.datetime.now().strftime("%y%m%d")

            
            
            # Loop for getting sensor readings

            while not CMD_in == IDLE_CMD:

                while ord(ts.ser.read(1))!= ts.STX:
                    continue

                 
                            
                #Read rest of the data
                tempSampledData = ts.readRestData()       

                if snsIndex == 0:
                    if IsTwoModeMerged:
                        # First Mode
                        sensor_1_data_history_first[read_count_sns_1_first,:] = tempSampledData[0,sensorIndexInData_1] - sensor_1_offset_first
                        #insert gripper CMD history if it is not -1
                        if gripperPosInput > -1:
                            GripperPosCMD_History[read_count_sns_1_first,0] = gripperPosInput
                            gripperPosInput = -1 #restore to default num

                        read_count_sns_1_first += 1

                        

                        # Second Mode
                        groupIndex = ts.groupIndex
                        sensor_1_data_history_second[read_count_sns_1_second, groupIndex*num_sensors_1:(groupIndex+1)*num_sensors_1]= tempSampledData[0,sensorIndexInData_2]
                        if groupIndex == groupIndexMax:
                            sensor_1_data_history_second[read_count_sns_1_second,:] = sensor_1_data_history_second[read_count_sns_1_second,:] - sensor_1_offset_second
                            read_count_sns_1_second = read_count_sns_1_second+1;
                            
                            # Calibrate Using Second Mode Data
                            if sensor_1_offsetObtained:                                
                                #Get moving Average of moving Average window
                                averagedMean = np.mean(sensor_1_data_history_second[read_count_sns_1_second-FT_movingAverageWindow_N:read_count_sns_1_second,:],axis=0)

                                # Multiply the calibration matrix accordingly
                                toCalibrate = np.append(averagedMean, np.square(averagedMean))

                                calData2 = np.matmul(CalMat_A2, toCalibrate)
                                # Save the relevant force torque to FT_data_history 
                                sensor_1_FT_history_second[read_count_sns_1_second-1,:] = calData2



                                ### Calculate Vorticity
                                
                                for j in range(0,taxelNum_y):
                                    for i in range(0,taxelNum_x):                                        
                                        west = averagedMean[0 + (j*taxelNum_x + i)*4];
                                        north = averagedMean[1 + (j*taxelNum_x + i)*4];
                                        east = averagedMean[2 + (j*taxelNum_x + i)*4];
                                        south = averagedMean[3 + (j*taxelNum_x + i)*4];
                                        
                                        u[j,i] = east - west;
                                        v[j,i] = north - south;
                                        
                                    

                                # Custom Curlz Calculate
                                du = u - u[1,1]
                                dv = v - v[1,1]
                                
                                curl_center = np.sum( np.multiply(dv, dvMask) - np.multiply(du,duMask) ) / 6.0; # Per each gradient, we see 6 element contributing
                                # curl_center = sum(sum( dv.*dvMask - du.*duMask))/6; # Per each gradient, we see 6 element contributing
                                

                                sensor_1_vorticity_history_second[read_count_sns_1_second-1,:] = curl_center





                        # Obtain Offset from initial few samples    
                        if not sensor_1_offsetObtained and  read_count_sns_1_second == initialSamplingNum:
                            sensor_1_offset_first = np.mean(sensor_1_data_history_first[read_count_sns_1_first-initialSamplingNum:read_count_sns_1_first,:],axis=0)
                            sensor_1_offset_second = np.mean(sensor_1_data_history_second[5:initialSamplingNum,:],axis=0)
                            sensor_1_offset_first = sensor_1_offset_first.astype(int)
                            sensor_1_offset_second = sensor_1_offset_second.astype(int)
                            sensor_1_offsetObtained = True
                            waitUntil_sns_1_first = winSizeN + read_count_sns_1_first

                    else:
                        sensor_1_data_history_first[read_count_sns_1_first,:] = tempSampledData - sensor_1_offset_first
                        read_count_sns_1_first += 1
                        if not sensor_1_offsetObtained and  read_count_sns_1_first == initialSamplingNum:
                            sensor_1_offset_first = np.mean(sensor_1_data_history_first[3:initialSamplingNum,:],axis=0)                    
                            sensor_1_offsetObtained = True
                            waitUntil_sns_1_first = winSizeN + read_count_sns_1_first

                    
                    #if the data overflows
                    if read_count_sns_1_first > sensor_1_data_history_first.shape[0]-1:
                        sensor_1_data_history_first = np.append(sensor_1_data_history_first, np.zeros((init_BufferSize,num_sensors_1)), axis=0)
                        GripperPosCMD_History = np.append(GripperPosCMD_History, -np.ones((init_BufferSize,1)), axis=0)

                    if read_count_sns_1_second > sensor_1_data_history_second.shape[0]-1:
                        sensor_1_data_history_second = np.append(sensor_1_data_history_second, np.zeros((init_BufferSize,num_sensors_2)), axis=0)
                        sensor_1_FT_history_second = np.append(sensor_1_FT_history_second, np.zeros((init_BufferSize,6)), axis=0)
                        sensor_1_vorticity_history_second = np.append(sensor_1_vorticity_history_second, np.zeros((init_BufferSize,1)), axis=0 )
                    
                    if SensorNum == 2:
                        snsIndex = 1

                    ######## Runs for 4 Ch case only

                    # Calculate FFT    
                    if read_count_sns_1_first == waitUntil_sns_1_first:
                        #process window data. filter out dc and multiply a hamming window
                        windowBuffer = sensor_1_data_history_first[waitUntil_sns_1_first - winSizeN  : waitUntil_sns_1_first, :]
                        Differential_N_S = windowBuffer[:,1] - windowBuffer[:,3]
                        Differential_W_E = windowBuffer[:,0] - windowBuffer[:,2]

                        # FFT for N-S
                        if slopeCompensateOn:
                            temp = np.transpose(Differential_N_S)
                            # temp = temp[0,:]
                            x_fit = np.array(range(0,np.shape(temp)[0]))
                            
                            fitObject = np.poly1d(np.polyfit(x_fit, temp, 1))

                            fftInput = np.multiply((Differential_N_S - fitObject(x_fit)), hammingW)

                        else:
                            fftInput = np.multiply((Differential_N_S - np.mean(Differential_N_S)), hammingW)




                        yfft = scipy.fftpack.fft(fftInput)
                        xfft = np.linspace(0.0, Fs/2, winSizeN//2+1)

                        fftStorage_sns_1_NS[fftStoreCounter_sns_1,:] = np.abs(yfft[:winSizeN//2+1])

                        #FFT for W-E
                        if slopeCompensateOn:
                            temp = np.transpose(Differential_W_E)
                            # temp = temp[0,:]
                            x_fit = np.array(range(0,np.shape(temp)[0]))
                            
                            fitObject = np.poly1d(np.polyfit(x_fit, temp, 1))

                            fftInput = np.multiply((Differential_W_E - fitObject(x_fit)), hammingW)
                            
                        else:
                            fftInput = np.multiply((Differential_W_E - np.mean(Differential_W_E)), hammingW)
                        #fftInput = np.multiply((Differential_W_E - np.mean(Differential_W_E)), hammingW)



                        yfft = scipy.fftpack.fft(fftInput)
                        xfft = np.linspace(0.0, Fs/2, winSizeN//2+1)

                        fftStorage_sns_1_WE[fftStoreCounter_sns_1,:] = np.abs(yfft[:winSizeN//2+1])


                        # print(fftStorage_sns_1_WE[fftStoreCounter_sns_1,0:4])

                        waitUntil_sns_1_first += overlapN
                        fftStoreCounter_sns_1 += 1

                        if fftStoreCounter_sns_1 > fftStorage_sns_1_NS.shape[0]-1:
                            fftStorage_sns_1_NS = np.append(fftStorage_sns_1_NS, np.zeros((init_BufferSize,winSizeN//2+1)), axis=0)
                            fftStorage_sns_1_WE = np.append(fftStorage_sns_1_WE, np.zeros((init_BufferSize,winSizeN//2+1)), axis=0)
                        



                
                elif snsIndex == 1:
                    if IsTwoModeMerged:
                        sensor_2_data_history_first[read_count_sns_2_first,:] = tempSampledData[0,sensorIndexInData_1]- sensor_2_offset_first
                        read_count_sns_2_first += 1

                        groupIndex = ts.groupIndex
                        sensor_2_data_history_second[read_count_sns_2_second, groupIndex*num_sensors_1:(groupIndex+1)*num_sensors_1]= tempSampledData[0,sensorIndexInData_2]
                        if groupIndex == groupIndexMax:
                            sensor_2_data_history_second[read_count_sns_2_second,:] = sensor_2_data_history_second[read_count_sns_2_second,:] - sensor_2_offset_second
                            read_count_sns_2_second = read_count_sns_2_second+1;

                            # Calibrate Using Second Mode Data
                            if sensor_2_offsetObtained:                                
                                #Get moving Average of moving Average window
                                averagedMean = np.mean(sensor_2_data_history_second[read_count_sns_2_second-FT_movingAverageWindow_N:read_count_sns_2_second,:],axis=0)

                                # Multiply the calibration matrix accordingly
                                toCalibrate = np.append(averagedMean, np.square(averagedMean))

                                calData2 = np.matmul(CalMat_B2, toCalibrate)
                                # Save the relevant force torque to FT_data_history 
                                sensor_2_FT_history_second[read_count_sns_2_second-1,:] = calData2


                                ### Calculate Vorticity
                                
                                for j in range(0,taxelNum_y):
                                    for i in range(0,taxelNum_x):                                        
                                        west = averagedMean[0 + (j*taxelNum_x + i)*4];
                                        north = averagedMean[1 + (j*taxelNum_x + i)*4];
                                        east = averagedMean[2 + (j*taxelNum_x + i)*4];
                                        south = averagedMean[3 + (j*taxelNum_x + i)*4];
                                        
                                        u[j,i] = east - west;
                                        v[j,i] = north - south;
                                    
                                # Custom Curlz Calculate
                                du = u - u[1,1]
                                dv = v - v[1,1]
                                
                                curl_center = np.sum( np.multiply(dv, dvMask) - np.multiply(du,duMask) ) / 6.0; # Per each gradient, we see 6 element contributing
                                # curl_center = sum(sum( dv.*dvMask - du.*duMask))/6; # Per each gradient, we see 6 element contributing
                                

                                sensor_2_vorticity_history_second[read_count_sns_2_second-1,:] = curl_center






                        #Obtain offsets
                        if not sensor_2_offsetObtained and  read_count_sns_2_second == initialSamplingNum:
                            sensor_2_offset_first = np.mean(sensor_2_data_history_first[read_count_sns_2_first-initialSamplingNum:read_count_sns_2_first,:],axis=0)                    
                            sensor_2_offset_second = np.mean(sensor_2_data_history_second[3:initialSamplingNum,:],axis=0)    
                            sensor_2_offset_first = sensor_2_offset_first.astype(int)
                            sensor_2_offset_second = sensor_2_offset_second.astype(int)                
                            sensor_2_offsetObtained = True
                            waitUntil_sns_2_first = winSizeN + read_count_sns_2_first
                    else:
                        sensor_2_data_history_first[read_count_sns_2_first,:] = tempSampledData
                        read_count_sns_2_first += 1

                        if not sensor_2_offsetObtained and  read_count_sns_2_first == initialSamplingNum:
                            sensor_2_offset_first = np.mean(sensor_2_data_history_first[3:initialSamplingNum,:],axis=0)                    
                            sensor_2_offsetObtained = True
                            waitUntil_sns_2_first = winSizeN + read_count_sns_2_first

                    
                    #if the data overflows
                    if read_count_sns_2_first > sensor_2_data_history_first.shape[0]-1:
                        sensor_2_data_history_first = np.append(sensor_2_data_history_first, np.zeros((init_BufferSize,num_sensors_1)), axis=0)
                    if read_count_sns_2_second > sensor_2_data_history_second.shape[0]-1:
                        sensor_2_data_history_second = np.append(sensor_2_data_history_second, np.zeros((init_BufferSize,num_sensors_2)), axis=0)
                        sensor_2_FT_history_second = np.append(sensor_2_FT_history_second, np.zeros((init_BufferSize,6)), axis=0)
                        sensor_2_vorticity_history_second = np.append(sensor_2_vorticity_history_second, np.zeros((init_BufferSize,1)), axis=0 )
                    
                    snsIndex = 0


                    ######## Runs for 4 Ch case only

                    # Calculate FFT    
                    if read_count_sns_2_first == waitUntil_sns_2_first:
                        #process window data. filter out dc and multiply a hamming window
                        windowBuffer = sensor_2_data_history_first[waitUntil_sns_2_first - winSizeN  : waitUntil_sns_2_first, :]
                        Differential_N_S = windowBuffer[:,1] - windowBuffer[:,3]
                        Differential_W_E = windowBuffer[:,0] - windowBuffer[:,2]

                        # FFT for N-S
                        if slopeCompensateOn:
                            temp = np.transpose(Differential_N_S)

                            # temp = temp[0,:]
                            x_fit = np.array(range(0,np.shape(temp)[0]))
                            
                            fitObject = np.poly1d(np.polyfit(x_fit, temp, 1))

                            fftInput = np.multiply((Differential_N_S - fitObject(x_fit)), hammingW)


                        else:
                            fftInput = np.multiply((Differential_N_S - np.mean(Differential_N_S)), hammingW)

                        yfft = scipy.fftpack.fft(fftInput)
                        xfft = np.linspace(0.0, Fs/2, winSizeN//2+1)

                        fftStorage_sns_2_NS[fftStoreCounter_sns_2,:] = np.abs(yfft[:winSizeN//2+1])

                        #FFT for W-E
                        if slopeCompensateOn:
                            temp = np.transpose(Differential_W_E)
                            # temp = temp[0,:]
                            x_fit = np.array(range(0,np.shape(temp)[0]))
                            
                            fitObject = np.poly1d(np.polyfit(x_fit, temp, 1))

                            fftInput = np.multiply((Differential_W_E - fitObject(x_fit)), hammingW)
                            
                        else:
                            fftInput = np.multiply((Differential_W_E - np.mean(Differential_W_E)), hammingW)
                        yfft = scipy.fftpack.fft(fftInput)
                        xfft = np.linspace(0.0, Fs/2, winSizeN//2+1)

                        fftStorage_sns_2_WE[fftStoreCounter_sns_2,:] = np.abs(yfft[:winSizeN//2+1])


                        # print(fftStorage_sns_2_WE[fftStoreCounter_sns_2,0:4])

                        waitUntil_sns_2_first += overlapN
                        fftStoreCounter_sns_2 += 1

                        if fftStoreCounter_sns_2 > fftStorage_sns_2_NS.shape[0]-1:
                            fftStorage_sns_2_NS = np.append(fftStorage_sns_2_NS, np.zeros((init_BufferSize,winSizeN//2+1)), axis=0)
                            fftStorage_sns_2_WE = np.append(fftStorage_sns_2_WE, np.zeros((init_BufferSize,winSizeN//2+1)), axis=0)



                    # After Doen with the 2nd sensor reading we Puablish the mesg to Main operating code
                    #Get and publish the Each Sensor Read
                    if sensor_1_offsetObtained and sensor_2_offsetObtained:
                        msg.sns_1_FFT_NS = fftStorage_sns_1_NS[fftStoreCounter_sns_1-1,0:6]
                        msg.sns_1_FFT_WE = fftStorage_sns_1_WE[fftStoreCounter_sns_1-1,0:6]
                        
                        msg.sns_2_FFT_NS = fftStorage_sns_2_NS[fftStoreCounter_sns_2-1,0:6]
                        msg.sns_2_FFT_WE = fftStorage_sns_2_WE[fftStoreCounter_sns_2-1,0:6]
                        
                        msg.sns_1_4Ch = sensor_1_data_history_first[read_count_sns_1_first-1,:]
                        msg.sns_1_F_M = sensor_1_FT_history_second[read_count_sns_1_second-1,:]
                        
                        msg.sns_2_4Ch = sensor_2_data_history_first[read_count_sns_2_first-1,:]
                        msg.sns_2_F_M = sensor_2_FT_history_second[read_count_sns_2_second-1,:]


                        msg.sns1_vorticity = sensor_1_vorticity_history_second[read_count_sns_1_second-1,0]
                        msg.sns2_vorticity = sensor_2_vorticity_history_second[read_count_sns_2_second-1,0]

                        counter = counter + 1
                        pub.publish(msg)   



                        














                    ##################33
        

            
            ts.sendChar("i")

            sensor_1_data_history_first = sensor_1_data_history_first[0:read_count_sns_1_first-1,:]
            GripperPosCMD_History = GripperPosCMD_History[0:read_count_sns_1_first-1,:]


            fftStorage_sns_1_NS = fftStorage_sns_1_NS[0:fftStoreCounter_sns_1,:]
            fftStorage_sns_1_WE = fftStorage_sns_1_WE[0:fftStoreCounter_sns_1,:]

            if IsTwoModeMerged:
                sensor_1_data_history_second = sensor_1_data_history_second[0:read_count_sns_1_second-1,:]
                sensor_1_FT_history_second = sensor_1_FT_history_second[0:read_count_sns_1_second-1,:]
                sensor_1_vorticity_history_second = sensor_1_vorticity_history_second[0:read_count_sns_1_second-1,:]


            if SensorNum == 2:
                sensor_2_data_history_first = sensor_2_data_history_first[0:read_count_sns_2_first-1,:]
                fftStorage_sns_2_NS = fftStorage_sns_2_NS[0:fftStoreCounter_sns_2,:]
                fftStorage_sns_2_WE = fftStorage_sns_2_WE[0:fftStoreCounter_sns_2,:]

                if IsTwoModeMerged:
                    sensor_2_data_history_second = sensor_2_data_history_second[0:read_count_sns_2_second-1,:]
                    sensor_2_FT_history_second = sensor_2_FT_history_second[0:read_count_sns_2_second-1,:]
                    sensor_2_vorticity_history_second = sensor_2_vorticity_history_second[0:read_count_sns_2_second-1,:]




            #r.stopMotion()
            
            ts.closePort()
            
            #r.demo()
            
            if plotShow:
                #%% Plot the data
                import matplotlib.pyplot as plt
                # plt.figure()
                # plt.plot(sensor_2_data_history_first)
                # plt.ylabel('Digital Count')
                # plt.xlabel('sample')
                # plt.grid()
                # plt.show()

                # plt.figure()
                # plt.plot(sensor_2_data_history_second)
                # plt.ylabel('Digital Count')
                # plt.xlabel('sample')
                # plt.grid()
                # plt.show()

                # plt.figure()
                # plt.plot(fftStorage_sns_1_NS[:,0:5])
                # plt.ylabel('| FFT |')
                # plt.xlabel('sample')
                # plt.grid()
                # plt.show()

                # plt.figure()
                # plt.plot(fftStorage_sns_2_NS[:,0:5])
                # plt.ylabel('Digital Count')
                # plt.xlabel('sample')
                # plt.grid()
                # plt.show()

                fig, axs = plt.subplots(4)                
                axs[0].plot(sensor_1_FT_history_second[30:,0:3])
                axs[0].set_title('Sensor A Force')
                axs[1].plot(sensor_1_FT_history_second[30:,3:])
                axs[1].set_title('Sensor A Moment')
                
                axs[2].plot(sensor_2_FT_history_second[30:,0:3])
                axs[2].set_title('Sensor B Force')
                axs[3].plot(sensor_2_FT_history_second[30:,3:])
                axs[3].set_title('Sensor B Moment')

                fig2,axs2 = plt.subplots(4)
                axs2[0].plot(fftStorage_sns_1_NS[:,0:3]) 
                axs2[0].set_ylim([0,2000])               
                axs2[0].set_title('Sensor A N-S FFT')
                axs2[1].plot(fftStorage_sns_2_NS[:,0:3])                
                axs2[1].set_ylim([0,2000])               
                axs2[1].set_title('Sensor B N-S FFT')


                axs2[2].plot(fftStorage_sns_1_WE[:,0:3]) 
                axs2[2].set_ylim([0,2000])            
                axs2[3].plot(fftStorage_sns_2_WE[:,0:3])                
                axs2[3].set_ylim([0,2000])               
                


                if sensingMode == MODE_TORSION_AND_INDIV:
                    axs2[2].set_title('Sensor A Torsion FFT')
                    axs2[3].set_title('Sensor B Torsion FFT')
                else:
                    axs2[2].set_title('Sensor A W-E FFT')
                    axs2[3].set_title('Sensor B W-E FFT')




                plt.show()
                


            #%% Save Output
            
            
            directory = ResultSavingDirectory +'/' + currDateOnlyString
            if not os.path.exists(directory):
            	os.makedirs(directory)

            
            output_file = directory + '/'+ 'result_' +currDateTimeString + SavingFileName + '.mat'
            	
            # # currDateString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
            # output_file = ResultSavingDirectory + '/'+ 'result_' +currDateString + SavingFileName + '.csv'
            

            # Save as .mat file
            savemat(output_file, 
                {'sensor_1_data_history_first':sensor_1_data_history_first,
                 'fftStorage_sns_1_NS':fftStorage_sns_1_NS,
                  'fftStorage_sns_1_WE':fftStorage_sns_1_WE,
                  'sensor_1_data_history_second' : sensor_1_data_history_second,
                  'sensor_1_FT_history_second' : sensor_1_FT_history_second,
                  'sensor_2_data_history_first':sensor_2_data_history_first,
                  'fftStorage_sns_2_NS':fftStorage_sns_2_NS,
                  'fftStorage_sns_2_WE':fftStorage_sns_2_WE,
                  'sensor_2_data_history_second' : sensor_2_data_history_second,
                  'sensor_2_FT_history_second' : sensor_2_FT_history_second,
                  'GripperPosCMD_History' : GripperPosCMD_History,
                  'sensingMode' : sensingMode,
                  'sensor_1_vorticity_history_second' : sensor_1_vorticity_history_second,
                  'sensor_2_vorticity_history_second' : sensor_2_vorticity_history_second,
                  'UR5_Y_pos' : UR5_Y_pos})

            # np.savetxt(output_file, sensor_1_data_history_second, delimiter=",")
            print("file Saved")

            CMD_in = NO_CMD
            currState = IDLE
            UR5_Y_pos = 0
            gripperPosInput = -1







                
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

