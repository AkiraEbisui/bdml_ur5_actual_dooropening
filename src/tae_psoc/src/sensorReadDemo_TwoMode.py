import time
import psocScanner as psoc
import struct
import numpy as np
import os
import math

runTime = 5



SensorExist = 1
plotShow = 1

ResultSavingDirectory = '/home/bdml/Tae_Data'
SavingFileName = 'debug'

##################################################
#%%
SensorNum = 1
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


sensingMode = MODE_FOUR_AND_INDIV

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
elif sensingMode == MODE_FOUR_AND_INDIV:
    SamplingFreq = 300
    num_sensors_2 = 36
    IsTwoModeMerged = True

#%% 
SamplingPeriod = 1.0/SamplingFreq
KitprogTimerClockFreq = 100e3
tempPeriodInput = divmod(math.floor(SamplingPeriod * KitprogTimerClockFreq) , 2**8)
periodInput = np.array([tempPeriodInput[1], tempPeriodInput[0]])

#%%

if SensorExist:
    import datetime
    currDateString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")
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
    sensor_1_data_history = np.zeros((init_BufferSize,num_sensors_1))
    sensor_1_data_history_second = np.zeros((init_BufferSize,num_sensors_2))
    read_count_first = 0
    read_count_second = 0
        
    #ts.sendChar("i")
    

#%%

tic = time.time()

stopCMDsent = False


while time.time() - tic < runTime:
    if SensorExist:
        while ord(ts.ser.read(1))!= ts.STX:
            continue
                    
        #Read rest of the data
        tempSampledData = ts.readRestData()       

        if IsTwoModeMerged:
            sensor_1_data_history[read_count_first,:] = tempSampledData[0,sensorIndexInData_1]
            read_count_first += 1

            groupIndex = ts.groupIndex
            sensor_1_data_history_second[read_count_second, groupIndex*num_sensors_1:(groupIndex+1)*num_sensors_1]= tempSampledData[0,sensorIndexInData_2]
            if groupIndex == groupIndexMax:
                read_count_second = read_count_second+1;
        else:
            sensor_1_data_history[read_count_first,:] = tempSampledData
            read_count_first += 1

        
        #if the data overflows
        if read_count_first > sensor_1_data_history.shape[0]-1:
            sensor_1_data_history = np.append(sensor_1_data_history, np.zeros((init_BufferSize,num_sensors_1)), axis=0)
        if read_count_second > sensor_1_data_history_second.shape[0]-1:
            sensor_1_data_history_second = np.append(sensor_1_data_history_second, np.zeros((init_BufferSize,num_sensors_2)), axis=0)
            

if SensorExist:    
    ts.sendChar("i")

    sensor_1_data_history = sensor_1_data_history[0:read_count_first-1,:]

    if IsTwoModeMerged:
        sensor_1_data_history_second = sensor_1_data_history_second[0:read_count_second-1,:]

    

#r.stopMotion()


if SensorExist:
    ts.closePort()
    
    #r.demo()
    
    if plotShow:
        #%% Plot the data
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(sensor_1_data_history)
        plt.ylabel('Digital Count')
        plt.xlabel('sample')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(sensor_1_data_history_second)
        plt.ylabel('Digital Count')
        plt.xlabel('sample')
        plt.grid()
        plt.show()
    
    # #%% Save Output
    # import datetime
    # currDateOnlyString = datetime.datetime.now().strftime("%y%m%d")
    # directory = ResultSavingDirectory +'/' + currDateOnlyString

    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # currDateTimeString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")


    # output_file = directory + '/'+ 'result_' +currDateTimeString + SavingFileName + '.csv'
    
    # np.savetxt(output_file, sensor_1_data_history, delimiter=",")

