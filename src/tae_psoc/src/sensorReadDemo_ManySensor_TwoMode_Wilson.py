import time
import psocScanner as psoc
import struct
import numpy as np
import os
import math
import scipy.fftpack

runTime = 10



SensorExist = 1
plotShow = 1

ResultSavingDirectory = '/home/bdml/Tae_Data'
SavingFileName = 'debug'

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
tempPeriodInput = divmod(math.floor(SamplingPeriod * KitprogTimerClockFreq - 1) , 2**8)
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
    initialSamplingNum = 16 # it should be an even number
    
    #%% Buffer
    init_BufferSize = 5000;
    sensor_1_data_history_first = np.zeros((init_BufferSize,num_sensors_1))
    sensor_1_data_history_second = np.zeros((init_BufferSize,num_sensors_2))
    sensor_1_offset_first = np.zeros((1, num_sensors_1))
    sensor_1_offset_second = np.zeros((1, num_sensors_2))
    read_count_sns_1_first = 0
    read_count_sns_1_second = 0
    sensor_1_offsetObtained = False

    sensor_2_data_history_first = np.zeros((init_BufferSize,num_sensors_1))
    sensor_2_data_history_second = np.zeros((init_BufferSize,num_sensors_2))
    sensor_2_offset_first = np.zeros((1, num_sensors_1))
    sensor_2_offset_second = np.zeros((1, num_sensors_2))
    sensor_2_offsetObtained = False
    read_count_sns_2_first = 0
    read_count_sns_2_second = 0

    
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

        
    fftStorage_sns_1_NS = np.zeros((init_BufferSize, winSizeN//2))
    fftStorage_sns_1_WE = np.zeros((init_BufferSize, winSizeN//2))

    fftStorage_sns_2_NS = np.zeros((init_BufferSize, winSizeN//2))
    fftStorage_sns_2_WE = np.zeros((init_BufferSize, winSizeN//2))

    



    #ts.sendChar("i")
    

#%%

tic = time.time()

stopCMDsent = False


snsIndex = 0

while time.time() - tic < runTime:
    if SensorExist:
        while ord(ts.ser.read(1))!= ts.STX:
            continue
                    
        #Read rest of the data
        tempSampledData = ts.readRestData()       

        if snsIndex == 0:
            if IsTwoModeMerged:
                # First Mode
                sensor_1_data_history_first[read_count_sns_1_first,:] = tempSampledData[0,sensorIndexInData_1] - sensor_1_offset_first
                read_count_sns_1_first += 1

                

                # Second Mode
                groupIndex = ts.groupIndex
                sensor_1_data_history_second[read_count_sns_1_second, groupIndex*num_sensors_1:(groupIndex+1)*num_sensors_1]= tempSampledData[0,sensorIndexInData_2]
                if groupIndex == groupIndexMax:
                    sensor_1_data_history_second[read_count_sns_1_second,:] = sensor_1_data_history_second[read_count_sns_1_second,:] - sensor_1_offset_second
                    read_count_sns_1_second = read_count_sns_1_second+1;

                # Obtain Offset from initial few samples    
                if not sensor_1_offsetObtained and  read_count_sns_1_second == initialSamplingNum:
                    sensor_1_offset_first = np.mean(sensor_1_data_history_first[3:initialSamplingNum,:],axis=0)
                    sensor_1_offset_second = np.mean(sensor_1_data_history_second[3:initialSamplingNum,:],axis=0)
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
            if read_count_sns_1_second > sensor_1_data_history_second.shape[0]-1:
                sensor_1_data_history_second = np.append(sensor_1_data_history_second, np.zeros((init_BufferSize,num_sensors_2)), axis=0)
            
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
                fftInput = np.multiply((Differential_N_S - np.mean(Differential_N_S)), hammingW)
                yfft = scipy.fftpack.fft(fftInput)
                xfft = np.linspace(0.0, Fs/2, winSizeN//2)

                fftStorage_sns_1_NS[fftStoreCounter_sns_1,:] = np.abs(yfft[:winSizeN//2])

                #FFT for W-E
                fftInput = np.multiply((Differential_W_E - np.mean(Differential_W_E)), hammingW)
                yfft = scipy.fftpack.fft(fftInput)
                xfft = np.linspace(0.0, Fs/2, winSizeN//2)

                fftStorage_sns_1_WE[fftStoreCounter_sns_1,:] = np.abs(yfft[:winSizeN//2])


                # print(fftStorage_sns_1_WE[fftStoreCounter_sns_1,0:4])

                waitUntil_sns_1_first += overlapN
                fftStoreCounter_sns_1 += 1

                if fftStoreCounter_sns_1 > fftStorage_sns_1_NS.shape[0]-1:
                    fftStorage_sns_1_NS = np.append(fftStorage_sns_1_NS, np.zeros((init_BufferSize,winSizeN//2)), axis=0)
                    fftStorage_sns_1_WE = np.append(fftStorage_sns_1_WE, np.zeros((init_BufferSize,winSizeN//2)), axis=0)
                



        
        elif snsIndex == 1:
            if IsTwoModeMerged:
                sensor_2_data_history_first[read_count_sns_2_first,:] = tempSampledData[0,sensorIndexInData_1]- sensor_2_offset_first
                read_count_sns_2_first += 1

                groupIndex = ts.groupIndex
                sensor_2_data_history_second[read_count_sns_2_second, groupIndex*num_sensors_1:(groupIndex+1)*num_sensors_1]= tempSampledData[0,sensorIndexInData_2]
                if groupIndex == groupIndexMax:
                    sensor_2_data_history_second[read_count_sns_2_second,:] = sensor_2_data_history_second[read_count_sns_2_second,:] - sensor_2_offset_second
                    read_count_sns_2_second = read_count_sns_2_second+1;

                #Obtain offsets
                if not sensor_2_offsetObtained and  read_count_sns_2_second == initialSamplingNum:
                    sensor_2_offset_first = np.mean(sensor_2_data_history_first[3:initialSamplingNum,:],axis=0)                    
                    sensor_2_offset_second = np.mean(sensor_2_data_history_second[3:initialSamplingNum,:],axis=0)                    
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
            
            snsIndex = 0


            ######## Runs for 4 Ch case only

            # Calculate FFT    
            if read_count_sns_2_first == waitUntil_sns_2_first:
                #process window data. filter out dc and multiply a hamming window
                windowBuffer = sensor_2_data_history_first[waitUntil_sns_2_first - winSizeN  : waitUntil_sns_2_first, :]
                Differential_N_S = windowBuffer[:,1] - windowBuffer[:,3]
                Differential_W_E = windowBuffer[:,0] - windowBuffer[:,2]

                # FFT for N-S
                fftInput = np.multiply((Differential_N_S - np.mean(Differential_N_S)), hammingW)
                yfft = scipy.fftpack.fft(fftInput)
                xfft = np.linspace(0.0, Fs/2, winSizeN//2)

                fftStorage_sns_2_NS[fftStoreCounter_sns_2,:] = np.abs(yfft[:winSizeN//2])

                #FFT for W-E
                fftInput = np.multiply((Differential_W_E - np.mean(Differential_W_E)), hammingW)
                yfft = scipy.fftpack.fft(fftInput)
                xfft = np.linspace(0.0, Fs/2, winSizeN//2)

                fftStorage_sns_2_WE[fftStoreCounter_sns_2,:] = np.abs(yfft[:winSizeN//2])


                print(fftStorage_sns_2_WE[fftStoreCounter_sns_2,0:4])

                waitUntil_sns_2_first += overlapN
                fftStoreCounter_sns_2 += 1

                if fftStoreCounter_sns_2 > fftStorage_sns_2_NS.shape[0]-1:
                    fftStorage_sns_2_NS = np.append(fftStorage_sns_2_NS, np.zeros((init_BufferSize,winSizeN//2)), axis=0)
                    fftStorage_sns_2_WE = np.append(fftStorage_sns_2_WE, np.zeros((init_BufferSize,winSizeN//2)), axis=0)
            




if SensorExist:    
    ts.sendChar("i")

    sensor_1_data_history_first = sensor_1_data_history_first[0:read_count_sns_1_first-1,:]
    fftStorage_sns_1_NS = fftStorage_sns_1_NS[0:fftStoreCounter_sns_1,:]
    fftStorage_sns_1_WE = fftStorage_sns_1_WE[0:fftStoreCounter_sns_1,:]

    if IsTwoModeMerged:
        sensor_1_data_history_second = sensor_1_data_history_second[0:read_count_sns_1_second-1,:]

    if SensorNum == 2:
        sensor_2_data_history_first = sensor_2_data_history_first[0:read_count_sns_2_first-1,:]
        fftStorage_sns_2_NS = fftStorage_sns_2_NS[0:fftStoreCounter_sns_2,:]
        fftStorage_sns_2_WE = fftStorage_sns_2_WE[0:fftStoreCounter_sns_2,:]

        if IsTwoModeMerged:
            sensor_2_data_history_second = sensor_2_data_history_second[0:read_count_sns_2_second-1,:]

    

#r.stopMotion()


if SensorExist:
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

        plt.figure()
        plt.plot(fftStorage_sns_1_NS[:,0:5])
        plt.ylabel('| FFT |')
        plt.xlabel('sample')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(fftStorage_sns_2_NS[:,0:5])
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
    
    # np.savetxt(output_file, sensor_1_data_history_first, delimiter=",")

