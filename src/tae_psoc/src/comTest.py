#!/usr/bin/env python

import rospy
import psocScanner as psoc
from tae_psoc.msg import SensorPacket
from tae_psoc.msg import Sensor_Fast

UpdateSync = False

callCounter = 0

def callback(data):
    global UpdateSync
    global callCounter
    
    # New data came
    UpdateSync = True
    callCounter += 1
    if callCounter >10:
        callCounter = 0
        print("hi")



def mainLoop():
    global UpdateSync

    rospy.init_node('comTest')

    #Each Sensor Reading is Published to topic 'SensorReading'
    #pub = rospy.Publisher('SensorPacket', SensorPacket, queue_size=1)
    rospy.Subscriber('Sensor_Fast',Sensor_Fast, callback)



#    targ = psoc.TactileSensor(port="/dev/ttyACM1") # new Psoc for the Sync
#    print("ComPort connected")
#    targ.ser.flushInput()

    while not rospy.is_shutdown():
        continue
#        if UpdateSync == True:
#            UpdateSync = False
#            targ.sendChar("z")
#            #print("z")


#####################################

#def callback(data):
#    global UpdateSync

#    # New data came
#    UpdateSync = True
#def mainLoop():
#    global UpdateSync
    
#    rospy.init_node('comTest')

#    #Each Sensor Reading is Published to topic 'SensorReading'
#    #pub = rospy.Publisher('SensorPacket', SensorPacket, queue_size=1)
#    rospy.Subscriber('SensorPacket',SensorPacket, callback)
    
    
    
#    targ = psoc.TactileSensor(port="/dev/ttyACM1") # new Psoc for the Sync
#    print("ComPort connected")
#    targ.ser.flushInput()
    
#    while not rospy.is_shutdown():
#        if UpdateSync == True:
#            UpdateSync = False
#            targ.sendChar("z")
#            #print("z")
    
    
                
if __name__ == '__main__':
    try:
        print("Started!")
        mainLoop()
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

