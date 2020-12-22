#!/usr/bin/env python

import serial
import serial.tools.list_ports
import struct
import numpy as np
from time import sleep, time

#from tactile_calibration_new import LoadCalibration, CalculateForce

''' searches for sensors as they show up in different systems '''
def findSensors():
    port_list = []
    search = ""

    # change the search term depending on OS:
    import platform
    os_name = platform.system()

    if os_name == "Darwin":
        search = "serial"
    elif os_name == "Windows":
        search = "COM"
    elif os_name == "Linux":
        search = "USB"
    else:
        raise Exception("Unsupported OS: " + os_name)

    for p in serial.tools.list_ports.grep(search):
        # print p[0] + " is available"
        port_list.append(p[0])

    if len(port_list) < 1:
        raise Exception("No sensors found!")

    return port_list

class TactileSensor():

    STX = 0x02
    ETX = 0x03

    # Commands to the sensors (simple, 1-byte transmissions)
    STREAM = 0x80       # Start streaming data from sensors
    SAMPLE = 0x81       # Send a single data sample
    IDLE = 0x82         # Idle mode
    STATUS_REQ = 0x83   # Report current status

    STATUS_TYPE = 0x17 # not sure why it's not 0x11
    # Data types
    DATA_TYPE = 0x10

    # Statuses
    INITIALIZING = 0x00
    IDLING = 0x01
    STREAMING = 0x02
    ERROR = 0x03
    
    packet_size = 0 # Length after STX
    num_sensors = 0 # Number of sensor = (packet_size - 1) / 2
    unpackFormat = ''

    groupIndex = 0 # For Two mode scanning

    ''' Constructor loads calibration and opens serial communication '''
    def __init__(self, port):        
        self.port = port
        self.baud = 115200
        self.status = ""
#        self.runtime = runtime
        #self.data_init = [0.0] * 6
        self.data_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        # set this to True to stop reading at any time
        self.stop = False

        try:
            self.ser = serial.Serial(port, baudrate=self.baud, timeout=1, writeTimeout=1)            
            print("Connected to Comport")

        except serial.SerialException:
            raise Exception("Unable to open serial port. Make sure the device is connected to {port}".format(port=port))
    
    def sendChar(self, cmd):
        txMsg = struct.pack('<B', ord(cmd))
        self.ser.write(txMsg)    
        
    def sendNum(self, inputNumpyArray):
        inputNumpyArray=np.uint8(inputNumpyArray)
        if inputNumpyArray.size<1:
            sys.exit("Wrong Input to SendNum")
        else:
            txMsg = struct.pack('<B', inputNumpyArray[0])
            for i in range(1,inputNumpyArray.size):
                txMsg += struct.pack('<B', inputNumpyArray[i])
            self.ser.write(txMsg)





        
        
        
    def readRestData(self):
        thisPacket = struct.unpack(self.unpackFormat, self.ser.read(self.packet_size) )
        if (thisPacket[self.packet_size-1] % 8) == self.ETX:
            
            self.groupIndex = (thisPacket[self.packet_size-1] - self.ETX) / 8

            processedPacket = np.zeros((1,self.num_sensors))            
                        
            #Pull the sensor data from the data stream
            for i in range(0,self.num_sensors):
                processedPacket[0,i]=thisPacket[2*i]+256*thisPacket[2*i+1]                
                      
            return processedPacket
            
        else:
            return 0
        
        
        
        
        
            
        ''' Send commands to the sensor '''
    def _commandIdle(self):
        txMsg = struct.pack('<BBB', self.STX, self.IDLE, self.ETX)
        self.ser.write(txMsg)

    def _commandStream(self):
        txMsg = struct.pack('<BBB', self.STX, self.STREAM, self.ETX)
        self.ser.write(txMsg)

    def _commandSample(self):
        txMsg = struct.pack('<BBB', self.STX, self.SAMPLE, self.ETX)
        self.ser.write(txMsg)

    def _commandStatusRequest(self):
        txMsg = struct.pack('<BBB', self.STX, self.STATUS_REQ, self.ETX)
        self.ser.write(txMsg)

    ''' Read from serial and parse status response '''
    def _readStatusResp(self):

        resp = self.ser.read(5)
        stx, rx_len, rx_type, rx_status, etx = struct.unpack('<BBBBB', resp)

        if (stx != self.STX) or (rx_len != 2) or (etx != self.ETX):
            self.status = "Bad status packet stx: " + str(stx) + " rx_len: " + str(rx_len) + " etx: " + str(etx)
            raise Warning(self.status)
            return

        if rx_status == self.INITIALIZING:
            self.status = 'INITIALIZING'
        elif rx_status == self.IDLING:
            self.status = 'IDLING'
        elif rx_status == self.STREAMING:
            self.status = 'STREAMING'
        elif rx_status == self.ERROR:
            self.status = 'ERROR'
        else:
            self.status = "Bad status content"
            raise Warning(self.status)

    ''' Read from serial and parse data response '''
    def _readSample(self):
        try:
            # read bytes until a start frame appears
            while ord(self.ser.read(1)) != self.STX:
                self.status = "Waiting for STX"
                pass

            data_length = ord(self.ser.read(1))
            #print(data_length)

            # read the rest
            raw_msg = self.ser.read(data_length)


            # verify end byte
            if ord(self.ser.read(1)) != self.ETX:
                self.status = "Bad end frame"
                raise Exception(self.status)

            #print('here 3')

            #format = "<B" + (data_length/2) * "H"
            format = "<BHHHHHHHH"

            # verify type of message
            msg = struct.unpack(format, raw_msg)

            if msg[0] != self.DATA_TYPE:
                self.status = "Did not receive message of DATA type"
                raise Exception(self.status)

            # cut off the data type byte
            # format is [Sx1, Sx2, Sy1, Sy2, T1, T2, T3, T4]
            data = list(msg[1:9])
            #print (data)

            # calculate forces. returns [Sx, Sy, T1, T2, T3, T4]
            return CalculateForce(data, self.calibration)

        except serial.SerialTimeoutException:
            raise Exception("Could not write to device, try connecting again.")
        except serial.SerialException:
            raise Exception("Could not configure device, make sure it is connected to " + self.port)
        except Exception as e:
            raise e

    ''' Verify that the sensor is idling before starting '''
    def _initialize(self):
        # Try to talk to the device and verify that it is in the IDLE state
        self._commandIdle()
        tries = 0
        while tries < 5 and not self.initialized:
            # print "trial", tries
            self.ser.flushInput()
            self._commandStatusRequest()

            # verify the response
            try:
                self._readStatusResp()
            except Warning as w:
                print ("Warning " + str(w))

            else:
                if self.status == "IDLING":
                    self.initialized = True
            finally:
                tries += 1

        # wait 0.05 seconds
        sleep(0.05)

    ''' read samples for a short amount of time to determine an offset in Newtons'''
    def _initData(self):
        self._commandStream()
        stoptime = time() + 0.5 # 0.5 seconds of wait
        numsamples = 0
        #print('here')
        #init = [0.0] * 6
        init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        while time() < stoptime:
            data = self._readSample()
            if len(data) != 0:
                numsamples += 1
                for i in range(6):
                    init[i] += data[i]

        for i in range(6):
            init[i] /= numsamples

        #print('here 2')

        self.data_init = init
        self._commandIdle()

    ''' subtract the offset from the data, all in Newtons '''
    def _offset(self, data):
        #offset = [0.0] * 6
        offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(6):
            data[i] -= self.data_init[i]
        return data

    def Initialize(self):
        self._initialize()
        if not self.initialized:
            raise Exception("Unplug and flush serial")

        print("initializing data...")
        self._initData()
        sleep(0.1)

    ''' verifies communication, then reads streaming data
        and writes it to the output file '''
    def closePort(self):
        self.ser.close();
        
        
    def run(self):

        try:
            self._commandStream()
        except Exception as e:
            print(e)
        except Warning as w:
            print(w)
        else:
            print("starting {0} second read".format(self.runtime))

            # File writing implementation
            print("saving to " + self.output)
            f = open(self.output, "w")
            f.write("%time,Sx,Sy,T1,T2,T3,T4\n")

            start = time()
            curr = start
            stop = start + self.runtime

            while not self.stop and curr < stop:
                curr = time()
                data = self._readSample()
                if len(data) != 0:
                        data = self._offset(data)

                        # write to file
                        formatted = str(curr - start) + ","
                        formatted += str(data).replace(" ","")[1:-1] + "\n"
                        f.write(formatted)

            f.close()
            self._commandIdle()
            print("read complete")

def RunSensor(runtime, filename):
    # try:
    #     port_list = findSensors()
    # except Exception as e:
    #     print e
    #     return

    # ''' open the alphabetically first connected sensor '''
    # ts = TactileSensor(output=filename, port=port_list[0], runtime=runtime)

    ''' open COM4 always '''
    ts = TactileSensor(output=filename, port="COM10", runtime=runtime)

    try:
        ts.Initialize()
        print('sensor intialized')
        ts.run()
    except Exception as e:
        print (e)

if __name__ == "__main__":
    print("Running sensor for 30 seconds")
    RunSensor(10, "sensor_output_test_2.csv")
