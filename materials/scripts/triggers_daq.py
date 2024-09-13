#54121 -+sycu 6y 
from psychopy import parallel
import serial
import os

##### Set port address to match your local machine

paddress = '/dev/parport0' # '0xDFF8' or 'COM1' in Windows

##################################################

try:
    port = serial.Serial(paddress)
    port_type = 'serial'
except NotImplementedError:
    port = parallel.setPortAddress(address=paddress)
    port_type = 'parallel'
except NotImplementedError:
    ## Change address to corresponding folder
    #os.chdir('~/LinuxDrivers/USB/python')
    from usb_1208FS import *
    usb1208FS = usb_1208FS()
    port_type = 'daq'
except:
    port_type = 'Not set'

print('port type: {}'.format(port_type))

if port_type == 'parallel':
    def setParallelData(code=1):
        port.setData(code)
        print('trigger sent {}'.format(code))
elif port_type == 'serial':
    def setParallelData(code=1):
        port.write(bytes(str(code).encode()))
        print('trigger sent {}'.format(code))
elif port_type == 'daq':
    def setParallelData(code=1):
        value = int(code,16)
        usb1208FS.AOut(0,value)
        print('trigger sent {}'.format(code))
else:
    def setParallelData(code=1):
        print('trigger not sent {}'.format(code))
