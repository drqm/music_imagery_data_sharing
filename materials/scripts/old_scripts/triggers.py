#54121 -+sycu 6y 
from psychopy import parallel
import platform

PLATFORM = platform.platform()
if 'Linux' in PLATFORM:
    port = parallel.ParallelPort(address='/dev/parport0')  # on MEG stim PC
else:  # on Win this will work, on Mac we catch error below
    try:
        port = parallel.setPortAddress(address='0xDFF8')
        #port = parallel.ParallelPort(address=0xDFF8)  # on MEG stim PC
    except NotImplementedError:
        port = []
# NB problems getting parallel port working under conda env
# from psychopy.parallel._inpout32 import PParallelInpOut32
# port = PParallelInpOut32(address=0xDFF8)  # on MEG stim PC
# parallel.setPortAddress(address='0xDFF8')
# port = parallel

# Figure out whether to flip pins or fake it
if port:
    def setParallelData(code=1):
        port.setData(code)
        print('trigger sent {}'.format(code))
else:
    def setParallelData(code=1):
        print('trigger not sent {}'.format(code))