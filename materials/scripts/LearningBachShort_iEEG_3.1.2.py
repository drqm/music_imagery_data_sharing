
"""
Created on Fri Jul 17 16:59:51 2020

@author: Leonardo Bonetti, Francesco Carlomagno

leonardo.bonetti@clin.au.dk

"""

#To use this script please do the following:
#-place the provided stimuli folders in your favorite directory
#-assign to the variable 'stimidr' (few lines below) such directory
#-define your log directory
#Please note that the csv file with the crucial stimuli presentation IDs and
# behavioural responses is created within the log directory

#################### Load libraries and set directories ####################
from random import shuffle
from psychopy import prefs
#prefs.hardware['audioLib'] = ['pyo']
prefs.hardware['audioLib'] = ['sounddevice']
from psychopy import visual, core, sound, event, gui, monitors, logging
import itertools as it
import os
import numpy as np
from sys import argv
# from triggers import setParallelData
# setParallelData(0)

#################### YOUR DIRECTORY ##########################################
#THIS IS YOUR DIRECTORY WHERE YOU PUT THE PROVIDED FOLDERS, PLEASE UPDATE THIS IF NECESSARY
my_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(my_path)
os.chdir('..')
cwd = os.getcwd()
print('current working directory: ' + cwd)
logdir = cwd + '/logs'
stimdir = cwd + '/stimuli'
#logdir = ('C:/Users/au571303/Documents/projects/memory_music_iEEG/logs')
#stimdir = ('C:/Users/au571303/Documents/projects/memory_music_iEEG/stimuli')

#monitor settings
#mon = monitors.Monitor('SonyG55')#fetch the most recent calib for this monitor
#mon.setDistance(50)#further away than normal?
#win = visual.Window(size=[1920,1080], monitor=mon)

#################### Actual codes ####################

##TEST

# quit key
def quitpd():
    if logfile:
       logfile.close()
    core.quit()
event.globalKeys.add(key='escape',func=quitpd,name='shutdown')

#GUI for subject ID
csid = ''
if len(argv)>1:
    csid = argv[1]

ID = gui.Dlg(title = 'subj ID')
ID.addField('ID (change if subject ID is incorrect): ',csid)
ID.addField('Photodiode? (yes/no): ', 'yes')
subID = ID.show()

#create csv file for log
filename = logdir + '/' + subID[0] + '_learning_bach_iEEG_custom.csv'
logfile = open(filename,'w')
logfile.write("subj;trial;trigger;response;RT;time \n")

#create log file
block_time = core.Clock()
logging.setDefaultClock(block_time)
filename2 = logdir + '/' + subID[0] + '_learning_bach_iEEG_default.log'
lastLog = logging.LogFile(filename2, level=logging.INFO, filemode='a')

#getting file for learning phase
os.chdir(stimdir + '/learningdef')
Bachminor = sound.Sound('Bachminorprelude.wav')

#getting files for recognition phase
os.chdir(stimdir + '/recogdef')
pathwave = stimdir + '/recogdef'
wavefilesd = []
wavenamesd = []
for file in sorted(os.listdir(pathwave)):
    if file.endswith(".wav"):
        wavenamesd.append(file)
        wavefilesd.append(sound.Sound(file))

#reproducing several times (here 8 times) the 3 Old excerpts
wnd = wavenamesd[24:] #extracting Old excerpts
wnd = wnd * 8 #multiplying them by 8
wfd = wavefilesd[24:]
wfd = wfd * 8
wavenames = wavenamesd[0:24] + wnd #getting the final list name
wavefiles = wavefilesd[0:24] + wfd #getting the final list files
wavezip = list(zip(wavenames,wavefiles))
shuffle(wavezip)

#initializing RT variable to get RTs
RT = core.Clock()
block_time = core.Clock() #this should not be useful anymore..

#this is a bit redundant.. anyway..
SPAAAAACE = 'space'

#preparing window for the screen
color_txt = 'white'
win = visual.Window(fullscr = True, color = 'black')#[.9,.9,.9])

#Set frame rate
# frate = np.round(win.getActualFrameRate())
# prd = 1000 / frate
# print('screen fps = {} - cycle duration = {}'.format(frate, prd))

#pd = visual.TextStim(win,text = 'First, you will listen to a complete musical piece '
#                                '(Learning phase). \n\n Please try to remember it as '
#                                'much as possible. \n\n Second, you will listen to 42 '
#                                'short musical excerpts (Recognition phase). \n\n For '
#                                'each of them you will be asked to indicate whether '
#                                'they were extracted from the musical piece of the '
#                                'learning phase or not. \n\n Press space to continue',
#                                 color = color_txt, height = 0.1)
pd = visual.TextStim(win,text = 'You will listen to a complete musical piece repeated twice. '
                                '\n\n Please try to remember it as '
                                'much as possible. \n\n Later you will do a memory task based '
                                'on this music.\n\n Press space to begin.',
                                color = color_txt, height = 0.1)
#instr = visual.TextStim(win,text = 'Learning phase \n\n Now you are going to listen '
#                                   'to a complete musical piece \n\n Please try to '
#                                   'remember it as much as possible \n\n Press space '
#                                  'to continue',color = color_txt)
instrrec = visual.TextStim(win,text = 'recognition part..',color = color_txt)
fix_c = visual.TextStim(win,text = '+', color = color_txt,height = 0.2)
block_time.reset() #this should not be useful anymore..
pcolor = 'black'
if subID[1]=='yes':
    pcolor = color_txt
pdiode = visual.Rect(win, size = (.3,.35), pos = (-1,-1),fillColor=pcolor)

# Flicker to signal task start
for i in range(5):
    pdiode.draw()
    win.flip()
    core.wait(0.05)
    win.flip()
    core.wait(0.05)
    
#actual experiment (learning phase)
pd.draw()
win.flip()
event.waitKeys(keyList = [SPAAAAACE])
#instr.draw()
#win.flip()
#event.waitKeys(keyList = [SPAAAAACE])
ttime = block_time.getTime() #this should not be useful anymore..

#playing the whole musical piece 2 times
for ll in range(2):
    dumm = 'Repetition ' + str(ll+1) + '\n Press space to continue'
    playlear = visual.TextStim(win,text = dumm, color = color_txt)
    playlear.draw()
    win.flip()
    event.waitKeys(keyList = [SPAAAAACE])
    fix_c.draw()
    win.flip()
    core.wait(0.5)
    #nextFlip = win.getFutureFlipTime(clock='ptb')
    #win.callOnFlip(setParallelData, 103) #SENDING TRIGGER
    Bachminor.play() #open again this line later

    fix_c.draw()
    pdiode.draw()
    win.flip()
    core.wait(0.05)
    fix_c.draw()
    win.flip()
    core.wait(0.05)
    
    event.clearEvents(eventType='keyboard')
    core.wait(25.9) #open again this line later

#actual experiment (recognition phase)
playrec = visual.TextStim(win,text = 'Now you will listen to 48 short musical excerpts. \n\n '
                                     'For each of them please indicate with the keyboard '
                                     'whether it was extracted from the musical '
                                     'piece that you have just listened to (press 1) '
                                     'or not (press 2). \n\n Press space to continue',
                                     color = color_txt)
playrec.draw()
win.flip()
event.waitKeys(keyList = [SPAAAAACE])
#over musical excerpts (trials)
for wavve in range(len(wavefiles)):
    #displaying the progressive trial number for participants' comfort
    jes = 'trial number ' + str((wavve + 1)) + ' / ' + str(len(wavefiles))
    instrrectr = visual.TextStim(win,text = jes,color = color_txt)
    instrrectr.draw()
    win.flip()
    core.wait(1)
    fix_c.draw()
    win.flip()
    core.wait(0.5)
    if 'old' in wavezip[wavve][0]:
        trigval = 10
    else:
        trigval = 50
    #nextFlip = win.getFutureFlipTime(clock='ptb')
    #win.callOnFlip(setParallelData, trigval) #SENDING TRIGGER
    ttime = block_time.getTime() #this should not be useful anymore..
    wavezip[wavve][1].play()
    fix_c.draw()
    pdiode.draw()
    win.flip()
    event.clearEvents(eventType='keyboard')
    RT.reset()
    resp = None
    core.wait(0.05)
    fix_c.draw()
    win.flip()
    core.wait(0.05)
    while resp == None:
        key = event.getKeys(keyList = ['1','2'])
        if len(key) > 0:
            rt = RT.getTime()
            resp = key[0][0]
        elif RT.getTime() > 4.9: #5 seconds of maximum wait if subject does not reply
            resp = 0
            rt = RT.getTime()
    if RT.getTime() > 1.7:
        core.wait(1)
    else:
        core.wait(1+1.7-RT.getTime())
    #writing RT, trial ID, subject's response
    lrow = '{};{};{};{};{};{} \n'
    lrow = lrow.format(subID[0],wavezip[wavve][0], trigval,resp, round(rt*1000),str(ttime))
    logfile.write(lrow) 

logging.flush()
logfile.close()

#final message
playlear = visual.TextStim(win,text = 'That was the end. \n'
                                      'Thank you very much! \n\n '
                                      'Press space to end this part of the experiment',
                            color = color_txt)
playlear.draw()
win.flip()
event.waitKeys(keyList = [SPAAAAACE])
# Flicker task end
for i in range(5):
    pdiode.draw()
    win.flip()
    core.wait(0.05)
    win.flip()
    core.wait(0.05)
####################
