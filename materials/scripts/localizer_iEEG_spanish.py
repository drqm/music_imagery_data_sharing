# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:11:11 2020

@author: au571303
"""
from random import shuffle
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
#prefs.hardware['audioLib'] = ['pyo']
from psychopy import visual, core, sound, event, gui, logging
import os
import numpy as np
from sys import argv

from triggers import setParallelData
setParallelData(0)

# set project directory:
my_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(my_path)
os.chdir('..')
#os.chdir('C:/Users/au571303/Documents/projects/memory_music_iEEG')
stim_dir = 'stimuli/manipulation_normalized'
log_dir = 'logs'
# uncomment to fixate randomization seed
#rgnSeed = np.random.randint(900509)
col = 'white'
n_tones = 45 # 40 trials per tone for iEEG
n_targets = 0

######################## create stimulus sequence ############################
seq = np.zeros(n_tones*3 + n_targets)
pattern = [1,2,3]

tidx = np.random.choice(range(n_tones),n_targets, replace = False)
tcount = 0
end = 0
p = 0
for n in range(n_tones):
    start = end
    end = start + 3
    if n > 0:
        p = None
    while p == None:
        shuffle(pattern)
        if pattern[0] != seq[start-1]:
            p = 0
    seq[start:end] = pattern 
    if np.isin(n,tidx):
        seq[end] = 4
        end = end + 1 

###################### Prepare psychopy task #################################

####preload stimuli:
sounds = [sound.Sound('{}/{}.wav'.format(stim_dir,int(s))) for s in np.unique(seq)]

#### Prepare relevant keys:
    
#keyNext = 'space' # key to advance

#### function to quit the experiment and save log file:
def quit_and_save():
    logfile.close()
    logging.flush()
    core.quit()
    
event.globalKeys.add(key='escape', func=quit_and_save, name='shutdown')

csid = ''
if len(argv)>1:
    csid = argv[1]

#### Collect participant identity:
ID_box = gui.Dlg(title = 'Subject identity')
ID_box.addField('ID: ', csid)
ID_box.addField('Photodiode? (yes/no): ', 'yes')
sub_id = ID_box.show()

##### create window to display text:
win = visual.Window(fullscr=True, color='black')

# set frame rate
frate = np.round(win.getActualFrameRate())
prd = 1000 / frate
print('screen fps = {} - cycle duration = {}'.format(frate, prd))

##### create text oBjects to display during the experiment:

instructions = visual.TextStim(win, text = "A continuación usted va a escuchar "
                                           "una serie de sonidos. \n\n"
                                           "Luego de que cada sonido termine, usted verá "
                                           "la palabra 'IMAGINE' en la pantalla. \n\n"
                                           "Cuando esto suceda, por favor reproduzca en su mente "
                                           "e imagine vívidamente el sonido escuchado. \n\n"
                                           "Por favor NO CANTE el sonido "
                                           "con su boca o produzca otros movimeintos.\n\n"
                                           "Presione la barra espaciadora para empezar.",
                                         wrapWidth=1.8, color = col)
                                         
endText = visual.TextStim(win, text='Este es el final de la prueba. \n'
                                    'Presione una tecla para terminar.',
                          wrapWidth=1.8, color = col)

count_txt = ['Escuche','Imagine']
#durs = [550,600,550,600]
durs = [900,900]
trig_dur = 50
fixationCross = visual.TextStim(win, text='+', color=col, height=0.2)
pcolor = 'black'
if sub_id[1] == 'yes':
    pcolor = col
pdiode = visual.Rect(win, size = (.3,.35), pos = (-1,-1),fillColor=pcolor)

# create a silent sound to prevent buffer issues
silentDur = .5
silent = sound.Sound('C', secs=silentDur, volume=0, sampleRate = 44100, stereo = False)

# start log file:
filename = log_dir + '/' + sub_id[0] + '_localizer_iEEG_spanish.log'
lastLog = logging.LogFile(filename, level=logging.INFO, filemode='w')
custom_logfname = log_dir + '/' + sub_id[0] + '_localizer_iEEG_spanish.csv'
logfile = open(custom_logfname,'w')
logfile.write("subject,time,soundID\n")

# Set clock
block_time = core.Clock()

##############################################################################

############################# Start the task #################################

# Flicker to signal task start

for i in range(5):
    pdiode.draw()
    win.flip()
    core.wait(0.05)
    win.flip()
    core.wait(0.05)

instructions.draw()
win.flip()
event.waitKeys()

fixationCross.draw()
win.flip()
silent.play()
core.wait(0.5)
nextFlip = win.getFutureFlipTime(clock='ptb')

for s in seq:
    win.callOnFlip(setParallelData, int(s))
    sounds[int(s)-1].play(when = nextFlip)
    for cidx,cc in enumerate(count_txt):
        count_msg = visual.TextStim(win, text=cc, color = col, height = 0.2)
        for frs in range(int(np.round(trig_dur/prd))): # 6 frames = 50 ms
            count_msg.draw()
            if cidx == 0:
                pdiode.draw()
            win.flip()
            if (frs == 0 and cidx == 0):
                ttime = block_time.getTime()
        if cidx == 0:
            win.callOnFlip(setParallelData, 0)
        for frs in range(int(np.round((durs[cidx]-trig_dur)/prd))): # 30 frames = 450 ms
            count_msg.draw()
            win.flip()
    logfile.write("{},{},{}\n".format(sub_id[0],ttime,s))
    nextFlip = win.getFutureFlipTime(clock='ptb')
    logging.flush()

logfile.close()
endText.draw()
win.flip()
event.waitKeys()
for i in range(5):
    pdiode.draw()
    win.flip()
    core.wait(0.05)
    win.flip()
    core.wait(0.05)
win.close()
core.quit()