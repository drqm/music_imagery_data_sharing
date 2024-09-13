# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:28:26 2020

@author: David Quiroga-Martinez
"""
#################### Load libraries and set directories #######################
from random import shuffle
from psychopy import prefs
prefs.hardware['audioLib'] = ['sounddevice']
from psychopy import visual, core, sound, event, gui, logging
import itertools as it
import os
import numpy as np
from sys import argv

#from triggers import setParallelData
#setParallelData(0)

# set project directory:
my_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(my_path)
os.chdir('..')
#os.chdir('C:/Users/au571303/Documents/projects/memory_music_iEEG')
stim_dir = 'stimuli/manipulation_normalized'
log_dir = 'logs'

# Uncomment this seed to reproduce the randomization:
#rgnSeed = np.random.randint(900509)
col = 'white' # text color
###############################################################################

######### Prepare block-specific texts to display in the experiment ##########

instructions = [['You will hear a melody.\n\n'
                 'After it finishes, you will see the word "IMAGINE" on the screen. \n'
                 'When this happens, please replay very vividly the SAME melody '
                 'in your mind. \n\n'
                 'After some time, you will hear a second melody. Please '
                 'indicate whether the second melody is EXACTLY THE SAME as the '
                 'first melody or not by pressing the keys as follows:\n\n'
                 '1 = same, 2 = different\n\n'],

                ['You will hear a melody.\n\n'
                 'After it finishes, you will see the word "IMAGINE" on the screen. \n'
                 'When this happens, please replay very vividly an INVERTED version '
                 'of the melody in your mind (for example: C B A is an inversion of A B C). \n\n'
                 'After some time, you will hear a second melody. Please indicate '
                 'whether the second melody is an INVERTED version of the '
                 'first melody or not by pressing the keys as follows:\n\n'
                 '1 = inverted, 2 = other\n\n']]


rehearse_texts = [["IMAGINE \n\n"
                   "Now take your time to imagine the original melody in your head.\n\n"
                   "Please DO NOT overtly sing or otherwise move while imagining.\n\n"
                   "When ready, press a key to hear the second melody and "
                   "provide an answer."],

                  ["IMAGINE \n\n"
                   "Now take your time to imagine how an INVERTED version of "
                   "this melody would sound.\n\n"
                   "Please DO NOT overtly sing or otherwise move while imagining.\n\n"
                   "Remember: A B C inverted "
                   "would become C B A.\n\nWhen ready, press a key to hear the "
                   "second melody and provide an answer."]]

continue_texts = [["Those were all the example melodies. Now we are ready\n"
                   "to begin with the real experiment. The melodies will play \n"
                   "automatically. Trials will go faster than the examples.\n"
                   "Remember that you will have to answer: \n\n1: same, 2: different\n\n"
                   "Try to replay the melody very vividly in your mind "
                   "exactly when the word 'IMAGINE' appears on the screen. \n"
                   "Press a key to start the experiment.\n"],

                  ["Those were all the example melodies. Now we are ready\n"
                   "to begin with the real experiment. The melodies will play\n"
                   "automatically. Trials will go faster than the examples.\n"
                   "Remember that you will have to answer:\n\n1: inverted, 2: other\n\n"
                   "Try to imagine the inverted melody very vividly in your mind "
                   "exactly when the word 'IMAGINE' appears on the screen. \n"
                   "Press a key to start the experiment.\n"]]

feedback_same_texts = [["Did you answer correctly?\n\n"
                        "The second melody was EXACTLY THE SAME as the first melody.\n"
                        "Now press a key to hear another example."],

                       ["Did you answer correctly?\n\n"
                        "The second melody was an INVERTED version of the first melody.\n"
                        "Now press a key to hear another example."]]

feedback_diff_texts = [["Did you answer correctly?\n\n"
                        "The second melody was totally different from the first melody.\n"
                        "Now press a key to continue."],

                       ["Did you answer correctly?\n\n"
                        "Be careful! This melody sounded similar to the first but\n"
                        "it is NOT an inverted version of it. Therefore, the correct\n"
                        "answer would be 2 in this case.\nNow press a key to continue."]]

orders_same = [[0,1,2],[2,1,0]] # for practice trials
orders_diff = [[0,2,1],[0,1,2]] # for practice trials
##############################################################################

############# create stimulus list for each experimental block ###############

blocks = {'recognize': [],'invert': []}
blocknames = ['recognize','invert']
stim = [1,2,3]
for bidx,b in enumerate(blocknames):
    blocks[b] = {'stim': [], 'primes': [], 'targets': [], 'melID': [],
                 'trialID': [],'type': [], 'melodies': [],'rehearse': [],
                 'instructions': [],'continue': [], 'feedback_same': [],
                 'feedback_diff': [], 'order_same': [], 'order_diff': []}

    # create prime melodies (12 repetitions of the two melodies)
    primes = [[1,2,3],[3,2,1]]*12 # 48 melodies per block

    # create targets:
    if b == 'recognize':
       same = primes.copy()
       diff1 = [[l[a] for a in [2,1,0]] for l in primes[0:12]]
       diff2 = [[l[a] for a in [0,2,1]] for l in primes[12:25]]
       diff = list(it.chain(diff1,diff2))

    elif b == 'invert':
       same = [[l[a] for a in [2,1,0]] for l in primes]
       diff1 = [[l[a] for a in [2,0,1]] for l in primes[0:12]]
       diff2 = [[l[a] for a in [0,1,2]] for l in primes[12:25]]
       diff = list(it.chain(diff1,diff2))

    # lists of trial information
    primes = list(it.chain(primes,primes))
    targets = list(it.chain(same,diff))
    trialID = list(range(len(primes)))   # trial number (before scrambling)

    # random trial order
    rand = trialID.copy()
    shuffle(rand)

    trialID = [id + 1 for id in trialID] # record trial identity
    melID = list(range(2))*24
    melID = [id + 1 for id in melID] # record melody identity

    # store randomized versions in corresponding block dictionary:
    blocks[b]['primes'] = [primes[p] for p in rand]
    blocks[b]['targets'] = [targets[t] for t in rand]
    blocks[b]['melID'] = [melID[m] for m in rand]
    blocks[b]['trialID'] = [trialID[t] for t in rand]
    blocks[b]['type'] = [1 if x < 24 else 2 for x in rand]
    blocks[b]['melodies'] = [list(l) for l in list(it.permutations(stim))]
    blocks[b]['stim'] = ['{}/{}.wav'.format(stim_dir,s) for s in stim]
    blocks[b]['instructions'] = instructions[bidx]
    blocks[b]['rehearse'] = rehearse_texts[bidx]
    blocks[b]['continue'] = continue_texts[bidx]
    blocks[b]['feedback_same'] = feedback_same_texts[bidx]
    blocks[b]['feedback_diff'] = feedback_diff_texts[bidx]
    blocks[b]['order_same'] = orders_same[bidx]
    blocks[b]['order_diff'] = orders_diff[bidx]

##############################################################################

####################### prepare Psychopy task ################################

#### Prepare relevant keys:

#keyNext = 'space' # key to advance

#### function to quit the experiment and save log file:
def quit_and_save():
    win.close()
    if logfile:
       logfile.close()
    logging.flush()
    core.quit()

event.globalKeys.add(key='escape', func=quit_and_save, name='shutdown')

#### Collect participant identity:
csid = ''
if len(argv)>1:
    csid = argv[1]

ID_box = gui.Dlg(title = 'Subject identity')
ID_box.addField('ID: (change if subject ID is incorrect) ', csid)
ID_box.addField('block order (random order: leave blank; maintenance first: 1; manipulation first: 2): ')
ID_box.addField('Photodiode? (yes/no): ', 'yes')

sub_id = ID_box.show()

block_order = [0,1]
shuffle(block_order)
pcolor = 'black'

if sub_id[1] == '1':
   block_order = [0,1]
if sub_id[1] == '2':
   block_order = [1,0]
if sub_id[2] == 'yes':
   pcolor = col

##### create window to display text:
win = visual.Window(fullscr=True, color='black')

# set frame rate
# frate = np.round(win.getActualFrameRate())
# prd = 1000 / frate
# print('screen fps = {} - cycle duration = {}'.format(frate, prd))

##### create other text ojects to display during the experiment:

nextText = visual.TextStim(win, text='(press a key to continue)',
                           color=col, pos=(0, -0.8))

endText = visual.TextStim(win, text= "This is the end of the task.\n\n "
                                     "Press a key to end this part of the experiment.",
                          color=col, height=0.2, wrapWidth=1.8)

practice_txt = visual.TextStim(win, text = "Before doing the task, let's "
                                           "hear a couple of examples.\n\n"
                                           "Press a key to hear and memorize "
                                           "a first melody.\n\nReady?",
                                         wrapWidth=1.8,
                                         color = col)

pause_txt = visual.TextStim(win, text = "Now it is time for a little pause\n\n"
                                        "Please rest as much as you need.\n "
                                        "Press a key when ready to continue",
                                         wrapWidth=1.8,
                                         color = col)

vividness_txt = visual.TextStim(win, text = "In the previous block, how vivid "
                                            "was your mental image of the sounds? \n\n"
                                            "Please type one of the numbers in "
                                            "the keyboard:\n\n "
                                            " not vivid at all <  1  2  3  4  5  > "
                                            "extremely vivid ",
                                         wrapWidth=1.8,
                                         color = col)

fixationCross = visual.TextStim(win, text='+', color=col, height=0.3)
listen_txt =  visual.TextStim(win, text='Listen', color=col, height=0.2)
imagine_txt = visual.TextStim(win, text='Imagine', color=col, height=0.2)
pdiode = visual.Rect(win, size = (.3,.35), pos = (-1,-1),fillColor=pcolor)

# set default log file
log_fn_def = log_dir + '/' + sub_id[0] +  '_iEEG_default.log'
lastLog = logging.LogFile(log_fn_def, level=logging.INFO, filemode='a')

## create a silent sound to prevent buffer issues
silentDur = .5
silent = sound.Sound('C', secs=silentDur, volume=0, sampleRate = 44100, stereo = False)

# set relevant clocks:
RT = core.Clock()
block_time = core.Clock()

##############################################################################

############## Now run the experimental blocks ###############################

#### select blocks to include:

bnames = ['recognize','invert'] # block names to loop over
bnames = [bnames[b] for b in block_order] # counterbalance

for bidx, b in enumerate(bnames): # loop over blocks

    ################### prepare block-specific variables #####################
    block = blocks[b]
    if bidx == (len(bnames) - 1):
        blockendText = visual.TextStim(win, text= "This is the end of the task.\n\n "
                                                  "Press a key to end this part of the experiment.",
                               color=col, wrapWidth=1.8)
    else:
        blockendText = visual.TextStim(win, text= "This is the end of the block.\n "
                                                  "Now take a little break.\n\n"
                                                  "Press a key to continue when ready. \n",
                               color=col, wrapWidth=1.8)

    instr = visual.TextStim(win, text=block['instructions'][0],
                            color=col, wrapWidth=1.8)

    continue_txt = visual.TextStim(win, text = block['continue'][0],
                                   color = col, wrapWidth=1.8)#, wrapWidth=1.8)

    rehearse_txt = visual.TextStim(win, text = block['rehearse'][0],
                                   color = col,wrapWidth=1.8)

    #### initialize custom log file:
    filename = log_dir + '/' + sub_id[0] + '_' + b + '_iEEG.csv'
    logfile = open(filename,'w')
    logfile.write("subject,block,time,melID,trialID,prime,"
                  "target,type,response,rt\n")

    #### preload sounds:
    sounds = []
    sounds = [sound.Sound('{}'.format(bb)) for bb in block['stim']]

    #################### Start experiment ####################################
    ## display instructions:
    for i in range(5):
        pdiode.draw()
        win.flip()
        core.wait(0.05)
        win.flip()
        core.wait(0.05)
    
    instr.draw()
    nextText.draw()
    win.flip()
    event.waitKeys()

    ##################### Run example trials #################################

    practice_txt.draw()
    nextText.draw()
    win.flip()
    event.waitKeys()

    # We will present a "same" (or "invert") and a "different" (or "other") trial.
    cases = ['same','diff']

    for tt in cases:

        ########## Prepare condition-specific variables #######################
        feedback_txt = visual.TextStim(win, text = block['feedback_' + tt][0],
                                       color = col, wrapWidth=1.8)
        order = block['order_' + tt]

        ################### begin stimulus presentation #######################

        #prime melody:

        listen_txt.draw(win)
        win.flip()
        silent.play()  # to prevent omission/cut of the first stimulus
        core.wait(1)
        for s in [0,1,2]:
            sounds[s].play()
            core.wait(0.5)

        rehearse_txt.draw(win)
        nextText.draw()
        win.flip()
        event.waitKeys()

        # target melody:

        listen_txt.draw(win)
        win.flip()
        silent.play()  # to prevent omission/cut of the first stimulus

        for s in order:
            sounds[s].play()
            core.wait(0.5)

        core.wait(2)
        feedback_txt.draw()
        nextText.draw()
        win.flip()
        event.waitKeys()

    continue_txt.draw()
    win.flip()
    event.waitKeys()

    ###################### Now we begin the real task ########################

    block_time.reset()

    silent.play()  # to prevent omission/cut of the first stimulus
    core.wait(silentDur)

    for tidx, t in enumerate(block['primes']):     # loop over trials:
        trialtxt = visual.TextStim(win, text='Trial {} / 48'.format(tidx + 1),
                                   color=col, height = 0.2)
        trialtxt.draw()
        win.flip()
        ttime = block_time.getTime() #  track trial onset
        core.wait(2.5)

        pmel = block['primes'][tidx] #prime melody
        tmel = block['targets'][tidx] # target melody
        # present prime (loop over sounds):
        #nextFlip = win.getFutureFlipTime(clock='ptb')
        for p,ps in enumerate(pmel):
            cwait = 0
            trigger = str(p + 1) + str(ps)
            #win.callOnFlip(setParallelData, int(trigger))
            sounds[ps-1].play()
            listen_txt.draw()
            win.flip()
            if p == 0:
                cwait = 0.05
                pdiode.draw()
                core.wait(cwait)
                listen_txt.draw()
                win.flip()
            core.wait(0.5-cwait)
        core.wait(0.5)
        imagine_txt.draw()
        win.flip()
        core.wait(2)

        #present target (loop over sounds)
        for midx, ts in enumerate(tmel):
            trigger = str(block['type'][tidx]) + str(midx + 1) + str(ts)
            #win.callOnFlip(setParallelData, int(trigger))
            sounds[ts-1].play()
            listen_txt.draw()
            win.flip()

        #clear events and reset the clock for RT
            if midx == 0:
                event.clearEvents(eventType='keyboard')
                RT.reset()
            core.wait(0.5)
        # if there is a response record keys and RT. Else record 0 and time limit:

        resp = None
        while resp == None:
           key = event.getKeys(timeStamped = RT)
           if len(key) > 0:
                resp = key[0][0]
                rt = key[0][1]
           elif RT.getTime() > 5:
                resp = 0
                rt = RT.getTime()

        ## gather log info and add to logfile:
        lrow = '{},{},{},{},{},{};{};{},{};{};{},{},{},{}\n'
        lrow = lrow.format(sub_id[0],b,ttime,block['melID'][tidx],block['trialID'][tidx],
                           pmel[0],pmel[1],pmel[2], tmel[0],tmel[1],tmel[2],
                           block['type'][tidx], resp,rt*1000)
        logfile.write(lrow)
        core.wait(0.1)  # 100 ms after response, to start new trial

        # now we introduce a small pause in trial 30 so that participants can rest
        if tidx == 23:
            pause_txt.draw()
            win.flip()
            event.waitKeys()
    vividness_txt.draw()
    win.flip()
    event.waitKeys(keyList = ['1','2','3','4','5'])
    logfile.close() # save log file

    blockendText.draw()
    win.flip()
    event.waitKeys()

core.wait(2)
for i in range(5):
    pdiode.draw()
    win.flip()
    core.wait(0.05)
    win.flip()
    core.wait(0.05)
core.quit()
