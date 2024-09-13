# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:28:26 2020

@author: David Quiroga-Martinez
"""
#################### Load libraries and set directories #######################
from random import shuffle
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
#prefs.hardware['audioLib'] = ['pyo']
from psychopy import visual, core, sound, event, gui, logging
import itertools as it
import os
import numpy as np
#from triggers import setParallelData
#setParallelData(0)

# set your own project directory:
os.chdir('C:/Users/au571303/Documents/projects/memory_music_iEEG')
stim_dir = 'stimuli/manipulation_normalized'
log_dir = 'logs'
# Set the frame rate of your screen. Not doing this may create timing issues
frate = 60 #48 #60 #120 
prd = 1000/frate # inter frame interval in ms
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
                 'first melody or not by pressing the buttons as follows:\n\n'
                 '1 = same, 2 = different\n\n'],
                
                ['You will hear a melody.\n\n'
                 'After it finishes, you will see the word "IMAGINE" on the screen. \n'
                 'When this happens, please replay very vividly an INVERTED version '
                 'of the melody in your mind (for example: C B A is an inversion of A B C). \n\n' 
                 'After some time, you will hear a second melody. Please indicate '
                 'whether the second melody is an INVERTED version of the '
                 'first melody or not by pressing the buttons as follows:\n\n'
                 '1 = inverted, 2 = other\n\n']]

rehearse_texts = [["Imagine \n\n"
                   "Now take your time to imagine the original melody in your head.\n"
                   "When ready, press a button to hear the second melody and"
                   " provide an answer."],
                  
                  ["Imagine \n\n"
                   "Now take your time to imagine how an INVERTED version of "
                   "this melody would sound.\n\n Remember: A B C inverted "
                   "would become C B A.\n\n When ready, press a button to hear the "
                   "second melody and provide an answer."]]

continue_texts = [["Those were all the example melodies. Now we are ready\n"
                   "to begin with the real experiment. The melodies will play \n"
                   "automatically. Trials will go faster than the examples.\n"
                   "Remember that you will have to answer: \n\n 1: same, 2: different\n\n"
                   "Try to replay the melody very vividly in your mind "
                   "exactly when the word 'IMAGINE' appears on the screen. \n"
                   "The experiment will start in a moment.\n"],
                  
                  ["Those were all the example melodies. Now we are ready\n"
                   "to begin with the real experiment. The melodies will play\n"
                   "automatically. Trials will go faster than the examples.\n"
                   "Remember that you will have to answer:\n\n 1: inverted, 2: other\n\n"
                   "Try to imagine the inverted melody very vividly in your mind "
                   "exactly when the word 'IMAGINE' appears on the screen. \n"
                   "The experiment will start in a moment.\n"]]

feedback_same_texts = [["Did you answer correctly?\n\n"
                        "The second melody was EXACTLY THE SAME as the first melody.\n"
                        "Now press a button to hear another example."],
                       
                       ["Did you answer correctly?\n\n"
                        "The second melody was an INVERTED version of the first melody.\n"
                        "Now press a button to hear another example."]]

feedback_diff_texts = [["Did you answer correctly?\n\n"
                        "The second melody was totally different from the first melody.\n"
                        "Now press a button to continue."],
                       
                       ["Did you answer correctly?\n\n"
                        "Be careful! This melody sounded similar to the first but\n"
                        "it is NOT an inverted version of it. Therefore, the correct\n" 
                        "answer would be 2 in this case.\n Now press a button to continue."]]

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

    # create prime melodies (15 repetitions of the two melodies)
    primes = [[1,2,3],[3,2,1]]*15 # 60 melodies per block
    
    # create targets:
    if b == 'recognize':
       same = primes.copy()
       diff1 = [[l[a] for a in [2,1,0]] for l in primes[0:15]]
       diff2 = [[l[a] for a in [0,2,1]] for l in primes[15:31]]
       diff = list(it.chain(diff1,diff2))

    elif b == 'invert':
       same = [[l[a] for a in [2,1,0]] for l in primes]
       diff1 = [[l[a] for a in [2,0,1]] for l in primes[0:15]]
       diff2 = [[l[a] for a in [0,1,2]] for l in primes[15:31]]
       diff = list(it.chain(diff1,diff2))
    
    # lists of trial information
    primes = list(it.chain(primes,primes))
    targets = list(it.chain(same,diff))    
    trialID = list(range(len(primes)))   # trial number (before scrambling)

    # random trial order
    rand = trialID.copy()
    shuffle(rand)   

    trialID = [id + 1 for id in trialID] # record trial identity
    melID = list(range(2))*30 
    melID = [id + 1 for id in melID] # record melody identity

    # store randomized versions in corresponding block dictionary:
    blocks[b]['primes'] = [primes[p] for p in rand]
    blocks[b]['targets'] = [targets[t] for t in rand]
    blocks[b]['melID'] = [melID[m] for m in rand]
    blocks[b]['trialID'] = [trialID[t] for t in rand]
    blocks[b]['type'] = [1 if x < 30 else 2 for x in rand]
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
    
keyNext = 'space' # key to advance

#### function to quit the experiment and save log file:
def quit_and_save():
    win.close()
    if logfile:
       logfile.close()
    logging.flush()
    core.quit()
    
event.globalKeys.add(key='escape', func=quit_and_save, name='shutdown')

#### Collect participant identity:
ID_box = gui.Dlg(title = 'Subject identity')
ID_box.addField('ID: ')
ID_box.addField('counterbalance (1 or 2): ')
sub_id = ID_box.show()

block_order = [0,1]
if sub_id[1] == '2':
   block_order = [1,0]

##### create window to display text:
win = visual.Window(fullscr=True, color='black')

##### create other text ojects to display during the experiment:
    
nextText = visual.TextStim(win, text='(press a button to continue)',
                           color=col, pos=(0, -0.8))

endText = visual.TextStim(win, text='The end\n\nThank you for your participation!',
                          color=col, height=0.2, wrapWidth=2)

practice_txt = visual.TextStim(win, text = "Before doing the task, let's "
                                           "hear a couple of examples.\n\n"
                                           "Press a button to hear and memorize"
                                           " a first melody.",
                                         wrapWidth=1.8, color = col)

pause_txt = visual.TextStim(win, text = "Now it is time for a little pause\n\n"
                                        "Please rest as much as you need.\n "
                                        "Press a button when ready to continue",
                                         wrapWidth=1.8, color = col)

fixationCross = visual.TextStim(win, text='+', color=col, height=0.3)
listen_txt =  visual.TextStim(win, text='Listen', color=col, height=0.2)
imagine_txt = visual.TextStim(win, text='Imagine', color=col, height=0.2)

# set default log file
log_fn_def = log_dir + '/' + sub_id[0] +  '_MEG_default.log'
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
        blockendText = visual.TextStim(win, text= "This is the end of this part of the experiment.\n\n "
                                                  "We will continue in a moment\n",
                               color=col, wrapWidth=1.8)
    else:
        blockendText = visual.TextStim(win, text= "This is the end of the block.\n "
                                                  "Now take a little break.\n\n"
                                                  "We will continue in a moment \n",
                               color=col,wrapWidth=1.8)
        
    instr = visual.TextStim(win, text=block['instructions'][0],
                            color=col, wrapWidth=1.8)
 
    continue_txt = visual.TextStim(win, text = block['continue'][0],
                                   wrapWidth=1.8, color = col)
        
    rehearse_txt = visual.TextStim(win, text = block['rehearse'][0],
                                wrapWidth=1.8, color = col)
       
    #### initialize custom log file:
    filename = log_dir + '/' + sub_id[0] + '_' + b + '_MEG.csv'
    logfile = open(filename,'w')
    logfile.write("subject,block,time,melID,trialID,prime,"
                  "target,type,response,rt\n")
    
    #### preload sounds:
    sounds = []
    sounds = [sound.Sound('{}'.format(bb)) for bb in block['stim']]       

    #################### Start experiment ####################################
    ## display instructions:
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
                                               wrapWidth=1.8, color = col)
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
    event.waitKeys(keyList=[keyNext])
        
    ###################### Now we begin the real task ########################
    
    block_time.reset()
    
    silent.play()  # to prevent omission/cut of the first stimulus
    core.wait(silentDur)

    for tidx, t in enumerate(block['primes']):     # loop over trials:
        trialtxt = visual.TextStim(win, text='Trial {} / 60'.format(tidx + 1),
                                   color=col, height = 0.2)
        trialtxt.draw()
        win.flip()
        ttime = block_time.getTime() #  track trial onset
        core.wait(2.5)

        pmel = block['primes'][tidx] #prime melody
        tmel = block['targets'][tidx] # target melody
        # present prime (loop over sounds):
        nextFlip = win.getFutureFlipTime(clock='ptb')
        for p,ps in enumerate(pmel):
            trigger = str(p + 1) + str(ps)
#            win.callOnFlip(setParallelData, int(trigger))
            win.callOnFlip(print, int(trigger))
            sounds[ps-1].play(when = nextFlip)
            for frs in range(int(np.round(50/prd))): # 6 frames = 50 ms
                listen_txt.draw()
                win.flip()
#            win.callOnFlip(setParallelData, 0)
            win.callOnFlip(print, 0)
            for frs in range(int(np.round(450/prd))): # 30 frames = 450 ms
                listen_txt.draw()
                win.flip()
            nextFlip = win.getFutureFlipTime(clock='ptb')
        
        for frs in range(int(np.round(500/prd))): # 30 frames = 450 ms
            listen_txt.draw()
            win.flip()
            
        #delay period:
        for frs in range(int(np.round(2000/prd))): # 30 frames = 450 ms
            imagine_txt.draw()
            win.flip()
       
        #present target (loop over sounds)
        for midx, ts in enumerate(tmel):
            trigger = str(block['type'][tidx]) + str(midx + 1) + str(ts) 
#            win.callOnFlip(setParallelData, int(trigger))           
            win.callOnFlip(print, int(trigger))            
            sounds[ts-1].play(when = nextFlip)
            #clear events and reset the clock for RT
            if midx == 0:
                event.clearEvents(eventType='keyboard')
                RT.reset()
            for frs in range(int(np.round(50/prd))): # 6 frames = 50 ms
                listen_txt.draw()
                win.flip()  
#            win.callOnFlip(setParallelData, 0)
            win.callOnFlip(print, 0)
            for frs in range(int(np.round(450/prd))): # 30 frames = 450 ms
                listen_txt.draw()
                win.flip()
            nextFlip = win.getFutureFlipTime(clock='ptb')
        
        # if there is a response record keys and RT. Else record 0 and time limit:
           
        resp = None 
        while resp == None: 
           key = event.getKeys(timeStamped = RT)
           if len(key) > 0:
                resp = key[0][0] 
                rt = key[0][1]
           elif RT.getTime() > 3.5:
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
        if tidx == 29:
            pause_txt.draw()
            win.flip()
            event.waitKeys()
            
    logfile.close() # save log file
    
    blockendText.draw()
    win.flip()
    event.waitKeys(keyList=[keyNext])
core.wait(2)
core.quit()