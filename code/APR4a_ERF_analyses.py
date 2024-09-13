#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Script to compute evoked responses and their inverse solution '''

# Append paths
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
scripts_dir = '/projects/' + proj_name + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

# Import libraries
import mne
import os
import os.path as op
import numpy as np
import pickle
from warnings import filterwarnings
from sys import argv
import matplotlib.pyplot as plt
from stormdb.access import Query
import pandas as pd
from src.preprocessing import WM_epoching, main_task_events_fun, default_events_fun
from src.decoding_functions import smooth_data

filterwarnings("ignore", category=DeprecationWarning)

# Set directories
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
os.environ['MINDLABPROJ']= project
os.environ['MNE_ROOT']='/users/david/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

raw_path = project_dir + '/scratch/maxfiltered_data/tsss_st16_corr96'
ica_path = project_dir + '/scratch/working_memory/ICA'
avg_path = project_dir + '/scratch/working_memory/averages'
log_path = project_dir + '/misc/working_memory_logs'

subjects_dir = project_dir + '/scratch/fs_subjects_dir'
fwd_path = project_dir + '/scratch/forward_models'

# Get subjects info
qy = Query(project)
subs = qy.get_subjects()

scode = 63
if len(argv) > 1:
    scode = int(argv[1])

sub = subs[scode-1] 

# Set other variabels
conds_orig = ['main','inv'] # original file names
conds = ['maintenance','manipulation'] # condition names
lnames = ['recognize','invert'] # logfile names

# Saving and plotting options
save_averages = True
plot_topo = True
compute_sources = True

# Epoching params
tmin = 2 #0 #2 #-.1
tmax = 4 #2 #4
smooth_tstep = 0.025
smooth_twin = 0.05
l_freq=.05
h_freq=None
reject = dict(mag = 4e-12, grad = 4000e-13)

# Name of analysis
suffix = "lf_{}_hf_{}_tstep_{}_twin_{}_{}_{}".format(l_freq,h_freq,smooth_tstep, smooth_twin, tmin, tmax)

# Initialize
epochs = {}
evokeds = {}
print('\n epoching \n')

for cidx, c in enumerate(conds_orig):
    nc = conds[cidx]
    fname = os.path.join(raw_path, sub, c + '_raw_tsss.fif') # raw file
    icaname = os.path.join(ica_path,sub, c + '_raw_tsss-ica.fif') # ICA solution
    lfname = op.join(log_path, sub[0:4] + '_' + lnames[cidx] + '_MEG.csv') #log file
    epochs[nc] = WM_epoching(data_path=fname, ica_path=icaname, tmin=tmin, tmax=tmax,
                                l_freq=l_freq, h_freq=h_freq, resample = 100, bads=[],
                                baseline=None, notch_filter=50,
                                events_fun=main_task_events_fun, events_fun_kwargs = {'cond': nc, 'lfname': lfname},
                                reject=reject)

epochs = mne.concatenate_epochs([epochs[e] for e in epochs]) # join all epochs
print(epochs)

# Smooth if required
if smooth_tstep:
    new_data, new_times = smooth_data(epochs.get_data(), tstart=epochs.times[0],
                                      tstep=smooth_tstep, twin=smooth_twin,
                                      Fs=epochs.info['sfreq'], taxis=2)

new_info = epochs.info.copy()
new_info['sfreq'] = 1/smooth_tstep
new_event_id = {eid: epochs.event_id[eid] for eid in epochs.event_id if epochs.event_id[eid] in epochs.events[:,2]}
epochs = mne.EpochsArray(new_data, info = new_info, events = epochs.events,
                         event_id = new_event_id,tmin = epochs.tmin)

# Calculate evoked responses
evokeds = {e: epochs[e].average() for e in epochs.event_id}

other_conds = ['maintenance','manipulation','mel1','mel2','maintenance/mel1',
               'maintenance/mel2','manipulation/mel1','manipulation/mel2']
for oc in other_conds:
    evokeds[oc] = epochs[oc].average()
    evokeds[oc].comment = oc

# Compute differences between specific conditions
evokeds_diff = {'melody': ['mel2','mel1'], 
                'block': ['manipulation','maintenance'],
                'melody_maintenance': ['maintenance/mel2','maintenance/mel1'],
                'melody_manipulation': ['manipulation/mel2','manipulation/mel1'],
                'interaction': ['melody_maintenance','melody_manipulation']}

for e in evokeds_diff:
    e2, e1 = evokeds_diff[e]
    evokeds[e] = mne.combine_evoked([evokeds[e2],evokeds[e1]],weights=[1,-1])
    evokeds[e].comment = e
    
#save output
if save_averages:
    evkd_fname = op.join(avg_path,'data',sub,sub + f'_evoked_{suffix}.p')
    evkd_file = open(evkd_fname,'wb')
    pickle.dump(evokeds,evkd_file)
    evkd_file.close()
    print('evoked file saved')

print('done epoching')

## Source analysis
if compute_sources:
    print('\n computing sources \n')
    fwd_fn = op.join(fwd_path, sub + '_vol-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fn)

    data_cov = mne.compute_covariance(epochs.load_data().copy().pick_types('mag'),
                                       tmin= tmin,
                                       tmax = tmax,rank ='info')
    ## inverse solution
    inv = mne.beamformer.make_lcmv(epochs.info,fwd,data_cov, reg=0.05,
                                    pick_ori='max-power',
                                    weight_norm= 'nai', rank = 'info')
    SNR = 3
    sources = {}
    for e in evokeds:
        sources[e] = mne.beamformer.apply_lcmv(evokeds[e],inv)#,max_ori_out='signed')

    src_fname = op.join(avg_path,'data',sub,sub + '_evoked_sources_{}.p'.format(suffix))
    src_file = open(src_fname,'wb')
    pickle.dump(sources,src_file)
    src_file.close()

    inv_fname = op.join(avg_path,'data',sub,sub + '_evoked_inverse_{}.p'.format(suffix))
    inv_file = open(inv_fname,'wb')
    pickle.dump(inv,inv_file)
    inv_file.close()
    print('\n sources file saved')

### Plotting
print('making some plots')
plot_conds = {'all': ['maintenance','manipulation'],
              'block_diff': ['block'],
              'mel_diff': ['melody_maintenance','melody_manipulation'],
              'interaction': ['interaction']}

ctypes = ['mag','grad']
for pc in plot_conds:
    for t in ctypes:
        merge_grads = False
        if t == 'grad':
            merge_grads = True
        fig, axis = plt.subplots(nrows=1,ncols=1,figsize=(60,30))
        mne.viz.plot_evoked_topo([evokeds[e].copy().pick_types(t) for e in plot_conds[pc]],
                               axes = axis,merge_grads=merge_grads,show=False, vline = [0,2,4])
        plt.tight_layout()
        fig.savefig(avg_path + '/figures/{}/{}_ERFs_topo_{}_{}_{}.pdf'.format(sub,sub,pc,t,suffix))

plt.close('all')
