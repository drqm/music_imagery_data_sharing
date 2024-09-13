#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Script to do stats on ROI data '''

# Append path
proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
scripts_dir = '/projects/' + proj_name + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

# Load libraries
import mne
import numpy as np
from matplotlib import pyplot as plt
from stormdb.access import Query
from pickle import load
from scipy import stats
from mne.datasets import sample
from mne.stats import spatio_temporal_cluster_1samp_test
import os
import os.path as op
import pickle
from copy import deepcopy
from sys import argv
import src.group_stats as gs
from random import choices

# Set other variables of interes
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
figures_dir = wdir + 'results/figures/'
src_sample = mne.read_source_spaces(subs_dir +
                                     '/fsaverage/bem/fsaverage-vol-5-src.fif')

stats_dir = wdir + 'results/stats/'

#Get subjects:
qr = Query(proj_name)
subjects = qr.get_subjects()

subs = range(11,91)

# Add exclusions
performance_exc = [55,58,60,73,76,82]
no_source_exc = [30,51,42]
noise_exc = [15]
no_data_exc = [32,33]

exclude = np.array(performance_exc + no_source_exc + noise_exc + no_data_exc)
subs = np.array([s for s in subs if s not in exclude])
subs.shape

suffix = 'patterns_sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_localized_ROI'

print('ROI grand average analyses')

# Load and store data
all_data = {}
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        dfname = op.join(data_dir,scode, scode + '_' + suffix + '.p')
        
        print('\n\nloading file {}\n'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)
        
        for cd in curdata:
            cdata = curdata[cd]['data']

            all_data.setdefault(cd,{})
            for ROI in cdata:
                print('appending subject {} condition {} ROI {}'.format(scode,cd,ROI))
                all_data[cd].setdefault(ROI,np.array([cdata[ROI]]))
                #all_data[cd][cd2].append([cdata])
                if sidx > 0:
                    all_data[cd][ROI] = np.vstack((all_data[cd][ROI],np.array([cdata[ROI]])))
    except Exception as e:
        print(e)
        continue

n = all_data[cd][ROI].shape[0]
times = curdata[cd]['times']

### Stats on main comparisons
conds = [k for k in all_data]

stat_results = {}
for cidx, cnd in enumerate(conds):
    stat_results[cnd] = {}
    for ROI in all_data[cnd]:
        cdata = all_data[cnd][ROI]
        print(cdata.shape)
        stat_results[cnd][ROI] = gs.do_stats(cdata, 'montecarlo', #adjacency=adjacency,
                                        FDR_alpha=.025, n_permutations=5000, cluster_alpha=.05)#, cluster_method='TFCE')
        print('reporting stats for cond {} ROI {}:\n'.format(cnd,ROI))
        print('minimum pval = ', np.round(np.min(stat_results[cnd][ROI]['pvals']),2))
        print('minimum tstat = ', np.round(np.min(stat_results[cnd][ROI]['tvals']),2))
        print('maximum tstat = ', np.round(np.max(stat_results[cnd][ROI]['tvals']),2),'\n')

# Save output    
print('saving stats results')
stats_fname = op.join(stats_dir,'ROI_source_stats_{}.p'.format(suffix))
sfile = open(stats_fname,'wb')
pickle.dump(stat_results,sfile)
sfile.close()

### Make plots
cnd2plot = ['maintenance1','manipulation1','interaction']
additional = [[],
              [],
              []]
ROI2plot = ['Right Auditory',
            'Right Memory',
            'Left Dorsal Cognitive Control',
            'Right Dorsal Cognitive Control',
            'Right Ventral Cognitive Control']

ncols = 5
for cndix, cnd in enumerate(cnd2plot):
        nrows = np.ceil(len(ROI2plot)/ncols).astype(int)
        fig, axes = plt.subplots(ncols=ncols,nrows=nrows,
                                  figsize = (4*ncols,nrows*3))
        for ROIx, ROI in enumerate(ROI2plot):
            cplts = []
            rix, cix = ROIx//ncols,ROIx%ncols

            if nrows == 1:
                cax = axes[cix]
            else:
                cax = axes[rix,cix]

            ci_upper = np.squeeze(stat_results[cnd][ROI]['data_mean'] + 2*stat_results[cnd][ROI]['data_sd']/np.sqrt(n-1))
            ci_lower = np.squeeze(stat_results[cnd][ROI]['data_mean'] - 2*stat_results[cnd][ROI]['data_sd']/np.sqrt(n-1))
            ccmask = np.squeeze(np.array(stat_results[cnd][ROI]['mask']).astype(float))
            ccmask[ccmask==0.] = np.nan
            cax.fill_between(times, ci_lower, ci_upper, color='k', alpha=.05)
            cax.plot(times,np.squeeze(stat_results[cnd][ROI]['data_mean']),color='k',
                                alpha=.8)#,label='difference')
            cax.plot(times,np.squeeze(stat_results[cnd][ROI]['data_mean'])*ccmask, 
                                color='k',linewidth=4,alpha = .4)
            for a in additional[cndix]:
                cax.plot(times,np.squeeze(stat_results[a][ROI]['data_mean']),label=a,alpha=.9)
            cax.set_title(ROI)
            cax.set_ylim(-1.5,1.5)
            cax.set_xlim(-.1,4)
            cax.set_xlabel('time (s)')
            cax.set_ylabel('source activation (a.u.)')
            cax.axhline(0., color='k')
            cax.axvline(0., color='k')
            cax.axvline(2., color='k',linestyle='--')
            cax.axvline(.5, color='k',linestyle=':')
            cax.axvline(1, color='k',linestyle=':')

        plt.tight_layout()
        plt.savefig(figures_dir + 'patterns_mels_ROI_{}.pdf'.format(cnd), orientation='portrait')