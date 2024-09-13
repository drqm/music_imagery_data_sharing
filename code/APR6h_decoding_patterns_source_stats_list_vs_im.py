#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Statistical comparison of listening and imagination, and of sound1 and sound3 in source space '''
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
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
src_sample = mne.read_source_spaces(subs_dir +
                                     '/fsaverage/bem/fsaverage-vol-5-src.fif')

print(src_sample)
stats_dir = wdir + 'results/stats/'
#Get subjects:
qr = Query(proj_name)
subjects = qr.get_subjects()

subs = range(11,91)#91) #, 27, 28, 29, 30, 31, 32, 33, 34, 35]
avg = False
periods = {'encoding': [0, 2],'delay': [2, 4], 'L1': [.2,.5], 'L3': [1.2,1.5]}#, 'retrieval': [4, 6]}
mode = 'patterns'
suffix = 'sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_localized'
suffix2 = ''
if len(argv) > 1:
    period = argv[1]

if len(argv) > 2:
    mode = argv[2]

subs = range(11,91)#91)#91) #, 27, 28, 29, 30, 31, 32, 33, 34, 35]
performance_exc = [55,58,60,73,76,82]
no_source_exc = [30,51,42]
noise_exc = [15]
no_data_exc = [32,33]

exclude = np.array(performance_exc + no_source_exc + noise_exc + no_data_exc)
subs = np.array([s for s in subs if s not in exclude])
print(subs.shape)
print('grand average analyses for lsitening vs imagining')

all_data = {}
scount = -1
for sidx,s in enumerate(subs):
    try:
        scode = subjects[s-1]
        morph = mne.read_source_morph(subs_dir + scode + '/bem/' + scode + '_vol-morph.h5')
        morph_mat = morph.vol_morph_mat
        dfname = op.join(data_dir,scode, scode + '_' + mode + '_' + suffix + '.p')
        print('loading file {}'.format(dfname))
        dfile = open(dfname,'rb')
        curdata = load(dfile)
        for cd in ['maintenance1','manipulation1']:
            times1, times2, times3, times4 = periods['encoding'], periods['delay'], periods['L1'], periods['L3']
            cdata_abs = (np.abs(morph_mat.dot(curdata[cd].copy().crop(times2[0],times2[1]).data)).mean(axis=1,keepdims=True) - 
                         np.abs(morph_mat.dot(curdata[cd].copy().crop(times1[0],times1[1]).data)).mean(axis=1,keepdims=True))
            cdata = (morph_mat.dot(curdata[cd].copy().crop(times2[0],times2[1]).data).mean(axis=1,keepdims=True) - 
                     morph_mat.dot(curdata[cd].copy().crop(times1[0],times1[1]).data).mean(axis=1,keepdims=True))#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
            cdata_s = (morph_mat.dot(curdata[cd].copy().crop(times4[0],times4[1]).data).mean(axis=1,keepdims=True) - 
                     morph_mat.dot(curdata[cd].copy().crop(times3[0],times3[1]).data).mean(axis=1,keepdims=True))#deepcopy(cmorphed.data) #morph_mat.dot(c.data)
       
            cd_name = cd + '_difference'
            cd_name_abs = cd + '_difference_abs'
            cd_name_s = cd + '_L3_L1'

            print('appending subject {} condition {}'.format(scode,cd_name))
            all_data.setdefault(cd_name,np.array([cdata]))
            all_data.setdefault(cd_name_abs,np.array([cdata_abs]))
            all_data.setdefault(cd_name_s,np.array([cdata_s]))
            #all_data[cd][cd2].append([cdata])
            if sidx > 0:
                all_data[cd_name] = np.vstack((all_data[cd_name],np.array([cdata])))
                all_data[cd_name_abs] = np.vstack((all_data[cd_name_abs],np.array([cdata_abs])))
                all_data[cd_name_s] = np.vstack((all_data[cd_name_s],np.array([cdata_s])))
        scount += 1
    except Exception as e:
        print(e)
        continue

all_data['interaction'] = all_data['manipulation1_difference'] - all_data['maintenance1_difference']
all_data['interaction_abs'] = all_data['manipulation1_difference_abs'] - all_data['maintenance1_difference_abs']   
all_data['interaction_s'] = all_data['manipulation1_L3_L1'] - all_data['maintenance1_L3_L1']
adjacency = mne.spatial_src_adjacency(src_sample)

### Stats on main comparisons
conds = [k for k in all_data]
#conds = ['maintenance','manipulation','melody','melody_maintenance','melody_manipulation','block','interaction']

#alphas = [.025,.025,.025,.025,.025,.025,.025]
stat_results = {}
for cidx, cnd in enumerate(conds):
    cdata = all_data[cnd]
    print(cdata.shape)
    #stat_results[cnd] = gs.do_stats(cdata, 'FDR', adjacency=adjacency, FDR_alpha=.025)
    stat_results[cnd] = gs.do_stats(cdata, 'montecarlo', adjacency=adjacency, FDR_alpha=.025,
                                    n_permutations=5000)
    print('reporting stats for {}:\n\n'.format(cnd))
    print('minimum pval = ', np.round(np.min(stat_results[cnd]['pvals'])))
#     print('minimum qval = ', np.round(np.min(stat_results[cnd]['qvals']),2))
    print('minimum tstat = ', np.round(np.min(stat_results[cnd]['tvals']),2))
    print('maximum tstat = ', np.round(np.max(stat_results[cnd]['tvals']),2))
    
print('saving stats results')
stats_fname = op.join(stats_dir,'{}_source_stats_list_vs_im.p'.format(mode))
sfile = open(stats_fname,'wb')
pickle.dump(stat_results,sfile)
sfile.close()
