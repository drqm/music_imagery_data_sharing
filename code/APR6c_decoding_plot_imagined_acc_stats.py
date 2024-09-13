''' Script to do stats on decoding accuracy data and make plots '''

# Set directories
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
scripts_dir = project_dir + '/scripts/working_memory/'
import sys
sys.path.append(scripts_dir)

import mne
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from stormdb.access import Query
from pickle import load
from scipy import stats
from mne.stats import spatio_temporal_cluster_1samp_test
import os
from os import path as op
import pickle
from copy import deepcopy
import warnings

# custom libraries
import src.group_stats as gs 
import src.decoding_functions as df
import importlib

importlib.reload(gs)
importlib.reload(df)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Set directories
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
project_dir = '/projects/' + project
stats_dir = project_dir + '/scratch/working_memory/results/stats/'
figures_dir = project_dir + '/scratch/working_memory/results/figures/'
dem_file = project_dir + '/misc/WM_demographics.csv'
acc_file = project_dir + '/misc/WM_accuracies.csv'

# Define subjects and exclusions
subs = np.arange(11,91)
performance_exc = [55,58,60,73,76,82]
no_source_exc = [30,51,42]
noise_exc = [15]
no_data_exc = [32,33]
exclude = np.array(performance_exc + noise_exc + no_data_exc)

block = 'task'
mtype = 'sensor'
suffix = block + '_' + mtype + '_lf_0.05_hf_None_tstep_0.025_twin_0.05'

# Load data
sdata, scodes, times = gs.load_scores(suffix, subs, exclude)

# Calculate difference conditions
subs = [['maintenance_from_maintenance','manipulation_from_manipulation'],
        ['maintenance_from_maintenance','maintenance_from_manipulation'],
        ['manipulation_from_manipulation','manipulation_from_maintenance']]

diffs = {}
for cs in subs:
    diffs[cs[1] + '_' + cs[0]] = sdata[cs[1]] - sdata[cs[0]]

# Get mean and confidence intervals
smean, sstd, sci_lower, sci_upper, smedian, siqr_lower, siqr_upper = gs.grand_avg_scores(sdata)
dmean, dstd, dsci_lower, dsci_upper, dsmedian, dsiqr_lower, dsiqr_upper = gs.grand_avg_scores(diffs)
CI = {ss: {'lower': sci_lower[ss], 'upper': sci_upper[ss]} for ss in sci_lower}
dCI = {ss: {'lower': dsci_lower[ss], 'upper': dsci_upper[ss]} for ss in dsci_lower}

# Cluster stats on main conditions (null = 0.5 = chance level)
cluster_stats = {}
for s in sdata:
    print('doing stats for {}'.format(s))
    cluster_stats[s] = gs.do_stats(sdata[s],method='montecarlo',cluster_method = 'normal',h0=1/2,n_permutations=5000)

# Cluster stats on contrast conditions (null=0)
dcluster_stats = {}
for s in diffs:
    print('doing stats for {}'.format(s))
    dcluster_stats[s] = gs.do_stats(diffs[s],method='montecarlo',cluster_method = 'normal',h0=0,n_permutations=5000)

# Get masks for significant clusters
cluster_masks = {sm: cluster_stats[sm]['mask'] for sm in cluster_stats}
dcluster_masks = {sm: dcluster_stats[sm]['mask'] for sm in dcluster_stats}

with open(stats_dir + 'accuracy_stats_main.p','wb') as cfile:
    pickle.dump(cluster_stats,cfile)

with open(stats_dir + 'accuracy_stats_contrasts.p','wb') as dfile:
    pickle.dump(dcluster_stats,dfile)

# Function to get peak of each cluster
def get_cluster_peaks(cstat,p_threshold=.05,diagonal=False):
        sigix = np.where(cstat['pvals'] <= p_threshold)[0]
        vx, vy, masks, p = [], [], [], []
        for six in sigix:
            cmask = cstat['clusters'][six].copy().astype(int).T
            cdata = cstat['data_mean']*cmask
            if diagonal:
                cdata = np.expand_dims(np.diagonal(cdata),axis=0)
            cvx, cvy  = np.unravel_index(np.abs(cdata).argmax(), cdata.shape)
            vx += [cvx]
            vy += [cvy]
            masks += [cmask]
            p += [cstat['pvals'][six]]
        vx = np.array(vx)
        vy = np.array(vy)
        p = np.array(p)
        return vx, vy, masks, p

# Generate statistical report
stats_report = {'comparison': [],'train_peak': [],'test_peak': [],'cluster_pval': [],'peak_tval': [], 'peak_val': []}
results = [cluster_stats, dcluster_stats]
for r in results:
    for c in r:
        vx,vy,dmask,p = get_cluster_peaks(r[c],diagonal=False)
        #print(c)
        for cpix, cp in enumerate(p):
            #cldir = np.sign(np.mean(cluster_stats[c]['tvals']*dmask[cpix]))
            tpeak = r[c]['tvals'][vx[cpix],vy[cpix]]
            vpeak = r[c]['data_mean'][vx[cpix],vy[cpix]]
            test_time, train_time = np.round(times['maintenance'][np.array([vy[cpix],vx[cpix]])],2)
            stats_report['comparison']+=[c]
            stats_report['train_peak']+=[train_time]
            stats_report['test_peak']+=[test_time]
            stats_report['cluster_pval']+=[np.round(cp,3)]
            stats_report['peak_tval']+=[np.round(tpeak,3)]
            stats_report['peak_val']+=[np.round(vpeak,3)]

stats_report = pd.DataFrame(stats_report)
print(stats_report)

# Plot time generalized matrices on main stats
df.plot_time_gen_accuracy(smean, times,  masks = cluster_masks,
                          nrows=2, ncols=2, vlines=[2], hlines=[2],
                          savefig = op.join(figures_dir,
                                            'accuracies_imagined_' +
                                             suffix +'_all_cluster.pdf'),
                          vmin=.4,vmax=.6, export_data=stats_dir)

# Plot time generalized matrices on contrast stats
df.plot_time_gen_accuracy(dmean, times,  masks = dcluster_masks,
                          nrows=2, ncols=2, vlines=[2], hlines=[2],
                          savefig = op.join(figures_dir,
                                            'accuracies_imagined_' +
                                             suffix +'_diff_cluster.pdf'),
                          vmin=-.1,vmax=.1, export_data=stats_dir)

# Plot diagonals on main stats
df.plot_diagonal_accuracy(smean, times,  masks = cluster_masks, CI=CI,
                          nrows=2, ncols=2, vlines=[2], hlines=[],
                          savefig = op.join(figures_dir,
                                            'accuracies_imagined_' +
                                             suffix +'_all_diagonal_cluster.pdf'),
                          ylims=(.45,.65), chance=1/2)

# Plot diagonals on contrast stats
df.plot_diagonal_accuracy(dmean, times,  masks = dcluster_masks, CI=dCI,
                          nrows=2, ncols=2, vlines=[2], hlines=[0],
                          savefig = op.join(figures_dir,
                                            'accuracies_imagined_' +
                                             suffix +'_diff_diagonal_cluster.pdf'),
                          ylims=(-.1,.1), chance=0)

#### Correlations with demographic info
# load demographics
dem = pd.read_csv(dem_file)
dem = dem[np.isin(dem['Subject'],scodes)]
print(dem)

# Load accuracies
acc = pd.read_csv(acc_file)
acc = acc[np.isin(acc['subject'],scodes)]
print(acc)

dem['acc_recall'] =  np.round(np.array(acc['accuracy'][acc['block']=='recall']),3)
dem['acc_manipulation'] = np.round(np.array(acc['accuracy'][acc['block']=='manipulation']),3)
dem['acc_all'] = np.nanmean(np.array([dem['acc_recall'],dem['acc_manipulation']]),axis=0)
print(dem)

# Function to correlate neural decoding (neural discrimination) with different demographic variables
def correlate_acc(sdata,dem,yvar,xvar,conds,periods):

    abs_cors, pvals = {}, {}
    print(f'\n\n##### correlation between {yvar} and {xvar} #######\n')

    for ccond in conds:
        c = ccond.split('_')[0]
        c2 = conds[ccond]

        if yvar == 'acc':
            cyvar = yvar + '_' + c2
        else:
            cyvar = yvar
        abs_cors[c2], pvals[c2] = {},{}

        for p in periods:
            if xvar == 'neuracc':
                cxvar = c2 + '_neuracc_' + p
            else:
                cxvar = xvar

            idx1 = (times[c] >= periods[p][0]) & (times[c] < periods[p][1])
            dem[c2 + '_neuracc_' + p] = np.round(np.mean(np.abs(sdata[ccond][:,idx1,:].copy()[:,:,idx1]-.5), axis=(1,2)),3)
            nanix = (np.isnan(np.array(dem[cyvar])) == False) & (np.isnan(np.array(dem[cxvar])) == False)
            abs_cors[c2][p], pvals[c2][p] = stats.pearsonr(dem[cxvar][nanix], np.array(dem[cyvar][nanix]))
            print('\n',c2, p)
            print('r = ', np.round(abs_cors[c2][p],2),' pval = ',np.round(pvals[c2][p],3))

    return abs_cors, pvals

# Run correlations
periods = {'listening': [0,2],'imagination': [2,4]}#, 'all': [0,4]}
conds = {'maintenance_from_maintenance': 'recall','manipulation_from_manipulation': 'manipulation'}
yvars = ['acc','vividness','TrainingGMSI']
xvar = 'neuracc'
cors,pvals = {},{}
for yv in yvars:
    cors[yv],pvals[yv] = correlate_acc(sdata,dem,yv,xvar,conds,periods)

# FDR correction
labels, qvals = [],[]
for yv in cors:
    for c2 in cors[yv]:
        for p in cors[yv][c2]:
            labels += [yv + ' ' + c2 + ' ' + p]
            qvals += [pvals[yv][c2][p]]
_,qvals = mne.stats.fdr_correction(qvals)
for cl, cq in zip(labels,list(qvals)):
    print(cl,np.round(cq,3))