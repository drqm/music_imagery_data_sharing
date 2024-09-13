''' Script to run and plot stats on sensor-level decoding patterns.
Cluster based permutations in space and time are used.'''

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
import sys
sys.path.append('/projects/' + proj_name + '/scripts/working_memory/')

import mne
import numpy as np
from matplotlib import pyplot as plt
from stormdb.access import Query
from pickle import load
from scipy import stats
from mne.datasets import sample
from mne.stats import spatio_temporal_cluster_1samp_test
import os
import pandas as pd
import os.path as op
import pickle
from copy import deepcopy
from sys import argv
import warnings
import src.group_stats as gs
import importlib

importlib.reload(gs)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set relevant variables
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

wdir = '/projects/' + proj_name + '/scratch/working_memory/'
data_dir = wdir + 'averages/data/'
subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'

stats_dir = wdir + 'results/stats/'
figures_dir = wdir + 'results/figures/'

#Get subjects:
qr = Query(proj_name)
sub_codes = qr.get_subjects()

## load data
suffix = 'task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05'

# Manage subjects and exclusions
sub_Ns = np.arange(11,91) 
performance_exc = [55,58,60,73,76,82]
no_source_exc = [30,51,42]
noise_exc = [15]
no_data_exc = [32,33]

exclude = np.array(performance_exc + noise_exc + no_data_exc)
sub_Ns = np.array([s for s in sub_Ns if s not in exclude])

gdata = {}
garray = {}
scount = 0
for sub in sub_Ns: #loop over subjects
    sub_code = sub_codes[sub-1]
    try:
        print('loading subject {}'.format(sub_code))
        
        # Load decoding patterns
        evkd_fname = op.join(data_dir,sub_code,sub_code + '_patterns_' + suffix + '.p')
        evkd_file = open(evkd_fname,'rb')
        evokeds = pickle.load(evkd_file)
        evkd_file.close()
        
        scount = scount +1
        
        #Loop over conditions and store
        for e in evokeds:
            if isinstance(evokeds[e], dict):
                cconds = {e + b: evokeds[e][b] for b in evokeds[e]}
            else:
                cconds = {e: evokeds[e]}
                
            for condname in cconds:
                if scount == 1:
                    gdata[condname] = []
                    garray[condname] = []
                    times_fname = op.join(data_dir,sub_code,sub_code + '_times_' + suffix + '.p')
                    times_file = open(times_fname,'rb')
                    times = pickle.load(times_file)
                    times_file.close()
                cconds[condname].times = times[condname[0:-1]]
                gdata[condname].append(cconds[condname].data)
                garray[condname].append(cconds[condname])
    except Exception as ex:
        print('could not load subject {}'.format(sub_code))
        print(ex)
        continue

# Transform into numpy arrays
for g in gdata:
    gdata[g] = np.array(gdata[g])

# Compute interaction conditions (difference of differences)
garray['interaction1'] = [mne.combine_evoked([garray['manipulation1'][cs],
                                              garray['maintenance1'][cs]],
                                              weights=[1,-1]) for cs,_ in enumerate(garray['maintenance1'])]

for gdix,gd in enumerate(garray['interaction1']):
    garray['interaction1'][gdix].comment = 'interaction'

gdata['interaction1'] = gdata['manipulation1'] - gdata['maintenance1']
times['interaction'] = times['maintenance']

# Compute imagine minus listen for both conditions
lix = (times['maintenance'] >= 0) & (times['maintenance'] <= 2)
iix = (times['maintenance'] >= 2) & (times['maintenance'] <= 4)
gdata['list_vs_im_maintenance'] = gdata['maintenance1'][:,:,iix].mean(axis = 2,keepdims=True) - gdata['maintenance1'][:,:,lix].mean(axis = 2,keepdims=True) 
gdata['list_vs_im_manipulation'] = gdata['manipulation1'][:,:,iix].mean(axis = 2,keepdims=True)  - gdata['manipulation1'][:,:,lix].mean(axis = 2,keepdims=True) 

# Compute sound3 minus sound1 for both conditions
fix = (times['maintenance'] >= .2) & (times['maintenance'] <= .5)
thix = (times['maintenance'] >= 1.2) & (times['maintenance'] <= 1.5)
gdata['1st_vs_3rd_maintenance'] = gdata['maintenance1'][:,:,thix].mean(axis = 2,keepdims=True) - gdata['maintenance1'][:,:,fix].mean(axis = 2,keepdims=True) 
gdata['1st_vs_3rd_manipulation'] = gdata['manipulation1'][:,:,thix].mean(axis = 2,keepdims=True)  - gdata['manipulation1'][:,:,fix].mean(axis = 2,keepdims=True) 

# Compute grand averages 
grand_avg = {}
for e in garray:
    grand_avg[e] = mne.grand_average(garray[e])
    grand_avg[e].data = np.mean(np.array(gdata[e]),0)
    grand_avg[e].comment = garray[e][0].comment
    grand_avg[e].info['srate'] = 1/.025
    grand_avg[e].times = times[e[0:-1]]

# Compute cluster-based stats on wanted conditions (over space and time)
conds = ['maintenance1','manipulation1','interaction1',
        'list_vs_im_maintenance','list_vs_im_manipulation',
        '1st_vs_3rd_maintenance','1st_vs_3rd_manipulation']
stats_results = {}

ch_type = ['mag','grad'] # channel types to loop over
for ct in ch_type:
    stats_results[ct] = {} # Initialize
    chidx = np.array(mne.pick_types(garray['interaction1'][0].info, meg = ct)) # Select channel type
    adjacency,_ = mne.channels.find_ch_adjacency(garray['interaction1'][0].info,ch_type=ct) # compute adjacency for cluster analyses
    
    # Loop over conditions and do stats
    for c in conds:
        X = gdata[c][:,chidx,:]
        stats_results[ct][c] = gs.do_stats(X, method='montecarlo', adjacency=adjacency, FDR_alpha=.025, h0=0,
                                       cluster_alpha = .05, p_threshold = .05, n_permutations=5000)

# Save stat results
with open(stats_dir + 'sensor_stats_' + suffix + '.p','wb') as cfile:
    pickle.dump(stats_results,cfile)

# Plot stat results
periods = {
           'L1': [.2,.5],
           'L2': [.7,1.],
           'L3': [1.2,1.5],
           'I1': [2,4],
           }

ch_type = ['mag','grad']
vlims = {'mag': [-100,100],'grad': [0,25]}
conds = ['maintenance1','manipulation1','interaction1']
export_data = True # whether to export data for publication figures

for cht in ch_type:
    print('\n############### {} ################\n'.format(cht))
    for p in periods:
        print('############# {} ################\n'.format(p))
        for c in conds:#stats_results[cht]:
            print('############# {} ################\n'.format(c))

            cERF = grand_avg[c].copy() # Get mean of data
            cERF = cERF.pick_types(meg=cht) # Select channel type
            tmin, tmax = periods[p] # Period to plot
            tidx = np.where([x and y for x,y in zip(cERF.times >= tmin, cERF.times <= tmax)])[0]
            
            cdata = stats_results[cht][c]['data_mean'].copy()
            cdata = cdata[:,tidx].mean(axis=1,keepdims=True)# retrieve data and average across period
            cERF.times=np.array([0])
            
            # Get mask for significant time points
            cmask = stats_results[cht][c]['mask'].copy()[:,tidx].astype(int).mean(axis=1,keepdims=True) > 0
            cERF.data = cdata
            
            # Make and save plot
            cfig = cERF.plot_topomap(mask=cmask, times = 0, vmin = vlims[cht][0], vmax = vlims[cht][1], sensors=False)
            cfig.savefig('{}patterns_sensor_{}_{}_topoplot_{}.pdf'.format(figures_dir,c,p,cht))
            plt.show()

            # Export data if required
            if export_data:
                topo_df = {'channel_name': [], 'x_coord': [], 'y_coord': [], 'z_coord': [], 'color_ERF': [], 'marker_significance': []}
                if cht == 'grad':
                    topo_df['planar1'] = []
                    topo_df['planar2'] = []
                    for chix in np.arange(0,len(cERF.ch_names),2):
                        topo_df['channel_name'] += [cERF.ch_names[chix] + '_' + cERF.ch_names[chix+1]]
                        topo_df['x_coord'] += [np.round(cERF.info['chs'][chix]['loc'][0],3)]
                        topo_df['y_coord'] += [np.round(cERF.info['chs'][chix]['loc'][1],3)]
                        topo_df['z_coord'] += [np.round(cERF.info['chs'][chix]['loc'][2],3)]
                        topo_df['planar1'] += [cERF.data[chix,0]*1e14]
                        topo_df['planar2'] += [cERF.data[chix+1,0]*1e14]
                        topo_df['color_ERF'] += [np.sqrt((cERF.data[chix,0]**2 + cERF.data[chix+1,0]**2)/2)*1e14]
                        topo_df['marker_significance'] += [int(int(cmask[chix,0] + cmask[chix,0])>0)]
                else:
                    for chix in np.arange(0,len(cERF.ch_names),1):
                        topo_df['channel_name'] += [cERF.ch_names[chix]]
                        topo_df['x_coord'] += [np.round(cERF.info['chs'][chix]['loc'][0],3)]
                        topo_df['y_coord'] += [np.round(cERF.info['chs'][chix]['loc'][1],3)]
                        topo_df['z_coord'] += [np.round(cERF.info['chs'][chix]['loc'][2],3)]
                        topo_df['color_ERF'] += [cERF.data[chix,0]*1e16]
                        topo_df['marker_significance'] += [int(cmask[chix])]

                topo_df = pd.DataFrame(topo_df).round(3)
                topo_df.to_csv('{}plot_data_patterns_sensor_{}_{}_topoplot_{}.csv'.format(stats_dir,c,p,cht),index=False)

plt.close('all')

