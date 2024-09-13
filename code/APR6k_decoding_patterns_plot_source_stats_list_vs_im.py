''' plot statistical contrast between listening and imagination and sound 1 and sound 3 '''
import mne
#%gui qt
#import matplotlib
from nilearn import surface
from nilearn import plotting
fsaverage = datasets.fetch_surf_fsaverage()
fsaverage

#%matplotlib qt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stormdb.access import Query
from pickle import load
from scipy import stats as spstats
from mne.datasets import sample
from mne.stats import spatio_temporal_cluster_1samp_test
import os
import pickle
from copy import deepcopy
import warnings
from os import path as op
warnings.filterwarnings("ignore", category=DeprecationWarning)

proj_name = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
wdir = '/projects/' + proj_name + '/scratch/working_memory/'
stats_dir = wdir + 'results/stats/'
data_dir = wdir + 'averages/data/'
figures_dir = wdir + 'results/figures/'

subs_dir = '/projects/' + proj_name + '/scratch/fs_subjects_dir/'
sample_path = sample.data_path()
sample_subjects_dir = sample_path + '/subjects'
src_sample = mne.read_source_spaces(subs_dir +
                                '/fsaverage/bem/fsaverage-vol-5-src.fif')
label_file = sample_subjects_dir + '/fsaverage/mri/aparc+aseg.mgz'

# Read stats
periods = ['list_vs_im']
stats = {}

for p in periods:
    csfname = '{}patterns_source_stats_{}.p'.format(stats_dir, p)
    print(csfname)
    cfile = open(csfname,'rb')
    stats[p] = pickle.load(cfile)
    cfile.close()
#print(stats)

# Load and morph a source time course
dfname = data_dir + '0021_LZW/0021_LZW_patterns_sources_task_sensor_lf_0.05_hf_None_tstep_0.025_twin_0.05_localized.p'
dfile = open(dfname,'rb')
all_stc = pickle.load(dfile)
dfile.close()
stc = deepcopy(all_stc['maintenance1'])
del all_stc

# load source morph
morph = mne.read_source_morph(subs_dir + '0021_LZW/bem/0021_LZW_vol-morph.h5')
stc = morph.apply(stc)
stc = stc.mean()
print(stc)

def get_cluster_peaks(cstat,p_threshold=.05):
        sigix = np.where(cstat['pvals']<=p_threshold)[0]
        vx, vy, masks, p = [], [], [], []
        for six in sigix:
            cmask = cstat['clusters'][six].copy().astype(int).T
            cdata = cstat['tvals']*cmask
            cvx, cvy = np.unravel_index(np.abs(cdata).argmax(), cdata.shape)
            vx += [cvx]
            vy += [cvy]
            masks += [cmask]
            p += [cstat['pvals'][six]]
        vx = np.array(vx)
        vy = np.array(vy)
        return vx, vy, masks, p

cperiods = ['list_vs_im']

labels = mne.get_volume_labels_from_aseg(label_file)
label_sel = [l for l in labels if ('White' not in l) & ('choroid' not in l) & ('vessel' not in l) & ('callosum' not in l) &
              ('Vent' not in l) & ('wall' not in l) & ('Unknown' not in l) & ('unknown' not in l) & ('Cerebellum' not in l)
              & ('CC_' not in l) & ('CSF' not in l) & ('Optic-Chiasm' not in l) & ('WM-hypointensities' not in l)]
MA = {}

clust_report = {'period': [], 'condition': [], 'cluster No': [],
                 'x': [], 'y': [], 'z': [],'p-value': [],
                 'peak t-value': [], 'peak activation': []} 
for p in cperiods:
    print(p)
    MA[p] = {}
    conds = stats[p].keys()
    for c in conds:#stats[p]: 
        print(c)
        cstc = stc.copy()
        vx, vy, masks, pvals = get_cluster_peaks(stats[p][c])
        MA[p][c] = np.zeros((vx.shape[0],len(label_sel)))
        for v in range(vx.shape[0]):
            ctime = cstc.times[vy[v]]
            print('cluster',v+1,' - peak: ', np.round(ctime,3), 's')
            print(np.unique(masks[v]))
            cstc.data = stats[p][c]['tvals'].copy()*masks[v]
            csign = np.sign(cstc.data.mean())
            cstc.data = np.abs(cstc.data)
            pos, _ = cstc.get_peak()
            pix, _ = cstc.get_peak(vert_as_index=True)
            coords = np.round(src_sample[0]['rr'][pos]*1000)
            clust_report['period'] += [p]
            clust_report['condition'] += [c]
            clust_report['cluster No'] += [v+1]
            clust_report['x'] += [coords[0]]
            clust_report['y'] += [coords[1]]
            clust_report['z'] += [coords[2]]
            clust_report['p-value'] += [pvals[v]]
            clust_report['peak t-value'] += [np.round(cstc.data[pix,0]*csign,2)]
            clust_report['peak activation'] += [np.round(stats[p][c]['data_mean'][pix,0],2)]

            MA[p][c][v] = np.squeeze(mne.extract_label_time_course(cstc,[label_file,label_sel],
                                                                     src=src_sample,mode='max'))*csign
clust_report = pd.DataFrame(clust_report)         
print(clust_report)
clust_report.to_csv(op.join(stats_dir,'patterns_source_stats_report_list_vs_im.csv'),index=False)

### Get labels of cluster activations
hems = {'right': ['rh','Right'],'left': ['lh','Left']}
MA_df = {'cluster': [], 'label': [], 'hemisphere': [], 'tval': []}
for p in MA:
    for c in MA[p]:#stats[p]:
        for cl in range(MA[p][c].shape[0]):
            act_ix = np.argsort(-np.abs(MA[p][c][cl,:]))
            #print( '\n', p, c,' cluster ', cl + 1)
            for h in hems:
                #print(h)
                for ix in act_ix:
                    if (np.abs(MA[p][c][cl,ix]) >= 2) & (hems[h][0] in label_sel[ix] or hems[h][1] in label_sel[ix]): 
                        print(label_sel[ix],np.round(MA[p][c][cl,ix],2))
                        MA_df['cluster'] += ['{} {} cluster {}'.format(p, c, cl + 1)]
                        MA_df['label'] += [label_sel[ix]]
                        MA_df['hemisphere'] += [h]
                        MA_df['tval'] += [np.round(MA[p][c][cl,ix],2)]
                             
MA_df = pd.DataFrame(MA_df)
print(MA_df)     
MA_df.to_csv(op.join(stats_dir,'patterns_stats_report_aparc_list_vs_im.csv'),index=False)

# Make plots
conds = ['maintenance1_difference','manipulation1_difference','interaction']#'maintenance1','manipulation1',]
views = ['lateral','medial']
hemi = ['left','right']
surfs = [fsaverage.pial_left, fsaverage.pial_right]
surfs2 = [fsaverage.infl_left, fsaverage.infl_right]
bgs = [fsaverage.sulc_left, fsaverage.sulc_right]
export_data = True
for p in stats:
    conds = stats[p].keys()
    for c in conds:
        
        cstc = stc.copy()
        cstc.data = stats[p][c]['data_mean']*stats[p][c]['mask']
        cstc = cstc.copy().mean()
        img = cstc.as_volume(src_sample,  mri_resolution=False)
        
        cname = '{} {}'.format(p,c)
        print(cname)
        cfig, caxis = plt.subplots(nrows=2,ncols=2,subplot_kw={'projection': '3d'})
        surf_df = {'x_coord':[],'y_coord': [],'z_coord': [],'color_source_activation': []}
        for hix,h in enumerate(hemi):
            for vix, vi in enumerate(views):
                surf = surface.vol_to_surf(img, surfs[hix])
                plotting.plot_surf_stat_map(surfs2[hix],surf, hemi=h,axes=caxis[vix,hix],
                                                symmetric_cbar=True,view=vi, vmax=.8,threshold=.15,
                                                colorbar=True, bg_map=bgs[hix],darkness=1)#,#,
                                                # engine='matplotlib',bg_on_data=True)
            if export_data:
                mesh = surface.load_surf_mesh(surfs[hix])
                surf_df['x_coord'] += [mesh[0][:,0]]
                surf_df['y_coord'] += [mesh[0][:,1]]
                surf_df['z_coord'] += [mesh[0][:,2]]
                surf_df['color_source_activation'] += [surf[:,0]]
        cfig.savefig(figures_dir + 'patterns_sources_' + cname + '_list_vs_im_surface.pdf')
        plt.show()
        if export_data:
            print('exporting data')
            surf_df = pd.DataFrame({ck: np.concatenate(surf_df[ck]) for ck in surf_df})
            surf_df = surf_df.round(3)
            surf_df.to_csv(stats_dir + 'plot_data_patterns_sources_' + cname + f'_list_vs_im_surface.csv', index=False)
