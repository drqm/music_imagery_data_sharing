'''Script to submit to cluster statistics of decoding patterns sources'''
from stormdb.cluster import ClusterBatch
import os
project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne'
scriptdir = '/projects/{}/scripts/working_memory/'.format(project)
periods = ['L1','L2','L3','I1']
avg = [1,1,1,1,1]
modes = ['patterns','filters']
cb = ClusterBatch(project)
queue = 'all.q'
for m in modes:    
    for pix,p in enumerate(periods):
        cmd = 'python {}APR6g_decoding_patterns_source_stats.py {} {} {}'.format(scriptdir,p,m,avg[pix])
        cb.add_job(cmd=cmd, queue=queue,n_threads =12,cleanup = False)
cb.submit()

