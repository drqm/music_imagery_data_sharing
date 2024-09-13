#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Script to submit to cluster the estimation of inverse solution '''
import os
from warnings import filterwarnings
from stormdb.cluster import ClusterBatch
from stormdb.access import Query

project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
os.environ['MINDLABPROJ']=project
os.environ['MNE_ROOT']='~/miniconda3/envs/mne' # for surfer
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'

script_dir = '/projects/{}/scripts/working_memory/'.format(project)
blocks = ['task'] #'localizer'
subNs = range(11,91)
ch_types = ['grad','mag','all']
cb = ClusterBatch(project)
for ch in ch_types:
    for b in blocks:
    # Loop over subjects:
        for sub in subNs:
            submit_cmd_base = 'python {}APR6e_decoding_invert_patterns.py {} {} {}'.format(script_dir,b,sub,ch)
            cb.add_job(cmd=submit_cmd_base, queue='short.q',n_threads = 1, cleanup = False)
cb.submit()
