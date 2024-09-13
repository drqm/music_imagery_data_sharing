import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ica

def default_events_fun(events):
    ''' Default event function. Returs same type of event as input '''
    event_id = {str(eo): eo for eo in np.unique(events[:,2])}
    return events, event_id

def WM_epoching(data_path, ica_path, tmin, tmax, l_freq=0.1, h_freq=None,
                resample = None, bads=[], reject = None, 
                baseline = None, notch_filter=None, 
                events_fun=default_events_fun, 
                events_fun_kwargs = {}):
    
    ''' This function takes raw data, prepocesses it and chunks it into epochs.
    
    Parameters
    ----------
    data_path: str
        Raw data file
    ica_path: str
        path of ICA solution for eog and ecg cleaning
    tmin, tmax: float 
        start and end epoch times around events
    l_freq, h_freq: None | float
        If float, cutoff for high-pass and low-pass filtering
    resample: None | float 
        If float, resampling frequency.
    bads: list
        list of bad channels to interpolate or exclude
    reject: dict
        Dictionary with thresholds for automatic trial rejection. See mne.Epochs for details.
    baseline: None | tuple
        If tuple, first and last timepoint of baseline period for baseline subtraction.
    notch_filter: None | float
        If float, frequency for notch filtering.
    events_fun: function
        function to map current events into new event defintions. The function must take an
        mne.Epochs.events array and output an mne.Epochs.events array and an mne.Epochs.event_id dict.
    events_fun_kwargs: **dict
        kwargs dict for events_fun

    Returns
    ----------
    epochs: mne.Epochs
    '''
    
    raw = mne.io.read_raw_fif(data_path, preload = True) # load data
    ica = mne.preprocessing.read_ica(ica_path) # load ica solution
    raw = ica.apply(raw) # apply ICA
    raw.pick_types(meg = True, stim = True, eog = False) # select MEG channels
    raw.info['bads'] += bads # Add bad channels
    
    if notch_filter: #notch_filter for power line
        raw.notch_filter(freqs=notch_filter)
        
    raw.filter(l_freq=l_freq,h_freq=h_freq) # further filtering
    
    if resample: #resample
        raw.resample(sfreq=resample)
        
    events = mne.find_events(raw, shortest_event = 1) #get events from raw data
    events = events[np.append(np.diff(events[:,0]) > 1,True)] # delete spurious triggers
    events, event_id = events_fun(events, **events_fun_kwargs) # apply custom events function
    
    picks = mne.pick_types(raw.info, meg = True) #select MEG channels
    # Do epoching
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        preload=True, baseline=baseline, picks=picks,on_missing='warn')
    return epochs

def main_task_events_fun(events, cond, lfname):
    ''' This function recodes events for the main task in the experiment

    Parameters
    ----------
    events: ndarray 
        Array with mne.Epochs.events
    cond: str
        condition to process. Either 'maintenance' (recall) or 'manipulation'
    lfname: str, tmax: float 
        Path to experiment log file. If provided, excludes incorrect trials.
    
    Returns
    ----------
    events: mne.Epochs.events
    event_id: mne.Epochs.event_id
    '''
    ## Custom events mapping
    mappings = {'maintenance': {'maintenance/mel1/same':   {'prev': [11,111,122], 'new': 11}, 
                            'maintenance/mel1/diff1':  {'prev': [11,213,222], 'new': 12},
                            'maintenance/mel1/diff2':  {'prev': [11,211,223], 'new': 13},
                            'maintenance/mel2/same':   {'prev': [13,113,122], 'new': 21},
                            'maintenance/mel2/diff1':  {'prev': [13,211,222], 'new': 22},
                            'maintenance/mel2/diff2':  {'prev': [13,213,221], 'new': 23}},

                'manipulation':{'manipulation/mel1/inv':    {'prev': [11,113,122], 'new': 31}, 
                            'manipulation/mel1/other1': {'prev': [11,211,222], 'new': 32},
                            'manipulation/mel1/other2': {'prev': [11,213,221], 'new': 33},
                            'manipulation/mel2/inv':    {'prev': [13,111,122], 'new': 41},
                            'manipulation/mel2/other1': {'prev': [13,213,222], 'new': 42},
                            'manipulation/mel2/other2': {'prev': [13,211,223], 'new': 43}}
           }

    events2 = events.copy()[events[:,2] < 20] # Get first tone of first melody
    events3 = events.copy()[np.isin(events[:,2]//10, [11,21]),:] # get first tone of target melody
    events4 = events.copy()[np.isin(events[:,2]//10, [12,22]),:] # get second tone of target melody
    new_events = events2
    for eid in mappings[cond]:
        p1,p4,p5 = mappings[cond][eid]['prev']
        idx = [x and y and z for x,y,z in zip(events2[:,2]==p1,events3[:,2]==p4,events4[:,2]==p5)]
        new_events[idx,2] = mappings[cond][eid]['new']
    if lfname:
        print('rejecting incorrect trials')
        ldf = pd.read_csv(lfname, sep = ',', header=0)
        ldf['acc'] = ldf['type'] == ldf['response']
        new_events = new_events[ldf['acc'], :]
        
    event_id = {ccc: mappings[cond][ccc]['new'] for ccc in mappings[cond]}
    return new_events, event_id

def main_task_decoding_events_fun(events, lfname=None):
    ''' This function recodes events for the main task in the experiment.
        Event labels are optimized for binary decoding.

    Parameters
    ----------
    events: ndarray 
        Array with mne.Epochs.events
    lfname: str, tmax: float 
        Path to experiment log file. If provided, excludes incorrect trials.
    
    Returns
    ----------
    events: mne.Epochs.events
    event_id: mne.Epochs.event_id
    '''
    events2 = events.copy()[events[:,2] < 20]
    events2[:,2] = [(ev-10)%3 * -1 + 1 for ev in events2[:,2]]
  
    if lfname:
        print('rejecting incorrect trials')
        ldf = pd.read_csv(lfname, sep = ',', header=0)
        ldf['acc'] = ldf['type'] == ldf['response']
        events2 = events2[ldf['acc'], :]

    event_id = {'melody' + str(ii): ii for ii in np.unique(events2[:,2])}
    return events2, event_id