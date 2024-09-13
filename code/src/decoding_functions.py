import numpy as np
import os
import os.path as op
import mne
import pandas as pd
import matplotlib.pyplot as plt
from pickle import dump
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA    
from mne.decoding import GeneralizingEstimator, get_coef, LinearModel,cross_val_multiscore,SlidingEstimator

def smooth_data(data, tstart, tstep, twin, Fs, taxis=2):

    """This function takes neural data and smooths them over time according to 
        time step size and time window size
    
    Parameters
    ----------
    data: ndarray
        Array of any dimension, usually ntrials x nchannels x ntimes
    tstart: float
        time of start for smoothing in seconds
    tstep: float
        time step for smoothing in seconds (e.g. 0.025s corresponding to output sfreq of 40Hz)
    twindow: float
        window for smoothing, e.g., 0.05s (+/- 0.025)
    Fs: float 
        original sampling frequency
    taxis: int
        axis of temporal dimension (typically the last)

    Returns
    ----------
    new_data: ndarray
        Smoothed data
    new_time: 1d ndarray 
        New time vector
    """

    # get data shape
    old_dims = data.shape
    
    # Arrange dimensions in standard form
    new_dimord = np.array([taxis] + [d for d in range(len(old_dims)) if d != taxis])
    old_dimord = np.argsort(new_dimord)
    data = np.transpose(data,new_dimord)
    new_dims = data.shape
    
    # Calculate old and new time vectors
    tend = tstart + new_dims[0] / Fs
    ctime = np.arange(tstart, tend + 1/Fs, 1/Fs)
    ntime = np.arange(tstart + twin/2, tend-twin/2 + 1/Fs, tstep)
    
    # Initialize output data
    new_data = np.ones((tuple([len(ntime)]) + new_dims[1:])) * np.nan
    
    # Loop over timesteps and smooth
    for ntix, nt in enumerate(ntime):
        lims = np.array([nt - twin / 2, nt + twin / 2]) # Current interval to average
        cix = [np.argmin(np.abs(l - ctime)) for l in lims] # Limit indices
        new_data[ntix] = np.mean(data[cix[0]:(cix[1]+1)],0) # Average interval and store
    
    # Reorder dimesions and return
    new_data = np.transpose(new_data, old_dimord)
    return new_data, ntime

def WM_time_gen_classify(epochs, mode='sensor', kind = 'Generalizing', inv = None, lmask=[], score = True, # n_features = 'all',
                         twindows = None, l_freq=None, h_freq =None, smooth=None, save_filters=None,ncomps=None,
                         save_scores = None, save_gen=None, save_patterns=None, save_times=None, save_comps = None,
                         penalty='l2',scoring_output = 'mean_acc', drop_incorrect=None):
        
        """This function takes a dictionary of MNE epochs arrays and performs sliding or generalized
         binary or multiclass classification on them. Training and testing is done within (cross-validated) 
         and between all arrays in the dictionary. mne.Epochs.events arrays are taken as different classes.
    
        Parameters
        ----------
        epochs: dict
            Dictionary containing mne.Epochs objects. Each dictionary key contains a different 
            condition or dataset.
        mode: string 
            decoding over sensors or over sources. Allowed: 'sensors', 'sources'.
        kind: string
            Type of neural decoder. 'Generalizing' or 'Sliding'.
        inv: mne inverse operator
            Inverse solution to project epochs into source space. Required if mode == 'source'.
        lmask: list of ndarray
            Arrays of 1s and 0s indicating which source vertices or regions of
            interest to include in the decoding.
        score: bool 
            Whether to get decoding accuracies.
        twindows: dict
            Optional. If provided, crops the epochs to a desired time window. Each
            dictionary entry is indexed by a key with the same name as in
            the epochs dictionary and contains a list with start and an end time.
        l_freq, h_freq: float
            Optional. low and high cutoffs for high and low pass filtering
        smooth: None | **dict
            Optional. If **kwargs, smooths data with parameters provided, e.g., **{'tstep': .025, 'twin': .05 } 
        save_filters: None | str
            Optional. If str, saves the model coefficients to str
        ncomps: int
            Optional. If provided, performs PCA before decoding using the specified number 
            of components
        save_filters: None | str
            Optional. If str, saves the decoding accuracies to str
        save_gen: None | str
            Optional. If str, saves the estimated model to str
        save_patterns: None | str
            Optional. If str, saves the decoding patterns to str
        save_times: None | str
            Optional. If str, saves the time vector to str
        save_comps: None | str
            Optional. If str, saves the PCA components to str
        penalty: str
            Type of regularization. 'l1' or 'l2'.
        scoring_output: str
            whether to output mean accuracy ('mean_acc') or single trial predicted labels ('trial_pred')
        drop_incorrect: None | bool
            If True, drops incorrect trials
        
        Returns
        ----------
        gen: dict
            each entry is an estimated decoder
        patterns: dict
            each entry is an epochs array containing decoding patterns
        filters: dict
            each entry is an epochs array containing decoding coefficients
        scores: dict
            each entry is a ndarray of decoding accuracies (1d for sliding, 2d for generalizing)
        times: dict
            each entry is a time vector
        comps: dict
            each entry is an epochs array with PCA components
        """
        ###### Select correct trials only
        
        ## Prepare the data and get timepoints
        if isinstance(epochs, mne.Epochs):
            epochs = {'epochs': epochs}
        
        if isinstance(twindows, list):
            cwin = twindows.copy()
            twindows = {e: cwin for e in epochs} 
        
        if twindows: 
            for e in epochs:
                epochs[e].crop(twindows[e][0], twindows[e][1])   
        
        # Get time vector per condition
        times = {e: epochs[e].times for e in epochs}
        
        # Get data depending on mode (source vs sensor)
        data, labels, containers, comps = {}, {}, {}, {}
        for e in epochs:
            if mode == 'source':
                csource = mne.beamformer.apply_lcmv_epochs(epochs[e],inv)
                containers[e] = csource[0].copy()
                data[e] = np.array([cs.data for cs in csource])
                if len(lmask)>0:
                    data[e] = data[e] * lmask
                    
            elif mode == 'sensor':
                data[e] = epochs[e].get_data()
                containers[e] = epochs[e].average().copy()

            labels[e] = epochs[e].events[:,2]
            comps[e] = []

            if ncomps:
                print('\n### PCA decomposition ####\n')
                pca = PCA(n_components=ncomps)
                cedata = data[e].copy().transpose([1,0,2])
                y,x,z = cedata.shape
                cedata = cedata.reshape((cedata.shape[0],-1)).T
                pdata = pca.fit(cedata).transform(cedata)
                
                ## Reshape back:
                pdata = pdata.reshape([x,z,ncomps]).transpose([0,2,1])
                data[e] = pdata
                comps[e] = pca.components_

        # Perform filtering if required
        if l_freq or h_freq:
            for e in data:
                data[e] = mne.filter.filter_data(data[e], epochs[e].info['sfreq'],
                                                 l_freq, h_freq, n_jobs = 8)
                    
        # Perform smoothing if requried
        if smooth:
            for e in data:
                data[e], times[e] = smooth_data(data[e], tstart = times[e][0],
                                                **smooth, Fs = epochs[e].info['sfreq'])
                
        # Initialize output:
        gen, patterns, filters, scores = {}, {}, {}, {}
        
        # Start classifier object:
        clf = make_pipeline(StandardScaler(),#SelectKBest(f_classif, k=n_features),
                            LinearModel(LogisticRegression(penalty=penalty,solver='liblinear')))
        
        # Loop over conditions
        for e in data:
            # Start and fit estimator
            print('fitting ', e)
             
            if kind == 'Generalizing':
                gen[e] = GeneralizingEstimator(clf, n_jobs=2, scoring = 'balanced_accuracy', verbose = True)
            elif kind == 'Sliding':
                gen[e] = SlidingEstimator(clf, n_jobs=2, scoring = 'balanced_accuracy', verbose = True)
            
            caeix = np.arange(data[e].shape[0])
            if drop_incorrect:
                print('dropping incorrect trials')
                caeix = np.array([a for a in caeix if a not in drop_incorrect[e]])
            print(caeix)
            gen[e].fit(X = data[e][caeix], y = labels[e][caeix])   
                
            ## Get patterns and filters
            print('extractig patterns and filters')
            cpatterns = get_coef(gen[e],'patterns_',inverse_transform=True)
            cfilters = get_coef(gen[e],'filters_', inverse_transform=True)

            # If only two classes (one set of patterns), expand dimensions for coherence with loops below:
            if len(cpatterns.shape) < 3:
                cpatterns = np.expand_dims(cpatterns,1)
                cfilters = np.expand_dims(cfilters,1)

            # Export patterns and filters to evoked, sources or components
            if (mode == 'source') & (not ncomps):
                # Loop over classes:
                for cl in range(cpatterns.shape[1]):
                    cname = e + str(cl+1)
                    patterns[cname] = containers[e]
                    patterns[cname].tstart = times[e][0]
                    patterns[cname].tstep = np.diff(times[e])[0]
                    filters[cname] = patterns[cname].copy()
                    patterns[cname].data = cpatterns[:,cl,:].copy()
                    filters[cname].data = cfilters[:,cl,:].copy()
                
            elif (mode == 'sensor') & (not ncomps):
                cinfo = containers[e].info.copy()
                cinfo['sfreq'] = 1 / np.diff(times[e])[0]
                # Loop over classes:
                for cl in range(cpatterns.shape[1]):
                    cname = e + str(cl+1)
                    patterns[cname] = mne.EvokedArray(cpatterns[:,cl,:].copy(), info = cinfo, baseline = None,
                                                      tmin=times[e][0], comment=e)
                    filters[cname] = patterns[cname].copy()
                    filters[cname].data = cfilters[:,cl,:].copy()
            
            elif ncomps:
                csrate = 1 / np.diff(times[e])[0]
                ch_names = ['ch' + str(ccc + 1) for ccc in range(ncomps)]
                cinfo = mne.create_info(ch_names, csrate)
                # Loop over classes:
                for cl in range(cpatterns.shape[1]):
                    cname = e + str(cl+1)
                    patterns[cname] = mne.EvokedArray(cpatterns[:,cl,:].copy(), info = cinfo, baseline = None,
                                                      tmin=times[e][0], comment=e)
                    filters[cname] = patterns[cname].copy()
                    filters[cname].data = cfilters[:,cl,:].copy()

            # Obtain scores looping over conditions to test:
            for e2 in data:
                scond = e2 + '_from_' + e
                print('scoring ', scond)
                if e != e2:
                    # If different conditions, perform test
                    if scoring_output == 'mean_acc':
                        scores[scond] = gen[e].score(data[e2], labels[e2])
                    elif scoring_output == 'trial_acc':
                        scores[scond] = gen[e].predict(data[e2])
                else:
                    # If same condition, cross-validate
                    if scoring_output == 'mean_acc':
                        scores[scond] = cross_val_multiscore(gen[e], data[e2][caeix],labels[e2][caeix],
                                                              cv = 5,n_jobs = 5).mean(0)
                    
                    elif scoring_output == 'trial_acc':
                        # We need a custom cross-validation in this case
                        if kind == 'Generalizing':
                            cgen = GeneralizingEstimator(clf, n_jobs=2, scoring = 'balanced_accuracy', verbose = True)
                        elif kind == 'Sliding':
                            cgen = SlidingEstimator(clf, n_jobs=2, scoring = 'balanced_accuracy', verbose = True)
                        
                        scores[scond] = []
                        nsamp = len(labels[e2])
                        # Divide the data in 5 chunks of cv_n length
                        cv_n = np.floor(nsamp / 5) 
                        ctix = np.arange(nsamp)
                        rix = ctix.copy()
                        
                        # Randomize trials
                        np.random.shuffle(rix)
                        
                        # Use crossvalidated predictions (loop over the 5 folds)
                        for ccv in range(5):                            
                            cstart, cend = int(ccv*cv_n), int((ccv+1)*cv_n*(ccv != 4) + nsamp*(ccv == 4)) # first and last trial index
                            print(cstart,cend)
                            crix = rix[cstart:cend] # select test trials
                            crix_neg = np.array([cctix for cctix in ctix if (cctix not in crix) & (cctix in caeix)]) #select training trials
                            
                            print(crix_neg)
                            print(crix)
                            
                            scores[scond] += [cgen.fit(X = data[e][crix_neg], y = labels[e][crix_neg]).predict(data[e2][crix])]
                        
                        scores[scond] = np.concatenate(scores[scond])
                        bix = np.argsort(rix)
                        print(rix[bix])
                        scores[scond] = scores[scond][bix]
                        print(scores[scond].shape)
                if scoring_output == 'trial_acc':
                    scores[scond] = np.array(scores[scond] - np.expand_dims(np.expand_dims(labels[e2],axis=-1),axis=-1) == 0).astype(int)
        
        ## Save output if requried
        
        if save_gen:
            print('saving models')
            gen_file = open(save_gen,'wb')
            dump(gen,gen_file)
            gen_file.close()

        if save_patterns:
            print('saving patterns')
            pat_file = open(save_patterns,'wb')
            dump(patterns,pat_file)
            pat_file.close()

        if save_filters:
            print('saving filters')
            fil_file = open(save_filters,'wb')
            dump(filters,fil_file)
            fil_file.close()
        
        if save_scores:
            print('saving scores')
            score_file = open(save_scores,'wb')
            dump(scores,score_file)
            score_file.close()
        
        if save_times:
            print('saving times')
            times_file = open(save_times,'wb')
            dump(times, times_file)
            times_file.close()
        
        if save_comps:
           print('saving components')
           comps_file = open(save_comps,'wb')
           dump(comps, comps_file)
           comps_file.close() 

        return gen, patterns, filters, scores, times, comps

def plot_time_gen_accuracy(scores, times, masks = None, nrows=2,
                           ncols=2,vlines=[],hlines=[],export_data=False,
                           savefig=None, vmin=None,vmax=None):
   
    """This function plots time-generalized accuracy matrices
    
    Parameters
    ----------
    scores: dict
        each entry is a 2d ndarray of decoding accuracies
    times: dict
        each entry is a time vector
    masks: dict
        Optional. Each entry contains a Boolean or numeric (1s and 0s) array
        indicating which matrix elements to draw contours around.
        Useful for highlighting significant points.
    nrows: int
        number of rows in the figure
    ncols: int 
        number of columns in the figure
    vlines, hlines: list
        list with floats indicating time points to draw vertical or horizontal lines
    export_data: Bool
        Whether to export data for publication figures
    savefig: None | str
        If str, saves the figure to str
    vmin, vmax: float
        Optional. Limits for color scale
    """
    #Create figure:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*8,ncols*8))
    
    # Loop over conditions:
    for six, s in enumerate(scores):
        
        # Get plot locations:
        x, y = six // ncols, six % ncols
        
        # Select axis:
        if nrows == 1:
            if ncols == 1:
                cax = axes
            else:
                cax = axes[y]
        else:
            cax = axes[x,y]
        
        # Get axes tick labels from condition times:
        #tr,te = s.split('_from_')
        strs = s.split('_')
        te, tr = strs[0], strs[2]
        tx1, tx2 = times[te][[0,-1]]
        ty1, ty2 = times[tr][[0,-1]]
        extent=[tx1, tx2, ty1, ty2]
                   
        # Make a plot
        im = cax.matshow(scores[s], vmin=vmin, vmax=vmax, cmap='RdBu_r', origin='lower',
                               extent=extent, aspect='auto',alpha=1)
        
        # Add mask contour if required (useful to display significant values)
        if masks:
            print('plotting clusters')
            cax.contour(masks[s].copy().astype('float'), levels=[-.1,1], colors='k',
                        extent=extent, origin='lower',corner_mask=False)
            
        # Axis lines:
        cax.axhline(0., color='k')
        cax.axvline(0., color='k')
        
        # Additional lines:
        for vl in vlines:
            cax.axvline(vl, color='k')
        
        for hl in hlines:
            cax.axhline(hl, color='k')            
        
        # Customize
        cax.xaxis.set_ticks_position('bottom')
        cax.set_xlabel('Testing Time (s)')
        cax.set_ylabel('Training Time (s)')
        cax.set_title(s)        
        plt.colorbar(im, ax=cax)
        ntrain, ntest =  scores[s].shape
        nall = ntrain*ntest

        if export_data:        
            cedata = {'x_testing_time': np.zeros((nall))*np.nan,'y_training_time':  np.zeros((nall))*np.nan,
                      'z_accuracy':  np.zeros((nall))*np.nan, 'contour_significance': np.zeros((nall))*np.nan}
            ccount = -1
            for trt in np.arange(scores[s].shape[0]):
                for tet in np.arange(scores[s].shape[1]):
                    ccount += 1
                    cedata['x_testing_time'][ccount] = times[te][tet]
                    cedata['y_training_time'][ccount] = times[tr][trt]
                    cedata['z_accuracy'][ccount] = scores[s][trt,tet]
                    cedata['contour_significance'][ccount] = masks[s][trt,tet].astype(int)

            cedata = pd.DataFrame(cedata)
            cedata=cedata.round(decimals=3)
            if not os.path.exists(export_data):
                os.makdirs(export_data)
            cedata.to_csv(op.join(export_data,'plot_data_timegen_acc_' + s + '.csv'),index=False)

    plt.tight_layout()
    if savefig:
        fig.savefig(savefig)

def plot_diagonal_accuracy(scores, times, CI = None, masks = None, nrows=2,
                           ncols=2,vlines=[],hlines=[],chance=1/2,
                           savefig=None, ylims = None, color = 'k'):
    """This function plots accuracies at the diagonal of time-generalized
    accuracy matrices or which are the output of sliding estimators
    
    Parameters
    ----------
    scores: dict
        each entry is a 1d or 2d ndarray of decoding accuracies
    times: dict
        each entry is a time vector
    CI: dict
        Optional. each entry contains confidence intervals to plot.
    masks: dict
        Optional. Each entry contains a Boolean or numeric (1s and 0s) array
        indicating which matrix elements highlight.
        Useful for marking significant time points.
    nrows: int
        number of rows in the figure
    ncols: int 
        number of columns in the figure
    vlines, hlines: list
        list with floats indicating time points to draw vertical or horizontal lines
    chance: float
        Chance level to draw horizontal line.
    savefig: None | str
        If str, saves the figure to str
    ylims: None | iterable
        If provided, set the limits of y axis
    color: str
        Line color 
    """

    #Create figure:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*6,ncols*6))
    for six, s in enumerate(scores):
        
        # Get plot locations:
        x, y = six // ncols, six % ncols
        
        # Select axis:
        if nrows == 1:
            if ncols == 1:
                cax = axes
            else:
                cax = axes[y]
        else:
            cax = axes[x,y]
        
        # Get axes tick labels from condition times:
        strsp = s.split('_from_')
        tr = strsp[0]
        te = strsp[1]
        # Setup masks
        if masks:
            cmask = masks[s].copy().astype('float')
        else:
            cmask = np.zeros(scores[s].shape,dtype='float')
        
        cmask[cmask == 0] = np.nan

        # Manage two-dimensional arrays
        if len(scores[s].shape) > 1:
            tseries = np.diagonal(scores[s])
            tseries_mask = np.diagonal(scores[s]*cmask)
        else: 
            tseries = scores[s]
            tseries_mask = scores[s]*cmask
        
        # Plot
        cax.plot(times[tr], tseries,color=color) #[:,range(scores[s].shape[0])].)
        cax.plot(times[tr], tseries_mask, linewidth=4,color=color)
        if CI:
            cax.fill_between(times[tr],  
                             np.diagonal(CI[s]['lower']),
                             np.diagonal(CI[s]['upper']),
                             alpha=.2,
                             color=color)
            
        cax.axvline(0., color='k')
        cax.axhline(chance, color='k')
        
        # Additional lines:
        for vl in vlines:
            cax.axvline(vl, color='k')
        
        for hl in hlines:
            cax.axhline(hl, color='k') 
        
        cax.set_xlabel('time')
        cax.set_ylabel('accuracy')
        cax.set_xlim((times[tr][0], times[tr][-1]))
        if ylims:
            cax.set_ylim(ylims)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)

