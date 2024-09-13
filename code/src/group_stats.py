from os import path as op
from stormdb.access import Query
import pickle
import numpy as np
from scipy import stats
from mne.stats import spatio_temporal_cluster_1samp_test, spatio_temporal_cluster_test, fdr_correction, ttest_1samp_no_p
from sklearn import linear_model
from sklearn.impute import SimpleImputer as Imputer
from functools import partial

def load_ERF_sensor(subs, suffix='', exclude=[]):
    """This function loads participant-wise event related fields and
    collects them for statistical analyses

    Parameters
    ----------
    sub: array of int
        participant numeric codes
    suffix: str
        identification of evoked file
    exclude: list
        subject codes to exclude

    Returns
    ----------
    sdata: dict of ndarray
        dict containing a stack data array per condition (subs x channels x times)
    scodes: dict of lists
        Subject codes included in each condition
    times: ndarray 
        Time vector
    """
   
    project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
    project_dir = '/projects/' + project
    avg_path = project_dir + '/scratch/working_memory/averages/data/'
    sdata, scodes = {}, []
    qr = Query(project)
    sub_codes = qr.get_subjects()
    for sub in subs:
        sub_code = sub_codes[sub-1]
        if sub not in exclude:
            try:
                print('loading subject {}'.format(sub))
                evkd_fname = op.join(avg_path,sub_code,sub_code + '_evoked_' + suffix + '.p')
                evkd_file = open(evkd_fname,'rb')
                evokeds = pickle.load(evkd_file)
                evkd_file.close()
                for c in evokeds:
                    times = evokeds[c].times
                    sdata.setdefault(c,[])
                    sdata[c] += [evokeds[c]]
                scodes += [sub]
            except Exception as e:
                print('could not load subject {}'.format(sub_code))
                print(e)
                continue
    try:
        print('converting to array')
        scodes = np.array(scodes)
        print('converted to array')
        
    except Exception as ee:
        print(ee)
    print('loaded data for {} subjects'.format(len(sdata[c])))
    return sdata, scodes, np.array(times)

def load_scores(suffix, subs, exclude=[]):
    """This function loads participant-wise decoding accuracies and
    collects them for statistical analyses

    Parameters
    ----------
    suffix: str
        identification of accuracy file
    subs: array of int
        participant numeric codes
    exclude: list
        subject codes to exclude

    Returns
    ----------
    sdata: dict of ndarray
        dict containing a stack data array per condition (subs x training_times x test_times)
    scodes: dict of lists
        Subject codes included in each condition
    times: dict of ndarrays
        Time vectors per condition
    """
    project = 'MINDLAB2020_MEG-AuditoryPatternRecognition'
    project_dir = '/projects/' + project
    avg_path = project_dir + '/scratch/working_memory/averages/data/'
    sdata, scodes = {}, []
    qr = Query(project)
    sub_codes = qr.get_subjects()
    for sub in subs:
        sub_code = sub_codes[sub-1]
        if sub not in exclude:
            try:
                print('loading subject {}'.format(sub))
                
                #score_fname = op.join(avg_path,sub_code + '_scores_imagined_smoothing25_50_hp005.p')
                score_fname = op.join(avg_path,sub_code,sub_code + '_scores_' + suffix + '.p')
                score_file = open(score_fname,'rb')
                score = pickle.load(score_file)
                if len(scodes) == 0:
                    times_fname = op.join(avg_path,sub_code,sub_code + '_times_' + suffix + '.p')
                    times_file = open(times_fname,'rb')
                    times = pickle.load(times_file)
                score_file.close()
                times_file.close()
                for c in score:
                    sdata.setdefault(c,[])
                    sdata[c].append(score[c].data)
                scodes += [sub]
            except Exception as e:
                print('could not load subject {}'.format(sub_code))
                print(e)
                continue
    try:
        print('converting to array')
        sdata = {s: np.array(sdata[s]) for s in sdata}
        scodes = np.array(scodes)
        print('converted to array')
        
    except Exception as ee:
        print(ee)
    print('loaded data for {} subjects'.format(sdata[c].shape[0]))
    return sdata, scodes, times

def grand_avg_scores(sdata):
    ''' grand average of stack group data'''

    # grand averages:
    smean, sstd, smedian, sci_lower, sci_upper, siqr_lower, siqr_upper = {},{},{},{},{},{},{}
    for s in sdata:
        smean[s] = np.mean(sdata[s],0)
        smedian[s] = np.median(sdata[s],0)
        sstd[s] = np.std(sdata[s],0)
        sci_lower[s] = smean[s] - 1.96*sstd[s]/np.sqrt(sdata[s].shape[0]-1)
        sci_upper[s] = smean[s] + 1.96*sstd[s]/np.sqrt(sdata[s].shape[0]-1)
        siqr_lower[s] = np.percentile(sdata[s],25,0)
        siqr_upper[s] = np.percentile(sdata[s],75,0)
        
    return smean, sstd, sci_lower, sci_upper, smedian, siqr_lower, siqr_upper

def do_stats(X, method='FDR', adjacency=None, FDR_alpha=.025, h0=0,sigma=1e-3,n_jobs=-1,
             cluster_alpha = .05, p_threshold = .05, n_permutations=500, cluster_method = 'normal'):
    """This performs stats based on 1-sample t-tests of provided data with multiple comparisons correction
       (either cluster based permutations or FDR).

    Parameters
    ----------
    X: ndarray
        stacked data array. First dimension corresponds to subjects.
    method: str
        False positive rate correction method. 
        Either 'montecarlo' for cluster-based permutations or 'FDR'.
    adjacency: None | matrix
        Adjacency matrix for clustering. See mne.spatio_temporal_cluster_1samp_test for details. 
        If None, it employs mne defaults.
    FDR_alpha: float
        Alpha level for FDR correction, if applicable.
    h0: float
        Null hypothesis value
    sigma: float 
        Sigma value for TFCE if required.
    n_jobs: float
        number of cores to employ
    cluster_alpha: float
        Alpha level
    p_threshold: float
        Cluster-forming threshold
    n_permutations: int
        number of permutations
    cluster_method: str
        either 'normal' or 'TFCE'

    Returns
    ----------
    stats_results: dict
        results with relevant statistical outputs
    """
    n_subjects = X.shape[0]
    if cluster_method == 'normal':
        t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
        print('Two sided alpha level: {} - t threshold: {}'.format(p_threshold,t_threshold))

    elif cluster_method == 'TFCE':
        t_threshold = dict(start=0, step=0.2)
        print('Two sided alpha level: {} - Using TFCE'.format(p_threshold))

    stat_fun = partial(ttest_1samp_no_p, sigma=sigma)
    if method == 'montecarlo':
        print('Clustering.')
        # Reshape for permutations function
        if len(X.shape) == 3:
            X = X.transpose(0,2,1)
            
        tvals, clusters, cluster_p_values, H0 = \
            spatio_temporal_cluster_1samp_test(X-h0, adjacency=adjacency, n_jobs=n_jobs,
                                                threshold=t_threshold, buffer_size=None,
                                                verbose=True, n_permutations = n_permutations,
                                                out_type='mask', stat_fun=stat_fun)
        good_cluster_inds = np.where(cluster_p_values <= cluster_alpha)[0]
        gclust = np.array([clusters[c] for c in good_cluster_inds])
        
        # Get mask of significant data points
        gmask = np.zeros(X.shape[1:])
        if gclust.shape[0] > 0:
            for tc in range(gclust.shape[0]):
                gmask = gmask + gclust[tc].astype(float)
        
        # Reshape back
        if len(X.shape) == 3:
            X = X.transpose(0,2,1)
        
        stat_results = {'mask': gmask.T, 'tvals': tvals.T, 'pvals': cluster_p_values,
                        'data_mean': np.mean(X,0), 'data_sd': np.std(X,0),'clusters': clusters,
                        'n': n_subjects,'alpha': cluster_alpha,'p_threshold': p_threshold}
                        
    elif method == 'FDR':
        print('\nPerforming FDR correction\n')
        tvals, pvals = stats.ttest_1samp(X, popmean=h0,axis=0)
        gmask, adj_pvals = fdr_correction(pvals, FDR_alpha)
        print(np.sum(gmask==0))
        stat_results = {'mask': gmask, 'tvals': tvals, 'pvals': pvals, 'qvals': adj_pvals,
                        'data_mean': np.mean(X,0), 'data_sd': np.std(X,0), 'n': n_subjects,
                        'alpha': FDR_alpha}
    return stat_results