# Import directories
import scipy as sp
import scipy.io as sio
import numpy as np , h5py,numpy
import matplotlib.pyplot as plt
import matplotlib
import scipy

rng = np.random
import pickle
import sklearn
import sklearn.cluster as clus
from sklearn.decomposition import PCA
import copy

import os.path

import cvxpy as cp


def get_file(src_dirs, pattern_no, movie_no):
    '''Get the pattern files.
    Args: 
        src_dirs : List of directories which might have pattern file.
        pattern_no : Pattern number. For single electrode scan, its the electrode number
        movie_no : Movie index. For single electrode scan, its the amplitude index (1-indexed).
        
    Returns : 
        filename : Full filename where traces are stored.
    '''
    import operator
    
    found = False
    for isrc in src_dirs:
        p_file = os.path.join(isrc, 'p%d' % pattern_no)
        if os.path.isdir(p_file):
            found = True
            break
    if not found: 
        raise ValueError('File not found')
    
    pm_files = os.listdir(p_file)
    movie_nos = [int(ifile.split('m')[1]) for ifile in pm_files]
    
    pm_data = dict(zip(movie_nos, pm_files))
    sorted_x = sorted(pm_data.items(), key=operator.itemgetter(0))
    return os.path.join(p_file, sorted_x[movie_no - 1][1])


def get_data_from_file(src_file):
    '''Get repeated stimuluation data.'''
    
    b = np.fromfile(src_file, dtype='<h')
    b0 = b[:1000]
    b1 = b[1000:]

    data_traces = np.reshape(b1, [b0[0],b0[1],b0[2]], order='F')
    channels = b0[3: 3+b0[2]]
    return data_traces


def get_data(src_dirs, ielec, 
             ntrials_xx, subtract_global_art, 
             start_trial, traces):
    
    if subtract_global_art:
       src_file = get_file(src_dirs, ielec + 1, 1)  # add 1 to matlab indexing.
       data_traces = get_data_from_file(src_file)
       first_art = data_traces.mean(0)
       
    dd = [[]] * ntrials_xx.shape[0]
    for iamp in range(ntrials_xx.shape[0]):
        
        ntrials = ntrials_xx[iamp]

        if ntrials <= 0 :
            continue
        
        src_file = get_file(src_dirs, ielec + 1, iamp + 1)  # add 1 to matlab indexing.
        data_traces = get_data_from_file(src_file)

        start = start_trial[iamp]
        dd[iamp] = data_traces[start: start + ntrials, :, :]
        if subtract_global_art:
            dd[iamp] = np.array(dd[iamp], dtype=np.float32)
            dd[iamp] -= first_art

    traces[ielec] = dd
    print('.', flush=True, end='')
    return
    
    
def get_data_wrapper(src_dirs, q, subtract_global_art, traces):
    
    while not q.empty():
        ielec, ntrials_xx, start_trial = q.get()
        get_data(src_dirs, ielec, ntrials_xx, subtract_global_art, start_trial, traces)
    
    return

def simulate_stimulation(src_dirs, trial_elec_amps, start_trial, subtract_global_art=False, n_processes=20):
    '''Give recorded traces for different stimulations.
    
    Args : 
        src_dir: preprocessed single electrode scan data, could be split in multiple folders
        trial_elec_amps : 2D array (Electrodes x amplitude) of trials
        start_trial : number of traials previously read from src_data
    
    '''
    from multiprocessing import Process, Manager, Queue
    
    n_elecs = 512
    n_amps = 38
   
    traces_master = [[]] * n_elecs
    
    for elec_start in range(0, 512, 40):
        print('+', end='', flush=True)
        elecs = np.arange(elec_start, np.minimum(elec_start + 40, 512)).astype(np.int32) 
        with Manager() as manager:
        
        
            #traces = manager.list([manager.list([] * n_amps)] * n_elecs)
            # Output list made
            traces = manager.list([[]] * n_elecs)
        
            # Input Queues made
            q = Queue()
            for ielec in elecs:
                ntrials_xx = trial_elec_amps[ielec, :].astype(np.int)
                q.put([ielec, ntrials_xx, start_trial[ielec, :].astype(np.int)])
           
            # Processes made
            p = []  
            for iprocess in range(n_processes):
                p.append(Process(target=get_data_wrapper, 
                             args=(src_dirs, q, 
                                   subtract_global_art, 
                                   traces)))
                p[-1].start()

            for ip in p:
                ip.join()
            
            traces = list(traces)
        
        for ielec in elecs:
            traces_master[ielec] += traces[ielec]            
    
    new_start_trial = start_trial + trial_elec_amps
        

    return traces_master, new_start_trial   


"""
def simulate_stimulation(src_dirs, trial_elec_amps, start_trial, subtract_global_art=False):
    '''Give recorded traces for different stimulations.
    
    Args : 
        src_dir: preprocessed single electrode scan data, could be split in multiple folders
        trial_elec_amps : 2D array (Electrodes x amplitude) of trials
        start_trial : number of traials previously read from src_data
    
    '''
    from multiprocessing import Process, Manager
    
    n_elecs = 512
    n_amps = 38
    
    with Manager() as manager:
        
        
        #traces = manager.list([manager.list([] * n_amps)] * n_elecs)
        traces = manager.list([[]] * n_elecs)

        p = []
        for ielec in range(n_elecs):

            ntrials_xx = trial_elec_amps[ielec, :].astype(np.int)

            src_file = get_file(src_dirs, ielec + 1, 1)  # add 1 to matlab indexing.
            data_traces = get_data_from_file(src_file)

            if subtract_global_art:
                first_art = data_traces.mean(0)
            else:
                first_art = None

            p.append(Process(target=get_data, 
                             args=(src_dirs, ielec + 1,
                                   ntrials_xx, 
                                   subtract_global_art, 
                                   first_art, start_trial, traces)))
            p[-1].start()

        for ip in p:
            ip.join()

        new_start_trial = start_trial + trial_elec_amps
        return list(traces), new_start_trial
"""
