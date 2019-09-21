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
import glob
import cvxpy as cp
from scipy.optimize import curve_fit

def get_nearby_elecs(elec_locx, elec_locy, center_electrode, n_elecs_nearby=6):
    """Return nearby electrodes in increasing order of distance."""
    elec_locx = np.double(elec_locx)
    elec_locy = np.double(elec_locy)
    distances = (elec_locx - elec_locx[center_electrode])**2 + (elec_locy - elec_locy[center_electrode])**2
    ii = np.argsort(distances)
    return ii[:n_elecs_nearby], distances[ii]

def get_autosort(loc, ielec):
    spk_sort_data = sio.loadmat(os.path.join(loc, 'elecRespAuto_p%d.mat' % (ielec + 1)))

    cids = spk_sort_data['elecRespAuto'][0][0][1][0][0][4][0]
    ncs = cids.shape[0]

    spks = []
    for icell in range(ncs):
        spks += [spk_sort_data['elecRespAuto'][0][0][1][0][0][3][0][icell] > 0]

    spks = np.array(spks)

    return spks, cids


def get_manualsort(loc, ielec):
    files = glob.glob(os.path.join(loc, 'elecResp_n*_p%d.mat' % (ielec + 1) ))
    
    data_acc = {}
    for ifile in files:
        print(ifile)
        dat = sio.loadmat(ifile)
        cid = int(ifile.split('/')[-1].split('_')[1][1:])
        curve = dat['elecResp'][0, 0][4][0, 0][5]
        data_acc.update({cid: curve})
    
    return data_acc


def get_yaess(loc, ielec, spk_th):
    
    dat = pickle.load(open(os.path.join(loc, '%d.pkl' % ielec), 'rb'))  # We store electrode IDs with 0 indexing.
    spks = dat['data']['yaess_y_spks']
    cids = dat['data']['yaess_cids']
    spks = np.sum(spks, 2) > spk_th
    spks = np.transpose(spks, [1, 0, 2])
    
    return spks, cids


def get_artifact_space(artifact_basis_retinas, dist, 
                       iamp, dim_artifact):
    norm_dist = int(dist / 3600)
    lookup_key = {0: 0, 1: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
    return artifact_basis_retinas[iamp, lookup_key[norm_dist]][:, :dim_artifact]


def get_recorded_traces(stim_elec, amps, ei, ei_cids, artifact_basis_retinas, traces_s, elec_coords, 
                        rec_length_use=55, t_min=0, t_max=70, dim_artifact=10, scale_ei_rec = 1/300, cids_use=None):
    
    trace_use = traces_s

    # Find nearby elecs
    nearby_elecs, nearby_distances = get_nearby_elecs(elec_coords[:, 0], elec_coords[:, 1], 
                                                      stim_elec, n_elecs_nearby=7)

    # Find cells with strong EI on these electrodes
    #print(ei.shape)
    ei_stim_elec = np.min(ei[:, stim_elec, :], 1)
    ei_rec_elec = np.min(np.min(ei[:, :, :], 2), 1)
    
    if cids_use is None :
        cells = np.where(np.logical_and(ei_stim_elec <= -10, ei_rec_elec <= -30) > 0)[0]
        #print(cells)
    else :
        cidx = [];
        #print(ei_cids.shape)
        for ic in cids_use:
            cidx += [np.where(ei_cids == ic)[0]]
        cells = np.array(cidx).astype(np.int)
        cells = np.squeeze(cells)
        
        #print('EI amp on stim elec', ei_stim_elec[cells])
        #print('EI amp on recording elec', ei_rec_elec[cells])
        
    ei_elecs = ei[:, nearby_elecs, :]
    relevant_ei = ei_elecs[cells, :, :]  # cells x electrodes x time
    relevant_ei *= scale_ei_rec
    relevant_cids = ei_cids[cells]
    n_cells, _, L2 = relevant_ei.shape
        
    #print(relevant_ei.shape)
    if n_cells == 0:
        raise ValueError('Zero cells.')
      
    # iamp = 16
    rec_log = []
    rec_2d_log = []
    trials_log = []
    for iamp in amps:
        recordings = trace_use[stim_elec][iamp][:, nearby_elecs, :rec_length_use]  # trials, electrodes, time
        recordings *= scale_ei_rec
        n_trials, n_elecs, L1 = recordings.shape
        recordings_2d = np.reshape(recordings, [n_trials, -1]) # trials x (electrodes stacked on each other).
        rec_log += [np.array(recordings).astype(np.float32)]
        rec_2d_log += [np.array(recordings_2d).astype(np.float32)]
        trials_log += [n_trials]
        
    # Make each amplitude have same number of trials (append 0s) and make a mask to keep track of this.
    max_trials = np.max(trials_log)
    mask = np.zeros((max_trials, np.squeeze(np.array(amps)).shape[0]))     
    for iamp in amps:
        rec_log[iamp] = np.append(rec_log[iamp], np.zeros((max_trials - trials_log[iamp], n_elecs, L1)), axis=0) 
        rec_2d_log[iamp] = np.append(rec_2d_log[iamp], np.zeros((max_trials - trials_log[iamp], n_elecs * L1)), axis=0)
        mask[:trials_log[iamp], iamp] = 1

    rec_log = np.array(rec_log)
    rec_2d_log = np.array(rec_2d_log)
    

    
    #t_min = 0
    #t_max = 70
    n_shifts = t_max - t_min
    
    # compute how many shifts needed
    for icnt, itime in enumerate(range(t_min, t_max)):
        if  np.maximum(itime, 0) > np.minimum(L2 + itime, L1):
            n_shifts = icnt
            break
    n_shifts = n_shifts - 1
    t_max = t_min + n_shifts
    
    shifted_eis = []
    for icell in range(n_cells):
        xx = np.zeros((n_shifts, n_elecs,  L1))
        for icnt, itime in enumerate(range(t_min, t_max)):
            if  np.maximum(itime, 0) > np.minimum(L2 + itime, L1):
                continue
            #op = relevant_ei[icell, :, :np.minimum(L2, L1-itime)]
            op = relevant_ei[icell, :,  np.maximum(-itime, 0):np.minimum(L2, L1-itime)]
            xx[icnt, :, np.maximum(itime, 0): np.minimum(L2 + itime, L1)] = op
        xx_2d = np.reshape(xx, [xx.shape[0], -1])
        shifted_eis += [xx_2d]

    shifted_eis = np.array(shifted_eis).T
    shifted_eis = np.transpose(shifted_eis, [0, 2, 1])
    shifted_eis = np.reshape(shifted_eis, [shifted_eis.shape[0], -1])

    # artifact_subspace = np.eye(L1 * n_elecs) 
    d = dim_artifact
    artifact_dict = {}
    for iamp in range(len(list(amps))):
        artifact_subspace = np.zeros((n_elecs * rec_length_use, n_elecs * (d + 1)))
        istart = 0
        jstart = 0
        for jelec in range(n_elecs):
            artifact_sp = get_artifact_space(artifact_basis_retinas, nearby_distances[jelec], 
                                             iamp, dim_artifact)
            artifact_subspace[istart: istart + L1, jstart: jstart + d] = artifact_sp  # np.eye(L1 * n_elecs)
            artifact_subspace[istart: istart + L1, jstart + d] = 1 # Add electrodewise bias to artifact
            istart = istart + L1
            jstart = jstart + d + 1
            
        artifact_dict.update({iamp: artifact_subspace})

    # Solve for all trials simultaneously
    #print('rec_2d_log', rec_2d_log.shape)
    #print('rec_log', rec_log.shape)
    
    n_trials = rec_2d_log.shape[1]
    b = rec_2d_log.T # np.reshape(recordings_2d, [-1])
    A = []
    for iamp in range(len(list(amps))):
        A += [np.append(shifted_eis, artifact_dict[iamp], axis=1)]

    return A, b, n_trials, n_cells, n_elecs, n_shifts, artifact_subspace, shifted_eis, rec_log, relevant_cids, mask


def initialize_artifact(A, b, n_shifts, n_cells):
    
    n_spks_params = n_shifts * n_cells
    n_amps = b.shape[2]
    artifact_subspace = A[0][:, n_spks_params:]
    x_art_init = np.zeros((artifact_subspace.shape[1], n_amps))  # artifact subspace x amplitude
    art_recons = np.zeros((b.shape[0], n_amps))
    y_spks = np.zeros((n_amps, n_spks_params, b.shape[1]))
    
    for iamp in range(n_amps):
        
        A_spk = A[iamp][:, :n_spks_params]
        artifact_subspace = A[iamp][:, n_spks_params:]
        
        # Initialize artifacts
        avg_trace = np.mean(b[:, :, iamp], 1)
        x_art_init[:, iamp] = np.linalg.pinv(artifact_subspace).dot(avg_trace)
        art_recons[:, iamp] = np.dot(artifact_subspace, x_art_init[:, iamp])
    
        # Initialize spks
        for itrial in range(b.shape[1]):
            
            residual = b[:, itrial, iamp] - artifact_subspace.dot(x_art_init[:, iamp])
            y_spks[iamp, :, itrial] = greedy_spks(A_spk, residual, n_shifts, n_cells)
            print(iamp, itrial, np.sum(y_spks[iamp, :, itrial]))
        
        '''
        # Initialize spikes
        ATA = A_spk.T.dot(A_spk)
        iinv = np.linalg.inv(ATA + 1 * np.eye(ATA.shape[0])).dot(A_spk.T)
        for itrial in range(n_trials) :
            y_spks[iamp, :, itrial] = iinv.dot(b[:, itrial, iamp] - artifact_subspace.dot(x_art_init[:, iamp]))
        '''
        
    return x_art_init.T, y_spks

def greedy_spks(A_spk, residual, n_shifts, n_cells):
    
    # copy data
    res = np.copy(residual)
    
    # spikes
    y_spks = np.zeros((n_shifts * n_cells))
    spkd_cells = []
    
    # reconstruction
    xx = np.eye(n_shifts * n_cells)
    recons_spk = A_spk.dot(xx)
    '''
    print(A_spk.shape, n_cells, n_shifts)
    plt.figure()
    plt.plot(A_spk[:, n_shifts - 1])
    plt.figure()
    plt.plot(A_spk[:, 0 ])
    asdasa
    '''
    
    while len(spkd_cells) < n_cells:
        # residual error
        errors = np.sum((np.expand_dims(res, 1) - recons_spk) ** 2, 0)
        for icell in spkd_cells:
            errors[n_shifts * icell: n_shifts * (icell + 1)] = np.inf
        ix = np.argmin(errors)
        if errors[ix] > np.sum( res **2 ) * 0.99:
            break

        # which cell spikes?
        icell = np.floor(ix / n_shifts).astype(np.int)
        y_spks[ix] = 1
        spkd_cells += [icell]
        
        # update residual
        res = res - recons_spk[:, ix]
    return y_spks
    
def initialize_and_normalize(A, b, n_shifts, n_cells):
    x_art_init, y_spks = initialize_artifact(A, b, n_shifts, n_cells)
    n_spks_params = n_shifts * n_cells
    ratio = [[]] * b.shape[-1]
    for iamp in range(b.shape[-1]):
        ratio[iamp] = np.sqrt(np.mean(x_art_init[:, iamp] ** 2) / np.mean(np.mean(y_spks[iamp, :, :], 1) ** 2))
        A[iamp][:, n_spks_params:] *= 1 / ratio[iamp]
    x_art_init_new, y_spks_new = initialize_artifact(A, b, n_shifts, n_cells)
    return A, b, x_art_init_new, y_spks_new


def sigmoid(xs,m,b):
	return 1/ (1 + np.exp(-m*(xs - b)))

def getActCurve(amps, probs, xvals):
	popt, pcov = curve_fit(sigmoid, amps, probs)
	y = sigmoid(xvals,*popt)
	return y

def compare_to_manual(manual_data, sort_data):
	th_logs = []
	rss_logs = []
	for icell in manual_data.keys():
		manual = manual_data[icell]
		if icell not in sort_data.keys():
			continue
		sort = sort_data[icell]
	
		manual_th = np.argmin((manual - 0.5) ** 2)
		sort_th = np.argmin((sort - 0.5) ** 2)
		print(icell, manual_th, sort_th) 
		th_logs += [[manual_th, sort_th]]
		
		# compute sigmoid fits and compute RSS
		manual = np.squeeze(np.array(manual))
		sort = np.squeeze(np.array(sort))
		try:
			manual_curve = getActCurve(np.arange(manual.shape[0]), manual, np.arange(manual.shape[0]))
			sort_curve = getActCurve(np.arange(sort.shape[0]), sort, np.arange(manual.shape[0]))
			rss = np.sum((manual_curve - sort_curve) ** 2)
		except:
			rss = np.nan
		
		rss_logs += [rss]
	return th_logs, rss_logs
        
        
def compile_and_compare(y_spks, relevant_cids, manual_data):
	# get model
	probs_model = y_spks.mean(-1)  # cells x model 
	model_data = {}
	for iicell, icell in enumerate(relevant_cids):
        	model_data.update({icell: probs_model[iicell, :]})
        
	model_th, model_rss = compare_to_manual(manual_data, model_data)
	model_th = np.array(model_th)
	
	return model_th, model_rss, model_data

def get_ei_data(ei_src):
	# Load EI data
	ei_data = sio.loadmat(ei_src)
	ei = ei_data['eis']
	ei_cids = np.squeeze(ei_data['cids'])
	return ei, ei_cids


def get_elec_coords(elec_coords_loc):
	elec_coords = sio.loadmat(elec_coords_loc)
	elec_coords = elec_coords['coords'].astype(np.double)
	return elec_coords


def get_artifacts_basis(art_basis):
	art_dat = sio.loadmat(art_basis)
	artifact_basis_retinas = art_dat['basis']
	return artifact_basis_retinas


def get_xy_priors(prior_src):
	return pickle.load(open(prior_src, 'rb'))


