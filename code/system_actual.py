import numpy as np
import sample_spks as ss
import spike_sort_util as ss_util
import spike_sort_alg_2 as ss_alg

class SortedRetina(object):
    '''Load autosorted spikes.'''
    def __init__(self, cids, loc, sort_alg='yaess'):
        self.cids = np.squeeze(np.array(cids))
        self.loc = loc 
        self.n_elecs = 512
        self.n_amps = 38
        self.start_trial = np.zeros((self.n_elecs, self.n_amps)).astype(np.int)
        self.sort_alg = sort_alg
        
    def stimulate(self, trial_elec_amps):
        spks_coll = self._get_spikes(trial_elec_amps=trial_elec_amps)
        return spks_coll

    def _get_spikes(self, trial_elec_amps):
        
        # spks_collect
        spks_collect = []
        for _ in range(self.n_elecs):
            yy = []
            for _ in range(self.cids.shape[0]):
                xx = [[], []]
                yy += [xx]
            spks_collect += [yy]
        
        for ielec in range(self.n_elecs):
            if trial_elec_amps[ielec, :].sum() > 0:
                try:
                    if self.sort_alg == 'autosort':
                        spks_autosort, cids_autosort = ss_util.get_autosort(self.loc, ielec)
                    if self.sort_alg == 'yaess':
                        spks_autosort, cids_autosort = ss_util.get_yaess(self.loc, ielec, spk_th=0.5)
                    
                except FileNotFoundError:
                    print('File not found for electrode %d' % ielec)
                    continue
         
                for iicell, icell in enumerate(cids_autosort):
                   if np.where(self.cids == icell)[0].shape[0] == 0:
                       continue
                   xcell = np.where(self.cids == icell)[0][0]
                   for iamp in range(self.n_amps):
                        st = int(self.start_trial[ielec, iamp])
                        n_tr = int(trial_elec_amps[ielec, iamp])
                        for itr in range(n_tr):
                            spks_collect[ielec][xcell][0] += [iamp]
                            spks_collect[ielec][xcell][1] += [spks_autosort[iicell, iamp, st + itr]]
                     
        self.start_trial += trial_elec_amps
                       
        return spks_collect         
                 
class PerfectlyObservedRetina(object):
    def __init__(self, dictionary, cids):
        self.dictionary = dictionary
        self.cids = np.squeeze(np.array(cids))
       
    def stimulate(self, trial_elec_amps):
        _, spks_coll = get_responses_3d(self.dictionary, 
                                        trial_elec_amps=trial_elec_amps)
        return spks_coll


def get_responses_3d(D, trial_elec_amps):
    
    n_elecs = D.shape[0]
    n_amps = D.shape[1]
    n_cells = D.shape[2]
    
    spks_collect = []
    for _ in range(n_elecs):
        yy = []
        for _ in range(n_cells):
            xx = [[], []]
            yy += [xx]
        spks_collect += [yy]
    
    probs_est = np.zeros((n_elecs, n_amps, n_cells))
    
    for ielec in range(D.shape[0]):
        # print(ielec, len(spks_collect[0][0][0]))
        ntrials_xx = trial_elec_amps[ielec, :].astype(np.int)
        for iamp in range(ntrials_xx.shape[0]):
            
            ntrials = ntrials_xx[iamp]
            
            if ntrials == 0:
                continue
            # print(n_cells, ntrials, ielec, iamp, D.shape) 
            spks = (np.random.rand(n_cells, ntrials) <= 
                    np.repeat(np.expand_dims(D[ielec, iamp, :], 1), ntrials, 1)).astype(np.float32)
            probs_est[ielec, iamp, :] = spks.mean(1)
            for icell in range(n_cells):
                
                if np.sum(D[ielec, :, icell]) == 0:  # cell will never be stimulated!!
                    continue
                spks_collect[ielec][icell][0] +=  ntrials * [iamp]
                spks_collect[ielec][icell][1] += list(spks[icell, :])
                
    return probs_est, spks_collect # elec x amp x cell


class SampleRealRetina(object):
    """Sample actual recorded spikes."""

    def __init__(self, preprocessed_data, ei_src, elec_coords_loc, art_basis, spk_th=0.6, cids_use=None):
        """Setup paths, load basic properties"""
        
        self.preprocessed_data_src = preprocessed_data
        self.ei, self.ei_cids = ss_util.get_ei_data(ei_src)
        self.ei_cids = np.squeeze(np.array(self.ei_cids))
        self.cids = self.ei_cids
        self.elec_coords = ss_util.get_elec_coords(elec_coords_loc)
        self.artifact_basis = ss_util.get_artifacts_basis(art_basis)
        
        self.n_elecs = self.elec_coords.shape[0]
        self.n_amps = 38
        self.n_cells = self.ei.shape[0] 
        self.start_trial = np.zeros((self.n_elecs, self.n_amps)).astype(np.int)
        self.spk_th = spk_th

        if cids_use is not None:
           self.cids_use = cids_use
        else:
           self.cids_use = self.cids
        self.n_cells_use = np.squeeze(np.array(cids_use.shape[0]))
          
    
    def stimulate(self, trial_elec_amps, dim_artifact=9, lam_logistic=0.005, lam_l1=1, retina_model=None):
        """Sample previously record spikes and do spike sorting.
           
           Note : Compared to other models, this results ALL spikes, starting from beginning!!
           """
        print('COMPARED TO OTHER MODELS, THIS RETURNS ALL ')
        # Sample spikes from stimulation
        # Re-spike sort all previously samples spikes
        '''
        traces, self.start_trial = ss.simulate_stimulation([self.preprocessed_data_src], 
                                                            (trial_elec_amps + self.start_trial).astype(np.int), 
                                                            np.zeros_like(trial_elec_amps).astype(np.int),
                                                            subtract_global_art=True)
        ''' 
        # Find electrodes which might have large probability of spiking.
        elec_to_analyze = np.where(trial_elec_amps.sum(1) > 0)[0] 
        # print(elec_to_analyze)  
        
        spk_coll = []
        for _ in range(self.n_elecs):
            yy = []
            for _ in range(self.n_cells_use):  # previously, self.n_cells
                xx = [[], []]
                yy += [xx]
            spk_coll += [yy]
 
        for ielec in elec_to_analyze:
            print('+', end='', flush=True)
            # Get traces for the analyzed electrode.
            
            ntrials_xx = trial_elec_amps[ielec, :] + self.start_trial[ielec, :]
            traces = [[]] * self.n_elecs
            # populate traces.
            ss.get_data([self.preprocessed_data_src], ielec,
                         ntrials_xx.astype(np.int32), True,
                         0 * ntrials_xx.astype(np.int32), traces)  
            
            # Format recorded data.
            try:
                op = ss_util.get_recorded_traces(ielec, range(self.n_amps), 
                                             self.ei, self.ei_cids, 
                                             self.artifact_basis, traces, 
                                             self.elec_coords, 
                                             rec_length_use=55, t_min=-10, 
                                             t_max=40, dim_artifact=dim_artifact, 
                                             scale_ei_rec = 1/100)
                A, b, n_trials, n_cells, n_elecs, n_shifts,  artifact_subspace, shifted_eis, recordings, relevant_cids, mask = op
            except ValueError:
                print('Zero cells!')
                continue

            # TODO(bhaishahster): Do spike sorting using previous spikes 
            q_in = None
            if retina_model is not None:
                q_in = np.zeros((self.n_amps, relevant_cids.shape[0]))
                for iicell, icell in enumerate(relevant_cids):
                    cidxxx = np.where(retina_model.cids == icell)[0][0]
                    q_in[:, iicell] = retina_model.probs_smoothen_recons[ielec, :, cidxxx]
                 

            # Run algorithm
            # TODO(bhaishahster): Deal with different number of trials for different amplitudes.
            op = ss_alg.spike_sort(A, b, n_cells, n_shifts, n_trials, 
                                       lr=0.01, opt_th=1e-4, 
                                       n_iter=6000-1, x_art_init=None, tau_in=1.0, tau_change_freq=400, 
                                       lam_z=0, lam_logistic=lam_logistic, lam_l1=lam_l1, phase_freq=5000, 
                                       n_outer_iter=0, mask=mask, q_in=q_in)
            y_spks, y_artifacts, spks_recons, art_recons, total_recons, sig_params, loss_log = op

            # Collect spikes
            for xcell, icell in enumerate(relevant_cids):
                if icell in self.cids_use:
                    iicell = np.where(self.cids_use == icell)[0][0]
                else:
                    continue
                for iamp in np.arange(self.n_amps):
                    nt = trial_elec_amps[ielec, iamp]
                    start = self.start_trial[ielec, iamp]
                    # print(ielec, icell, iicell, xcell, iamp)
                    #if nt > 0:
                    
                    nt = int(nt)
                    spk_coll[ielec][iicell][0] = spk_coll[ielec][iicell][0] + [iamp] * (nt + start)
                    spk_coll[ielec][iicell][1] = spk_coll[ielec][iicell][1] + list(y_spks[iamp, xcell, :, :nt + start].sum(0) > self.spk_th)
                        # print(len(spk_coll[ielec][iicell][0]))
        self.start_trial = trial_elec_amps + self.start_trial
        return spk_coll  
    
    '''   
    def stimulate(self, trial_elec_amps, dim_artifact=9, lam_logistic=0.005, lam_l1=1):
        """Sample previously record spikes and do spike sorting."""
        
        # Sample spikes from stimulation
        traces, self.start_trial = ss.simulate_stimulation([self.preprocessed_data_src], 
                                                            trial_elec_amps, self.start_trial, 
                                                            subtract_global_art=True)
         
        # Find electrodes which might have large probability of spiking.
        elec_to_analyze = np.where(trial_elec_amps.sum(1) > 0)[0] 
        #print(elec_to_analyze)  
        
        spk_coll = []
        for _ in range(self.n_elecs):
            yy = []
            for _ in range(self.n_cells):
                xx = [[], []]
                yy += [xx]
            spk_coll += [yy]
         
        params = {} 
        for ielec in elec_to_analyze:
            print('Electrode data prepared: %d' % ielec) 
            
            # Get recorded data
            try:
                op = ss_util.get_recorded_traces(ielec, range(self.n_amps), 
                                                  self.ei, self.ei_cids, 
                                                  self.artifact_basis, traces, 
                                                  self.elec_coords, 
                                                 rec_length_use=55, t_min=-10, 
                                                 t_max=40, dim_artifact=dim_artifact, 
                                                 scale_ei_rec = 1/100)
                A, b, n_trials, n_cells, n_elecs, n_shifts,  artifact_subspace, shifted_eis, recordings, relevant_cids, mask = op
                params.update({ielec: {'A': A, 'b': b, 'n_trials': n_trials, 'n_cells': n_cells, 
                                   'n_elecs': n_elecs, 'n_shifts': n_shifts, 'artifact_subspace': artifact_subspace, 
                                   'shifted_eis': shifted_eis, 'recordings': recordings, 
                                   'relevant_cids': relevant_cids, 'mask': mask}})
            except ValueError: 
                print('Zero cells')
            # TODO(bhaishahster): Do spike sorting using previous spikes 

        from multiprocessing import Process, Manager
        p_log = {}
        manager = Manager()
        y_spks_output = manager.dict()
        
        for ielec in params.keys():
            
            p = Process(target=self._spike_sort_wrapper, args=(params[ielec]['A'], params[ielec]['b'],
                                                            params[ielec]['n_cells'], params[ielec]['n_shifts'], 
                                                            params[ielec]['n_trials'],
                                                            lam_logistic, lam_l1, params[ielec]['mask'], y_spks_output, ielec ))
            print('Start %d' % ielec)
            p.start()
            p_log.update({ielec: p})
         
        for ielec in params.keys():
            p_log[ielec].join()
            print('Done %d' % ielec)
       
        # Collect spikes 
        y_spks_output = dict(y_spks_output)
        for ielec in params.keys():
            y_spks = y_spks_output[ielec]
            for xcell, icell in enumerate(params[ielec]['relevant_cids']):
                iicell = np.where(self.ei_cids == icell)[0][0]
                for iamp in np.arange(self.n_amps):
                    nt = trial_elec_amps[ielec, iamp]
                    print(ielec, icell, iicell, xcell, iamp)
                    if nt > 0:
                        nt = int(nt)
                        spk_coll[ielec][iicell][0] = spk_coll[ielec][iicell][0] + [iamp] * nt
                        spk_coll[ielec][iicell][1] = spk_coll[ielec][iicell][1] + list(y_spks[iamp, xcell, :, :nt].sum(0) > self.spk_th)
                        print(len(spk_coll[ielec][iicell][0]))
        return spk_coll  
    
    def _spike_sort_wrapper(self, A, b, n_cells, n_shifts, n_trials, lam_logistic, lam_l1, mask, y_spks_output, ielec):
        
        op = ss_alg.softmax_a_list(A, b, n_cells, n_shifts, n_trials, 
                                       lr=0.01, opt_th=1e-4, 
                                       n_iter=6000-1, x_art_init=None, tau_in=1.0, tau_change_freq=400, 
                                       lam_z=0, lam_logistic=lam_logistic, lam_l1=lam_l1, phase_freq=5000, n_outer_iter=0, mask=mask)
        
        y_spks, y_artifacts, spks_recons, art_recons, total_recons, sig_params, loss_log = op
        y_spks_output.update({ielec: y_spks})
        
    '''    
