import numpy as np
import cvxpy as cp

class Metrics(object):
    def __init__(self, dictionary, cids):
        self.loss = []
        self.loss_sig = []
        self.loss_pest_smoothen_dec = []
        
        self.nstims_log = []
        
        self.ntrials_per_elec_amps = []
        self.dictionary = dictionary
        self.cids = cids
        self.ntrials_cumsum = np.zeros((512, 38))
        
        self.dict_mismatch_stims = [] 
        self.dict_mismatch_percept_true_dict_log = [] 
        self.dict_mismatch_err_true_dict_log = [] 
        self.dict_mismatch_percept_est_dict_log = []
        self.dict_mismatch_err_est_dict_log = []

    def update(self, model, trial_elec_amps, p_th=0):
        
        probs_est = model.probs_est 
        probs_smoothen_recons = model.probs_smoothen_recons
        
        # extract probs_est, etc for relelvant cids
        if probs_est.shape[-1] != self.cids.shape[0]:
            probs_est = self._extract_cids(probs_est, model.cids, self.cids)     
        
        if probs_smoothen_recons.shape[-1] != self.cids.shape[0]:
            probs_smoothen_recons = self._extract_cids(probs_smoothen_recons, model.cids, self.cids)     
        
        cell_e  = np.expand_dims(self.dictionary.max(1) > p_th, 1)
        self.cell_e = cell_e
        l_pest_smoothen =  np.sum(((self.dictionary - probs_smoothen_recons) * cell_e) ** 2)
        l_pest =  np.sum(((self.dictionary - probs_est) * cell_e) ** 2)
        
        if hasattr(model, 'decoder'):
            dec_norm = np.sum(model.decoder ** 2, 0)
            l_p_dec =  np.sum( (self.dictionary - probs_smoothen_recons)**2 * dec_norm * cell_e)
            self.loss_pest_smoothen_dec += [l_p_dec]
                 
        self.loss += [l_pest]
        self.nstims_log += [np.sum(trial_elec_amps)]
        self.ntrials_per_elec_amps += [trial_elec_amps]
        self.ntrials_cumsum += trial_elec_amps
        self.loss_sig += [l_pest_smoothen]
        
        self.l_p_vary_log = []
        for p_th in np.arange(0, 1.0, 0.1):
            cell_e  = np.expand_dims(np.logical_and(self.dictionary.max(1) > p_th, self.dictionary.max(1) <= p_th + 0.1), 1)
            l_pest_smoothen =  np.sum(((self.dictionary - probs_smoothen_recons) * cell_e) ** 2)
            self.l_p_vary_log += [l_pest_smoothen]
        self.p_th_vary = np.arange(0, 1.0, 0.1)
       
        # If there is a decoder, then compute the perceptual error of using wrong dictionary
        '''
        if hasattr(model, 'decoder'):
           print('Finding perceptual error w.r.t. decoder')
           try:
              op = self._analyze_dictionary_accuracy(probs_smoothen_recons, self.dictionary, model.decoder, n_targets=5)
              self.dict_mismatch_stims += [op[0]] 
              self.dict_mismatch_percept_true_dict_log += [op[1]] 
              self.dict_mismatch_err_true_dict_log += [op[2]] 
              self.dict_mismatch_percept_est_dict_log += [op[3]]
              self.dict_mismatch_err_est_dict_log += [op[4]]
           except: 
              print('Error in decoder')
              self.dict_mismatch_stims += [np.nan]
              self.dict_mismatch_percept_true_dict_log += [np.nan] 
              self.dict_mismatch_err_true_dict_log += [np.nan] 
              self.dict_mismatch_percept_est_dict_log += [np.nan]
              self.dict_mismatch_err_est_dict_log += [np.nan]
        '''   

    def _analyze_dictionary_accuracy(self, dict_estimated, dict_true, decoder, stix_sz=32, n_targets=20):
        
        stas_norm = np.sum(decoder ** 2, 0)
        dict_est_2d = np.reshape(dict_estimated, [-1, dict_estimated.shape[-1]])
        dict_true_2d = np.reshape(dict_true, [-1, dict_true.shape[-1]])
         
        var_dict_est = np.squeeze(stas_norm * (dict_est_2d * (1 - dict_est_2d))).sum(-1)
        var_dict_true = np.squeeze(stas_norm * dict_true_2d * (1 - dict_true_2d)).sum(-1)
        
        dict_size = dict_est_2d.shape[0]
        
        from numpy.random import RandomState
        prng = RandomState(23) 
        dims = [int(320 / stix_sz), int(640 / stix_sz)]
        stims = np.repeat(np.repeat(prng.randn(dims[0], dims[1], 20), stix_sz / 8, axis=0),  stix_sz / 8, axis=1)
        stims = np.reshape(stims, [-1, stims.shape[-1]])
        stims = (stims > 0) - 0.5
        stims *= np.expand_dims((decoder.sum(1) != 0).astype(np.float32), 1)
        
        err_est_dict_log = []
        percept_est_dict_log = []
        err_good_dict_log = [] 
        percept_good_dict_log = []
        w_est = self._solve_stimulation_cp(var_dict_est, stims, decoder, dict_est_2d)
        w_true = self._solve_stimulation_cp(var_dict_true, stims, decoder, dict_true_2d)
        for itarget in np.arange(stims.shape[1]):
           print('.', end='', flush=True) 
           
           # use estimated dictionary 
           w = w_est[:, itarget] 
           error = var_dict_true.dot(w) + np.sum((stims[:, itarget] - decoder.dot(dict_true_2d.T.dot(w))) ** 2)
           err_est_dict_log += [error]
           percept = decoder.dot(dict_true_2d.T.dot(w))
           percept_est_dict_log += [percept]
           
           # use true dictionary 
           w = w_true[:, itarget]
           error = var_dict_true.dot(w) + np.sum((stims[:, itarget] - decoder.dot(dict_true_2d.T.dot(w))) ** 2)
           err_good_dict_log += [error]
           percept = decoder.dot(dict_true_2d.T.dot(w))
           percept_good_dict_log += [percept]
      
        percept_good_dict_log = np.array(percept_good_dict_log).T
        percept_est_dict_log = np.array(percept_est_dict_log).T
        err_est_dict_log = np.array(err_est_dict_log)
        err_good_dict_log = np.array(err_good_dict_log)
        
        return stims, percept_good_dict_log, err_good_dict_log, percept_est_dict_log, err_est_dict_log
    
    def _solve_stimulation_cp(self, var_dict_est, stim, decoder, dict_est_2d):    
        # compute stimulation using estimated dictionary
        
        dict_size = dict_est_2d.shape[0]
        n_targets = stim.shape[1]
        w = cp.Variable((dict_size, n_targets))
        objective = cp.Minimize(cp.sum(var_dict_est * w) + cp.sum((stim - decoder * (dict_est_2d.T * w)) ** 2))
        constraints = [w >= 0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=False)
        w = w.value
        return w
 
    def _extract_cids(self, probs_in, cids_true, cids_extract):
        
        probs_out = np.zeros((probs_in.shape[0], probs_in.shape[1], cids_extract.shape[0]))
        
        for iicids, icids in enumerate(cids_extract):
            xcid = np.where(cids_true == icids)[0][0]
            probs_out[:, :, iicids] = probs_in[:, :, xcid]
            
        return probs_out   
            
