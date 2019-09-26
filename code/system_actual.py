import numpy as np
                
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

    
