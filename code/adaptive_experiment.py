import numpy as np
import pickle



# Adaptive stimulation 2: treat different amplitudes for same electrode independently
def adaptive_experiment(metrics, model, retina, adaptive_stim_alg, ntrials_per_phase=5, trials_max = 25, n_amps=38, n_elecs=512, elec_consider=None, save_name=None):


    # first phase - non adaptive
    trial_elec_amps=np.ones((n_elecs, n_amps)) * ntrials_per_phase    
    if elec_consider is not None:
        print('Consider only a few electrodes')
        t = 0 * trial_elec_amps
        t[elec_consider, :] = trial_elec_amps[elec_consider, :]
        trial_elec_amps = t
           
    # print('Dummy experiment on 10 electrodes')
    spks_coll = retina.stimulate(trial_elec_amps)
    model.update(spks_coll)
    metrics.update(model, trial_elec_amps)
    if save_name is not None:
        pickle.dump({'model': model, 'metrics': metrics, 'trial_elec_amps': trial_elec_amps, 'spks_coll': spks_coll, 'retina': retina}, 
                   open(save_name + '_init' + '.pkl', 'wb'))
    
    print('Non adaptive : %d trials, loss %.5f' % (ntrials_per_phase, metrics.loss_sig[-1]))
    if len(metrics.loss_pest_smoothen_dec) > 0:
        print('Non adaptive: Evaluate loss, decoder weighted %.5f' % (metrics.loss_pest_smoothen_dec[-1]))
    
    # Solve for number of trials for each electrode
    for iphase in range(np.int((trials_max - ntrials_per_phase) / ntrials_per_phase)):

        # Choose adaptive stimulation pattern
        trial_elec_amps = adaptive_stim_alg(model, metrics, ntrials_per_phase)
        if elec_consider is not None:
            print('Consider only a few electrodes')
            t = 0 * trial_elec_amps
            t[elec_consider, :] = trial_elec_amps[elec_consider, :]
            trial_elec_amps = t
        
        print('%d: Choose stimulations' % (iphase + 1))
        if np.min(np.min(trial_elec_amps)) < 0:
            from IPython import embed; embed()
          
        # Stimulate - TODO: Sample real responses and do spike sorting instead! 
        spks_coll_new = retina.stimulate(trial_elec_amps)
        spks_coll = merge_data(spks_coll, spks_coll_new) 
        print('%d: Stimulate' % (iphase + 1))
        print('%d: Spike sort' % (iphase + 1))

        # Update model
        model.update(spks_coll)
        print('%d: Update model' % (iphase + 1))

        # Evaluate model performance
        metrics.update(model, trial_elec_amps)
        print('%d: Evaluate loss %.5f' % (iphase + 1, metrics.loss_sig[-1]))
        if len(metrics.loss_pest_smoothen_dec) > 0:
            print('%d: Evaluate loss, decoder weighted %.5f' % (iphase + 1, metrics.loss_pest_smoothen_dec[-1]))
         
        if save_name is not None:
            pickle.dump({'model': model, 'metrics': metrics, 'trial_elec_amps': trial_elec_amps, 'spks_coll': spks_coll, 'retina': retina}, 
                        open(save_name + str(iphase) + '.pkl', 'wb'))


    return metrics


def merge_data(spks_coll, spks_coll_new): 

    for ielec in range(len(spks_coll)):
        for icell in range(len(spks_coll[0])):
            if len(spks_coll_new[ielec][icell][0]) != 0:
                spks_coll[ielec][icell][0] += spks_coll_new[ielec][icell][0]
                spks_coll[ielec][icell][1] += spks_coll_new[ielec][icell][1]
    
    return spks_coll



