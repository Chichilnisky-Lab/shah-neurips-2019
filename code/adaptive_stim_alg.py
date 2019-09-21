import numpy as np
import cvxpy as cp
import tensorflow as tf



def adaptive_stim(model, metrics, ntrials_per_phase, n_amps=38,  n_trials_max_global=25):
    
    probs_est = model.probs_est 
    probs_smoothen_recons = model.probs_smoothen_recons
    ntrials_cumsum = metrics.ntrials_cumsum
    
    # Solve a convex problem!
    var_elec_cell = np.nansum(probs_est * (1 - probs_est), 1) 
    var_elecs = np.nansum(var_elec_cell, 1)
    T = cp.Variable(512)
    objective = cp.Minimize(cp.sum(var_elecs * cp.inv_pos(T + np.sum(ntrials_cumsum, 1) / n_amps)))
    constraints = [cp.sum(T) <= ntrials_per_phase  * 512, 0 <= T]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    n_trials_per_elec = (T.value)
    print('# Trials per electrode computed')
    
    # Convert # stimulations per electrode to elec_amp
    diff = np.abs(probs_est - probs_smoothen_recons) ** 2
    diff = np.sum(diff, 2)  # collapse across cells
    diff = diff / np.expand_dims(np.sum(diff, 1), 1)  # normalize
    trial_elec_amps = np.zeros((512, n_amps))
    for ielec in range(512):
        if n_trials_per_elec[ielec] > 0:
            trial_elec_amps[ielec, :] = np.round(n_amps * diff[ielec, :n_amps] * n_trials_per_elec[ielec])
    
    trial_elec_amps = np.maximum(trial_elec_amps, 0)
    # replace nans with 0
    trial_elec_amps[np.isnan(trial_elec_amps)] = 0
    
    return trial_elec_amps


def adaptive_stim_2(model, metrics, ntrials_per_phase, n_amps=38,  n_trials_max_global=25):
    
    probs_smoothen_recons = model.probs_smoothen_recons
    ntrials_cumsum = metrics.ntrials_cumsum
    probs_variance = model.probs_variance
     
    # Solve a convex problem!
    var_elec_cell = np.nansum(probs_variance, 1) 
    var_elecs = np.nansum(var_elec_cell, 1)
    T = cp.Variable(512)
    objective = cp.Minimize(cp.sum(var_elecs * cp.inv_pos(T + np.sum(ntrials_cumsum, 1) / n_amps)))
    constraints = [cp.sum(T) <= ntrials_per_phase  * 512, 0 <= T]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    n_trials_per_elec = (T.value)
    print('# Trials per electrode computed')
    
    # Convert # stimulations per electrode to elec_amp
    diff = probs_variance
    diff = np.sum(diff, 2)  # collapse across cells
    diff = diff / np.expand_dims(np.sum(diff, 1), 1)  # normalize
    trial_elec_amps = np.zeros((512, n_amps))
    for ielec in range(512):
        if n_trials_per_elec[ielec] > 0:
            trial_elec_amps[ielec, :] = np.round(n_amps * diff[ielec, :n_amps] * n_trials_per_elec[ielec])
    
    trial_elec_amps = np.maximum(trial_elec_amps, 0)
    # replace nans with 0
    trial_elec_amps[np.isnan(trial_elec_amps)] = 0
    
    return trial_elec_amps


def adaptive_stim_3(model, metrics, ntrials_per_phase, n_amps=38,  n_trials_max_global=25, use_decoder=False):
    
    probs_smoothen_recons = model.probs_smoothen_recons
    ntrials_cumsum = metrics.ntrials_cumsum
    probs_variance = model.probs_variance
     
    # Solve a convex problem!
    n_elecs = probs_smoothen_recons.shape[0]
    var_elec_amp_cell = probs_smoothen_recons * (1 - probs_smoothen_recons)
    if use_decoder:
        # use decoder
        print('Using decoder')
        decoder = model.decoder
        dec_energy = np.sum(decoder ** 2, 0)
        var_elec_amp_cell = var_elec_amp_cell * dec_energy
   
     
    var_elecs_amp = np.nansum(var_elec_amp_cell, 2)
    var_elecs_amp = np.ndarray.flatten(var_elecs_amp)
    print('Added extra statement :  var_elecs_amp = np.minimum(var_elecs_amp, 0.00000000001)')
    var_elecs_amp = np.minimum(var_elecs_amp, 0.000000001)
     
    T = cp.Variable(n_elecs * n_amps)
    objective = cp.Minimize(cp.sum(var_elecs_amp * cp.inv_pos(T + np.ndarray.flatten(ntrials_cumsum))))
    constraints = [cp.sum(T) <= ntrials_per_phase  * n_elecs * n_amps, 0 <= T]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    trial_elec_amps = (T.value).astype(np.int)
    
    trial_elec_amps = np.reshape(trial_elec_amps, [n_elecs, n_amps]) 
    trial_elec_amps[np.isnan(trial_elec_amps)] = 0
    
    return trial_elec_amps

def adaptive_stim_4(model, metrics, ntrials_per_phase, n_amps=38,  n_trials_max_global=25, use_decoder=False):
    
    probs_smoothen_recons = model.probs_smoothen_recons
    ntrials_cumsum = metrics.ntrials_cumsum
    probs_variance = model.probs_variance
     
    # Solve a convex problem!
    n_elecs = probs_smoothen_recons.shape[0]
    var_elec_amp_cell = probs_variance # probs_smoothen_recons * (1 - probs_smoothen_recons)
    if use_decoder:
        # use decoder
        print('Using decoder')
        decoder = model.decoder
        dec_energy = np.sum(decoder ** 2, 0)
        var_elec_amp_cell = var_elec_amp_cell * dec_energy
   
     
    var_elecs_amp = np.nansum(var_elec_amp_cell, 2)
    var_elecs_amp = np.ndarray.flatten(var_elecs_amp)
    print('scaled up the variance')
    var_elecs_amp *= np.ndarray.flatten(ntrials_cumsum)
    print('Added extra statement :  var_elecs_amp = np.minimum(var_elecs_amp, 0.00000000001)')
    var_elecs_amp = np.minimum(var_elecs_amp, 0.000000001)
     
    T = cp.Variable(n_elecs * n_amps)
    objective = cp.Minimize(cp.sum(var_elecs_amp * cp.inv_pos(T + np.ndarray.flatten(ntrials_cumsum))))
    constraints = [cp.sum(T) <= ntrials_per_phase  * n_elecs * n_amps, 0 <= T]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    trial_elec_amps = (T.value).astype(np.int)
    
    trial_elec_amps = np.reshape(trial_elec_amps, [n_elecs, n_amps]) 
    trial_elec_amps[np.isnan(trial_elec_amps)] = 0
    
    return trial_elec_amps

def adaptive_stim_5(model, metrics, ntrials_per_phase, 
                    n_amps=38,  n_trials_max_global=25, 
                    use_decoder=False, use_elec_weights=False):
    '''Do crazy stuff with fisher information matrix.'''
    probs_smoothen_recons = model.probs_smoothen_recons
    ntrials_cumsum = metrics.ntrials_cumsum
     
    # Solve a convex problem!
    elecs, cells = np.where(probs_smoothen_recons.sum(1) > 0) 
    p = probs_smoothen_recons[elecs, :, cells] 
    
    if use_elec_weights: 
        # get weights on different electrodes
        if not use_decoder:
            raise ValueError('How do we select electrode weights if no decoder used? ')
        elec_weights = get_elec_weights(model.decoder, probs_smoothen_recons, n_targets=20)        
    else:
        elec_weights = None

    if use_decoder:
        # use decoder
        print('Using decoder')
        decoder = model.decoder
    else:
        decoder = None
   
    trial_elec_amps = run_tf(p, elecs, cells, ntrials_cumsum.astype(np.float32), capacity=ntrials_per_phase * n_amps * 512 , n_amps=n_amps, decoder=decoder, elec_weights=elec_weights)
      
    return trial_elec_amps

def get_elec_weights(decoder, dict_, stix_sz=64, n_targets=40):
    '''How frequently are different electrodes used?'''
    
    stas_norm = np.sum(decoder ** 2, 0)
    dict_est_2d = np.reshape(dict_, [-1, dict_.shape[-1]])
    var_dict_est = np.squeeze(stas_norm * (dict_est_2d * (1 - dict_est_2d))).sum(-1)
    
    dict_size = dict_est_2d.shape[0]

    from numpy.random import RandomState
    prng = RandomState(50)
    dims = [int(320 / stix_sz), int(640 / stix_sz)]
    stims = np.repeat(np.repeat(prng.randn(dims[0], dims[1], 20), stix_sz / 8, axis=0),  stix_sz / 8, axis=1)
    stims = np.reshape(stims, [-1, stims.shape[-1]])
    stims = (stims > 0) - 0.5
    stims *= np.expand_dims((decoder.sum(1) != 0).astype(np.float32), 1)

    w_est = _solve_stimulation_cp(var_dict_est, stims/10, decoder/10, dict_est_2d)
    sum_elecs = np.reshape(w_est.sum(1), [dict_.shape[0], dict_.shape[1]] ).sum(1)    
    elec_weights = dict_.shape[1] * sum_elecs / np.sum(sum_elecs)
    
    return elec_weights

def _solve_stimulation_cp(var_dict_est, stim, decoder, dict_est_2d):
    # compute stimulation using estimated dictionary

    dict_size = dict_est_2d.shape[0]
    n_targets = stim.shape[1]
    w = cp.Variable((dict_size, n_targets))
    objective = cp.Minimize(cp.sum(var_dict_est * w) + cp.sum((stim - decoder * (dict_est_2d.T * w)) ** 2))
    constraints = [w >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    w = w.value
    return w


def run_tf(p, elecs, cells, n0, capacity, n_amps=38, decoder=None, eps=1e-3, elec_weights=None): 
   '''Distributre stimulations to electrodes and amplitudes from a budget.

   Args: 
       p : Probabilities. Electrode - cell pair x n_amps
       elecs : Electrodes for each of the pair
       n0 : Previous number of trials; Electrodes x n_amps
       capacity : Total number of stimulations
       n_amps : Number of amplitudes
   Returns: 
       n : How many times to stimulate. Electrodes x n_amps
   '''
   p = np.float32(p)
   entr = np.float32(p * (1 - p))
   
     
   xx_use = np.linspace(-1, 1, n_amps)
   x = np.float32(np.stack([np.squeeze(xx_use), np.ones(xx_use.shape[0])]))
   xxt = np.expand_dims(x, 1) * np.expand_dims(x, 0)
   xxt = np.transpose(xxt, [2, 0, 1])  # n_amps x 2 x 2
    
   with tf.Graph().as_default():
      tau = 1000
      n_raw = tf.Variable(np.zeros((512 * (n_amps + 1))).astype(np.float32))  # n_elecs x n_amps
      n = capacity * tf.nn.softmax(n_raw / tau)
      n = tf.reshape(n, [512, n_amps + 1])
      n = n[:, :n_amps]
      
      n_total = n + n0
      weighted_x = tf.gather(n_total, elecs) * entr # pairs x n_amps
      fisher_info_raw = tf.reduce_sum(tf.expand_dims(tf.expand_dims(weighted_x, 2), 3) * xxt, 1)  # pairs x 2 x 2
      fisher_info = fisher_info_raw + 0.0001 * tf.eye(2)

      covar = tf.matrix_inverse(fisher_info)  # pairs x 2 x 2
      A = tf.expand_dims(p * (1 - p), 2) *  x.T  # pairs x n_amps x 2
      Acovar = tf.einsum('ijk,ikl->ijl', A, covar)  # pairs x n_amps x 2
      AcovarA = tf.einsum('ijk,ilk->ijl', Acovar, A)  # pairs x n_amps x n_amps
      variance = tf.einsum('ill->i', AcovarA)  # pairs 
        
      if decoder is not None:
         dec_energy = np.sum(decoder ** 2, 0)
         dec_energy_cells = dec_energy[cells]  # pairs
         variance = variance * np.float32(dec_energy_cells)
         eps = 1e-5
      
      if elec_weights is not None:
         elec_w = tf.gather(elec_weights.astype(np.float32), elecs) 
         variance = elec_w * variance
           
      total_variance = tf.reduce_sum(variance)
      train_op = clipped_train_op(total_variance, 1, 0.5)
      # train_op = tf.train.AdamOptimizer(0.1).minimize(total_variance)
    
      with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
        
         loss_log = []
         l_prev = np.inf
         for iiter in range(20000):
            if iiter % 100 == 1:
               if np.abs(loss_log[-1] - l_prev) < eps:
                  break
               l_prev = loss_log[-1]
               print('--%d, %.5f--' % (iiter, loss_log[-1]), end='', flush=True) 
            _, tv, n_np, vv = sess.run([train_op, total_variance, n_total, AcovarA])
            loss_log += [tv]
             
         return np.round(sess.run(n)).astype(np.int)            
         
             
def clipped_train_op(loss, learning_rate, clip_norm):
    opt =  tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = opt.compute_gradients(loss)
    capped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads_and_vars]
    opt_fn = opt.apply_gradients(capped_grads_and_vars)
    return opt_fn


def fixed_stim(model, metrics, ntrials_per_phase, n_amps=38,  n_trials_max_global=25):
    
    trial_elec_amps = np.ones((512, n_amps)) * ntrials_per_phase
    return trial_elec_amps
