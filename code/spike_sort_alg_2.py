'''Soft-max spike sorting with prior sigmoid knowledge'''

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
import tensorflow as tf

def spike_sort(A, b, n_cells, n_shifts, n_trials, 
                           lr=0.01, opt_th=1e-4, 
                           n_iter=10000, x_art_init=None, 
                           tau_in=1.0, tau_change_freq = 100, n_outer_iter = 20,
                           lam_z=1, lam_l1=1, lam_logistic=1, phase_freq=1000, mask=None, q_in=None):
    
    
    n_spks_params = n_shifts * n_cells
    n_amps = b.shape[-1]
    n_artifact_params = A[0].shape[1] - n_spks_params
    total_params = A[0].shape[1]
    #print(n_spks_params, n_artifact_params, total_params)
    
    A_arr = np.array(A)
    A_arr_spks = A_arr[:, :, :n_spks_params]
    A_arr_spks = np.reshape(A_arr_spks, [n_amps, -1, n_cells, n_shifts])
    A_arr_artifacts = A_arr[:, :, n_spks_params:]
    
    eta_spks = np.ones((n_amps, n_cells, n_shifts + 1, n_trials))  # (# cells x # shifts) x trials x 2 (one for L2 and another for sparsity)
    
    if x_art_init is None:
        y_artifacts = np.zeros((n_amps, n_artifact_params))  # T x trials
    else:
        y_artifacts = np.copy(x_art_init)

    with tf.Graph().as_default():
        
        if mask is not None:
            mask_use = tf.constant(mask.astype(np.float32))  # mask : # trials x n_amps
        else:
            mask_use = tf.constant(np.ones(n_trials, n_amps).astype(np.float32))
        
        # setup tensorflow variables
        eta_spks_tf = tf.Variable(eta_spks.astype(np.float32), name='eta_spks_tf')
        tau_tf = tf.placeholder(shape=[], dtype=tf.float32, name='tau')
        y_spks_tf = tf.nn.softmax(eta_spks_tf / tau_tf, dim=2) 
        
        #print(y_spks_tf)
        y_spks_tf = y_spks_tf[:, :, 1:, :] # n_amps, n_cells, n_shifts x n_trials
        '''
        sig_params = tf.Variable(np.append(0.05 * np.ones((n_cells, 1)), 
                                           1.25 * np.ones((n_cells, 1)), axis=1).astype(np.float32))
        xx = np.arange(n_amps).astype(np.float32)
        fx = (tf.expand_dims(sig_params[:, 0], 0) * tf.expand_dims(xx, 1) - 
              tf.expand_dims(sig_params[:, 1], 0))  # n_amps x n_cells
        '''
        if q_in is not None:
            fx = q_in.astype(np.float32)  # n_amps x n_cells
            ysum = (tf.reduce_sum(y_spks_tf, 2))  # n_amps x n_cells x n_trials
            logit_plus = - tf.nn.softplus(tf.expand_dims(- fx, 2))
            logit_minus = - tf.nn.softplus(tf.expand_dims(fx, 2))
            loss_logistic_cellwise =  -(ysum * logit_plus + (1-ysum) * logit_minus)
            loss_logistic_cellwise = tf.einsum('ijk,ki->ijk', loss_logistic_cellwise, mask_use)
            normalize_logistic = n_cells * tf.reduce_sum(mask_use)        
            loss_logistic = lam_logistic * tf.reduce_sum(loss_logistic_cellwise) / normalize_logistic 
        else: 
            loss_logistic = tf.constant(np.float32(0.0))
        
        y_artifacts_tf = tf.Variable(y_artifacts.astype(np.float32), name='y_artifacts_tf')  # n_amps x n_artifact_dims
        
        # Get reconstruction filters
        A_spks_tf = tf.constant(A_arr_spks.astype(np.float32), name='A_spks')  # amp x rec_length x n_cells x n_shifts
        A_artifacts_tf = tf.constant(A_arr_artifacts.astype(np.float32), name='A_arts')  # n_amps x rec_length x n_art_dims
        
        # reconstruct spike waveform
        spks_recons = tf.einsum('ijkl,iklm->ijm', A_spks_tf, y_spks_tf)
        spks_recons = tf.transpose(spks_recons, [1, 2, 0])  # rec_length x n_trials x n_amps
        
        art_recons = tf.einsum('ijk,ik->ij', A_artifacts_tf, y_artifacts_tf) 
        art_recons = tf.transpose(art_recons, [1, 0])  # rec_length x n_amps
        
        total_recons = spks_recons + tf.expand_dims(art_recons, 1)  # rec_length x n_trials  x n_amps
        # TODO(bhaishahster): Add a way to  
        
        recons_err = (total_recons - b) ** 2
        recons_err = recons_err * mask_use
        normalize_recons_err = b.shape[0] * tf.reduce_sum(mask_use)
        loss_recons = tf.reduce_sum(recons_err) / normalize_recons_err
        
        y_spks_tf = tf.einsum('ijkl,li->ijkl', y_spks_tf, mask_use)
        normalize_y_spks = tf.reduce_sum(mask_use) * n_cells * n_shifts
        loss_ysparse = lam_l1 * tf.reduce_sum(y_spks_tf) / normalize_y_spks

        loss = loss_recons + loss_ysparse + loss_logistic
        
        opt_op1 = tf.train.AdamOptimizer(lr).minimize(loss, 
                                                      var_list=[eta_spks_tf, y_artifacts_tf])
       
        proj_eta1 = tf.assign(eta_spks_tf, tf.maximum(eta_spks_tf, -5))
        with tf.control_dependencies([proj_eta1]):
            proj_eta2 = tf.assign(eta_spks_tf, tf.minimum(eta_spks_tf, 5))
        proj_eta = tf.group(proj_eta1, proj_eta2)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tau = tau_in
            loss_log= []

            loss_prev = np.inf
            eps = 1e-4
            iter_cnt = -1
            for iiter in range(n_iter):
                iter_cnt += 1
                 
                l_np , lnp_recons, lnp_ysparse, lnp_logistic, _ = sess.run([loss, loss_recons, 
                                                                           loss_ysparse, loss_logistic, 
                                                                           opt_op1], 
                                                                          feed_dict={tau_tf: np.float32(tau)})
                sess.run(proj_eta)
                    
                loss_log += [[l_np , lnp_recons, lnp_ysparse, lnp_logistic]]
                
                #print('.', end='', flush=True)
                if np.abs(l_np - loss_prev)/np.abs(loss_prev) < eps or iter_cnt > tau_change_freq:
                    tau = 0.8 * tau
                    iter_cnt = -1
                    #print(iiter, 'Change tau %.3f' % tau)
                    #print(iiter, iter_cnt, l_np)

                    if tau < 0.0001:
                        break
                
                loss_prev = l_np
            #print('phase1', iiter,  l_np , lnp_recons, lnp_ysparse, lnp_logistic, tau, sess.run(eta_spks_tf).min(), sess.run(eta_spks_tf).max())
            op = sess.run([y_spks_tf, y_artifacts_tf, spks_recons, art_recons, total_recons, tf.constant(np.float32(0.0))], 
                          feed_dict={tau_tf: np.float32(tau)})
    return op + [loss_log]


