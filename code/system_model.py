import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
import tensorflow as tf

import sys
sys.path.append('/Volumes/Lab/Users/Sasi/pystim')
import eilib as eilib

import matplotlib
import matplotlib.pyplot as plt

def get_probs(sig_params):
    
    amp = np.expand_dims(np.expand_dims(np.arange(0, 38),1),2)
    sigmoid_params1, sigmoid_params2 = sig_params
    cell_elec = np.double(sigmoid_params2 != 0)
    cell_act = (1/(1+np.exp(-(amp * sigmoid_params2.T + sigmoid_params1.T)))) * cell_elec.T
    cell_act = np.transpose(cell_act, [1, 0, 2])
    return cell_act
    

def get_raw_probs(spks_coll):
    
    n_elecs = len(spks_coll)
    n_cells = len(spks_coll[0])
    
    probs_est = np.zeros((n_elecs, 38, n_cells))            
    for ielec in range(n_elecs):
        for icell in range(n_cells):
            amp_list = spks_coll[ielec][icell][0]
            if len(amp_list) == 0:
                continue
            ss = np.array(spks_coll[ielec][icell][1])
            for iamp in np.unique(amp_list):
                amp_idx = np.where(amp_list == iamp)[0].astype(np.int)
                probs_est[ielec, iamp, icell] = np.mean(ss[amp_idx])
    
    return probs_est

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-(b * (x - a))))

def fit_sigmoid_2(spks_coll):
    
    n_elecs = len(spks_coll)
    n_cells = len(spks_coll[-1])
    
    sig_p2 = np.zeros((n_elecs, n_cells))
    sig_p1 = np.zeros((n_elecs, n_cells))
    for ielec in range(n_elecs):
        #print(ielec)
        for icell in range(n_cells):
            if len(spks_coll[ielec][icell][0]) == 0: 
                continue
                
            if np.sum(spks_coll[ielec][icell][1]) == 0: # Zero spikes! 
                continue
            
             
            X = np.expand_dims(spks_coll[ielec][icell][0], 1)
            y = spks_coll[ielec][icell][1]
            clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X, y)
            # print(clf.coef_, clf.intercept_)
            
            sig_p1[ielec, icell] = clf.intercept_
            sig_p2[ielec, icell] = clf.coef_
            
            
    return sig_p1.T, sig_p2.T


def fit_model(spks_coll):
    probs_est = get_raw_probs(spks_coll)
    sigmoid_params = fit_sigmoid_2(spks_coll)
    probs_smoothen_recons = get_probs(sigmoid_params)
    
    return probs_est, probs_smoothen_recons, sigmoid_params

class Model(object):
    '''Model probabilities by empirical averaging.'''
    def __init__(self, cids, decoder=None):
        self.probs_est = None
        self.probs_smoothen_recons = None
        self.sigmoid_params = None
        self.cids = cids 
        
        if decoder is not None:
           self.decoder = decoder
         
    def update(self, spks_coll):
        self.probs_est, self.probs_smoothen_recons, self.sigmoid_params = fit_model(spks_coll)
        self.probs_variance = self.probs_smoothen_recons * (1 - self.probs_smoothen_recons) 


class Hierarchical1(object):
    '''Variational inference to account for prior knowledge about EI amplitudes'''
    def __init__(self, ei, cids, n_amps=38, use_prior=True, vi_global=False, model_interaction=False, xy_priors=None, decoder=None):
        self.ei = ei
        self.ei_amp = ei.min(2)
        
        self.probs_est = None
        self.probs_smoothen_recons = None
        self.sigmoid_params = None
        
        self.cids = cids 
        self.n_cells = np.squeeze(np.array(cids)).shape[0]
        
        self.s1_mean_param = None
        self.s1_sd_param = None
        self.s2_mean_param = None
        self.s2_sd_param = None
        
        self.n_amps = n_amps
             
        self.use_prior = use_prior
        self.vi_global = vi_global

        self.model_interaction = model_interaction
         
        if decoder is not None:
            self.decoder = decoder
        
        # Get axon - soma type information.
        # self.ei_type_dict = {'dendrite': 0, 'messy': 1, 'mixed': 2, 'axon': 3, 'soma': 4}
        self.ei_type_dict = {'dendrite': 3, 'messy': 3, 'mixed': 3, 'axon': 3, 'soma': 4}
        ei_type = np.zeros((ei.shape[0], ei.shape[1]))
        # print('Setting all cells to soma')
        for i in range(ei.shape[0]):
            for j in range(ei.shape[1]):
                ei_type_str = eilib.axonorsoma(self.ei[i, j, :])         
                ei_type[i, j] = self.ei_type_dict[ei_type_str]
        self.ei_type = ei_type
        
                               
        self.xy_priors = None
        if xy_priors is not None:
            self.xy_priors = {}
            self.xy_priors.update({self.ei_type_dict['soma']: xy_priors['soma'] })
            self.xy_priors.update({self.ei_type_dict['axon']: xy_priors['axon'] })  
     
    def set_default_prior(self):
        self.s2_mean_param = np.array([34.68914748, 20.95096066]).astype(np.float32)
        self.s2_sd_param = np.float32(2.9317003376201343)
        self.s1_mean_param = None
        self.s1_sd_param = None
        
        
    def update(self, spks_coll):
        
        self.set_default_prior()
         
        rr, n_trials, ei_amp_log, elec_cell_log, ei_type_log, ss_params = self._prepare_data(spks_coll) 
        # get raw probabilities
        self.probs_est = get_raw_probs(spks_coll)
        # Smoothen probabilities with variational inference

        params = self._vi(rr, n_trials, ei_amp_log, ei_type_log=ei_type_log)  #, ss_params=ss_params)
        self.probs_smoothen_recons, self.sigmoid_params, self.probs_variance = self._assemble(params, elec_cell_log)
        
        self.probs_est = self.probs_est[:, :self.n_amps, :]
        self.probs_smoothen_recons = self.probs_smoothen_recons[:, :self.n_amps, :]
        self.probs_variance = self.probs_variance[:, :self.n_amps, :]
    
    def _prepare_data(self, spks_coll):

        sigmoid_params = fit_sigmoid_2(spks_coll)  
        sigmoid_params = np.array(sigmoid_params)
        
        ss_params = []
        rr = []
        n_trials = []
        ei_amp_log = []
        elec_cell_log = []
        ei_type_log = []
        for ielec in range(len(spks_coll)):
            for icell in range(len(spks_coll[ielec])):
                if len(spks_coll[ielec][icell][0]) > 0 : 
                    irr = np.zeros(self.n_amps)
                    itr = np.zeros(self.n_amps)
                    for iamp in range(self.n_amps):
                        entries = np.array(spks_coll[ielec][icell][0]) == iamp
                        irr[iamp] = np.mean(np.array(spks_coll[ielec][icell][1])[entries])
                        itr[iamp] = np.sum(entries)
                    rr += [irr]
                    n_trials += [itr]
                    ei_amp_log += [ - self.ei_amp[icell, ielec]]
                    elec_cell_log += [[ielec, icell]]
                    ei_type_log += [self.ei_type[icell, ielec]]
                    ss_params += [ sigmoid_params[:, icell, ielec] ]
                     
        rr = np.array(rr)
        n_trials = np.array(n_trials) 
        ei_amp_log = np.array(ei_amp_log)
        elec_cell_log = np.array(elec_cell_log)
        ei_type_log = np.array(ei_type_log)
        ss_params = np.array(ss_params)
        return rr, n_trials, ei_amp_log, elec_cell_log, ei_type_log, ss_params
 
    def _assemble(self, params, elec_cell_log, n_samples=1000):
        
        prob_recons = np.zeros((512, self.n_amps, self.n_cells))
        prob_var = np.zeros((512, self.n_amps, self.n_cells))
        sigmoid_params = np.zeros((512, self.n_cells, 2))
        
        s1_mean = params[0]
        s1_sd = params[1]
        s2_mean = params[2]
        s2_sd = params[3]
        
        amps = np.arange(self.n_amps).astype(np.float32)
        for iparam in range(s1_mean.shape[0]):
           
           ss1_mean = s1_mean[iparam]
           ss1_sd = s1_sd[iparam]
           ss2_mean = s2_mean[iparam]
           ss2_sd = s2_sd[iparam]
           ielec, icell = elec_cell_log[iparam, :]
           
           p_log = []
           for isample in range(n_samples):
               eps = np.random.randn(1)
               ss1 = ss1_sd * eps + ss1_mean
               
               eps = np.random.randn(1) 
               ss2 = ss2_sd * eps + ss2_mean
               p_log += [1 / (1 + np.exp(-ss1 * (amps - ss2))) ]
           
           p_log = np.array(p_log)
           p_var = np.nanvar(p_log, 0)
           p_log = np.nanmean(p_log, 0)
           prob_recons[ielec, :, icell] = p_log 
           prob_var[ielec, :, icell] = p_var         
           sigmoid_params[ielec, icell, :] = [ss2_mean, ss1_mean]   # careful - its reversed! 
        
        return prob_recons, sigmoid_params, prob_var  
         

    def _vi(self, rr, n_trials, ei_amp_log, n_samples=10, ei_type_log=None, ss_params=None):
        # make rr, cells x amps. mane n_trials: cells x amps
                                           
        n_cells, n_amps = rr.shape
        with tf.Graph().as_default():
            
            # Data
            amps = tf.constant(np.arange(n_amps).astype(np.float32))
            y = tf.constant(rr.astype(np.float32))
            ei_amp_tf = tf.constant(ei_amp_log.astype(np.float32))
 
            # Variational parameter
            s1_mean = tf.Variable(5 * np.ones(n_cells).astype(np.float32))
            s1_sd = tf.Variable(5 * np.ones(n_cells).astype(np.float32))
            s2_mean = tf.Variable(20 * np.ones(n_cells).astype(np.float32))
            s2_sd = tf.Variable(5 * np.ones(n_cells).astype(np.float32))

            eps1 = tf.random_normal((n_samples, n_cells))
            eps2 = tf.random_normal((n_samples, n_cells))
            s1 = eps1 * s1_sd + s1_mean
            s2 = eps2 * s2_sd + s2_mean

            if self.vi_global: # Shall we do VI on prior?
                s2_mean_param_phi_mean = tf.Variable(self.s2_mean_param.astype(np.float32))
                s2_mean_param_phi_sd = tf.Variable(10 * np.ones(2).astype(np.float32))
                eps3 = tf.random_normal([n_samples, 2])
                s2_mean_param = s2_mean_param_phi_sd * eps3 + s2_mean_param_phi_mean

                s2_sd_param_phi_mean = tf.Variable(np.float32(self.s2_sd_param))
                s2_sd_param_phi_sd = tf.Variable(np.float32(1))
                eps4 = tf.random_normal([n_samples, 1])
                s2_sd_param = s2_sd_param_phi_sd * eps4 + s2_sd_param_phi_mean
                
                loss_entropy_global = tf.reduce_sum(-0.5 * tf.log(tf.pow(s2_mean_param_phi_sd, 2) ) - 
                                                     0.5 * tf.log(tf.pow(s2_sd_param_phi_sd, 2)) ) 
            '''
            # TODO(bhaishahster) : Implement this!  
            if self.model_interaction:
                # See if a cell-electrode pair even interacts!  
                logits = tf.Variable(np.zeros(n_cells, 2).astype(np.float32))
                temperature = 0.1
                dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
                q = dist.sample()
            '''
                
            # Prediction
            amps = np.arange(n_amps).astype(np.float32)
            logit = tf.expand_dims(s1, 2) * (tf.expand_dims(tf.expand_dims(amps, 0), 0) - 
                                             tf.expand_dims(s2, 2))  # ncells x namps
            logp_plus = - tf.nn.softplus(-logit)
            logp_neg = - tf.nn.softplus(logit)
            loss_pred = - tf.reduce_sum(y * logp_plus + (1 - y) * logp_neg, 0) / n_samples

            if self.use_prior:  
                if not self.vi_global : 
                    # Prior
                    s2_proir_params_tf = tf.constant(np.array(self.s2_mean_param).astype(np.float32))
                    s2_prior_mean = s2_proir_params_tf[0] / ei_amp_tf + s2_proir_params_tf[1]
                    s2_prior_sd = tf.constant(np.float32(self.s2_sd_param))
                else: 
                    s2_prior_mean = tf.expand_dims(s2_mean_param[:, 0], 1) / ei_amp_tf + tf.expand_dims(s2_mean_param[:, 1], 1)
                    s2_prior_sd = s2_sd_param
                
                loss_prior = tf.reduce_sum(tf.pow(s2_prior_mean - s2, 2) / (2 * tf.pow(s2_prior_sd, 2)) - tf.log(tf.pow(s2_prior_sd, 2) )) / n_samples  # tf.log(s1_normal.prob(s1))
                     
            else: 
                loss_prior = 0

            loss_entropy = tf.reduce_sum(-0.5 * tf.log(tf.pow(s1_sd, 2) ) - 0.5 * tf.log(tf.pow(s2_sd, 2) ))
            if self.vi_global and self.use_prior:
                loss_entropy += loss_entropy_global 
            loss = tf.reduce_sum(n_trials * loss_pred) + loss_prior + loss_entropy
            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                l_log = []
                for _ in range(100000):
                    _, loss_np = sess.run([train_op, loss])
                    l_log += [loss_np]

                params = sess.run([s1_mean, s1_sd, s2_mean, s2_sd])
            
            return params

 
class Hierarchical2(Hierarchical1):
    '''Variational inference to account for prior knowledge about EI amplitudes, EI type from previous experiments'''
      
    def _get_phi_xy_nu(self, n_samples):
       phi_xy_mean = tf.Variable(np.array([30.0, 30.0]).astype(np.float32))
       phi_xy_sd = tf.Variable(10 * np.ones(2).astype(np.float32))
       eps3 = tf.random_normal([n_samples, 2])
       q_xy = phi_xy_sd * eps3 + phi_xy_mean
       
       phi_nu_mean = tf.Variable(np.float32(2))
       phi_nu_sd = tf.Variable(np.float32(1))
       eps4 = tf.random_normal([n_samples, 1])
       q_nu = phi_nu_sd * eps4 + phi_nu_mean     
       phi_xy_nu_dict = {'phi_xy_mean': phi_xy_mean, 'phi_xy_sd': phi_xy_sd, 
                  'phi_nu_mean': phi_nu_mean, 'phi_nu_sd': phi_nu_sd}
       return q_xy, q_nu, phi_xy_nu_dict
        
    def _vi(self, rr, n_trials, ei_amp_log, n_samples=10, ei_type_log=None, ss_params=None):
        # make rr, cells x amps. mane n_trials: cells x amps
                                           
        n_cells, n_amps = rr.shape
        ei_type_log = ei_type_log.astype(np.int)
        u_ei, ei_type_indx = np.unique(ei_type_log, return_inverse=True)
          
        with tf.Graph().as_default():
            
            # Data
            print('Set up data')
            amps = tf.constant(np.arange(n_amps).astype(np.float32))
            y_rr = tf.constant(rr.astype(np.float32))
            ei_amp_tf = tf.constant(ei_amp_log.astype(np.float32))
            ei_type_tf = tf.transpose(tf.one_hot(ei_type_indx, depth=u_ei.shape[0]))

             
            # Variational parameter for 'a' and 'b'
            print('Variational parameterization of a and b')
            # Initialize using logistic regression
            if ss_params is not None:
                a_init = ss_params[:, 1]
                b_init = - ss_params[:, 0] / ss_params[:, 1]
                # give min and max range
                b_init[np.isnan(b_init)] = 20
                b_init[b_init < 0] = 20
                b_init[b_init > 40] = 20
                
            else: 
                a_init = 5 * np.ones(n_cells)
                b_init = 20 * np.ones(n_cells)
            
            phi_a_mean = tf.Variable(a_init.astype(np.float32))
            phi_a_sd = tf.Variable(5 * np.ones(n_cells).astype(np.float32))
            phi_b_mean = tf.Variable(b_init.astype(np.float32))
            phi_b_sd = tf.Variable(5 * np.ones(n_cells).astype(np.float32))
            eps1 = tf.random_normal((n_samples, n_cells))
            eps2 = tf.random_normal((n_samples, n_cells))
            q_a = eps1 * phi_a_sd + phi_a_mean
            q_b = eps2 * phi_b_sd + phi_b_mean
            loss_entropy = tf.reduce_sum( -0.5 * tf.log(tf.pow(phi_a_sd, 2)) 
                                          -0.5 * tf.log(tf.pow(phi_b_sd, 2)) ) 


            # Variational parameter for 'x, y' and 'nu'
            print('Variational parametrization of x and y')
            phi_log = {}
            x_log = []
            y_log = []
            nu_log = []
            for itype in u_ei:
                q_xy, q_nu, phi_xy_nu_dict = self._get_phi_xy_nu(n_samples)
                phi_log.update( {itype: {'q_xy': q_xy, 'q_nu': q_nu}} )
                loss_entropy += tf.reduce_sum(-0.5 * tf.log(tf.pow(phi_xy_nu_dict['phi_xy_sd'], 2)) 
                                              -0.5 * tf.log(tf.pow(phi_xy_nu_dict['phi_nu_sd'], 2)) ) 
                x_log += [q_xy[:, 0]]
                y_log += [q_xy[:, 1]]
                nu_log += [q_nu[:, 0]]
            x_log = tf.transpose(tf.stack(x_log, axis=0))
            y_log = tf.transpose(tf.stack(y_log, axis=0))
            nu_log = tf.transpose(tf.stack(nu_log, axis=0))
              
            # Prediction
            amps = np.arange(n_amps).astype(np.float32)
            logit = tf.expand_dims(q_a, 2) * (tf.expand_dims(tf.expand_dims(amps, 0), 0) - 
                                             tf.expand_dims(q_b, 2))  # ncells x namps
            logp_plus = - tf.nn.softplus(-logit)
            logp_neg = - tf.nn.softplus(logit)
            loss_pred = - tf.reduce_sum(y_rr * logp_plus + (1 - y_rr) * logp_neg, 0) / n_samples
            
            x = tf.matmul(x_log, ei_type_tf)
            y = tf.matmul(y_log, ei_type_tf)
            nu = tf.matmul(nu_log, ei_type_tf)
            
            ##### Priors #####
            # Prior on b using x and y
            b_prior_mean = x + y / ei_amp_tf
            b_prior_sd = nu
            loss_prior = tf.reduce_sum(tf.pow(b_prior_mean - q_b, 2) / (2 * tf.pow(b_prior_sd, 2)) - tf.log(tf.pow(b_prior_sd, 2) )) / n_samples  # tf.log(s1_normal.prob(s1))
            
            # Prior on x, y
            if self.xy_priors is not None:
                print('Adding priors on x and y')
                for itype in self.xy_priors.keys():
                    shift = np.array(self.xy_priors[itype]['shift']).astype(np.float32)
                    rotate = np.array(self.xy_priors[itype]['rotate']).astype(np.float32)
                    xy = phi_log[itype]['q_xy']
                    gaussian_samples = tf.matmul((xy - shift), rotate)
                    loss_prior += 0.5 * tf.reduce_sum(tf.pow(gaussian_samples, 2)) / n_samples  # n_samples x 2
            
             
            # Prior on xy, nu ? 
            loss = tf.reduce_sum(n_trials * loss_pred) + loss_prior + loss_entropy
            '''
            train_op0 =  tf.train.AdamOptimizer(0.05).minimize(loss)  # tf.train.AdamOptimizer(0.01).minimize(loss)
            train_op1 =  tf.train.AdamOptimizer(0.005).minimize(loss) 
            train_op2 =  tf.train.AdamOptimizer(0.0005).minimize(loss) 
            '''
            train_op0 = self._clipped_train_op(loss, 0.05, clip_norm=0.1)
            train_op1 = self._clipped_train_op(loss, 0.005, clip_norm=0.1)
            train_op2 = self._clipped_train_op(loss, 0.0005, clip_norm=0.1)
           

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                l_log = []
                l_smooth = []
                iiter = -1
                while True : 
                    
                    iiter += 1
                    if iiter > 180000:
                        break
                     
                    if iiter < 60000 : 
                        train_op = train_op0
                    elif iiter < 120000 :
                        train_op = train_op1
                    else:
                        train_op = train_op2
                       
                    _, loss_np = sess.run([train_op, loss])
                    l_log += [loss_np]
                    l_smooth += [np.min(l_log[-100:])]
                    
                    if np.isnan(loss_np):
                       print('NaN detecrted .. Restarting..')
                       sess.run(tf.global_variables_initializer())
                       iiter = -1
                     
                    if (iiter+1) % 9999 == 0:
                       plt.cla()
                       #plt.plot(l_log[2000:])
                       plt.plot(l_smooth[2000:])
                       #plt.ylim([np.min(l_smooth), np.max(l_smooth[2000:])])
                       plt.show()
                    
                params = sess.run([phi_a_mean, phi_a_sd, phi_b_mean, phi_b_sd])
            return params
              
    def _clipped_train_op(self, loss, learning_rate, clip_norm):
        opt =  tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = opt.compute_gradients(loss)
        capped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads_and_vars]
        opt_fn = opt.apply_gradients(capped_grads_and_vars)
        return opt_fn


        



 
