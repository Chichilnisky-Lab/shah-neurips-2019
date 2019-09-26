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


def sigmoid(xs,m,b):
	return 1/ (1 + np.exp(-m*(xs - b)))

def getActCurve(amps, probs, xvals):
	popt, pcov = curve_fit(sigmoid, amps, probs)
	y = sigmoid(xvals,*popt)
	return y


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


