"""
helper functions for EEG analyses

Created by Dirk van Moorselaar on 21-01-2018.
Copyright (c) 2018 DvM. All rights reserved.
"""

import numpy as np
from scipy.linalg import fractional_matrix_power
from IPython import embed 


def whiten_EEG(eegs, conditions):
	'''

	'''

	tr, el, tp = eegs.shape
	nr_c = np.unique(conditions).size
	# pre-allocate array
	E = np.empty((tp, nr_c, el, el)) * np.nan


	# loop over all timepoints
	for t in range(tp):
		for i, cnd in enumerate(np.unique(conditions)):

			# get condition mask
			cnd_mask = conditions == cnd
			X = eegs[cnd_mask,:,t]      # X = trials x channels 
			E[t,i,:,:] = covCor(X)

	EE = np.mean(E, axis = (0,1))	    # average over time and conditions

	# invert E: this is Mahalanobis or ZCA whitening
	IE = fractional_matrix_power(EE, -.5)

	# pre-allocate array
	whiten = np.empty(eegs.shape) * np.nan

	for t in range(tp):
		for n in range(tr):
			X = eegs[n,:,t]
			whiten[n,:,t] = np.dot(X,IE)

	return whiten		
 
def covCor(x, shrink = -1):
	'''

	'''

	# de-mean returns
	r, c = x.shape
	x -= np.tile(np.mean(x,axis = 0),(r,1))

	# compute sample covariance matrix
	sample = (1.0/r) * np.dot(x.T,x)

	# compute prior
	var = np.diag(sample)
	sqrtvar = np.sqrt(var)
	rBar = (np.sum(sample/ (np.tile(sqrtvar, (c,1)).T * np.tile(sqrtvar, (c,1)))) - c)/(c*(c-1))
	prior = rBar * np.tile(sqrtvar, (c,1)).T * np.tile(sqrtvar, (c,1))
	prior[np.identity(c, dtype = bool)] = var

	if shrink == -1:
		# compute shrinkage parameters and constant

		# what we call pi-hat
		y = x**2
		phi_mat = np.dot(y.T,y) / r - 2 * (np.dot(x.T,x)) * sample/r + sample**2
		phi = np.sum(phi_mat)

		# what we call rho-hat
		term1 = np.dot((x**3).T,x)/r
		helper = np.dot(x.T,x)/r
		help_diag = np.diag(helper)
		term2 = np.tile(help_diag,(c,1)).T * sample
		term3 = helper * np.tile(var,(c,1)).T
		term4 = np.tile(var,(c,1)).T * sample
		theta_mat = term1 - term2 -term3 + term4
		theta_mat[np.identity(c, dtype = bool)] = np.zeros(c)
		rho = np.sum(np.diag(phi_mat)) + rBar * np.sum(((1.0/sqrtvar) * sqrtvar.T) * theta_mat)

		# what we call gamma hat
		gamma = np.linalg.norm(sample - prior, ord = 'fro')**2

		# compute shrinkage constant
		kappa = (phi - rho) / gamma
		shrink = max(0, min(1, kappa/r))

	# compute the estimator
	sigma = shrink * prior + (1 - shrink) * sample	

	return sigma


def eog_filt(beh, EEG, heog, sfreq, windowsize = 50, windowstep = 25, threshold = 30):
	'''
	evaluate nan trials for eye movements

	'''

	windowstep /= 1000 / sfreq
	windowsize /= 1000 / sfreq

	# select trials with missing tracker data
	nan_idx = np.where(np.isnan(beh['eye_bins'])> 0)[0]
	s, e = 0, heog.shape[-1] - 1

	eye_trials = []
	for idx in nan_idx:
		
		for j in np.arange(s, e - windowstep, windowstep):

			w1 = np.mean(heog[idx][int(j):int(j + windowsize / 2) - 1])
			w2 = np.mean(heog[idx][int(j + windowsize / 2):int(j + windowsize) - 1])

			if abs(w1 - w2) > threshold:
				eye_trials.append(idx)
				break

	eye_trials = np.array(eye_trials, dtype = int)			

	print 'selected {} bad trials via eyethreshold'.format(eye_trials.size)	

	return eye_trials	


def paired_t(*args):
	"""Call scipy.stats.ttest_rel, but return only f-value."""
	from scipy.stats import ttest_rel
	return ttest_rel(*args)[0]

def bootstrap(X, b_iter = 10000):
	'''
	bootstrapSlopes uses a bootstrap procedure to calculate standard error of slope estimates.

	Arguments
	- - - - - 
	subject_id (list): list of subject id's 
	data (list): list of individual subjects slope dictionaries
	power (str): evoked or total power slopes
	info (dict): dictionary with variables set in spatialCTF
	b_iter (int): number of bootstrap iterations

	Returns
	- - - -
	bot(dict): dict of bootstrapped slopes
	'''	

	nr_obs = X.shape[0]
	bootstrapped = np.zeros((b_iter,X.shape[1]))

	for b in range(b_iter):
		idx = np.random.choice(nr_obs,nr_obs,replace = True) 				# sample nr subjects observations from the slopes sample (with replacement)
		bootstrapped[b,:] = np.mean(X[idx,:],axis = 0)

	error = np.std(bootstrapped, axis = 0)

	return error

def check_event_codes(events):
	'''
	Function that changes the relevant trigger codes based on behavioral log
	'''

	# read in triggers from behavioral file
	nr_practice = 16
	file = '/Users/dirk/Desktop/suppression/beh/raw/subject-2.csv'
	data = pd.read_csv(file)
	data = data[['target_loc','dist_loc','block_type','repetition']]

	# create trigger list for comparison
	triggers = []
	for i in range(nr_practice,data.shape[0]):
		triggers.append(int('{}{}'.format(data['target_loc'][i] + 1,data['dist_loc'][i] + 1)))

	# check trigger list
	if np.sum(events[np.where(events[:,2] == 3)[0] + 1,2] == np.array(triggers)) != data.shape[0] - nr_practice:
		print('behavioral and bdf file do not match')

	#reps = 0
	#for i,event in enumerate(events[:,2]):
	#	if np.logical_and(event > 3, event < 80):
	#		reps += 1
	#		events[i,2] = int('{}{}'.format(reps,event))
	#	if reps == 4:
	#		reps = 0

	#return events	

def update_epochs_beh(events, epochs):
	'''

	'''

	nr_practice = 16
	file = '/Users/dirk/Desktop/suppression/beh/raw/subject-3.csv'
	data = pd.read_csv(file)
	data = data[['target_loc','dist_loc','block_type','repetition']]

	idx_info = np.c_[np.where(events[:,2] == 3)[0], np.array(range(nr_practice,data.shape[0]))]
	idx_sel  = np.array([list(idx_info[:,0]).index(i) for i in epochs.selection])
	idx_beh  = idx_info[idx_sel,1]

	beh_info = {}

	beh_info.update({'condition': np.array(data['block_type'][idx_beh]),
						'repetition': np.array(data['repetition'][idx_beh]),
						'target_loc': np.array(data['target_loc'][idx_beh]),
						'dist_loc': np.array(data['dist_loc'][idx_beh])})

	with open('/Users/dirk/Desktop/suppression/eeg/subject-2.pickle','wb') as handle:
		pickle.dump(beh_info, handle)


def mark_epoch(events, main_trigger, tmin, tmax):

	time_trigger = events[events[:,2] == 3,0]
	trigger_min = time_trigger + 512 * tmin
	trigger_max = time_trigger + 512 * tmax

	int_times = np.sort(np.hstack((trigger_min, trigger_max)))

	int_triggers = np.zeros((int_times.shape[0],3))
	
	for i in range(int_times.shape[0]):
		int_triggers[i,0] = int_times[i]
		if i % 2 == 0:
			int_triggers[i,2] = 10
		else:
			int_triggers[i,2] = 20	
		
	return int_triggers		