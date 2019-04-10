import numpy as np
import mne
import scipy.sparse as sparse
import warnings

from IPython import embed 
from math import sqrt
from sklearn.feature_extraction.image import grid_to_graph
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
from scipy.stats import t, ttest_rel

def trial_exclusion(beh, eeg, excl_factor):


	mask = [(beh[key] == f).values for  key in excl_factor.keys() for f in excl_factor[key]]
	for m in mask: 
		mask[0] = np.logical_or(mask[0],m)
	mask = mask[0]
	if mask.sum() > 0:
		beh.drop(np.where(mask)[0], inplace = True)
		beh.reset_index(inplace = True)
		eeg.drop(np.where(mask)[0])
		print 'Dropped {} trials after specifying excl_factor'.format(sum(mask))
		print 'NOTE DROPPING IS DONE IN PLACE. PLEASE REREAD DATA IF THAT CONDITION IS NECESSARY AGAIN'
	else:
		print 'Trial exclusion: no trials selected that matched specified criteria'

	return beh, eeg

# functions that support reading in data
def select_electrodes(ch_names, subset):
	'''

	'''

	if subset == 'all':
		elecs = []
	elif subset == 'post':
		elecs = ['Iz','Oz','O1','O2','PO7','PO8','PO3','PO4','POz','Pz','P9','P10','P7','P8','P5','P6','P3','P4','P1','P2','Pz']	

	picks = mne.pick_channels(ch_names, include = elecs)

	return picks	

def filter_eye(beh, eeg, eye_window, eye_ch = 'HEOG', eye_thresh = 1, eye_dict = None):
	'''

	'''

	nan_idx = np.where(np.isnan(beh['eye_bins']) > 0)[0]
	print ('Trials without reliable eyetracking data {} out of {} clean trials ({}%)'.format(nan_idx.size, beh['eye_bins'].size, nan_idx.size/float(beh['eye_bins'].size)*100))

	# limit step algorhytm to trials without eye tracking data
	if nan_idx.size > 0:
		s,e = [np.argmin(abs(eeg.times - t)) for t in eye_window]
		eog = eeg._data[nan_idx,eeg.ch_names.index(eye_ch),s:e]

		if eye_dict != None:

			idx_eye = eog_filt(eog, sfreq = eeg.info['sfreq'], windowsize = eye_dict['windowsize'], 
								windowstep = eye_dict['windowstep'], threshold = eye_dict['threshold'])
			beh['eye_bins'][nan_idx[idx_eye]] = 99	
	
	# remove trials from beh and eeg objects
	to_drop = np.where(beh['eye_bins'] > eye_thresh)[0]	
	print ('Dropped {} trials based on threshold criteria ({})%'.format(to_drop.size, to_drop.size/float(beh['eye_bins'].size)*100))
	beh.drop(to_drop, inplace = True)
	beh.reset_index(inplace = True)
	eeg.drop(to_drop, reason='eye detection')

	return beh, eeg


def eog_filt(eog, sfreq, windowsize = 50, windowstep = 25, threshold = 30):
	'''
	Split-half sliding window approach. This function slids a window in prespecified steps over 
	eog data. If the change in voltage from the first half to the second half of the window is greater 
	than a threshold, this trial is marked for rejection

	Arguments
	- - - - - 
	eog(array): epochs X times. Data used for trial rejection
	sfreq (float): digitizing frequency
	windowsize (int): size of the sliding window (in ms)
	windowstep (int): step size to slide window over the trial
	threshold (int): threshold in microvolt

	Returns
	- - - -

	eye_trials: array that specifies for each epoch whether eog_filt detected an eye movement

	'''	

	# shift miliseconds to samples
	windowstep /= 1000.0 / sfreq
	windowsize /= 1000.0 / sfreq
	s, e = 0, eog.shape[-1]

	# create multiple windows based on window parameters (accept that final samples of epoch may not be included) 
	window_idx = [(i, i + int(windowsize)) for i in range(s, e, int(windowstep)) if i + int(windowsize) < e]

	# loop over all epochs and store all eye events into a list
	eye_trials = []
	for i, x in enumerate(eog):
		
		for idx in window_idx:
			window = x[idx[0]:idx[1]]

			w1 = np.mean(window[:window.size/2])
			w2 = np.mean(window[window.size/2:])

			if abs(w1 - w2) > threshold:
				eye_trials.append(i)
				break

	eye_trials = np.array(eye_trials, dtype = int)			
	print ('selected {0} bad trials via eyethreshold ({1:.0f}%)'.format(eye_trials.size, eye_trials.size/float(eog.shape[0]) * 100))	

	return eye_trials	


def confidence_int(data, p_value = .05, tail='two', morey=True):
	'''

	Cousineau's method (2005) for calculating within-subject confidence intervals
	If needed, Morey's correction (2008) can be applied (recommended).

	Arguments
	----------
	data (array): Data for which CIs should be calculated
	p_value (float): p-value for determining t-value (the default is .05).
	tail (string): Two-tailed ('two') or one-tailed t-value.
	morey (bool), Apply Morey correction (the default is True)

	Returns
	-------
	CI (array): Confidence intervals for each condition
	'''

	if tail=='two':
		p_value = p_value/2
	elif tail not in ['two','one']:
		p_value = p_value/2        
		warnings.warn('Incorrect argument for tail: using default')
	
	# normalize the data by subtracting the participants mean performance from each observation, 
	# and then add the grand mean to each observation
	ind_mean = data.mean(axis=1).reshape(data.shape[0],1)
	grand_mean = data.mean(axis=1).mean()
	data = data - ind_mean + grand_mean
	# Look up t-value and caluclate CIs
	t_value = abs(t.ppf([p_value], data.shape[0]-1)[0])
	CI = data.std(axis=0, ddof=1)/sqrt(data.shape[0])*t_value

	# correct CIs according to Morey (2008)
	if morey:
		CI = CI*(data.shape[1]/float((data.shape[1] - 1))) 

	return CI 

def curvefitting(x, y, bounds, func = lambda x, a, d: d + (1 - d) * a**x):
	'''

	'''

	from scipy.optimize import curve_fit
	popt, pcov = curve_fit(func, x, y, bounds = bounds)

	return popt, pcov

def permTestMask1D(diff, p_value = 0.05):
	'''

	'''

	T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)	
	print cluster_pv
	mask = np.zeros(diff[0].shape[1],dtype = bool)
	sig_clusters = []
	for cl in np.array(clusters)[np.where(cluster_pv < p_value)[0]]:
		mask[cl[0]] = True	
		sig_clusters.append(cl)

	return mask, sig_clusters	

def permTestMask2D(diff, p_value = 0.05):
	'''

	'''

	a = np.arange(diff[0].shape[1] * diff[0].shape[2]).reshape((diff[0].shape[1], diff[0].shape[2]))
	adj = connected_adjacency(a, '8').toarray()
	b = grid_to_graph(diff[0].shape[1], diff[0].shape[2]).toarray()
	conn = np.array(np.array(np.add(adj,b), dtype = bool),dtype = int)
	conn = sparse.csr_matrix(conn)

	#T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t, connectivity = conn)	
	T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)	
	#T_obs, clusters, cluster_pv, HO = spatio_temporal_cluster_test(diff, stat_fun = paired_t, connectivity = conn)	
	#T_obs, clusters, cluster_pv, HO = spatio_temporal_cluster_test(diff, stat_fun = paired_t)
	T_obs_plot = np.nan * np.ones_like(T_obs)
	print cluster_pv
	for c, p_val in zip(clusters, cluster_pv):
		if p_val <= p_value:
			print c.sum()
			T_obs_plot[c] = T_obs[c]

	return T_obs_plot	

def paired_t(*args):
	'''
	Call scipy.stats.ttest_rel, but return only f-value
	'''
	
	return ttest_rel(*args)[0]

def bootstrap(X, b_iter = 1000):
	'''
	bootstrapSlopes uses a bootstrap procedure to calculate standard error of slope estimates.

	Arguments
	- - - - - 
	test

	Returns
	- - - -

	'''	

	nr_obs = X.shape[0]
	bootstrapped = np.zeros((b_iter,X.shape[1]))

	for b in range(b_iter):
		idx = np.random.choice(nr_obs,nr_obs,replace = True) 				# sample nr subjects observations from the slopes sample (with replacement)
		bootstrapped[b,:] = np.mean(X[idx,:],axis = 0)

	error = np.std(bootstrapped, axis = 0)
	mean = X.mean(axis = 0)

	return error, mean

def permTTest(real, perm, nr_perms = 1000, p_thresh = 0.01):
	'''
	permTTest calculates p-values for the one-sample t-stat for each sample point across frequencies using a surrogate distribution generated with 
	permuted data. The p-value is calculated by comparing the t distribution of the real and the permuted slope data across sample points. 
	The t-stats for both distribution is calculated with

	t = (m - 0)/SEm

	, where m is the sample mean slope and SEm is the standard error of the mean slope (i.e. stddev/sqrt(n)). The p value is then derived by dividing 
	the number of instances where the surrogate T value across permutations is larger then the real T value by the number of permutations.  

	Arguments
	- - - - - 
	real(array):  
	perm(array): 
	p_thresh (float): threshold for significance. All p values below this value are considered to be significant

	Returns
	- - - -
	p_val (array): array with p_values across frequencies and sample points
	sig (array): array with significance indices (i.e. 0 or 1) across frequencies and sample points
	'''

	# get number observations
	nr_obs = real.shape[0]
	nr_perms = perm.shape[-1]

	# preallocate arrays
	p_val = np.empty((real.shape[1],real.shape[2])) * np.nan
	sig = np.zeros((real.shape[1],real.shape[2])) 	# will be filled with 0s (non-significant) and 1s (significant)

	# calculate the real and the surrogate one-sample t-stats
	r_M = np.mean(real, axis = 0); p_M = np.mean(perm, axis = 0)
	r_SE = np.std(real, axis = 0)/sqrt(nr_obs); p_SE = np.std(perm, axis = 0)/sqrt(nr_obs)
	r_T = r_M/r_SE; p_T = p_M/p_SE

	# calculate p-values
	for f in range(real.shape[1]):
		for s in range(real.shape[2]):
			surr_T = p_T[f,s,:]
			p_val[f,s] = len(surr_T[surr_T>r_T[f,s]])/float(nr_perms)
			if p_val[f,s] <= p_thresh:
				sig[f,s] = 1

	return p_val, sig


def connected_adjacency(image, connect, patch_size=(1, 1)):
	"""
	Creates an adjacency matrix from an image where nodes are considered adjacent 
	based on 4-connected or 8-connected pixel neighborhoods.

	:param image: 2 or 3 dim array
	:param connect: string, either '4' or '8'
	:param patch_size: tuple (n,m) used if the image will be decomposed into 
	               contiguous, non-overlapping patches of size n x m. The 
	               adjacency matrix will be formed from the smaller sized array
	               e.g. original image size = 256 x 256, patch_size=(8, 8), 
	               then the image under consideration is of size 32 x 32 and 
	               the adjacency matrix will be of size 
	               32**2 x 32**2 = 1024 x 1024
	:return: adjacency matrix as a sparse matrix (type=scipy.sparse.csr.csr_matrix)
	"""	
	

	r, c = image.shape[:2]

	r = r / patch_size[0]
	c = c / patch_size[1]

	if connect == '4':
		# constructed from 2 diagonals above the main diagonal
		d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
		d2 = np.ones(c*(r-1))
		upper_diags = s.diags([d1, d2], [1, c])
		return upper_diags + upper_diags.T

	elif connect == '8':
		# constructed from 4 diagonals above the main diagonal
		d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
		d2 = np.append([0], d1[:c*(r-1)])
		d3 = np.ones(c*(r-1))
		d4 = d2[1:-1]
		upper_diags = sparse.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
		return upper_diags + upper_diags.T
	else:
		raise ValueError('Invalid parameter \'connect\'={connect}, must be "4" or "8".'
				.format(connect=repr(connect)))