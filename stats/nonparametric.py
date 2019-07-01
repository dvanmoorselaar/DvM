"""
NonParametric statistical tests

Created by Dirk van Moorselaar on 27-02-2018.
Copyright (c) 2018 DvM. All rights reserved.
"""

import cv2

import numpy as np

from math import sqrt
from scipy.stats import ttest_rel, ttest_ind, wilcoxon
from IPython import embed 

def permutationTTest(X1, X2, nr_perm):
	'''

	'''

	# check whether X2 is a chance variable or a data array
	if isinstance(X2, (float, int)):
		X2 = np.tile(X2, X1.shape)
	X = X1 - X2

	# calculate T statistic
	nr_obs = X.shape[0]
	nr_test = X.shape[1:]
	T_0 = X.mean(axis = 0)/(X.std(axis = 0)/sqrt(nr_obs))

	# calculate surrogate T distribution
	surr = np.copy(X)
	T_p = np.stack([np.zeros(nr_test) for i in range(nr_perm)], axis = 0) 
	for p in range(nr_perm):
		perms = np.array(np.random.randint(2,size = X.shape), dtype = bool)
		surr[perms] *= -1
		T_p[p] = surr.mean(axis = 0)/(surr.std(axis = 0)/sqrt(nr_obs))

	# check how often surrogate T exceeds real T value
	thresh = np.sum(np.array((T_p > T_0),dtype = float), axis = 0)
	p_value = thresh/nr_perm
		
	return p_value, T_0


def clusterBasedPermutation(X1, X2, p_val = 0.05, cl_p_val = 0.05, paired = True, tail = 'both', nr_perm = 1000, mask = None, conn = None):

	'''
	Implements Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG- data. 
	Journal of Neurosience Methods, 164(1), 177?190. http://doi.org/10.1016/J.Jneumeth.2007.03.024

	Arguments
	- - - - - 

	X1 (array): subject X dim1 X dim2 (optional), where dim1 and dim2 can be any type of dimension 
				(time, frequency, electrode, etc). Values in array represent some dependent
				measure (e.g classification accuracy or power)
	X2 (array | float): either a datamatrix with same dimensions as X1, or a single value 
				against which X1 will be tested
	p_val (float): p_value used for inclusion into the cluster
	cl_p_val (float): p_value for evaluation overall cluster significance
	paired (bool): paired t testing (True) or independent t testing (False)
	tail (str): apply one- or two- tailed t testing
	nr_perm (int): number of permutations
	mask (array): dim1 X dim2 array. Can be used to restrict cluster based test to a specific region. 
	conn (array): outlines which dim1 points are connected to other dim1 points. Usefull
				  when doing a cluster based permutation test across electrodes 

	Returns
	- - - -

	cl_p_vals (array): dim1 X dim2 with p-values < cl_p_val for significant clusters and 1's for all other clusters

	'''

	# if no mask is provided include all datapoints in analysis
	if mask == None:
		mask = np.array(np.ones(X1.shape[1:]),dtype = bool)
		print('\nUsing all {} datapoints in cluster based permutation'.format(mask.size))
	elif mask.shape != X1[0].shape:
		print('\nMask does not have the same shape as X1. Adjust mask!')
	else:
		print('\nThere are {} out of {} datapoints in your mask during cluster based permutation'.format(int(mask.sum()), mask.size))	

	# check whether X2 is a chance variable or a data array
	if isinstance(X2, (float, int)):
		X2 = np.tile(X2, X1.shape)

	# compute observed cluster statistics
	pos_sizes, neg_sizes, pos_labels, neg_labels, sig_cl = computeClusterSizes(X1, X2, p_val, paired, tail, mask, conn)	
	cl_p_vals = np.ones(sig_cl.shape)

	# iterate to determine how often permuted clusters exceed the observed cluster threshold
	c_pos_cl = np.zeros(np.max(np.unique(pos_labels)))
	c_neg_cl = np.zeros(np.max(np.unique(neg_labels)))

	# initiate random arrays
	X1_rand = np.zeros(X1.shape)
	X2_rand = np.zeros(X1.shape)

	for p in range(nr_perm):

		print "\r{0}% of permutations".format((float(p)/nr_perm)*100),

		# create random partitions
		if paired: # keep observations paired under permutation
			rand_idx = np.random.rand(X1.shape[0])<0.5
			X1_rand[rand_idx,:] = X1[rand_idx,:] 
			X1_rand[~rand_idx,:] = X2[~rand_idx,:] 
			X2_rand[rand_idx,:] = X2[rand_idx,:] 
			X2_rand[~rand_idx,:] = X1[~rand_idx,:]
		else: # fully randomize observations under permutation
			all_X = np.vstack((X1,X2))	
			all_X = all_X[np.random.permutation(all_X.shape[0]),:]
			X1_rand = all_X[:X1.shape[0],:]
			X2_rand = all_X[X1.shape[0]:,:]

		# compute cluster statistics under random permutation
		rand_pos_sizes, rand_neg_sizes, _, _, _ = computeClusterSizes(X1_rand, X2_rand, p_val, paired, tail, mask, conn)
		max_rand = np.max(np.hstack((rand_pos_sizes, rand_neg_sizes)))

		# count cluster p values
		c_pos_cl += max_rand > pos_sizes
		c_neg_cl += max_rand > neg_sizes
			
	# compute cluster p values
	p_pos = c_pos_cl / nr_perm
	p_neg = c_neg_cl / nr_perm

	# remove clusters that do not pass threshold
	if tail == 'both':
		for i, cl in enumerate(np.unique(pos_labels)[1:]): # 0 is not a cluster
			if p_pos[i] < cl_p_val/2:
				cl_p_vals[pos_labels == cl] = p_pos[i]
			else:
				pos_labels[pos_labels == cl] = 0

		for i, cl in enumerate(np.unique(neg_labels)[1:]): # 0 is not a cluster
			if p_neg[i] < cl_p_val/2:
				cl_p_vals[neg_labels == cl] = p_neg[i]
			else:
				neg_labels[neg_labels == cl] = 0

	elif tail == 'right':
		for i, cl in enumerate(np.unique(pos_labels)[1:]): # 0 is not a cluster
			if p_pos[i] < cl_p_val:
				cl_p_vals[pos_labels == cl] = p_pos[i]
			else:
				pos_labels[pos_labels == cl] = 0

	elif tail == 'left':
		for i, cl in enumerate(np.unique(neg_labels)[1:]): # 0 is not a cluster
			if p_neg[i] < cl_p_val:
				cl_p_vals[neg_labels == cl] = p_neg[i]
			else:
				neg_labels[neg_labels == cl] = 0

	# ADD FUNCTION TO GET 			

	return cl_p_vals			
					

def computeClusterSizes(X1, X2, p_val, paired, tail, mask, conn):
	'''

	Helper function for clusterBasedPermutation (see documentation)
	
	NOTE!!!
	Add the moment only supports two tailed tests
	Add the moment does not support connectivity
	'''

	# STEP 1: determine 'actual' p value
	# apply the mask to restrict the data
	X1_mask = X1[:,mask]
	X2_mask = X2[:,mask]

	p_vals = np.ones(mask.shape)
	t_vals = np.zeros(mask.shape)

	if paired:
		t_vals[mask], p_vals[mask] = ttest_rel(X1_mask, X2_mask)
	else:
		t_vals[mask], p_vals[mask] = ttest_ind(X1_mask, X2_mask)		

	# initialize clusters and use mask to restrict relevant info
	sign_cl = np.mean(X1,0) - np.mean(X2,0)	
	sign_cl[~mask] = 0
	p_vals[~mask] = 1

	# STEP 2: apply threshold and determine positive and negative clusters
	cl_mask = p_vals < p_val
	pos_cl = np.zeros(cl_mask.shape)
	neg_cl = np.zeros(cl_mask.shape)
	pos_cl[sign_cl > 0] = cl_mask[sign_cl > 0]
	neg_cl[sign_cl < 0] = cl_mask[sign_cl < 0]

	# STEP 3: label clusters
	if conn == None:
		nr_p, pos_labels = cv2.connectedComponents(np.uint8(pos_cl))
		nr_n, neg_labels = cv2.connectedComponents(np.uint8(neg_cl))
		pos_labels = np.squeeze(pos_labels) # hack to control for onedimensional data (CHECK whether correct)
		neg_labels = np.squeeze(neg_labels)
	else:
		print('Function does not yet support connectivity')	

	# STEP 4: compute the sum of t stats in each cluster (pos and neg)
	pos_sizes, neg_sizes = np.zeros(nr_p - 1), np.zeros(nr_n - 1)
	for i, label in enumerate(np.unique(pos_labels)[1:]):
		pos_sizes[i] = np.sum(t_vals[pos_labels == label])

	for i, label in enumerate(np.unique(neg_labels)[1:]):
		neg_sizes[i] = abs(np.sum(t_vals[neg_labels == label]))

	if sum(pos_sizes) == 0:
		pos_sizes = 0

	if sum(neg_sizes) == 0:
		neg_sizes = 0

	return pos_sizes, neg_sizes, pos_labels, neg_labels, p_vals	


def permTTest(X_real, X_perm, p_thresh = 0.05):
	'''
	permTTest calculates p-values for the one-sample t-stat for each sample point across frequencies 
	using a surrogate distribution generated with permuted data. The p-value is calculated by comparing 
	the t distribution of the real and the permuted slope data across sample points. 
	The t-stats for both distribution is calculated with

	t = (m - 0)/SEm

	, where m is the sample mean slope and SEm is the standard error of the mean slope (i.e. stddev/sqrt(n)). 
	The p value is then derived by dividing the number of instances where the surrogate T value across permutations 
	is larger then the real T value by the number of permutations.  

	Arguments
	- - - - - 
	X_real(array): subject X dim1 X dim2 (optional), where dim1 and dim2 can be any type of dimension 
				(time, frequency, electrode, etc). Values in array represent some dependent measure 
				(e.g classification accuracy or power)
	X_perm(array): subject X nr_permutation X dim1 X dim2 (optional)
	p_thresh (float): threshold for significance. All p values below this value are considered to be significant

	Returns
	- - - -
	p_val (array): array with p_values across frequencies and sample points
	sig (array): array with significance indices (i.e. 0 or 1) across frequencies and sample points
	'''

	# FUNCTION DOES NOT YET SUPPORT ONE DIMENSIONAL DATA

	# preallocate arrays
	nr_perm = X_perm.shape [1]
	nr_obs = X_real.shape[0]
	p_val = np.zeros(X_real.shape[1:])
	sig = np.zeros(X_real.shape[1:])		# will be filled with 0s (non-significant) and 1s (significant)

	# calculate the real and the surrogate one-sample t-stats
	r_M = np.mean(X_real, axis = 0); p_M = np.mean(X_perm, axis = 0)
	r_SE = np.std(X_real, axis = 0)/sqrt(nr_obs); p_SE = np.std(X_perm, axis = 0)/sqrt(nr_obs)
	r_T = r_M/r_SE; p_T = p_M/p_SE

	# calculate p-values
	for f in range(X_real.shape[1]):
		for s in range(X_real.shape[2]):
			surr_T = p_T[f,s,:]
			p_val[f,s] = len(surr_T[surr_T>r_T[f,s]])/float(nr_perm)
			if p_val[f,s] < p_thresh:
				sig[f,s] = 1

	return p_val, sig

def FDR(p_vals, q = 0.05, method = 'pdep', adjust_p = False, report = True):
	'''
	Functions controls the false discovery rate of a family of hypothesis tests. FDR is
	the expected proportion of rejected hypotheses that are mistakingly rejected 
	(i.e., the null hypothesis is actually true for those tests). FDR is less 
	conservative/more powerfull method for correcting for multiple comparisons than 
	procedures like Bonferroni correction that provide strong control of the familiy-wise
	error rate (i.e. the probability that one or more null hypotheses are mistakingly rejected)

	Arguments
	- - - - - 

	p_vals (array): an array (one or multi-demensional) containing the p_values of each individual
					test in a family f tests
	q (float): the desired false discovery rate
	method (str): If 'pdep' the original Bejnamini & Hochberg (1995) FDR procedure is used, which 
				is guaranteed to be accurate if the individual tests are independent or positively 
				dependent (e.g., Gaussian variables that are positively correlated or independent).  
				If 'dep,' the FDR procedure described in Benjamini & Yekutieli (2001) that is guaranteed 
				to be accurate for any test dependency structure (e.g.,Gaussian variables with any 
				covariance matrix) is used. 'dep' is always appropriate to use but is less powerful than 'pdep.'
	adjust_p (bool): If True, adjusted p-values are computed (can be computationally intensive)	
	report (bool): If True, a brief summary of FDR results is printed 		

	Returns
	- - - -

	h (array): a boolean matrix of the same size as the input p_vals, specifying whether  
			   the test that produced the corresponding p-value is significant
	crit_p (float): All uncorrected p-values less than or equal to crit_p are significant.
					If no p-values are significant, crit_p = 0
	adj_ci_cvrg (float): he FCR-adjusted BH- or BY-selected confidence interval coverage.	
	adj_p (array): All adjusted p-values less than or equal to q are significant. Note, 
				   adjusted p-values can be greater than 1					   
	'''

	orig = p_vals.shape

	# check whether p_vals contains valid input (i.e. between 0 and 1)
	if np.sum(p_vals > 1) or np.sum(p_vals < 0):
		print ('Input contains invalid p values')

	# sort p_values	
	if p_vals.ndim > 1:
		p_vect = np.squeeze(np.reshape(p_vals,(1,-1)))
	else:
		p_vect = p_vals	
	
	sort = np.argsort(p_vect) # for sorting
	rev_sort = np.argsort(sort) # to reverse sorting
	p_sorted = p_vect[sort]

	nr_tests = p_sorted.size
	tests = np.arange(1.0,nr_tests + 1)

	if method == 'pdep': # BH procedure for independence or positive independence
		if report:
			print('FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests')
		thresh = tests * (q/nr_tests)
		wtd_p = nr_tests * p_sorted / tests 
	elif method == 'dep': # BH procedure for any dependency structure
		if report:
			print('FDR/FCR procedure used is guaranteed valid for independent or dependent tests')
		denom = nr_tests * sum(1/tests)
		thresh = tests * (q/denom)
		wtd_p = denom * p_sorted / tests
		# Note this method can produce adjusted p values > 1 (Compute adjusted p values)

	# Chec whether p values need to be adjusted	
	if adjust_p:
		adj_p = np.empty(nr_tests) * np.nan	
		wtd_p_sortidx = np.argsort(wtd_p)
		wtd_p_sorted = wtd_p[wtd_p_sortidx]
		next_fill = 0
		for i in range(nr_tests):
			if wtd_p_sortidx[i] >= next_fill:
				adj_p[next_fill:wtd_p_sortidx[i]+1] = wtd_p_sorted[i]
				next_fill = wtd_p_sortidx[i] + 1
				if next_fill > nr_tests:
					break	
		adj_p = np.reshape(adj_p[rev_sort], (orig))	
	else:
		adj_p = np.nan	

	rej = np.where(p_sorted <= thresh)[0]
		
	if rej.size == 0:
		crit_p = 0
		h = np.array(p_vals * 0, dtype = bool)
		adj_ci_cvrg = np.nan
	else:
		max_idx = rej[-1] # find greatest significant pvalue
		crit_p = p_sorted[max_idx]
		h = p_vals <= crit_p
		adj_ci_cvrg = 1 - thresh[max_idx]

	if report:
		nr_sig = np.sum(p_sorted <= crit_p)
		if nr_sig == 1:
			print('Out of {} tests, {} is significant using a false discovery rate of {}\n'.format(nr_tests,nr_sig,q))
		else:
			print('Out of {} tests, {} are significant using a false discovery rate of {}\n'.format(nr_tests,nr_sig,q))	

	return h, crit_p, adj_ci_cvrg, adj_p	

def threshArray(X, chance, method = 'ttest', p_value = 0.05):	
	'''
	Two step thresholding of a two dimensional data array.
	Step 1: use group level testing for each individual data point
	Step 2: apply clusterbased permutation on the thresholded data from step 1

	Arguments
	- - - - - 

	X (array): subject X dim1 X dim2, where dim1 and dim2 can be any type of dimension 
				(time, frequency, electrode, etc). Values in array represent some dependent
				measure (e.g classification accuracy or power)
	chance (int | float): chance value. All non-significant values will be reset to this value
	method (str): statistical test used in first step of thresholding
	p_value (float) | p_value used for thresholding


	Returns
	- - - -
	X (array): thresholded data 

	'''

	X_ = np.copy(X) # make sure original data remains unchanged
	p_vals = signedRankArray(X_, chance, method)
	X_[:,p_vals > p_value] = chance
	p_vals = clusterBasedPermutation(X_,chance)
	X_ = X_.mean(axis = 0)
	X_[p_vals > p_value] = chance

	return X_

def signedRankArray(X, Y, method = 'ttest'):
	'''

	Arguments
	- - - - - 

	X1 (array): subject X dim1 X dim2, where dim1 and dim2 can be any type of dimension 
				(time, frequency, electrode, etc). Values in array represent some dependent
				measure (e.g classification accuracy or power)
	Y (array | float): either a datamatrix with same dimensions as X1, or a single value 
				against which X1 will be tested
	method (str): type of test to calculate p values
	'''

	# check whether X2 is a chance variable or a data array
	if isinstance(Y, (float, int)):
		Y = np.tile(Y, X.shape)

	p_vals = np.ones(X[0].shape)

	for i in range(p_vals.shape[0]):
		for j in range(p_vals.shape[1]):
			if method == 'wilcoxon':
				_, p_vals[i,j] = wilcoxon(X[:,i,j], Y[:,i,j]) 
			elif method == 'ttest':
				_, p_vals[i,j] = ttest_rel(X[:,i,j], Y[:,i,j]) 

	return p_vals		


def bootstrap(X, b_iter = 1000):
	'''
	bootstrap uses a bootstrap procedure to calculate standard error of data in X.

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


def jacklatency(x1, x2, thresh_1, thresh_2, times):
	'''
	Helper function of jackknife. Calculates the latency difference between
	threshold crosses using linear interpolation

	Arguments
	- - - - - 
	x1 (array): subject X time. Values in array represent some dependent
				measure. (e.g. ERP voltages)
	x2 (array): array with same dimensions as X1
	thresh_1 (float): criterion value
	thresh_2 (float): criterion value
	times (array): timing of samples in X1 and X2
	times (str): calculate onset or offset latency differences

	Returns
	- - - -

	D (float): latency difference
	'''

	# get latency exceeding thresh 
	idx_1 = np.where(x1 >= thresh_1)[0][0]
	lat_1 = times[idx_1 - 1] + (times[idx_1] - times[idx_1 - 1]) * \
				(thresh_1 - x1[idx_1 - 1])/(x1[idx_1] - x1[idx_1-1])
	idx_2 = np.where(x2 >= thresh_2)[0][0]
	lat_2 = times[idx_2 - 1] + (times[idx_2] - times[idx_2 - 1]) * \
			(thresh_2 - x2[idx_2 - 1])/(x2[idx_2] - x2[idx_2-1])

	D = lat_2 - lat_1	

	return D	


def jackknife(X1, X2, times, peak_window, percent_amp = 50, timing = 'onset'):
	'''
	Implements Miller, J., Patterson, T., & Ulrich, R. (1998). Jackknife-based method for measuring 
	LRP onset latency differences. Psychophysiology, 35(1), 99-115. 

	Compares onset latencies between two grand-average waveforms. For each waveform a criterion 
	is determined based on a set percentage of the grand average peak. The latency at which this 
	criterion is first reached is then determined using linear interpolation. Next the jackknife 
	estimate of the standard error of the difference is used, which is then used to calculate the
	t value corresponding to the null hypothesis of no differences in onset latencies 

	Arguments
	- - - - - 
	X1 (array): subject X time. Values in array represent some dependent
				measure. (e.g. ERP voltages)
	X2 (array): array with same dimensions as X1
	times (array): timing of samples in X1 and X2
	peak_window (tuple | list): time window that contains peak of interest
	percent_amp (int): used to calculate criterion value
	timing (str): calculate onset or offset latency differnces

	Returns
	- - - -

	onset (float): onset differnce between grand waveform of X1 and X2
	t_value (float): corresponding to the null hypothesis of no differences in onset latencies
	'''	

	# set number of observations 
	nr_sj = X1.shape[0]

	# flip arrays if necessary
	if timing == 'offset':
		X1 = np.fliplr(X1)
		X2 = np.fliplr(X2)
		times = np.flipud(times)
	
	# get time window of interest
	s,e = np.sort([np.argmin(abs(times - t)) for t in peak_window])
	t = times[s:e]

	# slice data containing the peak average 
	x1 = np.mean(abs(X1[:,s:e]), axis = 0)
	x2 = np.mean(abs(X2[:,s:e]), axis = 0)
	
	# get the criterion based on peak amplitude percentage
	c_1 = max(x1) * percent_amp/ 100.0 
	c_2 = max(x2) * percent_amp/ 100.0 

	onset = jacklatency(x1, x2, c_1, c_2, t) 

	# repeat previous steps but exclude all data points once
	D = []
	idx = np.arange(nr_sj)
	for i in range(nr_sj):
		x1 = np.mean(abs(X1[np.where(idx != i)[0],s:e]), axis = 0)
		x2 = np.mean(abs(X2[:,s:e]), axis = 0)

		c_1 = max(x1) * percent_amp/ 100.0 
		c_2 = max(x2) * percent_amp/ 100.0 

		D.append(jacklatency(x1, x2, c_1, c_2, t) )

	# compute the jackknife estimate of the standard error of the differnce
	Sd = np.sqrt((nr_sj - 1.0)/ nr_sj * np.sum([(d - np.mean(D))**2 for d in np.array(D)]))	

	t_value = onset/ Sd 

	return onset, t_value

	






