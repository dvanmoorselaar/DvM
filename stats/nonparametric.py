"""
NonParametric statistical tests

Created by Dirk van Moorselaar on 27-02-2018.
Copyright (c) 2018 DvM. All rights reserved.
"""

import numpy as np

from typing import Optional, Generic, Union, Tuple, Any
from math import sqrt
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, ttest_1samp
from IPython import embed 

def cluster_test_mask(X:Union[np.array,list],p_value:float=0.05,
		      		test:str='within',**kwargs):

	if test == 'within':
		(t_obs,
		clusters,
		p_values,
		H0) = mne.stats.permutation_cluster_1samp_test(X, **kwargs)

	# create mask
	mask = np.zeros_like(t_obs, dtype=int)
	for c, p_val in enumerate(p_values):
		if p_val <= p_value:
			mask[clusters[c]] = 1
	
	return mask

def bootstrap_SE(X:np.array,nr_iter:int=9999):
	"""
	Uses bootstrapping to calculate the standard error of the mean 
	around timecourse X

	Args:
		X (np.array):timecourse data [nr_obs X nr_time]
		nr_iter (int, optional): Number of iterations for random 
		sampling (with replacment). Defaults to 1000.

	Returns:
		SE (np.array): standard error of the mean for each sample
		avg (np.array): mean timecourse
	"""

	nr_obs = X.shape[0]
	bootstr = np.zeros((nr_iter,X.shape[1]))

	print(f'bootstrapping using {nr_iter} iterations')
	for b in range(nr_iter):
		# sample nr observations from X (with replacement)
		idx = np.random.choice(nr_obs,size = nr_obs,replace = True) 				
		bootstr[b,:] = np.mean(X[idx,:],axis = 0)

	# calculate standard error of the mean
	SE = np.std(bootstr,ddof=1,axis = 0)
	avg_X = X.mean(axis = 0)

	return SE, avg_X

def threshArray(X, chance, method = 'ttest', paired = True, p_value = 0.05):	
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
	paired (bool): specifies whether ttest is a paired sampled test or not
	p_value (float) | p_value used for thresholding


	Returns
	- - - -
	X (array): thresholded data 

	'''

	X_ = np.copy(X) # make sure original data remains unchanged
	p_vals = signedRankArray(X_, chance, method)
	X_[:,p_vals > p_value] = chance
	p_vals = clusterBasedPermutation(X_,chance, paired = paired)
	X_ = X_.mean(axis = 0)
	X_[p_vals > p_value] = chance

	return X_

def signedRankArray(X, Y, method = 'ttest_rel'):
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
			elif method == 'ttest_rel':
				_, p_vals[i,j] = ttest_rel(X[:,i,j], Y[:,i,j]) 
			elif method == 	'ttest_1samp':
				_, p_vals[i,j] = ttest_1samp(X[:,i,j], Y[0,i,j]) 

	return p_vals		







	






