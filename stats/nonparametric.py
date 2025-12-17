"""
NonParametric statistical tests

Created by Dirk van Moorselaar on 27-02-2018.
Copyright (c) 2018 DvM. All rights reserved.
"""

import numpy as np
import scipy.sparse as sparse
import warnings

from typing import Optional, Generic, Union, Tuple, Any, Callable
from math import sqrt
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, ttest_1samp, t
from sklearn.feature_extraction.image import grid_to_graph
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test 

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


def confidence_int(
	data: np.ndarray,
	p_value: float = 0.05,
	tail: str = 'two',
	morey: bool = True
) -> np.ndarray:
	"""Calculate within-subject confidence intervals using Cousineau method.

	Computes confidence intervals appropriate for repeated-measures designs
	using Cousineau's (2005) normalization method with optional Morey's
	(2008) correction.

	Parameters
	----------
	data : np.ndarray
		Data array of shape (n_subjects, n_conditions). Each row is one
		subject, each column is one condition/time point.
	p_value : float, default=0.05
		Alpha level for confidence interval (default 95% CI).
	tail : str, default='two'
		'two' for two-tailed or 'one' for one-tailed test.
	morey : bool, default=True
		Whether to apply Morey's correction factor. Recommended.

	Returns
	-------
	CI : np.ndarray
		Confidence interval widths for each condition. Add/subtract from
		condition means for error bars.

	Notes
	-----
	Cousineau's method normalizes within-subject variability by:
	1. Computing each subject's mean across conditions
	2. Subtracting subject mean from each observation
	3. Adding grand mean back to center the data

	Morey's correction adjusts for the number of conditions to provide
	unbiased CIs.

	References
	----------
	.. [1] Cousineau, D. (2005). Confidence intervals in within-subject
	       designs: A simpler solution to Loftus and Masson's method.
	       Tutorials in Quantitative Methods for Psychology, 1(1), 42-45.
	.. [2] Morey, R. D. (2008). Confidence intervals from normalized data:
	       A correction to Cousineau (2005). Tutorials in Quantitative
	       Methods for Psychology, 4(2), 61-64.

	Examples
	--------
	>>> # Data: 10 subjects x 4 conditions
	>>> data = np.random.randn(10, 4)
	>>> ci = confidence_int(data, p_value=0.05)
	>>> means = np.mean(data, axis=0)
	>>> # Plot with error bars
	>>> plt.errorbar(range(4), means, yerr=ci)
	"""

	if tail == 'two':
		p_value = p_value/2
	elif tail not in ['two','one']:
		p_value = p_value/2        
		warnings.warn('Incorrect argument for tail: using default')
	
	# normalize the data by subtracting the participants mean performance from each observation, 
	# and then add the grand mean to each observation
	ind_mean = np.nanmean(data, axis=1).reshape(data.shape[0],1)
	grand_mean = np.nanmean(data, axis=1).mean()
	data = data - ind_mean + grand_mean
	# Look up t-value and caluclate CIs
	t_value = abs(t.ppf([p_value], data.shape[0]-1)[0])
	CI = np.nanstd(data, axis=0, ddof=1)/sqrt(data.shape[0])*t_value

	# correct CIs according to Morey (2008)
	if morey:
		CI = CI*(data.shape[1]/float((data.shape[1] - 1))) 

	return CI 

def curvefitting(
	x: np.ndarray,
	y: np.ndarray,
	bounds: tuple,
	func: Callable = lambda x, a, d: d + (1 - d) * a**x
) -> Tuple[np.ndarray, np.ndarray]:
	"""Fit a nonlinear function to data using least squares.

	Wrapper around scipy.optimize.curve_fit for convenient curve fitting
	with bounded parameters.

	Parameters
	----------
	x : np.ndarray
		Independent variable data (1D array).
	y : np.ndarray
		Dependent variable data (1D array, same length as x).
	bounds : tuple
		2-tuple of array_like containing lower and upper bounds for
		parameters. Format: (lower_bounds, upper_bounds).
		Example: ([0, 0], [1, 1]) for two parameters between 0 and 1.
	func : Callable, optional
		Function to fit. Must accept x as first argument followed by
		parameters to fit. Default is exponential:
		f(x, a, d) = d + (1 - d) * a**x

	Returns
	-------
	popt : np.ndarray
		Optimal parameter values that minimize the residual.
	pcov : np.ndarray
		Covariance matrix of the parameter estimates. Diagonal contains
		parameter variance estimates.

	Notes
	-----
	The default function is an exponential decay/growth model often used
	in learning curves or psychometric functions.

	See scipy.optimize.curve_fit documentation for details on:
	- Convergence criteria
	- Error handling
	- Absolute sigma estimation

	Examples
	--------
	>>> # Fit default exponential to data
	>>> x = np.array([1, 2, 3, 4, 5])
	>>> y = np.array([0.8, 0.65, 0.55, 0.5, 0.48])
	>>> popt, pcov = curvefitting(x, y, bounds=([0, 0], [1, 1]))
	>>> print(f"a={popt[0]:.3f}, d={popt[1]:.3f}")

	>>> # Custom sigmoid function
	>>> sigmoid = lambda x, L, k, x0: L / (1 + np.exp(-k*(x-x0)))
	>>> popt, pcov = curvefitting(
	...     x, y,
	...     bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]),
	...     func=sigmoid
	... )

	See Also
	--------
	scipy.optimize.curve_fit : Underlying fitting function
	"""
	from scipy.optimize import curve_fit
	popt, pcov = curve_fit(func, x, y, bounds=bounds)

	return popt, pcov

def permTestMask1D(
	diff: np.ndarray,
	p_value: float = 0.05
) -> Tuple[np.ndarray, list]:
	"""Identify significant time clusters using cluster-based permutation.

	Performs 1D cluster-based permutation testing on difference data
	(e.g., condition A - condition B) to identify temporally contiguous
	significant effects while controlling for multiple comparisons.

	Parameters
	----------
	diff : np.ndarray
		Difference data with shape (n_subjects, n_timepoints).
		Each row is one subject's difference wave.
	p_value : float, default=0.05
		Significance threshold for cluster p-values.

	Returns
	-------
	mask : np.ndarray
		Boolean array (n_timepoints,) indicating significant time points.
		True = part of significant cluster.
	sig_clusters : list
		List of tuples, each containing indices of one significant
		temporal cluster.

	Notes
	-----
	Uses MNE's permutation_cluster_test with paired t-test statistic.
	Cluster p-values are printed to console during execution.

	The test:
	1. Computes paired t-test at each time point
	2. Forms clusters of adjacent significant points
	3. Sums t-values within each cluster
	4. Compares cluster sums to permutation distribution

	This controls family-wise error rate (FWER) for multiple comparisons
	across time.

	Examples
	--------
	>>> # Test difference between two conditions
	>>> diff = condition_a - condition_b  # shape: (n_subjects, n_times)
	>>> mask, clusters = permTestMask1D(diff, p_value=0.05)
	>>> print(f"Found {len(clusters)} significant temporal clusters")
	>>> # Use mask to highlight significant times in plot
	>>> plt.plot(times, diff.mean(axis=0))
	>>> plt.fill_between(times, ymin, ymax, where=mask, alpha=0.3)

	See Also
	--------
	permTestMask2D : 2D cluster test (time-frequency)
	mne.stats.permutation_cluster_test : Underlying MNE function
	paired_t : T-test statistic function
	"""
	T_obs, clusters, cluster_pv, HO = permutation_cluster_test(
		diff, stat_fun=paired_t
	)
	print(cluster_pv)
	mask = np.zeros(diff[0].shape[1], dtype=bool)
	sig_clusters = []
	for cl in np.array(clusters)[np.where(cluster_pv < p_value)[0]]:
		mask[cl[0]] = True
		sig_clusters.append(cl)

	return mask, sig_clusters

def permTestMask2D(
	diff: np.ndarray,
	p_value: float = 0.05
) -> np.ndarray:
	"""Identify significant time-frequency clusters using permutation test.

	Performs 2D cluster-based permutation testing on time-frequency
	difference data to identify spatiotemporally contiguous significant
	effects while controlling for multiple comparisons.

	Parameters
	----------
	diff : np.ndarray
		Difference data with shape (n_subjects, n_freqs, n_timepoints).
		Each element [i] is one subject's difference TFR.
	p_value : float, default=0.05
		Significance threshold for cluster p-values.

	Returns
	-------
	T_obs_plot : np.ndarray
		2D array (n_freqs, n_timepoints) containing t-values only for
		significant clusters. Non-significant points are NaN.
		Useful for plotting significant effects overlaid on TFR.

	Notes
	-----
	Uses MNE's permutation_cluster_test with paired t-test statistic.
	Cluster p-values and sizes are printed to console.

	Connectivity matrix construction (currently disabled in code):
	- Uses grid_to_graph for basic adjacency
	- Can use connected_adjacency for 4- or 8-connected neighborhoods
	- Currently runs without explicit connectivity (6-connected default)

	The test:
	1. Computes paired t-test at each time-frequency point
	2. Forms clusters of adjacent significant points  
	3. Sums t-values within each cluster
	4. Compares cluster sums to permutation distribution

	Examples
	--------
	>>> # Test TFR difference between conditions
	>>> # diff shape: (n_subjects, n_freqs, n_times)
	>>> T_sig = permTestMask2D(diff, p_value=0.01)
	>>> # Plot only significant clusters
	>>> plt.imshow(T_sig, aspect='auto', cmap='RdBu_r')
	>>> plt.xlabel('Time')
	>>> plt.ylabel('Frequency')

	See Also
	--------
	permTestMask1D : 1D cluster test (time only)
	connected_adjacency : Custom adjacency matrix creation
	mne.stats.permutation_cluster_test : Underlying MNE function
	"""
	# Construct adjacency matrix (code below is commented out in original)
	a = np.arange(diff[0].shape[1] * diff[0].shape[2]).reshape(
		(diff[0].shape[1], diff[0].shape[2])
	)
	adj = connected_adjacency(a, '8').toarray()
	b = grid_to_graph(diff[0].shape[1], diff[0].shape[2]).toarray()
	conn = np.array(np.add(adj, b), dtype=bool).astype(int)
	conn = sparse.csr_matrix(conn)

	# Run permutation test (connectivity currently not used)
	T_obs, clusters, cluster_pv, HO = permutation_cluster_test(
		diff, stat_fun=paired_t
	)
	
	T_obs_plot = np.nan * np.ones_like(T_obs)
	print(cluster_pv)
	for c, p_val in zip(clusters, cluster_pv):
		if p_val <= p_value:
			print(c.sum())
			T_obs_plot[c] = T_obs[c]

	return T_obs_plot

def paired_t(*args) -> np.ndarray:
	"""Paired t-test statistic function for cluster permutation tests.

	Wrapper around scipy.stats.ttest_rel that returns only the t-statistic,
	designed for use with MNE permutation testing functions.

	Parameters
	----------
	*args
		Variable arguments passed to scipy.stats.ttest_rel.
		Typically two arrays of paired observations.

	Returns
	-------
	t_stat : np.ndarray
		T-statistic values. P-values are discarded.

	Notes
	-----
	This is a convenience function used as stat_fun argument in:
	- mne.stats.permutation_cluster_test
	- mne.stats.spatio_temporal_cluster_test

	Examples
	--------
	>>> # Used internally by permutation tests
	>>> from mne.stats import permutation_cluster_test
	>>> T_obs, clusters, pv, H0 = permutation_cluster_test(
	...     [cond_a, cond_b],
	...     stat_fun=paired_t
	... )

	See Also
	--------
	scipy.stats.ttest_rel : Underlying t-test function
	permTestMask1D : Uses this for 1D cluster tests
	permTestMask2D : Uses this for 2D cluster tests
	"""
	return ttest_rel(*args)[0]

def bootstrap(
	X: np.ndarray,
	b_iter: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
	"""Estimate standard error of the mean using bootstrap resampling.

	Generates a bootstrap distribution by resampling observations with
	replacement to estimate sampling variability.

	Parameters
	----------
	X : np.ndarray
		Data array of shape (n_observations, n_variables).
		Each row is one observation (e.g., subject), each column is
		one variable (e.g., time point, condition).
	b_iter : int, default=1000
		Number of bootstrap iterations. More iterations give more
		stable estimates but take longer.

	Returns
	-------
	error : np.ndarray
		Bootstrap standard error for each variable (n_variables,).
		Estimated as standard deviation across bootstrap samples.
	mean : np.ndarray
		Sample mean for each variable (n_variables,).

	Notes
	-----
	Bootstrap procedure:
	1. For each iteration, resample n observations with replacement
	2. Compute mean across resampled observations
	3. Repeat b_iter times to build bootstrap distribution
	4. Standard error = SD of bootstrap distribution

	Assumes observations are independent. For within-subject designs,
	consider confidence_int() instead.

	Examples
	--------
	>>> # Estimate SE across 20 subjects, 100 time points
	>>> data = np.random.randn(20, 100)
	>>> se, m = bootstrap(data, b_iter=2000)
	>>> # Plot with bootstrap error bars
	>>> plt.plot(m)
	>>> plt.fill_between(range(100), m-se, m+se, alpha=0.3)

	>>> # Single variable bootstrap
	>>> slopes = np.random.randn(15, 1)  # 15 subjects
	>>> se, mean = bootstrap(slopes, b_iter=5000)
	>>> print(f"Mean: {mean[0]:.3f} ± {se[0]:.3f}")

	See Also
	--------
	confidence_int : Within-subject confidence intervals
	scipy.stats.bootstrap : Modern scipy bootstrap function
	"""
	nr_obs = X.shape[0]
	bootstrapped = np.zeros((b_iter, X.shape[1]))

	for b in range(b_iter):
		# Sample nr_obs observations with replacement
		idx = np.random.choice(nr_obs, nr_obs, replace=True)
		bootstrapped[b, :] = np.mean(X[idx, :], axis=0)

	error = np.std(bootstrapped, axis=0)
	mean = X.mean(axis=0)

	return error, mean

def permTTest(
	real: np.ndarray,
	perm: np.ndarray,
	nr_perms: int = 1000,
	p_thresh: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute p-values using surrogate permutation distribution.

	Calculates significance of one-sample t-statistics by comparing
	against a null distribution generated from permuted data. Useful for
	time-frequency analyses or other multi-dimensional comparisons.

	Parameters
	----------
	real : np.ndarray
		Real data with shape (n_observations, n_freqs, n_timepoints).
		E.g., actual slope or effect estimates per subject.
	perm : np.ndarray
		Permuted/surrogate data with shape 
		(n_observations, n_freqs, n_timepoints, n_permutations).
		Null distribution generated by shuffling/permuting labels.
	nr_perms : int, default=1000
		Number of permutations. Should match perm.shape[-1].
	p_thresh : float, default=0.01
		Significance threshold. Points with p ≤ p_thresh are marked
		significant.

	Returns
	-------
	p_val : np.ndarray
		P-values for each frequency-time point (n_freqs, n_timepoints).
		Proportion of permutations where surrogate T > real T.
	sig : np.ndarray
		Binary significance mask (n_freqs, n_timepoints).
		1 = significant, 0 = non-significant.

	Notes
	-----
	Procedure for each time-frequency point:
	1. Compute one-sample t-stat for real data: t = M / SE
	   where M = mean, SE = std / sqrt(n)
	2. Compute t-stats for all permutations similarly
	3. P-value = proportion of permutation t-stats exceeding real t
	4. Mark as significant if p ≤ p_thresh

	This is a one-tailed test (permutation t > real t).

	Does NOT control for multiple comparisons across time-frequency
	points. Consider cluster-based permutation (permTestMask2D) for
	FWER control.

	Examples
	--------
	>>> # Real slopes: 15 subjects x 20 freqs x 50 times
	>>> real_slopes = np.random.randn(15, 20, 50) + 0.1
	>>> # Permuted: add 1000 permutations as 4th dimension  
	>>> perm_slopes = np.random.randn(15, 20, 50, 1000)
	>>> p_vals, sig_mask = permTTest(
	...     real_slopes, perm_slopes,
	...     nr_perms=1000, p_thresh=0.01
	... )
	>>> # Plot significance mask
	>>> plt.imshow(sig_mask, aspect='auto', cmap='binary')

	See Also
	--------
	permTestMask2D : Cluster-based permutation for FWER control
	paired_t : Related t-test function
	"""
	# Get number of observations
	nr_obs = real.shape[0]
	nr_perms = perm.shape[-1]

	# Preallocate arrays
	p_val = np.empty((real.shape[1], real.shape[2])) * np.nan
	sig = np.zeros((real.shape[1], real.shape[2]))

	# Calculate real and surrogate one-sample t-stats
	r_M = np.mean(real, axis=0)
	p_M = np.mean(perm, axis=0)
	r_SE = np.std(real, axis=0) / sqrt(nr_obs)
	p_SE = np.std(perm, axis=0) / sqrt(nr_obs)
	r_T = r_M / r_SE
	p_T = p_M / p_SE

	# Calculate p-values for each frequency-time point
	for f in range(real.shape[1]):
		for s in range(real.shape[2]):
			surr_T = p_T[f, s, :]
			# One-tailed: proportion where surrogate > real
			p_val[f, s] = len(surr_T[surr_T > r_T[f, s]]) / float(nr_perms)
			if p_val[f, s] <= p_thresh:
				sig[f, s] = 1

	return p_val, sig

def connected_adjacency(
	image: np.ndarray,
	connect: str,
	patch_size: Tuple[int, int] = (1, 1)
) -> sparse.csr_matrix:
	"""Create adjacency matrix for cluster-based permutation tests.

	Generates a sparse adjacency matrix defining which pixels/points
	are considered neighbors in a 2D grid. Used for spatial or
	time-frequency cluster analysis.

	Parameters
	----------
	image : np.ndarray
		2D or 3D array defining the grid structure. Shape determines
		adjacency matrix size.
	connect : str
		Neighborhood connectivity type:
		- '4': 4-connected (up, down, left, right neighbors)
		- '8': 8-connected (includes diagonals)
	patch_size : tuple of int, default=(1, 1)
		Patch dimensions (n, m) for grouping pixels into larger units.
		Original image is divided into non-overlapping n×m patches,
		with adjacency defined between patches rather than pixels.
		Example: (8, 8) converts 256×256 image to 32×32 patch grid.

	Returns
	-------
	adjacency : scipy.sparse.csr_matrix
		Sparse symmetric adjacency matrix where element [i, j] = 1 if
		nodes i and j are adjacent, 0 otherwise.

	Raises
	------
	ValueError
		If connect is not '4' or '8'.

	Notes
	-----
	The adjacency matrix is symmetric (undirected graph) and constructed
	efficiently using diagonal matrices.

	4-connected: Each interior pixel has 4 neighbors
	8-connected: Each interior pixel has 8 neighbors (including diagonals)

	Examples
	--------
	>>> # Create 8-connected adjacency for 10x10 grid
	>>> grid = np.zeros((10, 10))
	>>> adj = connected_adjacency(grid, '8')
	>>> print(adj.shape)  # (100, 100)

	>>> # With patching: 64x64 image → 32x32 patches
	>>> tfr_data = np.zeros((64, 64))  # time-frequency data
	>>> adj = connected_adjacency(tfr_data, '8', patch_size=(2, 2))
	>>> print(adj.shape)  # (1024, 1024) = 32*32 x 32*32

	See Also
	--------
	permTestMask2D : Uses this for 2D cluster tests
	sklearn.feature_extraction.image.grid_to_graph : Similar function
	"""
	r, c = image.shape[:2]

	r = int(r / patch_size[0])
	c = int(c / patch_size[1])

	if connect == '4':
		# Constructed from 2 diagonals above the main diagonal
		d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
		d2 = np.ones(c*(r-1))
		upper_diags = sparse.diags([d1, d2], [1, c])
		return upper_diags + upper_diags.T

	elif connect == '8':
		# Constructed from 4 diagonals above the main diagonal
		d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
		d2 = np.append([0], d1[:c*(r-1)])
		d3 = np.ones(c*(r-1))
		d4 = d2[1:-1]
		upper_diags = sparse.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
		return upper_diags + upper_diags.T
	else:
		raise ValueError(
			f'Invalid parameter \'connect\'={connect!r}, must be "4" or "8".'
		)



