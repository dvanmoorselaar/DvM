"""
NonParametric statistical tests

Created by Dirk van Moorselaar on 27-02-2018.
Copyright (c) 2018 DvM. All rights reserved.
"""

import numpy as np
import scipy.sparse as sparse
import warnings

from typing import Tuple
from math import sqrt
from scipy.stats import ttest_rel, t

def bootstrap_SE(
	X: np.ndarray, 
	nr_iter: int = 9999
) -> Tuple[np.ndarray, np.ndarray]:
	"""Estimate standard error of the mean using bootstrap resampling.

	Generates a bootstrap distribution by resampling observations with
	replacement to estimate sampling variability of the mean.

	Parameters
	----------
	X : np.ndarray
		Data array of shape (n_observations, n_variables).
		Each row is one observation (e.g., subject), each column is
		one variable (e.g., time point, condition).
	nr_iter : int, default=9999
		Number of bootstrap iterations. More iterations give more
		stable estimates but take longer.

	Returns
	-------
	SE : np.ndarray
		Bootstrap standard error for each variable (n_variables,).
		Estimated as standard deviation across bootstrap samples.
	avg_X : np.ndarray
		Sample mean for each variable (n_variables,).

	Notes
	-----
	Bootstrap procedure:
	1. For each iteration, resample n observations with replacement
	2. Compute mean across resampled observations
	3. Repeat nr_iter times to build bootstrap distribution
	4. Standard error = SD of bootstrap distribution (with ddof=1)

	Assumes observations are independent. For within-subject designs,
	consider confidence_int() instead.

	Progress is printed to console during execution.

	Examples
	--------
	>>> # Estimate SE across 20 subjects, 100 time points
	>>> data = np.random.randn(20, 100)
	>>> se, mean = bootstrap_SE(data, nr_iter=9999)
	>>> # Plot with bootstrap error bars
	>>> plt.plot(mean)
	>>> plt.fill_between(range(100), mean-se, mean+se, alpha=0.3)

	See Also
	--------
	confidence_int : Within-subject confidence intervals
	scipy.stats.bootstrap : Modern scipy bootstrap function
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

def confidence_int(
	data: np.ndarray,
	p_value: float = 0.05,
	tail: str = 'two',
	morey: bool = True
) -> np.ndarray:
	"""Calculate within-subject confidence intervals using Cousineau 
	method.

	Computes confidence intervals appropriate for repeated-measures
	designs using Cousineau's (2005) normalization method with optional 
	Morey's (2008) correction.

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
	       Tutorials in Quantitative Methods for Psychology, 1(1), 
		   42-45.
	.. [2] Morey, R. D. (2008). Confidence intervals from normalized 
		   data: A correction to Cousineau (2005). Tutorials in 
		   Quantitative Methods for Psychology, 4(2), 61-64.

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
	
	# normalize the data by subtracting the participants mean performance
	# from each observation, and then add the grand mean to each observation
	ind_mean = np.nanmean(data, axis=1).reshape(data.shape[0], 1)
	grand_mean = np.nanmean(data, axis=1).mean()
	data = data - ind_mean + grand_mean
	# Look up t-value and caluclate CIs
	t_value = abs(t.ppf([p_value], data.shape[0] - 1)[0])
	CI = (np.nanstd(data, axis=0, ddof=1) / 
		  sqrt(data.shape[0]) * t_value)

	# correct CIs according to Morey (2008)
	if morey:
		CI = CI * (data.shape[1] / float((data.shape[1] - 1))) 

	return CI 

def paired_t(*args) -> np.ndarray:
	"""Paired t-test statistic function for cluster permutation tests.

	Wrapper around scipy.stats.ttest_rel that returns only the 
	t-statistic,designed for use with MNE permutation testing functions.

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

	"""
	return ttest_rel(*args)[0]

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
	8-connected: Each interior pixel has 8 neighbors (including 
	diagonals)

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



