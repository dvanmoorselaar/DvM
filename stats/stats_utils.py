"""
NonParametric statistical tests

Created by Dirk van Moorselaar on 27-02-2018.
Copyright (c) 2018 DvM. All rights reserved.
"""

import numpy as np
import scipy.sparse as sparse
import warnings
import mne.stats

from typing import Tuple, Union, Optional
from math import sqrt
from scipy.stats import ttest_rel, t, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

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

def perform_stats(
		y: np.ndarray, 
		chance: float = 0, 
		stat_test: str = 'perm',
		p_thresh: float = 0.05,
		statfun: Optional[callable] = None,
		p_cluster: Optional[float] = None,
		threshold: Optional[float] = None
	) -> Tuple[np.ndarray, Union[list, np.ndarray], np.ndarray]:
	"""Perform statistical testing on group-level neural data.

	Conducts one-sample statistical tests comparing data against a 
	chance level. Supports multiple testing approaches 
	(permutation clustering, t-test, FDR correction) suitable for 
	time-series and 2D (frequency X time) neural data.

	Parameters
	----------
	y : np.ndarray
		Data array for statistical testing. Shape should be:
		- 2D array: (n_subjects, n_timepoints) for 1D data (timecourse)
		- 3D array: (n_subjects, n_frequencies, n_timepoints) for 
		  2D data (time-frequency)
	chance : float, default=0
		Chance level to test against. Data is centered on this value 
		before testing (y - chance). Typically 0 for deviation from 
		baseline, or 0.5 for classification accuracy.
	stat_test : {'perm', 'ttest', 'fdr'}, default='perm'
		Statistical test method:
		- 'perm': Permutation cluster test with sign-flipping to 
		  correct for multiple comparisons via spatiotemporal 
		  clustering. Tests if data distributions differ from chance 
		  level.
		- 'ttest': One-sample t-test (no multiple comparison correction)
		- 'fdr': T-test with False Discovery Rate correction
	p_thresh : float, default=0.05
		P-value threshold for filtering results. Only results with 
		p-value <= p_thresh are included in the output. Interpretation 
		varies by test type:
		- 'perm': Cluster-level p-value threshold. Only clusters with 
		  p <= p_thresh are returned (FWER control via clustering).
		- 'ttest': Individual timepoint p-value threshold. Only 
		  timepoints with p < p_thresh are marked as significant 
		  (no multiple comparison correction).
		- 'fdr': FDR-corrected p-value threshold. Only timepoints with 
		  corrected p < p_thresh are marked as significant.
	statfun : callable, optional
		Custom statistical function for permutation test. Only used when 
		stat_test='perm'. Function should accept data array and return 
		test statistic. If None (default), uses MNE's default 1-sample 
		t-test. For example: lambda x: np.mean(x, axis=0) for mean-based 
		test.
	p_cluster : float or None, default=None
		Cluster-forming p-value threshold (only for stat_test='perm'). 
		Use this for intuitive p-value-based specification. The 
		threshold test statistic is automatically calculated as:
		threshold = scipy.stats.t.ppf(1 - p_cluster/2, n_subjects - 1)
		For example, p_cluster=0.05 uses the t-value corresponding to 
		p=0.05 two-tailed. If both p_cluster and threshold are None 
		(default), MNE uses automatic threshold (default p=0.05). If 
		threshold is explicitly provided, p_cluster is ignored. Ignored 
		for stat_test='ttest' or 'fdr'.
	threshold : float or None, default=None
		Cluster-forming threshold as a test statistic value (only for 
		stat_test='perm'). Use this to directly specify the test 
		statistic threshold if you prefer not to use p-values. 
		For example:
		- For t-statistic: threshold=2.0 includes points with |t| > 2.0
		- For correlation: threshold=0.3 includes points with |r| > 0.3
		If specified, overrides p_cluster. If both are None (default), 
		MNE uses automatic threshold. 
		Ignored for stat_test='ttest' or 'fdr'.

	Returns
	-------
	test_stat : np.ndarray
		Test statistic values with same shape as input data.
	sig_mask : list or np.ndarray
		Significance indicators. Content depends on test type:
		- 'perm': List of significant clusters (filtered by p_thresh). 
		  Each cluster is a tuple of indices. Empty list if no 
		  significant clusters found.
		- 'ttest'/'fdr': Boolean array of same shape as input data, 
		  True for significant timepoints/pixels.
	p_vals : np.ndarray
		P-values for significant results only (filtered by p_thresh):
		- 'perm': One p-value per significant cluster
		- 'ttest'/'fdr': P-values for each significant timepoint/pixel

	Notes
	-----
	**Permutation test details**: Uses MNE's implementation which 
	performs a 1-sample cluster test via sign-flipping permutations. 
	For each permutation, data signs are randomly flipped and test 
	statistic recomputed. This naturally corrects for multiple 
	comparisons via spatiotemporal clustering, making it more 
	conservative than uncorrected tests.

	**FDR correction**: Applied across all timepoints (flattened for 2D 
	data) before reshaping back to original dimensions.

	**Important - Statistical control differences**: The three tests use 
	fundamentally different multiple comparison strategies:
	- 'perm': Controls family-wise error rate (FWER) via clustering - 
	  most conservative, best for strong effects
	- 'fdr': Controls false discovery rate (FDR) - moderate 
	  stringency, good balance
	- 'ttest': No correction - most liberal, many false positives
	
	Results are NOT directly comparable between methods with the same 
	p_thresh. Choose method based on your study design and effect size 
	expectations.

	Raises
	------
	ValueError
		If stat_test is not one of 'perm', 'ttest', or 'fdr'.
	TypeError
		If statfun is not callable.

	Examples
	--------
	>>> y = np.random.randn(30, 300)  # 30 subjects, 300 timepoints
	>>> t_vals, sig_mask, p_vals = perform_stats(y, chance=0, 
	...stat_test='ttest')
	>>> significant_points = np.sum(sig_mask)
	
	>>> # Using custom statfun with permutation test
	>>> custom_stat = lambda x: np.mean(x, axis=0)
	>>> t_vals, sig_mask, p_vals = perform_stats(y, stat_test='perm', 
	...                                            statfun=custom_stat)
	"""
	#TODO: add option to return a mask with significant points only

	# Determine input data dimensionality
	is_2d = y.ndim == 3

	if stat_test == 'perm':
		if statfun is not None and not callable(statfun):
			raise TypeError("statfun must be callable or None")
		
		# Build kwargs for MNE function - only include threshold if provided
		mne_kwargs = {'stat_fun': statfun}
		
		# Calculate threshold from p_cluster if provided and threshold is not
		if threshold is None and p_cluster is not None:
			# Convert p-value to t-statistic threshold
			n_subjects = y.shape[0]
			threshold = t.ppf(1 - p_cluster / 2, n_subjects - 1)
		
		if threshold is not None:
			mne_kwargs['threshold'] = threshold
		
		(t_obs, 
		clusters, 
		p_vals, 
		H0) = mne.stats.permutation_cluster_1samp_test(
			y - chance, **mne_kwargs
		)
		
		# Filter clusters by p_thresh to return only significant ones
		sig_indices = np.where(p_vals <= p_thresh)[0]
		significant_clusters = [clusters[i] for i in sig_indices]
		significant_p_vals = p_vals[sig_indices]
		
		return t_obs, significant_clusters, significant_p_vals
	elif stat_test == 'ttest':
		t_vals, p_vals = ttest_1samp(y, chance, axis=0)
		sig_mask = p_vals < p_thresh
		return t_vals, sig_mask, p_vals
	elif stat_test == 'fdr':
		t_vals, p_vals = ttest_1samp(y, chance, axis=0)
		if is_2d:
			# Apply FDR correction across all 2D points
			_, p_vals_fdr = fdrcorrection(p_vals.flatten())
			p_vals_fdr = p_vals_fdr.reshape(p_vals.shape)
		else:
			_, p_vals_fdr = fdrcorrection(p_vals)
		sig_mask = p_vals_fdr < p_thresh
		
		return t_vals, sig_mask, p_vals_fdr
	else:
		raise ValueError(
			f"stat_test must be 'perm', 'ttest', or 'fdr', got '{stat_test}'"
		)



