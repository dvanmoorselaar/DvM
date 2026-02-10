import mne
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from analysis.ERP import *
from stats.stats_utils import bootstrap_SE, perform_stats
from typing import Optional, Generic, Union, Tuple, Any, List, Dict
from support.preprocessing_utils import get_time_slice, get_diff_pairs
from visualization.visuals import MidpointNormalize

# set general plotting parameters
# inspired by http://nipunbatra.github.io/2014/08/latexify/
params = {
    'axes.labelsize': 10, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'arial',
}
matplotlib.rcParams.update(params)

meanlineprops = dict(linestyle='--', linewidth=1, color='black')
medianlineprops = dict(linestyle='-', linewidth=1, color='black')

def plot_timecourse(x: np.ndarray, y: np.ndarray,
                    show_SE: bool = False, smooth: bool = False,
                    **kwargs):
	"""
	Plot a timecourse line with optional error bands and smoothing.

	Plots neural timecourse data on the current matplotlib axis. If data 
	contains multiple subjects, can either average across subjects or 
	compute bootstrap standard errors for visualization of variability.

	Parameters
	----------
	x : np.ndarray
		X-axis values (typically time points in seconds). Should have 
		shape (n_timepoints,).
	y : np.ndarray
		Y-axis values. Can be 1D with shape (n_timepoints,) for 
		pre-averaged data, or 2D with shape (n_subjects, n_timepoints) 
		for subject-level data. If 2D, averaging or error computation 
		is applied based on show_SE parameter.
	show_SE : bool, default=False
		If True and y is 2D, compute bootstrap standard errors and shade 
		the area around the mean line. If False, simply average across 
		subjects. Ignored if y is 1D. Default is False.
	smooth : bool, default=False
		If True, apply Savitzky-Golay smoothing (window=9, order=1) to 
		the data before plotting. Provides mild smoothing suitable for 
		neural timecourse data. Default is False.
	**kwargs
		Additional keyword arguments passed to plt.plot() for styling 
		the line, such as label, color, linewidth, linestyle, etc.

	Returns
	-------
	None
		Modifies the current matplotlib figure by adding the plotted 
		timecourse line and optional error band.

	Notes
	-----
	When show_SE=True, the error band uses transparency (alpha=0.2) for 
	clear visualization. The Savitzky-Golay filter uses a window length 
	of 9 and polynomial order of 1, which provides mild smoothing 
	without over-filtering neural data.

	Examples
	--------
	>>> import numpy as np
	>>> x = np.linspace(-0.5, 1, 300)
	>>> y = np.random.randn(20, 300)  # 20 subjects, 300 timepoints
	>>> plt.figure()
	>>> plot_timecourse(x, y, show_SE=True, label='Mean ± SE', 
	...                 color='blue')
	>>> plt.xlabel('Time (s)')
	>>> plt.ylabel('Amplitude (µV)')
	"""

	if y.ndim > 1:
		if show_SE:
			err, y = bootstrap_SE(y)
		else:
			y = y.mean(axis=0)

	if smooth:
		y = savgol_filter(y, 9, 1)

	plt.plot(x, y, **kwargs)

	if show_SE:
		kwargs.pop('label', None)
		plt.fill_between(x, y + err, y - err, alpha=0.2, **kwargs)

def plot_2d(Z:np.array,x_val:np.array=None,
	    	y_val:np.array=None,colorbar:bool=True,nr_ticks_x:np.array=None,
			nr_ticks_y:np.array=5, set_y_ticks:bool=True,
			interpolation:str='bilinear',cbar_label:str=None,
			mask:Union[np.ndarray,list]=None,mask_value:float=np.nan,
			p_vals:np.ndarray=None,p_thresh:float=0.05,
			center_zero:bool=False,cmap:str=None,
			**kwargs):
	"""Plot 2D heatmap with optional masking of non-significant values.

	Creates a 2D heatmap visualization of neural data 
	(e.g., time-frequency representations) with flexible masking 
	capabilities. Supports both cluster-based and pointwise statistical 
	masks, automatic colorbar adjustment, and diverging colormaps 
	centered at zero.

	Parameters
	----------
	Z : np.ndarray
		2D or 3D data array to plot. If 3D with shape 
		(n_subjects, n_y, n_x),averaged over the first dimension. 
		For final visualization, shape shouldbe (n_y, n_x) representing 
		dimensions like (frequencies, timepoints) or 
		(train_time, test_time).
	x_val : np.ndarray or None, default=None
		X-axis coordinate values. If None, uses integer indices 
		[0, Z.shape[-1]]. Typically represents time values in seconds or 
		milliseconds.
	y_val : np.ndarray or None, default=None
		Y-axis coordinate values. If None, uses integer indices 
		[0, Z.shape[-2]].Typically represents frequency or time values.
	colorbar : bool, default=True
		Whether to display a colorbar indicating data value range.
	nr_ticks_x : int or None, default=None
		Number of x-axis ticks. If None, uses matplotlib defaults.
	nr_ticks_y : int or None, default=5
		Number of y-axis ticks. If None, defaults to 5 or total length 
		if smaller.
	set_y_ticks : bool, default=True
		Whether to manually set y-axis ticks. If True, uses nr_ticks_y 
		to determine tick spacing. Automatically detects linear vs. 
		log scaling for y_val.
	interpolation : str, default='bilinear'
		Interpolation method for imshow. Common options: 'nearest', 
		'bilinear', 'bicubic'. Automatically set to 'nearest' if using 
		masked arrays with  center_zero=True to prevent artifacts.
	cbar_label : str or None, default=None
		Label for the colorbar. If None, no label is displayed.
	mask : np.ndarray, list, or None, default=None
		Optional significance mask for displaying only significant data. 
		Format:
		- Boolean array (n_y, n_x): Shows only True values, masks rest
		- List of cluster tuples/arrays: Marks which elements are 
		  significant(converted to boolean mask internally using p_vals 
		  and p_thresh)
		If None, full data array is displayed without masking.
	mask_value : float, default=np.nan
		Value to assign to masked (non-significant) regions. Default 
		of np.nan creates visual gaps (white/transparent regions) for clean 
		visualization of only significant data. Use mask_value=0 to show 
		non-significant regions as zero instead of hiding them.
	p_vals : np.ndarray or None, default=None
		P-values for clusters (only used when mask is a list). Required 
		to filter clusters by p_thresh when converting cluster list to 
		boolean mask.
	p_thresh : float, default=0.05
		P-value threshold for significance. Only used with cluster list 
		mask:clusters with p_vals[i] <= p_thresh are included in the 
		final mask.
	center_zero : bool, default=False
		If True, use diverging colormap centered at zero (white). Useful 
		for data with positive and negative values (e.g., t-statistics, 
		effect sizes). Automatically computes symmetric vmin/vmax around 
		zero and sets interpolation to 'nearest' for masked arrays to 
		prevent artifacts.
	cmap : str or None, default=None
		Colormap name (e.g., 'RdBu_r', 'viridis'). If None and 
		center_zero=True, defaults to 'RdBu_r' 
		(red=positive, blue=negative, reversed).
	**kwargs
		Additional keyword arguments passed to plt.imshow, such as vmin, 
		vmax, or other matplotlib image display options.

	Returns
	-------
	None
		Modifies the current matplotlib figure by displaying the 2D 
		heatmap.

	Notes
	-----
	**Data preparation**: Input 3D arrays are averaged over the first 
	dimension(typically subjects) before visualization. Pre-average data 
	if custom aggregation is needed.

	**Masking behavior**: When using a mask with show_only_significant=True, 
	non-significant regions are set to mask_value (default np.nan, which 
	creates visual gaps). This isolates significant regions for focused 
	visualization. Non-masked plots show full data with significance 
	contours overlaid.

	**Cluster-based masking**: When mask is a list of clusters 
	(from permutation tests via plot_significance), p_vals and p_thresh 
	determine which clusters are included in the final boolean mask. 
	Clusters are filtered: only those with p <= p_thresh are marked as 
	True in the boolean mask.

	**Tick handling**: When set_y_ticks=True, the function 
	intelligently:
	- Detects linear vs. logarithmic spacing in y_val
	- Selects evenly-spaced indices for tick placement
	- Rounds tick labels if y_val is floating-point

	**Colorbar limits with masking**: When using center_zero with masked 
	data, colorbar limits are computed from unmasked values only, 
	preventing extreme values in masked regions from distorting the 
	color scale.

	Examples
	--------
	>>> import numpy as np
	>>> import matplotlib.pyplot as plt
	>>> times = np.linspace(0, 1, 300)
	>>> freqs = np.array([4, 8, 12, 16, 20, 25, 30, 40])
	
	>>> # Simple heatmap
	>>> tfr = np.random.randn(8, 300)  # freqs x times
	>>> plt.figure()
	>>> plot_2d(tfr, x_val=times, y_val=freqs, cbar_label='Power (dB)')
	>>> plt.xlabel('Time (s)')
	>>> plt.ylabel('Frequency (Hz)')
	
	>>> # With statistical masking
	>>> from stats.stats_utils import perform_stats
	>>> tfr_subjects = np.random.randn(20, 8, 300)
	>>> # subjects x freqs x times
	>>> _, sig_mask, _ = perform_stats(tfr_subjects, stats='ttest')
	>>> plt.figure()
	>>> plot_2d(tfr_subjects, x_val=times, y_val=freqs, 
	...         mask=sig_mask, mask_value=np.nan, center_zero=True)
	
	>>> # Diverging colormap for t-statistics
	>>> t_stats = np.random.randn(8, 300)  # t-values
	>>> plt.figure()
	>>> plot_2d(t_stats, x_val=times, y_val=freqs, center_zero=True,
	...         cmap='RdBu_r', cbar_label='t-statistic')

	See Also
	--------
	plot_significance : Overlay statistical significance markers on 
	2D plots
	plot_timecourse : Plot 1D timecourse data with optional error bands
	perform_stats : Compute statistical tests and generate significance 
	masks
	"""

	if Z.ndim > 2:
		Z = Z.mean(axis=0)
	
	# Apply mask to set non-significant values to mask_value
	# This is for visualization only - does not affect statistical testing
	if mask is not None:
		Z = Z.copy()  # Don't modify the original array
		
		# Handle different mask formats
		if isinstance(mask, list):
			# Convert cluster list to boolean mask (for perm test results)
			# Each cluster is a tuple of arrays: (row_indices, col_indices) for 2D
			bool_mask = np.zeros(Z.shape, dtype=bool)
			for i, cluster in enumerate(mask):
				if p_vals is None or p_vals[i] <= p_thresh:
					# cluster is already (row_indices, col_indices) tuple for 2D
					bool_mask[cluster] = True
			mask = bool_mask
		
		# Store unmasked data range BEFORE applying mask (for colorbar limits)
		Z_unmasked_values = Z[mask]
		
		# Set non-significant values to mask_value (np.nan creates visual gaps)
		Z[~mask] = mask_value
	else:
		Z_unmasked_values = None

	# set extent
	x_lim = [0,Z.shape[-1]] if x_val is None else [x_val[0],x_val[-1]]
	y_lim = [0,Z.shape[-2]] if y_val is None else [y_val[0],y_val[-1]]
	extent = [x_lim[0],x_lim[1],y_lim[0],y_lim[1]]

	# Set up colormap and normalization for diverging data
	#TODO: fix or remove
	if center_zero:
		if cmap is None:
			# Red-Blue colormap, reversed (red=positive, blue=negative)
			cmap = 'RdBu_r'  
		cmap_obj = plt.cm.get_cmap(cmap)

		# Calculate data range for colorbar limits (ignoring NaN values)
		if Z_unmasked_values is not None and len(Z_unmasked_values) > 0:
			valid_vals = Z_unmasked_values[~np.isnan(Z_unmasked_values)]
			if len(valid_vals) > 0:
				data_min = valid_vals.min()
				data_max = valid_vals.max()
			else:
				data_min, data_max = 0, 1
		else:
			valid_Z = Z[~np.isnan(Z)]
			if len(valid_Z) > 0:
				data_min = valid_Z.min()
				data_max = valid_Z.max()
			else:
				data_min, data_max = 0, 1

		# Ensure the range includes zero for MidpointNormalize to work 
		# correctly
		vmin = min(data_min, 0)
		vmax = max(data_max, 0)
		norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
		kwargs.setdefault('norm', norm)
		kwargs.setdefault('cmap', cmap_obj)
	
	# do actuall plotting
	plt.imshow(Z,interpolation=interpolation,aspect='auto',origin='lower',
	    	extent=extent, **kwargs)
	
	# set ticks
	if nr_ticks_x is not None:
		plt.xticks(np.linspace(x_lim[0],x_lim[1],nr_ticks_x))

	if set_y_ticks:
		if isinstance(y_val, list):
			y_val = np.array(y_val)
		if nr_ticks_y is None:
			nr_ticks_y = 5 if len(y_val) > 5 else len(y_val)	
		idx = np.linspace(0, len(y_val)-1, nr_ticks_y).astype(int)
		ticks = y_val[idx]
		if np.allclose(np.diff(y_val),np.diff(y_val)[0],rtol=1e-2, atol=1e-8):
			plt.yscale('linear')
		else:
			plt.yscale('log')
		if np.issubdtype(y_val.dtype, np.floating):
			tick_labels = np.round(ticks).astype(int)
		else:
			tick_labels = ticks
		plt.yticks(ticks, tick_labels)
		plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
	
	# add colorbar
	if colorbar:
		cbar = plt.colorbar()
		if cbar_label:
			cbar.set_label(cbar_label)

def _get_continuous_segments(mask: np.ndarray) -> List[np.ndarray]:
	"""Convert boolean mask into list of continuous segments.

	Identifies contiguous regions of True values in a boolean array and 
	returns the indices for each continuous segment. Useful for finding 
	time windows where statistical significance occurs.

	Parameters
	----------
	mask : np.ndarray
		1D boolean array indicating significant timepoints/pixels. 
		True values are grouped into continuous segments, False values 
		serve as boundaries.

	Returns
	-------
	segments : list of np.ndarray
		List of 1D arrays, each containing the integer indices of a 
		continuous segment of True values. If no True values exist in 
		mask, returns empty list. If entire array is True, returns 
		single array [0, 1, ..., len(mask)-1].

	Notes
	-----
	This function is commonly used with permutation test results to 
	identify which time windows show significant differences, or with 
	other boolean masking operations.

	Handles edge cases where significant segments start at index 0 or 
	extend to the end of the array.

	Examples
	--------
	>>> mask = np.array([False, True, True, False, True, False])
	>>> segments = _get_continuous_segments(mask)
	>>> len(segments)
	2
	>>> segments[0]
	array([1, 2])
	>>> segments[1]
	array([4])
	
	>>> # No significant values
	>>> mask_empty = np.array([False, False, False])
	>>> _get_continuous_segments(mask_empty)
	[]
	
	>>> # Entire array significant
	>>> mask_all = np.array([True, True, True])
	>>> _get_continuous_segments(mask_all)
	[array([0, 1, 2])]

	See Also
	--------
	plot_significance : Visualizes significance using this function
	"""
	# Find boundaries of continuous segments
	diff = np.diff(mask.astype(int))
	segment_starts = np.where(diff == 1)[0] + 1
	segment_ends = np.where(diff == -1)[0] + 1

	# Handle edge cases
	if mask[0]:
		segment_starts = np.r_[0, segment_starts]
	if mask[-1]:
		segment_ends = np.r_[segment_ends, len(mask)]

	# Return list of index arrays for each segment
	segments = [np.arange(start, end) for 	
			 				start, end in zip(segment_starts, segment_ends)]
	return segments

def plot_significance(x:np.array,y:np.array,chance:float=0,p_thresh:float=0.05,
					  color:str=None,stats:str='perm',
					  smooth:bool=False,line_width:float = 4,
					   y_val:np.array=None,
					  sig_mask:Union[np.ndarray,list]=None,
					  p_cluster:Optional[float]=None,
					  threshold:Optional[float]=None,
					  **kwargs):
	"""Plot significance markers on existing plots.

	Overlays statistical significance indicators on timecourse or 
	2D plots, showing which time points/regions differ significantly 
	from chance. Supports both cluster-based permutation tests and 
	uncorrected/correctedpointwise tests with flexible visualization 
	options.

	Parameters
	----------
	x : np.ndarray
		X-axis values (typically time points in seconds or 
		milliseconds). Should have length matching temporal dimension of 
		y or sig_mask.
	y : np.ndarray
		Data array for statistical testing. Shape:
		- 2D: (n_subjects, n_timepoints) for 1D timecourse
		- 3D: (n_subjects, n_frequencies, n_timepoints) for 
		  2D time-frequency. Only used if sig_mask is not provided 
		  (for computing statistics).
	chance : float, default=0
		Chance level to test against (same as used in perform_stats).
		Only used if sig_mask is not provided.
	p_thresh : float, default=0.05
		P-value threshold for significance. Only used if sig_mask is not
		provided (filtering is done by perform_stats). For pre-computed
		sig_mask, this parameter is ignored since filtering should 
		alreadybe applied upstream.
	color : str or None, default=None
		Color for significance markers. If None:
		- For 1D plots: uses color of last plotted line
		- For 2D plots: uses 'white' for contrast
	stats : {'perm', 'ttest', 'fdr'}, default='perm'
		Statistical test type used (only relevant if sig_mask not 
		provided). Determines how significance is visualized:
		- 'perm': Cluster-based results. In 1D: thick line segments 
		  over cluster boundaries. In 2D: dashed contours outlining 
		  cluster shapes.
		- 'ttest'/'fdr': Pointwise results. In 1D: continuous line 
		  segments showing all significant timepoints. In 2D: dashed 
		  contours (if smooth gradient) OR filled regions (if discrete 
		  significant points).
	smooth : bool, default=False
		If True, apply Savitzky-Golay smoothing (1D) or Gaussian 
		smoothing (2D) to the data before plotting. For 1D plots: 
		smooths y_data using Savitzky-Golay filter (window=9, order=1). 
		For 2D plots: smooths the significance mask using 
		Gaussian filter (sigma=1.0) to reduce pixelated appearance. 
		Default is False.
	line_width : float, default=4
		Line width for significance markers on 1D plots. Default is 4.
	y_val : np.ndarray or None, default=None
		Y-axis values for 2D plots (e.g., frequency values). Required if
		plot is 2D (y.ndim == 3). Ignored for 1D plots.
	sig_mask : np.ndarray, list, or None, default=None
		Pre-computed significance mask/clusters (optional). If provided,
		skips statistical computation. Format depends on test type:
		- 'perm': List of cluster tuples (already filtered by p_thresh)
		- 'ttest'/'fdr': Boolean array of same shape as y
		If None, statistics are computed using perform_stats.
	p_cluster : float or None, optional
		Cluster-forming p-value threshold (permutation test only). 
		Automatically converts to threshold value as t-statistic. 
		If both p_cluster and threshold are None, MNE uses automatic.
		Default: None.
	threshold : float or None, optional
		Cluster-forming threshold as test statistic value (permutation 
		test only). Overrides p_cluster if specified. Default: None.
	**kwargs
		Additional keyword arguments passed to matplotlib plotting 
		functions(plt.contour, plt.contourf, or plt.plot depending on 
		plot type).

	Returns
	-------
	None
		Modifies the current matplotlib figure by adding significance 
		markers.

	Notes
	-----
	**1D vs 2D plot detection**: Automatically inferred from y.ndim:
	- 2D array (n_subjects, n_timepoints) → 1D timecourse plot
	- 3D array (n_subjects, n_frequencies, n_timepoints) → 2D 
	  time-frequencyplot

	**Permutation test visualization**: Clusters are outlined with 
	dashed contours, optionally after smoothing (sigma=1.0) to highlight 
	spatial extent while avoiding pixelated appearance.

	**Pointwise test visualization (ttest/fdr)**: Continuous segments of
	significance are identified and plotted as thick lines. Multiple
	disconnected segments are plotted separately using 
	_get_continuous_segments.

	**Color inference for 1D plots**: If color=None, automatically 
	extracts the color of the last plotted line on the axis. This allows 
	seamlessoverlays of significance on existing plots without 
	specifying colors.

	Examples
	--------
	>>> import numpy as np
	>>> import matplotlib.pyplot as plt
	>>> x = np.linspace(-0.5, 1, 300)
	>>> y = np.random.randn(20, 300)
	
	>>> # Plot timecourse with automatic statistics
	>>> plt.figure()
	>>> plot_timecourse(x, y.mean(axis=0), color='blue', label='Mean')
	>>> plot_significance(x, y, stats='ttest', color='red', 
	... 					smooth=False)
	>>> plt.xlabel('Time (s)')
	>>> plt.ylabel('Amplitude (µV)')
	
	>>> # Pre-compute statistics for efficiency (multiple plots)
	>>> t_vals, sig_mask, _ = perform_stats(y, stat_test='perm')
	>>> # ... plot multiple conditions ...
	>>> plot_significance(x, y, stats='perm', sig_mask=sig_mask)
	
	>>> # 2D time-frequency plot
	>>> tfr = np.random.randn(20, 8, 300)  # subjects, freqs, times
	>>> freqs = np.array([4, 8, 12, 16, 20, 25, 30, 40])
	>>> plot_2d(tfr.mean(axis=0), x_val=x, y_val=freqs)
	>>> plot_significance(x, tfr, y_val=freqs, stats='perm')

	See Also
	--------
	perform_stats : Compute statistical significance
	_get_continuous_segments : Extract continuous significance regions
	plot_timecourse : Plot underlying data
	plot_2d : Plot 2D heatmaps with optional masking
	"""
	
	# Infer plot type from data dimensions
	plot_type = '2d' if y.ndim == 3 else '1d'

	# Only compute stats if not provided
	if sig_mask is None:
		_, sig_mask, _ = perform_stats(y, chance, stats, p_thresh,
										p_cluster=p_cluster,
										threshold=threshold)

	if plot_type == '2d':
		# Require y_val for 2D plots
		if y_val is None:
			raise ValueError("y_val (e.g., freq values) required for 2D plots")
		
		# Handle 2D significance plotting
		x_lim = [x[0], x[-1]]
		y_lim = [y_val[0], y_val[-1]]
		extent = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]

		if color is None:
			color = 'white'

		# Plot significance contours based on test type
		if stats == 'perm':
			# Handle cluster-based results 
			# (already filtered by p_thresh in perform_stats)
			for cluster in sig_mask:
				cluster_mask = np.zeros(y.shape[1:])  # (n_y, n_x)
				
				# cluster is a tuple of arrays: (train_indices, test_indices)
				# For 2D GAT with shape (n_subjects, train_time, test_time),
				# after removing subject dimension we have (train_time, test_time)
				# cluster is already the correct (row_indices, col_indices) tuple
				
				cluster_mask[cluster] = 1  # Direct indexing with tuple

				# Smooth the cluster mask if requested
				if smooth:
					cluster_mask_smooth = gaussian_filter(
										cluster_mask.astype(float), sigma=1.0)
				else:
					cluster_mask_smooth = cluster_mask.astype(float)
				
				plt.contour(cluster_mask_smooth, levels=[0.5], colors=color,
							linestyles='dashed', linewidths=1,
							extent=extent, **kwargs)
		else:
			# Handle boolean mask results (ttest, fdr)
			# Apply smoothing if requested
			if smooth:
				sig_mask_smooth = gaussian_filter(sig_mask.astype(float), 
									  				sigma=1.0)
			else:
				sig_mask_smooth = sig_mask.astype(float)
			
			if sig_mask_smooth.max() > 0.5 and sig_mask_smooth.min() < 0.5:
				plt.contour(sig_mask_smooth, levels=[0.5], colors=color,
							linestyles='dashed', linewidths=1,
							extent=extent, **kwargs)
			elif sig_mask.any():  # If there are any significant points
				# Use alternative visualization - highlight significant regions
				plt.contourf(sig_mask.astype(float), levels=[0.5, 1.0], 
							colors=[color], alpha=0.3, extent=extent, **kwargs)
	
	else:
		# Get current line properties
		if color is None:
			current_line = plt.gca().get_lines()[-1]
			color = current_line.get_color()
			y_data = current_line.get_ydata()
		else:
			y_data = np.mean(y, axis=0)

		if smooth:
			y_data = savgol_filter(y_data, 9, 1)

		if stats == 'perm':
			for cl in sig_mask:
				plt.plot(x[cl], y_data[cl], linewidth=line_width, 
					color=color, **kwargs)
		else:
			for segment in _get_continuous_segments(sig_mask):
				plt.plot(x[segment], y_data[segment], linewidth=line_width,
						color=color, **kwargs)

def plot_erp_timecourse(
	erps: Union[list, dict], 
	times: np.array, 
	elec_oi: list, 
	lateralized: bool = False, 
	cnds: list = None, 
	colors: list = None, 
	show_SE: bool = False, 
	smooth: bool = False, 
	stats: Union[str, bool] = False,
	p_thresh: float = 0.05,
	p_cluster: Optional[float] = None,
	threshold: Optional[float] = None,
	window_oi: Tuple = None, 
	offset_axes: int = 10, 
	onset_times: Union[list, bool] = [0], 
	show_legend: bool = True, 
	**kwargs
):
	"""Visualize event-related potential (ERP) timecourses.

	Plots averaged ERP waveforms for specified conditions and 
	electrodes. Supports single electrode, multiple electrode groups 
	(averaged within group), and lateralized comparisons 
	(contralateral - ipsilateral differences). Automatically handles 
	unit conversion and time scaling.

	Parameters
	----------
	erps : list or dict
		ERP data. If list, treated as single condition. If dict, keys 
		are condition names, values are lists of mne.Evoked objects 
		(one per subject).
	times : ndarray
		Time array (seconds or milliseconds) corresponding to ERP 
		timepoints. Automatically converted to milliseconds if detected 
		in seconds (average difference < 0.1).
	elec_oi : list
		Electrode(s) of interest. Can be list of strings for single 
		group (e.g., ['Cz', 'CPz']), or list of lists for multiple 
		groups (e.g., [['C3', 'C5'], ['C4', 'C6']]). 
		Data averaged within groups.
	lateralized : bool, optional
		If True, plots contra-ipsilateral difference. Requires elec_oi 
		to have exactly 2 groups. Default: False.
	cnds : list, optional
		Condition names to include. If None, plots all conditions.
		Default: None.
	colors : list, optional
		Colors for waveforms. If fewer colors than waveforms, uses
		default tableau colors. Default: None.
	show_SE : bool, optional
		If True, displays shaded standard error band around waveform.
		Default: False.
	smooth : bool, optional
		If True, applies Savitzky-Golay smoothing to waveform.
		Default: False.
	stats : {'perm', 'ttest', 'fdr'} or False, optional
		Statistical test type. 'perm': permutation cluster test; 
		'ttest': t-test; 'fdr': false discovery rate correction; 
		False: no statistics. Default: False.
	p_thresh : float, optional
		P-value threshold for significance. Clusters/timepoints with 
		p <= p_thresh considered significant. Default: 0.05.
	p_cluster : float or None, optional
		Cluster-forming p-value threshold (permutation test only). 
		Automatically converts to threshold value as t-statistic. 
		If both p_cluster and threshold are None, MNE uses automatic.
		Default: None.
	threshold : float or None, optional
		Cluster-forming threshold as test statistic value (permutation 
		test only). Overrides p_cluster if specified. Default: None.
	window_oi : tuple, optional
		Time window (start_ms, end_ms) or 
		(start_ms, end_ms, 'pos'/'neg') to highlight with rectangle. 
		Third element specifies polarity to restrict y-axis 
		(e.g., show only positive half). Default: None.
	offset_axes : int, optional
		Pixel offset for despine. Default: 10.
	onset_times : list or False, optional
		Time points (ms) to mark with vertical dashed lines 
		(e.g., stimulus onset, response). If False, no lines drawn. 
		Default: [0].
	show_legend : bool, optional
		If True, displays legend for waveforms. Default: True.
	**kwargs
		Additional keyword arguments passed to plot_timecourse() and 
		plot_significance() (e.g., linewidth, alpha).

	Returns
	-------
	None
		Modifies matplotlib figure directly.

	Notes
	-----
	1. **Data preparation**: Data averaged across subjects and within 
	   electrode groups using ERP.group_erp(). Automatically converts 
	   volts to microvolts if detected (data range 1 nV - 1 mV).

	2. **Electrode grouping**: Single electrode names auto-wrapped as 
	   single-item lists. Multiple groups (e.g., contra/ipsi pairs) 
	   plotted as separate waveforms unless lateralized=True.
	
	3. **Lateralization**: When lateralized=True, requires exactly 2 
	   electrode groups. Computes difference (group1 - group2), 
	   typically contra-ipsi. Result shown as single waveform per 
	   condition.

	4. **Time unit handling**: Automatically detects and converts 
	   seconds to milliseconds. If window_oi specified, converted 
	   accordingly.

	5. **Window highlighting**: Rectangle drawn with dashed outline. 
	   Polarity filtering (pos/neg) limits display to upper/lower half 
	   of y-axis, useful for highlighting amplitude without obscuring 
	   data.

	6. **Color management**: Colors assigned per waveform 
	   (condition x group combinations). Removed colors not re-used; 
	   auto-cycles through tableau colors if insufficient provided.

	7. **Statistical testing**: When stats enabled, tests each waveform 
	   against zero baseline across subjects. For permutation tests, 
	   control cluster formation via p_cluster (intuitive p-value) or 
	   threshold (direct test statistic). Returns only significant 
	   clusters for 'perm' or significant timepoints for 'ttest'/'fdr'.

	Examples
	--------
	1. Plot single electrode ERP for all conditions:

	    >>> import numpy as np
	    >>> # Simulate ERP data: 30 subjects, 1 condition, 512 samples
	    >>> erps_data = {}\n
	    >>> for cnd in ['target', 'nontarget']:
	    ...     erps_data[cnd] = []\n
	    ...     for _ in range(30):  # 30 subjects
	    ...         evoked = type('obj', (), {
	    ...             'ch_names': ['Pz', 'Cz', 'Fz'],
	    ...             'times': np.linspace(-0.2, 0.8, 512),
	    ...             'data': np.random.randn(3, 512) * 1e-6,  # volts
	    ...             'pick': lambda x: type('obj', (), {
	    ...                 'data': np.random.randn(1, 512) * 1e-6
	    ...             })()
	    ...         })()\n
	    ...         erps_data[cnd].append(evoked)

	    >>> plot_erp_timecourse(
	    ...     erps=erps_data,
	    ...     times=np.linspace(-200, 800, 512),
	    ...     elec_oi=['Pz'],
	    ...     cnds=['target', 'nontarget'],
	    ...     colors=['red', 'blue'],
	    ...     show_SE=True,
	    ...     onset_times=[0],
	    ...     show_legend=True
	    ... )

	2. Plot contra/ipsi waveforms with statistical significance:

	    >>> plot_erp_timecourse(
	    ...     erps=erps_data,
	    ...     times=np.linspace(-200, 800, 512),
	    ...     elec_oi=[['P3', 'P5'], ['P4', 'P6']],
	    ...     cnds=['target'],
	    ...     colors=['red', 'blue'],
	    ...     window_oi=(300, 500, 'pos'),  # Highlight P300 window
	    ...     smooth=True,
	    ...     show_SE=True,
	    ...     stats='ttest',
	    ...     p_thresh=0.05
	    ... )

	3. Plot lateralized difference with permutation test:

	    >>> plot_erp_timecourse(
	    ...     erps=erps_data,
	    ...     times=np.linspace(-200, 800, 512),
	    ...     elec_oi=[['P3', 'P5'], ['P4', 'P6']],
	    ...     lateralized=True,
	    ...     cnds=['target', 'nontarget'],
	    ...     colors=['purple', 'orange'],
	    ...     window_oi=(300, 500),
	    ...     smooth=True,
	    ...     stats='perm',
	    ...     p_cluster=0.01,  # More stringent clustering
	    ...     show_legend=True
	    ... )

	See Also
	--------
	plot_timecourse : Plot generic 1D timecourse data
	plot_significance : Overlay statistical significance on plots
	plot_erp_topography : Plot topographic scalp maps of ERP amplitudes
	"""

	# Convert times from seconds to milliseconds if needed
	time_diff = np.diff(times).mean()
	if time_diff < 0.1:  # If average difference < 0.1, assume seconds
		times = times * 1000
		print(f"Times converted from seconds to milliseconds")
		if window_oi is not None:
			window_oi = [window_oi[0]*1000, window_oi[1]*1000] + window_oi[2:]	

	if isinstance(erps, list):
		erps = {'temp':erps}

	if cnds is not None:
		erps = {key:value for (key,value) in erps.items() if key in cnds}

	if isinstance(elec_oi[0],str): 
		elec_oi = [elec_oi]

	# Calculate the actual number of waveforms that will be plotted
	n_waveforms_per_condition = 1 if lateralized else len(elec_oi)
	total_waveforms = len(erps) * n_waveforms_per_condition

	if colors is None or len(colors) < total_waveforms:
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())
	
	for cnd in erps.keys():
		# extract all time courses for the current condition
		y = []
		for c, elec in enumerate(elec_oi):
			y_,_ = ERP.group_erp(erps[cnd],elec_oi = elec)
			y.append(y_)
		
		# Auto-detect and convert volts to microvolts for all conditions
		for i, y_ in enumerate(y):
			typical_value = np.abs(y_).mean()
			# Between 1 nV and 1 mV
			if typical_value < 1e-3 and typical_value > 1e-9:  
				y[i] = y_ * 1e6  # Convert volts to microvolts
				if i == 0:  # Only print once per condition
					print(f"Data for condition '{cnd}' converted from volts to" 
		   			" microvolts" )		
				
		# set up timecourses to plot	
		if lateralized:
			y = [y[0] - y[1]]
		
		# Create appropriate labels based on lateralization and 
		# number of conditions
		if lateralized:
			labels = [f'{cnd} (contra-ipsi)']
		elif len(y) == 1:
			labels = [cnd]
		else:
			labels = [f'{cnd} contra', f'{cnd} ipsi']

		#do actual plotting
		for i, y_ in enumerate(y):
			color = colors.pop(0) 
			plot_timecourse(times,y_,show_SE,smooth,
							label=labels[i],color=color,**kwargs)
			if stats:
				plot_significance(times, y_, 0, color=color, stats=stats,
							 p_thresh=p_thresh, p_cluster=p_cluster,
							 threshold=threshold, smooth=smooth, **kwargs)

	# clarify plot
	if window_oi is not None:
		_, _, ymin, ymax = plt.axis()
		if len(window_oi) == 3:
			ymin, ymax = (0, ymax) if window_oi[-1] == 'pos' else (ymin, 0)

		# Add a small margin to ymin and ymax to ensure visibility
		# Adjust the margin as needed (5% of height)
		margin = 0.05 * (ymax - ymin)  
		ymin += margin
		ymax -= margin
		# Add dashed grey outline for the time window of interest
		rect = plt.Rectangle(
			(window_oi[0], ymin),  
			window_oi[1] - window_oi[0],  
			ymax - ymin, 
			edgecolor='black',  
			facecolor='none', 
			linestyle='--')  
		plt.gca().add_patch(rect)

	if show_legend:
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys(),loc = 'best',
		prop={'size': 7},frameon=False)
	plt.xlabel('Time (ms)')
	plt.ylabel('\u03BC' + 'V')
	plt.axhline(0, color = 'black', ls = '--', lw=1)
	if onset_times:
		for t in onset_times:
			plt.axvline(t,color = 'black',ls='--',lw=1)

	sns.despine(offset = offset_axes)

def plot_tfr_timecourse(tfr:Union[dict,mne.time_frequency.AverageTFR], 
	elec_oi: list, 
	freq_oi: Union[int,Tuple] = None,
	lateralized: bool = False, 
	cnds: list = None, 
	colors: list = None, 
	timecourse: str = '2d',
	stats:Union[str,bool]=False,
	p_thresh:float=0.05,
	show_only_significant:bool=False,
	center_zero:bool=False,
	show_SE: bool = False, 
	smooth: bool = False, 
	window_oi: Tuple = None, 
	offset_axes: int = 10, 
	onset_times: Union[list, bool] = [0], 
	show_legend: bool = True, 
	ls: str = '-', 
	**kwargs
):
	"""Visualize time-frequency representation (TFR) data as timecourses.

	Plots time-frequency power spectrograms for specified electrodes and frequencies.
	Supports both 1D timecourse plots (averaged across frequencies) and 2D time-frequency
	heatmaps. Can display statistical significance testing results overlaid on the data.

	Parameters
	----------
	tfr : dict or mne.time_frequency.AverageTFR
		Time-frequency representation data. If dict, should have condition names as keys,
		each mapping to a list of individual TFR objects (typically one per subject).
		Each TFR object should have .times, .freqs, .ch_names, and .data attributes.
	elec_oi : list
		Electrode(s) of interest. Can be a list of single electrode names
		(e.g., ['Cz']) or a list of lists for multiple electrode groups
		(e.g., [['C3', 'C5'], ['C4', 'C6']] for contra/ipsi). Data will be averaged
		across electrodes within each group.
	freq_oi : int or tuple, optional
		Frequency-of-interest for filtering. If int, extracts single frequency closest
		to specified value. If tuple (start, end), extracts frequency range. If None,
		uses all frequencies. Default: None.
	lateralized : bool, optional
		If True, computes contra-ipsilateral difference (y[0] - y[1]). Requires elec_oi
		to have exactly 2 groups. Default: False.
	cnds : list, optional
		Condition names to plot. If None, uses all conditions in tfr. Default: None.
	colors : list, optional
		RGB or named colors for each condition (1D timecourse only). If fewer colors
		than conditions, uses default tableau colors. Default: None.
	timecourse : {'1d', '2d'}, optional
		Plot type. '1d' shows power averaged across frequencies vs time (supports
		multiple conditions/electrodes). '2d' shows time-frequency heatmap (single
		condition only). Default: '2d'.
	stats : {'perm', 'ttest', 'fdr'} or False, optional
		Statistical test type. 'perm': permutation test; 'ttest': t-test across
		subjects; 'fdr': false discovery rate correction; False: no statistics.
		Default: False.
	p_thresh : float, optional
		P-value threshold for significance. Clusters with p-value <= p_thresh are
		considered significant. Default: 0.05.
	show_only_significant : bool, optional
		If True, masks (sets to NaN) non-significant regions in 2D plots,
		showing only significant data regions. If False, shows full data
		with significance markers overlaid. Default: False.
	center_zero : bool, optional
		If True, centers colormap at zero (for 2D plots with diverging data).
		Default: False.
	show_SE : bool, optional
		If True, shows shaded standard error band around 1D timecourse.
		Default: False.
	smooth : bool, optional
		If True, applies smoothing (1D: Savitzky-Golay filter; 2D: Gaussian).
		Default: False.
	window_oi : tuple, optional
		Time window (start_ms, end_ms) to display. If None, shows full time range.
		Default: None.
	offset_axes : int, optional
		Pixel offset for despine operation. Default: 10.
	onset_times : list or False, optional
		Time points (ms) to mark with vertical lines (e.g., stimulus onset).
		If False, no lines drawn. Default: [0].
	show_legend : bool, optional
		If True, displays legend for 1D plots. Default: True.
	ls : str, optional
		Line style for 1D timecourses ('-', '--', '-.', ':'). Default: '-'.
	**kwargs
		Additional keyword arguments passed to plot_significance() and plot_timecourse().

	Returns
	-------
	None
		Modifies matplotlib figure directly.

	Notes
	-----
	1. **Time unit handling**: Automatically converts times from seconds to milliseconds
	   if detected (average time difference < 0.1).

	2. **2D timecourse limitation**: Only one condition can be plotted as 2D heatmap.
	   If multiple conditions specified, uses first condition only.

	3. **Electrode grouping**: Single electrode names are auto-wrapped in lists.
	   For lateralization, must provide exactly 2 electrode groups (contra, ipsi).

	4. **Frequency selection**: With freq_oi, individual frequencies are extracted
	   directly; frequency ranges are averaged across the range.

	5. **Statistical overlays**: For 2D plots, significance is shown as contours
	   (if show_only_significant=False) or as only significant regions (if show_only_significant=True).
	   For 1D plots, uses colored bands overlaid on timecourse.

	6. **Data averaging**: Power values are averaged across subjects (all TFR objects
	   in condition list) and within electrode groups.

	Examples
	--------
	1. Plot 2D time-frequency heatmap with statistical masking::

		>>> import numpy as np
		>>> # Simulate TFR data: 20 subjects, 64 channels, 10 frequencies, 300 timepoints
		>>> tfr_data = {}
		>>> for cnd in ['stim', 'rest']:
		...     tfr_data[cnd] = []
		...     for _ in range(20):  # 20 subjects
		...         tfr_obj = type('obj', (), {
		...             'times': np.linspace(-0.5, 1.5, 300),
		...             'freqs': np.arange(2, 31, 3),  # 10 frequencies
		...             'ch_names': ['Cz', 'C3', 'C4'],
		...             'data': np.random.randn(3, 10, 300)  # channels x freqs x times
		...         })()
		...         tfr_data[cnd].append(tfr_obj)

		>>> plot_tfr_timecourse(
		...     tfr=tfr_data,
		...     elec_oi=['Cz'],
		...     freq_oi=(8, 12),  # Alpha band
		...     timecourse='2d',
		...     cnds=['stim'],
		...     stats='ttest',
		...     show_only_significant=True,
		...     center_zero=False
		... )

	2. Plot 1D timecourse with multiple electrode groups (contra/ipsi)::

		>>> plot_tfr_timecourse(
		...     tfr=tfr_data,
		...     elec_oi=[['C3', 'C5'], ['C4', 'C6']],
		...     freq_oi=(10, 15),  # Theta-alpha boundary
		...     timecourse='1d',
		...     lateralized=False,
		...     cnds=['stim', 'rest'],
		...     colors=['red', 'blue'],
		...     show_SE=True,
		...     smooth=True,
		...     stats='perm',
		...     p_thresh=0.01
		... )

	3. Plot lateralized 1D timecourse with significance overlay::

		>>> plot_tfr_timecourse(
		...     tfr=tfr_data,
		...     elec_oi=[['C3', 'C5'], ['C4', 'C6']],
		...     freq_oi=10,  # Single frequency (Hz)
		...     timecourse='1d',
		...     lateralized=True,
		...     cnds=['stim'],
		...     show_SE=True,
		...     smooth=True,
		...     stats='fdr',
		...     p_thresh=0.05,
		...     onset_times=[0, 500]
		... )

	See Also
	--------
	plot_timecourse : Plot 1D timecourse data
	plot_2d : Plot 2D heatmap data
	plot_significance : Overlay statistical significance on plots
	"""
	
	if cnds is not None:
		tfr = {key:value for (key,value) in tfr.items() if key in cnds}
	else:
		print('No conditions specified. Using first condition in tfr')
		cnds = list(tfr.keys())

	if timecourse == '2d' and len(cnds) > 1:
		print(f'2d timecourse only supports one condition. '
		f'will show first condition only: {cnds[0]}')
		cnds = [cnds[0]]

	if colors is None or len(colors) < len(cnds):
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())

	if isinstance(elec_oi[0],str): 
		elec_oi = [elec_oi]

	times = tfr[cnds[0]][0].times
	# Convert times from seconds to milliseconds if needed
	time_diff = np.diff(times).mean()
	if time_diff < 0.1:  # If average difference < 0.1, assume seconds
		times = times * 1000
		print(f"Times converted from seconds to milliseconds")


	for cnd in cnds:
		# get indices of frequencies of interest
		freqs = tfr[cnd][0].freqs
		if freq_oi is not None:
			if isinstance(freq_oi, tuple):
				freq_idx = (np.abs(freqs - freq_oi[0]).argmin(),
							np.abs(freqs - freq_oi[1]).argmin())
				freq_idx = slice(freq_idx[0],freq_idx[1]+1)
			else:
				freq_idx = np.abs(freqs - freq_oi).argmin()

			freqs = freqs[freq_idx]

		# extract all time courses for the current condition
		y = []
		for c, elec in enumerate(elec_oi):
			# Stack individual TFR data for the specified electrodes
			idx = [tfr[cnd][0].ch_names.index(e) for e in elec]
			y_ = np.stack([tfr_.data[idx] for tfr_ in tfr[cnd]]).mean(axis = 1)
			if freq_oi is not None:
				if isinstance(freq_idx, int):
					y_ = np.expand_dims(y_[:, freq_idx], axis=-1)
				else:
					y_ = y_[:, freq_idx]
			y.append(y_)

		# Create appropriate labels based on lateralization and number of conditions
		if lateralized:
			labels = [f'{cnd} (contra-ipsi)']
		elif len(y) == 1:
			labels = [cnd]
		else:
			labels = [f'{cnd} contra', f'{cnd} ipsi']
				
		# set up timecourses to plot	
		if lateralized:
			y = [y[0] - y[1]]
			
		#do actual plotting
		for i, y_ in enumerate(y):
			if timecourse == '2d':
				# Calculate stats once if needed
				sig_mask = None
				p_vals = None
				if stats:
					_, sig_mask, p_vals = perform_stats(y_, 0, stats, p_thresh)
				
				# Plot with optional masking
				plot_2d(y_, x_val=times, y_val=freqs, cbar_label='Power (au)',
						colorbar=True,
						mask=sig_mask if show_only_significant else None,
						mask_value=np.nan, center_zero=center_zero)
				
				# Add contours only if not masking (when masking, zeros show significance)
				if stats and not show_only_significant:		
					plot_significance(times, y_, 0, color='white', stats=stats,
							y_val=freqs, sig_mask=sig_mask, p_vals=p_vals,
							**kwargs)
			else:
				color = colors.pop(0) 
				plot_timecourse(times,y_.mean(axis = 1),show_SE,smooth,
						label=labels[i],color=color,ls=ls,**kwargs)
				if stats:		
					plot_significance(times,y_.mean(axis = 1),0,
									color=color,stats=stats,
									smooth=smooth,**kwargs)

	if show_legend and timecourse == '1d':
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys(),loc = 'best',
		prop={'size': 7},frameon=False)


	plt.xlabel('Time (ms)')
	if timecourse == '2d':
		plt.ylabel('Frequency (Hz)')
	else:
		plt.ylabel('Power (au)')
	
	sns.despine(offset = offset_axes)
				
def plot_bdm_timecourse(bdms:Union[list,dict],cnds:list=None,timecourse:str='1d',
					colors:list=None,
					show_SE:bool=False,smooth:bool=False,method:str='auc',
					chance_level:float=0.5,stats:Union[str,bool]='perm',
					p_thresh:float=0.05,mask_nonsig:bool=False,
					center_zero:bool=False,
					p_cluster:Optional[float]=None,
					threshold:Optional[float]=None,
					freq_oi: Union[int,Tuple] = None,
					onset_times:Union[list,bool]=[0],offset_axes:int=10,
					show_legend:bool=True,ls = '-',**kwargs):

	if isinstance(bdms, dict):
		bdms = [bdms]	
	times = bdms[0]['info']['times']
	# Convert times from seconds to milliseconds if needed
	time_diff = np.diff(times).mean()
	if time_diff < 0.1:  # If average difference < 0.1, assume seconds
		times = times * 1000
		print(f"Times converted from seconds to milliseconds")
	
	if cnds is None:
		cnds = [key for key in bdms[0] if 'info' not in key]

	if timecourse != '1d' and len(cnds) > 1:
		print('2d timecourse only supports one condition. Plotting first ' \
		f'condition only {cnds[0]}')
		cnds = [cnds[0]]

	if colors is None or len(colors) < len(cnds) and timecourse == '1d':
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())

	for c, cnd in enumerate(cnds):
		# extract data
		y = np.stack([bdm[cnd]['dec_scores'] for bdm in bdms])
		color = colors[c]
		
		# Initialize y_label with default (updated in 2D branches)
		y_label = method

		if timecourse == '1d':
			# Extract diagonal if input is 2D (e.g., from GAT/TFR)
			if y.ndim > 2:
				# Get the minimum dimension to extract diagonal safely
				original_shape = y.shape
				min_dim = min(y.shape[1], y.shape[2])
				y = y[:, np.arange(min_dim), np.arange(min_dim)]
				print(f"Extracted diagonal from 2D GAT data "
					f"{original_shape[1:]} → {y.shape[1:]} "
					f"for condition '{cnd}'")
			
			# do actual plotting
			plot_timecourse(times,y,show_SE,smooth,
							label=cnd,color=color,ls=ls)
			if stats:		
				plot_significance(times,y,chance_level,
					 			color=color,stats=stats,
								smooth=smooth,**kwargs)
		else: 
			if timecourse == '2d_tfr':
				y_range = bdms[0]['info']['freqs']	
				y_label = 'Frequency (Hz)'
				y_ticks = True
			elif timecourse == '2d_GAT':
				test_times_x = bdms[0]['info']['test_times']
				y_range = bdms[0]['info']['times']
				y_label = 'Train time (ms)'
				y_ticks = False
			
			# Calculate stats once if needed
			sig_mask = None
			p_vals = None
			if stats:
				_, sig_mask, p_vals = perform_stats(y, 0, stats, p_thresh,
													p_cluster=p_cluster,
													threshold=threshold)
			
			# Plot with optional masking
			# For GAT: use test_times_x for x-axis (test time), train times for y-axis
			x_vals = test_times_x if timecourse == '2d_GAT' else times
			plot_2d(y, x_val=x_vals, y_val=y_range, colorbar=True,
		   				set_y_ticks=y_ticks, cbar_label=method,
						mask=sig_mask if mask_nonsig else None,
						mask_value=0, p_vals=p_vals, p_thresh=p_thresh,
						center_zero=center_zero, **kwargs)
			
			# Add contours only if not masking (when masking, zeros show significance)
			if stats and not mask_nonsig:
				plot_significance(x_vals, y, 0, color='white', stats=stats,
								y_val=y_range, sig_mask=sig_mask,
								p_vals=p_vals, **kwargs)				

	# fine tune plot	
	if show_legend:
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys(),loc = 'best',
		prop={'size': 7},frameon=False)

	if onset_times and timecourse == '1d':
		for t in onset_times:
			plt.axvline(t,color = 'black',ls='--',lw=1)
	
	if timecourse == '2d_GAT':
		plt.xlabel('Test time (ms)')
	else:
		plt.xlabel('Time (ms)')
	plt.ylabel(y_label)
	if timecourse == '1d':
		plt.axhline(chance_level, color = 'black', ls = '--', lw=1)

	sns.despine(offset = offset_axes)

def plot_ctf_timecourse(ctfs:Union[list,dict],cnds:list=None,colors:list=None,
						show_SE:bool=False,smooth:bool=False,timecourse:str='1d',
						output:str='raw_slopes',band_oi: str=None,
						stats:Union[str,bool]='perm',p_thresh:float=0.05,
						mask_nonsig:bool=False,center_zero:bool=False,
						onset_times:Union[list,bool]=[0],offset_axes:int=10,
						show_legend:bool=True,avg_bins:bool=False,**kwargs):
	
	if isinstance(ctfs, dict):
		ctfs = [ctfs]
	times = ctfs[0]['info']['times']

	# Convert times from seconds to milliseconds if needed
	time_diff = np.diff(times).mean()
	if time_diff < 0.1:  # If average difference < 0.1, assume seconds
		times = times * 1000
		print(f"Times converted from seconds to milliseconds")

	if cnds is None:
		cnds = [key for key in ctfs[0] if 'info' not in key]

	if timecourse != '1d' and len(cnds) > 1:
		print('2d timecourse only supports one condition. Plotting first ' \
		f'condition only {cnds[0]}')
		cnds = [cnds[0]]

	if isinstance(output, str):
		output = [output]

	if colors is None or len(colors) < len(cnds) and timecourse == '1d':
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())

	if band_oi is not None:
		band_idx = ctfs[0]['info']['bands'].index(band_oi)

	ylabel = f'CTF slope (au) - {band_oi}' if band_oi is not None \
											else 'CTF slope (au)'
	for c, cnd in enumerate(cnds):
		color = colors[c] 
		for o, out in enumerate(output):
			# extract data
			y = np.stack([ctf[cnd][out] for ctf in ctfs])
			if len(output) > 1:
				label = f'{cnd} - {out}'
			else:
				label = cnd

			# Select frequency band if needed (applies to both 1d and 2d_gat)
			if timecourse in ['1d', '2d_gat']:
				if y.shape[1]> 1:
					if band_oi is not None:
						y = y[:,band_idx,:]
					else:
						Warning('Multiple frequency bands detected but no ' \
						'band_oi specified. Averaging across all frequency ' \
						'bands.')
						y = y.mean(axis=1)
				else:
					y = np.squeeze(y,axis=1)

			# do actual plotting
			if timecourse == '1d':
				if y.ndim > 2 and avg_bins:
					y = y[:,:,~np.all(y == 0, axis=(0,1))].mean(axis=-1)
				if y.ndim > 2:
					for b in range(y.shape[-1]):
						y_ = y[:,:,b]
						if not np.all(y_ == 0):
							bin_label = f'{label} - bin_{b}'
							plot_timecourse(times,y_,show_SE,smooth,
										label=bin_label,color=colors[b],
										ls=['-','--'][o])
				else:
					plot_timecourse(times,y,show_SE,smooth,
									label=label,color=color,ls=['-','--'][o])

				if stats:
					#TODO: make also work for individual bins	
					#TODO: add chance level if needed	
					plot_significance(times,y,0,
									color=color,stats=stats,
									smooth=smooth,**kwargs)
			elif timecourse == '2d_tfr' or timecourse == '2d_gat':
				if timecourse == '2d_tfr':
					freqs = ctfs[0]['info']['freqs']	
					y_range = [np.mean(band) for band in freqs]
					ylabel = 'Frequency (Hz)'
					y_ticks = True
				else:  # 2d_gat
					y_range = times
					ylabel = 'Train time (ms)'
					y_ticks = False
				
				# Calculate stats once if needed
				sig_mask = None
				p_vals = None
				if stats:
					_, sig_mask, p_vals = perform_stats(y, 0, stats, p_thresh)
				
				# Plot with optional masking
				plot_2d(y, x_val=times, y_val=y_range, colorbar=True,
		   				set_y_ticks=y_ticks, cbar_label='CTF slope',
						mask=sig_mask if mask_nonsig else None,
						mask_value=0, p_vals=p_vals, p_thresh=p_thresh,
						center_zero=center_zero, **kwargs)
			
				# Add contours only if not masking (when masking, zeros show significance)
				if stats and not mask_nonsig:
					plot_significance(times, y, 0, color='white', stats=stats,
									y_val=y_range, sig_mask=sig_mask, 
									p_vals=p_vals, **kwargs)	
			elif timecourse == '2d_ctf':
				if y.shape[1]> 1:
					Warning('2d CTF timecourse only supports single output.' \
					' Plotting first output only.')
					y = y[:,0]
				else:
					y = np.squeeze(y,axis=1)
				if y.ndim > 3:
					Warning('2d CTF timecourse only supports single channel.' \
					'Individual channels will be averaged.')
					y = y.mean(axis=-2)
				if y.shape[-1]%2 == 0:
					y = np.concatenate([y, y[:, :, 0:1] ], axis=2)
				y = np.swapaxes(y,1,2)
				y_range = np.linspace(-180,180,y.shape[1])
				ylabel = 'Channel offset (deg)'
				plot_2d(y,x_val=times,y_val=y_range,colorbar=True,
		   				set_y_ticks=False,cbar_label='Channel response',**kwargs)

	# fine tune plot	
	if show_legend:
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys(),loc = 'best',
		prop={'size': 7},frameon=False)
	
	plt.xlabel('Time (ms)')
	plt.ylabel(ylabel)
	if timecourse == '1d':
		if onset_times:
			for t in onset_times:
				plt.axvline(t,color = 'black',ls='--',lw=1)
		plt.axhline(0, color = 'black', ls = '--', lw=1)

	sns.despine(offset = offset_axes)
	
def plot_erp_topography(erps:Union[list,dict],times:np.array,
						window_oi:tuple=None,cnds:list=None,
						topo:str='raw',montage:str='biosemi64',**kwargs):

	if isinstance(erps, list):
		erps = {'temp':erps}

	if cnds is not None:
		erps = {key:value for (key,value) in erps.items() if key in cnds}

	if window_oi is None:
		window_oi = (times[0],times[-1])
	idx = get_time_slice(times, window_oi[0],window_oi[1])

	for c, cnd in enumerate(erps.keys()):
		ax = plt.subplot(1,len(erps), c+1, title = cnd)
		_, evoked = ERP.group_erp(erps[cnd],set_mean=True)
		data = evoked._data[:,idx].mean(axis = 1)

		if topo == 'diff':
			preflip = np.copy(data)
			# visualize contra vs. ipsi
			ch_names = evoked.ch_names
			pairs = get_diff_pairs(montage, ch_names)
			# flip data
			for el, pair in pairs.items():
				data[ch_names.index(el)] = preflip[pair[0]] - preflip[pair[1]]

		# do actual plotting
		plot_topography(data,montage=montage,axes=ax,**kwargs)

def plot_topography(X:np.array,ch_types:str='eeg',montage:str='biosemi64',
					sfreq:int=512.0,**kwargs):

	# create montage 
	ch_names = mne.channels.make_standard_montage(montage).ch_names
	info = mne.create_info(ch_names, ch_types=ch_types,sfreq=sfreq)
	info.set_montage(montage)

	# do actuall plotting
	mne.viz.plot_topomap(X, info,**kwargs)











