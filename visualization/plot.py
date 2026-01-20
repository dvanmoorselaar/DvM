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
from statsmodels.stats.multitest import fdrcorrection
from analysis.ERP import *
from stats.stats_utils import bootstrap_SE
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

def plot_timecourse(x:np.array,y:np.array,
					show_SE:bool=False,smooth:bool=False,**kwargs):

	if y.ndim > 1:
		if show_SE:
			err, y = bootstrap_SE(y)
		else:
			y = y.mean(axis=0)

	if smooth:
		y = savgol_filter(y, 9, 1)

	plt.plot(x,y,**kwargs)

	if show_SE:
		kwargs.pop('label', None)
		plt.fill_between(x,y+err,y-err,alpha=0.2,**kwargs)

def _perform_stats(
		y: np.ndarray, 
		chance: float = 0, 
		stat_test: str = 'perm',
		p_thresh: float = 0.05
	) -> Tuple[np.ndarray, list, np.ndarray]:
	#TODO: add option to return a mask with signifacant points only
	#TODO: add statfun option for permutation test
	"""Perform statistical testing and return consistent output format.

	Args:
		y: Data array. For 1D: (n_subjects, n_timepoints)
						For 2D: (n_subjects, n_frequencies, n_timepoints)
		chance: Chance level for statistical testing
		stat_test: Statistical test type ('perm', 'ttest', 'fdr')
		p_thresh: P-value threshold for significance
		adjacency: Adjacency matrix for 2D clustering (only for perm test)

	Returns:
		Tuple containing:
			- test_stat: Test statistic (t-values)
			- clusters/sig_mask: For perm test: list of significant clusters
								For others: boolean mask of significant timepoints/pixels
			- p_vals: P-values for each cluster/timepoint/pixel
	"""

    # Determine input data dimensionality
	is_2d = y.ndim == 3

	if stat_test == 'perm':
		(t_obs, 
		clusters, 
		p_vals, 
		H0) = mne.stats.permutation_cluster_1samp_test(y - chance)
		return t_obs, clusters, p_vals
	elif stat_test == 'ttest':
		t_vals, p_vals = stats.ttest_1samp(y, chance, axis=0)
		sig_mask = p_vals < p_thresh
		return t_vals, sig_mask, p_vals
	elif stat_test == 'fdr':
		t_vals, p_vals = stats.ttest_1samp(y, chance, axis=0)
		if is_2d:
			# Apply FDR correction across all 2D points
			_, p_vals_fdr = fdrcorrection(p_vals.flatten())
			p_vals_fdr = p_vals_fdr.reshape(p_vals.shape)
		else:
			_, p_vals_fdr = fdrcorrection(p_vals)
		sig_mask = p_vals_fdr < p_thresh
		
		return t_vals, sig_mask, p_vals_fdr

def _get_continuous_segments(mask: np.ndarray) -> List[np.ndarray]:
	"""Convert boolean mask into list of continuous segments.

	Args:
		mask: Boolean array indicating significant timepoints

	Returns:
		List of arrays containing indices for each continuous segment
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
					  p_vals:np.ndarray=None,
					  **kwargs):
	"""
	Plot significance markers on existing plots.
	
	Args:
		x: X-axis values (time points)
		y: Data array for stats calculation (only used if sig_mask not provided)
		chance: Chance level for statistical testing
		p_thresh: P-value threshold for significance
		color: Color for significance markers
		stats: Statistical test type ('perm', 'ttest', 'fdr')
		smooth: Whether to smooth the data
		line_width: Width of significance lines (1D plots)
		y_val: Y-axis values for 2D plots (e.g., frequencies)
		sig_mask: Pre-computed significance mask (optional, avoids re-calculating stats)
		p_vals: Pre-computed p-values (optional, used with sig_mask)
		**kwargs: Additional plotting arguments
	"""
	
	# Infer plot type from data dimensions
	plot_type = '2d' if y.ndim == 3 else '1d'

	# Only compute stats if not provided
	if sig_mask is None:
		_, sig_mask, p_vals = _perform_stats(y, chance, stats, p_thresh)

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
			for cluster, p_val in zip(sig_mask, p_vals):
				if p_val <= p_thresh:
					cluster_mask = np.zeros(y.shape[1:])  # (n_freqs, n_times)
					cluster_mask[cluster] = 1

					# Smooth the cluster mask
					cluster_mask_smooth = gaussian_filter(
										cluster_mask.astype(float), sigma=1.0)
					plt.contour(cluster_mask_smooth, levels=[0.5], colors=color,
								linestyles='dashed', linewidths=1,
								extent=extent, **kwargs)
		else:
			# Handle boolean mask results (ttest, fdr)
			# Apply same smoothing for visual consistency
			sig_mask_smooth = gaussian_filter(sig_mask.astype(float), sigma=1.0)
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
			for cl, p_val in zip(sig_mask, p_vals):
				if p_val <= p_thresh:
					plt.plot(x[cl], y_data[cl], linewidth=line_width, 
						color=color, **kwargs)
		else:
			for segment in _get_continuous_segments(sig_mask):
				plt.plot(x[segment], y_data[segment], linewidth=line_width,
						color=color, **kwargs)

def plot_2d(Z:np.array,x_val:np.array=None,
	    	y_val:np.array=None,colorbar:bool=True,nr_ticks_x:np.array=None,
			nr_ticks_y:np.array=5, set_y_ticks:bool=True,
			interpolation:str='bilinear',cbar_label:str=None,
			mask:Union[np.ndarray,list]=None,mask_value:float=0,
			p_vals:np.ndarray=None,p_thresh:float=0.05,
			center_zero:bool=False,cmap:str=None,
			**kwargs):
	"""
	Plot 2D heatmap with optional masking of non-significant values.
	
	Args:
		Z: 2D or 3D array to plot (if 3D, averaged over first dimension)
		x_val: X-axis values
		y_val: Y-axis values
		colorbar: Whether to show colorbar
		nr_ticks_x: Number of x-axis ticks
		nr_ticks_y: Number of y-axis ticks
		set_y_ticks: Whether to set y-axis ticks
		interpolation: Interpolation method for imshow
		cbar_label: Label for colorbar
		mask: Boolean array or cluster list indicating which values to keep
		mask_value: Value to set masked pixels to (default: 0 for chance level)
		p_vals: P-values for each cluster (only used with cluster list mask)
		p_thresh: P-value threshold for significance (only used with cluster list mask)
		center_zero: If True, use diverging colormap centered at zero (white)
		cmap: Colormap name. If None and center_zero=True, uses 'RdBu_r'
		**kwargs: Additional arguments passed to plt.imshow
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
			bool_mask = np.zeros(Z.shape, dtype=bool)
			for i, cluster in enumerate(mask):
				# Only include significant clusters
				if p_vals is None or p_vals[i] <= p_thresh:
					bool_mask[cluster] = True
			mask = bool_mask
		
		# Store unmasked data range BEFORE applying mask (for colorbar limits)
		Z_unmasked_values = Z[mask]
		
		# Use masked array to properly handle non-significant values
		# This prevents interpolation artifacts - masked pixels won't be displayed
		Z = np.ma.masked_where(~mask, Z)
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
			cmap = 'RdBu_r'  # Red-Blue colormap, reversed (red=positive, blue=negative)
		cmap_obj = plt.cm.get_cmap(cmap)
		cmap_obj.set_bad(color='white')

		# Calculate data range for colorbar limits
		if Z_unmasked_values is not None and len(Z_unmasked_values) > 0:
			data_min = Z_unmasked_values.min()
			data_max = Z_unmasked_values.max()
		else:
			if isinstance(Z, np.ma.MaskedArray):
				data_min = Z.compressed().min()
				data_max = Z.compressed().max()
			else:
				data_min = Z.min()
				data_max = Z.max()

		# Ensure the range includes zero for MidpointNormalize to work correctly
		vmin = min(data_min, 0)
		vmax = max(data_max, 0)
		norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
		kwargs.setdefault('norm', norm)
		kwargs.setdefault('cmap', cmap_obj)

		# Prevent interpolation artifacts for masked arrays
		if isinstance(Z, np.ma.MaskedArray) and np.ma.is_masked(Z):
			interpolation = 'nearest'
	
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


def plot_erp_timecourse(
	erps: Union[list, dict], 
	times: np.array, 
	elec_oi: list, 
	lateralized: bool = False, 
	cnds: list = None, 
	colors: list = None, 
	show_SE: bool = False, 
	smooth: bool = False, 
	window_oi: Tuple = None, 
	offset_axes: int = 10, 
	onset_times: Union[list, bool] = [0], 
	show_legend: bool = True, 

	ls: str = '-', 
	**kwargs
):
	"""
	Plots the ERP time course for specified conditions and 
	electrodes.

	This function visualizes ERP time courses for selected 
	conditions and electrodes, optionally showing contralateral vs. 
	ipsilateral waveforms or difference waveforms. 
	It supports customization of colors, smoothing, 
	statistical output, and highlighting specific time windows.

	Args:
		erps (Union[list, dict]): ERP data to plot. Can be a list of 
			evoked objects (mne.Evoked) or a dictionary where keys 
			are condition names and values are lists of evoked 
			objects.
		times (np.array): Array of time points corresponding to the 
			ERP data.
		elec_oi (list): Electrodes of interest. If plotting 
			contralateral vs. ipsilateral waveforms, specify a list 
			of lists (e.g., `[['O1'], ['O2']]`).
		lateralized (bool, optional): Specifies whether to plot 
			difference waveforms. If True, plots the difference between
			the contralateral and ipsilateral waveforms as specified in
			elec_oi. If False, plots the average of the specicified 
			electrodes, which can be single list or a list of lists 
			for contralateral and ipsilateral waveforms
			Defaults to False.
		cnds (list, optional): List of conditions to include in the
			plot. If None, all conditions are plotted. 
			Defaults to None.
		colors (list, optional): List of colors for the conditions. 
			If not enough colors are specified, default Tableau 
			colors are used. Defaults to None.
		show_SE (bool, optional): If True, plots the standard error 
			of the mean (SEM) for the ERP data. Defaults to False.
		smooth (bool, optional): If True, applies smoothing to the 
			ERP time course. Defaults to False.
		window_oi (Tuple, optional): Time window of interest 
			(start, end, polarity). The tuple specifies the start and 
			end times of the window in seconds, and optionally the 
			polarity (`'pos'` or `'neg'`). If the polarity is not 
			provided, the window is highlighted without adjusting ymin 
			or ymax. Defaults to None.
		offset_axes (int, optional): Offset for the axes when using 
			seaborn's `despine` function. Defaults to 10.
		onset_times (Union[list, bool], optional): List of onset 
			times to mark with vertical lines on the plot. If False, 
			no onset times are marked. Defaults to `[0]`.
		show_legend (bool, optional): If True, displays a legend for 
			the plot. Defaults to True.
		ls (str, optional): Line style for the ERP time course. 
			Defaults to `'-'`.
		**kwargs: Additional keyword arguments passed to the 
			`plot_timecourse` function.

	Returns:
		None: The function generates a plot but does not return 
		any value.
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
		
		# Create appropriate labels based on lateralization and number of conditions
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
							label=labels[i],color=color,ls=ls,**kwargs)

	# clarify plot
	if window_oi is not None:
		_, _, ymin, ymax = plt.axis()
		if len(window_oi) == 3:
			ymin, ymax = (0, ymax) if window_oi[-1] == 'pos' else (ymin, 0)

		# Add a small margin to ymin and ymax to ensure visibility
		margin = 0.05 * (ymax - ymin)  # Adjust the margin as needed (5% of height)
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
	mask_nonsig:bool=False,
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
	
	#TODO: make it work with mne.time_frequency.AverageTFR objects
	
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
					_, sig_mask, p_vals = _perform_stats(y_, 0, stats, p_thresh)
				
				# Plot with optional masking
				plot_2d(y_, x_val=times, y_val=freqs, cbar_label='Power (au)',
						colorbar=True,
						mask=sig_mask if mask_nonsig else None,
						mask_value=0, p_vals=p_vals, p_thresh=p_thresh,
						center_zero=center_zero)
				
				# Add contours only if not masking (when masking, zeros show significance)
				if stats and not mask_nonsig:		
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

		if timecourse == '1d':
			y_label = method
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
				times = bdms[0]['info']['test_times']
				y_range = times
				y_label = 'Train time (ms)'
				y_ticks = False
			
			# Calculate stats once if needed
			sig_mask = None
			p_vals = None
			if stats:
				_, sig_mask, p_vals = _perform_stats(y, 0, stats, p_thresh)
			
			# Plot with optional masking
			plot_2d(y, x_val=times, y_val=y_range, colorbar=True,
		   				set_y_ticks=y_ticks, cbar_label=method,
						mask=sig_mask if mask_nonsig else None,
						mask_value=0, p_vals=p_vals, p_thresh=p_thresh,
						center_zero=center_zero, **kwargs)
			
			# Add contours only if not masking (when masking, zeros show significance)
			if stats and not mask_nonsig:
				plot_significance(times, y, 0, color='white', stats=stats,
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
					_, sig_mask, p_vals = _perform_stats(y, 0, stats, p_thresh)
				
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











