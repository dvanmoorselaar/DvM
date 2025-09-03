import mne
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

from scipy.signal import savgol_filter
from eeg_analyses.ERP import *
from stats.nonparametric import bootstrap_SE
from typing import Optional, Generic, Union, Tuple, Any
from support.support import get_time_slice, get_diff_pairs

from IPython import embed

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

def plot_time_course(x:np.array,y:np.array,
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

def plot_2d(X:np.array,mask:np.array=None,x_val:np.array=None,
	    	y_val:np.array=None,colorbar:bool=True,nr_ticks_x:np.array=None,
			nr_ticks_y:np.array=5,**kwargs):

	if X.ndim > 2:
		X = X.mean(axis=0)

	# set extent
	x_lim = [0,X.shape[-1]] if x_val is None else [x_val[0],x_val[-1]]
	y_lim = [0,X.shape[-2]] if y_val is None else [y_val[0],y_val[-1]]
	extent = [x_lim[0],x_lim[1],y_lim[0],y_lim[1]]

	# do actuall plotting
	plt.imshow(X,interpolation='nearest',aspect='auto',origin='lower',
	    	extent=extent, **kwargs)
	
	# set ticks
	if nr_ticks_x is not None:
		plt.xticks(np.linspace(x_lim[0],x_lim[1],nr_ticks_x))

	if nr_ticks_y is None:
		nr_ticks_y = 5

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
		plt.colorbar()

	# plot the mask
	if mask is not None:
		plt.contour(mask,levels=[0],colors='black',linestyles='dashed',
	      			linewidths=0.5,extent=extent)

def plot_significance(x:np.array,y:np.array,p_thresh:float=0.05,
					  stats:str='perm',marker_y:float=0.48,**kwargs):

	if stats == 'perm':
		(t_obs, 
		clusters, 
		clust_pv, 
		H0) = mne.stats.permutation_cluster_1samp_test(y)

	for cl, p_val in zip(clusters, clust_pv):
		if p_val <= p_thresh:
			plt.plot(x[cl], marker_y * np.ones_like(x[cl]),**kwargs)

def plot_erp_time_course(
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
			`plot_time_course` function.

	Returns:
		None: The function generates a plot but does not return 
		any value.
	"""

	if isinstance(erps, list):
		erps = {'temp':erps}

	if cnds is not None:
		erps = {key:value for (key,value) in erps.items() if key in cnds}
	
	if colors is None or len(colors) < len(erps):
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())

	if isinstance(elec_oi[0],str): 
		elec_oi = [elec_oi]
	
	for cnd in erps.keys():
		# extract all time courses for the current condition
		y = []
		for c, elec in enumerate(elec_oi):
			y_,_ = ERP.group_erp(erps[cnd],elec_oi = elec)
			y.append(y_)
				
		# set up timecourses to plot	
		if lateralized:
			y = [y[0] - y[1]]
		labels = [cnd] if len(y) == 1 else ['contralateral','ipsilateral']

		#do actual plotting
		for i, y_ in enumerate(y):
			color = colors.pop(0) 
			plot_time_course(times,y_,show_SE,smooth,
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
		print('2d timecourse only supports one condition. ' \
		'will show first condition only')
		timecourse = 'raw_slopes'

	if colors is None or len(colors) < len(cnds):
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())

	if isinstance(elec_oi[0],str): 
		elec_oi = [elec_oi]

	times = tfr[cnds[0]][0].times

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
				
		# set up timecourses to plot	
		if lateralized:
			y = [y[0] - y[1]]
		labels = [cnd] if len(y) == 1 else ['contralateral','ipsilateral']
			
		#do actual plotting
		for i, y_ in enumerate(y):
			if timecourse == '2d':
				plot_2d(y_, None,times,freqs,colorbar=True)
			else:
				color = colors.pop(0) 
				plot_time_course(times,y_.mean(axis = 1),show_SE,smooth,
						label=labels[i],color=color,ls=ls,**kwargs)

def plot_ctf_time_course(ctfs:Union[list,dict],cnds:list=None,colors:list=None,
						show_SE:bool=False,smooth:bool=False,
						output:str='raw_slopes',stats:Union[str,bool]='perm',
						onset_times:Union[list,bool]=[0],offset_axes:int=10,
						show_legend:bool=True,avg_bins:bool=False,**kwargs):
	
	if isinstance(ctfs, dict):
		ctfs = [ctfs]
	times = ctfs[0]['info']['times']

	if cnds is None:
		cnds = [key for key in ctfs[0] if 'info' not in key]

	if isinstance(output, str):
		output = [output]

	if colors is None or len(colors) < len(cnds):
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())

	for c, cnd in enumerate(cnds):
		color = colors[c] 
		for o, out in enumerate(output):
			# extract data
			y = np.stack([ctf[cnd][out] for ctf in ctfs])
			if len(output) > 1:
				label = f'{cnd} - {out}'
			else:
				label = cnd

			if y.ndim > 2 and avg_bins:
				y = y[:,:,~np.all(y == 0, axis=(0,1))].mean(axis=-1)
			if y.ndim > 2:
				for b in range(y.shape[-1]):
					y_ = y[:,:,b]
					if not np.all(y_ == 0):
						bin_label = f'{label} - bin_{b}'
						plot_time_course(times,y_,show_SE,smooth,
									label=bin_label,color=colors[b],
									ls=['-','--'][o])
			else:
				plot_time_course(times,y,show_SE,smooth,
								label=label,color=color,ls=['-','--'][o])
			if stats:
				if c == 0:
					y_ = np.stack([ctf[cnd][out] for cnd in cnds 
															for ctf in ctfs])
					
					if y_.ndim > 2 and avg_bins:
						y_ = y_[:,:,~np.all(y_ == 0, axis=(0,1))].mean(axis=-1)				
					y_ = np.reshape(y_,(len(cnds),-1,y_.shape[-1]))
					y_min = np.mean(y_, axis = 1).min()
					y_max = np.mean(y_, axis = 1).max()
					step = (y_max - y_min)/20

				
				marker_y = y_min + np.abs(step*c)
				plot_significance(times,y,stats=stats,
								color=color,marker_y=marker_y,
								ls=['-','--'][o])
			
	# fine tune plot	
	if show_legend:
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys(),loc = 'best',
		prop={'size': 7},frameon=False)

	if onset_times:
		for t in onset_times:
			plt.axvline(t,color = 'black',ls='--',lw=1)
	
	plt.xlabel('Time (ms)')
	plt.ylabel(output)
	plt.axhline(0, color = 'black', ls = '--', lw=1)

	sns.despine(offset = offset_axes)


def plot_bdm_time_course(bdms:Union[list,dict],cnds:list=None,colors:list=None,
						show_SE:bool=False,smooth:bool=False,method:str='auc',
						chance_level:float=0.5,stats:Union[str,bool]='perm',
						onset_times:Union[list,bool]=[0],offset_axes:int=10,
						show_legend:bool=True,ls = '-',**kwargs):

	if isinstance(bdms, dict):
		bdms = [bdms]	
	times = bdms[0]['info']['times']
	
	if cnds is None:
		cnds = [key for key in bdms[0] if 'info' not in key]

	if colors is None or len(colors) < len(cnds):
		print('not enough colors specified. Using default colors')
		colors = list(mcolors.TABLEAU_COLORS.values())

	for c, cnd in enumerate(cnds):
		# extract data
		y = np.stack([bdm[cnd]['dec_scores'] for bdm in bdms])
		color = colors[c] 
		plot_time_course(times,y,show_SE,smooth,
							label=cnd,color=color,ls=ls)
		if stats:
			if c == 0:
				y_ = np.stack([bdm[cnd]['dec_scores'] for cnd in cnds 
				   										for bdm in bdms])
				y_ = np.reshape(y_,(len(cnds),-1,y_.shape[-1]))
				y_min = np.mean(y_, axis = 1).min()
				y_max = np.mean(y_, axis = 1).max()
				step = (y_max - y_min)/25

			marker_y = y_min - np.abs(y_min * step*c)
			plot_significance(times,y-chance_level,stats=stats,
					 		 color=color,marker_y = marker_y)

	# fine tune plot	
	if show_legend:
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys(),loc = 'best',
		prop={'size': 7},frameon=False)

	if onset_times:
		for t in onset_times:
			plt.axvline(t,color = 'black',ls='--',lw=1)
	
	plt.xlabel('Time (ms)')
	plt.ylabel(method)
	plt.axhline(chance_level, color = 'black', ls = '--', lw=1)

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
			








def contraIpsiPlotter(contra_X, ipsi_X, times, labels, colors, sig_mask = False, p_val = 0.05, errorbar = False, legend_loc = 'best'):
	'''

	'''

	# loop over all waveforms
	for i, (contra, ipsi) in enumerate(zip(contra_X, ipsi_X)):
		d_wave = contra - ipsi
		
		# do actual plotting
		plotTimeCourse(times, d_wave, color = colors[i], label = labels[i],
					   mask = sig_mask, mask_p_val = p_val, errorbar = errorbar)

	plt.axvline(x = 0, ls = '--', color = 'black')
	plt.axhline(y = 0, ls = '--', color = 'black')			
	plt.legend(loc = legend_loc)		
	sns.despine(offset=10, trim = False)


def plotTimeCourse(times, X, color = 'blue', label = None, mask = False, mask_p_val = 0.05, paired = False, errorbar = False):
	'''

	'''

	if X.ndim > 1:
		err, x = bootstrap(X)
	else:
		x = X	
	if errorbar:
		plt.fill_between(times, x + err, x - err, alpha = 0.2, color = color)

	if type(mask) != bool:
		mask = clusterMask(X, mask, mask_p_val)
		x_sig = np.ma.masked_where(~mask, x)
		x_nonsig = np.ma.masked_where(mask, x)
		plt.plot(times, x_sig, color = color, ls = ':')
		plt.plot(times, x_nonsig, label = label, color = color)
	else:
		plt.plot(times, x, label = label, color = color)


def plotSignificanceBars(X1, X2, times, y, color, p_val = 0.05, paired = True, show_descriptives = False, lw = 2, ls = '-'):
	'''

	'''

	# find significance mask
	mask = clusterMask(X1, X2, p_val, paired = paired)
	y_sig = np.ma.masked_where(~mask, np.ones(mask.size) * y)
	plt.plot(times, y_sig, color = color, ls = ls, lw = lw)
	if show_descriptives:
		embed()















