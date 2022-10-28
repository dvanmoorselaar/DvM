import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from eeg_analyses.ERP import *
from stats.nonparametric import bootstrap_SE
from typing import Optional, Generic, Union, Tuple, Any
from support.support import get_time_slice

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


def plot_time_course(x:np.array,y:np.array,show_SE:bool=False,**kwargs):

	if y.ndim > 1:
		if show_SE:
			err, y = bootstrap_SE(y)
		else:
			y = y.mean(axis=0)

	plt.plot(x,y,**kwargs)

	if show_SE:
		kwargs.pop('label', None)
		plt.fill_between(x,y+err,y-err,alpha=0.2,**kwargs)
	
def plot_erp_time_course(erps:Union[list,dict],times:np.array,elec_oi:list,
						contra_ipsi:str=None,colors:list=None,
						show_SE:bool=False,window_oi:Tuple=None,
						offset_axes:int=10,onset_times:Union[list,bool]=[0],
						show_legend:bool=True,ls = '-'):

	if isinstance(erps, list):
		erps = {'temp':erps}
	
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
		if contra_ipsi == 'd_wave':
			y = [y[0] - y[1]]
		labels = [cnd] if len(y) == 1 else ['contra','ipsi']

		#do actual plotting
		for i, y_ in enumerate(y):
			color = colors.pop() 
			plot_time_course(times,y_,show_SE,
							label=labels[i],color=color,ls=ls)

	# clarify plot
	if window_oi is not None:
		_, _, ymin, ymax = plt.axis()
		if len(window_oi) == 3:
			if window_oi[-1] == 'pos':
				ymin= 0
				color = 'red'
			elif window_oi[-1] == 'neg':
				ymax = 0
				color = 'blue'
		idx = get_time_slice(times, window_oi[0],window_oi[1])
		plt.fill_between(times[idx], ymin, ymax, color = color, alpha = 0.2)

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















