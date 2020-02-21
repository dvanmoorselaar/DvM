import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from stats.nonparametric import *

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


def plotTimeCourse(times, X, color = 'blue', label = None, mask = False, mask_p_val = 0.05, errorbar = False):
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


def plotSignificanceBars(X1, X2, times, y, color, p_val = 0.05, show_descriptives = False, lw = 2, ls = '-'):
	'''

	'''

	# find significance mask
	mask = clusterMask(X1, X2, p_val)
	y_sig = np.ma.masked_where(~mask, np.ones(mask.size) * y)
	plt.plot(times, y_sig, color = color, ls = ls, lw = lw)
	if show_descriptives:
		embed()















