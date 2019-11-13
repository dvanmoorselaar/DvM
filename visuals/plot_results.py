import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from stats.nonparametric import *

def contraIpsiPlotter(contra_X, ipsi_X, times, labels, colors, sig_mask = False, errorbar = False, legend_loc = 'best'):
	'''

	'''

	# loop over all waveforms
	for i, (contra, ipsi) in enumerate(zip(contra_X, ipsi_X)):
		d_wave = contra - ipsi
		err, x = bootstrap(d_wave)
		
		# do actual plotting
		if sig_mask:
			mask = clusterMask(d_wave, 0, 0.05)
			x_sig = np.ma.masked_where(~mask, x)
			x_nonsig = np.ma.masked_where(mask, x)
			plt.plot(times, x_sig, color = colors[i], ls = '--')
			plt.plot(times, x_nonsig, label = labels[i], color = colors[i])
		else:	
			plt.plot(times, x, label = labels[i], color = colors[i])

		if errorbar:
			plt.fill_between(times, x + err, x - err, alpha = 0.2, color = colors[i])

	plt.axvline(x = 0, ls = '--', color = 'black')
	plt.axhline(y = 0, ls = '--', color = 'black')			
	plt.legend(loc = legend_loc)		
	sns.despine(offset=10, trim = False)


def plotTimeCourse(times, x, color = 'blue', label = None, mask = None, errorbar = False):
	'''

	'''

	if errorbar:
		err, x = bootstrap(x)
		plt.fill_between(times, x + err, x - err, alpha = 0.2, color = color)

	if type(mask) == 'NoneType':
		plt.plot(times, x, label = label, color = color)
	else:
		x_sig = np.ma.masked_where(~mask, x)
		x_nonsig = np.ma.masked_where(mask, x)
		plt.plot(times, x_sig, color = color, ls = '--')
		plt.plot(times, x_nonsig, label = label, color = color)








