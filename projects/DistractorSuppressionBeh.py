import os
import mne
import glob
import pickle
import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed 
from support.support import *
from support.FolderStructure import FolderStructure

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('white')
sns.set_style('white', {'axes.linewidth': 2})

class DistractorSuppressionBeh(FolderStructure):

	def __init__(self): pass

	def repetitionExp1():
		'''

		'''

		embed()

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		file = '/Users/dirk/Dropbox/projects/UVA/Suppression/repetition/analysis/preprocessed.csv'
		data = pd.read_csv(file)

		# create pivot
		data = data.query("RT_filter == True")
		pivot = data.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','set_size','repetition'], aggfunc = 'mean')
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# plot conditions
		plt.figure(figsize = (20,10))

		ax = plt.subplot(1,2, 1, title = 'Repetition effect', ylabel = 'RT (ms)', xlabel = 'repetition', ylim = (300,650), xlim = (0,11))
		for cnd in ['dist','target']:
			for load in [4,8]:
				pivot[cnd][load].mean().plot(yerr = pivot_error[cnd][load], label = '{}-{}'.format(cnd,load))
		
		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		# and plot normalized data
		norm = pivot.values
		for i,j in [(0,12),(12,24),(24,36),(36,48)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		pivot = pd.DataFrame(norm, index = np.unique(data['subject_nr']), columns = pivot.keys())
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())	

		ax = plt.subplot(1,2, 2, title = 'Normalized RT', ylabel = 'au', xlabel = 'repetition', ylim = (0.5,1), xlim = (0,11))
		for cnd in ['dist','target']:
			for load in [4,8]:

				popt, pcov = curvefitting(range(12),np.array(pivot[cnd][load].mean()),bounds=(0, [1,1])) 
				pivot[cnd][load].mean().plot(yerr = pivot_error[cnd][load], label = '{0}-{1}: alpha = {2:.2f}; delta = {3:.2f}'.format(cnd,load,popt[0],popt[1]))

		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		plt.savefig('/Users/dirk/Dropbox/projects/UVA/Suppression/repetition/analysis/figs/repetition_effect.pdf')		
		plt.close()	

if __name__ == '__main__':

	print 'what the fuck'
	
	os.chdir('/home/dvmoors1/big_brother/Dist_suppr')

	PO = DistractorSuppressionBeh()

	PO.repetitionExp1()