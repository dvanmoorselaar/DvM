import os
import mne
import sys
import glob
import pickle
import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed 
from scipy.stats import ttest_rel
sys.path.append('/home/dvmoors1/big_brother/ANALYSIS/DvM')
from support.support import *
from support.FolderStructure import FolderStructure

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class DistractorSuppressionBeh(FolderStructure):

	def __init__(self): pass

	def DTsim(self, exp = 'DT_sim',column = 'dist_high'):
		'''

		'''

		# read in data
		file = self.FolderTracker([exp,'analysis'], filename = 'preprocessed.csv')
		DF = pd.read_csv(file)

		# creat pivot (with RT filtered data)
		DF = DF.query("RT_filter == True")
		
		# create unbiased DF
		if column == 'dist_high':
			DF = DF[DF['target_high'] == 'no']
		elif column == 'target_high':
			DF = DF[DF['dist_high'] == 'no']

		pivot = DF.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type',column], aggfunc = 'mean')
		error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# plot the seperate conditions (3 X 1 plot design)
		plt.figure(figsize = (20,10))

		levels = np.unique(pivot.keys().get_level_values('block_type'))
		for idx, block in enumerate(levels):
			
			# get effect and p-value
			diff = pivot[block]['yes'].mean() -  pivot[block]['no'].mean()
			t, p = ttest_rel(pivot[block]['yes'], pivot[block]['no'])

			ax = plt.subplot(1,3, idx + 1, title = '{0}: \ndiff = {1:.0f}, p = {2:.3f}'.format(block, diff, p), ylabel = 'RT (ms)', ylim = (300,800))
			df = pd.DataFrame(np.hstack((pivot[block].values)), columns = ['RT (ms)'])
			df['sj'] = range(pivot.index.size) * 2
			df[column] = ['yes'] * pivot.index.size + ['no'] * pivot.index.size

			ax = sns.stripplot(x = column, y = 'RT (ms)', data = df, hue = 'sj', size = 10,jitter = True)
			ax.legend_.remove()
			sns.violinplot(x = column, y = 'RT (ms)', data = df, color= 'white', cut = 1)

			sns.despine(offset=50, trim = False)
		
		plt.tight_layout()
		plt.savefig(self.FolderTracker([exp,'analysis','figs'], filename = 'block_effect_{}.pdf'.format(column)))		
		plt.close()	

		# create text file with pair wise ANOVA comparisons
		output = open(self.FolderTracker([exp,'analysis','figs'], filename = 'ANOVA_{}.txt'.format(column)), 'w')
		# sim vs P
		t, p = ttest_rel(pivot[levels[0]]['yes']- pivot[levels[0]]['no'], pivot[levels[1]]['yes']- pivot[levels[1]]['no'])
		output.write('{0} vs {1} = {2:0.3f} \n'.format(levels[0],levels[1],p))
		# sim vs DP
		t, p = ttest_rel(pivot[levels[0]]['yes']- pivot[levels[0]]['no'], pivot[levels[2]]['yes']- pivot[levels[2]]['no'])
		output.write('{0} vs {1} = {2:0.3f} \n'.format(levels[0],levels[2],p))
		# P vs DP
		t, p = ttest_rel(pivot[levels[2]]['yes']- pivot[levels[2]]['no'], pivot[levels[1]]['yes']- pivot[levels[1]]['no'])
		output.write('{0} vs {1} = {2:0.3f} \n'.format(levels[2],levels[1],p))
		output.close()

	def repetitionExp1():
		'''

		'''

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
	
	os.chdir('/home/dvmoors1/big_brother/Dist_suppr')

	PO = DistractorSuppressionBeh()
	PO.DTsim(exp = 'DT_sim',column = 'dist_high')
	PO.DTsim(exp = 'DT_sim',column = 'target_high')
	PO.DTsim(exp = 'DT_sim2',column = 'dist_high')
	PO.DTsim(exp = 'DT_sim2',column = 'target_high')
	#PO.repetitionExp1()