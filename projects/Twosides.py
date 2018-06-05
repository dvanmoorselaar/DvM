import sys
import matplotlib
matplotlib.use('agg') #
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import os
import mne
import glob
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed
#from eeg_analyses.EEG import * 
#from eeg_analyses.ERP import * 
#from eeg_analyses.BDM import BDM 
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info
sj_info = {'1': {'tracker': (False, '', ''),  'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			'2': {'tracker': (False,'', ''), 'replace':{}},
			'3': {'tracker': (False, '', ''), 'replace':{}},
			'4': {'tracker': (True, 'asc', 500), 'replace':{}},
			'5': {'tracker': (True, 'asc', 500), 'replace':{}}}

# project specific info
montage = mne.channels.read_montage(kind='biosemi64')
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
trigger = [3]
t_min = 0
t_max = 0
flt_pad = 0.5
eeg_runs = [1,2]
binary =  0
project_param = ['practice','nr_trials','trigger','condition','RT']

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class Twosides(FolderStructure):

	def __init__(self): pass

	def ctfReader(self, sj_id = 'all', channels = 'all_channels_no_eye', header = 'target_loc',ctf_name = '*_slopes_all.pickle', fband = 'all'):
		'''
		Reads in preprocessed CTF data

		sj_id (str): 
		channels (str): name of channel folder that contains ctf data
		header (str): CTF tuned to target location or distractor
		ctf_name (str): name of preprocessed ctfs
		fbanc (str): frequency band(s) of interest

		Returns
		- - - -

		ctf (dict): dictionary of ctfs as specified by ctf_name
		info (dict): EEG object used for plotting
		times (array): times shifted in time by 0.25 such that 0 ms is target display onset

		'''

		if sj_id == 'all':
			files = glob.glob(self.FolderTracker(['ctf',channels,'{}_loc'.format(header)], filename = ctf_name))
		else:
			ctf_name = '{}_' + ctf_name + '.pickle'
			files = [self.FolderTracker(['ctf',channels,'{}_loc'.format(header)], filename = ctf_name.format(sj)) for sj in sj_id]	
		ctf = []
		for file in files:
			# read in classification dict
			with open(file ,'rb') as handle:
				ctf.append(pickle.load(handle))	

		with open(self.FolderTracker(['ctf',channels, '{}_loc'.format(header)], filename = '{}_info.pickle'.format(fband)),'rb') as handle:
			info = pickle.load(handle)

		#times = info['times'] - 250	
		times =np.linspace(-400,600,269)

		return ctf, info, times

	def repetitionPlot(self, T, D, times, chance = 0, p_val = 0.05):
		'''
		Standard main plots. A 2*2 visualization of the repetition effect. Top two graphs show
		analysis tuned to the target location. Bottom two graphs show analysis tuned to distractor
		location. Variable blocks are shown in blue, target repetition is shown in green and distractor
		repetition is shown in red. 
		'''

		# initialize Permutation object
		PO = Permutation()

		# nice format for 2x2 subplots
		plt.figure(figsize = (15,10))
		plt_idx = 1
		y_lim = (-0.25,0.2)
		step = ((y_lim[1] - y_lim[0])/70.0)

		for to_plot in [T,D]:
			embed()
			# set plotting colors and legend labels
			for plot in ['V','R']:
				# initialize subplot and beatify plots
				ax = plt.subplot(2,2 , plt_idx) 
				ax.tick_params(axis = 'both', direction = 'outer')
				plt.axhline(y=chance, color = 'black')
				plt.axvline(x= -250, color = 'black') # onset placeholders
				plt.axvline(x= 0, color = 'black')	# onset gabors
				sns.despine(offset=50, trim = False)
				plt.ylim(y_lim)

				if plot == 'V':
					cnds, color = ['DvTv_0','DvTv_3'], 'blue'
				elif plot == 'R':
					if plt_idx > 2:
						cnds, color = ['DrTv_0','DrTv_3'], 'red'	
					elif plt_idx <= 2:
						cnds, color = ['DvTr_0','DvTr_3'], 'green'
	
				# loop over condititions	
				for i, cnd in enumerate(cnds):
					err, diff = bootstrap(to_plot[cnd])	
					# plot timecourse with bootstrapped error bar
					plt.plot(times, diff, label = cnd, color = color, ls = ['-','--'][i])
					plt.fill_between(times, diff + err, diff - err, alpha = 0.2, color = color)		

					# indicate significant clusters of individual timecourses
					sig_cl = PO.clusterBasedPermutation(to_plot[cnd], chance, p_val = 0.05)
					mask = np.where(sig_cl < 1)[0]
					sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
					for cl in sig_cl:
						plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * i), ls = ['-','--'][i], color = color)

				sig_cl = PO.clusterBasedPermutation(to_plot[cnds[0]], to_plot[cnds[1]], p_val = 0.01)	
				mask = np.where(sig_cl < 1)[0]
				sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
				for cl in sig_cl:
					plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * 2), color = 'black')	

				sig_cl = PO.clusterBasedPermutation(to_plot[cnds[0]] - to_plot[cnds[1]], to_plot['DvTv_0'] - to_plot['DvTv_3'], p_val = 0.05, cl_p_val = 0.01)	
				mask = np.where(sig_cl < 1)[0]
				sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
				for cl in sig_cl:
					plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * 3), color = 'grey')	
				
				plt.legend(loc = 'best')		
				# update plot counter
				plt_idx += 1	
	
		plt.tight_layout()

	def alphaSlopes(self):
		'''
		Main analysis alpha band as reported in MS. CTF slopes is contrasted using cluster-based permutation 
		in variable and repetition sequences bewteen the first and last repetition.
		'''

		# read in target repetition
		slopes, info, times = self.ctfReader(sj_id = 'all', channels = 'all_channels_no-eye', 
											header = 'target', ctf_name = 'cnds_*_slopes_alpha.pickle', fband = 'alpha')

		T = {}
		for cnd in ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']:
			T[cnd] = np.vstack([slopes[i][cnd]['T_slopes'] for i in range(len(slopes))])

		# read in dist repetition
		slopes, info, times = self.ctfReader(sj_id = 'all', channels = 'all_channels_no-eye', 
											header = 'dist', ctf_name = 'cnds_*_slopes_alpha.pickle', fband = 'alpha')
		D = {}
		for cnd in ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']:
			D[cnd] = np.vstack([slopes[i][cnd]['total'] for i in range(len(slopes))])

		self.repetitionPlot(T,D, times)	
		plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'alpha-slopes.pdf'))
		plt.close()


if __name__ == '__main__':
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/Dist_suppression'
	os.chdir(project_folder)

	# run preprocessing
	# sj = 2
	# preprocessing(sj = 5, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 			  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 			  trigger = trigger, project_param = project_param, 
	# 			  project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)

	# ERP analysis

	# BDM analysis


	# plot project analysis
	PO = Twosides()
	PO.alphaSlopes()
	


