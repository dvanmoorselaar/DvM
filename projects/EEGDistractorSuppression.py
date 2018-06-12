import os
import sys
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')
import mne
import glob
import pickle
import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from stats.nonparametric import *
from eeg_analyses.helperFunctions import *
from visuals.taskdisplays import *
from visuals.visuals import MidpointNormalize
from support.support import *
from IPython import embed 
from scipy.stats import pearsonr
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from scipy.stats import ttest_rel
from eeg_analyses.FolderStructure import FolderStructure

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})


class EEGDistractorSuppression(FolderStructure):

	def __init__(self): pass


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
				print sig_cl
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
			#T[cnd] = mne.filter.resample(np.vstack([slopes[i][cnd]['E_slopes'] for i in range(len(slopes))]), down = 4)
			T[cnd] = np.vstack([slopes[i][cnd]['total'] for i in range(len(slopes))])

		# read in dist repetition
		slopes, info, times = self.ctfReader(sj_id = 'all', channels = 'all_channels_no-eye', 
											header = 'dist', ctf_name = 'cnds_*_slopes_alpha.pickle', fband = 'alpha')
		D = {}
		for cnd in ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']:
			#D[cnd] = mne.filter.resample(np.vstack([slopes[i][cnd]['E_slopes'] for i in range(len(slopes))]), down = 4)
			D[cnd] = np.vstack([slopes[i][cnd]['total'] for i in range(len(slopes))])
		
		times = np.linspace(-300, 800, 141) - 250	
		#times = mne.filter.resample(times, down = 2)
		self.repetitionPlot(T,D, times)	
		plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'alpha-slopes.pdf'))
		plt.close()

	def crossTraining(self):
		'''
		Add description if included in MS
		'''	

		slopes, info, times = self.ctfReader(sj_id = 'all', channels = 'all_channels_no-eye', 
											header = 'target', ctf_name = '*_slopes-sub_cross.pickle', fband = 'alpha')

		plt.figure(figsize = (20,10))
		for pl, cnd in enumerate(['DvTr_0','DvTr_3']):

			ax = plt.subplot(2,2 , pl + 1, title = cnd, ylabel = 'train time (ms)', xlabel = 'test time (ms)') 
			X = np.stack([np.squeeze(slopes[i][cnd]['cross']) for i in range(len(slopes))])
			#p_vals = signedRankArray(X, 0)
			X = np.mean(X, axis = 0)
			#X[p_vals > 0.05] = 0

			plt.imshow(X, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], vmin = -0.3, vmax = 0.3)
			plt.colorbar()

		slopes, info, times = self.ctfReader(sj_id = 'all', channels = 'all_channels_no-eye', 
											header = 'dist', ctf_name = '*_slopes-sub_cross.pickle', fband = 'alpha')
 	
		for pl, cnd in enumerate(['DrTv_0','DrTv_3']):

			ax = plt.subplot(2,2 , pl + 3, title = cnd, ylabel = 'train time (ms)', xlabel = 'test time (ms)') 
			X = np.stack([np.squeeze(slopes[i][cnd]['cross']) for i in range(len(slopes))])
			#p_vals = signedRankArray(X, 0)
			X = np.mean(X, axis = 0)
			#X[p_vals > 0.05] = 0
			plt.imshow(X, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], vmin = -0.3, vmax = 0.3)
			plt.colorbar()

		plt.tight_layout()

		plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'cross-training.pdf'))
		plt.close()

	def conditionCheck(self, window = (-0.3,0.8), thresh_bin = 1):
		'''
		Checks the mimimum number of conditions after preprocessing

		'''

		text = ''
		for sj in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:

			# read in beh
			with open(self.FolderTracker(extension = ['beh','processed'], filename = 'subject-{}_all.pickle'.format(sj)),'rb') as handle:
				beh = pickle.load(handle)

			# read in eeg data 
			EEG = mne.read_epochs(self.FolderTracker(extension = ['processed'], filename = 'subject-{}_all-epo.fif'.format(sj)))

			# exclude trials contaminated by unstable eye position
			s, e = [np.argmin(abs(EEG.times - t)) for t in window]
			nan_idx = np.where(np.isnan(beh['eye_bins'])> 0)[0]
			heog = EEG._data[:,EEG.ch_names.index('HEOG'),s:e]

			eye_trials = eog_filt(beh, EEG, heog, sfreq = EEG.info['sfreq'], windowsize = 50, windowstep = 25, threshold = 30)
			beh['eye_bins'][eye_trials] = 99
	
			# use mask to select conditions and position bins (flip array for nans)
			eye_mask = ~(beh['eye_bins'] > thresh_bin)	
			
			# select conditions
			cnds = beh['condition'][eye_mask]

			min_cnd, cnd = min([sum(cnds == c) for c in np.unique(cnds)]), np.unique(cnds)[
            np.argmin([sum(cnds== c) for c in np.unique(cnds)])]

			text += 'sj {}, min cnd is {} ({} trials, {}%) \n'.format(sj, cnd, min_cnd, min_cnd/612.0)

		with open('eye-{}.txt'.format(thresh_bin), 'a') as the_file:
			the_file.write(text)

	def inspectTimeCourse(self, plotting_data, times, y_lim = (-7,5), chance = 0, file = ''):
		'''
		Creates general plotting structure. Left plot repetition 1 and 4 in variable condition.
		Middle plot repetition 1 and 4 in repeat condition. Right plot effect of repetition.
		Uses unique colors for variable and repetition blocks. Significant parts of the line are 
		set to black.
		'''

		# initialize Permutation object
		PO = Permutation()
	
		# nice format for 2x2 subplots
		plt.figure(figsize = (15,10)) 

		step = ((y_lim[1] - y_lim[0])/70.0)

		# loop over variable and repetition plots and D and T blocks
		T_jasp, D_jasp = np.zeros((24,8)), np.zeros((24,8))
		all_data = {'T':{},'D':{}}
		plt_idx = 1
		for analysis in ['T', 'D']:
			# initialize subplot
			
			for plot in ['Var','Rep']:
				ax = plt.subplot(2,2 , plt_idx) 
				
				# beautify plots
				ax.tick_params(axis = 'both', direction = 'outer')
				plt.axhline(y=chance, color = 'grey')
				plt.axvline(x=-0.25, color = 'grey') # onset placeholders
				plt.axvline(x=0, color = 'grey')	# onset gabors
				plt.ylim(y_lim)

				# set plotting colors and legend labels
				if plot == 'Var':
					cnds, color = ['DvTv_0','DvTv_3'], 'blue'
				elif plot == 'Rep':
					if analysis == 'D':
						cnds, color = ['DrTv_0','DrTv_3'], 'red'	
					elif analysis == 'T':
						cnds, color = ['DvTr_0','DvTr_3'], 'green'	

				# loop over condititions	
				for i, cnd in enumerate(cnds):
					all_data[analysis][cnd] = plotting_data[analysis][cnd]
					err, diff = bootstrap(plotting_data[analysis][cnd])

					# plot timecourse with bootstrapped error bar
					plt.plot(times, diff, label = '{}'.format(cnd), color = color, ls = ['-','--'][i])
					plt.fill_between(times, diff + err, diff - err, alpha = 0.2, color = color)

					# indicate significant clusters of individual timecourses
					sig_cl = PO.clusterBasedPermutation(plotting_data[analysis][cnd], chance)
					mask = np.where(sig_cl < 1)[0]
					sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
					for cl in sig_cl:
						plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * i), ls = ['-','--'][i], color = color)
						file.write('{} vs {}: window = {} - {}'.format(cnd, chance, times[cl[0]],times[cl[-1]]))

				# plot the repetition effect
				sig_cl = PO.clusterBasedPermutation(all_data[analysis][cnds[1]], all_data[analysis][cnds[0]])	
				mask = np.where(sig_cl < 1)[0]
				sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
				for cl in sig_cl:
					plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * 2), color = 'black')
					if cl.size != 0:
						file.write('repetition effect {}: window = {} - {} \n'.format(analysis, times[cl[0]],times[cl[-1]]))						

				plt_idx += 1		
				plt.legend(loc = 'best')
				sns.despine(offset=50, trim = False)

			# plot the baseline effect (rep 3 - rep 0 vs baseline)
			sig_cl = PO.clusterBasedPermutation(all_data[analysis][cnds[1]] - all_data[analysis][cnds[0]], all_data[analysis]['DvTv_3'] - all_data[analysis]['DvTv_0'])	
			mask = np.where(sig_cl < 1)[0]
			sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
			for cl in sig_cl:
				plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * 3), color = 'grey')
				if cl.size != 0:
					file.write('repetition effect baseline {}: window = {} - {} \n'.format(analysis, times[cl[0]],times[cl[-1]]))

		# plot the analysis effect
		sig_cl = PO.clusterBasedPermutation(all_data['T']['DvTr_0'],  all_data['D']['DrTv_0'])	
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * 4), color = 'yellow')
			if cl.size != 0:
				file.write('D vs T rep 1 window = {} - {} \n'.format(times[cl[0]],times[cl[-1]]))

		sig_cl = PO.clusterBasedPermutation(all_data['T']['DvTr_3'],  all_data['D']['DrTv_3'])	
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * 5), color = 'purple')
			if cl.size != 0:
				file.write('D vs T rep 4 window = {} - {} \n'.format(times[cl[0]],times[cl[-1]]))

		sig_cl = PO.clusterBasedPermutation(all_data['T']['DvTr_3'] - all_data['T']['DvTr_0'],  all_data['D']['DrTv_3'] - all_data['D']['DrTv_0'])	
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size) * (y_lim[0] + step * 6), color = 'brown')
			if cl.size != 0:
				file.write('D vs T repetition effect window = {} - {} \n'.format(times[cl[0]],times[cl[-1]]))
		
		file.close()
		plt.tight_layout()

	def inspectTimeCourseSEP(self, header, plotting_data, times, y_lim = (-7,5), chance = 0, analysis = ''):
		'''
		Creates general plotting structure. Left plot repetition 1 and 4 in variable condition.
		Middle plot repetition 1 and 4 in repeat condition. Right plot effect of repetition.
		Uses unique colors for variable and repetition blocks. Significant parts of the line are 
		set to black.
		'''

		# initialize Permutation object
		PO = Permutation()

		# get height permutation bar
		h = (abs(y_lim[0] - y_lim[1]))/40.0/2
	
		if header == 'target':
			rep_cnds = ['DvTr_0','DvTr_3']
			rep_color = 'green'
		elif header == 'dist':
			rep_cnds = ['DrTv_0','DrTv_3']
			rep_color = 'red'
	
		# loop over variable and repetition plots
		all_data = []
		for idx,  plot in enumerate(['Var','Rep']):

			# initialize plot
			#ax = plt.subplot(1,3 , idx + 1 ) #, title = plot, ylabel = 'mV')
			plt.figure(figsize = (15,5)) 
			#ax = plt.subplot(1,1 , 1, xlabel = 'Time (ms)') #, title = plot, ylabel = 'mV')
			# beautify plots
			plt.tick_params(axis = 'both', direction = 'out')
			plt.xlabel('Time (ms)')
			plt.axhline(y=chance, ls = '--', color = 'grey')
			plt.axvline(x=-0.25, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')	
			plt.ylim(y_lim)
			plt.xlim((times[0],times[-1]))

			# set plotting colors and legend labels
			if plot == 'Var':
				cnds, color = ['DvTv_0','DvTv_3'], 'blue'
			elif plot == 'Rep':
				cnds, color = rep_cnds, rep_color	

			# loop over condititions	
			for i, cnd in enumerate(cnds):
				all_data.append(plotting_data[cnd])
				err, diff = bootstrap(all_data[-1])

				# plot timecourse with bootstrapped error bar
				plt.plot(times, diff, label = '{}'.format(cnd), color = color, ls = ['-','--'][i])
				plt.fill_between(times, diff + err, diff - err, alpha = 0.2, color = color)

				# change parts of line that are significant
				sig_cl = PO.clusterBasedPermutation(all_data[-1], chance)
				mask = np.where(sig_cl < 1)[0]
				sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
				for cl in sig_cl:
					plt.plot(times[cl], np.ones(cl.size) * (0.08 + 0.01 * i), ls = ['-','--'][i], color = color)
					
			# add markers for significant clusters (condition repetition effect)
			sig_cl = PO.clusterBasedPermutation(all_data[-2], all_data[-1])	
			plt.fill_between(times, chance - 0.25, chance + 0.25, where = sig_cl < 1, color = 'black', label = 'p < 0.05')
			plt.legend(loc = 'best')
			sns.despine(offset=50, trim = False)
			plt.tight_layout()
			plt.savefig(self.FolderTracker([analysis,'MS-plots'], filename = '{}-{}.pdf'.format(header,plot)))

		# Compare variable and repetition plots (rep 1 - rep 4)	
		plt.figure(figsize = (15,5)) 
		ax = plt.subplot(1,1 ,1) #, title = 'Difference', ylabel = 'mV')
		
		# beatify plot
		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=-0.25, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')

		var = all_data[0] - all_data[1]
		rep = all_data[2] - all_data[3]

		for i, signal in enumerate([var, rep]):
			err, diff = bootstrap(signal)
			plt.plot(times, diff, label = ['var','rep'][i], color = ['blue', rep_color][i])
			plt.fill_between(times, diff + err, diff - err, alpha = 0.2, color =['blue', rep_color][i])

		sig_cl = PO.clusterBasedPermutation(var, rep)

		# set new height permutation bar
		y_lim = ax.get_ylim()
		h = (abs(y_lim[0] - y_lim[1]))/40.0/2
		plt.xlim((times[0],times[-1]))
		plt.fill_between(times, -h, h, where = sig_cl < 1, color = 'black', label = 'p < 0.05')
		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)
		plt.tight_layout()
		plt.savefig(self.FolderTracker([analysis,'MS-plots'], filename = '{}-plotdiff.pdf'.format(header)))

	

	### DT comparison
	def DT(self):

		# initialize Permutation object
		PO = Permutation()
		embed()
		# read in target slopes
		slopes_t, info, times = self.ctfReader(sj_id = 'all',channels = 'all_channels_no-eye', 
												header = 'target', cnd_name = 'cnds', ctf_name = 'slopes_alpha')

		slopes_d, info, times = self.ctfReader(sj_id = 'all',channels = 'all_channels_no-eye', 
												header = 'dist', cnd_name = 'cnds', ctf_name = 'slopes_alpha')

		power = 'total'

		t_effect = np.vstack([slopes_t[i]['DvTr_3'][power] for i in range(len(slopes_t))]) #- np.vstack([slopes_t[i]['DvTr_3'][power] for i in range(len(slopes_t))])
		d_effect = np.vstack([slopes_d[i]['DrTv_3'][power] for i in range(len(slopes_d))]) #- np.vstack([slopes_d[i]['DrTv_3'][power] for i in range(len(slopes_d))])

		sig_cl = PO.clusterBasedPermutation(t_effect, d_effect)	
		plt.figure(figsize = (15,5)) 
		plt.tick_params(axis = 'both', direction = 'out')
		plt.xlabel('Time (ms)')
		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=-0.25, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')	
		plt.xlim((times[0],times[-1]))

		plt.plot(times, t_effect.mean(0), label = 't_effect')
		err, diff = bootstrap(t_effect)
		plt.fill_between(times, diff + err, diff - err, alpha = 0.2)

		plt.plot(times, d_effect.mean(0), label = 'd_effect')
		err, diff = bootstrap(d_effect)
		plt.fill_between(times, diff + err, diff - err, alpha = 0.2)
		plt.fill_between(times, -0.05, 0.05, where = sig_cl < 1, color = 'black', label = 'p < 0.05')

		plt.savefig(self.FolderTracker(['poster', 'ctf'], filename = 'DTdiff.pdf'))
		plt.close()

		# read in decoding
		files = glob.glob(self.FolderTracker(['bdm', 'target_loc'], filename = 'class_*_perm-False.pickle'))
		bdmT = []
		for file in files:
			with open(file ,'rb') as handle:
				bdmT.append(pickle.load(handle))

		files = glob.glob(self.FolderTracker(['bdm', 'dist_loc'], filename = 'class_*_perm-False.pickle'))
		bdmD = []
		for file in files:
			with open(file ,'rb') as handle:
				bdmD.append(pickle.load(handle))

		T = np.stack([np.diag(bdmT[j]['DvTr_3']['standard']) for j in range(len(bdmT))])
		D = np.stack([np.diag(bdmD[j]['DrTv_3']['standard']) for j in range(len(bdmD))])

		sig_cl = PO.clusterBasedPermutation(D, T)	

		# read in ERP
		erps_T, info, times = self.erpReader('target', 'lat-down1')
		erps_D, info, times = self.erpReader('target', 'lat-down1')
		elecs = ['PO7','PO3','O1']
		e_idx = np.array([erps_T[erps_T.keys()[0]]['all']['elec'][0].index(e) for e in elecs])

		ipsi = np.vstack([erps_T[str(key)]['DvTr_3']['ipsi'][e_idx].mean(0) for key in erps_T.keys()])
		contra = np.vstack([erps_T[str(key)]['DvTr_3']['contra'][e_idx].mean(0) for key in erps_T.keys()])
		T = contra - ipsi

		ipsi = np.vstack([erps_D[str(key)]['DrTv_3']['ipsi'][e_idx].mean(0) for key in erps_D.keys()])
		contra = np.vstack([erps_D[str(key)]['DrTv_3']['contra'][e_idx].mean(0) for key in erps_D.keys()])
		D = contra - ipsi

		sig_cl = PO.clusterBasedPermutation(D, T)
		plt.figure(figsize = (15,5)) 
		#ax = plt.subplot(1,1 , 1, xlabel = 'Time (ms)') #, title = plot, ylabel = 'mV')
		# beautify plots
		plt.tick_params(axis = 'both', direction = 'out')
		plt.xlabel('Time (ms)')
		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=-0.25, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')	
		plt.xlim((times[0],times[-1]))
		plt.fill_between(times, -1, -2, where = sig_cl < 1, color = 'black')

		plt.savefig(self.FolderTracker(['poster', 'erp'], filename = 'DTdiff.pdf'))
		plt.close()

	def diff_ERPS(self, elecs, erp_name, y_lim = (-5,2)):
		'''
		plots ERP difference waves (contra - ipsi) for the repeat and the variable sequences seperately.
		Calls inspectTimeCourse to visualize the condition comparisons and plot the significant clusters.

		Arguments
		- - - - - 

		elecs (list): list of electrodes used for ERP's
		header (str): ERP tuned to target location or distractor
		erp_name (str): name of preprocessed erps
		'''

		PO = Permutation()

		# open file to store timing of significant clusters
		f = open(self.FolderTracker(['erp','MS-plots'], filename = 'main_erp-{}.txt'.format(erp_name)),'w')

		# read in data and shift timing
		T_erps, info, times = self.erpReader('target', erp_name)
		D_erps, info, times = self.erpReader('dist', erp_name)

		# get indices of electrodes of interest
		e_idx = np.array([T_erps[T_erps.keys()[0]]['all']['elec'][0].index(e) for e in elecs])

		# plot difference wave form collapsed across all conditions
		ipsi_T = np.mean(np.stack(([[T_erps[str(key)][cnd]['ipsi'][e_idx].mean(0) for key in T_erps.keys()] for cnd in ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']])),0)
		contra_T =np.mean(np.stack(([[T_erps[str(key)][cnd]['contra'][e_idx].mean(0) for key in T_erps.keys()] for cnd in ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']])),0)
		ipsi_D = np.mean(np.stack(([[D_erps[str(key)][cnd]['ipsi'][e_idx].mean(0) for key in D_erps.keys()] for cnd in ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']])),0)
		contra_D =np.mean(np.stack(([[D_erps[str(key)][cnd]['contra'][e_idx].mean(0) for key in D_erps.keys()] for cnd in ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']])),0)

		plt.figure(figsize = (15,5)) 
		for i, effect in enumerate([contra_T - ipsi_T, contra_D - ipsi_D]):

			sig_cl = PO.clusterBasedPermutation(effect, 0)
			ax = plt.subplot(1,2 ,i + 1) 
				
			# beautify plots
			ax.tick_params(axis = 'both', direction = 'outer')
			plt.axhline(y=0, color = 'grey')
			plt.axvline(x=-0.25, color = 'grey') # onset placeholders
			plt.axvline(x=0, color = 'grey')	# onset gabors
			plt.ylim(y_lim)
			err, diff = bootstrap(effect)
			plt.plot(times, diff, color = ['green', 'red'][i])
			plt.fill_between(times, diff + err, diff - err, alpha = 0.2, color = ['green', 'red'][i])
			mask = np.where(sig_cl < 1)[0]
			sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
			for j, cl in enumerate(sig_cl):
				if i == 0:
					self.ERPJASP(T_erps, cl, ['DvTv_0','DvTv_3','DvTr_0','DvTr_3'], e_idx, 'target', j, nr_sj = 24)
				elif i == 1:
					self.ERPJASP(D_erps, cl, ['DvTv_0','DvTv_3','DrTv_0','DrTv_3'], e_idx, 'dist', j, nr_sj = 24)	
				plt.plot(times[cl], np.ones(cl.size) * (y_lim[0]), color = ['green', 'red'][i])

		sns.despine(offset=50, trim = False)
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'main_allerp-{}.pdf'.format(erp_name)))
		plt.close()	

		# get plotting data
		plotting_data = {'T':{},'D':{}}
		for cnd in ['DvTv_0','DvTv_3','DrTv_0','DrTv_3','DvTr_0','DvTr_3']:
				if e_idx.size > 1:
					ipsi_T = np.vstack([T_erps[str(key)][cnd]['ipsi'][e_idx].mean(0) for key in T_erps.keys()])
					contra_T = np.vstack([T_erps[str(key)][cnd]['contra'][e_idx].mean(0) for key in T_erps.keys()])
					ipsi_D = np.vstack([D_erps[str(key)][cnd]['ipsi'][e_idx].mean(0) for key in D_erps.keys()])
					contra_D = np.vstack([D_erps[str(key)][cnd]['contra'][e_idx].mean(0) for key in D_erps.keys()])
				else:
					ipsi_T = np.vstack([T_erps[str(key)][cnd]['ipsi'][e_idx] for key in T_erps.keys()])
					contra_T = np.vstack([T_erps[str(key)][cnd]['contra'][e_idx] for key in T_erps.keys()])
					ipsi_D = np.vstack([D_erps[str(key)][cnd]['ipsi'][e_idx] for key in D_erps.keys()])
					contra_D = np.vstack([D_erps[str(key)][cnd]['contra'][e_idx] for key in D_erps.keys()])

				plotting_data['T'][cnd] = contra_T - ipsi_T
				plotting_data['D'][cnd] = contra_D - ipsi_D

		self.inspectTimeCourse(plotting_data, times, y_lim = y_lim, chance = 0, file = f)
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'main_erp-{}.pdf'.format(erp_name)))
		plt.close()	

	def ERPJASP(self, erps, cluster, cnds, e_idx, header, clust_nr, nr_sj = 24):
		'''
		Select contra and ipsi waveforms for JASP analysis
		'''

		JASP = np.zeros((nr_sj, len(cnds)*2))
		for i, cnd in enumerate(cnds):
			ipsi = np.vstack([erps[str(key)][cnd]['ipsi'][e_idx, cluster[0]:cluster[-1]].mean() for key in erps.keys()])
			contra = np.vstack([erps[str(key)][cnd]['contra'][e_idx, cluster[0]:cluster[-1]].mean() for key in erps.keys()])

			JASP[:,i*2] = ipsi.T
			JASP[:, i + (i + 1)] = contra.T

		headers = ['_'.join(np.array(labels,str)) for labels in product(*[cnds,['ipsi','contra']])]	
		np.savetxt(self.FolderTracker(['erp','MS-plots'], filename = '{}_cl{}-JASP.csv'.format(header, clust_nr)), JASP, delimiter = "," ,header = ",".join(headers), comments='')

	def bdmdiag(self):
		'''

		'''

		# read in data
		with open(self.FolderTracker(['bdm','{}_loc'.format('target')], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		times = info['times'] - 0.25


		files = glob.glob(self.FolderTracker(['bdm', '{}_loc'.format('target')], filename = 'class_*_perm-False.pickle'))
		bdm_T = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm_T.append(pickle.load(handle))

		files = glob.glob(self.FolderTracker(['bdm', '{}_loc'.format('dist')], filename = 'class_*_perm-False.pickle'))
		bdm_D = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm_D.append(pickle.load(handle))		

		# get plotting data
		plotting_data = {'T':{},'D':{}}
		for cnd in ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']:
				plotting_data['D'][cnd] = np.stack([np.diag(bdm_D[j][cnd]['standard']) for j in range(len(bdm_D))])

		for cnd in ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']:
				plotting_data['T'][cnd] = np.stack([np.diag(bdm_T[j][cnd]['standard']) for j in range(len(bdm_T))])

		# open file to store timing of significant clusters
		f = open(self.FolderTracker(['bdm','MS-plots'], filename = 'diag-bdm.txt'),'w')
		self.inspectTimeCourse(plotting_data, times, y_lim = (0.1, 0.3), chance = 1/6.0, file = f)
		plt.savefig(self.FolderTracker(['bdm','MS-plots'], filename = 'diag-bdm.pdf'))
		plt.close()		



	def bdmACC(self, header):
		'''

		'''

		PO = Permutation()

		if header == 'target':
			conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
		elif header == 'dist':
			conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']	

		# read in data
		with open(self.FolderTracker(['bdm','{}_loc'.format(header)], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		times = info['times'] - 0.25

		files = glob.glob(self.FolderTracker(['bdm', '{}_loc'.format(header)], filename = 'class_*_perm-False.pickle'))
		bdm = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm.append(pickle.load(handle))			
		
		#plt.figure(figsize = (20,10))
		perm = []
		plt_idx = [1,2,4,5]
		X2 = 1/6.0
		# normalize colorbar 
		norm = MidpointNormalize(midpoint=X2)
		data = []
		for i, cnd in enumerate(conditions):

			plt.figure(figsize = (10,10))
			#ax = plt.subplot(2,3 , plt_idx[i])#, title = cnd, ylabel = 'Time (ms)', xlabel = 'Time (ms)')
			#ax.tick_params(direction = 'in', length = 5)
			plt.tick_params(direction = 'in', length = 5)

			X1 = np.stack([bdm[j][cnd]['standard'] for j in range(len(bdm))])[:, times >= 0, :][:,:,times >= 0]
			data.append(X1)
			p_vals = signedRankArray(X1, X2)
			h,_,_,_ = FDR(p_vals)
			
			dec = np.mean(X1,0)
			dec[~h] = X2

			plt.imshow(dec, norm = norm, cmap = cm.bwr, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[0,times[-1],0,times[-1]], vmin = 0.1, vmax = 0.3)
		
			plt.colorbar()
			plt.savefig(self.FolderTracker(['poster', 'bdm'], filename = 'bdm_{}-plot{}.pdf'.format(header,cnd)))
			plt.close()

		plt_idx = [3,6]
		for i, cnd in enumerate(['var','rep']):
			plt.figure(figsize = (10,10))

			#ax = plt.subplot(2,3 , plt_idx[i])#, title = cnd, ylabel = 'Time (ms)', xlabel = 'Time (ms)')
			#ax.tick_params(direction = 'in', length = 0.5)
			plt.tick_params(direction = 'in', length = 0.5)
			if i == 0:
				X, Y = data[0], data[1]
			else:
				X, Y = data[2], data[3]

			sig_cl = PO.clusterBasedPermutation(X,Y)

			x = times[times > 0]
			X = np.tile(x,(x.size,1)).T
			Y = np.tile(x,(len(x),1))
			Z = sig_cl.T
			plt.contour(X,Y,Z,1)

			#plt.imshow(sig_cl, interpolation='none', aspect='auto', 
			#		origin = 'lower', extent=[0,times[-1],0,times[-1]]) 
			#plt.coloribar()

			#plt.colorbar()
			plt.savefig(self.FolderTracker(['poster', 'bdm'], filename = 'bdm_{}-diffplot{}.pdf'.format(header,cnd)))
			plt.close()

		# plot repetition effect
		plt.figure(figsize = (10,10))	
		plt.tick_params(direction = 'in', length = 0.5)	
		sig_cl = PO.clusterBasedPermutation(data[0] - data[1],data[2] - data[3])
		x = times[times > 0]
		X = np.tile(x,(x.size,1)).T
		Y = np.tile(x,(len(x),1))
		Z = sig_cl.T
		plt.contour(X,Y,Z,1)

		plt.savefig(self.FolderTracker(['poster', 'bdm'], filename = 'bdm_{}-diffplot-rep3.pdf'.format(header)))
		plt.close()

		#plt.tight_layout()
		#plt.savefig(self.FolderTracker(['bdm','MS-plots'], filename = 'class_{}.pdf'.format(header)))		
		#plt.close()




	### EXTRA ANALYSIS
	def cndTOPO(self, header, topo_name = 'topo_lat-down1', start = -0.1, stop = 0.15, step = 0.01):
		'''

		'''

		# get conditions of interest
		if header == 'target':
			cnds = ['DvTv_0', 'DvTv_3','DvTr_0','DvTr_3']
		elif header == 'dist':
			cnds = ['DvTv_0', 'DvTv_3','DrTv_0','DrTv_3']

		# define segments
		segments = np.arange(start, stop, step)	

		# read in data and shift timing
		topo, info, times = self.erpReader(header, topo_name)

		# create figure
		plt.figure(figsize = (50,20))

		# loop over conditions
		idx_cntr = 1
		for cnd in cnds:

			# loop over time segments
			for start_seg in segments:

				# select time window of interest
				s, e = [np.argmin(abs(times - t)) for t in (start_seg,start_seg+0.01)]

				# extract mean TOPO for window of interest
				T = np.mean(np.stack(
					[topo[j][cnd][:,s:e] for j in topo.keys()], 
					axis = 0), axis = (0,2))

				if cnd == 'DvTv_0':
					ax = plt.subplot(len(cnds), segments.size ,idx_cntr, title = '{0:.2f}'.format(start_seg))
				else:
					ax = plt.subplot(len(cnds), segments.size ,idx_cntr)

				im = mne.viz.plot_topomap(T, info['info'], names = info['ch_names'],
				show_names = False, show = False, axes = ax, cmap = cm.jet, vmin = -7,vmax = 5)

				idx_cntr += 1

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'cnd_topos_{}.pdf'.format(header)))
		plt.close()	

		# create figure
		

		# loop over conditions
		
		
		for block in ['var','rep']:
			idx_cntr = 1
			plt.figure(figsize = (40,10))
			# loop over time segments
			for start_seg in segments:


				# select time window of interest
				s, e = [np.argmin(abs(times - t)) for t in (start_seg,start_seg+0.01)]

				# extract mean TOPO for window of interest
				if block == 'var':
					ax = plt.subplot(1, segments.size ,idx_cntr, title = '{0:.2f}'.format(start_seg))
					cnd = cnds[:2]
				elif block == 'rep':
					ax = plt.subplot(1, segments.size ,idx_cntr)
					cnd = cnds[-2:]

				T = np.mean(np.stack(
					[topo[j][cnd[0]][:,s:e] for j in topo.keys()], 
					axis = 0), axis = (0,2)) - np.mean(np.stack(
					[topo[j][cnd[1]][:,s:e] for j in topo.keys()], 
					axis = 0), axis = (0,2))

				print T.min(), T.max()

				im = mne.viz.plot_topomap(T, info['info'], names = info['ch_names'],
				show_names = False, show = False, axes = ax, cmap = cm.jet, vmin = -2,vmax = 2)
				idx_cntr += 1

			plt.tight_layout()
			plt.savefig(self.FolderTracker(['poster','erp'], filename = 'cnd-diff_{}_{}.pdf'.format(header,block)))
			plt.close()	

		
	### CTF PLOTS
	def plotCTF(self, header = 'target', cnd_name = 'cnds', ctf_name = 'ctf_alpha'):
		'''

		'''

		ctfs, info, times = self.ctfReader(sj_id = 'all', channels = 'all_channels_no-eye', header = header, cnd_name = cnd_name, ctf_name = ctf_name)
		
		power = 'total'
		if header == 'target':
			cnds = ['DvTv_0', 'DvTv_3','DvTr_0','DvTr_3']
		elif header == 'dist':
			cnds = ['DvTv_0', 'DvTv_3','DrTv_0','DrTv_3']

		#norm = MidpointNormalize(midpoint=0)
		for cnd in cnds:

			plt.figure(figsize = (40,15))
			plt.tick_params(direction = 'in', length = 5)
			xy = np.squeeze(np.stack([np.mean(np.mean(ctfs[i][cnd]['ctf']['total'],1),2) for i in range(len(ctfs))]))
			xy = np.mean(xy,0)
			xy = np.hstack((xy,xy[:,0].reshape(-1,1))).T
			plt.imshow(xy, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],-180,180], vmin = 0.0, vmax = 0.45) 

			plt.colorbar()
			plt.savefig(self.FolderTracker(['poster', 'ctf'], filename = 'ctf_{}_{}.pdf'.format(header,cnd)))
			plt.close()



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
			print(file)
			# resad in classification dict
			with open(file ,'rb') as handle:
				ctf.append(pickle.load(handle))	

		with open(self.FolderTracker(['ctf',channels, '{}_loc'.format(header)], filename = '{}_info.pickle'.format(fband)),'rb') as handle:
			info = pickle.load(handle)

		times = info['times'] - 250	

		return ctf, info, times

	def erpReader(self, header, erp_name):
		'''
		Reads in preprocessed EEG data

		Arguments
		- - - - - 

		header (str): ERP tuned to target location or distractor
		erp_name (str): name of preprocessed erps

		Returns
		- - - -

		erp (dict): dictionary of erps as specified by erp_name
		info (dict): EEG object used for plotting
		times (array): times shifted in time by 0.25 such that 0 ms is target display onset
		'''

		# read in data and shift timing
		with open(self.FolderTracker(['erp','{}_loc'.format(header)], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','{}_loc'.format(header)], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		return erp, info, times

	def rawCTF(self, header, ctf_name):
		'''

		'''

		if header == 'dist_loc':
			conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
		elif header == 'target_loc':
			conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']

		# read in data	
		raw, info, times = self.ctfReader(sj_id = 'all'
			,channels = 'all_channels_no-eye', header = header, ctf_name = ctf_name)

		s, e = [np.argmin(abs(times - t)) for t in (150,200)]

		# read in info for topoplot
		with open(self.FolderTracker(['erp',header], filename = 'plot_dict.pickle') ,'rb') as handle:
			info_viz = pickle.load(handle)
		

		for i, cnd in enumerate(conditions):

			plt.figure(figsize = (30,10))
			ax =  plt.subplot(2,1, 1, title = cnd, ylabel = 'channels', xlabel = 'time (ms)')
			ctf = np.mean(np.dstack([np.mean(raw[sj][cnd]['ctf']['raw_eeg'], axis = (0,2)) for sj in range(len(raw))]),2)
			ctf = np.vstack((ctf.T,ctf[:,0]))

			plt.imshow(ctf, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],1,6])

			for loc in range(6):
				ax =  plt.subplot(2,6, loc + 7, title = str(loc))
				w = np .mean(np.array([np.mean(raw[sj][cnd]['W']['raw_eeg'], axis = (0,2)) for sj in range(len(raw))]),0)	

				im = mne.viz.plot_topomap(np.mean(w[s:e,loc,:],0), info_viz['info'], names = info_viz['ch_names'],
				show_names = False, show = False, axes = ax, cmap = cm.jet, vmin = -4.5, vmax = 4.5 )

			sns.despine(offset=50, trim = False)
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'raw-{}-{}.pdf'.format(header, cnd)))
			plt.close()


	def timeFreqCTF(self, header, cnd_name, perm = True, p_map = False):
		'''

		'''

		PO = Permutation()

		slopes, info, times = self.ctfReader(sj_id = 'all',
			 channels = 'all_channels_no-eye', header = header, cnd_name = cnd_name, ctf_name = 'slopes_all', fband = 'all')

		#if perm:
		#	slopes, info, times = self.ctfReader(sj_id = [2,5,6,7,10,13,14,15,18,19,22,23,24],
		#	 channels = 'all_channels_no-eye', header = header, ctf_name = 'slopes_perm_all', fband = 'all')
			
		freqs = (info['freqs'].min(), info['freqs'].max())

		if header == 'dist':
			conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
		elif header == 'target':
			conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']

		if cnd_name == 'all':
			plt.figure(figsize = (20,15))
			plt.tick_params(direction = 'in', length = 5)
			xy = np.stack([slopes[j]['all']['total'] for j in range(len(slopes))])
			p_vals = signedRankArray(xy, 0)
			h,_,_,_ = FDR(p_vals)
			XY = np.mean(xy,axis = 0)
			XY[~h] = 0

			plt.imshow(XY, cmap = cm.jet, interpolation='none', aspect='auto', 
				origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[-1]], vmin = 0, vmax = 0.20)
			plt.colorbar()
			plt.savefig(self.FolderTracker(['poster', 'ctf'], filename = '{}-all-freqs.pdf'.format(header)))
			plt.close()
		else:	

			for power in ['evoked', 'total']:

				crange = (-0.15,0.15)	
				repeat = []
				variable = []
				plt.figure(figsize = (20,15))
				data = []

				plt_idx = [1,2,4,5]
				for i, cnd in enumerate(conditions):
					ax =  plt.subplot(2,3, plt_idx[i], title = cnd, ylabel = 'freqs', xlabel = 'time (ms)')
					xy = np.stack([slopes[j][cnd][power] for j in range(len(slopes))])[:,:6,:]
					data.append(xy)
					p_vals = signedRankArray(xy, 0)
					
					h,_,_,_ = FDR(p_vals)
					XY = np.mean(xy,axis = 0)
					XY[~h] = 0

					if 'r' in cnd:
						repeat.append(xy)
					else:
						variable.append(xy)	


					plt.imshow(XY, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],freqs[0],18], vmin = crange[0], vmax = crange[1])

					plt.axvline(x=-250, ls = '--', color = 'white')
					plt.axvline(x=0, ls = '--', color = 'white')
					plt.colorbar(ticks = (crange[0],crange[1]))

				plt_idx = [3,6]
				for i, cnd in enumerate(['var','rep']):

					ax = plt.subplot(2,3 , plt_idx[i], title = cnd, ylabel = 'freqs', xlabel = 'time (ms)')
					if i == 0:
						X, Y = data[0], data[1]
					else:
						X, Y = data[2], data[3]

					sig_cl = PO.clusterBasedPermutation(X, Y)	

					plt.imshow(sig_cl, cmap = cm.jet, interpolation='none', aspect='auto', 
							origin = 'lower', extent=[times[0],times[-1],freqs[0],18]) 

					plt.colorbar()

				plt.tight_layout()
				if perm:
					if p_map:
						plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'tf-p_map_{}_{}.pdf'.format(header, power)))
					else:	
						plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'tf_{}_{}.pdf'.format(header, power)))
				else:
					plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'tf_{}_{}.pdf'.format(header, power)))	
				plt.close()	


	def threeDSlopes(self, header):
		'''

		'''

		if header == 'target_loc':
			cnds  = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
		elif header == 'dist_loc':
			cnds  = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']

		# read in data
		ctfs, info, times = self.ctfReader(sj_id = 'all',
			 channels = 'all_channels_no-eye', header = header, ctf_name = 'ctf_all', fband = 'all')

		# get X (time),Y (channel), Z data (channel response)
		X = info['times'][::info['downsample']]
		Y = np.arange(7)
		X, Y = np.meshgrid(X, Y)

		power = 'total'
		
		for fr, band in enumerate(info['freqs']):
			if band[1] <= 14:
				f = plt.figure(figsize = (20,15))
				for i, cnd in enumerate(cnds):

					ax = f.add_subplot(2, 2, i + 1, projection='3d', title = cnd)
					if header == 'target_loc':
						crange = (0,1)
					elif header == 'dist_loc':
						crange = (-0.5,0.5)

					Z = np.dstack([np.mean(ctfs[j][cnd]['ctf'][power][fr,:], axis = (0,2)).T for j in range(len(ctfs))])
					Z = np.vstack((Z.mean(axis =2), Z.mean(axis =2)[0,:]))
					surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
		                      linewidth=0, antialiased=False, rstride = 1, cstride = 1, vmin = crange[0], vmax = crange[1])
					
					ax.set_zlim(crange)
					f.colorbar(surf, shrink = 0.5, ticks = crange)

				plt.tight_layout()
				plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'ctfs_{}_{}-{}.pdf'.format(header,band[0],band[1])))
				plt.close()

	### BDM PLOTS
				


	
	### ERP PLOTS



	def topoChannelSelection(self, header, topo_name, erp_window = dict(P1 = (0.09, 0.13), N1 = (0.15, 0.2), N2Pc = (0.18, 0.25), Pd = (0.25, 0.3))):
		'''
		Creates topoplots for time windows of interest. Can be used to select which electrodes shows the largest component (averaged across all conditions)
		NOTE: SHOULD THIS BE ALL CONDITIONS OR ONLY THE CONDITIONS OF INTEREST???????

		Arguments
		- - - - - 

		header (str): ERP tuned to target location or distractor
		topo_name (str): name of preprocessed evoked data
		erp_window (dict): dictionary of time windows of interest (tuple). Key of dict is the name of the component

		'''
		
		# read in data and shift timing
		topo, info, times = self.erpReader(header, topo_name)
	
		# first plot continuous segments of data
		plt.figure(figsize = (30,30))
		for idx, tp in enumerate(np.arange(-0.25,0.35, 0.01)):

			# select time window of interest
			s, e = [np.argmin(abs(times - t)) for t in (tp,tp+0.01)]

			# extract mean TOPO for window of interest
			T = np.mean(np.stack(
				[topo[j]['all'][:,s:e] for j in topo.keys()], axis = 0), 
				axis = (0,2))

			ax = plt.subplot(6,10 ,idx + 1, title = '{0:.2f}'.format(tp))
			im = mne.viz.plot_topomap(T, info['info'], names = info['ch_names'],
				show_names = False, show = False, axes = ax, cmap = cm.jet, vmin = -7,vmax = 5)

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'evoked-all_{}.pdf'.format(header)))
		plt.close()

		# loop over all ERP components of interest
		for erp in erp_window.keys():


			# select time window of interest
			s, e = [np.argmin(abs(times - t)) for t in erp_window[erp]]

			# extract mean TOPO for window of interest
			T = np.mean(np.stack(
				[topo[j]['all'][:,s:e] for j in topo.keys()], axis = 0), 
				axis = (0,2))

			# create figure
			plt.figure(figsize = (10,10))
			ax = plt.subplot(1,1 ,1, title = 'erp-{}'.format(header))
			im = mne.viz.plot_topomap(T, info['info'], names = info['ch_names'],
				show_names = True, show = False, axes = ax, cmap = cm.jet)

			plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'evoked-{}-{}.pdf'.format(erp, header)))
			plt.close()	

	def erpInspection(self, header, erp_name):
		'''
		Shows ERPs across whole time indow collapsed across conditions. Can also be used with 
		topoChannelSelection to select channels with largest component

		Arguments
		- - - - - 

		header (str): ERP tuned to target location or distractor
		erp_name (str): name of preprocessed erps

		'''

		# read in data and shift timing
		erps, info, times = self.erpReader(header, erp_name)

		# extract mean ERP
		ipsi = np.mean(np.stack(
				[erps[key]['all']['ipsi'] for key in erps.keys()], axis = 0),
				axis = 0)

		contra = np.mean(np.stack(
				[erps[key]['all']['contra'] for key in erps.keys()], axis = 0),
				axis = 0)

		# initiate figure
		plt.figure(figsize = (20,10))
		
		for plot, data in enumerate([ipsi, contra]):
			ax = plt.subplot(1,2 , plot + 1, title = ['ipsi','contra'][plot], ylabel = 'mV')
			ax.tick_params(axis = 'both', direction = 'outer')

			for i, erp in enumerate(data):
				plt.plot(times, erp, label = '{}-{}'.
				format(erps['2']['all']['elec'][0][i], erps['2']['all']['elec'][1][i]))

			plt.legend(loc = 'best')
			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=-0.25, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
		
		sns.despine(offset=50, trim = False)
		plt.tight_layout()

		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'elecs-{}.pdf'.format(header)))
		plt.close()	

	def ipsi_contra_ERPS(self, elecs, header, erp_name):
		'''
		plots ipsilateral and contalateral waveforms seperately. 

		Arguments
		- - - - - 

		elecs (list): list of electrodes used for ERP's
		header (str): ERP tuned to target location or distractor
		erp_name (str): name of preprocessed erps

		'''
		PO = Permutation()

		color_var = 'blue'

		if header == 'target_loc':
			color_rep = 'green'
			erp_types = ['DvTr_0','DvTr_3']
		elif header == 'dist_loc':
			erp_types = ['DrTv_0','DrTv_3']	
			color_rep = 'red'


		# read in data and shift timing
		erps, info, times = self.erpReader(header, erp_name)

		# get indices of electrodes of interest
		e_idx = np.array([erps[erps.keys()[0]]['all']['elec'][0].index(e) for e in elecs])

		plt.figure(figsize = (30,20))

		# plot ipsi and contralateral erps with bootsrapped error bar
		for idx, plot in enumerate([1,2,4,5]):

			ax = plt.subplot(2,3 , plot, title = ['Var-Ipsi','Rep-Ipsi','Var-Contra','Rep-Contra'][idx], ylabel = 'mV')
			ax.tick_params(axis = 'both', direction = 'outer')

			perm = []
			if plot == 1 or plot == 4:
				cnds = ['DvTv_0','DvTv_3']
				color= color_var
			elif plot == 2 or plot == 5:
				cnds = erp_types
				color = color_rep

			for i, cnd in enumerate(cnds):
				if plot == 1 or plot == 2:
					if e_idx.size > 1:
						erp = np.vstack([erps[str(key)][cnd]['ipsi'][e_idx].mean(0) for key in erps.keys()])
					else:	
						erp = np.vstack([erps[str(key)][cnd]['ipsi'][e_idx] for key in erps.keys()])
				elif plot == 4 or plot == 5:
					if e_idx.size > 1:
						erp = np.vstack([erps[str(key)][cnd]['contra'][e_idx].mean(0) for key in erps.keys()])
					else:	
						erp = np.vstack([erps[str(key)][cnd]['contra'][e_idx] for key in erps.keys()])

				err, signal = bootstrap(erp)
				perm.append(erp)
				plt.plot(times, signal, label = '{}-{}'.format(cnd,str(elecs)), color = color, ls = ['-','--'][i])
				plt.fill_between(times, signal + err, signal - err, alpha = 0.2, color = color)
			

			sig_cl = PO.clusterBasedPermutation(perm[0], perm[1])

			plt.ylim(-7,5)	
			
			plt.fill_between(times, -0.05, 0.05, where = sig_cl < 1, color = 'black', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=-0.25, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')

		sns.despine(offset=50, trim = False)
		plt.tight_layout()

		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'ipsi-contra_{}_erps.pdf'.format(header)))
		plt.close()




	def linkBehErp(self, elecs, header, erp_name, window = (0.27,0.33)):
		'''

		elecs (list): list of electrodes used for ERP's
		header (str): ERP tuned to target location or distractor
		erp_name (str): name of preprocessed erps

		'''

		# read in data and shift timing
		erps, info, times = self.erpReader(header, erp_name)

		# get indices of electrodes of interest
		e_idx = np.array([erps['1']['all']['elec'][0].index(e) for e in elecs])

		# select time window of interest
		s, e = [np.argmin(abs(times - t)) for t in window]

		# read in beh and get RT reduction
		RT = []
		for sj in erps.keys():
			beh_files = glob.glob(self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_ses_*.csv'.format(sj)))

			# get triggers logged in beh file
			beh = pd.concat([pd.read_csv(file) for file in beh_files])
			beh =beh[beh['practice'] == 'no']
			if header == 'dist_loc':
				RT.append(
					beh['RT'][beh['condition'] == 'DrTv_0'].values.mean() - 
					beh['RT'][beh['condition'] == 'DrTv_3'].values.mean())
			elif header == 'target_loc':
				RT.append(
					beh['RT'][beh['condition'] == 'DvTr_0'].values.mean() - 
					beh['RT'][beh['condition'] == 'DvTr_3'].values.mean())					

		# get ERP reduction
		if e_idx.size > 1:
			if header == 'dist_loc':
				diff = (np.vstack([erps[str(key)]['DrTv_0']['contra'][e_idx].mean(0) for key in erps.keys()]) - \
					np.vstack([erps[str(key)]['DrTv_0']['ipsi'][e_idx].mean(0) for key in erps.keys()])) - \
						(np.vstack([erps[str(key)]['DrTv_3']['contra'][e_idx].mean(0) for key in erps.keys()]) - \
					np.vstack([erps[str(key)]['DrTv_3']['ipsi'][e_idx].mean(0) for key in erps.keys()]))
			elif header == 'target_loc':
				diff = (np.vstack([erps[str(key)]['DvTr_0']['contra'][e_idx].mean(0) for key in erps.keys()]) - \
					np.vstack([erps[str(key)]['DvTr_0']['ipsi'][e_idx].mean(0) for key in erps.keys()])) - \
						(np.vstack([erps[str(key)]['DvTr_3']['contra'][e_idx].mean(0) for key in erps.keys()]) - \
					np.vstack([erps[str(key)]['DvTr_3']['ipsi'][e_idx].mean(0) for key in erps.keys()]))			

		diff = diff[:,s:e].mean(axis = 1)	

		# do plotting
		sns.regplot(np.array(RT), diff)	
		r, p = pearsonr(np.array(RT), diff)
		plt.title('r = {0:0.2f}, p = {1:0.2f}'.format(r,p))	
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'corr-{}-{}.pdf'.format(elecs, header)))
		plt.close()

	def repetitionRaw(self):

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# create pivot (only include trials valid trials from RT_filter)
		RT = beh.query("RT_filter == True")
		RT_piv = RT.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')
		pivot_error = pd.Series(confidence_int(RT_piv.values), index = RT_piv.keys())
		
		# plot conditions
		plt.figure(figsize = (10,10))

		ax = plt.subplot(1,1,1, title = 'Repetition effect', ylabel = 'RT (ms)', xlabel = 'repetition', ylim = (300,650))
		for i, cnd in enumerate(['DvTv','DrTv','DvTr']):
			RT_piv[cnd].mean().plot(yerr = pivot_error[cnd], label = cnd, color = ['blue','red','green'][i])
		
		plt.xlim(-0.5,3.5)
		plt.xticks([0,1,2,3])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=50, trim = False)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'repetition_effect.pdf'))		
		plt.close()


		# and plot normalized data
		norm = RT_piv.values
		for i,j in [(0,4),(4,8),(8,12)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		pivot = pd.DataFrame(norm, index = np.unique(beh['subject_nr']), columns = RT_piv.keys())
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())	

		ax = plt.subplot(1,2, 2, title = 'Normalized RT', ylabel = 'au', xlabel = 'repetition', ylim = (0.5,1), xlim = (0,4))
		for cnd in ['DvTv','DrTv','DvTr']:

			popt, pcov = curvefitting(range(4),np.array(pivot[cnd].mean()),bounds=(0, [1,1])) 
			pivot[cnd].mean().plot(yerr = pivot_error[cnd], label = '{0}: alpha = {1:.2f}; delta = {2:.2f}'.format(cnd,popt[0],popt[1]))

		plt.xlim(-0.5,3.5)
		plt.xticks([0,1,2,3])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		#plt.tight_layout()
		#plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'main_beh.pdf'))		
		#plt.close()


	def spatialGradient(self, yrange = (350,500)):
		'''

		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# add spatial dist filter
		beh['dist_bin'] = abs(beh['dist_loc'] - beh['target_loc'])
		beh['dist_bin'][beh['dist_bin'] > 3] = 6 - beh['dist_bin'][beh['dist_bin'] > 3]

		# create pivot
		beh = beh.query("RT_filter == True")
		gradient = beh.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition','dist_bin'], aggfunc = 'mean')
		gradient_err = pd.Series(confidence_int(gradient.values), index = gradient.keys())

		# Create pivot table and extract individual headers for .csv file (input to JASP)
		gradient_array = np.hstack((np.array(gradient.index).reshape(-1,1),gradient.values))
		headers = ['sj'] + ['_'.join(np.array(labels,str)) for labels in product(*gradient.keys().levels)]
		np.savetxt(self.FolderTracker(['beh','analysis'], filename = 'gradient_JASP.csv'), gradient_array, delimiter = "," ,header = ",".join(headers), comments='')

		for cnd in ['DvTr','DrTv','DvTv']:
			plt.figure(figsize = (15,15 ))
			for i in range(4):

				ax = plt.subplot(2,2, i + 1, title = 'Repetition {}'.format(i) , ylim = yrange)
				if i % 2 == 0:
					plt.ylabel('RT (ms)')
				gradient[cnd].mean()[i].plot(kind = 'bar', yerr = gradient_err[cnd][i], color = 'grey')
			
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'gradient_{}.pdf'.format(cnd)))
			plt.close()


	def primingCheck(self):
		'''

		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# filter out RT outliers
		DR = beh.query("RT_filter == True")

		# get effect of first repetition in distractor repetition block
		DR = DR.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')
		DR = DR['DrTv'][1] - DR['DrTv'][0]

		# get priming effect (only look at chance repetitions within DvTv); first get repetitions and then filter out outliers
		beh['priming'] = np.nan
		beh['priming'] = beh['priming'].apply(pd.to_numeric)

		rep = False
		for i, idx in enumerate(beh.index[1:]):

			if (beh.loc[idx - 1,'dist_loc'] == beh.loc[idx,'dist_loc']) and \
			(beh.loc[idx -1 ,'subject_nr'] == beh.loc[idx,'subject_nr']) and \
			(beh.loc[idx - 1,'block_cnt'] == beh.loc[idx,'block_cnt']) and \
			(rep == False) and beh.loc[idx,'RT_filter'] == True and beh.loc[idx - 1,'RT_filter'] == True:
				rep = True
				beh.loc[idx,'priming'] = beh.loc[idx,'RT'] - beh.loc[idx - 1,'RT']
			else:
				rep = False	
							
		# get priming effect
		PR = beh.pivot_table(values = 'priming', index = 'subject_nr', columns = ['block_type'], aggfunc = 'mean')['DvTv']	
		t, p = ttest_rel(DR, PR)


		# plot comparison
		plt.figure(figsize = (15,10))
		df = pd.DataFrame(np.hstack((DR.values,PR.values)),columns = ['effect'])
		df['subject_nr'] = range(DR.index.size) * 2
		df['block_type'] = ['DR'] * DR.index.size + ['PR'] * DR.index.size

		ax = sns.stripplot(x = 'block_type', y = 'effect', data = df, hue = 'subject_nr', size = 10,jitter = True)
		ax.legend_.remove()
		sns.violinplot(x = 'block_type', y = 'effect', data = df, color= 'white', cut = 1)

		plt.title('p = {0:.3f}'.format(p))
		plt.tight_layout()
		sns.despine(offset=10, trim = False)
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'priming.pdf'))	
		plt.close()

	def splitHalf(self, header, sj_id, index):
		'''

		'''	

		if header == 'dist_loc':
			block_type = 'DrTv'
		elif header == 'target_loc':
			block_type = 'DvTr'	

		# read in beh
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# create pivot (only include trials valid trials from RT_filter)
		RT = beh.query("RT_filter == True")
		RT_piv = RT.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')[block_type]
		
		# get repetition effect and sort
		effect = RT_piv[3] - RT_piv[0]
		if sj_id != 'all':
			effect = effect[sj_id]

		if index == 'index':	
			sj_order = np.argsort(effect.values)
		elif index == 'sj_nr':
			sj_order = effect.sort_values().index.values	

		groups = {'high':sj_order[:sj_order.size/2],
		'low':sj_order[sj_order.size/2:]}

		return groups, block_type	


	def indDiffBeh(self):
		'''

		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		RT = beh.query("RT_filter == True")
		RT_piv = RT.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')

		target = RT_piv['DvTr'][0] - RT_piv['DvTr'][3]
		dist = RT_piv['DrTv'][0] - RT_piv['DrTv'][3]

		plt.figure(figsize = (30,10))

		# plot correlation between target and distractor (repetition effect) 
		r, p = pearsonr(target,dist)
		ax = plt.subplot(1,3, 1, title = 'r = {0:0.2f}, p = {1:0.2f}'.format(r,p))
		sns.regplot(target, dist)
		plt.ylabel('distractor suppression')
		plt.xlabel('target facilitation')

		# plot individual learning effects (normalized data relative to first repetition)
		norm = RT_piv.values
		for i,j in [(0,4),(4,8),(8,12)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		normed_RT = pd.DataFrame(norm, index = np.unique(beh['subject_nr']), columns = RT_piv.keys())

		ax = plt.subplot(1,3, 2, title = 'Distractor', 
			xlabel = 'repetition', ylabel = 'RT (ms)')
		plt.plot(normed_RT['DrTv'].T)

		ax = plt.subplot(1,3, 3, title = 'Target', 
			xlabel = 'repetition', ylabel = 'RT (ms)')
		plt.plot(normed_RT['DvTr'].T)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'individual.pdf'))	
		plt.close()



	def timeFreqCTFInd(self, channel, header):
		'''

		'''

		# read in CTF data
		slopes, info = self.readCTFdata('all',channel, header, '*_slopes_all.pickle')

		times = info['times'] -250
		freqs = (info['freqs'].min(), info['freqs'].max())

		if header == 'dist_loc':
			conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
		elif header == 'target_loc':
			conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']


		power = 'total'
		for sj in range(len(slopes)):

			crange = (-0.15,0.15)	

			plt.figure(figsize = (20,15))
			for i, cnd in enumerate(conditions):
				ax =  plt.subplot(2,2, i + 1, title = cnd, ylabel = 'freqs', xlabel = 'time (ms)')
				xy = slopes[sj][cnd][power]

				plt.imshow(xy, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = crange[0], vmax = crange[1])
				plt.axvline(x=-250, ls = '--', color = 'white')
				plt.axvline(x=0, ls = '--', color = 'white')
				plt.colorbar(ticks = (crange[0],crange[1]))

			plt.tight_layout()
			plt.savefig(self.FolderTracker(['ctf',channel,'figs','ind'], filename = 'tf_{}_{}.pdf'.format(sj,header)))	
			plt.close()	

	def splitTimeFreqCTF(self, channel, header, perm = False):
		'''

		'''

		sj_id = np.array([1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
		groups, block_type = self.splitHalf(header, sj_id, 'index')
  
		# read in ctf
		slopes, info = self.readCTFdata(sj_id,channel, header, '*_slopes_all.pickle')
		times = info['times']
		freqs = (info['freqs'].min(), info['freqs'].max())

		if perm:
			slopes_p, info = self.readCTFdata(sj_id, channel, header,'*_slopes_perm_all.pickle')

		crange = (-0.15,0.15)	
		repeat = []
		for power in ['total','evoked']:

			plt.figure(figsize = (20,15))
			idx = 1
			for rep in [0,3]:
				for group in groups.keys():

					ax = plt.subplot(2,2, idx, title = 'rep_{}_{}'.format(rep,group), ylabel = 'freqs', xlabel = 'time (ms)')
					xy = np.stack([slopes[j]['{}_{}'.format(block_type,rep)][power] for j in groups[group]])
					XY = np.mean(xy,axis = 0)

					if power == 'total' and rep == 3:
						repeat.append(np.swapaxes(xy,1,2))
					
					if perm:
						xy_perm = np.stack([slopes_p[j]['{}_{}'.format(block_type,rep)][power] for j in groups[group]])
						p_val, sig = permTTest(xy, xy_perm,  p_thresh = 0.05)
						XY[sig == 0] = 0

					plt.imshow(XY, cmap = cm.jet, interpolation='none', aspect='auto', 
						origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = crange[0], vmax = crange[1])
					plt.axvline(x=0, ls = '--', color = 'white')
					plt.axvline(x=250, ls = '--', color = 'white')
					plt.colorbar(ticks = (crange[0],crange[1]))

					idx += 1

			plt.tight_layout()
			if perm:
				plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'split_{}_{}.pdf'.format(header, power)))	
			else:
				plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'split_noperm_{}_{}.pdf'.format(header, power)))	
			plt.close()	


	def clusterTestTimeFreq(self, variable, repeat, times, freqs, channel,header, power):
		'''

		'''

		plt.figure(figsize = (30,10))

		ax = plt.subplot(1,3, 1, title = 'variable', ylabel = 'freqs', xlabel = 'time (ms)')
		print 'variable'
		T_obs_plot = permTestMask2D(variable, p_value = 0.05)
		# plot 3rd - 1st repetition
		plt.imshow(T_obs_plot.T, cmap = cm.jet, interpolation='none', aspect='auto', 
			origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = 0, vmax = 5)
		plt.colorbar(ticks = (0,5))

		print 'repeat'
		ax = plt.subplot(1,3, 2, title = 'repeat', ylabel = 'freqs', xlabel = 'time (ms)')
		# plot 3rd - 1st repetition
		T_obs_plot = permTestMask2D(repeat, p_value = 0.05)
		plt.imshow(T_obs_plot.T, cmap = cm.jet, interpolation='none', aspect='auto', 
			origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = 0, vmax = 5)
		plt.colorbar(ticks = (0,5))
		
		print 'interaction'
		ax = plt.subplot(1,3, 3, title = 'interaction', ylabel = 'freqs', xlabel = 'time (ms)')
		# plot repeat - variable
		T_obs_plot = permTestMask2D([variable[1] - variable[0], repeat[1] - repeat[0]], p_value = 0.05)
		plt.imshow(T_obs_plot.T, cmap = cm.jet, interpolation='none', aspect='auto', 
			origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = 0, vmax = 5)
		plt.colorbar(ticks = (0,5))

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'TF_comparison_{}_{}.pdf'.format(header, power)))
		plt.close()	





	def ipsiContraCheck(self, header, erp_name):
		'''

		'''


		# read in data
		with open(self.FolderTracker(['erp','dist_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		with open(self.FolderTracker(['erp','target_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			t_erps = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			d_erps = pickle.load(handle)

		print t_erps.keys(), d_erps.keys()
		plt.figure(figsize = (20,20))
		titles = ['T0-left','T0-right', 'T3-left','T3-right','D0-left','D0-right','D3-left','D3-right']
		for i, cnd in enumerate(['DvTr_0','DvTr_0','DvTr_3','DvTr_3','DrTv_0','DrTv_0','DrTv_3','DrTv_3']):
			
			ax = plt.subplot(4,2 , i + 1, title = titles[i], ylabel = 'mV')
			
			if i < 4:
				if i % 2 == 0:
					ipsi = np.vstack([t_erps[str(key)][cnd]['l_ipsi'] for key in t_erps.keys()])
					contra = np.vstack([t_erps[str(key)][cnd]['l_contra'] for key in t_erps.keys()])
				else:
					ipsi = np.vstack([t_erps[str(key)][cnd]['r_ipsi'] for key in t_erps.keys()])
					contra = np.vstack([t_erps[str(key)][cnd]['r_contra'] for key in t_erps.keys()])
			else:
				if i % 2 == 0:
					ipsi = np.vstack([d_erps[str(key)][cnd]['l_ipsi'] for key in d_erps.keys()])
					contra = np.vstack([d_erps[str(key)][cnd]['l_contra'] for key in d_erps.keys()])
				else:
					ipsi = np.vstack([d_erps[str(key)][cnd]['r_ipsi'] for key in d_erps.keys()])
					contra = np.vstack([d_erps[str(key)][cnd]['r_contra'] for key in d_erps.keys()])

			err, ipsi = bootstrap(ipsi)
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'blue')
			plt.fill_between(info['times'], ipsi + err, ipsi - err, alpha = 0.2, color = 'blue')

			err, contra = bootstrap(contra)
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], contra + err, contra  - err, alpha = 0.2, color = 'green')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','figs'], filename = '{}-check-1.pdf'.format(erp_name)))
		plt.close()

		plt.figure(figsize = (20,20))

		# plot repetition effect
		ax = plt.subplot(2,2 , 1, title = 'Target repetition Left', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DvTr_0','DvTr_3']):
			L_ipsi = np.vstack([t_erps[str(key)][cnd]['l_ipsi'] for key in t_erps.keys()])
			L_contra = np.vstack([t_erps[str(key)][cnd]['l_contra'] for key in t_erps.keys()])
			err, diff = bootstrap(L_contra - L_ipsi)
			perm.append(L_contra - L_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		ax = plt.subplot(2,2 , 2, title = 'Target repetition Right', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DvTr_0','DvTr_3']):
			R_ipsi = np.vstack([t_erps[str(key)][cnd]['r_ipsi'] for key in t_erps.keys()])
			R_contra = np.vstack([t_erps[str(key)][cnd]['r_contra'] for key in t_erps.keys()])
			err, diff = bootstrap(R_contra - R_ipsi)
			perm.append(R_contra - R_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		ax = plt.subplot(2,2 , 3, title = 'Distractor repetition Left', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DrTv_0','DrTv_3']):
			L_ipsi = np.vstack([d_erps[str(key)][cnd]['l_ipsi'] for key in d_erps.keys()])
			L_contra = np.vstack([d_erps[str(key)][cnd]['l_contra'] for key in d_erps.keys()])
			err, diff = bootstrap(L_contra - L_ipsi)
			perm.append(L_contra - L_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		ax = plt.subplot(2,2 , 4, title = 'Distractor repetition Right', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DrTv_0','DrTv_3']):
			R_ipsi = np.vstack([d_erps[str(key)][cnd]['r_ipsi'] for key in d_erps.keys()])
			R_contra = np.vstack([d_erps[str(key)][cnd]['r_contra'] for key in d_erps.keys()])
			err, diff = bootstrap(R_contra - R_ipsi)
			perm.append(R_contra - R_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','figs'], filename = '{}-check-2.pdf'.format(erp_name)))
		plt.close()


	def N2pCvsPd(self, erp_name, split = False):
		'''		

		'''

		sj_id = np.array([1,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21])

		# read in data
		with open(self.FolderTracker(['erp','dist_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		with open(self.FolderTracker(['erp','target_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			t_erps = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			d_erps = pickle.load(handle)

		if split:
			groups, block_type = self.splitHalf(split, sj_id, 'sj_nr')

		else:
			groups = {'all':t_erps.keys()}	

		for group in groups.keys():	

			# get ipsilateral and contralateral erps tuned to the target and tuned to the distractor (collapsed across all conditions)
			#T_ipsi = np.vstack([t_erps[str(key)]['all']['ipsi'] for key in t_erps.keys()])
			#T_contra = np.vstack([t_erps[str(key)]['all']['contra'] for key in t_erps.keys()])
			T_ipsi = np.vstack([t_erps[str(key)]['all']['ipsi'] for key in groups[group]])
			T_contra = np.vstack([t_erps[str(key)]['all']['contra'] for key in groups[group]])

			#D_ipsi = np.vstack([d_erps[str(key)]['all']['ipsi'] for key in d_erps.keys()])
			#D_contra = np.vstack([d_erps[str(key)]['all']['contra'] for key in d_erps.keys()])
			D_ipsi = np.vstack([d_erps[str(key)]['all']['ipsi'] for key in groups[group]])
			D_contra = np.vstack([d_erps[str(key)]['all']['contra'] for key in groups[group]])

			plt.figure(figsize = (20,20))
			
			# plot ipsi and contralateral erps with bootsrapped error bar
			ax = plt.subplot(4,2 , 1, title = 'Target ERPs', ylabel = 'mV')

			err, ipsi = bootstrap(T_ipsi)
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'blue')
			plt.fill_between(info['times'], ipsi + err, ipsi - err, alpha = 0.2, color = 'blue')

			err, contra = bootstrap(T_contra)
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], contra + err, contra  - err, alpha = 0.2, color = 'green')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			ax = plt.subplot(4,2 , 2, title = 'Distractor ERPs', ylabel = 'mV')
			
			err, ipsi = bootstrap(D_ipsi)
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'blue')
			plt.fill_between(info['times'], ipsi + err, ipsi - err, alpha = 0.2, color = 'blue')
			plt.legend(loc = 'best')	

			err, contra = bootstrap(D_contra)
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], contra + err, contra - err, alpha = 0.2, color = 'green')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			# plot diff wave collapsed across all conditions
			ax = plt.subplot(4,2 , 3, title = 'Target diff', ylabel = 'mV')
			
			err, diff = bootstrap(T_contra - T_ipsi)
			plt.plot(info['times'], diff, color = 'black')
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = 'black')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')
			
			ax = plt.subplot(4,2 , 4, title = 'Distractor diff', ylabel = 'mV')
			
			err, diff = bootstrap(D_contra - D_ipsi)
			plt.plot(info['times'], diff, color = 'black')
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = 'black')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')
		
			# plot repetition effect
			ax = plt.subplot(4,2 , 5, title = 'Target repetition', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DvTr_0','DvTr_3']):
				T_ipsi = np.vstack([t_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				T_contra = np.vstack([t_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(T_contra - T_ipsi)
				perm.append(T_contra - T_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			ax = plt.subplot(4,2 , 6, title = 'Distractor repetition', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DrTv_0','DrTv_3']):
				D_ipsi = np.vstack([d_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				D_contra = np.vstack([d_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(D_contra - D_ipsi)
				perm.append(D_contra - D_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			# plot repetition effect (control)
			ax = plt.subplot(4,2, 7, title = 'Target repetition (control)', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DvTv_0','DvTv_3']):
				T_ipsi = np.vstack([t_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				T_contra = np.vstack([t_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(T_contra - T_ipsi)
				perm.append(T_contra - T_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			ax = plt.subplot(4,2, 8, title = 'Distractor repetition (control)', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DvTv_0','DvTv_3']):
				D_ipsi = np.vstack([d_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				D_contra = np.vstack([d_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(D_contra - D_ipsi)
				perm.append(D_contra - D_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			sns.despine(offset=10, trim = False)
			plt.tight_layout()
			if split:
				plt.savefig(self.FolderTracker(['erp','figs'], filename = 'n2pc-Pd-{}-{}_{}.pdf'.format(group,split,erp_name)))	
			else:
				plt.savefig(self.FolderTracker(['erp','figs'], filename = 'n2pc-Pd_{}_{}.pdf'.format(group,erp_name)))		
			plt.close()


	def clusterTopo(self, header, fname = ''):
		'''

		'''

		# read in data
		files = glob.glob(self.FolderTracker(['erp', header], filename = fname))
		topo = []
		for file in files:
			with open(file ,'rb') as handle:
				topo.append(pickle.load(handle))



	def topoAnimation(self, header):
		'''

		'''


		# read in data
		files = glob.glob(self.FolderTracker(['erp', header], filename = 'topo_*.pickle'))
		topo = []
		for file in files:
			print file
			# read in erp dict
			with open(file ,'rb') as handle:
				topo.append(pickle.load(handle))

		# read in processed data object (contains info for plotting)
		EEG = mne.read_epochs(self.FolderTracker(extension = ['processed'], filename = 'subject-1_all-epo.fif'))

		# read in plot dict		
		with open(self.FolderTracker(['erp',header], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		plt_idx = [1,3,7,9]
		for image in range(564):
			f = plt.figure(figsize = (20,20))	
			for i, cnd in enumerate(np.sort(topo[0].keys())):

				ax = plt.subplot(3,3 , plt_idx[i], title = cnd)

				T = np.mean(np.dstack([np.mean(topo[j][cnd], axis = 0) for j in range(len(topo))]), axis = 2)
				mne.viz.plot_topomap(T[:,image], EEG.info, show_names = False, show = False, vmin = -4, vmax = 3)

			ax = plt.subplot(3,3 , 5, title = '{0:0.2f}'.format(info['times'][image]))
			if info['times'][image] <= 0:	
				searchDisplayEEG(ax, fix = True)
			elif info['times'][image] <= 0.25:
				searchDisplayEEG(ax, fix = False)	
			else:
				searchDisplayEEG(ax, fix = False, stimulus = 4, erp_type = header)		

			plt.tight_layout()	
			plt.savefig(self.FolderTracker(['erp', 'figs','video'], filename = 'topo_{0}_{1:03}.png'.format(header,image + 1)))
			plt.close()

		plt_idx = [1,3]	
		for image in range(564):
			f = plt.figure(figsize = (20,20))	
			for i in range(2):

				if i == 0:
					title = 'variable'
					T = np.mean(np.dstack([np.mean(topo[j]['DvTv_0'], axis = 0) for j in range(len(topo))]), axis = 2) - \
					np.mean(np.dstack([np.mean(topo[j]['DvTv_3'], axis = 0) for j in range(len(topo))]), axis = 2)
				else:
					T = np.mean(np.dstack([np.mean(topo[j]['DrTv_0'], axis = 0) for j in range(len(topo))]), axis = 2) - \
					np.mean(np.dstack([np.mean(topo[j]['DrTv_3'], axis = 0) for j in range(len(topo))]), axis = 2)
					title = 'repeat'	
				ax = plt.subplot(1,3 ,plt_idx[i] , title = title)
				mne.viz.plot_topomap(T[:,image], EEG.info, show_names = False, show = False, vmin = -1, vmax = 1)

			ax = plt.subplot(1,3 , 2, title = '{0:0.2f}'.format(info['times'][image]))
			if info['times'][image] <= 0:	
				searchDisplayEEG(ax, fix = True)
			elif info['times'][image] <= 0.25:
				searchDisplayEEG(ax, fix = False)	
			else:
				searchDisplayEEG(ax, fix = False, stimulus = 4, erp_type = header)		

			plt.tight_layout()	
			plt.savefig(self.FolderTracker(['erp', 'figs','video'], filename = 'topo_diff_{0}_{1:03}.png'.format(header,image + 1)))
			plt.close()


if __name__ == '__main__':
	
	os.chdir('/home/dvmoors1/BB/Dist_suppression') 

	PO = EEGDistractorSuppression()

	#PO.conditionCheck(thresh_bin = 1.00)

	# ANALYSIS PAPER 
	#PO.diff_ERPS(elecs = ['PO7','PO3','O1'], erp_name= 'lat-down1-mid')
	#PO.diff_ERPS(elecs = ['PO7','PO3','O1'], erp_name= 'lat-down1')

	# Behavior plots 
	#PO.repetitionRaw()	
	#PO.spatialGradient()
	#PO.primingCheck()
	#PO.indDiffBeh()

	# CTF plots
	#PO.alphaSlopes()
	PO.crossTraining()

	#PO.CTFslopes(header = 'target', ctf_name = 'slopes_alpha', fband = 'alpha')
	#PO.CTFslopes(header = 'dist', ctf_name = 'slopes_alpha', fband = 'alpha')
	#PO.timeFreqCTF(header = 'target', cnd_name = 'all', perm = False, p_map = False)
	#PO.timeFreqCTF(header = 'dist', cnd_name = 'cnds',perm = False, p_map = False)
	#PO.plotCTF(header = 'dist')
	#PO.plotCTF(header = 'dist')
	#PO.threeDSlopes(header = 'dist_loc')
	#PO.threeDSlopes(header = 'target_loc')
	#PO.rawCTF(header = 'dist_loc', ctf_name = 'ctf-raw')
	#PO.rawCTF(header = 'target_loc', ctf_name = 'ctf-raw')

	#PO.splitTimeFreqCTF(channel = 'posterior_channels', header = 'target_loc', perm = True)
	#PO.splitTimeFreqCTF(channel = 'posterior_channels', header = 'dist_loc', perm = True)


	# BDM plots
	#PO.bdmdiag()
	#PO.bdmACC(header = 'target')

	# ERP plots
	# PO.topoChannelSelection(header = 'dist_loc', topo_name = 'topo_lat-down1')
	# PO.erpInspection(header = 'dist_loc', erp_name = 'lat-down1')
	# PO.topoChannelSelection(header = 'target_loc', topo_name = 'topo_lat-down1')
	# PO.erpInspection(header = 'target_loc', erp_name = 'lat-down1')
	#PO.diff_ERPS(elecs = ['PO7','PO3','O1'], header = 'dist', erp_name= 'lat-down1')
	#PO.diff_ERPS(elecs = ['PO7','PO3','O1'], header = 'target', erp_name= 'lat-down1')
	#PO.cndTOPO('dist', start = 0.05, stop = 0.15, step = 0.01)
	#PO.cndTOPO('dist')
	# PO.ipsi_contra_ERPS(elecs = ['PO7','PO3','O1'], header = 'dist_loc', erp_name = 'lat-down1')
	# PO.ipsi_contra_ERPS(elecs = ['PO7','PO3','O1'], header = 'target_loc', erp_name = 'lat-down1')
	# PO.linkBehErp(elecs = ['PO7','PO3'], header = 'dist_loc', erp_name = 'lat-down1', window = (0.29,0.36))
	# PO.linkBehErp(elecs = ['PO7','PO3'], header = 'target_loc', erp_name = 'lat-down1', window = (0.16,0.22))

	# TARGET VS DISTRACTOR
	#PO.DT()


