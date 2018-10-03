import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import os
import mne
import sys
import glob
import pickle
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import argrelextrema
from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.TF import * 
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
from eeg_analyses.CTF import * 
from eeg_analyses.Spatial_EM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info
sj_info = {'1': {'tracker': (False, '', ''),  'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			}

# project specific info
project = 'DT_reps'
part = 'beh'
factors = ['block_type','repetition']
labels = [['DrTv','DvTr','DvTv'],[1,2,3,4]]
to_filter = ['RT'] 
project_param = ['practice','nr_trials','trigger','condition','RT', 'subject_nr',
				'block_type', 'correct','dist_loc','dist_orient','target_loc',
				'target_orient','repetition','fixed_pos']

montage = mne.channels.read_montage(kind='biosemi64')
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
trigger = []
t_min = 0
t_max = 0
flt_pad = 0.5
eeg_runs = [1,2]
binary =  0

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class DT_reps(FolderStructure):

	def __init__(self): pass

	def prepareBEH(self, project, part, factors, labels, project_param):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and correct == 1', cnd_sel = True, save = True)
		PP.exclude_outliers(criteria = dict(RT = 'RT_filter == True', correct = ''))
		PP.prep_JASP(agg_func = 'mean', voi = 'RT', data_filter = 'RT_filter == True', save = True)
		PP.save_data_file()

	def BEHexp1(self):
		'''
		analyzes experiment 1 as reported in the MS
		'''

		# read in data
		file = self.FolderTracker(['beh-exp1','analysis'], filename = 'preprocessed.csv')
		data = pd.read_csv(file)

		# create pivot (main analysis)
		data = data.query("RT_filter == True")
		pivot = data.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','set_size','repetition'], aggfunc = 'mean')
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# plot conditions
		plt.figure(figsize = (20,10))

		ax = plt.subplot(1,2, 1, title = 'Repetition effect', ylabel = 'RT (ms)', xlabel = 'repetition', ylim = (300,650), xlim = (0,11))
		for b, bl in enumerate(['target','dist']):
			for s, set_size in enumerate([4,8]):
				pivot[bl][set_size].mean().plot(yerr = pivot_error[bl][set_size], 
											label = '{}-{}'.format(bl,set_size), 
											ls = ['-','--'][s], color = ['green','red'][b])
		
		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		# create pivot (normalized data)
		norm = pivot.values
		for i,j in [(0,12),(12,24),(24,36),(36,48)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		pivot = pd.DataFrame(norm, index = np.unique(data['subject_nr']), columns = pivot.keys())
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# fit data to exponential decay function (and plot normalized data)
		ax = plt.subplot(1,2, 2, title = 'Normalized RT', ylabel = 'au', xlabel = 'repetition', ylim = (0.5,1), xlim = (0,11))
		alpha, delta = np.zeros((pivot.shape[0],4)),  np.zeros((pivot.shape[0],4))
		c_idx = 0
		headers = []
		for b, bl in enumerate(['target','dist']):
			for s, set_size in enumerate([4,8]):
				headers.append('{}_{}'.format(bl,set_size))
				X = pivot[bl][set_size].values
				pivot[bl][set_size].mean().plot(yerr = pivot_error[bl][set_size], 
												label = '{0}-{1}'.format(bl,set_size),
												ls = ['-','--'][s], color = ['green','red'][b])
				for i, x in enumerate(X):
					popt, pcov = curvefitting(range(12),x,bounds=(0, [1,1])) 
					alpha[i, c_idx] = popt[0]
					delta[i, c_idx] = popt[1]
				c_idx += 1

		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		plt.savefig(self.FolderTracker(['beh-exp1','figs'], filename = 'main-ana.pdf'))		
		plt.close()	

		# save parameters for JASP 		
		np.savetxt(self.FolderTracker(['beh-exp1','analysis'], filename = 'fits_alpha-JASP.csv'), alpha, delimiter = "," ,header = ",".join(headers), comments='')
		np.savetxt(self.FolderTracker(['beh-exp1','analysis'], filename = 'fits_delta-JASP.csv'), delta, delimiter = "," ,header = ",".join(headers), comments='')

	def clusterPlot(self, X1, X2, p_val, times, y, color, ls = '-'):
		'''
		plots significant clusters in the current plot
		'''	

		# indicate significant clusters of individual timecourses
		sig_cl = clusterBasedPermutation(X1, X2, p_val = p_val)
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size) * y, color = color, ls = ls)

	def beautifyPlot(self, y = 0, xlabel = 'Time (ms)', ylabel = 'Mv', ls = '-'):
		'''
		Adds markers to the current plot. Onset placeholder and onset search and horizontal axis
		'''

		plt.axhline(y=y, ls = ls, color = 'black')
		plt.axvline(x=-0.25, ls = ls, color = 'black')
		plt.axvline(x=0, ls = ls, color = 'black')
		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

	### ERP ANALYSIS	
	def erpSelection(self, header, topo_name, elec = ['PO3','PO7']):
		'''

		Uses condition averaged data to select electrodes and time windows of interest for erp analysis

		Arguments
		- - - - - 

		header (str): ERP tuned to target location or distractor
		topo_name (str): name of preprocessed evoked data
		erp_window (dict): dictionary of time windows of interest (tuple). Key of dict is the name of the component

		'''

		# read in erp and topo data
		with open(self.FolderTracker(['erp',header], filename = '{}.pickle'.format(topo_name)) ,'rb') as handle:
			erp = pickle.load(handle)

		with open(self.FolderTracker(['erp',header], filename = 'topo_{}.pickle'.format(topo_name)) ,'rb') as handle:
			topo = pickle.load(handle)

		with open(self.FolderTracker(['erp',header], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		# visualization step 1: plot topograpic plots across time
		plt.figure(figsize = (80,80))
		for idx, tp in enumerate(np.arange(0,0.4, 0.01)):
			ax = plt.subplot(5,8 ,idx + 1, title = '{0:.2f}'.format(tp))
			# select time window of interest and extract mean topgraphy
			s, e = [np.argmin(abs(times - t)) for t in (tp,tp+0.01)]
			T = np.mean(np.stack(
				[topo[j]['all'][:,s:e] for j in topo.keys()], axis = 0), 
				axis = (0,2))
			im = mne.viz.plot_topomap(T, info['info'], names = info['ch_names'],
				show_names = True, show = False, axes = ax, cmap = cm.jet, vmin = -7,vmax = 5)

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'evoked-all-acrosstime_{}.pdf'.format(header)))
		plt.close()

		# visualization step 2: plot difference waves across time for selected electrodes
		elec_idx = [erp[erp.keys()[0]]['all']['elec'][0].index(e) for e in elec]			
		ipsi = np.mean(np.stack([erp[key]['all']['ipsi'] for key in erp.keys()])[:,elec_idx], axis = 1)
		contra = np.mean(np.stack([erp[key]['all']['contra'] for key in erp.keys()])[:,elec_idx], axis = 1)
		d_wave = (contra - ipsi).mean(axis = 0)

		# initiate figure
		plt.figure(figsize = (20,10))
		plt.plot(times, d_wave)
		self.beautifyPlot(y = 0)
		self.clusterPlot(contra - ipsi, 0, 0.05, times, -3, color = 'black')
		sns.despine(offset=50, trim = False)
		plt.tight_layout()

		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'grand_mean-{}.pdf'.format(header)))
		plt.close()	

		# visualization step 3: plot difference waves across time for selected electrodes 
		# (only include conditions without repetition)
		cnds = ['DvTv_0','DrTv_0','DvTr_0']
		ipsi = np.mean(np.stack([np.mean(np.stack([erp[key][cnd]['ipsi'] 
				for key in erp.keys()])[:,elec_idx], axis = 1) 
				for cnd in cnds]), axis = 0)
		contra = np.mean(np.stack([np.mean(np.stack([erp[key][cnd]['contra'] 
				for key in erp.keys()])[:,elec_idx], axis = 1) 
				for cnd in cnds]), axis = 0)
		d_wave = (contra - ipsi).mean(axis = 0)
		# initiate figure
		plt.figure(figsize = (20,10))
		plt.plot(times, d_wave)
		self.beautifyPlot(y = 0)
		self.clusterPlot(contra - ipsi, 0, 0.05, times, -3, color = 'black')
		sns.despine(offset=50, trim = False)
		plt.tight_layout()

		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'subset_mean-{}.pdf'.format(header)))
		plt.close()	 


	def componentSelection(self, header, cmp_name, cmp_window, ext, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1']):
		'''
		for each component of interest finds the local maximum which is then extended by 
		a specified window. Mean amplitudes across conditions and hemifilields are then
		saved in a .csv file for statistical analysis in JASP.
		'''

		# read in erp  data and times
		with open(self.FolderTracker(['erp',header], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			erp = pickle.load(handle)
		with open(self.FolderTracker(['erp','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		# general parameters
		if header == 'target_loc':
			cnds = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
			blocks =  ['DvTv_','DvTr_']
		elif header == 'dist_loc':	
			cnds = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
			blocks =  ['DvTv_','DrTv_']

		# start with creating condition averaged waveform
		e_idx = [erp[erp.keys()[0]]['all']['elec'][0].index(e) for e in elec]	
		contra = np.mean(np.stack([erp[e][c]['contra'][e_idx] 
						for e in erp for c in cnds if c[:-1] in blocks]), 
						axis = (0,1))
		ipsi = np.mean(np.stack([erp[e][c]['ipsi'][e_idx] 
						for e in erp for c in cnds if c[:-1] in blocks]), 
						axis = (0,1))
		gr_diff = contra - ipsi

		# zoom in on component window and find local minimum/maximum (seperate for ipsi and contra)
		strt, end = [np.argmin(abs(times - t)) for t in cmp_window]	
		if 'P1' == cmp_name:
			ipsi_peak = times[strt:end][np.argmax(ipsi[strt:end])]
			contra_peak = times[strt:end][np.argmax(contra[strt:end])]
		elif 'N1' == cmp_name:
			ipsi_peak = times[strt:end][np.argmin(ipsi[strt:end])]
			contra_peak = times[strt:end][np.argmin(contra[strt:end])]
		elif 'N2pc' == cmp_name:
			diff_peak = times[strt:end][np.argmin(gr_diff[strt:end])]
		elif 'Pd' == cmp_name:
			diff_peak = times[strt:end][np.argmin(gr_diff[strt:end])]	


		# show plots of ipsi and contra waveforms (left) and diff wave (right)
		plt.figure(figsize = (30,15))
		ax = plt.subplot(1,2, 1, title = 'lateralization peaks')
		plt.plot(times, contra, label = 'contra', color = 'red')
		plt.plot(times, ipsi, label = 'ipsi', color = 'green')
		self.beautifyPlot(y = 0, xlabel = 'Time (ms)', ylabel = 'Mv', ls = '-')
		if cmp_name in ['P1','N1']:
			plt.axvline(x = ipsi_peak, ls = '--', color = 'green')
			plt.axvline(x = contra_peak, ls = '--', color = 'red')
			plt.axvline(x = cmp_window[0], ls = '--', color = 'black')
			plt.axvline(x = cmp_window[1], ls = '--', color = 'black')
		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)

		ax = plt.subplot(1,2, 2, title = 'diff wave peak')
		plt.plot(times,  gr_diff, color = 'blue')
		if cmp_name in ['N2pc','Pd']:
			plt.axvline(x = diff_peak, ls = '--', color = 'red')
			plt.axvline(x = cmp_window[0], ls = '--', color = 'black')
			plt.axvline(x = cmp_window[1], ls = '--', color = 'black')
		self.beautifyPlot(y = 0, xlabel = 'Time (ms)', ylabel = 'Mv', ls = '-')
		plt.axvline(x = cmp_window[0], ls = '--', color = 'black')
		plt.axvline(x = cmp_window[1], ls = '--', color = 'black')
		sns.despine(offset=50, trim = False)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','cmps'], filename = '{}-{}.pdf'.format(header,cmp_name)))
		plt.close()	

		# store data in array
	
		data = []
		headers = []
		lateralization = ['contra','ipsi']
		if cmp_name in ['P1','N1']:
			s_ipsi, e_ipsi = [np.argmin(abs(times - t)) for t in (ipsi_peak - ext, ipsi_peak + ext)]	
			s_contra, e_contra = [np.argmin(abs(times - t)) for t in (contra_peak - ext, contra_peak + ext)]

		for block in blocks:
			for rep in ['0','3']:
				for lat in lateralization:
					headers.append('{}{}_{}'.format(block, rep, lat))
					if lat == 'contra' and cmp_name in ['P1','N1']:
						data.append(np.stack([erp[e][block + rep][lat][e_idx,s_contra:e_contra] for e in erp]).mean(axis = (1,2))) 
					elif lat == 'ipsi' and cmp_name in ['P1','N1']:	
						data.append(np.stack([erp[e][block + rep][lat][e_idx,s_ipsi:e_ipsi] for e in erp]).mean(axis = (1,2)))
					else:					
						data.append(np.stack([erp[e][block + rep][lat][e_idx,strt:end] for e in erp]).mean(axis = (1,2)))	

		X = np.vstack(data).T	
		# save as .csv file
		np.savetxt(self.FolderTracker(['erp',header], filename = '{}.csv'.format(cmp_name)), X, delimiter = "," ,header = ",".join(headers), comments='')

	def erpLateralized(self, header, erp_name, elec):
		'''
		Plots the ipsi and contralateral waveforms (and difference waves) seperately across conditions 
		for first and final repetition in the sequence
		''' 

		# general parameters
		with open(self.FolderTracker(['erp',header], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25
		
		if header == 'target_loc':
			blocks =  ['DvTv_','DvTr_']
			colors = ['blue', 'green']
		elif header == 'dist_loc':	
			blocks =  ['DvTv_','DrTv_']
			colors = ['blue', 'red']

		# read in erp data
		with open(self.FolderTracker(['erp',header], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			erp = pickle.load(handle)
		e_idx = [erp[erp.keys()[0]]['all']['elec'][0].index(e) for e in elec]	

		# show plots of ipsi (left) and contra (right) waveforms for baseline (up) and repetition (down) blocks
		plt.figure(figsize = (30,30))
		plt_idx = 1

		diff = {'ipsi':{},'contra':{}}
		for lat in ['ipsi', 'contra']:
			
			for color, bl in zip(colors,blocks):
				plt.subplot(3,2, plt_idx, title = '{}{}'.format(bl,lat),  ylim = (-8,6))
				# select data
				rep_effect = []
				for i, (rep, ls) in enumerate(zip(['0','3'],['-','--'])):
					diff[lat].update({bl+rep:{}})
					x = np.mean(
						np.stack([erp[e][bl+rep][lat][e_idx] for e in erp]),
						axis = 1)
					rep_effect.append(x)
					diff[lat][bl+rep] = x

					plt.plot(times, x.mean(axis = 0), label = bl+rep, color = color, ls = ls)
					self.clusterPlot(rep_effect[-1], 0, 0.05, times, plt.ylim()[0] + 0.5 + 0.5*i, color, ls)	

				self.clusterPlot(rep_effect[0],rep_effect[1], 0.05, times, plt.ylim()[0] + 1.5, 'black', '-')	
				self.beautifyPlot(y = 0, xlabel = 'Time (ms)', ylabel = 'Mv', ls = '-')	
				plt.legend(loc = 'best')
				sns.despine(offset=50, trim = False)	
				plt_idx += 1
	
		for i, (rep, ls) in enumerate(zip(['0','3'],['--','-'])):
			plt.subplot(3,2, plt_idx, title = 'Diff-{}'.format(bl[:-1]), ylim = (-5,1.5))
			perm = []
			for color, bl in zip(colors,blocks):
			
				x = diff['contra'][bl+rep] - diff['ipsi'][bl+rep]
				perm.append(x)
				plt.plot(times, x.mean(axis = 0), label = bl+rep, color = color, ls = ls)
				self.clusterPlot(x, 0, 0.05, times, plt.ylim()[0] + 0.5 + 0.5*i, color, ls)	

			# jacknife procedure (N2pc)
			print '\n {}'.format(header)
			onset, t_value = jackknife(perm[0],perm[1], times, [0.17,0.23], percent_amp = 45, timing = 'offset')	
			print bl+rep, 'onset', onset *1000, t_value
			self.clusterPlot(perm[0],perm[1], 0.05, times, plt.ylim()[0] + 1.5, 'black', '-')
			plt_idx += 1
			self.beautifyPlot(y = 0, xlabel = 'Time (ms)', ylabel = 'Mv', ls = '-')			
			plt.legend(loc = 'best')
			sns.despine(offset=50, trim = False)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = '{}-ipsi_contra.pdf'.format(header)))
		plt.close()			



	def erpContrast(self, erp_name, elec):
		'''
		Contrasts differnce waveforms across conditions between first and final repetition in the sequence
		'''

		# read in erp and topo data
		with open(self.FolderTracker(['erp','target_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			T_erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			D_erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25
		elec_idx = [D_erp[D_erp.keys()[0]]['all']['elec'][0].index(e) for e in elec]

		# step 1: analyse lateralized effects (contra - ipsi)
		for a, cnds in enumerate([['DvTv_0','DvTv_3','DvTr_0','DvTr_3'],['DvTv_0','DvTv_3','DrTv_0','DrTv_3']]):
			plt.figure(figsize = (30,10))
			plt_idx = 1
			diff = []
			for b, cnd in enumerate(cnds):
				if b % 2 == 0:
					ax = plt.subplot(1,2, plt_idx, ylim = (-3.5,1.5)) 
					plt_idx += 1

				# get data to plot
				if a == 0:
					erp = T_erp
					colors = ['blue'] * 2 + ['green'] * 2
				elif a == 1:
					erp = D_erp
					colors = ['blue'] * 2 + ['red'] * 2	
				ipsi = np.mean(np.stack([erp[key][cnd]['ipsi'] for key in erp.keys()])[:,elec_idx], axis = 1)
				contra = np.mean(np.stack([erp[key][cnd]['contra'] for key in erp.keys()])[:,elec_idx], axis = 1)
				diff.append(contra - ipsi)
				d_wave = (contra - ipsi).mean(axis = 0)
				plt.plot(times, d_wave, label = cnd, color = colors[b], ls = ['-','--','-','--'][b])
				
				if b % 2 == 1:
					plt.legend(loc = 'best')
					self.beautifyPlot(y = 0)
					# cluster based permutation
					self.clusterPlot(diff[-2], diff[-1], 0.05, times, ax.get_ylim()[0] + 0.05, color = 'black')
					# compare onset and offset N2pc
					onset, t_value = jackknife(diff[-2], diff[-1], times, peak_window = (0.10,0.3), percent_amp = 50, timing = 'onset')
					if abs(t_value) > stats.t.ppf(1 - 0.025, ipsi.shape[0] - 1):
						print 'Onset between {} and {} is significant (t value = {}, onset = {})'.format(cnds[b], cnds [b - 1], t_value, onset)
					offset, t_value = jackknife(diff[-2], diff[-1], times, peak_window = (0.10,0.3), percent_amp = 50, timing = 'offset')
					if abs(t_value) > stats.t.ppf(1 - 0.025, ipsi.shape[0] - 1):
						print 'Offset between {} and {} is significant (t value = {}, offset = {})'.format(cnds[b], cnds [b - 1], t_value, offset * 1000)

				sns.despine(offset=50, trim = False)
				
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'main_{}.pdf'.format(['target','dist'][a])))
			plt.close()

			# step 2: analyse ipsi and contralateral effects seperately
			for a, cnds in enumerate([['DvTr_0','DvTr_3'],['DrTv_0','DrTv_3']]):
				plt.figure(figsize = (30,10))
				for b, cnd in enumerate(cnds):
					ax = plt.subplot(1,2, b + 1, ylim = (-7.5, 5.5), title = cnd) 
					contra = np.mean(np.stack([erp[key][cnd]['contra'] for key in erp.keys()])[:,elec_idx], axis = 1)
					ipsi = np.mean(np.stack([erp[key][cnd]['ipsi'] for key in erp.keys()])[:,elec_idx], axis = 1)
					plt.plot(times, contra.mean(axis = 0), label = 'contra', color = 'red')
					plt.plot(times, ipsi.mean(axis = 0), label = 'ipsi', color = 'green')
					#plt.fill_between(times, m_wave + err, m_wave - err, alpha = 0.2, color = ['red', 'green'][b])

					plt.legend(loc = 'best')
					self.beautifyPlot(y = 0)
					sns.despine(offset=50, trim = False)
				
				# save figures
				plt.tight_layout()
				plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'main-latnew_{}.pdf'.format(['target','dist'][a])))
				plt.close()		

	def frontalBias(self, erp_name = 'topo_lat-down1-mid', elec = ['AF3','Fz','AFz','AF4','F3','F1','F2','F4']):
		'''

		'''

		# read in erp and topo data
		with open(self.FolderTracker(['erp','target_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			T_erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			D_erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		times = info['times'] - 0.25
		elec_idx = [info['ch_names'].index(e) for e in elec]

		s_,e_ = [np.argmin(abs(t - times)) for t in (0.2,0.27)]
		# step 1: analyse lateralized effects (contra - ipsi)
		for a, cnds in enumerate([['DvTv_0','DvTv_3','DvTr_0','DvTr_3'],['DvTv_0','DvTv_3','DrTv_0','DrTv_3']]):
			plt.figure(figsize = (30,10))
			plt_idx = 1
			diff = []
			data = []
			headers = []
			for b, cnd in enumerate(cnds):
				headers.append(cnd)
				if b % 2 == 0:
					ax = plt.subplot(1,2, plt_idx, ylim = (-4,4)) 
					plt_idx += 1

				# get data to plot
				if a == 0:
					erp = T_erp
					colors = ['blue'] * 2 + ['green'] * 2
				elif a == 1:
					erp = D_erp
					colors = ['blue'] * 2 + ['red'] * 2	
				X = np.mean(np.stack([erp[key][cnd] for key in erp.keys()])[:,elec_idx], axis = 1)
				data.append(X[:,s_:e_].mean(axis = 1))
				diff.append(X)
				plt.plot(times, X.mean(axis = 0), label = cnd, color = colors[b], ls = ['-','--','-','--'][b])
				
				if b % 2 == 1:
					plt.axvline(x=0.2, ls = '--', color = 'black')
					plt.axvline(x=0.27, ls = '--', color = 'black')
					plt.legend(loc = 'best')
					self.beautifyPlot(y = 0)
					# cluster based permutation
					self.clusterPlot(diff[-2], diff[-1], 0.05, times, ax.get_ylim()[0] + 0.05, color = 'black')

				sns.despine(offset=50, trim = False)
			
			X = np.vstack(data).T	
			# save as .csv file
			np.savetxt(self.FolderTracker(['erp',['target_loc','dist_loc'][a]], filename = '{}.csv'.format('FB')), X, delimiter = "," ,header = ",".join(headers), comments='')	
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'frontal-bias_{}.pdf'.format(['target','dist'][a])))
			plt.close()

	# BDM ANALYSIS	
	def bdmAcc(self,band = 'alpha'):
		'''
		plots decoding across time across conditions
		'''
		with open(self.FolderTracker(['bdm','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		files = glob.glob(self.FolderTracker(['bdm', 'target_loc'], filename = 'class_*_perm-False.pickle'.format(band)))
		bdm_T = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm_T.append(pickle.load(handle))

		files = glob.glob(self.FolderTracker(['bdm', 'dist_loc'], filename = 'class_*_perm-False.pickle'.format(band)))
		bdm_D = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm_D.append(pickle.load(handle))

		# step 2: analyse ipsi and contralateral effects seperately
		plt.figure(figsize = (30,20))
		idx_plot = 1
		for a, cnds in enumerate([['DvTv_0','DvTr_0','DvTv_3','DvTr_3'],['DvTv_0','DrTv_0','DvTv_3','DrTv_3']]):
			title = ['Target','Dist'][a]
			bdm = [bdm_T, bdm_D][a]
			rep_color = ['green','red'][a]
			for b, cnd in enumerate(cnds):
				color = ['blue', rep_color][0 if b < 2 else 1]
				if b % 2 == 0:
					ax = plt.subplot(2,2, idx_plot, ylim = (0.1,0.3), title = title) 
					self.beautifyPlot(y = 1/6.0, ylabel = 'Decoding acc (%)')
					idx_plot += 1
					
				X = np.stack([bdm[sj][cnd]['standard'] for sj in range(len(bdm))])
				err, x = bootstrap(X)
				plt.plot(times, x, label = cnd, ls = ['-','--'][b%2], color = color)
				plt.fill_between(times, x + err, x - err, alpha = 0.2, color = color)
				
				if b % 2 == 1:
					sns.despine(offset=50, trim = False)
					plt.legend(loc = 'best')
				
		# save figure
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['bdm','MS-plots'], filename = 'bdm_acc.pdf'.format(band)))
		plt.close()	



	def bdmDiag(self):

		# read in data
		with open(self.FolderTracker(['bdm','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		files = glob.glob(self.FolderTracker(['bdm', 'target_loc'], filename = 'class_*_perm-False-alpha.pickle'))
		bdm_T = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm_T.append(pickle.load(handle))

		files = glob.glob(self.FolderTracker(['bdm', 'dist_loc'], filename = 'class_*_perm-False-alpha.pickle'))
		bdm_D = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm_D.append(pickle.load(handle))
	
		# step 1: analyze diagnal 
		for a, cnds in enumerate([['DvTv_0','DvTr_0','DvTv_3','DvTr_3'],['DvTv_0','DrTv_0','DvTv_0','DrTv_3']]):
			plt.figure(figsize = (30,10))
			plt_idx = 1
			diff = []
			for b, cnd in enumerate(cnds):
				if b % 2 == 0:
					ax = plt.subplot(1,2, plt_idx, ylim = (0.10,0.3)) 
					plt_idx += 1

				# get data to plot
				if a == 0:
					bdm = bdm_T
					colors = ['blue'] * 2 + ['green'] * 2
				elif a == 1:
					bdm = bdm_D
					colors = ['blue'] * 2 + ['red'] * 2	
				#diag = np.stack([np.diag(bdm[i][cnd]['standard']) for i in range(len(bdm))])
				diag = np.stack([bdm[i][cnd]['standard'] for i in range(len(bdm))])
				diff.append(diag)
				plt.plot(times, diag.mean(axis = 0), label = cnd, color = colors[b], ls = ['-','--','-','--'][b])
				
				if b % 2 == 1:
					plt.legend(loc = 'best')
					self.beautifyPlot(y = 0)
					# cluster based permutation
					self.clusterPlot(diff[-2], diff[-1], 0.05, times, ax.get_ylim()[0] + 0.05, color = 'black')
				
				self.beautifyPlot(y = 1/6.0)
			sns.despine(offset=50, trim = False)
				
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['bdm','MS-plots'], filename = 'main_{}-post.pdf'.format(['target','dist'][a])))
			plt.close()

	def bdmSelection(self, header, cmp_name, cmp_window):
		'''

		'''	

		# general parameters
		if header == 'target_loc':
			blocks =  ['DvTv_','DvTr_']
		elif header == 'dist_loc':	
			blocks =  ['DvTv_','DrTv_']	

		with open(self.FolderTracker(['bdm','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		files = glob.glob(self.FolderTracker(['bdm', header], filename = 'class_*_perm-False-post.pickle'))
		bdm = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm.append(pickle.load(handle))

		# select time indices
		s, e = [np.argmin(abs(times - t)) for t in cmp_window]	
		# store data in array
		data = []
		headers = []
		for block in blocks:
			for rep in ['0','3']:
				headers.append('{}{}'.format(block, rep))
				#data.append(np.mean(np.stack([np.diag(bdm[i][block+rep]['standard']) for i in range(len(bdm))])[:,s:e], axis = 1))
				data.append(np.mean(np.stack([bdm[i][block+rep]['standard'] for i in range(len(bdm))])[:,s:e], axis = 1))
				
		X = np.vstack(data).T	
		# save as .csv file
		np.savetxt(self.FolderTracker(['bdm',header], filename = '{}.csv'.format(cmp_name)), X, delimiter = "," ,header = ",".join(headers), comments='')


	# CTF ANALYSIS
	def ctfSlopes(self):
		'''
		CTF ALPHA ANALYSIS as preregistred at https://osf.io/4bx7y/
		Contrasts first (left subplot) and final repetition (right subplot) across conditions
		'''
	
		# read in data (and shift times such that stimulus onset is at 0ms)
		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'alpha_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		# read in CTF slopes tuned to the target and the distractor
		ctfs = {'target_loc':[],'dist_loc':[]}
		for loc in ['target_loc', 'dist_loc']:
			files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',loc], filename = 'cnds_*_slopes-Foster_alpha.pickle'))
			for file in files:
				with open(file ,'rb') as handle:
					ctfs[loc].append(pickle.load(handle))
	
		# plot repetition effect (first repetition on the left, final repetition on the right)
		for a, (power, sl_name) in enumerate(zip(['total', 'evoked'],['T_slopes','E_slopes'])):
			for b, cnds in enumerate([['DvTv_0','DvTr_0','DvTv_0','DvTr_3'],['DvTv_0','DrTv_0','DvTv_3','DrTv_3']]):
				plt.figure(figsize = (30,10))
				# get data to plot
				if b == 0:
					ctf = ctfs['target_loc']
					colors = ['blue', 'green'] * 2
				elif b == 1:
					ctf = ctfs['dist_loc']
					colors = ['blue', 'red'] * 2 

				plt_idx = 1
				diff = []
				for c, cnd in enumerate(cnds):
					if c % 2 == 0:
						ax = plt.subplot(1,2, plt_idx, ylim = [[(-0.05,0.1),(-0.2,0.1)][b],(-0.05,0.2)][a]) 
						yrange = np.diff(plt.ylim())[0]/30
						plt_idx += 1

					slopes = np.squeeze(np.stack([ctf[i][cnd][sl_name] for i in range(len(ctf))]))
					diff.append(slopes)
					err, slopes = bootstrap(slopes)
					plt.plot(times, slopes, label = cnd, color = colors[c], ls = ['--','--','-','-'][c])
					plt.fill_between(times, slopes + err, slopes - err, alpha = 0.2, color = colors[c])		
					self.clusterPlot(diff[-1], 0, 0.05, times, ax.get_ylim()[0] + (1 + c%2) * yrange, color = colors[c], ls = ['--','--','-','-'][c])		
					
					if c % 2 == 1:
						# contrast first and final repetition across conditions
						plt.legend(loc = 'best')
						self.beautifyPlot(y = 0, ylabel = 'CTF slope')
						# cluster based permutation
						self.clusterPlot(diff[-2], diff[-1], 0.05, times, ax.get_ylim()[0] + yrange*3, color = 'black')

					if c == 3:	
						# baseline comparison
						self.clusterPlot(diff[-4] - diff[-3], diff[-2] - diff[-1], 0.05, times, ax.get_ylim()[0] + yrange*4, color = 'grey')
					
				sns.despine(offset=50, trim = False)
					
				# save figures
				plt.tight_layout()
				plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'ctf-alpha_{}_{}.pdf'.format(power, ['target','dist'][b])))
				plt.close()	

	def ctfCrossTrain(self):
		'''
		Exploratory CTF analysis examaning possible negative tuning
		'''

		# read in data
		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'alpha_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		# plot 
		norm = MidpointNormalize(midpoint=0)
		for header in ['dist_loc','target_loc']:
			plt.figure(figsize = (30,15))
			for a, cnd in enumerate(['DvTv','DrTv','DvTr']):
				print cnd
				ax = plt.subplot(2,3, a + 1, title = cnd, xlabel = 'Time (ms)', ylim = (-0.1,0.1)) 
		
				files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',header], filename = '*_cross-training_{}-perm_500.pickle'.format(cnd)))
				ctf = []
				for file in files:
					with open(file ,'rb') as handle:
						ctf.append(pickle.load(handle))

				X_r = np.stack([np.squeeze(ctf[i][cnd+'_3']['slopes']) for i in range(len(ctf))])
				#X_p = np.stack([np.squeeze(ctf[i][cnd+'_3']['slopes_p']) for i in range(len(ctf))])
				#p_val, sig = permTTest(X_r, X_p, p_thresh = 0.01)
				
				
				diag = np.stack([np.diag(x) for x in X_r])
				plt.plot(times, np.mean(diag, axis = 0))
				self.clusterPlot(diag, 0, 0.05, times, ax.get_ylim()[0] + 0.01, color = 'black')
				self.beautifyPlot(y = 0, ylabel = 'CTF slope')
				
				X = X_r.mean(axis = 0)
				p_val, t = permutationTTest(X_r,0,1000)
				X_r[:,p_val > 0.05] = 0
				cl_p_vals = clusterBasedPermutation(X_r,0)
				X[cl_p_vals == 1] = 0
				ax = plt.subplot(2,3, a + 4, title = cnd, xlabel = 'Test Time (ms)', ylabel = 'Train Time (ms)' ) 
				plt.imshow(X, norm = norm, cmap = cm.bwr, interpolation='none', aspect='auto', 
						  origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], vmin = -0.1, vmax = 0.2)
				plt.colorbar()

				sns.despine(offset=50, trim = False)
					
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'cross-train_{}.pdf'.format([header])))
			plt.close()	

	def ctfallFreqs(self):
		'''

		'''

		power = 'evoked'
		if power == 'total':
			sl_name = 'T_slopes'
		elif power == 'evoked':
			sl_name = 'E_slopes'	

		# read in data
		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'alpha_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'cnds_*_slopes-Foster_alpha.pickle'))
		ctf_T = []
		for file in files:
			print file
			with open(file ,'rb') as handle:
				ctf_T.append(pickle.load(handle))

		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye','dist_loc'], filename = 'cnds_*_slopes-Foster_alpha.pickle'))
		ctf_D = []
		for file in files:
			with open(file ,'rb') as handle:
				ctf_D.append(pickle.load(handle))

		# step 1
		for a, cnds in enumerate([['DvTv_0','DvTv_3','DvTr_0','DvTr_3'],['DvTv_0','DvTv_3','DrTv_0','DrTv_3']]):
			plt.figure(figsize = (30,10))
			plt_idx = 1
			diff = []
			for b, cnd in enumerate(cnds):
				ax = plt.subplot(2,2, plt_idx, title = cnd) 
				plt_idx += 1

				# get data to plot
				if a == 0:
					ctf = ctf_T
				elif a == 1:
					ctf = ctf_D
				slopes = np.stack([ctf[i][cnd][sl_name] for i in range(len(ctf))])
				diff.append(slopes)
				plt.imshow(slopes.mean(axis = 0), aspect = 'auto', interpolation = None, origin = 'lower', 
										cmap = cm.viridis, extent = [times[0],times[-1],4,32], vmin = 0, vmax = 0.2)
				plt.colorbar()
				
			sns.despine(offset=50, trim = False)
				
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'all-freqs-ctf_{}_{}.pdf'.format(power, ['target','dist'][a])))
			plt.close()	

	def ctfCrossTrainold(self):
		'''

		'''

		# read in data
		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'alpha_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = '*_cross-training_ind-perm_0.pickle'))
		ctf_T = []
		for file in files:
			with open(file ,'rb') as handle:
				ctf_T.append(pickle.load(handle))

		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye','dist_loc'], filename = '*_cross-training_ind-perm_0.pickle'))
		ctf_D = []
		for file in files:
			with open(file ,'rb') as handle:
				ctf_D.append(pickle.load(handle))

		# step 1: analyze diagnal 
		norm = MidpointNormalize(midpoint=0)
		for a, cnds in enumerate([['DvTv_3','DvTr_3'],['DvTv_3','DrTv_3']]):
			plt.figure(figsize = (30,30))

			for b, cnd in enumerate(cnds):
				ax = plt.subplot(2,2, b + 1, title = cnd, xlabel = 'Time (ms)', ylim  = (-0.2,0.2)) 
	
				# get data to plot
				if a == 0:
					ctf = ctf_T
				elif a == 1:
					ctf = ctf_D

				X_r = np.stack([np.squeeze(ctf[i][cnd]['slopes']) for i in range(len(ctf))])
				X_p = np.stack([np.squeeze(ctf[i][cnd]['slopes_p']) for i in range(len(ctf))])
				p_val, sig = permTTest(X_r, X_p, p_thresh = 0.01)

				diag = np.stack([np.diag(x) for x in X_r])
				plt.plot(times, np.mean(diag, axis = 0))
				self.clusterPlot(diag, 0, 0.05, times, ax.get_ylim()[0] + 0.01, color = 'black')
				self.beautifyPlot(y = 0, ylabel = 'CTF slope')
				

				X = X_r.mean(axis = 0)
				#p_val, t = permutationTTest(X_r,0,1000)
				#X_r[:,p_val > 0.05] = 0
				#cl_p_vals = clusterBasedPermutation(X_r,0)
				#X[cl_p_vals == 1] = 0
				
				X[sig == 0] = 0
				ax = plt.subplot(2,2, b + 3, title = cnd, xlabel = 'Test Time (ms)', ylabel = 'Train Time (ms)') 
				plt.imshow(X, norm = norm, cmap = cm.bwr, interpolation='none', aspect='auto', 
							   origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], vmin = -0.1, vmax = 0.1)
				plt.colorbar()

			sns.despine(offset=50, trim = False)
				
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'cross-train_{}.pdf'.format(['target','dist'][a])))
			plt.close()	




if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '3' 
	os.environ['NUMEXP_NUM_THREADS'] = '3'
	os.environ['OMP_NUM_THREADS'] = '3'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/DT_reps'
	os.chdir(project_folder)
	PO = DT_reps()

	#preprocessing and main analysis
	for sj in range(1,25):

		for header in ['dist_loc']:

			if header == 'target_loc':
				midline = {'dist_loc': [0,3]}
				cnds = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
			elif header == 'dist_loc':
				midline = {'target_loc': [0,3]}
				cnds = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']	
	
			# ERP analysis
			erp = ERP(header = header, baseline = [-0.3, 0], eye = True)
			erp.selectERPData(sj = sj, time = [-0.3, 0.8], l_filter = 30) 
			# erp.ipsiContra(sj = sj, left = [2,3], right = [4,5], l_elec = ['PO7','PO3','O1','P3','P5','P7'], 
			# 				r_elec = ['PO8','PO4','O2','P4','P6','P8'], midline = midline, balance = False, erp_name = 'main-unbalanced')
			# erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1','P3','P5','P7'], 
			# 				r_elec = ['PO8','PO4','O2','P4','P6','P8'], midline = midline, balance = True, erp_name = 'main-low')
			# erp.topoFlip(left = [1, 2])
			# erp.topoSelection(sj = sj, loc = [2,4], midline = midline, topo_name = 'main', balance = True)
			erp.topoSelection(sj = sj, loc = [0,1,2,3,4,5], topo_name = 'frontal-bias', balance = True)

			# BDM analysis
			#bdm = BDM(decoding = header, nr_folds = 10, eye = True, elec_oi = 'post', downsample = 128, bdm_filter = dict(alpha = (8,12)))
			#bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.3, 0.8), bdm_matrix = False)

			# # CTF analysis
			#ctf = CTF('all_channels_no-eye', header, nr_iter = 10, nr_blocks = 3, nr_bins = 6, nr_chans = 6, delta = False)
			#ctf.spatialCTF(sj, [-0.3, 0.8], cnds, method = 'Foster', freqs = dict(alpha = [8,12]), downsample = 4, nr_perm = 0, plot = False)
			#ctf.crosstrainCTF(sj, [-0.3, 0.8], train_cnds = ['DvTv_0'], test_cnds = ['DvTv_3'], 
			#				freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = True, nr_perm = 500, name = 'DvTv-perm_500')
			# ctf.crosstrainCTF(sj, [-0.3, 0.8], train_cnds = ['DrTv_0'], test_cnds = ['DrTv_3'], 
			#  			 freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = True, nr_perm = 500, name = 'DrTv-perm_500')
			# ctf.crosstrainCTF(sj, [-0.3, 0.8], train_cnds = ['DvTr_0'], test_cnds = ['DvTr_3'], 
			#  			 freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = True, nr_perm = 500, name = 'DvTr-perm_500')
			#ctf.spatialCTF(sj, [-300, 800], 500, cnds, freqs = dict(all = [4,30]), downsample = 4)
	# analysis manuscript
	# BEH
	#PO.BEHexp1()

	# ERP 
	#for header in ['target_loc', 'dist_loc']:
		#PO.componentSelection(header, cmp_name = 'P1', cmp_window = [0.11,0.15], ext = 0.0175, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.componentSelection(header, cmp_name = 'N1', cmp_window = [0.16,0.22], ext = 0.0175, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.componentSelection(header, cmp_name = 'N2pc', cmp_window = [0.17,0.23], ext = 0, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.componentSelection(header, cmp_name = 'Pd', cmp_window = [0.28,0.35], ext = 0, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.erpLateralized(header, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])

	#PO.erpSelection(header = 'dist_loc', topo_name = 'main', elec = ['PO3','PO7','O1'])
	#PO.erpSelection(header = 'target_loc', topo_name = 'main', elec = ['PO3','PO7','O1'])
	#PO.erpContrast(erp_name = 'lat-down1-mid', elec = ['PO7','PO3','O1'])
	#PO.frontalBias()

	# PO.componentSelection(header = 'dist_loc', erp_name = 'lat-down1-mid', elec = ['PO7','PO3','O1'], 
	# 						cmpnts = dict(N2pc = (0.2, 0.3), Pd = (0.25, 0.4)))

	# BDM
	#PO.bdmAcc()
	#PO.bdmDiag()
	#for header in ['target_loc', 'dist_loc']:
	#		PO.bdmSelection(header, 'Pd', (0.28,0.35))

	# CTF
	#PO.ctfSlopes()
	#PO.ctfCrossTrainold()
	#PO.ctfallFreqs()


