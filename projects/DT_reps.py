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
from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.TF import * 
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
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

	def clusterPlot(self, X1, X2, p_val, times, y, color):
		'''
		plots significant clusters in the current plot
		'''	

		# indicate significant clusters of individual timecourses
		sig_cl = clusterBasedPermutation(X1, X2, p_val = p_val)
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size) * y, color = color, ls = '--')

	def beautifyPlot(self, y = 0, xlabel = 'Time (ms)', ylabel = 'Mv'):
		'''
		Adds markers to the current plot. Onset placeholder and onset search and horizontal axis
		'''

		plt.axhline(y=y, ls = '-', color = 'black')
		plt.axvline(x=-0.25, ls = '-', color = 'black')
		plt.axvline(x=0, ls = '-', color = 'black')
		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

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
			for s, side in enumerate(['ipsi','contra']):
				diff = []
				ax = plt.subplot(1,2, s + 1, ylim = (-7.5, 5.5), title = side) 
				for b, cnd in enumerate(cnds):
					wave = np.mean(np.stack([erp[key][cnd][side] for key in erp.keys()])[:,elec_idx], axis = 1)
					diff.append(wave)
					err, m_wave = bootstrap(wave)
					m_wave = wave.mean(axis = 0)
					plt.plot(times, m_wave, label = cnd, color = ['red','green'][b])
					plt.fill_between(times, m_wave + err, m_wave - err, alpha = 0.2, color = ['red', 'green'][b])
				
				plt.legend(loc = 'best')
				self.clusterPlot(diff[-2], diff[-1], 0.05, times, ax.get_ylim()[0] + 0.05, color = 'black')
				self.beautifyPlot(y = 0)
				sns.despine(offset=50, trim = False)
			
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'main-lat_{}.pdf'.format(['target','dist'][a])))
			plt.close()		

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '2' 
	os.environ['NUMEXP_NUM_THREADS'] = '2'
	os.environ['OMP_NUM_THREADS'] = '2'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/DT_reps'
	os.chdir(project_folder)
	PO = DT_reps()

	#run preprocessing
	for sj in range(1,25):

		for header in ['target_loc','dist_loc']:

			if header == 'target_loc':
				midline = {'dist_loc': [0,3]}
			elif header == 'dist_loc':
				midline = {'target_loc': [0,3]}	
	
			# ERP analysis
			erp = ERP(header = header, baseline = [-0.3, 0], eye = True)
			erp.selectERPData(sj = sj, time = [-0.3, 0.8], l_filter = 30) 
			erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1','P3','P5','P7'], 
							r_elec = ['PO8','PO4','O2','P4','P6','P8'], midline = midline, balance = True, erp_name = 'main_4')
			erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1','P3','P5','P7'], 
							r_elec = ['PO8','PO4','O2','P4','P6','P8'], midline = midline, balance = True, erp_name = 'main_5')
			erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1','P3','P5','P7'], 
							r_elec = ['PO8','PO4','O2','P4','P6','P8'], midline = midline, balance = True, erp_name = 'main_6')
			erp.topoFlip(left = [1, 2])
			erp.topoSelection(sj = sj, loc = [2,4], midline = midline, topo_name = 'main', balance = True)


	# analysis manuscript
	# ERP
	#PO.erpSelection(header = 'dist_loc', topo_name = 'main', elec = ['PO3','PO7','O1'])
	#PO.erpSelection(header = 'target_loc', topo_name = 'main', elec = ['PO3','PO7','O1'])
	PO.erpContrast(erp_name = 'main_4', elec = ['PO3','PO7','O1'])


