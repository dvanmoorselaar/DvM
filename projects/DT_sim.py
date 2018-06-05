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

from IPython import embed
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import BDM 
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
trigger = [12,13,14,15,16,21,23,24,25,26,31,32,34,35,36,41,42,43,45,46,51,52,53,54,56,61,62,63,64,65,101,102,103,104,105,106]
t_min = -0.75
t_max = 0.55
flt_pad = 0.5
eeg_runs = [1]
binary =  61440
project_param = ['practice','nr_trials','trigger','condition','RT',
				'block_type', 'correct','dist_high','dist_loc','dist_orient',
		         'dist_type','high_loc', 'target_high','target_loc','target_type']

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class DT_sim(FolderStructure):

	def __init__(self): pass

	def plotERP(self):
		'''

		'''

		# read in cda data
		with open(self.FolderTracker(['erp','dist_loc'], filename = 'main.pickle') ,'rb') as handle:
			erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		plt.figure(figsize = (30,20))
		for idx, cnd in enumerate(['DTsim-no','DTsim-yes','DTdisDP-no', 'DTdisDP-yes','DTdisP-no','DTdisP-yes']):
			ax =  plt.subplot(3,2, idx + 1, title = cnd, ylabel = 'mV', xlabel = 'time (ms)')
			ipsi = np.squeeze(np.stack([erp[e][cnd]['ipsi'] for e in erp]))
			contra = np.squeeze(np.stack([erp[e][cnd]['contra'] for e in erp]))
			err_i, ipsi  = bootstrap(ipsi)	
			err_c, contra  = bootstrap(contra)	

			#plt.ylim(-8,8)
			plt.axhline(y = 0, color = 'black')
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'red')
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], ipsi + err_i, ipsi - err_i, alpha = 0.2, color = 'red')	
			plt.fill_between(info['times'], contra + err_c, contra - err_c, alpha = 0.2, color = 'green')	

			plt.legend(loc = 'best')
			sns.despine(offset=50, trim = False)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','dist_loc'], filename = 'ipsi-contra.pdf'))
		plt.close()

	def plotBDM(self, header, cnds):
		'''
		
		'''

		# read in data
		with open(self.FolderTracker(['bdm','{}_type'.format(header)], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		files = glob.glob(self.FolderTracker(['bdm', '{}_type'.format(header)], filename = 'class_*_perm-False.pickle'))
		bdm = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm.append(pickle.load(handle))	

		plt.figure(figsize = (10,10))
		for i, cnd in enumerate(cnds):

			#ax = plt.subplot(2,3 , plt_idx[i])#, title = cnd, ylabel = 'Time (ms)', xlabel = 'Time (ms)')
			plt.tick_params(direction = 'in', length = 5)
			X = np.mean(np.stack([bdm[j][cnd]['standard'] for j in range(len(bdm))]), axis = 0)
			plt.plot(info['times'], X, label = cnd)

		plt.legend(loc = 'best')
		plt.savefig(self.FolderTracker(['bdm','{}_type'.format(header)], filename = 'dec.pdf'))
		plt.close()	


if __name__ == '__main__':
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/DT_sim'
	os.chdir(project_folder)

	# run preprocessing
	sj = 2
	# preprocessing(sj = 5, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 			  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 			  trigger = trigger, project_param = project_param, 
	# 			  project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)

	# ERP analysis
	# erp = ERP(header = 'dist_loc', baseline = [-0.45,-0.25])
	# erp.selectERPData(sj = sj, time = [-0.45, 0.55], l_filter = 40) 
	# erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7'], 
	# 								r_elec = ['PO8'], midline = {'target_loc': [0,3]}, balance = False, erp_name = 'main')
	# erp.topoFlip(left = [2])
	# erp.topoSelection(sj = sj, loc = [2,4], midline = {'target_loc': [0,3]}, topo_name = 'main')

	# BDM analysis
	#BDM = BDM('all_channels','target_type', nr_folds = 10)
	#for sj in [4,7]:
	#	BDM.Classify(sj, ['DTsim','DTdisDP','DTdisP'], 'block_type', time = (-0.5, 0.6), nr_perm = 0, bdm_matrix = False)


	# plot project analysis
	PO = DT_sim()
	PO.plotERP()
	#PO.plotBDM(header = 'target', cnds = ['DTsim','DTdisDP','DTdisP']) 


