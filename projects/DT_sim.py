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
			'2': {'tracker': (False,'', ''), 'replace':{}},
			'3': {'tracker': (False, '', ''), 'replace':{}},
			'4': {'tracker': (True, 'asc', 500,'',''), 'replace':{}},
			'5': {'tracker': (True, 'asc', 500), 'replace':{}}}

# project specific info

project = 'DT_sim'
part = 'beh'
factors = ['block_type','dist_high']
labels = [['DTsim','DTdisP','DTdisDP'],['yes','no']]
to_filter = ['RT'] 
project_param = ['practice','nr_trials','trigger','condition','RT', 'subject_nr',
				'block_type', 'correct','dist_high','dist_loc','dist_orient',
		         'dist_type','high_loc', 'target_high','target_loc','target_type']

montage = mne.channels.read_montage(kind='biosemi64')
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
trigger = [12,13,14,15,16,21,23,24,25,26,31,32,34,35,36,41,42,43,45,46,51,52,53,54,56,61,62,63,64,65,101,102,103,104,105,106]
t_min = -0.75
t_max = 0.55
flt_pad = 0.5
eeg_runs = [1]
binary =  61440


# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class DT_sim(FolderStructure):

	def __init__(self): pass


	def prepareBEH(self, project, part, factors, labels, project_param):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and correct == 1', cnd_sel = False, save = True)
		PP.exclude_outliers(criteria = dict(RT = 'RT_filter == True', correct = ''))
		PP.prep_JASP(agg_func = 'mean', voi = 'RT', data_filter = 'RT_filter == True', save = True)
		PP.save_data_file()

	def countCndCheck(self):
		'''
		Checks min condition number per cnd after preprocessing
		'''

		# loop across all processed behavior files
		files = glob.glob(self.FolderTracker(['beh', 'processed'], filename = 'subject-*_all.pickle'))
		for file in files:
			# open file
			with open(file, 'rb') as handle:
				beh = pickle.load(handle)

			# get minimum condition number
			cnd_info = np.unique(beh['condition'], return_counts = True)	
			cnd_sort = np.argsort(cnd_info)[1]

			if 'no' in cnd_info[0][cnd_sort[0]]:
				max_trial = 756.0
			elif 'yes' in cnd_info[0][cnd_sort[0]]:
				max_trial = 1260.0	

			# calculate percentage of min condition
			min_p =  cnd_info[1][cnd_sort[0]]/max_trial * 100
			
			if min_p < 75:
				print min_p, cnd_info[0][cnd_sort[0]], file

	def mainBEH(self, exp = 'beh', column = 'dist_high', ylim = (350,750)):
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
		f = plt.figure(figsize = (30,10))

		#levels = np.unique(pivot.keys().get_level_values('block_type'))
		levels = ['DTsim','DTdisDP','DTdisP']
		for idx, block in enumerate(levels):
			
			# get effect and p-value
			diff = pivot[block]['yes'].mean() -  pivot[block]['no'].mean()
			t, p = ttest_rel(pivot[block]['yes'], pivot[block]['no'])

			ax = plt.subplot(1,3, idx + 1, title = '{0}: \ndiff = {1:.0f}, p = {2:.3f}'.format(block, diff, p), ylabel = 'RT (ms)', ylim = ylim)
			df = pd.melt(pivot[block], value_name = 'RT (ms)')
			df['sj'] = range(pivot.index.size) * 2
			ax = sns.stripplot(x = column, y = 'RT (ms)', data = df, hue = 'sj', size = 10, jitter = True)
			ax.legend_.remove()
			sns.violinplot(x = column, y = 'RT (ms)', data = df, color= 'white', cut = 1)

			sns.despine(offset=50, trim = False)
		
		f.subplots_adjust(wspace=50)
		plt.tight_layout()
		plt.savefig(self.FolderTracker([exp,'figs'], filename = 'block_effect_{}.pdf'.format(column)))		
		plt.close()	

	def singleTarget(self):
		'''

		'''

		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		DF = pd.read_csv(file)

		# creat pivot (with RT filtered data) 
		DF = DF.query("RT_filter == True")
		DF = DF[DF['dist_loc'] == 'None']

		# read in single target data
		T_pivot = DF.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','target_high'], aggfunc = 'mean')
		T_error = pd.Series(confidence_int(T_pivot.values), index = T_pivot.keys())

		# read in distractor data
		DF = pd.read_csv(file)
		DF = DF.query("RT_filter == True")
		DF = DF[DF['target_high'] == 'no']
		D_pivot = DF.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','dist_high'], aggfunc = 'mean')
		D_error = pd.Series(confidence_int(D_pivot.values), index = D_pivot.keys())

		# create line plot (DTsim, DTdisDP, DTdisP)
		blocks = ['DTsim','DTdisP']

		# show effects
		f = plt.figure(figsize = (20,10))
		
		# plot 
		ax =  plt.subplot(1,2, 2, title = 'Single target trials', ylabel = 'RT (ms)', ylim = (390,430),xlim = (-0.25,1.25))
		plt.xticks((0,1), blocks)
		plt.plot((0,1),[T_pivot.mean()[bl]['no'] for bl in blocks], color = 'green')
		plt.plot((0,1),[T_pivot.mean()[bl]['yes'] for bl in blocks], color = 'red')
		plt.errorbar((0,1),[T_pivot.mean()[bl]['no'] for bl in blocks], 
							yerr = [T_error[bl]['no'] for bl in blocks], 
							fmt = 'o', label = 'single target - low', color = 'green')
		plt.errorbar((0,1),[T_pivot.mean()[bl]['yes'] for bl in blocks], 
							yerr = [T_error[bl]['yes'] for bl in blocks], 
							fmt = 's', label = 'single target - high', color = 'red')

		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)

		ax =  plt.subplot(1,2, 1, title = 'Distractor trials', ylabel = 'RT (ms)', ylim = (410,470),xlim = (-0.25,1.25))
		plt.xticks((0,1), blocks)
		plt.plot((0,1),[D_pivot.mean()[bl]['no'] for bl in blocks], color = 'green')
		plt.plot((0,1),[D_pivot.mean()[bl]['yes'] for bl in blocks], color = 'red')
		plt.errorbar((0,1),[D_pivot.mean()[bl]['no'] for bl in blocks], 
							yerr = [D_error[bl]['no'] for bl in blocks], 
							fmt = 'o', label = 'dist - low', color = 'green')
		plt.errorbar((0,1),[D_pivot.mean()[bl]['yes'] for bl in blocks], 
							yerr = [D_error[bl]['yes'] for bl in blocks], 
							fmt = 's', label = 'dist - high', color = 'red')

		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)

		plt.tight_layout()
		f.subplots_adjust(wspace=0.5)
		plt.savefig(self.FolderTracker(['beh','figs'], filename = 'single-target.pdf'))		
		plt.close()


	def beautifyPlot(self, y = 0, xlabel = 'Time (ms)', ylabel = 'Mv'):
		'''
		Adds markers to the current plot. Onset placeholder and onset search and horizontal axis
		'''

		plt.axhline(y=y, ls = '-', color = 'black')
		plt.axvline(x=-0.25, ls = '-', color = 'black')
		plt.axvline(x=0, ls = '-', color = 'black')
		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

	def clusterPlot(self, X1, X2, p_val, times, y, color):
		'''
		plots significant clusters in the current plot
		'''	

		# indicate significant clusters of individual timecourses
		sig_cl = clusterBasedPermutation(X1, X2, p_val = 0.05)
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size) * y, color = color, ls = '--')


	def plotERP(self):
		'''
		plot ipsi and contra and difference waves

		'''

		# read in cda data
		with open(self.FolderTracker(['erp','dist_loc'], filename = 'main.pickle') ,'rb') as handle:
			erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		# ipsi and contra plots
		diff_waves = {}
		plt.figure(figsize = (30,20))
		for idx, cnd in enumerate(['DTsim-no','DTsim-yes','DTdisDP-no', 'DTdisDP-yes','DTdisP-no','DTdisP-yes']):
			ax =  plt.subplot(3,2, idx + 1, title = cnd, ylabel = 'mV', xlabel = 'time (ms)')
			ipsi = np.mean(np.stack([erp[e][cnd]['ipsi'] for e in erp]), axis = 1)
			contra = np.mean(np.stack([erp[e][cnd]['contra'] for e in erp]), axis = 1)
			block, supp = cnd.split('-')
			if block not in diff_waves.keys():
				diff_waves.update({block:{}})
			diff_waves[block][supp] = contra - ipsi	

			err_i, ipsi  = bootstrap(ipsi)	
			err_c, contra  = bootstrap(contra)	

			plt.ylim(-8,8)
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

		# difference waves
		f = plt.figure(figsize = (30,10))
		for idx, block in enumerate(diff_waves.keys()):
			ax =  plt.subplot(1,3, idx + 1, title = block, ylim = (-5,3))
			to_test = []
			for i, supp in enumerate(diff_waves[block].keys()):
				to_test.append(diff_waves[block][supp])
				err_d, diff = bootstrap(diff_waves[block][supp])
				plt.plot(info['times'], diff, label = supp, color = ['red', 'green'][i])
				plt.fill_between(info['times'], diff + err_d, diff - err_d, alpha = 0.2, color = ['red', 'green'][i])
				self.clusterPlot(to_test[-1], 0, p_val = 0.05, times = info['times'], y = (-5 + 0.2*i), color = ['red', 'green'][i])

			self.clusterPlot(to_test[0], to_test[1], p_val = 0.05, times = info['times'], y = (-5 + 0.2*2), color = 'black')
			self.beautifyPlot(y = 0, xlabel = 'Time (ms)', ylabel = 'mV')	
			plt.legend(loc = 'best')
			sns.despine(offset=50, trim = False)
		f.subplots_adjust(wspace=50)
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','dist_loc'], filename = 'diffwaves.pdf'))
		plt.close()


	def plotBDM(self, header):
		'''
		
		'''

		# read in data
		with open(self.FolderTracker(['bdm','{}_type'.format(header)], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	

		files = glob.glob(self.FolderTracker(['bdm', '{}_type'.format(header)], filename = 'class_*_perm-False.pickle'))
		bdm = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm.append(pickle.load(handle))	

		# plot diagonal decoding (all conditions in one plot)
		plt.figure(figsize = (15,10))	
		diff = []
		for i, cnd in enumerate(['DTsim', 'DTdisP', 'DTdisDP']):
			ax = plt.subplot(1,1 ,1, ylim = (0.3, 0.45))
			X = np.stack([bdm[j][cnd]['standard'] for j in range(len(bdm))])
			X_diag = np.stack([np.diag(x) for x in X])
			diff.append(X_diag)
			self.clusterPlot(X_diag, 1/3.0, p_val = 0.05, times = times, y = plt.ylim()[0] + i * 0.002, color = ['red','purple', 'darkblue'][i])	
			err_diag, X_diag  = bootstrap(X_diag)	
			plt.plot(times, X_diag, label = cnd, color = ['red','purple', 'darkblue'][i])
			plt.fill_between(info['times'], X_diag + err_diag, X_diag - err_diag, alpha = 0.2, color = ['red','purple', 'darkblue'][i])
			
		self.clusterPlot(diff[-1], diff[-2], p_val = 0.05, times = times, y = plt.ylim()[0] + 3 * 0.002, color = 'black')	
		plt.legend(loc = 'best')
		self.beautifyPlot(y = 1/3.0, xlabel = 'Time (ms)', ylabel = 'Decoding accuracy (%)')	
		sns.despine(offset=50, trim = False)
		plt.tight_layout()			
		plt.savefig(self.FolderTracker(['bdm','{}_type'.format(header)], filename = 'cnd-dec.pdf'))
		plt.close()	


		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/3.0)
		plt_idx = 1
		for i, cnd in enumerate(['DTsim', 'DTdisP', 'DTdisDP']):
			for plot in ['matrix', 'diag']:
				ax = plt.subplot(3,2 , plt_idx, title = cnd)#, title = cnd, ylabel = 'Time (ms)', xlabel = 'Time (ms)')
				X = np.stack([bdm[j][cnd]['standard'] for j in range(len(bdm))])
				p_vals = signedRankArray(X, 1/3.0)
				h,_,_,_ = FDR(p_vals)
				dec = np.mean(X,0)
				dec[~h] = 1/3.0
				X_diag = np.stack([np.diag(x) for x in X])
				if plot == 'matrix':
					plt.imshow(dec, norm = norm, cmap = cm.bwr, interpolation='none', aspect='auto', 
							   origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
							   vmin = 0.3, vmax = 0.40)
					plt.colorbar()
				elif  plot == 'diag':
					plt.ylim(0.3,0.45)
					self.clusterPlot(X_diag, 1/3.0, p_val = 0.05, times = times, y = (0.31), color = 'blue')
					err_diag, X_diag  = bootstrap(X_diag)	
					plt.plot(times, X_diag)	
					plt.fill_between(times, X_diag + err_diag, X_diag - err_diag, alpha = 0.2)
					plt.axhline(y = 0.33, color = 'black', ls = '--')
					plt.axvline(x = -0.2, color = 'black', ls = '--')
					plt.axvline(x = 0, color = 'black', ls = '--')
					sns.despine(offset=50, trim = False)
				plt_idx += 1	
			
		plt.tight_layout()			
		plt.savefig(self.FolderTracker(['bdm','{}_type'.format(header)], filename = 'dec.pdf'))
		plt.close()	

	def plotTF(self, c_elec, i_elec, method = 'hilbert'):


		with open(self.FolderTracker(['tf', method], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		times = info['times']	

		time = (-0.05, 0)
		s, e = [np.argmin(abs(info['times'] - t)) for t in time]
		files = glob.glob(self.FolderTracker(['tf', method], filename = '*-tf.pickle'))
		print files
		tf = []
		for file in files:
			with open(file ,'rb') as handle:
				tf.append(pickle.load(handle))	

		contra_idx = [info['ch_names'].index(e) for e in c_elec]
		ipsi_idx = [info['ch_names'].index(e) for e in i_elec]

		plt.figure(figsize = (30,10))
		for plt_idx, cnd in enumerate(['DTsim','DTdisP','DTdisDP']):
			ax = plt.subplot(1,3 , plt_idx + 1, title = cnd, xlabel = 'time (ms)', ylabel = 'freq')
			contra = np.stack([np.mean(tf[i][cnd]['base_power'][:,contra_idx,:], axis = 1) for i in range(len(tf))])
			ipsi = np.stack([np.mean(tf[i][cnd]['base_power'][:,ipsi_idx,:], axis = 1) for i in range(len(tf))])
			X = contra - ipsi
			#sig = clusterBasedPermutation(X,0)
			X = X.mean(axis = 0)
			#print X[:,s:e].mean(axis = 1)
			#X[sig == 1] = 0
			plt.imshow(X, cmap = cm.jet, interpolation='none', aspect='auto', 
							   origin = 'lower', extent=[times[0],times[-1],5,40],
							   vmin = -0.1, vmax = 0.1)
			
			plt.yticks(info['frex'][::3])
			plt.colorbar()

		plt.tight_layout()			
		plt.savefig(self.FolderTracker(['tf','figs'], filename = 'tf-main-basetest.pdf'))
		plt.close()	

		embed()

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '2' 
	os.environ['NUMEXP_NUM_THREADS'] = '2'
	os.environ['OMP_NUM_THREADS'] = '2'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/DT_sim'
	os.chdir(project_folder)

	# initiate current project
	PO = DT_sim()

	# analyze behavior
	#PO.prepareBEH(project, part, factors, labels, project_param)
	#PO.mainBEH(exp = 'beh', column = 'dist_high',  ylim = (200,600))
	#PO.singleTarget()
	#PO.mainBEH(exp = 'exp_2', column = 'target_high',  ylim = (350,900))

	# analyze eeg
	#PO.countCndCheck()
	#PO.plotERP()
	#PO.plotBDM(header = 'target')
	#PO.plotBDM(header = 'dist')
	PO.plotTF(c_elec = ['PO7','PO3','O1'], i_elec= ['PO8','PO4','O2'], method = 'wavelet')


	# run preprocessing
	for sj in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
	
	# 	preprocessing(sj = sj, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 			  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 			  trigger = trigger, project_param = project_param, 
	# 			  project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)

	#  	#TF analysis
	 	tf = TF()
	 	tf.TFanalysis(sj = sj, cnds = ['DTsim','DTdisP','DTdisDP'], 
	 			  cnd_header ='block_type', base_period = (-0.8,-0.6), 
	 			  time_period = (-0.6,0.5), method = 'wavelet', flip = dict(high_prob = 'left'), downsample = 4)

	# 	# ERP analysis
	# 	erp = ERP(header = 'dist_loc', baseline = [-0.45,-0.25], eye = False)
	# 	erp.selectERPData(sj = sj, time = [-0.45, 0.55], l_filter = 40) 
	# 	erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1'], 
	# 									r_elec = ['PO8','PO4','O2'], midline = {'target_loc': [0,3]}, balance = False, erp_name = 'main')
	# 	erp.topoFlip(left = [2])
	# 	erp.topoSelection(sj = sj, loc = [2,4], midline = {'target_loc': [0,3]}, topo_name = 'main')

	# 	# BDM analysis
	# 	# feature decoding (dist)
	# 	bdm = BDM('all_channels', 'dist_type', nr_folds = 10, eye = False)
	# 	bdm.Classify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', subset = None, time = (-0.45, 0.55), nr_perm = 0, bdm_matrix = True)

	# 	# feature decoding (target)
	# 	bdm = BDM('all_channels', 'target_type', nr_folds = 10, eye = False)
	# 	bdm.Classify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', subset = None, time = (-0.45, 0.55), nr_perm = 0, bdm_matrix = True)
	
	#	# location decoding (dist_loc)
	#	bdm = BDM('dist_loc', nr_folds = 10, eye = False, elec_oi = 'all', downsample = 128, bdm_filter = None)
	#	bdm.Classify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', 
	#				bdm_labels = ['0','1','2','3','4','5'], time = (-0.45, 0.55), nr_perm = 0, bdm_matrix = True)
