import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import os
import mne
import sys
import glob
import pickle
import logging
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

sj_info = {'1': {'tracker': (True, 'tsv', 30, 'Onset task display',0), 'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			'2': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'3': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'4': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'5': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'6': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'7': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'8': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'9': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'10': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'11': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'12': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'13': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'14': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'15': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'16': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'17': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'18': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'19': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'20': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'21': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'22': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'23': {'tracker': (False, 'tsv', 30, 'Onset task display',0), 'replace':{}},
			'24': {'tracker': (True, 'tsv', 30,'Onset task display',0), 'replace':{}}}


# project specific info

project = 'DT_sim'
part = 'beh'
factors = ['block_type','dist_high']
labels = [['DTsim','DTdisP','DTdisDP'],['yes','no']]
to_filter = ['RT'] 
project_param = ['practice','nr_trials','trigger','condition','RT', 'subject_nr',
				'block_type', 'correct','dist_high','dist_loc','dist_orient',
		         'dist_type','high_loc', 'target_high','target_loc','target_type', 'block_cnt']

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


	def prepareBEH(self, project, part, factors, labels, project_param, to_filter):
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

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, channel_plots, inspect):
		'''
		EEG preprocessing as preregistred @ https://osf.io/b2ndy/register/5771ca429ad5a1020de2872e
		'''

		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']
		tracker, ext, t_freq, start_event, shift = sj_info[str(sj)]['tracker']

		# start logging
		logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename= self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocess_sj{}_ses{}.log'.format(
                        sj, session), overwrite = False),
                    filemode='w+')

		logging.info('Started preprocessing subject {}, session {}'.format(sj, session))
		# READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
		EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
		                                   montage=None, preload=True, eog=eog) for run in eeg_runs])

		#EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['V_do','H_l','Ref_r','Ref_l','EXG7','EXG8'])
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEGica = EEG.filter(h_freq=None, l_freq=1,
		                    fir_design='firwin', skip_by_annotation='edge')
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		             skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
		#if sj == 5 and session == 1: # correct for starting eeg recording during practice
		#	events = events[59:,:]

		beh, missing = EEG.matchBeh(sj, session, events, trigger, 
		                             headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		epochs.selectBadChannels(channel_plots = channel_plots, inspect = inspect, RT = None)    
		epochs.artifactDetection(inspect=inspect, run = True)

		# ICA
		epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = inspect)

		# EYE MOVEMENTS
		epochs.detectEye(missing, time_window=(t_min*1000, t_max*1000), tracker = tracker, tracker_shift = shift, start_event = start_event, extension = ext, eye_freq = t_freq)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, trigger)


	def detectSaccades(self, sj):
		'''
		Visual inspection of saccades detected by automatic step algorythm
		'''

		# read in epochs
		beh, eeg = self.loadData(sj, (-0.75,0.55),False)# 'HEOG', 1, eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20))
		front_electr = [eeg.ch_names.index(e) for e in [
			'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'VEOG', 'HEOG']]

		# get eog data of interest
		s, e = [np.argmin(abs(eeg.times - t)) for t in (-0.75,0.55)]
		eog = eeg._data[:,eeg.ch_names.index('HEOG'),s:e]
		eye_idx = eog_filt(eog, eeg.info['sfreq'], windowsize = 200, windowstep = 10, threshold = 25)	


		bad_eogs = eeg[eye_idx]
		idx_bads = bad_eogs.selection

		bad_eogs.plot(block=True, n_epochs=5, n_channels=len(
			front_electr), picks=front_electr, scalings='auto')

		missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eogs.selection])
		eye_idx = np.delete(eye_idx, missing)

		beh['eye_bins'][eye_idx] = 99
		beh = beh.to_dict('list')

		pickle.dump(beh,open(self.FolderTracker(extension=['beh', 'processed'],
            	filename='subject-{}_all.pickle'.format(sj)),'wb'))

	def countCndCheck(self):
		'''
		Checks min condition number per cnd after preprocessing
		'''

		# loop across all subjects
		sj_info = []
		for sj in range(1,25):
			beh, eeg = self.loadData(sj, (-0.75,0.55),True, 'HEOG', 1, eye_dict = None)

			# get minimum condition number
			cnd_info = np.unique(beh['condition'], return_counts = True)	
			cnd_sort = np.argsort(cnd_info)[1]

			if 'no' in cnd_info[0][cnd_sort[0]]:
				max_trial = 756.0
			elif 'yes' in cnd_info[0][cnd_sort[0]]:
				max_trial = 1260.0	

			# calculate percentage of min condition
			min_p =  cnd_info[1][cnd_sort[0]]/max_trial * 100
			
			if min_p < 70:
				sj_info.append((min_p, cnd_info[0][cnd_sort[0]], sj))

		embed()

	def plotExp1_2_3(self, exp = 'exp_1', excl_pos_bias = False):
		'''
		creates a bar plot with individual datapoints overlayed
		'''

		file = self.FolderTracker([exp ,'analysis'], filename = 'preprocessed.csv')
	
		# seperate plots for target and distractor suppression
		for to_plot in ['dist_high', 'target_high']:
			DF = pd.read_csv(file)

			# creat pivot (with RT filtered data)
			DF = DF.query("RT_filter == True")

			if excl_pos_bias:
				if 'dist' in to_plot:
					DF = DF[DF['target_high'] == 'no']
				elif 'target' in to_plot:
					DF = DF[DF['dist_high'] == 'no']	

			pivot = DF.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type',to_plot], aggfunc = 'mean')	
			error = pd.Series(confidence_int(pivot.values), index = pivot.keys())
			# save output to JASP for analysis
			headers = ['-'.join(string) for string in pivot.keys()]
			np.savetxt(self.FolderTracker([exp,'analysis'], filename = 'RTs-JASP-{}.csv'.format(to_plot[:-5])), pivot.values, delimiter = "," ,header = ",".join(headers), comments='')
	
			# plot the seperate conditions (3 X 1 plot design)
			f = plt.figure(figsize = (30,10))
			levels = np.unique(pivot.keys().get_level_values('block_type'))
			for idx, block in enumerate(levels):
				
				# get effect and p-value
				diff = pivot[block]['yes'].mean() -  pivot[block]['no'].mean()
				t, p = ttest_rel(pivot[block]['yes'], pivot[block]['no'])

				ax = plt.subplot(1,3, idx + 1, title = '{0}: \ndiff = {1:.0f}, p = {2:.3f}'.format(block, diff, p), ylabel = 'RT (ms)', ylim = (300,800))
				df = pd.melt(pivot[block], value_name = 'RT (ms)')
				df['sj'] = range(pivot.index.size) * 2
				ax = sns.stripplot(x = to_plot, y = 'RT (ms)', data = df, hue = 'sj', size = 20, jitter = True, edgecolor = 'black', color = 'white', linewidth = 3)
				ax.legend_.remove()
				sns.barplot(x = to_plot, y = 'RT (ms)', data = df, color= 'white')

				sns.despine(offset=50, trim = False)
			
			f.subplots_adjust(wspace=50)
			plt.tight_layout()
			plt.savefig(self.FolderTracker([exp,'figs'], filename = 'main_ana_{}.pdf'.format(to_plot[:-5])))		
			plt.close()	


	def singleTargetEEG(self):
		'''

		'''

		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		DF = pd.read_csv(file)

		# creat pivot (with RT filtered data) 
		DF = DF.query("RT_filter == True")
		DF = DF[DF['dist_loc'] == 'None']

		# read in single target data
		pivot = DF.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','target_high'], aggfunc = 'mean')
		error = pd.Series(confidence_int(T_pivot.values), index = T_pivot.keys())
		levels = np.unique(pivot.keys().get_level_values('block_type'))

		# save output to JASP for analysis
		headers = ['-'.join(string) for string in pivot.keys()]
		np.savetxt(self.FolderTracker([exp,'analysis'], filename = 'RTs-JASP-{}.csv'.format(to_plot[:-5])), pivot.values, delimiter = "," ,header = ",".join(headers), comments='')

		#do actual plotting 
		f = plt.figure(figsize = (20,10))
		plt.bar(range(1,6,2), [pivot[level]['no'].mean() for level in levels], 
				yerr = [error[level]['no'].mean() for level in levels] , 
				width = 0.5, color = 'green', label = 'low probability', ecolor = 'green')
		plt.bar(np.arange(1,6,2)  + 0.5, [pivot[level]['yes'].mean() for level in levels], 
				yerr = [error[level]['yes'].mean() for level in levels] , 
				width = 0.5, color = 'red', label = 'high probability', ecolor = 'red')

		plt.xticks(np.arange(1,6,2) + 0.5, levels)
		plt.ylim(390,430)
		plt.xlim(0,7)
		plt.legend(loc = 'best')

		sns.despine(offset=50, trim = False)
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


	def adjustBeh(self):
		'''
		Add a factor to the beh file so that decoding can be done on trials with and without target type repeats
		in DTdisDP 
		'''	

		# read in preprocessed beh	
		for sj in range(1,25):	
			beh = pickle.load(open(self.FolderTracker(extension = ['beh','processed'], 
								filename = 'subject-{}_all.pickle'.format(sj)),'rb'))

			# read in raw behavior
			raws = []
			files = glob.glob(self.FolderTracker(extension = ['beh','raw'], 
								filename = 'subject-{}_ses_*.csv'.format(sj)))
			for file in files:
				raws.append(pd.read_csv(file))
			raws = pd.concat(raws)
			raws = raws[raws['practice'] == 'no']
			raws.reset_index(inplace = True)

			# check for target repeats
			target_rep =  np.zeros(raws.shape[0], dtype = bool)
			for i in range(raws.shape[0]):
				if raws['nr_trials'][i] > 1:
					if raws['target_type'][i] == raws['target_type'][i - 1]:
						target_rep[i] = True

			# find indices that survived preprocessin
			ses = 0
			idx = []
			tr_old = 1
			for tr in beh['nr_trials']:
				if tr < tr_old: # first trial of block can't be a repeat
					ses = 1
				raw_idx = np.where(raws['nr_trials'] == tr)[0]
				if raw_idx.size == 1:
					idx.append(raw_idx[0])
				else:
					idx.append(raw_idx[ses])	
				tr_old = tr
	
			# check selection 	
			if np.sum(raws['nr_trials'][idx].values == beh['nr_trials']) != len(beh['nr_trials']):
				print 'selected the wrong trials for subject {}'.format(sj)

			beh['target_rep'] = target_rep[idx]

			# save updated pickle file
			pickle.dump(beh, open(self.FolderTracker(extension = ['beh','processed'], 
						filename = 'subject-{}_all.pickle'.format(sj)),'wb'))	


	def adjustBeh2(self):
		'''
		Add a factor to the beh file so that erp to targets can be analyzed on hig and low probability locations
		in DTdisDP 
		'''	

		# read in preprocessed beh	
		for sj in range(1,25):	
			beh = pickle.load(open(self.FolderTracker(extension = ['beh','processed'], 
								filename = 'subject-{}_all.pickle'.format(sj)),'rb'))

			target_cnds = []
			for i, block in enumerate(beh['block_type']):
				target_cnds.append('{}-{}'.format(block, beh['target_high'][i]))

			beh['target_info'] = np.array(target_cnds)

			# save updated pickle file
			pickle.dump(beh, open(self.FolderTracker(extension = ['beh','processed'], 
						filename = 'subject-{}_all.pickle'.format(sj)),'wb'))	


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

		#time = (-0.05, 0)
		#s, e = [np.argmin(abs(info['times'] - t)) for t in time]
		files = glob.glob(self.FolderTracker(['tf', method], filename = '*-tf.pickle'))
		tf = []
		for file in files:
			with open(file ,'rb') as handle:
				tf.append(pickle.load(handle))	

		contra_idx = [info['ch_names'].index(e) for e in c_elec]
		ipsi_idx = [info['ch_names'].index(e) for e in i_elec]

		embed()
		plt.figure(figsize = (40,15))
		for plt_idx, cnd in enumerate(['DTsim','DTdisP','DTdisDP']):
			ax = plt.subplot(1,3 , plt_idx + 1, title = cnd, xlabel = 'time (ms)', ylabel = 'freq')
			contra = np.stack([np.mean(tf[i][cnd]['base_power'][:,contra_idx,:], axis = 1) for i in range(len(tf))])
			ipsi = np.stack([np.mean(tf[i][cnd]['base_power'][:,ipsi_idx,:], axis = 1) for i in range(len(tf))])
			X = (contra - ipsi) 
			X_thresh = threshArray(X, 0, method = 'ttest', p_value = 0.05)
			X_thresh = np.array(X_thresh, dtype = bool)
			plt.imshow(X.mean(axis = 0), cmap = cm.jet, interpolation='none', aspect='auto', 
							   origin = 'lower', extent=[times[0],times[-1],5,40],
							   vmin = -1, vmax = 1)
			plt.contour(X_thresh,origin = 'lower')
			
			plt.axvline (x = -0.25, color = 'white', ls ='--')
			plt.axvline (x = 0, color = 'white', ls ='--')
			plt.yticks(info['frex'][::5])
			plt.colorbar()
			sns.despine(offset=50, trim = False)

		plt.tight_layout()			
		plt.savefig(self.FolderTracker(['tf','figs'], filename = 'tf-main-basetest.pdf'))
		plt.close()	

		embed()

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/DT_sim'
	os.chdir(project_folder)

	# initiate current project
	PO = DT_sim()


	# analyze behavior
	# behavioral experiments 1 and 2
	#PO.prepareBEH(project, 'exp_1', factors, [['DTsim','DTdisU','DTdisP'],['yes','no']], project_param, to_filter)
	#PO.prepareBEH(project, 'exp_2', factors, [['DT_sim','DT_dis_DP','DT_dis_P'],['yes','no']], project_param, to_filter)
	#PO.plotExp1_2_3(exp = 'exp_1', excl_pos_bias = True)
	#PO.plotExp1_2_3(exp = 'exp_2', excl_pos_bias = True)
	
	# eeg experiment
	#PO.prepareBEH(project, part, factors, labels, project_param, to_filter)
	#PO.plotExp1_2_3(exp = 'beh', excl_pos_bias = True)
	#PO.singleTargetEEG()

	# adjust behavior file
	#PO.adjustBeh()
	#PO.adjustBeh2()

	# analyze eeg
	#PO.countCndCheck()

	
	#PO.plotERP()
	#PO.plotBDM(header = 'target')
	#PO.plotBDM(header = 'dist')
	#PO.plotTF(c_elec = ['PO7','PO3','O1'], i_elec= ['PO8','PO4','O2'], method = 'wavelet')

	# # run preprocessing
	for sj in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
	 	print 'starting subject {}'.format(sj)

		#for session in [2]:
		# 	PO.prepareEEG(sj = sj, session = session, eog = eog, ref = ref, eeg_runs = eeg_runs, 
		# 		t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
		# 		trigger = trigger, project_param = project_param, 
		# 		project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)

		#PO.detectSaccades(sj = sj)
		beh, eeg = PO.loadData(sj,True, (-0.75,0.55),'HEOG', 1, eye_dict = None)#dict(windowsize = 200, windowstep = 10, threshold = 20))

		# # ERP analysis (distractor tuned)
		#erp = ERP(eeg, beh, header = 'dist_loc', baseline = [-0.45,-0.25])
		#erp.selectERPData(time = [-0.45, 0.55], l_filter = 30, excl_factor = dict(dist_loc = ['None'])) 
		#erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1'], 
		# 								r_elec = ['PO8','PO4','O2'], midline = {'target_loc': [0,3]}, balance = False, erp_name = 'main')
		#erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7'], 
		# 								r_elec = ['PO8'], midline = {'target_loc': [0,3]}, balance = False, erp_name = 'main-PO7')
		#erp.topoFlip(left = [1,2])
		#erp.topoSelection(sj = sj, loc = [2,4], midline = {'target_loc': [0,3]}, topo_name = 'main')

		#  ERP analysis (target tuned)
		#erp = ERP(eeg, beh, header = 'target_loc', baseline = [-0.45,-0.25])
		#erp.selectERPData(time = [-0.45, 0.55], l_filter = 30, excl_factor = dict(dist_loc = ['0','1','2','3','4','5'])) 
		#erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1'], 
		# 					r_elec = ['PO8','PO4','O2'], cnd_header = 'target_info',balance = False, erp_name = 'main')
		#erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7'], 
		# 					r_elec = ['PO8'], balance = False, cnd_header = 'target_info', erp_name = 'main-PO7')

		# # BDM analysis (collapsed across low and high; exclude single target trials)
		# # feature decoding (dist)
		bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 10, method = 'acc', elec_oi = 'all', downsample = 128)
		bdm.Classify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55), 
		 			excl_factor = dict(dist_loc = ['None']), gat_matrix = False)

		# # # feature decoding (target)
		#bdm = BDM(beh, eeg, decoding = 'target_type', nr_folds = 10, elec_oi = 'all', downsample = 128)
		#bdm.Classify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55), 
		#			excl_factor = dict(dist_loc = ['None']), gat_matrix = True)

		#bdm = BDM(beh, eeg, decoding = 'target_type', nr_folds = 10, elec_oi = 'all', downsample = 128)
		#bdm.Classify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55), 
		# 			excl_factor = dict(target_rep = [False]), gat_matrix = False)	
	
		# cross feature decoding (independent)
		#bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 1, elec_oi = 'all', downsample = 128)
		#bdm.crossClassify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55), 
		# 				tr_factor = dict(dist_loc = ['0','1','2','3','4','5']), te_factor = dict(dist_loc = ['None']),
		# 				tr_header = 'dist_type', te_header = 'target_type', gat_matrix = True)

		#bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 1, elec_oi = 'all', downsample = 128)
		#bdm.crossClassify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55), 
		# 				tr_factor = dict(dist_loc = ['0','1','2','3','4','5']), te_factor = dict(dist_loc = ['None']),
		# 				tr_header = 'target_type', te_header = 'target_type', gat_matrix = False)

		#bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 1, elec_oi = 'all', downsample = 128, bdm_filter = dict(alpha = (8,12)))
		#bdm.crossClassify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55), 
		#				tr_factor = dict(dist_loc = ['0','1','2','3','4','5']), te_factor = dict(dist_loc = ['None']),
		#				tr_header = 'dist_type', te_header = 'target_type', gat_matrix = False)

		#bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 1, elec_oi = 'all', downsample = 128, bdm_filter = dict(theta = (4,8)))
		#bdm.crossClassify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55), 
		#				tr_factor = dict(dist_loc = ['0','1','2','3','4','5']), te_factor = dict(dist_loc = ['None']),
		#				tr_header = 'dist_type', te_header = 'target_type', gat_matrix = False)

		# cross feature decoding (dependent)
		#beh, eeg = PO.loadData(sj, (-0.75,0.55),True, 'HEOG', 1, eye_dict = None)
		#bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 10, elec_oi = 'all', downsample = 128)
		#bdm.crossClassify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55),
		#				tr_te_rel = 'dep', excl_factor = dict(dist_loc = ['None']),
		#				tr_header = 'dist_type', te_header = 'target_type', gat_matrix = False) # excl_factor already set

		# beh, eeg = PO.loadData(sj, (-0.75,0.55),True, 'HEOG', 1, eye_dict = None)
		# bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 10, elec_oi = 'all', downsample = 128, bdm_filter = dict(alpha = (8,12)))
		# bdm.crossClassify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55),
		# 				tr_te_rel = 'dep', excl_factor = dict(dist_loc = ['None']),
		# 				tr_header = 'dist_type', te_header = 'target_type', gat_matrix = False) # excl_factor already set

		# beh, eeg = PO.loadData(sj, (-0.75,0.55),True, 'HEOG', 1, eye_dict = None)
		# bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 10, elec_oi = 'all', downsample = 128, bdm_filter = dict(theta = (4,8)))
		# bdm.crossClassify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', time = (-0.75, 0.55),
		# 				tr_te_rel = 'dep', excl_factor = dict(dist_loc = ['None']),
		# 				tr_header = 'dist_type', te_header = 'target_type', gat_matrix = False) # excl_factor already set
		# BDM analysis ( low and high seperate)
		# feature decoding (dist)
		#bdm = BDM(beh, eeg, decoding = 'dist_type', nr_folds = 10, elec_oi = 'all', downsample = 128)
		#bdm.Classify(sj, cnds = ['DTsim-no','DTdisP-no','DTdisDP-no', 'DTsim-yes','DTdisP-yes','DTdisDP-yes'], cnd_header = 'condition', time = (-0.75, 0.55), nr_perm = 0, gat_matrix = False)

		#bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.3, 0.8), gat_matrix = False, nr_perm = 0)
		# feature decoding (target)
		#bdm = BDM(beh, eeg, decoding = 'target_type', nr_folds = 10, elec_oi = 'all', downsample = 128)
		#bdm.Classify(sj, cnds = ['DTsim-no','DTdisP-no','DTdisDP-no', 'DTsim-yes','DTdisP-yes','DTdisDP-yes'], cnd_header = 'condition', time = (-0.75, 0.55), nr_perm = 0, gat_matrix = False)	
		

		# location decoding (dist_loc)
		#bdm = BDM(beh, eeg, decoding = 'dist_loc', nr_folds = 10, elec_oi = 'all', downsample = 128)
		#bdm.Classify(sj, cnds = ['DTsim','DTdisP','DTdisDP'], cnd_header = 'block_type', 
		#			bdm_labels = ['0','1','2','3','4','5'], time = (-0.75, 0.55), nr_perm = 0, gat_matrix = False)

		# TF analysis
		tf = TF(beh, eeg)
		tf.TFanalysis(sj = sj, cnds = ['DTsim','DTdisP','DTdisDP'], 	
		 		  	cnd_header ='block_type', elec_oi = ['PO7', 'PO3', 'O1', 'PO8', 'PO4', 'O2'], base_period = None, 
		 			base_type = 'Z', time_period = (-0.6,0.5), method = 'wavelet', flip = dict(high_loc = [2]), downsample = 4)

	#  	#TF analysis
	# 	tf = TF()
	# 	tf.TFanalysis(sj = sj, cnds = ['DTsim','DTdisP','DTdisDP'], 
	# 			  cnd_header ='block_type', base_period = (-0.8,-0.6), 
	# 			  time_period = (-0.6,0.5), method = 'wavelet', flip = dict(high_prob = 'left'), downsample = 4)




