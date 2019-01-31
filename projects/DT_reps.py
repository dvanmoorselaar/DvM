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
#from eeg_analyses.Spatial_EM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info
sj_info = {'1': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
		  	'2': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
		  	'3': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
		  	'4': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}}, 
		  	'5': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
		  	'6': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}}, 
		  	'7': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
		  	'8': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
		  	'9': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}}, 
		  	'10': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
		  	'11': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
		  	'12': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
		  	'13': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
		  	'14': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'15': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'16': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}}, 
			'17': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'18': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'19': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'20': {'tracker': (False, '', None, 'Onset placeholder',0), 'replace':{}},
			'21': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'22': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'23': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			'24': {'tracker': (True, 'asc', 500, 'Onset placeholder',0), 'replace':{}},
			} 

# project specific info
project = 'DT_reps'
part = 'beh'
factors = ['block_type','repetition']
labels = [['DrTv','DvTr','DvTv'],[0,1,2,3]]
to_filter = ['RT'] 
project_param = ['practice','nr_trials','trigger','condition','RT', 'subject_nr',
				'block_type', 'correct','dist_loc','dist_orient','target_loc',
				'target_orient','repetition','fixed_pos', 'set_size', 'block_cnt']

montage = mne.channels.read_montage(kind='biosemi64')
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
trigger = [3]
t_min = 0
t_max = 0
flt_pad = 0.5
eeg_runs = [1,2,3] # 3 runs for subject 15 session 2
binary =  3840

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
		#PP.exclude_outliers(criteria = dict(RT = 'RT_filter == True', correct = ''))
		#PP.prep_JASP(agg_func = 'mean', voi = 'RT', data_filter = 'RT_filter == True', save = True)
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
                    filemode='w')

		# READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
		EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
		                                   montage=None, preload=True, eog=eog) for run in eeg_runs])

		#EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['V_do','H_l','Ref_r','Ref_l','EXG7','EXG8'])
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		# EEGica = EEG.filter(h_freq=None, l_freq=1,
		#                    fir_design='firwin', skip_by_annotation='edge')
		# EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		#             skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
		beh, missing, events = self.matchBeh(sj, session, events, trigger, 
		                             headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		epochs.selectBadChannels(channel_plots = False, inspect=False, RT = None)    
		epochs.artifactDetection(inspect=False, run = False)

		# ICA
		epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = True)

		# EYE MOVEMENTS
		self.binEye(epochs, missing, time_window=(t_min*1000, t_max*1000), tracker = tracker, tracker_shift = shift, start_event = start_event, extension = ext, eye_freq = t_freq)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, trigger)


	def matchBeh(self, sj, session, events, trigger, headers, max_trigger = 66):
		'''
		 '''
		 # read in data file
		beh_file = self.FolderTracker(extension=[
		            'beh', 'raw'], filename='subject-{}_ses_{}.csv'.format(sj, session))
		 # get triggers logged in beh file
		beh = pd.read_csv(beh_file)
		beh = beh[headers]
		beh = beh[beh['practice'] == 'no']
		beh = beh.drop(['practice'], axis=1)

		# get triggers bdf file
		idx_trigger = np.where(events[:,2] == trigger)[0] + 1
		trigger_bdf = events[idx_trigger,2] 
		 # log number of unique triggers
		unique = np.unique(trigger_bdf)
		logging.info('{} detected unique triggers (min = {}, max = {})'.
		                format(unique.size, unique.min(), unique.max()))
		
		# make sure trigger info between beh and bdf data matches
		missing_trials = []
		while beh.shape[0] != trigger_bdf.size:
			 # remove spoke triggers, update events
			if missing_trials == []:
				logging.info('removed {} spoke triggers from bdf'.format(sum(trigger_bdf > max_trigger)))
				# update events
				to_remove = idx_trigger[np.where(trigger_bdf > max_trigger)[0]] -1
				events = np.delete(events, to_remove, axis = 0)
				trigger_bdf = trigger_bdf[trigger_bdf < max_trigger]
			trigger_beh = beh['trigger'].values
			if trigger_beh.size > trigger_bdf.size:
				for i, trig in enumerate(trigger_bdf):
					if trig != trigger_beh[i]:
						beh.drop(beh.index[i], inplace=True)  
						miss = beh['nr_trials'].iloc[i]
						missing_trials.append(miss)
						logging.info('Removed trial {} from beh file,because no matching trigger exists in bdf file'.format(miss))
						break

		missing = np.array(missing_trials)        
		# log number of matches between beh and bdf       
		logging.info('{} matches between beh and epoched data'.
		    format(sum(beh['trigger'].values == trigger_bdf)))           

		return beh, missing, events

	def updateBEH(self, sj):
		'''
		Function updates the preprocessed behavior pickle file. It adds a new column 
		that allows to test whether the anticipatory alpha as observed with target location
		repetition should be attributed to lingering effects from the previous trial
		'''

		# read in the raw csv files and filter practice trials
		beh_files = glob.glob(PO.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_ses_*.csv'.format(sj)))

		raw_beh = pd.concat([pd.read_csv(f) for f in beh_files], ignore_index = True)
		raw_beh = raw_beh[raw_beh['practice'] == 'no']

		# shift the labels from DvTv_2 to DvTv_3
		for loc in ['target_loc','dist_loc']:
			upd = raw_beh[loc][raw_beh['condition'] == 'DvTv_2']
			raw_beh[loc + '_new'] = np.nan
			raw_beh[loc + '_new'][raw_beh['condition'] == 'DvTv_3'] = upd.values

		# read in beh file after preprocessing
		beh = pickle.load(open(self.FolderTracker(extension = ['beh','processed'], 
							filename = 'subject-{}_all.pickle'.format(sj)),'rb'))
		# checks which trials from raw_beh survived preprocessing
		sel_tr = beh['clean_idx']
	
		# Adjust selected trials and update raw_beh to correct for missing triggers in bdf file
		if sj in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,24]:
			sel_tr[np.where(np.diff(sel_tr) <0)[0] + 1:] += 3672
		elif sj == 5:
			sel_tr[np.where(np.diff(sel_tr) <0)[0] + 1:] += 3168
		elif sj == 15: # subject has missing data in bdf file in session 2 DOES NOT YET WORK
			raw_beh['nr_trials_upd'] = range(1, 7345)
			for i in [2277 + 3672]*27:                                        
				raw_beh.drop(raw_beh.index[[i]], inplace = True)
			sel_tr[np.where(np.diff(sel_tr) <0)[0] + 1:] += 3672 
		elif sj == 17: # subject has missing data in bdf file in session 1  DOES NOT YET WORK
			raw_beh.reset_index(inplace = True)
			for i in [21,1062,2635,3528]:                                         
				raw_beh.drop(raw_beh.index[[i]], inplace = True)
			sel_tr[np.where(np.diff(sel_tr) <0)[0] + 1:] += 3668
		elif sj == 23:
			raw_beh['nr_trials_upd'] = range(1, 7345)
			raw_beh = raw_beh[raw_beh['nr_trials_upd'] != 2214]
			sel_tr[np.where(np.diff(sel_tr) <0)[0] + 1:] += 3671

		raw_beh.reset_index(inplace = True)

		try:
			if sum(raw_beh['trigger'].values[sel_tr] == beh['trigger']) == beh['condition'].shape[0]:
				print 'selection via selected trials will work for sj {}'.format(sj)

				# update the target and dist postion bins for DvTv_3 based on n-1
				for loc in ['target_loc','dist_loc']:
					beh['{}-1'.format(loc[:-4])] = raw_beh[loc + '_new'].values[sel_tr]

				# save updated pickle file
				pickle.dump(beh, open(self.FolderTracker(extension = ['beh','processed'], 
							filename = 'subject-{}_all.pickle'.format(sj)),'wb'))	
		except:
			print('what happened?')		

	def BEHexp1(self, set_size_control = False):
		'''
		analyzes experiment 1 as reported in the MS
		'''

		# read in data
		file = self.FolderTracker(['beh-exp1','analysis'], filename = 'preprocessed.csv')
		data = pd.read_csv(file)

		# create pivot (main analysis)
		data = data.query("RT_filter == True")
		if set_size_control: # filter out all trials in set size 8 where distractor and target are presented next to one another
			data['distance'] = abs(data['target_loc'] - data['dist_loc']) 
			data['distance'][data['distance'] > 4] = 8 - data['distance'][data['distance'] > 4]
			data['loc_filter'] = np.logical_and(data['set_size'] == 8, data['distance'] == 1)
			data = data.query("loc_filter == False")

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
		sns.despine(offset=50, trim = False)

		if set_size_control:
			plt.savefig(self.FolderTracker(['beh-exp1','figs'], filename = 'control-ana.pdf'))	
		else:	
			plt.savefig(self.FolderTracker(['beh-exp1','figs'], filename = 'main-ana.pdf'))		
		plt.close()	

		# save parameters for JASP 	
		if not set_size_control:	
			np.savetxt(self.FolderTracker(['beh-exp1','analysis'], filename = 'fits_alpha-JASP.csv'), alpha, delimiter = "," ,header = ",".join(headers), comments='')
			np.savetxt(self.FolderTracker(['beh-exp1','analysis'], filename = 'fits_delta-JASP.csv'), delta, delimiter = "," ,header = ",".join(headers), comments='')

	def BEHexp2(self, set_size_control = False):
		'''
		analyzes experiment 2 as reported in the MS
		'''

		# read in data
		file = self.FolderTracker(['beh-exp2','analysis'], filename = 'preprocessed.csv')
		data = pd.read_csv(file)

		# create pivot (main analysis)
		data = data.query("RT_filter == True")
		if set_size_control: # filter out all trials in set size 8 where distractor and target are presented next to one another
			data['distance'] = abs(data['target_loc'] - data['dist_loc']) 
			data['distance'][data['distance'] > 4] = 8 - data['distance'][data['distance'] > 4]
			data['loc_filter'] = np.logical_and(data['set_size'] == 8, data['distance'] == 1)
			data = data.query("loc_filter == False")
		pivot = data.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','set_size','repetition'], aggfunc = 'mean')
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# plot conditions
		plt.figure(figsize = (20,10))

		ax = plt.subplot(1,2, 1, title = 'Repetition effect', ylabel = 'RT (ms)', xlabel = 'repetition', ylim = (300,650), xlim = (0,11))
		for b, bl in enumerate(['DrTv','DvTv','Tv']):
			for s, set_size in enumerate([4,8]):
				pivot[bl][set_size].mean().plot(yerr = pivot_error[bl][set_size], 
											label = '{}-{}'.format(bl,set_size), 
											ls = ['-','--'][s], color = ['red','blue','yellow'][b])
		
		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		# create pivot (normalized data)
		norm = pivot.values
		for i,j in [(0,12),(12,24),(24,36),(36,48),(48,60),(60,72)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		pivot = pd.DataFrame(norm, index = np.unique(data['subject_nr']), columns = pivot.keys())
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# fit data to exponential decay function (and plot normalized data)
		ax = plt.subplot(1,2, 2, title = 'Normalized RT', ylabel = 'au', xlabel = 'repetition', ylim = (0.5,1), xlim = (0,11))
		alpha, delta = np.zeros((pivot.shape[0],6)),  np.zeros((pivot.shape[0],6))
		c_idx = 0
		headers = []
		for b, bl in enumerate(['DrTv','DvTv','Tv']):
			for s, set_size in enumerate([4,8]):
				headers.append('{}_{}'.format(bl,set_size))
				X = pivot[bl][set_size].values
				pivot[bl][set_size].mean().plot(yerr = pivot_error[bl][set_size], 
												label = '{0}-{1}'.format(bl,set_size),
												ls = ['-','--'][s], color = ['red','blue','yellow'][b])
				for i, x in enumerate(X):
					popt, pcov = curvefitting(range(12),x,bounds=(0, [1,1])) 
					alpha[i, c_idx] = popt[0]
					delta[i, c_idx] = popt[1]
				c_idx += 1

		plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=50, trim = False)


		if set_size_control:
			plt.savefig(self.FolderTracker(['beh-exp2','figs'], filename = 'control-ana.pdf'))	
		else:	
			plt.savefig(self.FolderTracker(['beh-exp2','figs'], filename = 'main-ana.pdf'))			
		plt.close()	

		# save parameters for JASP 
		if not set_size_control:		
			np.savetxt(self.FolderTracker(['beh-exp2','analysis'], filename = 'fits_alpha-JASP.csv'), alpha, delimiter = "," ,header = ",".join(headers), comments='')
			np.savetxt(self.FolderTracker(['beh-exp2','analysis'], filename = 'fits_delta-JASP.csv'), delta, delimiter = "," ,header = ",".join(headers), comments='')

	def BEHexp3(self):
		'''
		analyzes experiment 3 as reported in the MS
		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		data = pd.read_csv(file)

		# create pivot (main analysis)
		data = data.query("RT_filter == True")
		pivot = data.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# plot conditions
		plt.figure(figsize = (20,10))

		ax = plt.subplot(1,2, 1, title = 'Repetition effect', ylabel = 'RT (ms)', xlabel = 'repetition', ylim = (350,550), xlim = (0,4))
		for b, bl in enumerate(['DrTv','DvTv','DvTr']):
			pivot[bl].mean().plot(yerr = pivot_error[bl], 
											label = bl, 
											ls = '-', color = ['red','blue','green'][b])
		
		plt.xticks([0,1,2,3])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		# create pivot (normalized data)
		norm = pivot.values
		for i,j in [(0,4),(4,8),(8,12)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		pivot = pd.DataFrame(norm, index = np.unique(data['subject_nr']), columns = pivot.keys())
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# fit data to exponential decay function (and plot normalized data)
		ax = plt.subplot(1,2, 2, title = 'Normalized RT', ylabel = 'au', xlabel = 'repetition', ylim = (0.5,1), xlim = (0,4))
		alpha, delta = np.zeros((pivot.shape[0],6)),  np.zeros((pivot.shape[0],6))
		c_idx = 0
		headers = []
		for b, bl in enumerate(['DrTv','DvTv','DvTr']):
			for s, set_size in enumerate([4,8]):
				headers.append('{}_{}'.format(bl,set_size))
				X = pivot[bl].values
				pivot[bl].mean().plot(yerr = pivot_error[bl], 
												label = bl,
												ls = '-', color = ['red','blue','green'][b])
				for i, x in enumerate(X):
					popt, pcov = curvefitting(range(4),x,bounds=(0, [1,1])) 
					alpha[i, c_idx] = popt[0]
					delta[i, c_idx] = popt[1]
				c_idx += 1

		plt.xticks([0,1,2,3])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=50, trim = False)

		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'main-ana.pdf'))			
		plt.close()	

		# save parameters for JASP 
		np.savetxt(self.FolderTracker(['beh','analysis'], filename = 'fits_alpha-JASP.csv'), alpha, delimiter = "," ,header = ",".join(headers), comments='')
		np.savetxt(self.FolderTracker(['beh','analysis'], filename = 'fits_delta-JASP.csv'), delta, delimiter = "," ,header = ",".join(headers), comments='')

	def primingCheck(self):
		'''
		Analyze whether the observed effects simply reflect priming or priming plus
		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# first determine the priming/suppresssive effects in distractor repeat sequences (now collapsed across conditions)
		beh = beh.query("RT_filter == True")
		DR = np.diff(beh.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')['DrTv'])
		DR_error = pd.Series(confidence_int(DR))

		# now calculate the priming effect in baseline sequences (still contains multiple reps)
		beh = beh[beh['block_type'] == 'DvTv']
		beh['priming'] = np.nan
		beh['priming'] = beh['priming'].apply(pd.to_numeric)
		for i in range(1, beh.shape[0]):
			if (beh['dist_loc'].iloc[i - 1] == beh['dist_loc'].iloc[i]) and \
			(beh['nr_trials'].iloc[i] - 1 == beh['nr_trials'].iloc[i-1]) and\
			(beh['subject_nr'].iloc[i -1] == beh['subject_nr'].iloc[i]) and \
			(beh['repetition'].iloc[i] > beh['repetition'].iloc[i-1]): # search for repetitions, that are direct, within the same subject and within the same sequence
				beh['priming'].iloc[i] = beh['RT'].iloc[i]- beh['RT'].iloc[i - 1]

		# get priming effect
		PR = beh.pivot_table(values = 'priming', index = 'subject_nr', columns = ['repetition'], aggfunc = 'mean')	
		PR_error = pd.Series(confidence_int(PR.values))
		PR_mean_error = PR.mean(axis = 1).values.std()/sqrt(PR.shape[0])

		# save output for JASP
		headers = ['DR1','DR2','DR3','PR1','PR2','PR3','PRave']
		x = np.hstack((np.hstack((DR, PR)), PR.mean(axis =1).reshape(DR.shape[0],1)))
			
		# save as .csv file
		np.savetxt(self.FolderTracker(['beh','analysis'], filename = 'priming.csv'), x, delimiter = "," ,header = ",".join(headers), comments='')	
		t, p = ttest_rel(DR, PR)
		print t, p

		# plot comparison / plot average
		plt.figure(figsize = (10,10))
		plt.bar(np.arange(1,4)-0.1,abs(DR.mean(axis = 0)), width = 0.2, yerr = DR_error, align = 'center', color = 'red', label = 'Dr')
		plt.bar(np.arange(1,4)+0.1,abs(PR.mean(axis = 0).values), width = 0.2,  yerr = PR_error,align = 'center', color = 'blue', label = 'baseline')
		plt.ylim((0,40))
		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'priming.pdf'))	
		plt.close()

		plt.figure(figsize = (10,10))
		plt.bar(np.arange(1,4)-0.1,abs(DR.mean(axis = 0)), width = 0.2, yerr = DR_error, align = 'center', color = 'red', label = 'Dr')
		plt.bar(np.arange(1,4)+0.1,[abs(PR.mean().mean())]*3, width = 0.2,  yerr = [PR_mean_error]*3,align = 'center', color = 'blue', label = 'baseline')
		plt.ylim((0,40))
		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'priming_ave.pdf'))	
		plt.close()

	def ctfTemp(self):
		'''
		Plot as shown in MS.
			- CTF across a range of frequencies collapsed across conditions (top left)
			- Evoked slopes within the alpha band (top right)
			- Total slopes within the alpha band (bottom)
			- statistics contrast each repetition against it's baseline in the variable condition (e.g. repeat 0 vs variable 0)
		'''

		# create new figure and set plotting parameters
		
		#colors = ['blue', color] 

		
		# read in plotting info (and shift times such that stimulus onset is at 0ms)
		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'all_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		# read in Total and Evoked slopes within alpha band
		alpha = []
		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'cnds_*_slopes-Foster_alpha1.pickle'))
		for file in files:
			alpha.append(pickle.load(open(file, 'rb')))
		colors = ['grey','black','red','green']
		for i, file in enumerate(files):
			plt.figure(figsize = (20,20))
			sj = file.split('_')[5]
			for c, cnd in enumerate(['DvTv_3','DvTr_3']):
				slope = np.squeeze(alpha[i][cnd]['T_slopes'])
				plt.plot(times, slope, label = cnd, color = colors[c])
			plt.legend(loc = 'best')
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'test-{}.pdf'.format(sj), overwrite = False))
			plt.close()		


	def ctfPlot(self, repetition, color = ['red'], window = (-0.55,0)):
		'''
		Plot as shown in MS.
			- CTF across a range of frequencies collapsed across conditions (top left)
			- Evoked slopes within the alpha band (top right)
			- Total slopes within the alpha band (bottom)
			- statistics contrast each repetition against it's baseline in the variable condition (e.g. repeat 0 vs variable 0)
		'''

		# create new figure and set plotting parameters
		plt.figure(figsize = (30,20))
		colors = ['blue', color] 
		if repetition == 'target_loc':
			ylim = [[-0.05,0.16],[-0.05,0.1]]
			yrange = [0.005, 0.004]
			rep_cnds = ['DvTr_0','DvTr_3']
		elif repetition == 'dist_loc':
			ylim = [[-0.05,0.16],[-0.35,0.1]]
			yrange = [0.005, 0.005]
			rep_cnds = ['DrTv_0','DrTv_3']	
		
		# read in plotting info (and shift times such that stimulus onset is at 0ms)
		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'all_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		# read in Total slopes collapsed across conditions
		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',repetition], filename = 'all_*_slopes-Foster_all.pickle'))
		print files
		freqs = [pickle.load(open(file, 'rb')) for file in files]

		# read in Total and Evoked slopes within alpha band
		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',repetition], filename = 'cnds_*_slopes-Foster_alpha.pickle'))
		alpha = [pickle.load(open(file, 'rb')) for file in files]
		#self.ctfANOVAinput(times, window, alpha, ['DvTv_0','DvTv_3'] + rep_cnds, repetition )

		# read in repetition effect within alpha band
		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',repetition[:-4] +'-1'], filename = 'cnds_*_slopes-Foster_alpha.pickle'))
		reps = [pickle.load(open(file, 'rb')) for file in files]
		
		# plot total across frequencies
		ax = plt.subplot(221, xlabel = 'Time (ms)', ylabel = 'freqs')  
		slopes = np.stack([ctf['all']['T_slopes'] for ctf in freqs])
		X = slopes.mean(axis = 0)
		p_vals = signedRankArray(slopes, 0)
		slopes[:,p_vals > 0.01] = 0
		p_vals = clusterBasedPermutation(slopes,0)
		X[p_vals > 0.01] = 0
		plt.imshow(X, cmap = cm.viridis, interpolation='none', aspect='auto', 
						  origin = 'lower', extent=[times[0],times[-1],4,34])
		plt.axhline(y = 8, color = 'white', ls = '--')
		plt.axhline(y = 12, color = 'white', ls = '--')
		plt.colorbar()

		# plot evoked and total effects
		plt_idx = [422,424,223,224]
		plt_cntr = 0
		for p, power in enumerate(['E_slopes','T_slopes']):
			perm = {}
			for i, cnds in enumerate((['DvTv_0','DvTv_3'],rep_cnds)):
				ax = plt.subplot(plt_idx[plt_cntr], xlabel = 'Time (ms)', ylim = ylim[p]) 
				plt.yticks([ylim[p][0],0,ylim[p][1]])
				for c, cnd in enumerate(cnds):
					slopes = np.squeeze(np.stack([ctf[cnd][power] for ctf in alpha]))
					perm.update({cnd: slopes})
					err, X = bootstrap(slopes)
					plt.plot(times, X, label = cnd, color = colors[i], ls = ['--','-'][c])
					plt.fill_between(times, X + err, X - err, alpha = 0.2, color = colors[i])	
					self.clusterPlot(perm[cnd], 0, 0.05, times, ylim[p][0] + yrange[p] * (c + 1), color = colors[i], ls = ['--','-'][c])
				
				if c == 1:
					# contrast first and final repetition across conditions
					self.beautifyPlot(y = 0, ylabel = 'CTF slope')
					#self.clusterPlot(perm[cnds[0]], perm[cnds[1]], 0.05, times,ylim[p][0] + yrange[p] * (c + 2), color = 'black')
					if i == 1:
						rep_slopes = np.mean(np.squeeze(np.stack([r['DvTv_3'][power] for r in reps])), axis = 0)
						plt.plot(times, rep_slopes, label = 'N-1', color = 'black', ls = '--')
						self.clusterPlot(perm['DvTv_0'], perm[rep_cnds[0]], 0.05, times,ylim[p][0] + yrange[p] * (c + 2), color = 'black', ls = '--')
						self.clusterPlot(perm['DvTv_3'], perm[rep_cnds[1]], 0.05, times,ylim[p][0] + yrange[p] * (c + 3), color = 'black', ls = '-')
						self.clusterPlot(perm[rep_cnds[0]] - perm['DvTv_0'], perm[rep_cnds[1]] - perm['DvTv_3'],0.05, times, ylim[p][0] + yrange[p] * (c + 4), color = 'grey')
					plt.legend(loc = 'topleft')	
					sns.despine(offset=50, trim = False)
				plt_cntr += 1
						
		# save figures
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'main-{}-alpha.pdf'.format(repetition)))
		plt.close()	

	def ctfANOVAinput(self, times, window, slopes, factors, repetition ):
		'''
		gets the average in a predefined window which can be used as input for a repeated measures ANOVA
		'''	

		# get indices time window
		s, e = [np.argmin(abs(t - times)) for t in window]
		x = []
		for factor in factors:
			slope = np.squeeze(np.stack([ctf[factor]['T_slopes'] for ctf in slopes]))[:,s:e].mean(axis = 1)
			x.append(slope)

		X = np.vstack(x).T	
		# save as .csv file
		np.savetxt(self.FolderTracker(['ctf','all_channels_no-eye','anova'], filename = '{}.csv'.format(repetition)), X, delimiter = "," ,header = ",".join(factors), comments='')

	def ctfCrossPlot(self, repetition ):
		'''

		'''	

		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=0)
		#colors = ['blue', color] 
		
		# read in plotting info (and shift times such that stimulus onset is at 0ms)
		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'all_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		# read in Cross Training repetition 0
		rep_0 = []
		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',repetition], filename = '*_cross-training_baseline-0.pickle'))
		for file in files:
			rep_0.append(pickle.load(open(file, 'rb')))

		rep_3 = []
		files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',repetition], filename = '*_cross-training_baseline-3.pickle'))
		for file in files:
			rep_3.append(pickle.load(open(file, 'rb')))

		plt_idx = 1
		for r, rep in enumerate([rep_0, rep_3]):
			for c, cnd in enumerate(['DvTr_', 'DrTv_']):
				ax = plt.subplot(2,2,plt_idx, xlabel = 'Train ?? Time (ms)', ylabel = 'Test ?? Time (ms)', title = cnd + ['0','3'][r])
				ctf = np.squeeze(np.stack([x[cnd + ['0','3'][r]]['slopes'] for x in rep]))  
				X = threshArray(ctf, 0, method = 'ttest', p_value = 0.05)
				plt.imshow(X, norm = norm, cmap = cm.bwr, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], vmin = -0.1, vmax = 0.1)
				plt.colorbar()
				plt_idx += 1

		# save figures
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'cross-{}.pdf'.format(repetition)))
		plt.close()	


	def erpPlot(self, repetition, color = ['red']):
		'''
		Plot as shown in MS.
			- IPSI (left) and CONTRA (right) plots (top 4 plots)
			- Contra - Ipsi (bottom 2 plots)
			- statistics contrast each repetition against it's baseline in the variable condition (e.g. repeat 0 vs variable 0)
		'''

		# create new figure and set plotting parameters
		plt.figure(figsize = (30,20))
		colors = ['blue', color] 
		if repetition == 'target_loc':
			ylim = [-8,6]
			rep_cnds = ['DvTr_0','DvTr_3']
			yrange = 0.5
		elif repetition == 'dist_loc':
			ylim = [-8,6]
			yrange = 0.5
			rep_cnds = ['DrTv_0','DrTv_3']	
		
		# read in Total slopes collapsed across conditions
		file = self.FolderTracker(['erp',repetition], filename = 'lat-down1-mid.pickle')
		erps = pickle.load(open(file, 'rb'))

		with open(self.FolderTracker(['erp','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25
		elec_idx = [erps['1']['all']['elec'][0].index(e) for e in ['PO7','PO3','O1']]
		self.erpANOVAinput(times, (0.17,0.23), erps, repetition, 'N2pc', elec_idx)
		self.erpANOVAinput(times, (0.27,0.34), erps, repetition, 'Pd', elec_idx)

		# plot ipsi and contralateral seperately
		plt_idx = 1
		erp_dict = {'ipsi':{},'contra':{}}
		for l, lat in enumerate(['ipsi','contra']):
			for i, cnds in enumerate((['DvTv_0','DvTv_3'],rep_cnds)):
				ax = plt.subplot(4,2,plt_idx, xlabel = 'Time (ms)',ylim = ylim, title = lat) 
				plt.yticks([ylim[0],0,ylim[1]])
				for c, cnd in enumerate(cnds):
					erp = np.squeeze(np.stack([erps[key][cnd][lat][elec_idx,:] for key in erps])).mean(axis = 1)
					erp_dict[lat].update({cnd:erp})
					err, X = bootstrap(erp)
					plt.plot(times, X, label = cnd, color = colors[i], ls = ['--','-'][c])
					#plt.fill_between(times, X + err, X - err, alpha = 0.2, color = colors[c])	
				
				if i == 1:
					self.clusterPlot(erp_dict[lat]['DvTv_0'], erp_dict[lat][rep_cnds[0]], 0.05, times,ylim[0] + yrange * (1), color = 'black', ls = '--')
					self.clusterPlot(erp_dict[lat]['DvTv_3'], erp_dict[lat][rep_cnds[1]], 0.05, times,ylim[0] + yrange * (2), color = 'black', ls = '-')

				self.beautifyPlot(y = 0, ylabel = 'micro Volt')
				sns.despine(offset=50, trim = False)
				plt.legend(loc = 'topleft')	
					
				plt_idx += 1
		
		# plof difference waveforms
		for i, cnds in enumerate((['DvTv_0','DvTv_3'],rep_cnds)):
			ax = plt.subplot(2,2,i + 3, xlabel = 'Time (ms)', ylim = (-4,2)) 
			plt.yticks([-4,0,2])
			for c, cnd in enumerate(cnds):
				erp = erp_dict['contra'][cnd] - erp_dict['ipsi'][cnd]
				err, X = bootstrap(erp)
				plt.plot(times, X, label = cnd, color = colors[i], ls = ['--','-'][c])

			if i == 1:
				# repetition effect	
				self.clusterPlot(erp_dict['contra']['DvTv_0'] - erp_dict['ipsi']['DvTv_0'], 
								erp_dict['contra'][rep_cnds[0]] - erp_dict['ipsi'][rep_cnds[0]], 0.05, times,-4 + 0.15, color = 'black', ls = '--')
				self.clusterPlot(erp_dict['contra']['DvTv_3'] - erp_dict['ipsi']['DvTv_3'], 
								erp_dict['contra'][rep_cnds[1]] - erp_dict['ipsi'][rep_cnds[1]], 0.05, times,-4 + 0.3, color = 'black', ls = '-')
				# baseline correction
				self.clusterPlot((erp_dict['contra']['DvTv_0'] - erp_dict['ipsi']['DvTv_0']) - (erp_dict['contra'][rep_cnds[0]] - erp_dict['ipsi'][rep_cnds[0]]), 
								(erp_dict['contra']['DvTv_3'] - erp_dict['ipsi']['DvTv_3']) - (erp_dict['contra'][rep_cnds[1]] - erp_dict['ipsi'][rep_cnds[1]]),
								 0.05, times,-4 + 0.45, color = 'grey')
				
			self.beautifyPlot(y = 0, ylabel = 'micro Volt')
			sns.despine(offset=50, trim = False)
			plt.legend(loc = 'topleft')	

		# save figures
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'main-{}.pdf'.format(repetition)))
		plt.close()	

	def erpANOVAinput(self, times, window, erps, repetition, component, elec_idx):
		'''
		gets the average in a predefined window which can be used as input for a repeated measures ANOVA
		'''	

		# get indices time window
		s, e = [np.argmin(abs(t - times)) for t in window]
		if repetition == 'target_loc':
			factors = ['DvTv_0','DvTv_3'] + ['DvTr_0','DvTr_3']
		elif repetition == 'dist_loc':
			factors = ['DvTv_0','DvTv_3'] + ['DrTv_0','DrTv_3']
		x = []
		headers = []
		for lat in ['contra', 'ipsi']:
			for factor in factors:
				erp = np.stack([erps[key][factor][lat][elec_idx,:] for key in erps]).mean(axis = 1)[:,s:e].mean(axis = 1)
				x.append(erp)
				headers.append(lat + '-' + factor)

		# add difference between contra and ipsi
		for i in range(4):
			x.append(x[i] - x[i+4] )
			headers.append(factors[i])

		X = np.vstack(x).T	
		# save as .csv file
		np.savetxt(self.FolderTracker(['erp','anova'], filename = '{}-{}.csv'.format(component,repetition)), X, delimiter = "," ,header = ",".join(headers), comments='')	

		# apply jackknife procedure
		rep0 = np.stack([erps[key]['DvTr_0']['contra'][elec_idx,:] for key in erps]).mean(axis = 1) - np.stack([erps[key]['DvTr_0']['ipsi'][elec_idx,:] for key in erps]).mean(axis = 1)
		rep3 = np.stack([erps[key]['DvTr_3']['contra'][elec_idx,:] for key in erps]).mean(axis = 1) - np.stack([erps[key]['DvTr_3']['ipsi'][elec_idx,:] for key in erps]).mean(axis = 1)
		onset, t_value = jackknife(rep0,rep3, times, [0.17,0.23], percent_amp = 45, timing = 'offset')	
		print 'onset ' + repetition, onset *1000, t_value


	def bdmPlot(self, repetition, color = ['red'], window = (-0.55,0), plot_clusters):
		'''
		Plot as shown in MS.
			- CTF across a range of frequencies collapsed across conditions (top left)
			- Evoked slopes within the alpha band (top right)
			- Total slopes within the alpha band (bottom)
			- statistics contrast each repetition against it's baseline in the variable condition (e.g. repeat 0 vs variable 0)
		'''

		# create new figure and set plotting parameters
		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/6.0)
		colors = ['blue', color] 
		if repetition == 'target_loc':
			ylim = [0.14,0.28]
			yrange = 0.003
			rep_cnds = ['DvTr_0','DvTr_3']
		elif repetition == 'dist_loc':
			ylim = [0.14,0.28]
			yrange = 0.003
			rep_cnds = ['DrTv_0','DrTv_3']	

		# read in plotting info (and shift times such that stimulus onset is at 0ms)
		with open(self.FolderTracker(['bdm',repetition], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25
	
		# read in broadband EEG across all electrodes
		bdm = []
		files = glob.glob(self.FolderTracker(['bdm',repetition], filename = 'class_*_perm-False.pickle'))
		for file in files:
			bdm.append(pickle.load(open(file, 'rb')))

		self.bdmANOVAinput(times, (0.17,0.23), bdm, repetition, 'N2pc')
		self.bdmANOVAinput(times, (0.27,0.34), bdm, repetition, 'Pd')
		
		# plot diagonal decoding
		perm = {}
		for i, cnds in enumerate((['DvTv_0','DvTv_3'],rep_cnds)):
			ax = plt.subplot(2,2,i + 1, xlabel = 'Time (ms)', ylim = ylim)
			plt.yticks([ylim[0],1/6.0,ylim[1]])
			for c, cnd in enumerate(cnds):
				dec =  np.stack([np.diag(b[cnd]['standard']) for b in bdm])
				perm.update({cnd: dec})
				err, X = bootstrap(dec)
				plt.plot(times, X, label = cnd, color = colors[i], ls = ['--','-'][c])
				plt.fill_between(times, X + err, X - err, alpha = 0.2, color = colors[i])	
				self.clusterPlot(perm[cnd], 1/6.0, 0.05, times, ylim[0] + yrange * (c + 1), color = colors[i], ls = ['--','-'][c])
			
			if i == 1:
				self.clusterPlot(perm['DvTv_0'], perm[rep_cnds[0]], 0.05, times,ylim[0] + yrange * (c + 2), color = 'black', ls = '--')
				self.clusterPlot(perm['DvTv_3'], perm[rep_cnds[1]], 0.05, times,ylim[0] + yrange * (c + 3), color = 'black', ls = '-')
				self.clusterPlot(perm[rep_cnds[0]] - perm['DvTv_0'], perm[rep_cnds[1]] - perm['DvTv_3'],0.05, times, ylim[0] + yrange * (c + 4), color = 'grey')
			self.beautifyPlot(y = 1/6.0, ylabel = 'decoding acc')
			plt.legend(loc = 'topleft')	
			sns.despine(offset=50, trim = False)

		# # plot GAT matrices
		plt_idx = [9,10,13,14]
		perm_info = {}
		for c, cnd in enumerate((['DvTv_0','DvTv_3'] + rep_cnds)):
			ax = plt.subplot(4,4,plt_idx[c], xlabel = 'Train ?? time (ms)',  ylabel = 'Test ?? time (ms)', title = cnd) 
			dec = np.stack([b[cnd]['standard'] for b in bdm])
			perm_info[cnd] = dec
			X = threshArray(dec, 1/6.0, method = 'ttest', p_value = 0.05)
			if plot_clusters:
				#if c == 1:
				#	sig_cl = clusterBasedPermutation(perm_info[rep_cnds[0]], perm_info[rep_cnds[1]], p_val = 0.05)	
				if c == 3:	
					sig_cl = clusterBasedPermutation(perm_info[rep_cnds[0]] - perm_info[rep_cnds[0]], perm_info['DvTv_0'] - perm_info['DvTv_3'], p_val = 0.05)	
				plt.imshow(np.array(sig_cl, dtype = bool), aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]])
			else:
				plt.imshow(X, norm = norm, cmap = cm.bwr, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], vmin = 0.14, vmax = 0.28)
			#if c in [1,3]:
			#	plt.colorbar() 

			sns.despine(offset=50, trim = False)

		# read in broadband EEG across posterior electrodes
		if repetition == 'target_loc':
			dec_info = [('post','alpha'),('post','theta')]
		elif repetition == 'dist_loc':
			dec_info = [('post','alpha'),('post','theta')]

		for idx, (elec, band) in enumerate(dec_info):
			# if repetition == 'dist_loc':
			# 	ax = plt.subplot(6,2,8 + idx*2, xlabel = 'Time (ms)', ylim = ylim, title = '{}-{}'.format(elec, band))
			# elif repetition == 'target_loc':
			# 	ax = plt.subplot(4,2,6 + idx*2, xlabel = 'Time (ms)', ylim = ylim, title = '{}-{}'.format(elec, band))
			ax = plt.subplot(4,2,6 + idx*2, xlabel = 'Time (ms)', ylim = ylim, title = '{}-{}'.format(elec, band))
			plt.yticks([ylim[0],1/6.0,ylim[1]])
			bdm = []
			files = glob.glob(self.FolderTracker(['bdm',elec,repetition], filename = 'class_*_perm-False-{}.pickle'.format(band)))
			for file in files:
				bdm.append(pickle.load(open(file, 'rb')))
			if repetition == 'dist_loc':	
				self.bdmANOVAinput(times, (0.27,0.34), bdm, repetition, 'Pd-{}'.format(band), diag = False)
			color_idx = 0
			for c, cnd in enumerate((['DvTv_0','DvTv_3'] + rep_cnds)):

				dec = np.squeeze(np.stack([b[cnd]['standard'] for b in bdm]))
				perm.update({cnd: dec})
				err, X = bootstrap(dec)
				if cnd in ['DvTv_3',rep_cnds[1]]:
					plt.plot(times, X, label = cnd, color = colors[color_idx], ls = '-')
					self.clusterPlot(perm[cnd], 1/6.0, 0.05, times, ylim[0] + yrange * (c + 1), color = colors[color_idx], ls = '-')
					plt.fill_between(times, X + err, X - err, alpha = 0.2, color = colors[color_idx])	
					color_idx += 1

			self.clusterPlot(perm['DvTv_3'], perm[rep_cnds[1]], 0.05, times,ylim[0] + yrange * (c + 2), color = 'black', ls = '-')
			self.clusterPlot(perm[rep_cnds[0]] - perm['DvTv_0'], perm[rep_cnds[1]] - perm['DvTv_3'], 0.05, times,ylim[0] + yrange * (c + 3), color = 'grey', ls = '-')
			self.beautifyPlot(y = 1/6.0, ylabel = 'decoding acc')
			if repetition == 'dist_loc':
				plt.axvline(x = 0.27, ls = '--', color = 'grey')
				plt.axvline(x = 0.34, ls = '--', color = 'grey')
			plt.legend(loc = 'topleft')	
			sns.despine(offset=50, trim = False)


		# save figures
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['bdm','MS-plots'], filename = 'main-{}.pdf'.format(repetition)))
		plt.close()	

	def bdmANOVAinput(self, times, window, bdms, repetition, component, diag = True):
		'''
		gets the average in a predefined window which can be used as input for a repeated measures ANOVA
		'''	

		# get indices time window
		s, e = [np.argmin(abs(t - times)) for t in window]
		if repetition == 'target_loc':
			factors = ['DvTv_0','DvTv_3'] + ['DvTr_0','DvTr_3']
		elif repetition == 'dist_loc':
			factors = ['DvTv_0','DvTv_3'] + ['DrTv_0','DrTv_3']
		x = []
		headers = []
		for factor in factors:
			if diag:
				bdm = np.stack([np.diag(b[factor]['standard']) for b in bdms])[:,s:e].mean(axis = 1)
			else:
				bdm = np.stack([b[factor]['standard'] for b in bdms])[:,s:e].mean(axis = 1)
			x.append(bdm)

		X = np.vstack(x).T	
		# save as .csv file
		np.savetxt(self.FolderTracker(['bdm','anova'], filename = '{}-{}.csv'.format(component,repetition)), X, delimiter = "," ,header = ",".join(factors), comments='')	

		# apply jackknife procedure
		#rep0 = np.stack([erps[key]['DvTr_0']['contra'][elec_idx,:] for key in erps]).mean(axis = 1) - np.stack([erps[key]['DvTr_0']['ipsi'][elec_idx,:] for key in erps]).mean(axis = 1)
		#rep3 = np.stack([erps[key]['DvTr_3']['contra'][elec_idx,:] for key in erps]).mean(axis = 1) - np.stack([erps[key]['DvTr_3']['ipsi'][elec_idx,:] for key in erps]).mean(axis = 1)
		#onset, t_value = jackknife(rep0,rep3, times, [0.17,0.23], percent_amp = 45, timing = 'offset')


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

	def frontalBias(self, erp_name = 'topo_frontal-bias', elec = ['AF3','Fz','AFz','AF4','F3','F1','F2','F4']):
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
				embed()
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
			for b, cnds in enumerate([['DvTv_0','DvTr_0','DvTv_3','DvTr_3'],['DvTv_0','DrTv_0','DvTv_3','DrTv_3']]):
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

	def ctfrepBias(self):
		'''

		'''		

		with open(self.FolderTracker(['ctf','all_channels_no-eye','target_loc'], filename = 'alpha_info.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		plt.figure(figsize = (30,10))
		# loop over target and dist repetitions
		for idx, loc in enumerate(['target_loc','dist_loc']):
			ax = plt.subplot(1,2, idx + 1, title = loc, xlabel = 'Time (ms)') 
			clust = []
			p_values = []
			for c, cnd in enumerate([loc, loc[:-4] + '-1']):
				files = glob.glob(self.FolderTracker(['ctf','all_channels_no-eye',cnd], filename = 'cnds_*_slopes-Foster_alpha.pickle'))
				ctf = []
				for file in files:
					ctf.append(pickle.load(open(file,'rb')))
				if cnd[-1] == '1':
					X = np.squeeze(np.stack([ctf[i]['DvTv_3']['T_slopes'] for i in range(len(ctf))]))
				else:
					X = np.squeeze(np.stack([ctf[i][['DvTr_3','DrTv_3'][idx]]['T_slopes'] for i in range(len(ctf))]))	
				clust.append(X)
				p_values.append(np.mean(X[:,times < -0.25],1))

				err, slopes = bootstrap(X)
				plt.plot(times, slopes, label = ['repeat','variable'][c], color = ['red','green'][c])
				plt.fill_between(times, slopes + err, slopes - err, alpha = 0.2, color = ['red','green'][c])		

			print loc, stats.ttest_rel(p_values[0],p_values[1])
			self.clusterPlot(clust[0], clust[1], 0.05, times, ax.get_ylim()[0] + 0.01, color = 'black')
			self.beautifyPlot(y = 0, ylabel = 'CTF slope')
			plt.legend(loc = 'best')	

			sns.despine(offset=50, trim = False)
					
		# save figures
		plt.tight_layout()
		plt.savefig(self.FolderTracker(['ctf','all_channels_no-eye','MS-plots'], filename = 'repBias.pdf'))
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
 

	def plotTF(self):
		'''

		'''
		norm = MidpointNormalize(midpoint=0)

		# read in data
		with open(self.FolderTracker(['tf','wavelet','target_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		contra_idx = info['ch_names'].index('PO7')	
		ipsi_idx = info['ch_names'].index('PO8')

		# read in TF data
		tfs = {'target_loc':[],'dist_loc':[]}
		for loc in ['target_loc', 'dist_loc']:
			files = glob.glob(self.FolderTracker(['tf','wavelet',loc], filename = '*-tf.pickle'))
			for file in files:
				with open(file ,'rb') as handle:
					#tfs[loc].append(pickle.load(handle))
					tf = pickle.load(handle)
				if loc == 'target_loc':
					x_i = tf['DvTr_3']['base_power'][:,ipsi_idx,:]
					x_c = tf['DvTr_3']['base_power'][:,contra_idx,:]
				elif loc == 'dist_loc':
					x_i = tf['DrTv_3']['base_power'][:,ipsi_idx,:]
					x_c = tf['DrTv_3']['base_power'][:,contra_idx,:]					
				x = x_c - x_i
				plt.imshow(x, cmap = cm.jet, interpolation='none', aspect='auto', 
						  origin = 'lower', extent=[times[0], times[-1],0,40])
				plt.yticks(info['frex'][::4])
				plt.colorbar()
				sj = file.split('-')[0][-2:]
				if sj[0] == '/':
					sj = sj[1]

				plt.savefig(self.FolderTracker(['tf','wavelet','Mike-test'], filename = 'tf_{}-{}.pdf'.format(loc, sj)))
				plt.close()			



		# plot 
		norm = MidpointNormalize(midpoint=0)
		for header in ['dist_loc','target_loc']:
			plt.figure(figsize = (30,15))
			if header == 'target_loc':
				cnds = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
			elif header == 'dist_loc':
				cnds = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']

			for a, cnd in enumerate(cnds):
				ax = plt.subplot(2,2, a + 1, title = cnd, xlabel = 'Time (ms)', ylabel = 'Freq') 

				X_c = np.stack([tfs[header][i][cnd]['power'][:,contra_idx,:] for i in range(len(tfs[header]))])
				X_i = np.stack([tfs[header][i][cnd]['power'][:,ipsi_idx,:] for i in range(len(tfs[header]))])
				X = (X_c - X_i).mean(axis = 0)/(X_c + X_i).mean(axis = 0)

				plt.imshow(X, cmap = cm.jet, interpolation='none', aspect='auto', 
						  origin = 'lower', extent=[times[0], times[-1],0,40], vmin = -0.1, vmax = 0.1)
				plt.yticks(info['frex'][::4])
				plt.colorbar()

				sns.despine(offset=50, trim = False)
					
			# save figures
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['tf','wavelet','MS-plots'], filename = 'tf_{}no-base.pdf'.format([header])))
			plt.close()					


if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/DT_reps'
	os.chdir(project_folder)
	PO = DT_reps()

	# exp 1 and 2
	#PO.prepareBEH(project, 'beh-exp1', ['block_type','set_size','repetition'], [['target','dist'],[4,8],range(12)], project_param)
	#PO.prepareBEH(project,'beh-exp2',['block_type','set_size','repetition'], [['DvTv','DrTv','Tv'],[4,8],range(12)], project_param)
	# exp 3 (eeg)
	#PO.prepareBEH(project, part, factors, labels, project_param)

	#preprocessing and main analysis
	for sj in [20,21,22,23,24]: 


		# RUN PREPROCESSING
		# for session in range(1,3):
		# 	PO.prepareEEG(sj = sj, session = 2, eog = eog, ref = ref, eeg_runs = eeg_runs, 
		#   				t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
		#   				trigger = trigger, project_param = project_param, 
		#  				project_folder = project_folder, binary = binary, channel_plots = False, inspect = False)

		# READ IN PREPROCESSED DATA FOR FURTHER ANALYSIS
		#PO.updateBEH(sj)
		#beh, eeg = PO.loadData(sj, (-0.3,0.8),True, 'HEOG', 1)

		for header in ['target_loc', 'dist_loc']:

			if header == 'target_loc':
				midline = {'dist_loc': [0,3]}
				cnds = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
			elif header == 'dist_loc':
				midline = {'target_loc': [0,3]}
				cnds = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']	

			# CTF analysis
			#ctf = CTF(beh, eeg, 'all_channels_no-eye', header, nr_iter = 10, nr_blocks = 3, nr_bins = 6, nr_chans = 6, delta = False)

			# step 1: search broad band of frequencies collapsed across all conditions
			#ctf.spatialCTF(sj, [-0.3, 0.8], cnds, method = 'Foster', freqs = dict(all = [4,30]), downsample = 4, nr_perm = 0, collapse = True)
			# step 2: compare conditions within the alpha band
			#ctf.spatialCTF(sj, [-0.3, 0.8], cnds, method = 'Foster', freqs = dict(alpha = [8,12]), downsample = 4, nr_perm = 0)
			# step 3: cross train between first and final repetition to examine learning effects
			#ctf.crosstrainCTF(sj, [-0.3, 0.8], train_cnds = ['DvTv_0'], test_cnds = ['DvTr_0','DrTv_0'], 
			#				freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = True, nr_perm = 0, name = 'baseline-0')
			#ctf.crosstrainCTF(sj, [-0.3, 0.8], train_cnds = ['DvTv_3'], test_cnds = ['DvTr_3','DrTv_3'], 
			#				freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = True, nr_perm = 0, name = 'baseline-3')
			# ctf.crosstrainCTF(sj, [-0.3, 0.8], train_cnds = ['DrTv_0'], test_cnds = ['DrTv_3'], 
			#  			 freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = True, nr_perm = 0, name = 'DrTv-perm_0')
			# ctf.crosstrainCTF(sj, [-0.3, 0.8], train_cnds = ['DvTr_0'], test_cnds = ['DvTr_3'], 
			#  			 freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = True, nr_perm = 0, name = 'DvTr-perm_0')
			# step 4: compare conditions within the theta band
			#ctf.spatialCTF(sj, [-0.3, 0.8], cnds, method = 'Foster', freqs = dict(theta = [4,8]), downsample = 4, nr_perm = 0)
			# step 5: test whether anticipatory effects can be explained by lingering effects frm previous trial
			#ctf = CTF(beh, eeg, 'all_channels_no-eye', '{}-1'.format(header[:-4]), nr_iter = 10, nr_blocks = 3, nr_bins = 6, nr_chans = 6, delta = False)
			#ctf.spatialCTF(sj, [-0.3, 0.8], ['DvTv_3'], method = 'Foster', freqs = dict(alpha = [8,12]), downsample = 4, nr_perm = 0, plot = False)


			# ERP analysis
			#erp = ERP(eeg, beh, header = header, baseline = [-0.3, 0], eye = True)
			#erp.selectERPData(time = [-0.3, 0.8], l_filter = 30) 
			#erp.ipsiContra(sj = sj, left = [2,3], right = [4,5], l_elec = ['PO7','PO3','O1','P3','P5','P7'], 
			#  				r_elec = ['PO8','PO4','O2','P4','P6','P8'], midline = midline, balance = False, erp_name = 'main-unbalanced')
			#erp.ipsiContra(sj = sj, left = [2], right = [4], l_elec = ['PO7','PO3','O1','P3','P5','P7'], 
			#  				r_elec = ['PO8','PO4','O2','P4','P6','P8'], midline = midline, balance = False, erp_name = 'main-unbalanced-low')
			# erp.topoFlip(left = [1, 2])
			# erp.topoSelection(sj = sj, loc = [2,4], midline = midline, topo_name = 'main', balance = True)
			#erp.topoSelection(sj = sj, loc = [0,1,2,3,4,5], topo_name = 'frontal-bias', balance = True)

			# BDM analysis
			#bdm = BDM(beh ,eeg, decoding = header, nr_folds = 10, elec_oi = 'all', downsample = 128, bdm_filter = dict(alpha = (8,12)))
			#bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.3, 0.8), gat_matrix = False, nr_perm = 0)

			#bdm = BDM(beh, eeg, decoding = header, nr_folds = 10, elec_oi = 'post', downsample = 128, bdm_filter = None)
			#bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.3, 0.8), gat_matrix = False, nr_perm = 0)

			#bdm = BDM(beh, eeg, decoding = header, nr_folds = 10, elec_oi = 'post', downsample = 128, bdm_filter = dict(alpha = (8,12)))
			#bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.3, 0.8), gat_matrix = False, nr_perm = 0)

			#bdm = BDM(beh, eeg, decoding = header, nr_folds = 10, elec_oi = 'post', downsample = 128, bdm_filter = dict(theta = (4,8)))
			#bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.3, 0.8), gat_matrix = False, nr_perm = 0)

			#bdm = BDM(beh, eeg, decoding = header, nr_folds = 10, elec_oi = 'post', downsample = 128, bdm_filter = dict(beta = (12,20)))
			#bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.3, 0.8), gat_matrix = False, nr_perm = 0)

			# TF analysis
			#tf = TF(beh, eeg)
			#tf.TFanalysis(sj = sj, cnds = cnds, 	
			#		  	cnd_header ='condition', base_period = (-0.5,-0.3), 
			#			time_period = (-0.3,0.8), method = 'wavelet', flip = {header: [1,2]}, factor = {header: [1,2,4,5]}, downsample = 4)


	# analysis manuscript
	# BEH
	#PO.BEHexp1(set_size_control = False)
	#PO.BEHexp2(set_size_control = False)
	#PO.primingCheck()
	#PO.BEHexp3()	

	# ERP 
	#PO.erpPlot(repetition = 'target_loc', color = 'green')
	#PO.erpPlot(repetition = 'dist_loc', color = 'red')
	#for header in ['target_loc', 'dist_loc']:
		#PO.componentSelection(header, cmp_name = 'P1', cmp_window = [0.11,0.15], ext = 0.0175, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.componentSelection(header, cmp_name = 'N1', cmp_window = [0.16,0.22], ext = 0.0175, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.componcentSelection(header, cmp_name = 'N2pc', cmp_window = [0.17,0.23], ext = 0, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.componentSelection(header, cmp_name = 'Pd', cmp_window = [0.28,0.35], ext = 0, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])
		#PO.erpLateralized(header, erp_name = 'lat-down1-mid', elec = ['PO3','PO7','O1'])

	#PO.erpSelection(header = 'dist_loc', topo_name = 'main', elec = ['PO3','PO7','O1'])
	#PO.erpSelection(header = 'target_loc', topo_name = 'main', elec = ['PO3','PO7','O1'])
	#PO.erpContrast(erp_name = 'lat-down1-mid', elec = ['PO7','PO3','O1'])
	#PO.frontalBias()

	# PO.componentSelection(header = 'dist_loc', erp_name = 'lat-down1-mid', elec = ['PO7','PO3','O1'], 
	# 						cmpnts = dict(N2pc = (0.2, 0.3), Pd = (0.25, 0.4)))

	# BDM
	PO.bdmPlot(repetition = 'target_loc', color = 'green')
	PO.bdmPlot(repetition = 'dist_loc', color = 'red')
	
	
	#PO.bdmDiag()
	#for header in ['target_loc', 'dist_loc']:
	#		PO.bdmSelection(header, 'Pd', (0.28,0.35))

	# CTF
	#PO.ctfTemp()
	#PO.ctfPlot(repetition = 'target_loc', color = 'green')
	#PO.ctfPlot(repetition = 'dist_loc', color = 'red')
	#PO.ctfCrossPlot(repetition = 'target_loc')
	#PO.ctfCrossPlot(repetition = 'dist_loc')
	#PO.ctfSlopes()
	#PO.ctfCrossTrainold()
	#PO.ctfallFreqs()
	#PO.ctfrepBias()

	# TF.
	#PO.plotTF()



