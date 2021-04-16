"""
analyze EEG data

Created by Dirk van Moorselaar on 01-07-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""
import os
import mne
import pickle
import math
import warnings

import numpy as np
import pandas as pd

from IPython import embed
from scipy.fftpack import fft, ifft
from scipy.signal import butter, lfilter, freqz
from support.FolderStructure import *
from support.support import select_electrodes, trial_exclusion

class ERP(FolderStructure):

	def __init__(self, eeg, beh, header, baseline, flipped = False):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		self.eeg = eeg
		self.beh = beh
		self.header = header
		self.baseline = baseline
		self.flipped = flipped


	def selectERPData(self, time, l_filter = None, h_filter = None, excl_factor = None):
		''' 

		THIS NEEDS TO BE ADJUSTED FOR TIMINGS

		Arguments
		- - - - - 


		Returns
		- - - -

		'''
 
		beh = self.beh
		EEG = self.eeg

		# check whether trials need to be excluded
		if type(excl_factor) == dict: # remove unwanted trials from beh
			beh, EEG = trial_exclusion(beh, EEG, excl_factor)

		# read in eeg data 
		if l_filter != None or h_filter != None:
			EEG.filter(l_freq = l_filter, h_freq = h_filter)
	
		# select time window and EEG electrodes
		EEG = EEG.crop(tmin = time[0],tmax = time[1])    
		ch_names = EEG.ch_names

		self.eeg = EEG 

		# store dictionary with variables for plotting
		plot_dict = {'ch_names': ch_names, 'times': EEG.times, 'info':EEG.info}

		with open(self.FolderTracker(['erp',self.header], filename = 'plot_dict.pickle'.format(self.header)),'wb') as handle:
			pickle.dump(plot_dict, handle)	


	@staticmethod	
	def baselineCorrect(X, times, base_period):
		''' 

		Applies baseline correction to an array of data by subtracting the average 
		from the base_period from data array.

		Arguments
		- - - - - 
		X (array): numpy array (trials x electrodes x timepoints)
		times (array): eeg timepoints 
		base_period (list): baseline window (start and end time)


		Returns
		- - - 
		X (array): baseline corrected EEG data
		'''

		# select indices baseline period
		start, end = [np.argmin(abs(times - b)) for b in base_period]

		nr_time = X.shape[2]
		nr_elec = X.shape[1]

		X = X.reshape(-1,X.shape[2])

		X = np.array(np.matrix(X) - np.matrix(X[:,start:end]).mean(axis = 1)).reshape(-1,nr_elec,nr_time)

		return X

	@staticmethod
	def selectMaxTrial(idx, cnds, all_cnds):
		''' 

		Loops over all conditions to determine which conditions contains the 
		least number of trials. Can be used to balance the number of trials
		per condition.

		Arguments
		- - - - - 
		idx (array): array of trial indices
		cnds (list| str): list of conditions checked for trial balancing. If all, all conditions are used
		all_cnds (array): array of trial specific condition information

		Returns
		- - - 
		max_trial (int): number of trials that can maximally be taken from a condition 
		'''

		if cnds == 'all':
			cnds = np.unique(all_cnds)

		max_trial = []
		for cnd in cnds:
			count = sum(all_cnds[idx] == cnd)
			max_trial.append(count)

		max_trial = min(max_trial)	

		return max_trial


	def topoFlip(self, left , header):
		''' 
		Flips the topography of trials where the stimuli of interest was presented 
		on the left (i.e. right hemifield). After running this function it is as if 
		all stimuli are presented right (i.e. the left hemifield)

		Arguments
		- - - - - 
		left (list): list containing stimulus labels indicating spatial position 
		header (string): column name used for flipping

		Returns
		- - - -
		inst (instance of ERP): The modified instance 

		'''	

		picks = mne.pick_types(self.eeg.info, eeg=True, exclude='bads')   
		# dictionary to flip topographic layout
		flip_dict = {'Fp1':'Fp2','AF7':'AF8','AF3':'AF4','F7':'F8','F5':'F6','F3':'F4',\
					'F1':'F2','FT7':'FT8','FC5':'FC6','FC3':'FC4','FC1':'FC2','T7':'T8',\
					'C5':'C6','C3':'C4','C1':'C2','TP7':'TP8','CP5':'CP6','CP3':'CP4',\
					'CP1':'CP2','P9':'P10','P7':'P8','P5':'P6','P3':'P4','P1':'P2',\
					'PO7':'PO8','PO3':'PO4','O1':'O2'}

		idx_l = np.sort(np.hstack([np.where(self.beh[header] == l)[0] for l in left]))

		# left stimuli are flipped as if presented right
		pre_flip = np.copy(self.eeg._data[idx_l][:,picks])

		# do actual flipping
		for l_elec, r_elec in flip_dict.items():
			l_elec_data = pre_flip[:,self.eeg.ch_names.index(l_elec)]
			r_elec_data = pre_flip[:,self.eeg.ch_names.index(r_elec)]
			self.eeg._data[idx_l,self.eeg.ch_names.index(l_elec)] = r_elec_data
			self.eeg._data[idx_l,self.eeg.ch_names.index(r_elec)] = l_elec_data

		self.flipped = True	

	def ipsiContraElectrodeSelection(self):
		'''

		'''	

		# left and right electrodes in standard set-up
		left_elecs = ['Fp1','AF7','AF3','F7','F5','F3','F1','FT7','FC5','FC3',
					'FC1','T7','CP1','P9','P7','P5','P3','P1','PO7','PO3','O1']
		right_elecs = ['Fp2','AF8','AF4','F8','F6','F4','F2','FT8','FC6','FC4',
						'FC2','T8','CP2','P10','P8','P6','P4','P2','PO8','PO4','O2']

		# check which electrodes are present in the current set-up				
		left_elecs = [l for l in left_elecs if l in self.ch_names]
		right_elecs = [r for r in right_elecs if r in self.ch_names]

		# select indices of left and right electrodes
		idx_l_elec = np.sort([self.ch_names.index(e) for e in l_elec])
		idx_r_elec = np.sort([self.ch_names.index(e) for e in r_elec])

		return idx_l_elec, idx_r_elec

	def cndSplit(self, beh, conditions, cnd_header):
		'''
		splits condition data in fast and slow data based on median split	

		Arguments
		- - - - - 
		beh (dataframe): pandas dataframe with trial specific info
		conditions (list): list of conditions. Each condition will be split individually
		cnd_header (str): string of column in beh that contains conditions
		'''	

		for cnd in conditions:
			median_rt = np.median(beh.RT[beh[cnd_header] == cnd])
			beh.loc[(beh.RT < median_rt) & (beh[cnd_header] == cnd), cnd_header] = '{}_{}'.format(cnd, 'fast')
			beh.loc[(beh.RT > median_rt) & (beh[cnd_header] == cnd), cnd_header] = '{}_{}'.format(cnd, 'slow')

		return beh	

	def createDWave(self, data, idx_l, idx_r, idx_l_elec, idx_r_elec):
		"""Creates a baseline corrected difference wave (contralateral - ipsilateral).
		For this function stimuli need not have been artificially shifted to the same hemifield
		
		Arguments:
			data {array}  -- eeg data (epochs X electrodes X time)
			idx_l {array} -- Indices of trials where stimuli of interest is presented left 
			idx_r {array} -- Indices of trials where stimuli of interest is presented right 
			l_elec {array} -- list of indices of left electrodes
			r_elec {array} -- list of indices from right electrodes
		
		Returns:
			d_wave {array} -- contralateral vs. ipsilateral difference waveform
		"""

		# create ipsi and contra waveforms
		ipsi = np.vstack((data[idx_l,:,:][:,idx_l_elec], data[idx_r,:,:][:,idx_r_elec]))
		contra = np.vstack((data[idx_l,:,:][:,idx_r_elec], data[idx_r,:,:][:,idx_l_elec]))
	
		# baseline correct data	
		ipsi = self.baselineCorrect(ipsi, self.eeg.times, self.baseline)
		contra = self.baselineCorrect(contra, self.eeg.times, self.baseline)

		# create ipsi and contra ERP
		ipsi = np.mean(ipsi, axis = (0,1)) 
		contra = np.mean(contra, axis = (0,1))

		d_wave = contra - ipsi

		return d_wave

	def permuteIpsiContra(self, eeg, contra_idx, ipsi_idx, nr_perm = 1000):
		"""Calls upon createDWave to create permuted difference waveforms. Can for example be used to calculate 
		permuted area under the curve to establish reliability of a component. Function assumes that it is if all 
		stimuli are presented within one hemifield
		
		Arguments:
			eeg {mne object}  -- epochs object mne
			contra_idx {array} -- Indices of contralateral electrodes 
			ipsi_idx {array} -- Indices of ipsilateral electrodes
		
		Keyword Arguments:
			nr_perm {int} -- number of permutations (default: {1000})
		
		Returns:
			d_wave {array} -- contralateral vs. ipsilateral difference waveform (can be used as sanity check)
			d_waves_p {array} -- permuted contralateral vs. ipsilateral difference waveforms (nr_perms X timepoints)
		"""		

		data = eeg._data
		nr_epochs = data.shape[0]

		# create evoked objects using mne functionality
		evoked = eeg.average().apply_baseline(baseline = self.baseline)
		d_wave = np.mean(evoked._data[contra_idx] - evoked._data[ipsi_idx], axis = 0)

		# initiate empty array for shuffled waveforms
		d_waves_p = np.zeros((nr_perm, eeg.times.size))

		for p in range(nr_perm):
			idx_p = np.random.permutation(nr_epochs)
			idx_left = idx_p[::2]
			idx_right = idx_p[1::2]
			d_waves_p[p] = self.createDWave(data, idx_left, idx_right, contra_idx, ipsi_idx)

		return d_wave, d_waves_p


	def createERP(self, beh, eeg, idx, fname, RT_split = False):
		"""Uses mne evoked functionality to create and save ERPs
		
		Arguments:
			beh {dataFrame} -- trial specific info
			eeg {mne object} -- Epochs object MNE
			idx {array} -- indices of trials of interest
			fname {str} -- fname to store evoked object
		
		Keyword Arguments:
			RT_split {bool} -- If True data will also be analyzed seperately for fast and slow trials (default: {False})
		"""

		beh = beh.iloc[idx].copy()
		eeg = eeg[idx]

		# create evoked objects using mne functionality and save file
		evoked = eeg.average().apply_baseline(baseline = self.baseline)
		evoked.save(self.FolderTracker(['erp', self.header],'{}-ave.fif'.format(fname)))

		if RT_split:
			median_rt = np.median(beh.RT)
			beh.loc[beh.RT < median_rt, 'RT_split'] = 'fast'
			beh.loc[beh.RT > median_rt, 'RT_split'] = 'slow'
			for rt in ['fast', 'slow']:
				mask = beh['RT_split'] == rt
				# create evoked objects using mne functionality and save file
				evoked = eeg[mask].average().apply_baseline(baseline = self.baseline)
				evoked.save(self.FolderTracker(['erp', self.header],'{}_{}-ave.fif'.format(fname, rt)))

	def ipsiContra(self, sj, left, right, l_elec = ['PO7'], r_elec = ['PO8'], conditions = 'all', cnd_header = 'condition', midline = None, erp_name = '', RT_split = False, permute = False):
		''' 

		Creates laterilized ERP's by cross pairing left and right electrodes with left and right position labels.
		ERPs are made for all conditios collapsed and for individual conditions

		Arguments
		- - - - - 
		sj (int): subject id (used for saving)
		left (list): stimulus labels indicating left spatial position(s) as logged in beh 
		right (list): stimulus labels indicating right spatial position(s) as logged in beh
		l_elec (list): left electrodes (right hemisphere)
		r_elec (list): right hemisphere (left hemisphere)
		conditions (str | list): list of conditions. If all, all unique conditions in beh are used
		cnd_header (str): name of condition column
		midline (None | dict): Can be used to limit analysis to trials where a specific 
								stimulus (key of dict) was presented on the midline (value of dict)
		erp_name (str): name of the pickle file to store erp data
		RT_split (bool): If true each condition is also analyzed seperately for slow and fast RT's (based on median split)
		permute (bool | int): If true (in case of a number), randomly flip the hemifield of the stimulus of interest and calculate ERPs

		Returns
		- - - -
		 

		'''

		# make sure it is as if all stimuli of interest are presented right from fixation	
		if not self.flipped:
			print('Flipping is done based on {} column in beh and relative to values {}. \
				If not correct please flip trials beforehand'.format(self.header, left))
			self.topoFlip(left , self.header)
		else:
			print('It is assumed as if all stimuli are presented right')

		# report that left right specification contains invalid values
		# ADD WARNING!!!!!!
		idx_l, idx_r = [],[]
		if len(left)>0:
			idx_l = np.sort(np.hstack([np.where(self.beh[self.header] == l)[0] for l in left]))
		if len(right)>0:	
			idx_r = np.sort(np.hstack([np.where(self.beh[self.header] == r)[0] for r in right]))

		# if midline, only select midline trials
		if midline != None:
			idx_m = []
			for key in midline.keys():
				idx_m.append(np.sort(np.hstack([np.where(self.beh[key] == m)[0] for m in midline[key]])))
			idx_m = np.hstack(idx_m)
			idx_l = np.array([idx for idx in idx_l if idx in idx_m])
			idx_r = np.array([idx for idx in idx_r if idx in idx_m])

		#if balance:
		#	max_trial = self.selectMaxTrial(np.hstack((idx_l, idx_r)), conditions, self.beh[cnd_header])

		# select indices of left and right electrodes
		idx_l_elec = np.sort([self.eeg.ch_names.index(e) for e in l_elec])
		idx_r_elec = np.sort([self.eeg.ch_names.index(e) for e in r_elec])

		if conditions == 'all':
			conditions = ['all'] + list(np.unique(self.beh[cnd_header]))

		for cnd in conditions:

			# select left and right trials for current condition
			# first select condition indices
			if cnd == 'all':
				idx_c = np.arange(self.beh[cnd_header].size)
			else:	
				idx_c = np.where(self.beh[cnd_header] == cnd)[0]
		
			# split condition indices in left and right trials	
			idx_c_l = np.array([l for l in idx_c if l in idx_l], dtype = int)
			idx_c_r = np.array([r for r in idx_c if r in idx_r], dtype = int)

			if idx_c_l.size == 0 and idx_c_r.size == 0:
				print('no data found for {}'.format(cnd))
				continue
			
			fname = 'sj_{}-{}-{}'.format(sj, erp_name, cnd)
			idx = np.hstack((idx_c_l, idx_c_r))
			self.createERP(self.beh, self.eeg, idx, fname, RT_split = RT_split)

			if permute:
				# STILL NEEDS DOUBLE CHECKING AGAINST MNE OUTPUT (looks ok!)
				# make it as if stumuli were presented left and right
				eeg = self.eeg[idx]
				d_wave, d_waves_p = self.permuteIpsiContra(eeg, idx_l_elec, idx_r_elec, nr_perm = permute)

				# save results
				perm_erps = {'d_wave': d_wave, 'd_waves_p': d_waves_p}
				pickle.dump(perm_erps, open(self.FolderTracker(['erp',self.header],'sj_{}_{}_{}_perm.pickle'.format(sj, erp_name, cnd)) ,'wb'))

			# create erp (nr_elec X nr_timepoints)
			#if cnd != 'all' and balance:
			#	idx_balance = np.random.permutation(ipsi.shape[0])[:max_trial]
			#	ipsi = ipsi[idx_balance,:,:]
			#	contra = contra[idx_balance,:,:]
			

	def topoSelection(self, sj, conditions = 'all', loc = 'all', midline = None, balance = False, topo_name = ''):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		try:
			with open(self.FolderTracker(['erp',self.header],'topo_{}.pickle'.format(topo_name)) ,'rb') as handle:
				topos = pickle.load(handle)
		except:
			topos = {}

		# update dictionary
		if str(sj) not in topos.keys():
			topos.update({str(sj):{}})

		if conditions == 'all':
			conditions = ['all'] + list(np.unique(self.beh['condition']))
		else:
			conditions = ['all'] + conditions

		# filthy hack to get rid of 'None' in index array
		if self.beh[self.header].dtype != 'int64':
			self.beh[self.header][self.beh[self.header] == 'None'] = np.nan

		if loc != 'all':
			idx_l = np.sort(np.hstack([np.where(np.array(self.beh[self.header], dtype = float) == l)[0] for l in loc]))

		if balance:
			max_trial = self.selectMaxTrial(idx_l, conditions, self.beh['condition'])
			
			# if midline, only select midline trials
			if midline != None:
				idx_m = []
				for key in midline.keys():
					idx_m.append(np.sort(np.hstack([np.where(self.beh[key] == m)[0] for m in midline[key]])))
				idx_m = np.hstack(idx_m)
				idx_l = np.array([idx for idx in idx_l if idx in idx_m])

		for cnd in conditions:

			# get condition data
			if cnd == 'all':
				idx_c = np.arange(self.beh['condition'].size)	
			else:	
				idx_c = np.where(self.beh['condition'] == cnd)[0]
			
			if loc != 'all':
				idx_c_l = np.array([l for l in idx_c if l in idx_l])
			elif loc == 'all' and midline != None:
				idx_c_l = np.array([l for l in idx_c if l in idx_m])
			else:
				idx_c_l = idx_c	

			if idx_c_l.size == 0:
				print('no topo data found for {}'.format(cnd))
				continue				

			topo = self.eeg[idx_c_l,:,:]

			# baseline correct topo data
			topo = self.baselineCorrect(topo, self.times, self.baseline)
			
			if cnd != 'all' and balance:
				idx_balance = np.random.permutation(topo.shape[0])[:max_trial]
				topo = topo[idx_balance,:,:]

			topo = np.mean(topo, axis = 0)

			topos[str(sj)].update({cnd:topo})
		
		with open(self.FolderTracker(['erp',self.header],'topo_{}.pickle'.format(topo_name)) ,'wb') as handle:
			pickle.dump(topos, handle)	


	def ipsiContraCheck(self,sj, left, right, l_elec = ['PO7'], r_elec = ['PO8'], conditions = 'all', midline = None, erp_name = ''):
		'''

		'''

		file = self.FolderTracker(['erp',self.header],'{}.pickle'.format(erp_name))

		if os.path.isfile(file):
			with open(file ,'rb') as handle:
				erps = pickle.load(handle)
		else:
			erps = {}

		# update dictionary
		if str(sj) not in erps.keys():
			erps.update({str(sj):{}})

		# select left and right trials
		idx_l = np.sort(np.hstack([np.where(self.beh[self.header] == l)[0] for l in left]))
		idx_r = np.sort(np.hstack([np.where(self.beh[self.header] == r)[0] for r in right]))

		# select indices of left and right electrodes
		idx_l_elec = np.sort([self.ch_names.index(e) for e in l_elec])
		idx_r_elec = np.sort([self.ch_names.index(e) for e in r_elec])

		if conditions == 'all':
			conditions = ['all'] + list(np.unique(self.beh['condition']))

		for cnd in conditions:

			erps[str(sj)].update({cnd:{}})

			# select left and right trials for current condition
			if cnd == 'all':
				idx_c = np.arange(self.beh['condition'].size)
			else:	
				idx_c = np.where(self.beh['condition'] == cnd)[0]
			
			idx_c_l = np.array([l for l in idx_c if l in idx_l])
			idx_c_r = np.array([r for r in idx_c if r in idx_r])

			l_ipsi = self.eeg[idx_c_l,:,:][:,idx_l_elec,:]
			l_contra = self.eeg[idx_c_l,:,:][:,idx_r_elec,:] 
			r_ipsi = self.eeg[idx_c_r,:,:][:,idx_r_elec,:]
			r_contra = self.eeg[idx_c_r,:,:][:,idx_l_elec,:] 

			# baseline correct data	
			l_ipsi = self.baselineCorrect(l_ipsi, self.times, self.baseline)
			l_contra = self.baselineCorrect(l_contra, self.times, self.baseline)

			r_ipsi = self.baselineCorrect(r_ipsi, self.times, self.baseline)
			r_contra = self.baselineCorrect(r_contra, self.times, self.baseline)

			# create erp
			l_ipsi = np.mean(l_ipsi, axis = (0,1))
			l_contra = np.mean(l_contra, axis = (0,1))

			r_ipsi = np.mean(r_ipsi, axis = (0,1))
			r_contra = np.mean(r_contra, axis = (0,1))

			erps[str(sj)][cnd].update({'l_ipsi':l_ipsi,'l_contra':l_contra,'r_ipsi':r_ipsi,'r_contra':r_contra})	

		# save erps	
		with open(self.FolderTracker(['erp',self.header],'{}.pickle'.format(erp_name)) ,'wb') as handle:
			pickle.dump(erps, handle)	

			
if __name__ == '__main__':

	pass

	# project_folder = '/home/dvmoors1/big_brother/Dist_suppression'
	# os.chdir(project_folder) 
	# subject_id = [1,2,5,6,7,8,10,12,13,14,15,18,19,21,22,23,24]
	# subject_id = [3,4,9,11,17,20]	
	# header = 'dist_loc'

	# session = ERP(header = header, baseline = [-0.3,0])
	# if header == 'target_loc':
	# 	conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
	# 	midline = {'dist_loc': [0,3]}
	# else:
	# 	conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
	# 	midline = {'target_loc': [0,3]}
	
	# for sj in subject_id:

	# 	session.selectERPData(sj, time = [-0.3, 0.8], l_filter = 30) 
	# 	session.ipsiContra(sj, left = [2], right = [4], l_elec = ['P7','P5','P3','PO7','PO3','O1'], 
	# 								r_elec = ['P8','P6','P4','PO8','PO4','O2'], midline = None, balance = True, erp_name = 'lat-down1')
	# 	session.ipsiContra(sj, left = [2], right = [4], l_elec = ['P7','P5','P3','PO7','PO3','O1'], 
	# 								r_elec = ['P8','P6','P4','PO8','PO4','O2'], midline = midline, balance = True, erp_name = 'lat-down1-mid')

	# 	session.topoFlip(left = [1,2])
		
	# 	session.topoSelection(sj, loc = [2,4], midline = None, topo_name = 'lat-down1')
		# session.topoSelection(sj, loc = [2,4], midline = midline, topo_name = 'lat-down1-mid')





