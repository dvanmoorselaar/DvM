"""
analyze EEG data

Created by Dirk van Moorselaar on 01-07-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""
import os
import mne
import pickle
import math
#import matplotlib
#matplotlib.use('agg') # now it works via ssh connection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helperFunctions import *
from IPython import embed
from scipy.fftpack import fft, ifft
from scipy.signal import butter, lfilter, freqz
from support.FolderStructure import *
from eeg_support import *

class ERP(FolderStructure):

	def __init__(self, eeg, beh, header, baseline):
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
		self.flipped = False

	def selectERPData(self, time = [-0.3, 0.8], l_filter = False, excl_factor = None):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		beh = self.beh
		EEG = self.eeg
		embed()

		# check whether trials need to be excluded
		if type(excl_factor) == dict: # remove unwanted trials from beh
			mask = [(beh[key] == f).values for  key in excl_factor.keys() for f in excl_factor[key]]
			for m in mask: 
				mask[0] = np.logical_or(mask[0],m)
			mask = mask[0]
			if mask.sum() > 0:
				beh.drop(np.where(mask)[0], inplace = True)
				beh.reset_index(inplace = True)
				EEG.drop(np.where(mask)[0])
				print 'Dropped {} trials after specifying excl_factor'.format(sum(mask))
				print 'NOTE DROPPING IS DONE IN PLACE. PLEASE REREAD DATA IF THAT CONDITION IS NECESSARY AGAIN'

			else:
				print 'Trial exclusion: no trials selected that matched specified criteria'
				mask = np.zeros(beh.shape[0], dtype = bool)

		# read in eeg data 
		self.flipped = False
		if l_filter:
			EEG.filter(l_freq = None, h_freq = l_filter)

		# select time window and EEG electrodes
		s, e = [np.argmin(abs(EEG.times - t)) for t in time]
		picks = mne.pick_types(EEG.info, eeg=True, exclude='bads')
		eegs = EEG._data[:,picks,s:e]
		times = EEG.times[s:e]
		ch_names = EEG.ch_names

		# store dictionary with variables for plotting
		plot_dict = {'ch_names': EEG.ch_names, 'times':times, 'info':EEG.info}

		with open(self.FolderTracker(['erp',self.header], filename = 'plot_dict.pickle'.format(self.header)),'wb') as handle:
			pickle.dump(plot_dict, handle)	

		self.eeg = eegs
		self.beh = beh	
		self.ch_names = ch_names
		self.times = times	

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


	def topoFlip(self, left = []):
		''' 
		Flips the topography of trials where the stimuli of interest was presented 
		on the left (i.e. right hemifield). After running this function it is as if 
		all stimuli are presented right (i.e. the left hemifield)

		Arguments
		- - - - - 
		left (list): list containing stimulus labels indicating spatial position 

		Returns
		- - - -
		inst (instance of ERP): The modified instance 

		'''	
		
		# dictionary to flip topographic layout
		flip_dict = {'Fp1':'Fp2','AF7':'AF8','AF3':'AF4','F7':'F8','F5':'F6','F3':'F4',\
					'F1':'F2','FT7':'FT8','FC5':'FC6','FC3':'FC4','FC1':'FC2','T7':'T8',\
					'C5':'C6','C3':'C4','C1':'C2','TP7':'TP8','CP5':'CP6','CP3':'CP4',\
					'CP1':'CP2','P9':'P10','P7':'P8','P5':'P6','P3':'P4','P1':'P2',\
					'PO7':'PO8','PO3':'PO4','O1':'O2','Fpz':'Fpz','AFz':'AFz','Fz':'Fz',\
					'FCz':'FCz','Cz':'Cz','CPz':'CPz','Pz':'Pz','POz':'POz','Oz':'Oz',\
					'Fp2':'Fp1','AF8':'AF7','AF4':'AF3','F8':'F7','F6':'F5','F4':'F3',\
					'F2':'F1','FT8':'FT7','FC6':'FC5','FC4':'FC3','FC2':'FC1','T8':'T7',\
					'C6':'C5','C4':'C3','C2':'C1','TP7':'TP8','CP5':'CP6','CP3':'CP4',\
					'CP1':'CP2','P9':'P10','P7':'P8','P5':'P6','P3':'P4','P1':'P2','PO8':'PO7',\
					'PO4':'PO3','O2':'O1'}


		idx_l = np.sort(np.hstack([np.where(self.beh[self.header] == l)[0] for l in left]))

		# left stimuli are flipped as if presented right
		pre_flip = self.eeg[idx_l,:,:]
		flipped = np.zeros(pre_flip.shape)

		# do actual flipping
		for key in flip_dict.keys():
			flipped[:,self.ch_names.index(flip_dict[key]),:] = pre_flip[:,self.ch_names.index(key),:]

		self.eeg[idx_l,:,:] = flipped
		self.flipped = True	


	def ipsiContra(self, sj, left, right, l_elec = ['PO7'], r_elec = ['PO8'], conditions = 'all', cnd_header = 'condition', midline = None, balance = False, erp_name = ''):
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

		Returns
		- - - -
		 

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

		# filthy hack to get rid of 'None' in index array
		if self.beh[self.header].dtype != 'int64':
			self.beh[self.header][self.beh[self.header] == 'None'] = np.nan

		# select left and right trials
		idx_l = np.sort(np.hstack([np.where(np.array(self.beh[self.header], dtype = float) == l)[0] for l in left]))
		idx_r = np.sort(np.hstack([np.where(np.array(self.beh[self.header], dtype = float) == r)[0] for r in right]))

		# if midline, only select midline trials
		if midline != None:
			idx_m = []
			for key in midline.keys():
				idx_m.append(np.sort(np.hstack([np.where(self.beh[key] == m)[0] for m in midline[key]])))
			idx_m = np.hstack(idx_m)
			idx_l = np.array([idx for idx in idx_l if idx in idx_m])
			idx_r = np.array([idx for idx in idx_r if idx in idx_m])

		if balance:
			max_trial = self.selectMaxTrial(np.hstack((idx_l, idx_r)), conditions, self.beh[cnd_header])

		# select indices of left and right electrodes
		idx_l_elec = np.sort([self.ch_names.index(e) for e in l_elec])
		idx_r_elec = np.sort([self.ch_names.index(e) for e in r_elec])

		if conditions == 'all':
			conditions = ['all'] + list(np.unique(self.beh[cnd_header]))
	
		for cnd in conditions:

			erps[str(sj)].update({cnd:{}})

			# select left and right trials for current condition
			if cnd == 'all':
				idx_c = np.arange(self.beh[cnd_header].size)
			else:	
				idx_c = np.where(self.beh[cnd_header] == cnd)[0]
		
			idx_c_l = np.array([l for l in idx_c if l in idx_l], dtype = int)
			idx_c_r = np.array([r for r in idx_c if r in idx_r], dtype = int)

			if idx_c_l.size == 0 and idx_c_r.size == 0:
				print 'no data found for {}'.format(cnd)
				continue

			if self.flipped:
				# as if all stimuli presented right: left electrodes are contralateral, right electrodes are ipsilateral
				ipsi = np.vstack((self.eeg[idx_c_l,:,:][:,idx_r_elec], self.eeg[idx_c_r,:,:][:,idx_r_elec]))
				contra = np.vstack((self.eeg[idx_c_l,:,:][:,idx_l_elec], self.eeg[idx_c_r,:,:][:,idx_l_elec]))
			else:
				# stimuli presented bilataral
				ipsi = np.vstack((self.eeg[idx_c_l,:,:][:,idx_l_elec], self.eeg[idx_c_r,:,:][:,idx_r_elec]))
				contra = np.vstack((self.eeg[idx_c_l,:,:][:,idx_r_elec], self.eeg[idx_c_r,:,:][:,idx_l_elec]))

			# baseline correct data	
			ipsi = self.baselineCorrect(ipsi, self.times, self.baseline)
			contra = self.baselineCorrect(contra, self.times, self.baseline)

			# create erp (nr_elec X nr_timepoints)
			if cnd != 'all' and balance:
				idx_balance = np.random.permutation(ipsi.shape[0])[:max_trial]
				ipsi = ipsi[idx_balance,:,:]
				contra = contra[idx_balance,:,:]
			
			ipsi = np.mean(ipsi, axis = (0,1)) 
			contra = np.mean(contra, axis = (0,1))

			erps[str(sj)][cnd].update({'ipsi':ipsi,'contra':contra,'diff_wave':contra - ipsi, 'elec': [l_elec, r_elec]})	

		# save erps	
		with open(self.FolderTracker(['erp',self.header],'{}.pickle'.format(erp_name)) ,'wb') as handle:
			pickle.dump(erps, handle)	

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
				print 'no topo data found for {}'.format(cnd)
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





