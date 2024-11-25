"""
analyze EEG data

Created by Dirk van Moorselaar on 30-03-2016.
Copyright (c) 2016 DvM. All rights reserved.
"""

import json
from sre_constants import MAX_REPEAT
import time
import glob
import os
import pickle
import mne
import random
import warnings
import matplotlib
matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Generic, Union, Tuple, Any
from eeg_analyses.BDM import *  
from support.FolderStructure import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import pi, sqrt
from mne.filter import filter_data
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage.measurements import label

from support.support import select_electrodes, trial_exclusion, get_time_slice
from IPython import embed 


class CTF(BDM):
	'''
	Spatial encoding scripts modeled after: "The topography of alpha-band 
	activity tracks the content of spatial working memory" by Foster et al. 
	(2015). Scipts based on Matlab scripts published on open science Framework 
	(https://osf.io/bwzjj/) and lab visit to Chicago University 
	(spring quarter 2016).
	'''

	def __init__(self,sj:int,epochs:mne.Epochs,beh:pd.DataFrame, 
				to_decode:str,nr_bins:int,nr_chans:int,shift_bins:int=0,
				nr_iter:int=10,nr_folds:int=3,elec_oi:Union[str,list]='all',
				sin_power = 7,delta:bool=False,method:str='Foster',
				avg_ch:bool=True,ctf_param:Union[str,bool]='slope',
				power:str='band',min_freq:int=4,max_freq:int=40,
				num_frex:int=25,freq_scaling:str='log',slide_window:int=0,
				laplacian:bool=False,pca_cmp:int=0,avg_trials:int=1,
				baseline:Optional[tuple]=None,seed:Union[int, bool] = 42213):
		''' 
		
		Arguments
		- - - - - 
		beh (DataFrame): behavioral infor across epochs
		eeg (mne object): eeg object
		channel_folder (str): folder specifying which electrodes are used for CTF analysis (e.g. posterior channels)
		decoding (str)): String specifying what to decode. 
		but can be changed to different locations (e.g. decoding of the location of an intervening stimulus).
		nr_iter (int):  number iterations to apply forward model 	
		nr_folds (str): number of folds used for cross validation
		nr_bins (int): number of location bins used in experiment
		nr_chans (int): number of hypothesized underlying channels
		delta (bool): should basisset assume a shape of CTF or be a delta function
		power (str): shuould model be run on filtered data or on raw voltages 

		Returns
		- - - -
		self (object): SpatialEM object
		'''

		self.sj = sj
		self.beh = beh
		self.epochs = epochs
		self.to_decode = to_decode
		self.elec_oi = elec_oi
		self.ctf_param = ctf_param
		self.avg_ch = avg_ch
		self.cross = False
		self.baseline = baseline
		self.seed = seed
		self.shift_bins = shift_bins

		# specify model parameters
		self.method = method
		self.nr_bins = nr_bins													# nr of spatial locations 
		self.nr_chans = nr_chans 												# underlying channel functions coding for spatial location
		self.nr_iter = nr_iter													# nr iterations to apply forward model 		
		self.nr_folds = nr_folds												# nr blocks to split up training and test data with leave one out test procedure
		self.sfreq = 512														# shift of channel position
		self.power = power
		self.min_freq = min_freq
		self.max_freq = max_freq
		self.num_frex = num_frex
		self.freq_scaling = freq_scaling
		self.slide_wind = slide_window
		self.laplacian = laplacian
		self.pca=pca_cmp
		self.avg_trials = avg_trials
		# hypothesized set tuning functions underlying power measured across electrodes
		self.basisset = self.calculate_basis_set(self.nr_bins, self.nr_chans,
													sin_power,delta)
		
	def calculate_basis_set(self,nr_bins:int,nr_chans:int, 
			 				sin_power:int=7,delta:bool=False)->np.array:
		"""
		calculateBasisset returns a basisset that is used to reconstruct 
		location-selective CTFs from the topographic distribution 
		of oscillatory power across electrodes. It is assumed that power 
		measured at each electrode reflects the weighted sum of 
		a specific number of spatial channels (i.e. neural populations), 
		each tuned for a different angular location. 

		The basisset either assumes a particular shape of the CTF. 
		In this case the profile of each channel across angular 
		locations is modeled as a half sinusoid raised to the sin_power, 
		given by: 

		R = sin(0.50)**sin_power

		, where 0 is angular location and R is response of the spatial 
		channel in arbitrary units. If no shape is assumed the channel 
		response for each target position is set to 1, while setting all 
		the other channel response to 0 (delta function).

		The R is then circularly shifted for each channel such that the 
		peak response of each channel is centered over one of the 
		location bins.
		
		Arguments
		- - - - - 
		nr_bins (int): 
		nr_chans(int): assumed nr of spatial channels 
		sin_power (int): power for sinussoidal function

		Returns
		- - - -
		bassisset(array): set of basis functions used to predict channel 
		responses in forward model
		'''

		Args:
			nr_bins (int, optional): nr of spatial locations to decode. 
			nr_chans (int, optional): assumed nr of spatial channels. 
			sin_power (int, optional): power for sinussoidal function. 
			delta (bool, optional): defines whether the ctf has a 
			particular shape (as set by sin_power), or alternatively is 
			a stick function. 

		Returns:
			bassisset (np.array): basisset used in during forward 
			encoding
		"""

		# specify basis set
		if nr_bins % 2 == 0:
			x = np.linspace(0, 2*pi - 2*pi/nr_bins, nr_bins)
		else:
			x = np.linspace(0, 2*pi - 2*pi/nr_bins, nr_bins) 
			x += np.deg2rad(180/nr_bins)

		# c_centers = np.linspace(0, 2*pi - 2*pi/nr_chans, nr_chans)			
		# c_centers = np.rad2deg(c_centers)
		
		if delta:
			pred = np.zeros(nr_bins)
			pred[-1] = 1
		else:
			# hypothetical channel responses
			pred = np.sin(0.5*x)**sin_power									
		
		# shift the initial basis function
		if nr_bins % 2 == 0:												
			pred = np.roll(pred,nr_chans - (np.argmax(pred) + 1))
		else:
			pred = np.roll(pred,np.argmax(pred))	

		basisset = np.zeros((nr_chans,nr_bins))
		for c in range(nr_chans):
			basisset[c,:] = np.roll(pred,c+1)
		
		return basisset	
	
	def spatial_ctf(self,pos_labels:dict ='all',cnds:dict=None, 
					excl_factor:dict=None,window_oi:tuple=(None,None),
					freqs:dict='main_param',
					downsample:int = 1,GAT:bool=False, 
					nr_perm:int= 0,collapse:bool=False,name:int='main'):
		"""
		calculate spatially based channel tuning functions across 
		conditions. Training and testing can be done either witin or 
		across conditions (see cnds argument)

		Args:
			pos_labels (dict, optional): key, item pair where key points 
			to the column in beh that contains position labels for ctf
			analysis, and the item is a list that contains all values 
			that will be considered in the ctf analysis. Defaults to all 
			(i.e., all values as specified in the class argument 
			to_decode will be considered in the ctf analysis). 
			cnds (dict, optional): key, item pair where key points to 
			the column in beh that contains conditions of  interest, and 
			the item is a list that contains conditions of interets. 
			To run a cross decoding analysis, the first value in the 
			cnds dict needs to be a list specifying the conditions to 
			train the model. Currently only a single condition can be 
			tested per run. Defaults to None (i.e., include all trials)
			excl_factor (dict, optional): This gives the option to 
			exclude specific conditions from analysis. For example, 
			to only include trials where the cue was pointed to the left 
			and not to the right specify the following: 
			factor = dict('cue_direc': ['right']). 
			Mutiple column headers and multiple variables per header can 
			be specified . Defaults to None.
			window_oi (tuple, optional): time window of interest 
			(start to end) for ctf analysis. Time window will  be 
			cropped after filtering. Defaults to (None,None).
			freqs (dict, optional): key, item pair where key is the name 
			of the frequency band of interest, and the item is a list 
			with the lower and upper frequency. Defaults to main_param, 
			which will use the time frequency settings as specified in 
			the CTF class. 
			downsample (int, optional): factor to downsample data 
			(is applied after filtering). Defaults to 1.
			GAT (bool, optional): perform encoding across
			all combinations of timepoints (i.e., generate a 
			generalization across time encoding matrix).
			nr_perm (int, optional): _description_. Defaults to 0.
			collapse (bool, optional): _description_. Defaults to False.
			name (int, optional): _description_. Defaults to 'main'.
		"""
 
		# read in data
		epochs = self.epochs.copy()
		beh = self.beh.copy()
		if cnds is None:
			(cnd_head,_) = (None,['all_data'])
		else:
			(cnd_head,_), = cnds.items()
		headers = [cnd_head]
		epochs, beh = self.select_ctf_data(epochs, beh,self.elec_oi, 
											headers, excl_factor)

		# set params
		nr_itr = self.nr_iter * self.nr_folds
		ctf_name = f'{self.sj}_{name}'
		nr_perm += 1
		nr_elec = len(epochs.ch_names)
		tois = get_time_slice(epochs.times, window_oi[0], window_oi[1])
		times_oi = epochs.times[tois]
		nr_samples = epochs.times[tois][::downsample].size
		ctf, info = {}, {}
		freqs, nr_freqs = self.set_frequencies(freqs)
		data_type = 'power' if freqs != ['raw'] else 'raw'
		if self.method == 'k-fold':
			# TODO: fix
			print('Method not yet  implemented')
			print('nr_folds is irrelevant and will be reset to 1')
			#self.nr_folds = 1						
		if collapse:
			pass
			# TODO: fix
			#conditions += ['all_trials']
	
		if type(cnds) == dict:
			train_cnds, test_cnd = self.check_cnds_input(cnds)
			if test_cnd is not None:
				self.cross = True
				nr_itr = self.nr_iter
		else:
			train_cnds = ['all_data']
			
		# based on conditions get position bins
		(pos_bins, 
		cnds, 
		epochs, 
		max_tr) = self.select_ctf_labels(epochs, beh, pos_labels, cnds)

		if GAT:
			print('Creating generalization across time matrix.',
	 			'This may take some time and is generally not ',
				'recommended')
			
		# Frequency loop (ensures that data is only filtered once)
		for fr in range(nr_freqs):
			print('Frequency {} out of {}'.format(str(fr + 1), str(nr_freqs)))

			# Time-Frequency Analysis
			E, T = self.tf_decomposition(epochs.copy(),freqs[fr],tois,downsample)

			# Loop over conditions
			for c, cnd in enumerate(train_cnds):
				print(f'Running ctf for {cnd} condition')

				# preallocate arrays
				if GAT:
					C2_E = np.zeros((nr_perm,nr_freqs, nr_itr,
									nr_samples-self.slide_wind,
									nr_samples-self.slide_wind,self.nr_bins, 
									self.nr_chans))
					W_E = np.zeros((nr_perm,nr_freqs, nr_itr,
									nr_samples-self.slide_wind, 
									nr_samples-self.slide_wind,
									self.nr_chans, 
									nr_elec))						
				else:
					C2_E = np.zeros((nr_perm,nr_freqs, nr_itr,
									nr_samples-self.slide_wind,self.nr_bins, 
									self.nr_chans))
					W_E = np.zeros((nr_perm,nr_freqs, nr_itr,
									nr_samples-self.slide_wind, self.nr_chans, 
									nr_elec))							 
				C2_T, W_T  = C2_E.copy(), W_E.copy()				 
	
				# partition data into training and testing sets
				# is done once to ensure each frequency has the same sets
				if fr == 0:
					# update ctf dicts to keep track of output	
					info.update({cnd:{}})
					ctf.update({cnd:{'C2_E':C2_E,'C2_T':C2_T,
									'W_E':W_E,'W_T':W_T}})
					# get condition indices
					cnd_idx = cnds == cnd
					if self.cross:
						test_idx = cnds == test_cnd
						test_bins = np.unique(pos_bins[test_idx])
						(train_idx, 
						test_idx) = self.train_test_cross(pos_bins, cnd_idx,
														test_idx, self.nr_iter)
					else:
						(train_idx, 
						test_idx) = self.train_test_split(pos_bins,
														cnd_idx,max_tr)
						test_bins = np.unique(pos_bins)

					info[cnd]['train_idx'] = train_idx
					info[cnd]['test_idx'] = test_idx
					if self.method == 'Foster':
						C1 = np.empty((self.nr_bins * (self.nr_folds - 1), 
										self.nr_chans)) * np.nan
					else:
						C1 = self.basisset

				# iteration loop
				for itr in range(nr_itr):

					# TODO: insert permutation loop
					p = 0
					train_idx = info[cnd]['train_idx'][itr]

					# initialize evoked and total power arrays
					bin_te_E = np.zeros((self.nr_bins, nr_elec, nr_samples)) 
					bin_te_T = bin_te_E.copy()
					if self.method == 'k-fold':
						pass
						#TODO: implement 
					elif self.method == 'Foster':
						nr_itr_tr = self.nr_bins * (self.nr_folds - 1)
						bin_tr_E = np.zeros((nr_itr_tr, nr_elec, nr_samples)) 
						bin_tr_T = bin_tr_E.copy()
						
					# position bin loop
					bin_cnt = 0
					for bin in range(self.nr_bins):
						if bin in test_bins:
							test_idx=np.squeeze(info[cnd]['test_idx'][itr][bin])
							bin_te_T[bin] = np.mean(T[test_idx], axis = 0)
							bin_te_E[bin] = self.extract_power(np.mean(\
											E[test_idx], axis = 0),freqs[fr])

						if self.method == 'Foster':
							for j in range(self.nr_folds - 1):
								evoked = self.extract_power(np.mean(\
											E[train_idx[bin][j]], axis = 0),
											freqs[fr])
								bin_tr_E[bin_cnt] = evoked
								total = np.mean(T[train_idx[bin][j]], axis = 0)
								bin_tr_T[bin_cnt] = total
								C1[bin_cnt] = self.basisset[bin]
								bin_cnt += 1
						elif self.method == 'k-fold':
							pass
							#TODO: implement
					
					(ctf[cnd]['C2_E'][p,fr,itr], 
					ctf[cnd]['W_E'][p,fr,itr],
					ctf[cnd]['C2_T'][p,fr,itr],
					ctf[cnd]['W_T'][p,fr,itr]) = self.forward_model_loop(
															bin_tr_E, 
															bin_te_E,
															bin_tr_T, 
															bin_te_T,C1,GAT)
		# take the average across model iterations
		for cnd in train_cnds:
			for key in ['C2_E','C2_T','W_E','W_T']:
				ctf[cnd][key] = ctf[cnd][key].mean(axis = 2)

			if freqs == ['raw']:
				ctf[cnd]['C2_raw'] = ctf[cnd].pop('C2_E')
				ctf[cnd]['W_raw'] = ctf[cnd].pop('W_E')
				del ctf[cnd]['C2_T']
				del ctf[cnd]['W_T']

		# save output
		times_oi = epochs.times[tois][::downsample]
		if self.slide_wind > 0:
			times_oi = times_oi[:-self.slide_wind]
		if self.ctf_param:
			print('get ctf tuning params')
			ctf_param = self.get_ctf_tuning_params(ctf,self.ctf_param,
					  							data_type, GAT,
												avg_ch=self.avg_ch,
												test_bins=test_bins)
			
			ctf_param.update({'info':{'times':times_oi}})

			with open(self.folder_tracker(['ctf',self.to_decode], 
					fname=f'ctf_param_{ctf_name}.pickle'),'wb') as handle:
				print('saving ctf params')
				pickle.dump(ctf_param, handle)

		ctf.update({'info':{'times':times_oi}})
		with open(self.folder_tracker(['ctf',self.to_decode], 
				fname = f'ctfs_{ctf_name}.pickle'),'wb') as handle:
			print('saving ctfs')
			pickle.dump(ctf, handle)

		with open(self.folder_tracker(['ctf',self.to_decode], 
				fname = f'ctf_info_{ctf_name}.pickle'),'wb') as handle:
			pickle.dump(info, handle)
		
	def select_ctf_data(self,epochs:mne.Epochs,beh:pd.DataFrame,
						elec_oi:Union[list, str]= 'all',headers:list = [],
						excl_factor:dict = None) -> Tuple[mne.Epochs,
															pd.DataFrame]:
		"""
		Selects the data of interest by selecting epochs and electrodes
		of interest

		Args:
			elec_oi ([list, str], optional): Electrodes used for 
			ctf analysis. Defaults to 'all'.
			excl_factor (dict, optional): exclusion dictionary that 
			allows for dropping of certain conditions. Defaults to None.

		Returns:
			epochs (mne.Epochs): EEG data used for ctf analysis
			beh (pd.DataFrame): behavioral parameters
		"""

		# if specified remove trials matching specified criteria
		if excl_factor is not None:
			beh, epochs = trial_exclusion(beh, epochs, excl_factor)

		# if not already done reset index (to properly align beh and epochs)
		beh.reset_index(inplace = True, drop = True)

		if self.laplacian:
			epochs = mne.preprocessing.compute_current_source_density(epochs)

		# if specified # average across trials
		(epochs, 
		beh) = self.average_trials(epochs,beh,[self.to_decode] + headers) 

		# limit analysis to electrodes of interest
		picks = select_electrodes(epochs.ch_names, elec_oi) 
		epochs.pick(picks)

		return epochs, beh

	def check_cnds_input(self, cnds:dict)-> Tuple[list, str]:
		"""
		checks whether ctf should be run within or across conditions

		Args:
			cnds (dict): dictionary with conditions of interest as 
			values

		Returns:
			train_cnds (list): Conditions used train and test the model.
			In case test_cnd is not None, these conditions will be used 
			solely to train the model
			test_cnd (str): condition to test the model
		"""
		
		(_, cnds_oi), = cnds.items()
		if type(cnds_oi[0]) == list:
			train_cnds, test_cnd = cnds_oi
		else:
			train_cnds, test_cnd = cnds_oi, None

		return train_cnds, test_cnd

	def select_ctf_labels(self, epochs: mne.Epochs, beh: pd.DataFrame, 
						pos_labels: dict, cnds: dict) -> Tuple[np.array, 
															np.array,
															np.array, int]:
		"""
		Selects data of interest by selecting all specified position 
		(in pos_labels) and conditions (in cnds). Function also ensures 
		that there are no more unique position bins then the number of 
		bins used to calculate ctfs

		Args:
			epochs (mne.Epochs): Epochs object 
			beh (pd.DataFrame): dataframe with behavioral parameters
			pos_labels (dict): key, item pair where key points to the 
			column in beh that contains labels for ctf analysis, and 
			the item contains all values that will be considered in the 
			ctf analysis
			cnds (dict): key, item pair where key points to the column
			in beh that contains condition labels, and the item contains
			all unique conditions that are considered

		Returns:
			pos_bins (np.array): array with position labels
			cnds (np.array): array with condition labels
			epochs (mne.Epochs): eeg data of interest
			max_tr (int): maximum number of trials to include per
			position 
			label (ensures that labels are balanced across conditions)
		"""

		# extract conditions of interest
		if cnds is None:
			cnds = np.array(['all_data']*beh.shape[0])
			cnd_idx = np.arange(beh.shape[0])
		else:
			(cnd_header, cnds_oi), = cnds.items()
			cnd_idx = np.where(beh[cnd_header].isin(np.hstack(cnds_oi)))[0]
			cnds = beh[cnd_header]
		
		# extract position labels of interest
		if pos_labels != 'all':
			(pos_header, pos_labels), = pos_labels.items()
			pos_idx =  np.where(beh[pos_header].isin(pos_labels))[0]
		else:
			pos_header = self.to_decode
			pos_idx = np.arange(beh.shape[0]) 
		
		# get data of interest
		idx = np.intersect1d(cnd_idx, pos_idx)
		pos_bins = beh[pos_header].values[idx]
		cnds = cnds[idx]
		epochs = epochs[idx]
		pos_bins, cnds, epochs = self.select_bins_oi(pos_bins, cnds, epochs)
		max_tr = self.set_max_trial(cnds, pos_bins, self.method)

		return pos_bins, cnds, epochs, max_tr

	def select_bins_oi(self, pos_bins: np.array, cnds: np.array, 
					  epochs: mne.Epochs) -> Tuple[np.array, 
					  							  np.array, np.array]:
		"""
		Function ensures that there are no ctf labels that exceed 
		the number of bins as specified at class initialization

		Args:
			pos_bins (np.array): array with position labels
			cnds (np.array): array with condition labels
			epochs (mne.Epochs): eeg data of interest

		Returns:
			pos_bins (np.array): array with position labels
			cnds (np.array): array with condition labels
			epochs (mne.Epochs): eeg data of interest
		"""

		mask = np.logical_and(pos_bins >= 0,pos_bins < self.nr_bins)
		pos_bins = pos_bins[mask]
		cnds = cnds[mask]
		epochs = epochs[np.where(mask)[0]]

		return pos_bins, cnds, epochs

	def set_max_trial(self, cnds: np.array, pos_bins: np.array, 
					 method: str) -> int:
		"""
		Determines the maximum number of trials that can be used in 
		the block/fold assignment of the forward model such that the 
		number of trials used in the IEM is equated across conditions

		Args:
			cnds (np.array): array with condition labels
			pos_bins (np.array): array with position labels
			method (str): method used for block assignment

		Returns:
			nr_per_bin (int): maximum number of trials per cell used to
			create train and test data
		"""

		nr_cnds = np.unique(cnds).size

		if nr_cnds == 1:
			_, bin_count = np.unique(pos_bins, return_counts = True)
		else:
			bin_count = np.zeros((nr_cnds,self.nr_bins))
			for c, cnd in enumerate(np.unique(cnds)):
				temp_bin = pos_bins[cnds == cnd]
				for b in range(self.nr_bins):
					bin_count[c,b] = sum(temp_bin == b)

		# select cell with lowest number of observations
		min_count = np.min(bin_count)
		if method == 'Foster':
			nr_per_bin = int(np.floor(min_count/self.nr_folds))						
		elif method == 'k-fold':
			nr_per_bin = int(np.floor(min_count/self.nr_iter)*self.nr_iter)

		return nr_per_bin

	def extract_power(self,x:np.array,band:Union[str,list]=None)->np.array:
		"""
		To extract power the signal is computed by squaring the complex 
		magnitude of the complex signal. 

		Args:
			x (np.array): complex signal
			band (str|list): if band is raw no power is extracted

		Returns:
			power_x (np.array): power extracted from 
	
		"""
		if isinstance(band, (list,tuple)):
			power = abs(x)**2
		else:
			power = x

		return power

	def tf_decomposition(self, epochs: mne.Epochs,  band: tuple, tois: slice, 
					  downsample: int) -> Tuple[np.array, np.array]:
		"""
		Isolates frequency-specific activity using a 5th order 
		butterworth filter. A Hilbert Transform is then applied to the 
		filtered data to extract the complex analytic signal. 
		To extract power this signal is computed by squaring the complex 
		magnitude of the complex signal. 

		Args:
			epochs (mne.Epochs): epochs object used for TF analysis
			band (tuple): frequency band used for TF decomposition
			tois (slice): time windows of interest
			downsample (int): factor to downsample the data 
			(after filtering)

		Returns:
			E (np.array): analytic representation obtain with hilbert
			transform (used to calculated evoked activity: ongoing 
			activity irrespective of phase relation to stimulus onset)
			T (np.array): squared magnitude of E. Used to calculated 
			total power ctf	
		"""

		# initiate arrays for evoked and total power
		_, nr_chan, nr_time = epochs._data.shape

		# extract power using hilbert or wavelet convolution
		if isinstance(band, (list,tuple)):
			if self.power == 'band':
				epochs.filter(band[0], band[1], method = 'iir', 
						iir_params = dict(ftype = 'butterworth', order = 5))
				epochs.apply_hilbert()
				E = epochs._data
				T = self.extract_power(E,band)
			elif self.power == 'wavelet':
				print('Method not yet implemented')
				pass 
		elif band == 'raw':
			epochs.apply_baseline(baseline = self.baseline)
			T = epochs._data
			E = epochs._data

		# trim filtered data (after filtering to avoid artifacts)
		E = E[:,:,tois]
		T = T[:,:,tois]

		# downsample 
		if band != 'raw':
			E = E[:,:,::downsample]
			T = T[:,:,::downsample]
		else:
			#TODO: addd mne downsampling method
			E = E[:,:,::downsample]
			T = T[:,:,::downsample]		

		return E, T

	def forward_model(self, train_X: np.array, test_X: np.array, 
					C1: np.array) -> Tuple[np.array, np.array]:
		"""
		Applies an inverted encoding model (IEM) to each sample. This 
		routine proceeds in two stages (train and test).
		1. In the training stage, training data (B1) is used to estimate 
		weights (W) that approximate the relative contribution of each 
		spatial channel to the observed response measured at each 
		electrode, with:

		B1 = W*C1

		,where B1 is power at each electrode, C1 is the predicted 
		response of each spatial channel, and W is weight matrix that 
		characterizes a linear mapping from channel space to electrode 
		space. The equation is solved via least-squares estimation.
		
		2. In the test phase, the model is inverted to transform the 
		observed test data B2 into estimated channel responses. The 
		estimated channel response are then shifted to a common center 
		by aligning the estimate channel response to the channel tuned 
		for the stimulus bin. 

		Args:
			train_X (np.array): train data (epochs X elec X samples)
			test_X (np.array): test data (epochs X elec X samples)
			C1 (np.array):basisset

		Returns:
			C2s (np.array): shifted predicted channels response
			W (np.array): weight matrix
		"""

		# check input shapes and adjust data shapes if needed
		## TODO: potentially remove
		if train_X.ndim == 3:
			train_X = train_X.mean(axis=-1)
			test_X = test_X.mean(axis=-1)

		# apply forward model
		B1 = train_X
		B2 = test_X

		if self.pca:
			pca = PCA(n_components=self.pca, svd_solver = 'full').fit(B1)
			B1 = pca.transform(B1)
			B2 = pca.transform(B2)

		# estimate weight matrix W (nr_chans x nr_electrodes)
		W, resid_w, rank_w, s_w = np.linalg.lstsq(C1,B1, rcond = -1)
		# estimate channel response C2 (nr_chans x nr test blocks)		
		C2, resid_c, rank_c, s_c = np.linalg.lstsq(W.T,B2.T, rcond = -1)	
		
		# TRANSPOSE C2 so that we average across channels 
		# rather than across position bins
		C2 = C2.T
		C2s = np.zeros(C2.shape)

		# shift tunings to common center
		bins = np.arange(self.nr_bins)
		nr_2_shift = int(np.ceil(C2.shape[1]/2.0))
		for i in range(C2.shape[0]):
			idx_shift = abs(bins - bins[i]).argmin()
			shift = idx_shift - nr_2_shift
			if self.nr_bins % 2 == 0:							
				C2s[i,:] = np.roll(C2[i,:], - shift)	
			else:
				C2s[i,:] = np.roll(C2[i,:], - shift - 1)

		# shift the predicted channel responses
		# to a common spatial reference frame
		C2s = np.roll(C2s, shift = self.shift_bins, axis = 0)
		W = np.roll(W, shift = self.shift_bins, axis = 0)

		return C2s, W 

	def forward_model_loop(self,E_train:np.array,E_test:np.array,
							T_train:np.array,T_test: np.array,C1: np.array,
							GAT:bool)->Tuple[np.array, np.array,
													np.array,np.array]:
		"""
		Applies inverted encoding model (see forward_model) for each 
		individual time point

		Args:
			E_train (np.array): train data with evoked power
			E_test (np.array): test data with evoked power
			T_train (np.array): train data with total power
			T_test (np.array): test data with total power
			C1 (np.array): basisset
			GAT (bool): specifies whether training and testing is done
			across all combiantions of time points or only along the 
			diagonal of the generalization across time matrix (
			i.e., training and testing is done only at the same time 
			points)

		Returns:
			C2_E (np.array): evoked power based shifted predicted 
			channels response across samples
			W_E (np.array): evoked power weight matrices across samples
			C2_T (np.array):total power based shifted predicted channels 
			response across samples
			W_T (np.array): total power weight matrices across samples
		"""

		# set necessary parameters
		slide = 1+self.slide_wind	
		nr_bins = self.nr_bins
		_, nr_elec, nr_samples_tr = E_train.shape 
		nr_samples_tr -= self.slide_wind
		if GAT:
			nr_samples_te = E_test.shape[-1] - self.slide_wind
		else:
			nr_samples_te = 1	

		# initialize output arrays
		C2_E = np.zeros((nr_samples_tr,nr_samples_te, nr_bins, self.nr_chans))
		W_E =  np.zeros((nr_samples_tr,nr_samples_te, nr_bins,nr_elec))
		C2_T, W_T = C2_E.copy(), W_E.copy()

		# TODO: parallelize loop
		for tr_t in range(nr_samples_tr):
			for te_t in range(nr_samples_te):
				if not GAT:
					te_t = tr_t

				# evoked power model fit
				c2_e, w_e = self.forward_model(E_train[...,tr_t:tr_t+slide],
											E_test[...,te_t:te_t+slide], 
											C1)			

				# total power model fit
				c2_t, w_t = self.forward_model(T_train[...,tr_t:tr_t+slide], 
												T_test[...,te_t:te_t+slide], 
												C1)
				
				if not GAT:
					te_t = 0
				C2_E[tr_t,te_t] = c2_e
				W_E[tr_t,te_t,:,:w_e.shape[1]] = w_e
				C2_T[tr_t,te_t] = c2_t
				W_T[tr_t,te_t,:,:w_t.shape[1]] = w_t
		
		C2_E = np.squeeze(C2_E)
		W_E = np.squeeze(W_E)
		C2_T = np.squeeze(C2_T)
		W_T = np.squeeze(W_T)

		return C2_E, W_E, C2_T, W_T

	def train_test_cross(self, pos_bins: np.array,train_idx:np.array,
						test_idx:np.array,
						nr_iter:int)->Tuple[np.array, np.array]:
		"""
		selects trial indices for the train and the test set

		Args:
			pos_bins (np.array): array with position bins
			train_idx (np.array): array with indices for training 
			condition
			test_idx (np.array): array with indices for test condition
			nr_iter (int): number of iterations in cross validation

		Returns:
			train_idx (np.array): indices used to train the model (with 
			bin labels balanced)
			test_idx (np.array): indices used to test the model (with 
			bin labels balanced)
		"""

		if self.seed:
			np.random.seed(self.seed) # set seed 

		# select training data (ensure that number of bins is balanced)
		train_bins = pos_bins[train_idx]
		train_idx = np.arange(pos_bins.size)[train_idx]
		bins,  counts = np.unique(train_bins, return_counts=True)
		min_obs = min(counts)
		train_idx = [np.random.choice(train_idx[train_bins==b],min_obs,False) 
															for b in bins]
		
		#randomly split training blocks in half for each iteration
		train_idx = np.stack(train_idx)[:,None,:]
		split_arrays = []
		if train_idx.shape[-1] % 2 == 0:
			to_split = train_idx.shape[-1] 
		else:
			to_split = train_idx.shape[-1] -1 
		split = to_split // 2
		for i in range(nr_iter):
			split_idx = np.random.permutation(train_idx.shape[-1])[:split*2]
			splitted = np.split(train_idx[..., split_idx],[split],axis=-1)
			split_arrays.append(np.concatenate(splitted, axis=1))
		
		train_idx = np.stack(split_arrays)

		# select test data (ensure that number of bins is balanced)
		test_bins = pos_bins[test_idx]
		idx = np.arange(pos_bins.size)[test_idx]
		bins,  counts = np.unique(test_bins, return_counts=True)
		min_obs = min(counts)
		idx = [np.random.choice(idx[test_bins==b],min_obs,False) 
															for b in bins]
		# add new axis so that trial indexing does not crash
		if bins.size == np.unique(train_bins).size:											
			test_idx = np.stack(idx)[None,:,None,:]
		else:
			test_idx = np.zeros((np.unique(train_bins).size, min_obs),
								dtype = int)
			bin_cnt = 0
			for bin in np.unique(train_bins):
				if bin in bins:
					test_idx[int(bin)] = idx[bin_cnt]
					bin_cnt += 1
			test_idx = test_idx[None,:,None,:]

		#TODO: check whether test data needs to be split up in iterations
		test_idx = np.tile(test_idx, (nr_iter, 1, 1, 1))

		return train_idx, test_idx	

	def train_test_split(self, pos_bins: np.array,cnd_idx: np.array, 
						max_tr: int)->Tuple[np.array, np.array]:
	
		'''
		creates a random block assignment (train and test blocks)

		Arguments
		- - - - - 
		pos_bin (array): array with position bins
		idx_tr (array): array of trial numbers corresponding to pos_bin 
		labels
		nr_per_bin(int): maximum number of trials to be used per 
		location bin
		nr_iter (int): number of iterations used for IEM
		Returns
		- - - -
		blocks(array): randomly assigned block indices
		'''

		if self.seed:
			np.random.seed(self.seed) # set seed 
		
		# get trial count and condition indices
		cnd_bins = pos_bins[cnd_idx]
		nr_tr = cnd_bins.size
		trial_idx = np.arange(pos_bins.size)[cnd_idx]
		# initiate array
		bl_assign = np.zeros((self.nr_iter, nr_tr))

		# loop over iterations
		for i in range(self.nr_iter):
			#  initiate new array for each block assignment
			blocks = np.empty(nr_tr) * np.nan
			shuf_blocks = np.empty(nr_tr) * np.nan

			idx_shuf = np.random.permutation(nr_tr) 			
			shuf_bin = cnd_bins[idx_shuf]

			# take the 1st max_tr x nr_blocks trials for each position bin
			for bin in range(self.nr_bins):
				idx = np.where(shuf_bin == bin)[0] 	
				idx = idx[:max_tr * self.nr_folds] 
				x = np.tile(np.arange(self.nr_folds),(max_tr,1))
				shuf_blocks.flat[idx] = x	

			# unshuffle block assignment and save to CTF
			blocks[idx_shuf] = shuf_blocks	
			bl_assign[i] = blocks
		tr_per_block = int(sum(blocks == 0)/self.nr_bins)

		# after block assignment split into train and test set
		train_idx = np.zeros((self.nr_iter * self.nr_folds, self.nr_bins,
							 self.nr_folds - 1, tr_per_block), dtype = int)									
		test_idx = np.zeros((self.nr_iter * self.nr_folds, self.nr_bins, 
						  tr_per_block), dtype = int)

		idx = 0	
		for i in range(self.nr_iter):
			for bl in range(self.nr_folds):
				for bin in range(self.nr_bins):	
					test_mask = (bl_assign[i] == bl) * (cnd_bins == bin)
					test_idx[idx,bin] = trial_idx[test_mask]
					train_mask = ((~np.isnan(bl_assign[i])) * 
								 (bl_assign[i] != bl) * (cnd_bins == bin))
					# split all train data into seperate train blocks
					train = np.array_split(trial_idx[train_mask], 
														self.nr_folds - 1)
					for j in range(self.nr_folds - 1):
						train_idx[idx, bin, j] = train[j]
				idx += 1

		return train_idx, test_idx

	def set_frequencies(self, freqs):
		# DOUBLE CHECK THIS: self.power=='band'???

		if freqs == 'main_param':
			if self.freq_scaling == 'log':
				frex = np.logspace(np.log10(self.min_freq), 
								np.log10(self.max_freq), 
							  	self.num_frex)
			elif self.freq_scaling == 'linear':
				frex = np.linspace(self.min_freq,self.max_freq,self.num_frex)
			if self.power == 'band':
				frex = [(frex[i], frex[i+1]) for i in range(self.num_frex -1)]
		elif type(freqs) == dict:
			frex = [(freqs[band][0],freqs[band][1]) for band in freqs.keys()]
		elif freqs == 'raw':
			frex = [freqs]
		nr_frex = len(frex)

		return frex, nr_frex

	def localizer_spatial_ctf(self,pos_labels_tr:dict ='all',
							pos_labels_te:dict ='all',
							freqs:dict='main_param',te_cnds:dict=None,
							te_header:str=None,
							window_oi_tr:tuple=None,window_oi_te:tuple=None,
							excl_factor_tr:dict=None,excl_factor_te:dict=None,
							downsample:int = 1,nr_perm:int=0,GAT:bool=False,
							name:str='loc_ctf'):

		# set train and test data
		epochs_tr, beh_tr = self.select_ctf_data(self.epochs[0], self.beh[0],
												self.elec_oi,excl_factor_tr)

		epochs_te, beh_te = self.select_ctf_data(self.epochs[1], self.beh[1],
												self.elec_oi, excl_factor_te)

		if window_oi_tr is None:
			window_oi_tr = (epochs_tr.tmin, epochs_tr.tmax)

		if window_oi_te is None:
			window_oi_te = window_oi_tr

		nr_itr = 1
		ctf_name = f'{self.sj}_{name}'
		nr_perm += 1
		nr_elec = len(epochs_tr.ch_names)
		tois_tr = get_time_slice(epochs_tr.times,
								window_oi_tr[0],window_oi_tr[1])
		tois_te = get_time_slice(epochs_te.times,
								window_oi_te[0],window_oi_te[1])
		
		avg_tr = True if not GAT else False
		GAT = True
			
		ctf, info = {}, {}
		freqs, nr_freqs = self.set_frequencies(freqs)
		data_type = 'power' if freqs != ['raw'] else 'raw'

		if type(te_cnds) == dict:
			(cnd_header, test_cnds), = te_cnds.items()
		else:
			test_cnds = ['all_data']
			cnd_header = None

		# set train and test data
		(pos_bins_tr, 
		_,
		epochs_tr, 
		_) = self.select_ctf_labels(epochs_tr, beh_tr, pos_labels_tr, None)

		(pos_bins_te, 
		cnds, 
		epochs_te, 
		_) = self.select_ctf_labels(epochs_te, beh_te, pos_labels_te, te_cnds)

		# Frequency loop (ensures that data is only filtered once)
		for fr in range(nr_freqs):
			print('Frequency {} out of {}'.format(str(fr + 1), str(nr_freqs)))

			# Time-Frequency Analysis 
			(E_tr, 
			T_tr) = self.tf_decomposition(epochs_tr.copy(),freqs[fr],
									 tois_tr,downsample)	
			nr_samp_tr = 1 if avg_tr else E_tr.shape[-1]
			(E_te, 
			T_te) = self.tf_decomposition(epochs_te.copy(),freqs[fr],
									 tois_te,downsample)	
			nr_samp_te = E_te.shape[-1]

			# Loop over conditions
			for c, cnd in enumerate(test_cnds):
				print(f'Running localizer ctf for condition: {cnd} ')	

				# preallocate arrays
				C2_E = np.zeros((nr_perm,nr_freqs, nr_samp_tr, nr_samp_te,
									self.nr_bins, self.nr_chans))
				W_E = np.zeros((nr_perm,nr_freqs, nr_samp_tr, nr_samp_te,
									self.nr_chans, nr_elec))												 
				C2_T, W_T  = C2_E.copy(), W_E.copy()	

				# partition data into training and testing sets
				# is done once to ensure each frequency has the same sets
				if fr == 0:
					# update ctf dicts to keep track of output	
					info.update({cnd:{}})
					ctf.update({cnd:{'C2_E':C2_E,'C2_T':C2_T,
									'W_E':W_E,'W_T':W_T}})

					# get train and test indices
					idx = np.arange(pos_bins_tr.size)
					(train_idx,_) = self.train_test_cross(pos_bins_tr,idx,
					   									None, nr_itr)
					idx = cnd == cnds
					(_,test_idx) = self.train_test_cross(pos_bins_te,idx,
					  									idx,nr_itr)
					test_bins = np.unique(pos_bins_te[test_idx])

					info[cnd]['train_idx'] = train_idx
					info[cnd]['test_idx'] = test_idx
					if self.method == 'Foster':
						C1 = np.empty((self.nr_bins * self.nr_folds, 
		     						   self.nr_chans)) * np.nan
					else:
						C1 = self.basisset	

				# TODO: insert permutation loop
				p = 0
				#TODO: make sure this works with iterations
				train_idx = info[cnd]['train_idx'][0]

				# initialize evoked and total power arrays
				bin_te_E = np.zeros((self.nr_bins, nr_elec, nr_samp_te)) 
				bin_te_T = bin_te_E.copy()
				if self.method == 'k-fold':
					pass
					#TODO: implement 
				elif self.method == 'Foster':
					nr_itr_tr = self.nr_bins * (self.nr_folds)
					bin_tr_E = np.zeros((nr_itr_tr, nr_elec, nr_samp_tr)) 
					bin_tr_T = bin_tr_E.copy()
					
				# position bin loop
				bin_cnt = 0

				for bin in range(self.nr_bins):
					if bin in test_bins:
						bin_idx = np.where(bin == test_bins)[0]
						test_idx = np.squeeze(info[cnd]['test_idx'][0][bin_idx])
						bin_te_E[bin] = abs(np.mean(E_te[test_idx], axis = 0))**2
						bin_te_T[bin] = np.mean(T_te[test_idx], axis = 0)

					if self.method == 'Foster':
						for j in range(self.nr_folds):
							evoked = abs(np.mean(E_tr[train_idx[bin][j]], 
												axis = 0))**2
							total = np.mean(T_tr[train_idx[bin][j]], axis = 0)
							if avg_tr:
								evoked = evoked.mean(axis = -1)[:, np.newaxis]
								total = total.mean(axis = -1)[:, np.newaxis]
							bin_tr_E[bin_cnt] = evoked
							bin_tr_T[bin_cnt] = total
							C1[bin_cnt] = self.basisset[bin]
							bin_cnt += 1
					elif self.method == 'k-fold':
						pass
						#TODO: implement

				(ctf[cnd]['C2_E'][p,fr], 
				ctf[cnd]['W_E'][p,fr],
				ctf[cnd]['C2_T'][p,fr],
				ctf[cnd]['W_T'][p,fr]) = self.forward_model_loop(
														bin_tr_E, 
														bin_te_E,
														bin_tr_T, 
														bin_te_T,C1,GAT)
				
		# take the average across model iterations
		for cnd in test_cnds:

			if freqs == ['raw']:
				ctf[cnd]['C2_raw'] = ctf[cnd].pop('C2_E')
				ctf[cnd]['W_raw'] = ctf[cnd].pop('W_E')
				del ctf[cnd]['C2_T']
				del ctf[cnd]['W_T']

		# save output
		with open(self.folder_tracker(['ctf',self.to_decode], 
				fname = f'ctfs_{ctf_name}.pickle'),'wb') as handle:
			print('saving ctfs')
			pickle.dump(ctf, handle)

		with open(self.folder_tracker(['ctf',self.to_decode], 
				fname = f'ctf_info_{ctf_name}.pickle'),'wb') as handle:
			pickle.dump(info, handle)	

		if self.ctf_param:
			print('get ctf tuning params')
			ctf_param = self.get_ctf_tuning_params(ctf,self.ctf_param,
					  							data_type, GAT,
												avg_ch=self.avg_ch)
			
			with open(self.folder_tracker(['ctf',self.to_decode], 
					fname=f'ctf_param_{ctf_name}.pickle'),'wb') as handle:
				print('saving ctf params')
				pickle.dump(ctf_param, handle)

	def summarize_ctfs_old(self, X, nr_freqs, nr_samps):
		'''	Captures a range of summary statistics of the channel tuning function. Slopes are calculated by 	
		collapsing across symmetric data points in the tuning curve. The collapsed data create a linear 
		increasing vector array from the tails to the peak of the tuning curve. A first order polynomial 
		is then fitted to this array to estimate the slope of the tuning curve. In addition, the tuning curve is
		fitted to a Gaussian. The amplitude, mean and sigma of this function are also computed. 

		Arguments:
			X {array} -- CTF data across frequencies and sample points (nr freqs, nr samps, nr chans) 
			nr_freqs {int} --  number of frequencies in CTF data
			nr_samps {int} -- number of sample points in CTF data
		
		Returns:
			[type] -- [description]
		'''

		if self.nr_chans % 2 == 0:
			steps = int(self.nr_chans / 2 + 1) 
		else:
			steps = int(np.ceil(self.nr_chans / 2.0))	

		slopes = np.empty((nr_freqs, nr_samps)) * np.nan
		amps = np.copy(slopes)
		baselines = np.copy(slopes)
		concentrations = np.copy(slopes)
		means = np.copy(slopes)
		preds = np.copy(slopes)
		rmses = np.copy(slopes)

		for f in range(nr_freqs):
			for s in range(nr_samps):
				d = X[f,s] 
				# collapse symmetric slope positions, 
				# eg (0, (45,315), (90,270), (135,225), 180)
				if self.nr_chans % 2 == 0:
					d = np.array([d[0]] + [np.mean((d[i],d[-i]))
											 for i in range(1,steps)])
				else:
					d = np.array([np.mean((d[i],d[i+(-1-2*i)])) 
												for i in range(steps)])
				slopes[f,s] = np.polyfit(range(1, len(d) + 1), d, 1)[0]
				#amps[f,s], means[f,s], sigmas[f,s] = self.fitGaussian(X[f,s])
				(amps[f,s], 
				baselines[f,s], 
				concentrations[f,s], 
				means[f,s], 
				rmses[f,s]) = self.fitCosToCTF(X[f,s])

		return slopes, amps, baselines, concentrations, means, rmses

	def extract_slopes(self,X:np.array)->float:
		"""
		Slopes are calculated by collapsing across symmetric data points 
		in the tuning curve. The collapsed data create a linear 
		increasing vector array from the tails to the peak of the tuning 
		curve. A first order polynomial is then fitted to this array to 
		estimate the slope of the tuning curve.

		Args:
			X (np.array): estimated ctf

		Returns:
			slope (float): estimated slope of ctf 
		"""

		if self.nr_chans % 2 == 0:
			steps = int(self.nr_chans / 2 + 1) 
		else:
			steps = int(np.ceil(self.nr_chans / 2.0))
		
		# collapse acrosssymmetric slope positions, 
		# eg (0, (45,315), (90,270), (135,225), 180)
		if self.nr_chans % 2 == 0:
			x = np.array([X[0]] + [np.mean((X[i],X[-i]))
											for i in range(1,steps)])
		else:
			x = np.array([np.mean((X[i],X[i+(-1-2*i)])) 
											for i in range(steps)])
		
		slope = np.polyfit(range(1, len(x) + 1), x, 1)[0]	

		return slope	
	
	def extract_ctf_params(self,ctfs:np.array,params:dict,signal:str,
						  perm_idx:int,ch_idx:int=None)->dict:
		"""
		Helper function of get_ctf_tuning_params and summarize_ctfs
		that extracts the ctf parameters

		Args:
			ctfs (np.array): ctf function per frequency
			params (dict): output parameters as specified by 
			get_ctf_tuning_params. Dict with initialized arrays to save
			tuning parameters
			signal (str): which signal is characterized. Evoked power(E)  
			total (T) power, or raw (raw) eeg 
			perm_idx (int): current permuation index 
			ch_idx (int, optional): If quantification is done per 
			channel, specifies the index of the current channel. 
			Defaults to None.

		Returns:
			params (dict): output parameters
		"""

		if ctfs.ndim == 3:
			nr_freqs, nr_samples_tr, nr_chan = ctfs.shape
			nr_samples_te = 1
			GAT = False
			# insert new dimension so that indexing does not crash
			ctfs = ctfs[...,np.newaxis,:]
		else:
			nr_freqs, nr_samples_tr, nr_samples_te, nr_chan = ctfs.shape 
			GAT = True

		# hack to deal with line breaks
		p = params
		s = signal 

		for f in range(nr_freqs):
			for tr_s in range(nr_samples_tr):
				for te_s in range(nr_samples_te):
	
					slopes = self.extract_slopes(ctfs[f,tr_s,te_s])
					if ch_idx is None:
						p[f'{s}_slopes'][perm_idx,f,tr_s,te_s] = slopes
					else:
						p[f'{s}_slopes'][perm_idx,f,tr_s,te_s,ch_idx] = slopes
					if any([key for key in params.keys() if 'amps' in key]):
						(amps, base,
						conc,mu,_) = self.fit_cos_to_ctf(ctfs[f,tr_s,te_s])
						if ch_idx is None:
							p[f'{s}_amps'][perm_idx,f,tr_s,te_s] = amps
							p[f'{s}_base'][perm_idx,f,tr_s,te_s] = base
							p[f'{s}_conc'][perm_idx,f,tr_s,te_s] = conc
							p[f'{s}_means'][perm_idx,f,tr_s,te_s] = mu
						else:
							p[f'{s}_amps'][perm_idx,f,tr_s,te_s,ch_idx] = amps
							p[f'{s}_base'][perm_idx,f,tr_s,te_s,ch_idx] = base
							p[f'{s}_conc'][perm_idx,f,tr_s,te_s,ch_idx] = conc
							p[f'{s}_means'][perm_idx,f,tr_s,te_s,ch_idx] = mu	

		params = p	

		return params

	def summarize_ctfs(self,ctfs:dict,params:dict,nr_samples:Union[int,tuple],
					   nr_freqs:int,nr_perm:int=1,avg_ch:bool=True,
					   test_bins:np.array=None)->dict:
		"""
		
		Args:
			ctfs (dict): condition specific ctf (contains C2 and W)
			params (dict): output parameters as specified by 
			get_ctf_tuning_params. Dict with initialized arrays to save
			tuning parameters
			nr_samples (Union[int,tuple]): number of samples used in ctf
			analysis. Is used to determine whether analysis contains
			independent train and test data 
			(i.e., generalization across time)
			nr_freqs (int): nr of independent frequencies in ctf 
			analysis
			nr_perm (int, optional): Number of permuations. 
			Defaults to 1. (i.e., no permutation)
			avg_ch (bool, optional): Should the ctf be characterized
			individually for each spatial channel (False) or averaged
			across channels. Defaults to True.
			test_bins (np.array, optional): test bins actually used to 
			fit ctf

		Returns:
			params (dict): output parameters
		"""
		
		# check whether ctfs contains matrix of train and test samples
		GAT = True if isinstance(nr_samples,tuple) else False
	
		# infer datatypes to extract
		signals = np.unique([key.split('_')[0] for key in params.keys()])

		for signal in signals:		
			# get params for each permutation
			for p in range(nr_perm):
				signal_ctfs = ctfs[f'C2_{signal}'][p]
				if avg_ch:
					# check whether ctfs contains unfitted bins
					if test_bins.size < self.nr_bins:
						signal_ctfs = signal_ctfs[:,:,test_bins]
					signal_ctfs = signal_ctfs.mean(axis = -2)
					params = self.extract_ctf_params(signal_ctfs, params,
					 								signal, p)
				else:
					nr_chans = signal_ctfs.shape[-2]
					
					for ch in range(nr_chans):
						if GAT:
							ch_ctfs = signal_ctfs[:,:,:,ch]
						else:
							ch_ctfs = signal_ctfs[:,:,ch]
						if np.all(signal_ctfs[:,:,ch] == 0):
				 			continue
						params = self.extract_ctf_params(ch_ctfs,params, 
														signal, p, ch)
		
		return params

	def get_ctf_tuning_params(self,ctf:dict,params:str='slopes',
			   				data_type:str='power',GAT:bool=False,
							avg_ch:bool=True,test_bins:np.array=None)->dict:
		"""
		Quantifies the ctfs as calulated by spatial_ctf and 
		localizer_spatial_ctf by either only the slopes, or more fine 
		grained by additionaly returning parameters that describe the 
		ctf function (i.e., amplitude, width, basline and mean).

		Args:
			ctf (dict): ctfs per condition as returned by ctf functions
			(spatial_ctf and localizer_spatial_ctf)
			params (str, optional): Specifies the detail of 
			quantification. 'cos_fit' gives additional parameters.
			Defaults to 'slopes'.
			data_type (str, optional): Are ctfs calculated on 
			time-frequency data or on raw eeg. Defaults to 'power'.
			GAT (bool, optional): Specifies whether ctf is trained and 
			tested on independent timepoints (i.e., generalization 
			across time matrix).
			avg_ch (bool, optional): Should the ctf be characterized
			individually for each spatial channel (False) or averaged
			across channels. Defaults to True.

		Returns:
			_type_: _description_
		"""

		# determine the output parameters
		if test_bins is None:
			test_bins = np.arange(self.nr_bins)
		output_params = []
		signals = ['T','E'] if data_type == 'power' else ['raw']
		output_params += [f'{signal}_slopes' for signal in signals]
		if params == 'cos_fit':
			tuning_params = ['amps', 'base','conc','means']
			output_params += [f'{signal}_{param}' for param in tuning_params
		     										for signal in signals]	
		# initiate output dict
		ctf_param = {}
		data = ctf[list(ctf.keys())[0]]
		if data_type == 'power':
			data = data['C2_E']
		else: 
			data = data['C2_raw']

		if GAT:
			(nr_perm, nr_freq, 
			nr_samples_tr, nr_samples_te) = list(data.shape)[:4]
			output = np.zeros((nr_perm,nr_freq, nr_samples_tr,nr_samples_te))
			nr_samples = (nr_samples_tr,nr_samples_te)
		else:
			nr_perm, nr_freq, nr_samples = list(data.shape)[:3]
			output = np.zeros((nr_perm,nr_freq,nr_samples,1))		

		if not avg_ch:
			output = output[..., np.newaxis] * np.zeros(self.nr_chans)

		# loop over all conditions
		for cnd in ctf.keys():
			ctf_param.update({cnd:{}})
			# initiate output data
			for param in output_params:
				ctf_param[cnd].update({param:output.copy()})

			# get paramters
			ctf_param[cnd] = self.summarize_ctfs(ctf[cnd],ctf_param[cnd],
				     							nr_samples,nr_freq,nr_perm,
												avg_ch,test_bins)
		
			# get rid of unecessary dimensions
			ctf_param[cnd] = {k: np.squeeze(v) 
		     							for k, v in ctf_param[cnd].items()}		

		return ctf_param			

	def fit_cos_to_ctf(self,X:np.array,kstep:float=0.1):
		"""_summary_

		Args:
			X (np.array): _description_
			kstep (float, optional): _description_. Defaults to 0.1.

		Returns:
			_type_: _description_
		"""


		num_bins = X.size

		# possible concentration parameters to consider
		# step size of 0.1 is usually good, but smaller makes for better fits
		k = np.arange(3, 40 + kstep, kstep)

		# allocate storage arrays
		sse, baseline, amps = np.zeros((3, k.size))

		# hack to find best central point
		m_idx = np.argmax(X[int(np.round(num_bins/2.0) - np.round(num_bins * .1) - 1): 
		  				    int(np.round(num_bins/2.0) + np.round(num_bins * .1))])
		x = np.linspace(0, np.pi - np.pi/num_bins, num_bins)
		#u = x[int(m_idx + np.round(num_bins/2.0) - np.round(num_bins * .1) - 1)]
		u = x[int(num_bins/2)]

		# loop over all concentration parameters
		# estimate best amp and baseline offset 
		# and find combination that minimizes sse
		a, b = 1,0
		for i in range(k.size):
			# create the vm function
			pred = a * np.exp(k[i] * (np.cos(u - x) - 1)) + b
			# build a design matrix and use GLM to estimate amp and baseline
			dm = np.zeros((num_bins, 2))
			dm[:,0] = pred
			dm[:,1] = np.ones(num_bins)
			betas, _, _, _ = np.linalg.lstsq(dm, X, rcond=None)
			amps[i], baseline[i] = betas
			est = pred * betas[0] + betas[1]
			sse[i] = sum((est - X)**2)

		idx_min = np.argmin(sse)
		b = baseline[idx_min]
		a = amps[idx_min]
		concentration = k[idx_min]
		pred = a * np.exp(concentration * (np.cos(u - x) - 1)) + b
		rmse = np.sqrt(sum((pred - X)**2)/pred.size)

		return a, b, concentration, u, rmse

	def gaussian(self, x, amp, mu, sig):
		'''Standard Gaussian function
		
		Arguments:
			x {array} -- gaussian x coordinates 
			amp {int|float} -- gaussian amplitude
			mu {int|foat} -- gaussian mean
			sig {int|float} -- gaussian width
		
		Returns:
			y{array} -- gaussian y coordinates
		'''

		y = amp * np.exp(-(x - mu)**2/(2*sig**2))

		return y
		
	def fit_gaussian(self, y):
		'''fits a Gaussian to input y
		
		Arguments:
			y {array} -- to be fitted data
		
		Returns:
			popt {array} -- array that contains amplitude, mean and sigma of the fitted Gaussian
		'''

		x = np.arange(y.size)
		mu = sum(x * y) / sum(y)
		sig = np.sqrt(sum(y * mu)**2) / sum(y)

		popt, pcov = curve_fit(self.gaussian, x, y, p0=[max(y), mu, sig] )

		return popt 
