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
				to_decode:str,nr_bins:int,nr_chans:int,nr_iter:int=10, 
				nr_blocks:int=3,elec_oi:Union[str,list]='all',
				sin_power = 7,delta:bool=False,method:str='Foster',
				avg_ch:bool=True,ctf_param:bool=True,power:str='band',
				min_freq:int=4,max_freq:int=40,num_frex:int=25,
				freq_scaling:str='log',slide_window:int=0,
				laplacian:bool=False,pca_cmp:int=10):
		''' 
		
		Arguments
		- - - - - 
		beh (DataFrame): behavioral infor across epochs
		eeg (mne object): eeg object
		channel_folder (str): folder specifying which electrodes are used for CTF analysis (e.g. posterior channels)
		decoding (str)): String specifying what to decode. 
		but can be changed to different locations (e.g. decoding of the location of an intervening stimulus).
		nr_iter (int):  number iterations to apply forward model 	
		nr_blocks (str): number of blocks to apply forward model
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

		# specify model parameters
		self.method = method
		self.nr_bins = nr_bins													# nr of spatial locations 
		self.nr_chans = nr_chans 												# underlying channel functions coding for spatial location
		self.nr_iter = nr_iter													# nr iterations to apply forward model 		
		self.nr_blocks = nr_blocks												# nr blocks to split up training and test data with leave one out test procedure
		self.sfreq = 512														# shift of channel position
		self.power = power
		self.min_freq = min_freq
		self.max_freq = max_freq
		self.num_frex = num_frex
		self.freq_scaling = freq_scaling
		self.slide_wind = slide_window
		self.laplacian = laplacian
		self.pca=pca_cmp
		# hypothesized set tuning functions underlying power measured across electrodes
		self.basisset = self.calculate_basis_set(self.nr_bins, self.nr_chans, 
											  sin_power,delta)

	def calculate_basis_set(self,nr_bins:int,nr_chans:int, 
			 				sin_power:int,delta:bool)->np.array:
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
			x = np.linspace(0, 2*pi - 2*pi/nr_bins, nr_bins) + np.deg2rad(180/nr_bins)

		c_centers = np.linspace(0, 2*pi - 2*pi/nr_chans, nr_chans)			
		c_centers = np.rad2deg(c_centers)
		
		if delta:
			pred = np.zeros(nr_bins)
			pred[-1] = 1
		else:
			pred = np.sin(0.5*x)**sin_power									# hypothetical channel responses
		
		if nr_bins % 2 == 0:											# shift the initial basis function	
			pred = np.roll(pred,nr_chans - (np.argmax(pred) + 1))
		else:
			pred = np.roll(pred,np.argmax(pred))	

		basisset = np.zeros((nr_chans,nr_bins))
		for c in range(nr_chans):
			basisset[c,:] = np.roll(pred,c+1)
	
		return basisset	

	def select_ctf_data(self,epochs:mne.Epochs,beh:pd.DataFrame,
						elec_oi: Union[list, str]=  'all',
						excl_factor: dict = None) -> Tuple[mne.Epochs,
															pd.DataFrame]:
		"""
		Selects the data of interest by selectiong epochs and elecrodes
		of interest

		Args:
			elec_oi ([list, str], optional): Electrodes used for ctf analysis. 
			Defaults to 'all'.
			excl_factor (dict, optional): exclusion dictionary that allows for
			dropping of certai conditions. Defaults to None.

		Returns:
			epochs (mne.Epochs): EEG data used for ctf analysis
			beh (pd.DataFrame): behavioral parameters
		"""

		# if not already done reset index (to properly align beh and epochs)
		beh.reset_index(inplace = True, drop = True)

		# if specified remove trials matching specified criteria
		if excl_factor is not None:
			beh, epochs = trial_exclusion(beh, epochs, excl_factor)

		if self.laplacian:
			epochs = mne.preprocessing.compute_current_source_density(epochs)

		# limit analysis to electrodes of interest
		picks = select_electrodes(epochs.ch_names, elec_oi) 
		epochs.pick(picks)

		return epochs, beh

	def check_cnds_input(self, cnds:dict)-> Tuple[list, str]:
		"""
		checks whether ctf should be run within or across conditions

		Args:
			cnds (dict): dictionary with conditions of interest as values

		Returns:
			train_cnds (list): Conditions used train and test the model.
			In case test_cnd is not None, these conditions will be used solely 
			to train the model
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
		(in pos_labels) and conditions (in cnds). Function also ensures that
		there are no more unique position bins then the number of bins used to 
		calculate ctfs

		Args:
			epochs (mne.Epochs): Epochs object 
			beh (pd.DataFrame): dataframe with behavioral parameters
			pos_labels (dict): key, item pair where key points to the column
			in beh that contains labels for ctf analysis, and the item contains
			all values that will be considered in the ctf analysis
			cnds (dict): key, item pair where key points to the column
			in beh that contains condition labels, and the item contains
			all unique conditions that are considered

		Returns:
			pos_bins (np.array): array with position labels
			cnds (np.array): array with condition labels
			epochs (mne.Epochs): eeg data of interest
			max_tr (int): maximum number of trials to include per position 
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
		Function ensures that there are no ctf labels that exceed the number
		of bins as specified at class initialization

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
		the block/fold assignment of the forward model such that the number 
		of trials used in the IEM is equated across conditions

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
			nr_per_bin = int(np.floor(min_count/self.nr_blocks))						
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
		butterworth filter. A Hilbert Transform is then applied to the filtered 
		data to extract the complex analytic signal. To extract power this 
		signal is computed by squaring the complex magnitude of the complex 
		signal. 

		Args:
			epochs (mne.Epochs): epochs object used for TF analysis
			band (tuple): frequency band used for TF decomposition
			tois (slice): time windows of interest
			downsample (int): factor to downsample the data (after filtering)

		Returns:
			E (np.array): analytic representation obtain with hilbert transform
			(used to calculated evoked activity: ongoing activity irrespective 
			of phase relation to stimulus onset)
			T (np.array): squared magnitude of E. Used to calculated total 
			power ctf	
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
			T = epochs._data
			E = epochs._data

		# trim filtered data (after filtering to avoid artifacts)
		E = E[:,:,tois]
		T = T[:,:,tois]

		# downsample 
		E = E[:,:,::downsample]
		T = T[:,:,::downsample]

		return E, T

	def forward_model(self, train_X: np.array, test_X: np.array, 
					C1: np.array) -> Tuple[np.array, np.array]:
		"""
		Applies an inverted encoding model (IEM) to each sample. This routine 
		proceeds in two stages (train and test).
		1. In the training stage, training data (B1) is used to estimate 
		weights (W) that approximate the relative contribution of each spatial 
		channel to the observed response measured at each electrode, with:

		B1 = W*C1

		,where B1 is power at each electrode, C1 is the predicted response 
		of each spatial channel, and W is weight matrix that characterizes a 
		linear mapping from channel space to electrode space. The equation is 
		solved via least-squares estimation.
		
		2. In the test phase, the model is inverted to transform the observed 
		test data B2 into estimated channel responses. The estimated channel 
		response are then shifted to a common center by aligning the estimate 
		channel response to the channel tuned for the stimulus bin. 

		Args:
			train_X (np.array): train data (epochs X electrodes X timepoints)
			test_X (np.array): test data (epochs X electrodes X timepoints)
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

		return C2s, W 

	def forward_model_loop(self, E_train: np.array,  E_test: np.array,
							T_train: np.array,  T_test: np.array, 
							C1: np.array) -> Tuple[np.array, np.array,
													np.array,np.array]:
		"""
		Applies inverted encoding model (see forward_model) for each individual
		time point

		Args:
			E_train (np.array): train data with evoked power
			E_test (np.array): test data with evoked power
			T_train (np.array): train data with total power
			T_test (np.array): test data with total power
			C1 (np.array): basisset

		Returns:
			C2_E (np.array): evoked power based shifted predicted channels 
			response across samples
			W_E (np.array): evoked power weight matrices across samples
			C2_T (np.array):total power based shifted predicted channels 
			response across samples
			W_T (np.array): total power weight matrices across samples
		"""

		# initialize arrays
		_, nr_elec, nr_samples = E_train.shape
		nr_bins = self.nr_bins
		C2_E = np.zeros((nr_samples-self.slide_wind, nr_bins, self.nr_chans))
		W_E =  np.zeros((nr_samples-self.slide_wind, nr_bins,nr_elec))
		C2_T, W_T = C2_E.copy(), W_E.copy()

		# TODO: parallelize loop
		# TODO: potentially remove
		for t in range(nr_samples-self.slide_wind):
			# evoked power model fit
			c2_e, w_e = self.forward_model(E_train[...,t:t+1+self.slide_wind],
											E_test[...,t:t+1+self.slide_wind], 
											C1)			
			
			C2_E[t], W_E[t,:,:w_e.shape[1]] = c2_e, w_e

			# total power model fit
			c2_t, w_t = self.forward_model(T_train[...,t:t+1+self.slide_wind], 
											T_test[...,t:t+1+self.slide_wind], 
											C1)
			C2_T[t], W_T[t,:,:w_t.shape[1]] = c2_t, w_t

		return C2_E, W_E, C2_T, W_T

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

	def train_test_cross(self, pos_bins: np.array,train_idx:np.array,
						test_idx:np.array)->Tuple[np.array, np.array]:
		"""
		selects trial indices for the train and the test set

		Args:
			pos_bins (np.array): array with position bins
			train_idx (np.array): array with indices for training condition
			test_idx (np.array): array with indices for test condition

		Returns:
			train_idx (np.array): indices used to train the model (with 
			bin labels balanced)
			test_idx (np.array): indices used to test the model (with 
			bin labels balanced)
		"""

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
		for i in range(self.nr_iter):
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

		test_idx = np.tile(test_idx, (self.nr_iter, 1, 1, 1))

		return train_idx, test_idx

	def train_test_split(self, pos_bins: np.array,cnd_idx: np.array, 
						max_tr: int)->Tuple[np.array, np.array]:
	
		'''
		creates a random block assignment (train and test blocks)

		Arguments
		- - - - - 
		pos_bin (array): array with position bins
		idx_tr (array): array of trial numbers corresponding to pos_bin labels
		nr_per_bin(int): maximum number of trials to be used per location bin
		nr_iter (int): number of iterations used for IEM
		Returns
		- - - -
		blocks(array): randomly assigned block indices
		'''
		
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
				idx = idx[:max_tr * self.nr_blocks] 
				x = np.tile(np.arange(self.nr_blocks),(max_tr,1))
				shuf_blocks.flat[idx] = x	

			# unshuffle block assignment and save to CTF
			blocks[idx_shuf] = shuf_blocks	
			bl_assign[i] = blocks
		tr_per_block = int(sum(blocks == 0)/self.nr_bins)

		# after block assignment split into train and test set
		train_idx = np.zeros((self.nr_iter * self.nr_blocks, self.nr_bins,
							 self.nr_blocks - 1, tr_per_block), dtype = int)									
		test_idx = np.zeros((self.nr_iter * self.nr_blocks, self.nr_bins, 
						  tr_per_block), dtype = int)

		idx = 0	
		for i in range(self.nr_iter):
			for bl in range(self.nr_blocks):
				for bin in range(self.nr_bins):	
					test_mask = (bl_assign[i] == bl) * (cnd_bins == bin)
					test_idx[idx,bin] = trial_idx[test_mask]
					train_mask = ((~np.isnan(bl_assign[i])) * 
								 (bl_assign[i] != bl) * (cnd_bins == bin))
					# split all train data into seperate train blocks
					train = np.array_split(trial_idx[train_mask], 
														self.nr_blocks - 1)
					for j in range(self.nr_blocks - 1):
						train_idx[idx, bin, j] = train[j]
				idx += 1

		return train_idx, test_idx

	def spatial_ctf(self, pos_labels:dict ='all',cnds:dict=None, 
					excl_factor:dict=None, window_oi:tuple=(None,None),
					freqs:dict='main_param',downsample:int = 1, 
					nr_perm:int= 0,collapse:bool=False,name:int='main'):
		"""
		calculate spatially based channel tuning functions across conditions.
		Training and testing cab be donne either witin or across conditions 
		(see cnds argument)

		Args:
			pos_labels (dict, optional): key, item pair where key points to the 
			column in beh that contains position labels for ctf analysis, and 
			the item is a list that contains all values that will be considered 
			in the ctf analysis. Defaults to all (i.e., all values as specified
			in the class argument to_decode will be considered in the ctf 
			analysis). 
			cnds (dict, optional): key, item pair where key points to the 
			column in beh that contains conditions of  interest, and 
			the item is a list that contains conditions of interets. To run a 
			cross decoding analysis, the first value in the cnds dict needs to 
			be a list specifying the conditions to train the model. Currently 
			only a single condition can be tested per run. Defaults to None 
			(i.e., include all trials)
			excl_factor (dict, optional): This gives the option to exclude 
			specific conditions from analysis. For example, to only include 
			trials where the cue was pointed to the left and not to the right 
			specify the following: factor = dict('cue_direc': ['right']). 
			Mutiple column headers and multiple variables per header can be 
			specified . Defaults to None.
			window_oi (tuple, optional): time window of interest (start to end) 
			for ctf analysis. Time window will  be cropped after filtering. 
			Defaults to (None,None).
			freqs (dict, optional): key, item pair where key is the name of 
			the frequency band of interest, and the item is a list with the 
			lower and upper frequency. Defaults to main_param, which will use 
			the time frequency settings as specified in the CTF class. 
			downsample (int, optional): factor to downsample data (is applied
			after filtering). Defaults to 1.
			nr_perm (int, optional): _description_. Defaults to 0.
			collapse (bool, optional): _description_. Defaults to False.
			name (int, optional): _description_. Defaults to 'main'.
		"""
 
		# read in data
		epochs, beh = self.select_ctf_data(self.epochs, self.beh,
											self.elec_oi, excl_factor)

		# set params
		nr_itr = self.nr_iter * self.nr_blocks
		ctf_name = f'{self.sj}_{name}'
		nr_perm += 1
		nr_elec = len(epochs.ch_names)
		tois = get_time_slice(epochs.times, window_oi[0], window_oi[1])
		nr_samples = epochs.times[tois][::downsample].size
		ctf, info = {}, {}
		freqs, nr_freqs = self.set_frequencies(freqs)
		if self.method == 'k-fold':
			# TODO: fix
			print('Method not yet  implemented')
			print('nr_blocks is irrelevant and will be reset to 1')
			#self.nr_blocks = 1						
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
  
		# Frequency loop (ensures that data is only filtered once)
		for fr in range(nr_freqs):
			print('Frequency {} out of {}'.format(str(fr + 1), str(nr_freqs)))

			# Time-Frequency Analysis
			E, T = self.tf_decomposition(epochs.copy(),freqs[fr],tois,downsample)

			# Loop over conditions
			for c, cnd in enumerate(train_cnds):
				print(f'Running ctf for {cnd} condition')

				# preallocate arrays
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
						test_idx) = self.train_test_cross(pos_bins,cnd_idx,
														 test_idx)
					else:
						(train_idx, 
						test_idx) = self.train_test_split(pos_bins,
														cnd_idx,max_tr)
						test_bins = np.unique(pos_bins)

					info[cnd]['train_idx'] = train_idx
					info[cnd]['test_idx'] = test_idx
					if self.method == 'Foster':
						C1 = np.empty((self.nr_bins* (self.nr_blocks - 1), 
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
						nr_itr_tr = self.nr_bins * (self.nr_blocks - 1)
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
							for j in range(self.nr_blocks - 1):
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
															bin_te_T, C1)
		# take the average across model iterations
		for cnd in train_cnds:
			for key in ['C2_E','C2_T','W_E','W_T']:
				ctf[cnd][key] = ctf[cnd][key].mean(axis = 2)

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
			ctf_param = self.ctfs_tuning_params(ctf)
			with open(self.folder_tracker(['ctf',self.to_decode], 
					fname=f'ctf_param_{ctf_name}.pickle'),'wb') as handle:
				print('saving ctf params')
				pickle.dump(ctf_param, handle)

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

		ctf, info = {}, {}
		freqs, nr_freqs = self.set_frequencies(freqs)

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
			nr_samp_tr = E_tr.shape[-1]
			(E_te, 
			T_te) = self.tf_decomposition(epochs_te.copy(),freqs[fr],
									 tois_te,downsample)	
			nr_samp_te = E_te.shape[-1]

			# Loop over conditions
			for c, cnd in enumerate(test_cnds):
				print(f'Running localizer ctf for condition: {cnd} ')	

				# preallocate arrays
				if GAT:
					C2_E = np.zeros((nr_perm,nr_freqs, nr_samp_tr, nr_samp_te,
									self.nr_bins, self.nr_chans))
					W_E = np.zeros((nr_perm,nr_freqs, nr_samp_tr, nr_samp_te,
									self.nr_chans, nr_elec))	
				else:
					C2_E = np.zeros((nr_perm,nr_freqs,nr_samp_tr, 
									self.nr_bins, self.nr_chans))
					W_E = np.zeros((nr_perm,nr_freqs,nr_samp_tr, 
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
					(train_idx,_) = self.train_test_cross(pos_bins_tr,idx,None)
					idx = cnd == cnds
					(_,test_idx) = self.train_test_cross(pos_bins_te,idx,idx)
					test_bins = np.unique(pos_bins_te[test_idx])

					info[cnd]['train_idx'] = train_idx
					info[cnd]['test_idx'] = test_idx
					if self.method == 'Foster':
						C1 = np.empty((self.nr_bins, self.nr_chans)) * np.nan
					else:
						C1 = self.basisset	

				# TODO: insert permutation loop
				p = 0
				train_idx = info[cnd]['train_idx'][0]

				# initialize evoked and total power arrays
				bin_te_E = np.zeros((self.nr_bins, nr_elec, nr_samp_te)) 
				bin_te_T = bin_te_E.copy()
				if self.method == 'k-fold':
					pass
					#TODO: implement 
				elif self.method == 'Foster':
					bin_tr_E = np.zeros((self.nr_bins, nr_elec, nr_samp_tr)) 
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
						evoked = abs(np.mean(E_tr[train_idx[bin][0]], 
											axis = 0))**2
						bin_tr_E[bin_cnt] = evoked
						total = np.mean(T_tr[train_idx[bin][0]], axis = 0)
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
														bin_te_T,C1)
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
			ctf_param = self.ctfs_tuning_params(ctf)
			with open(self.folder_tracker(['ctf',self.to_decode], 
					fname=f'ctf_param_{ctf_name}.pickle'),'wb') as handle:
				print('saving ctf params')
				pickle.dump(ctf_param, handle)

	def summarize_ctfs(self, X, nr_freqs, nr_samps):
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

	def ctfs_tuning_params(self, ctf: dict):


		params = ['T_slopes','E_slopes',
				'T_amps','E_amps',
				'T_base','E_base',
				'T_conc','E_conc',
				'T_means','E_means']
	
		# get ctf params
		ctf_param = {}
		(nr_perm, 
		nr_freq, 
		nr_samples, _, _,) =  ctf[list(ctf.keys())[0]]['C2_E'].shape
		output = np.zeros((nr_perm,nr_freq, nr_samples))
		if not self.avg_ch:
			output = np.zeros((nr_perm,nr_freq, nr_samples, self.nr_chans))

		# loop over  all conditions
		for cnd in ctf.keys():
			ctf_param.update({cnd:{}})
			# initiate output data
			for param in params:
				ctf_param[cnd].update({param:output.copy()})

			# get tuning params (seperate for evoked and total power)
			for p in range(nr_perm):
				if self.avg_ch:
					ctf_evoked = ctf[cnd]['C2_E'][p].mean(axis = -2)
					ctf_total = ctf[cnd]['C2_T'][p].mean(axis = -2)

					(ctf_param[cnd]['E_slopes'][p],
					ctf_param[cnd]['E_amps'][p],
					ctf_param[cnd]['E_base'][p],
					ctf_param[cnd]['E_conc'][p],
					ctf_param[cnd]['E_means'][p],_
					) = self.summarize_ctfs(ctf_evoked,nr_freq,nr_samples)

					(ctf_param[cnd]['T_slopes'][p],
					ctf_param[cnd]['T_amps'][p],
					ctf_param[cnd]['T_base'][p],
					ctf_param[cnd]['T_conc'][p],
					ctf_param[cnd]['T_means'][p],_
					) = self.summarize_ctfs(ctf_total,nr_freq,nr_samples)
				else:
					ctf_evoked = ctf[cnd]['C2_E'][p]
					ctf_total = ctf[cnd]['C2_T'][p]
					for ch in range(self.nr_chans):
						if np.all(ctf_evoked[:,:,ch] == 0):
							continue
						(ctf_param[cnd]['E_slopes'][p,:,:,ch],
						ctf_param[cnd]['E_amps'][p,:,:,ch],
						ctf_param[cnd]['E_base'][p,:,:,ch],
						ctf_param[cnd]['E_conc'][p,:,:,ch],
						ctf_param[cnd]['E_means'][p,:,:,ch],_
						)= self.summarize_ctfs(ctf_evoked[:,:,ch],
												nr_freq,nr_samples)

						(ctf_param[cnd]['T_slopes'][p,:,:,ch],
						ctf_param[cnd]['T_amps'][p,:,:,ch],
						ctf_param[cnd]['T_base'][p,:,:,ch],
						ctf_param[cnd]['T_conc'][p,:,:,ch],
						ctf_param[cnd]['T_means'][p,:,:,ch],_
						) = self.summarize_ctfs(ctf_total[:,:,ch],
											nr_freq,nr_samples)

		return ctf_param

	def forwardModel(self, train_X, test_X, C1):
		'''
		forwardModel applies an inverted encoding model (IEM) to each time point in the analyses. This routine proceeds in two stages (train and test).
		1. In the training stage, training data (B1) is used to estimate weights (W) that approximate the relative contribution of each spatial channel 
		to the observed response measured at each electrode, with:

		B1 = W*C1

		,where B1 is power at each electrode, C1 is the predicted response of each spatial channel, and W is weight matrix that characterizes a 
		linear mapping from channel space to electrode space. Equation is solved via least-squares estimation.
		2. In the test phase, the model is inverted to transform the observed test data B2 into estimated channel responses. The estimated channel response 
		are then shifted to a common center by aligning the estimate dchannel response to the channel tuned for the stimulus bin. 

		Arguments
		- - - - - 
		train_X (array): training data (trials X electrodes X timepoints)
		test_X (array): test data (trials X electrodes X timepoints)
		C1 (array): basisset

		Returns
		- - - -
		C2(array): unshifted predicted channel response
		C2s(array): shifted predicted channels response

		'''

		# apply forward model
		B1 = train_X
		B2 = test_X
		W, resid_w, rank_w, s_w = np.linalg.lstsq(C1,B1, rcond = -1)		# estimate weight matrix W (nr_chans x nr_electrodes)
		C2, resid_c, rank_c, s_c = np.linalg.lstsq(W.T,B2.T, rcond = -1)	# estimate channel response C2 (nr_chans x nr test blocks) 

		# TRANSPOSE C2 so that we take the average across channels rather than across position bins
		C2 = C2.T
		C2s = np.zeros(C2.shape)

		bins = np.arange(self.nr_bins)
		# shift eegs to common center
		nr_2_shift = int(np.ceil(C2.shape[1]/2.0))
		for i in range(C2.shape[0]):
			idx_shift = abs(bins - bins[i]).argmin()
			shift = idx_shift - nr_2_shift
			if self.nr_bins % 2 == 0:							# CHECK THIS: WORKS FOR 5 AND 8 BINS
				C2s[i,:] = np.roll(C2[i,:], - shift)	
			else:
				C2s[i,:] = np.roll(C2[i,:], - shift - 1)			

		return C2, C2s, W 											# return the unshifted and the shifted channel response



	def equateBins(self, bins, X):
		'''
		removes trials such that number of observations per bin is equalized
		'''	

		idx_to_keep = []
		min_tr = np.min(np.unique(bins, return_counts= True)[1])
		for b in np.unique(bins):
			idx = np.where(bins == b)[0]	
			np.random.shuffle(idx)
			idx_to_keep.append(idx[:min_tr])

		idx = np.sort(np.hstack(idx_to_keep))
		
		bins = bins[idx]
		X = X[idx]

		return bins, X	

	def crosstrainCTF(self, sj, window, train_cnds, test_cnds, freqs = dict(alpha = [8,12]), filt_art = 0.5, downsample = 4, tgm = False, nr_perm = 0, name = ''):
		'''
		Calculates spatial CTFs across subjects and conditions using the filter-Hilbert method. 
		Is a specfic case of CTF as training and testing is not done within conditions but across conditions
			
		Arguments
		- - - - - 
		sj (int): subject number
		window (list | tuple): specifying the time window of interest
		freqs (dict): Frequency band used for bandpass filter. If key set to all, analyses will 
					  be performed in increments off ? Hz
		train_cnds (list): conditions used for training. That is the same training set is used for all testing conditions
		test_cnds (list): test conditions
		filt_art (float): time in s added to correct for filter artifacts
		downsample (int): factor used to downsample the data (e.g. downsample = 4 downsamples 512 Hz to 128 Hz)
		tgm (bool): create a train test generalization matrix
		nr_perm (int): Number of permutations. If 0, analysis is only performed on non-permuted data
		name (str): name of cross training combination

		'''

		# set number of permutations
		nr_perm += 1

		CTF = {} 

		if 'all' in freqs.keys():
			freqs = np.vstack([[i, i + 4] for i in range(freqs['all'][0],freqs['all'][1] + 1,2)])
		else:									
			freqs = np.vstack([[freqs[band][0],freqs[band][1]] for band in freqs.keys()])
		nr_freqs = freqs.shape[0]

		# read in all data (train plus test data)
		pos_bins, cnds, eegs = self.selectData(train_cnds + test_cnds)
		samples = np.logical_and(self.eeg.times >= window[0], self.eeg.times <= window[1])
		nr_samples = self.eeg.times[samples][::downsample].size
	
		# loop over test conditions
		for test_cnd in test_cnds:

			# select train and test data
			train_mask = np.array([True if cnd in train_cnds and cnd != test_cnd else False for cnd in cnds])
			train_X = eegs[train_mask,:,:]
			train_bins = pos_bins[train_mask]
			train_bins, train_X = self.equateBins(train_bins, train_X)

			test_X = eegs[cnds == test_cnd,:,:]
			test_bins = pos_bins[cnds == test_cnd]
			test_bins, test_X = self.equateBins(test_bins, test_X)

			if tgm:
				nr_te_samples = nr_samples
			else:
				nr_te_samples = 1	

			# tf = np.empty((nr_perm, nr_freqs, nr_samples, nr_te_samples, self.nr_chans)) * np.nan
			# C2 = np.empty((nr_perm,nr_freqs, nr_samples, nr_te_samples, self.nr_bins, self.nr_chans)) * np.nan
			# W = np.empty((nr_perm, nr_freqs, nr_samples, nr_te_samples, self.nr_chans, eegs.shape[1])) * np.nan

			tf = np.zeros((nr_perm, nr_freqs, nr_samples, nr_te_samples, self.nr_chans))
			C2 = np.zeros((nr_perm,nr_freqs, nr_samples, nr_te_samples, self.nr_bins, self.nr_chans))
			W = np.zeros((nr_perm, nr_freqs, nr_samples, nr_te_samples, self.nr_chans, eegs.shape[1]))

			# Loop over frequency bands
			for fr in range(nr_freqs):
				print('Started cross training of frequency {} out of {}'.format(fr + 1, freqs.shape[0]))

				# TF analysis
				_, tr_total = self.powerAnalysis(train_X, samples, self.sfreq, [freqs[fr][0],freqs[fr][1]], downsample)
				_, te_total = self.powerAnalysis(test_X, samples, self.sfreq, [freqs[fr][0],freqs[fr][1]], downsample)

				# average data for each position
				bin_tr_X = np.empty((self.nr_bins, eegs.shape[1], tr_total.shape[2])) * np.nan
				bin_te_X = np.empty((self.nr_bins, eegs.shape[1],te_total.shape[2])) * np.nan

				for p in range(nr_perm):
					print("\r{0:.2f}% of permutations".format((float(p)/nr_perm)*100),)

					# first time train labels are not shuffled. # PERMUTATION INDICES ATE NOT YET SAVED
					if p > 0:
						np.random.shuffle(train_bins)

					for b in range(self.nr_bins):
						idx_tr = train_bins == b
						idx_te = test_bins == b
						bin_tr_X[b] = np.mean(tr_total[idx_tr], axis = 0)
						bin_te_X[b] = np.mean(te_total[idx_te], axis = 0)

					# Loop over all samples
					for tr_smpl in range(nr_samples):
						#print "\r{0:.0f}% of tr matrix".format((float(tr_smpl)/nr_samples)*100),
						for te_smpl in range(nr_te_samples):
							if not tgm:
								te_smpl = tr_smpl
								te_idx = 0
							else:
								te_idx = te_smpl	

							###### ANALYSIS ON TOTAL POWER ######
							c2, c2s, w = self.forwardModel(bin_tr_X[:,:,tr_smpl], bin_te_X[:,:,te_smpl], self.basisset)
							C2[p,fr,tr_smpl,te_idx] = c2 						# save the unshifted channel response
							tf[p,fr,tr_smpl, te_idx] = np.mean(c2s,0)			# save average of shifted channel response
							W[p,fr, tr_smpl, te_idx] = w 						# save estimated weights per channel and electrode
			
			# calculate slopes (for non-permuted data)
			slopes = np.zeros((nr_perm,nr_freqs, nr_samples, nr_te_samples))
			for p_ in range(nr_perm):
				for t in range(tf.shape[2]):
					slopes[p_,:,:,t] = self.calculateSlopes(tf[p_,:,:,t,:],nr_freqs,nr_samples) 
	
			#CTF[test_cnd] = {'C2':C2, 'ctf': tf, 'W': w, 'slopes': slopes}
			
			if nr_perm == 1:
				CTF[test_cnd] = {'slopes': slopes[0]}
			else:
				CTF[test_cnd] = {'slopes_p': slopes[1:], 'slopes': slopes[0]}	

		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_cross-training_{}.pickle'.format(sj, name)),'wb') as handle:
			print('saving cross training CTF dict')
			pickle.dump(CTF, handle)

	def spatial_ctf_old(self, pos_labels: dict = 'all', cnds: dict = None, 
					excl_factor: dict = None, window_oi: tuple = (None, None),
					freqs = dict(alpha = [8,12]), downsample = 1, 
					nr_perm = 0, collapse = False):
		'''
		Calculates spatial CTFs across subjects and conditions using the filter-Hilbert method. 

		Arguments
		- - - - - 
		sj (list): list of subject id's 
		window (list): time interval in ms used for ctf calculation
		conditions (list)| Default ('all'): List of conditions. Defaults to all conditions
		freqs (dict) | Default is the alpha band: Frequency band used for bandpass filter. If key set to all, analyses will be performed in increments 
		from 1Hz
		downsample (int): factor to downsample the bandpass filtered data
		method (str): method used for splitting data into training and testing sets
		nr_perm (int): number of times model is run with permuted data
		collapse (bool): create CTF collapsed across conditions

		Returns
		- - - -
		self(object): dict of CTF data
		'''	

		# read in data
		epochs, beh = self.select_ctf_data(self.elec_oi, excl_factor)

		# based on conditions get position bins
		(pos_bins, 
		cnds, 
		epochs, 
		max_tr) = self.select_ctf_labels(epochs, beh, pos_labels, cnds)

		# set params
		tois = get_time_slice(epochs.times, window_oi[0], window_oi[1])
		ctf = {}
		ctf_info = {}
		nr_perm += 1
		ctf_info['times'] = self.eeg.times[samples][::downsample]
		nr_samples = ctf_info['times'].size
		if self.method == 'k-fold':
			print('nr_blocks is irrelevant and will be reset to 1')
			self.nr_blocks = 1
								
		if 'all' in freqs.keys():
			frqs = np.vstack([[i, i + 4] for i in range(freqs['all'][0],freqs['all'][1] + 1,2)])
		else:									
			frqs = np.vstack([[freqs[band][0],freqs[band][1]] for band in freqs.keys()])
		nr_freqs = frqs.shape[0]

		if collapse:
			TODO: fix
			conditions += ['all_data']	

		# Frequency loop (ensures that data is only filtered once)
		for fr in range(nr_freqs):
			print('Frequency {} out of {}'.format(str(fr + 1), str(nr_freqs)))

			# Time-Frequency Analysis 
			E, T = self.tf_decomposition(epochs, tois, downsample)
			#E, T = self.powerAnalysis(eeg, samples, self.sfreq, [frqs[fr][0],frqs[fr][1]], downsample, self.power)

			# Loop over conditions
			for c, cnd in enumerate(np.unique(cnds)):

				# preallocate arrays

				# update CTF dict with preallocated arrays such that condition data can be saved later 
				if cnd not in ctf_info.keys():
					ctf_info.update({cnd:{}})
					ctf.update({cnd: {'tf_E': np.zeros((nr_perm,nr_freqs,self.nr_iter * self.nr_blocks,nr_samples, self.nr_chans)),
									 'C2_E': np.zeros((nr_perm,nr_freqs,self.nr_iter * self.nr_blocks,nr_samples, self.nr_bins, self.nr_chans)),
									 'W_E':	np.zeros((nr_perm,nr_freqs,self.nr_iter * self.nr_blocks,nr_samples, self.nr_chans, eegs.shape[1])),
									 'tf_T': np.zeros((nr_perm,nr_freqs,self.nr_iter * self.nr_blocks,nr_samples, self.nr_chans)),
									 'C2_T': np.zeros((nr_perm,nr_freqs,self.nr_iter * self.nr_blocks,nr_samples, self.nr_bins, self.nr_chans)),
									 'W_T':	np.zeros((nr_perm,nr_freqs,self.nr_iter * self.nr_blocks,nr_samples, self.nr_chans, eegs.shape[1])) }})	

				# select data and create training and test sets across folds for each condition
				# this is done only on the first frequency loop to ensure that datasets are identical across frequencies
				if fr == 0:
					if cnd == 'all':
						cnd_idx = np.arange(cnds.size)
						labels = pos_bins[:]
					else:
						# select train and test trials 
						cnd_idx = np.where(cnds == cnd)[0]
						labels = pos_bins[cnds == cnd]
					self.nr_folds = self.nr_iter
					if method == 'Foster':
						ctf_info[cnd]['tr'], ctf_info[cnd]['te']  = self.assignBlocks(labels, cnd_idx, nr_per_bin, self.nr_iter)
						C1 = np.empty((self.nr_bins* (self.nr_blocks - 1), self.nr_chans)) * np.nan 										
					elif method == 'k-fold':
						ctf_info[cnd]['tr'], ctf_info[cnd]['te'], _ = self.bdmTrialSelection(cnd_idx, labels, nr_per_bin, {})
						C1 = self.basisset

				# Loop through each iteration (is folds in the new set up)
				for itr in range(self.nr_iter * self.nr_blocks):
	
					# loop over permutations
					for p in range(nr_perm):
						print("\r{0:.2f}% of permutations".format((float(p)/nr_perm)*100),)
						# first time train labels are not shuffled. # PERMUTATION INDICES ATE NOT YET SAVED
						tr_idx = ctf_info[cnd]['tr'][itr]
						if p > 0:	
							to_shuffle = tr_idx.ravel()
							np.random.shuffle(to_shuffle)
							tr_idx = to_shuffle.reshape((self.nr_bins, self.nr_blocks - 1, -1))

						# average data for each position
						bin_te_E = np.empty((self.nr_bins, eegs.shape[1], T.shape[2])) * np.nan
						bin_te_T = np.empty((self.nr_bins, eegs.shape[1], T.shape[2])) * np.nan
						if method == 'k-fold':
							bin_tr_E = np.empty((self.nr_bins, eegs.shape[1], T.shape[2])) * np.nan
							bin_tr_T = np.empty((self.nr_bins, eegs.shape[1], T.shape[2])) * np.nan
						elif method == 'Foster':
							bin_tr_E = np.empty((self.nr_bins * (self.nr_blocks - 1), eegs.shape[1], T.shape[2])) * np.nan
							bin_tr_T = np.empty((self.nr_bins * (self.nr_blocks - 1), eegs.shape[1], T.shape[2])) * np.nan	
						
						bin_cntr = 0
						for b in range(self.nr_bins):
							bin_te_E[b] = abs(np.mean(E[ctf_info[cnd]['te'][itr][b]], axis = 0))**2
							bin_te_T[b] = np.mean(T[ctf_info[cnd]['te'][itr][b]], axis = 0)
							if method == 'fold':
								bin_tr_E[b] = abs(np.mean(E[tr_idx[b]], axis = 0))**2
								bin_tr_T[b] = np.mean(T[tr_idx[b]], axis = 0)	
							elif method == 'Foster':
								for j in range(self.nr_blocks - 1):
									bin_tr_E[bin_cntr] = abs(np.mean(E[tr_idx[b][j]], axis = 0))**2
									bin_tr_T[bin_cntr] = np.mean(T[tr_idx[b][j]], axis = 0)
									C1[bin_cntr] = self.basisset[b]
									bin_cntr += 1
															
						# Loop over all samples CHECK WHETHER THIS LOOP CAN BE PARALLIZED 
						for smpl in range(nr_samples):

							###### ANALYSIS ON EVOKED POWER ######
							c2, c2s, w = self.forwardModel(bin_tr_E[:,:,smpl], bin_te_E[:,:,smpl], C1)
							ctf[cnd]['C2_E'][p,fr,itr,smpl] = c2							# save the unshifted channel response
							ctf[cnd]['tf_E'][p,fr,itr, smpl] = np.mean(c2s,0)				# save average of shifted channel response
							ctf[cnd]['W_E'][p,fr, itr, smpl] = w 							# save estimated weights per channel and electrode

							###### ANALYSIS ON TOTAL POWER ######
							c2, c2s, w = self.forwardModel(bin_tr_T[:,:,smpl], bin_te_T[:,:,smpl], C1)
							ctf[cnd]['C2_T'][p,fr, itr, smpl] = c2 							# save the unshifted channel response
							ctf[cnd]['tf_T'][p,fr,itr, smpl] = np.mean(c2s,0)				# save average of shifted channel response
							ctf[cnd]['W_T'][p,fr, itr, smpl] = w 							# save estimated weights per channel and electrode 

		# calculate slopese
		ctf_inf = {}
		for cnd in ctf.keys():
			e_slopes, t_slopes = np.zeros((2, nr_perm,nr_freqs, nr_samples))
			e_amps, t_amps = np.zeros((2, nr_perm,nr_freqs, nr_samples))
			e_baselines, t_baselines = np.zeros((2, nr_perm,nr_freqs, nr_samples))
			e_concentrations, t_concentrations = np.zeros((2, nr_perm,nr_freqs, nr_samples))
			e_means, t_means = np.zeros((2, nr_perm,nr_freqs, nr_samples))
			#e_rmses, t_rmses = np.zeros((2, nr_perm,nr_freqs, nr_samples))
		
		
			ctf_inf.update({cnd:{}})
			for p_ in range(nr_perm):
				e_slopes[p_,:,:], e_amps[p_,:,:], e_baselines[p_,:,:], e_concentrations[p_,:,:], e_means[p_,:,:], _  = self.summarizeCTFs(np.mean(ctf[cnd]['tf_E'][p_], axis = 1),nr_freqs,nr_samples)
				t_slopes[p_,:,:], t_amps[p_,:,:], t_baselines[p_,:,:], t_concentrations[p_,:,:], t_means[p_,:,:], _   = self.summarizeCTFs(np.mean(ctf[cnd]['tf_T'][p_], axis = 1),nr_freqs,nr_samples)
			
			if nr_perm == 1:
				ctf_inf[cnd]['T_slopes'] = t_slopes[0]
				ctf_inf[cnd]['E_slopes'] = e_slopes[0]
				ctf_inf[cnd]['T_amps'] = t_amps[0]
				ctf_inf[cnd]['E_amps'] = e_amps[0]
				ctf_inf[cnd]['T_baselines'] = t_baselines[0]
				ctf_inf[cnd]['E_baselines'] = e_baselines[0]
				ctf_inf[cnd]['T_concentrations'] = t_concentrations[0]
				ctf_inf[cnd]['E_concentrations'] = e_concentrations[0]
				ctf_inf[cnd]['T_means'] = t_means[0]
				ctf_inf[cnd]['E_means'] = e_means[0]
			else:
				slopes[cnd] = {'T_slopes_p': t_slopes[1:], 'T_slopes': t_slopes[0], 
							   'E_slopes_p': e_slopes[1:], 'E_slopes': e_slopes[0],
							   'T_amps_p': t_amps[1:], 'T_amps': t_amps[0], 
							   'E_amps_p': e_amps[1:], 'E_amps': e_amps[0],
							   'T_baselines_p': t_means[1:], 'T_baselines': t_means[0], 
							   'E_baselines_p': e_means[1:], 'E_baselines': e_means[0],
							   'T_concentrations_p': t_sigmas[1:], 'T_concentrations': t_sigmas[0], 
							   'E_concentrations_p': e_sigmas[1:], 'E_concentrations': e_sigmas[0]}

		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding, self.power], filename = '{}_{}_slopes-{}_{}.pickle'.format(cnd_name,str(sj),method, list(freqs.keys())[0])),'wb') as handle:
			print('saving slopes dict')
			pickle.dump(ctf_inf, handle)

		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding, self.power], filename = '{}_{}_ctfs-{}_{}.pickle'.format(cnd_name,str(sj),method, list(freqs.keys())[0])),'wb') as handle:
			print('saving ctfs')
			pickle.dump(ctf, handle)
			
		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding, self.power], filename = '{}_info.pickle'.format(list(freqs.keys())[0])),'wb') as handle:
			print('saving info dict')
			pickle.dump(ctf_info, handle)

	def permuteSpatialCTF(self, subject_id, nr_perms = 1000):
		'''
		permuteSpatialCTF calculates spatial CTFs across subjects and conditions similar to spatialCTF, but with permuted channel labels.
		All relevant variables are read in from the CTF and info dict created in SpatialCTF, such that the only difference between functions
		is the permutation (i.e. block assignment is read in from spatialCTF)

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		nr_perms (int): number of permutations 

		Returns
		- - - -
		self(object): dict of CTF data
		'''
		
		warnings.filterwarnings('ignore')

		# load CTF parameters
		info = self.readCTF(subject_id, 'cnds','all', info = True)
		CTF = self.readCTF([subject_id], 'cnds','all', 'ctf')[0]
	
		# initialize dict to save CTFs
		CTFp = {}
		for cond in info['conditions']:													
			CTFp.update({cond:{}})			# update CTF dict such that condition data can be saved later			

		#get EEG data
		#file = self.FolderTrackerCTF('data', filename = str(subject_id) + '_EEG.mat')
		#EEG = h5py.File(file)
		pos_bins, conditions, eegs, EEG = self.readData(subject_id, info['conditions'])	

		#idx_channel, nr_selected, nr_total, all_channels = self.selectChannelId(subject_id, channels = 'all')
		idx_channel, nr_selected, nr_total, all_channels = np.arange(64), 64, 64, EEG.ch_names[:64]

		#eegs = EEG['erp']['trial']['data'][:,idx_channel,:]						# read in eeg data (drop scalp elecrodes)
		#eegs = np.swapaxes(eegs,0,2)											# eeg needs to be a (trials x electrodes x timepoints) array
		#eegs = eegs[CTF['perm']['idx_art'] == 0,:,:]
		nr_electrodes = eegs.shape[1]

		# loop through conditions
		for cond, curr_cond in enumerate(info['conditions']):

			eeg = eegs[CTF['perm']['condition'] == curr_cond,:,:len(info['tois'])] # selected condition eeg data equated for nr of time points 
			pos_bin = CTF['perm']['pos_bins'][CTF['perm']['condition'] == curr_cond]
			nr_trials = len(pos_bin)	

			# Preallocate arrays
			nr_trials_block = CTF[curr_cond]['nr_trials_block']			# SHOULD BE SAME PER CONDITION SO CAN BE CHANGED
			tf_evoked = np.empty((info['nr_freqs'],self.nr_iter, nr_perms,info['nr_samps'], self.nr_chans)) * np.nan; tf_total = np.copy(tf_evoked)
			idx_perm = np.empty((info['nr_freqs'],self.nr_iter,nr_perms, self.nr_blocks,nr_trials_block))			

			# loop through each frequency
			timings = []
			for freq in range(info['nr_freqs']):
				print('\n Frequency {} out of {}'.format(str(freq + 1), str(info['nr_freqs'])))
				
				# Time-Frequency Analysis
				filt_data_evoked, filt_data_total = self.powerAnalysis(eeg, info['tois'], self.sfreq, [info['freqs'][freq][0],info['freqs'][freq][1]], info['downsample'])

				# Loop through each 
				for itr in range(self.nr_iter):

					# grab block assignment for current iteration
					blocks = CTF[curr_cond]['blocks_' + str(itr)]
						
					# loop through permutations
					for p in range(nr_perms):
						print('\r{0}% of permutations ({1} out of {2} conditions; {3} out of {4} frequencies); iter {5}'.format((float(p)/nr_perms)*100, cond + 1, len(info['conditions']),str(freq + 1),str(info['nr_freqs']),itr + 1),)

						# Permute trial assignment within each block
						perm_pos_bin = np.empty((pos_bin.size)) * np.nan
						for b in range(self.nr_blocks):
							idx_p = np.random.permutation(nr_trials_block)			# create a permutation index
							permed_bins = np.empty(idx_p.size) * np.nan
							permed_bins[idx_p] = pos_bin[blocks == b]	 			# grab block b data and permute according to the index
							perm_pos_bin[blocks == b] = permed_bins					# put permuted data into perm_pos_bin
							idx_perm[freq, itr, p, b,:] = idx_p						# store the permutation NOT YET SAVED TO DICTIONARY!!!!!

						# average data for each position bin across blocks
						all_bins = np.arange(self.nr_bins)
						block_data_evoked = np.empty((self.nr_bins*self.nr_blocks, nr_electrodes, info['nr_samps'])) * np.nan 	# averaged evoked data
						block_data_total = np.copy(block_data_evoked) 								  							# averaged total data
						labels = np.empty((self.nr_bins*self.nr_blocks,1)) * np.nan 											# bin labels for averaged data
						block_nr = np.copy(labels)																				# block numbers for averaged data
						c = np.empty((self.nr_bins*self.nr_blocks,self.nr_chans)) * np.nan 										# predicted channel responses for averaged data

						bin_cntr = 0
						for i in range(self.nr_bins):
							for j in range(self.nr_blocks):
								idx = np.logical_and(perm_pos_bin == all_bins[i],blocks == j)
								evoked_2_mean = filt_data_evoked[idx,:,:][:,:,:] 												# ADJUST TOIS FOR DOWNSAMPLING				
								block_data_evoked[bin_cntr,:,:] = abs(np.mean(evoked_2_mean,axis = 0))**2						# evoked power is averaged over trials before calculating power
								total_2_mean = filt_data_total[idx,:,:][:,:,:]													# CHECK WARNING: NUMBERS SHOULD NO LONGER BE COMPLEX
								block_data_total[bin_cntr,:,:] = np.mean(total_2_mean,axis = 0)									# total power is calculated before average over trials takes place (see frequency loop)
								labels[bin_cntr] = i
								block_nr[bin_cntr] = j
								c[bin_cntr,:] = self.basisset[i,:]
								bin_cntr += 1		
							
						# Loop over all samples CHECK WHETHER THIS LOOP CAN BE PARALLIZED 
						for smpl in range(info['nr_samps']):

							data_evoked = block_data_evoked[:,:,smpl] # CHECK WHY I DO NOT AVERAGE HERE!!!!!!
							data_total = block_data_total[:,:,smpl]

							# FORWARD MODEL
							tmpeCR = np.empty((self.nr_blocks,self.nr_chans)) * np.nan
							tmptCR = np.copy(tmpeCR) 				# for shifted channel response	

							for blck in range(self.nr_blocks):

								###### ANALYSIS ON EVOKED POWER ######
								C2, C2s, W = self.forwardModel(data_evoked, labels, c, block_nr, blck, all_bins)
								tmpeCR[blck,:] = np.mean(C2s, axis = 0)	# CHECK AXIS IN MATLAB!!!!!!
								###### ANALYSIS ON TOTAL POWER ######
								
								C2, C2s, W = self.forwardModel(data_total, labels, c, block_nr, blck, all_bins)
								tmptCR[blck,:] = np.mean(C2s, axis = 0)
							
							tf_evoked[freq,itr,p, smpl,:] = np.mean(tmpeCR, axis = 0)			# save average of shifted channel response
							tf_total[freq,itr,p, smpl,:] = np.mean(tmptCR, axis = 0)		
	
			# save relevant data to CTF dictionary
			CTFp[curr_cond]['ctf'] = {}
			CTFp[curr_cond]['ctf'] = {'evoked': tf_evoked,'total': tf_total}

		# save data
		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_ctfperm_all.pickle'.format(str(subject_id))),'wb') as handle:
			print('saving CTF perm dict')
			pickle.dump(CTFp, handle)

		

	def Gaussian(self, x, amp, mu, sig):
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
		
	def fitGaussian(self, y):
		'''fits a Gaussian to input y
		
		Arguments:
			y {array} -- to be fitted data
		
		Returns:
			popt {array} -- array that contains amplitude, mean and sigma of the fitted Gaussian
		'''

		x = np.arange(y.size)
		mu = sum(x * y) / sum(y)
		sig = np.sqrt(sum(y * mu)**2) / sum(y)

		popt, pcov = curve_fit(self.Gaussian, x, y, p0=[max(y), mu, sig] )

		return popt 

	def fitCosToCTF(self, d, kstep=0.1):
		'''[summary]
		
		Arguments:
			d {[type]} -- [description]
		
		Keyword Arguments:
			kstep {float} -- [description] (default: {0.1})
		
		Returns:
			[type] -- [description]
		'''

		
		num_bins = d.size

		# possible concentration parameters to consider
		# step size of 0.1 is usually good, but smaller makes for better fits
		k = np.arange(3, 40 + kstep, kstep)

		# allocate storage arrays
		sse, baseline, amps = np.zeros((3, k.size))

		# hack to find best central point
		m_idx = np.argmax(d[int(np.round(num_bins/2.0) - np.round(num_bins * .1) - 1): 
		  				    int(np.round(num_bins/2.0) + np.round(num_bins * .1))])
		x = np.linspace(0, np.pi - np.pi/num_bins, num_bins)
		#u = x[int(m_idx + np.round(num_bins/2.0) - np.round(num_bins * .1) - 1)]
		u = x[int(num_bins/2)]

		# loop over all concentration parameters
		# estimate best amp and baseline offset and find combination that minimizes sse
		a, b = 1,0
		for i in range(k.size):
			# create the vm function
			pred = a * np.exp(k[i] * (np.cos(u - x) - 1)) + b
			# build a design matrix and use GLM to estimate amp and baseline
			X = np.zeros((num_bins, 2))
			X[:,0] = pred
			X[:,1] = np.ones(num_bins)
			betas, _, _, _ = np.linalg.lstsq(X, d, rcond=None)
			amps[i], baseline[i] = betas
			est = pred * betas[0] + betas[1]
			sse[i] = sum((est - d)**2)

		idx_min = np.argmin(sse)
		b = baseline[idx_min]
		a = amps[idx_min]
		concentration = k[idx_min]
		pred = a * np.exp(concentration * (np.cos(u - x) - 1)) + b
		rmse = np.sqrt(sum((pred - d)**2)/pred.size)

		return a, b, concentration, u, rmse


	def summarizeCTFs(self, X, nr_freqs, nr_samps):
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
				# collapse symmetric slope positions, eg (0, (45,315), (90,270), (135,225), 180)
				if self.nr_chans % 2 == 0:
					d = np.array([d[0]] + [np.mean((d[i],d[-i])) for i in range(1,steps)])
				else:
					d = np.array([np.mean((d[i],d[i+(-1-2*i)])) for i in range(steps)])
				slopes[f,s] = np.polyfit(range(1, len(d) + 1), d, 1)[0]
				#amps[f,s], means[f,s], sigmas[f,s] = self.fitGaussian(X[f,s])
				amps[f,s], baselines[f,s], concentrations[f,s], means[f,s], rmses[f,s] = self.fitCosToCTF(X[f,s])

		return slopes, amps, baselines, concentrations, means, rmses

	def CTFSlopes(self, subject_id, conditions,cnd_name = 'cnds', band = 'alpha', perm = False):
		'''
		CTFSlopes sets up data to calculate slope values for real or permuted data. Actual slope calculations across frequencies and sample points 
		is done with helper function calculateSlopes 

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		band (str): calculate slopes for a single frequency band or across a range of frequency bands (i.e. all)
		perm (bool): use real or permuted data for slope calculation 
		'''


		if perm:
			ctfs = self.readCTF(subject_id, cnd_name, band, 'ctfperm')
			sname = 'slopes_perm'
		else:	
			sname = 'slopes'
			ctfs = self.readCTF(subject_id, cnd_name, band)

		info = self.readCTF(subject_id, cnd_name, band, dicts = 'ctf', info = True)

		for i, sbjct in enumerate(subject_id):
			slopes = {}

			for cond in conditions:
				slopes.update({cond:{}})
				for power in ['evoked','total']:
					if perm: 
						data = np.mean(ctfs[i][cond]['ctf'][power],axis = 1)					# average across iterations
						nr_perms = data.shape[1]
						p_slopes = np.empty((data.shape[0],data.shape[2],nr_perms)) * np.nan
						for p in range(nr_perms):											# loop over permutations
							p_slopes[:,:,p] = self.calculateSlopes(data[:,p,:,:], info['nr_freqs'], info['nr_samps'])
						slopes[cond].update({power: p_slopes})
					else:	
						data = np.mean(np.mean(ctfs[i][cond]['ctf'][power],axis = 1),axis = 2) 	# average across iteration and cross-validation blocks
						sl = self.calculateSlopes(data, info['nr_freqs'], info['nr_samps'])
						slopes[cond].update({power: sl})

			with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_{}_{}_{}.pickle'.format(cnd_name,str(sbjct),sname,band)),'wb') as handle:
				print('saving slopes dict')
				pickle.dump(slopes, handle)	
	
	def topoWeights(self, subject_id, band = 'alpha', movie = False, ioi = [(-300,0),(0,200),(200,1200),(1200,1700),(1700,2200)]):
		'''

		'''

		file = '/home/moorselaar/Spatial_IEM/analysis/' 
		layout = mne.channels.read_layout('subject_layout_64_std.lout',file)

		all_channels = []
		channel_labels = ['A','B']
		for label in channel_labels:
			all_channels += [label + str(i) for i in range(1,33)]
		
		ctfs = self.readCTF(subject_id, band)
		info = self.readCTF(subject_id, band, info = True)		

		for cond in info['conditions']:
			
			# preallocate arrays
			We = np.empty((len(subject_id),info['nr_samps'],64)) * np.nan; Wt = np.copy(We)
			for i, sbjct in enumerate(subject_id):
				# get electrode indices
				a, b, c, used_channels = self.selectChannelId(sbjct, channels = 'all')
				idx_ch = np.array([all_channels.index(ch) for ch in used_channels])
				We[i,:,idx_ch] = np.squeeze(np.mean(np.mean(np.mean(ctfs[i][cond]['W']['evoked'],1),2),2)).T
				Wt[i,:,idx_ch] = np.squeeze(np.mean(np.mean(np.mean(ctfs[i][cond]['W']['total'],1),2),2)).T

			if movie: 	
				for i in range(Wt.shape[1]):
					f = plt.figure(figsize = (20,20))
					We_samp = np.nanmean(We[:,i,:],0)	
					Wt_samp = np.nanmean(Wt[:,i,:],0)
					# normalize against maximum
					We_samp = np.array([float(i)/np.max(We_samp) for i in We_samp])
					Wt_samp = np.array([float(i)/np.max(Wt_samp) for i in Wt_samp])

					ax = plt.subplot(2,1,1,title = 'CTF EVOKED (up), CTF TOTAL (down) ')	
					img = plt.imshow(np.array([[0,np.nanmax(np.nanmean(We,axis = 0))]]))
					img.set_visible(False)
					plt.colorbar(orientation='vertical')
					mne.viz.plot_topomap(We_samp, layout.pos[:,:2], names = all_channels, show_names = True, show = False, axis = ax, vmin = 0, vmax = 1)
					
					ax = plt.subplot(2,1,2)		
					img = plt.imshow(np.array([[0,np.nanmax(np.nanmean(Wt,axis = 0))]]))
					img.set_visible(False)
					plt.colorbar(orientation='vertical')
					mne.viz.plot_topomap(Wt_samp, layout.pos[:,:2], names = all_channels, show_names = True, show = False, axis = ax, vmin = 0, vmax = 1)

					plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder,self.decoding,'figs','topo'], filename = 'topo_{}_samp_{}.pdf'.format(cond, "%04d" % (i,))))
					plt.close()

			else:

				for ii in ioi:
					
					ind_i = np.array([(abs(info['times']- i)).argmin() for i in ii])
					f = plt.figure(figsize = (20,20))
					We_int = np.mean(np.nanmean(We[:,ind_i[0]:ind_i[1],:],0),0)	
					Wt_int = np.mean(np.nanmean(Wt[:,ind_i[0]:ind_i[1],:],0),0)
					# normalize against maximum
					We_int = np.array([float(i)/np.max(We_int) for i in We_int])
					Wt_int = np.array([float(i)/np.max(Wt_int) for i in Wt_int])

					ax = plt.subplot(2,1,1,title = 'Topo EVOKED (up), Topo TOTAL (down) ')	
					img = plt.imshow(np.array([[0,1]]))
					img.set_visible(False)
					plt.colorbar(orientation="vertical")
					mne.viz.plot_topomap(We_int, layout.pos[:,:2], names = all_channels, show_names = True, show = False, axis = ax, vmin = 0, vmax = 1)
					
					ax = plt.subplot(2,1,2)		
					img = plt.imshow(np.array([[0,1]]))
					img.set_visible(False)
					plt.colorbar(orientation="vertical")
					mne.viz.plot_topomap(Wt_int, layout.pos[:,:2], names = all_channels, show_names = True, show = False, axis = ax, vmin = 0, vmax = 1)

					plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder,self.decoding,'figs'], filename = 'topo_{}_samp_{}.pdf'.format(cond, ii)))
					plt.close()
		
	def plotCTF(self, subject_id, band = 'alpha', plot = 'individual'):
		'''
		plotCTF plots individual or group CTF data. At the moment plotting is only done in 2d space. Plots evoked (up) and total (down) 
		power in seperate subplots for all conditions specified in SpatialCTF.   

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		band (str): frequency band to plot
		plot (str): plot averaged grup data or individual data. If individual, plotCTF returns multiple plots dependent on the number of 
		subjects in subject_id (8 subjects per plot) 
		''' 		

		ctfs = self.readCTF(subject_id, band)
		info = self.readCTF(subject_id, band, info = True)
	
		for cond in info['conditions']:	
			# preallocate arrays
			e_ctf = np.zeros((len(subject_id),info['nr_samps'],self.nr_chans)); t_ctf = np.copy(e_ctf)

			for i, sbjct in enumerate(subject_id):
				e_ctf[i,:,:] = np.squeeze(np.mean(np.mean(ctfs[i][cond]['ctf']['evoked'],1),2))
				t_ctf[i,:,:] = np.squeeze(np.mean(np.mean(ctfs[i][cond]['ctf']['total'],1),2))

			if self.nr_chans % 2 == 0:
				x = np.linspace(-180,180,self.nr_chans + 1)
			else:
				x = np.linspace(-180,180,self.nr_chans)

			X = np.tile(x,(info['nr_samps'],1)).T
			Y = np.tile(info['times'],(len(x),1))

			if plot == 'individual':
				for i, sbjct in enumerate(list(subject_id)):
					if i % 8 == 0: 
						plt.figure(figsize= (15,10))
						idx = [1,2,7,8,13,14,19,20]
						idx_cntr = 0
					
					ZE = e_ctf[i,:,:]
					ZT = t_ctf[i,:,:]
					if self.nr_chans % 2 == 0:
						ZE = np.hstack((ZE,ZE[:,0].reshape(-1,1))).T
						ZT = np.hstack((ZT,ZT[:,0].reshape(-1,1))).T
					
					levels = np.linspace(-0.2,0.9,1000)
					ax = plt.subplot(11,2,idx[idx_cntr],title = 'SUBJECT ' + str(sbjct), ylabel = 'Channel')
					ax.set_xticklabels([])
					plt.contourf(Y,X,ZE, levels, cmap = cm.viridis)											# plot evoked
					plt.colorbar(ticks = (levels[0],levels[-1]))
					plt.subplot(11,2,idx[idx_cntr]+2,xlabel = 'Time (ms)', ylabel = 'Channel')			# plot total
					plt.contourf(Y,X,ZT, levels, cmap = cm.viridis)
					plt.colorbar(ticks = (levels[0],levels[-1]))
					idx_cntr += 1
	 
				nr_plots = int(np.ceil(len(subject_id)/8.0))
				for i in range(nr_plots,0,-1):
					plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = '{}_ctfs__ind_{}_{}.pdf'.format(band,cond,i)))
					plt.close()

			elif plot == 'group':
				ectf = np.nanmean(e_ctf,0) 
				tctf = np.nanmean(t_ctf,0) 
				if self.nr_chans % 2 == 0:
					ectf = np.hstack((ectf,ectf[:,0].reshape(-1,1)))
					tctf = np.hstack((tctf,tctf[:,0].reshape(-1,1)))
				
				levels = np.linspace(-0.2,0.9,1000)
				ZE = ectf.T
				ZT = tctf.T
				
				plt.figure(figsize= (15,10))
				ax = plt.subplot(2,1,1,title = 'CTF EVOKED (up), CTF TOTAL (down) ', ylabel = 'Channel')
				ax.set_xticklabels([])
				plt.contourf(Y,X,ZE, levels, cmap = cm.viridis)
				plt.colorbar(ticks = (levels[0],levels[-1]))
				ax = plt.subplot(2,1,2,xlabel = 'Time (ms)', ylabel = 'Channel')
				plt.contourf(Y,X,ZT, levels, cmap = cm.viridis)
				plt.colorbar(ticks = (levels[0],levels[-1]))
				plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = '{}_ctfs_group_{}.pdf'.format(band,cond)))
				plt.close()	

	def corrCTFSlopes(self, subject_id):
		'''

		'''

		from scipy.stats.stats import pearsonr

		info = self.readCTF(subject_id, 'alpha', info = True)
		ctf_mem, ctf_search = [],[]
		for sbjct in subject_id:
			with open(self.FolderTracker(['ctf',self.channel_folder,'memory'], filename = '{}_{}_{}.pickle'.format(str(sbjct),'slopes','alpha')),'rb') as handle:
				ctf_mem.append(pickle.load(handle))		
			with open(self.FolderTracker(['ctf',self.channel_folder,'search'], filename = '{}_{}_{}.pickle'.format(str(sbjct),'slopes','alpha')),'rb') as handle:
				ctf_search.append(pickle.load(handle))	
	
		slopes_mem_dual = np.vstack([ctf_mem[i]['dual_task']['total'] for i in range(len(subject_id))])
		slopes_mem_single = np.vstack([ctf_mem[i]['single_task']['total'] for i in range(len(subject_id))])				
		slopes_search_single = np.vstack([ctf_search[i]['single_task']['total'] for i in range(len(subject_id))])
		slopes_search_dual = np.vstack([ctf_search[i]['dual_task']['total'] for i in range(len(subject_id))])

		slopes_search_cor = slopes_search_dual - slopes_search_single
		slopes_mem_cor = slopes_mem_dual - slopes_mem_single

		bins = np.arange(0,2300,100)

		for plot in ['ind','group']:

			if plot == 'ind':
				plt.figure(figsize= (20,10))
				for sj in range(16):
					ax = plt.subplot(8,2, sj + 1)

					plt.plot(info['times'],slopes_mem_cor[sj,:], color = 'red', label = 'memory')
					plt.plot(info['times'],slopes_search_cor[sj,:], color = 'green', label = 'search')

					plt.xlim(-300,2200)

				plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder], filename = 'correction_ind.pdf'))
				plt.close()		

			if plot == 'group':
				plt.figure(figsize= (20,10))

				plt.plot(info['times'],np.mean(slopes_mem_cor, axis = 0), color = 'red', label = 'memory')
				plt.plot(info['times'],np.mean(slopes_search_cor, axis = 0), color = 'green', label = 'search')
				plt.xlim(-300,2200)
				plt.legend(loc = 'best')

				plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder], filename = 'correction_group.pdf'))
				plt.close()	

				plt.figure(figsize= (10,10))
				ax = plt.subplot(111)
				bin_mem, bin_search = [],[]

				for i in range(bins.shape[0] - 1):
					idx = [np.argmin(abs(info['times'] - j)) for j in bins[i:i+2]]
					bin_mem.append(np.mean(slopes_mem_cor[:,idx[0]:idx[1]]))
					bin_search.append(np.mean(slopes_search_cor[:,idx[0]:idx[1]]))

				z = np.polyfit(bin_mem, bin_search, 1)	
				p = np.poly1d(z)
				ax = plt.subplot(111)
				plt.plot(bin_mem,p(bin_mem),'r--')

				for i in range(len(bin_mem)):
					plt.plot(bin_mem[i],bin_search[i],'o', color = mcolors.cnames.keys()[i], label = '{}-{}'.format(0 + 100*i,100 + 100*i))
				
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])	
				ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		
				plt.savefig(self.FolderTracker( extension = ['ctf',self.channel_folder], filename = 'WM-search_slopeselectivity_corrected.pdf'))
				plt.close()	

		bin_mem, bin_search = [],[]
		
		for i in range(bins.shape[0] - 1):
			idx = [np.argmin(abs(info['times'] - j)) for j in bins[i:i+2]]
			if plot == 'individual':
				bin_mem.append(np.mean(slopes_mem[:,idx[0]:idx[1]], axis = 1))
				bin_search.append(np.mean(slopes_search[:,idx[0]:idx[1]], axis = 1))
			else:
				bin_mem.append(np.mean(slopes_mem_dual[:,idx[0]:idx[1]]))
				bin_search.append(np.mean(slopes_search[:,idx[0]:idx[1]]))

		if plot == 'individual':
			for sj in range(16):
				z = np.polyfit(bin_mem[sj], bin_search[sj], 1)	
				p = np.poly1d(z)
				ax = plt.subplot(4,4, sj + 1)
				plt.plot(bin_mem[sj],p(bin_mem[sj]),'r--')

				plt.plot(bin_mem[sj],bin_search[sj],'o')
				
			plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder], filename = 'WM-search_slopeselectivity_ind.pdf'))
			plt.close()	


		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])	
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.xlabel('Memory CTF slope')
		plt.ylabel('Search CTF slope')
		plt.xlim(0, 0.14)
		plt.ylim(0, 0.14)

		plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder], filename = 'scatter_group.pdf'))
		plt.close()


		plt.figure(figsize= (10,10))
		bins = np.arange(500,2300,100)
		bin_mem, bin_search = [],[]

		for i in range(bins.shape[0] - 1):
			idx = [np.argmin(abs(info['times'] - j)) for j in bins[i:i+2]]
			bin_mem.append(np.polyfit(range(1, len(slopes_mem[idx[0]:idx[1]]) + 1), slopes_mem[idx[0]:idx[1]],1)[0])
			bin_search.append(np.polyfit(range(1, len(slopes_search[idx[0]:idx[1]]) + 1), slopes_search[idx[0]:idx[1]],1)[0])

		z = np.polyfit(bin_mem, bin_search, 1)	
		p = np.poly1d(z)
		ax = plt.subplot(111)
		plt.plot(bin_mem,p(bin_mem),'r--')

		for i in range(len(bin_mem)):
			plt.plot(bin_mem[i],bin_search[i],'o', color = mcolors.cnames.keys()[i], label = '{}-{}'.format(500 + 100*i,600 + 100*i))

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])	
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.xlabel('Memory CTF slope')
		plt.ylabel('Search CTF slope')
		plt.xlim(-0.001, 0.001)
		plt.ylim(-0.001, 0.001)

		plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder], filename = 'scatter_intervalslope_group_newint.pdf'))
		plt.close()	


		cor_matrix = np.zeros((slopes_search.shape[1],slopes_mem.shape[1]))
		p_matrix = np.zeros((slopes_search.shape[1],slopes_mem.shape[1]))

		for a in range(slopes_search.shape[1]):
			for b in range(slopes_mem.shape[1]):
				cor = pearsonr(slopes_search[:,a],slopes_mem[:,b])
				cor_matrix[a,b] = cor[0]
				p_matrix[a,b] = cor[1]

		plt.figure(figsize= (20,10))
		idx = 1		
		for i in range(16):	
			ax = plt.subplot(4,4,idx)
			plt.xlim(0, 0.2)
			plt.ylim(0, 0.2)
			plt.plot(slopes_mem[i,:], slopes_search[i,:],'o')
			idx += 1

	
		plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder], filename = 'scatter_individual.pdf'))
		plt.close()
	
		cor_matrix[p_matrix > 0.05] = 0		
		plt.imshow(cor_matrix, cmap = cm.viridis, interpolation='none', aspect='auto', origin = 'lower', extent=[-300,2200,-300,2200])
		plt.colorbar(ticks = (-1,1))
		plt.xlabel('memory')
		plt.ylabel('search')

		print('saving')
		plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder], filename = 'slope_corr_matrix.pdf'))
		plt.close()
	

	def plotCTFSlopes(self, subject_id,band = 'alpha', plot = 'individual'):		
		'''
		plotCTFSlopes plots individual or group average (bootstrapped) CTF slopes. Plots evoked and total power in seperate plots. All conditions are shown within a single plot  

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		band (str): frequency band to plot
		plot (str): plot averaged grup data or individual data. If individual, plotCTFSlopes returns multiple plots dependent on the number of 
		subjects in subject_id (8 subjects per plot) 
		'''

		all_slopes = self.readCTF(subject_id, band,'slopes')
		info = self.readCTF(subject_id, band, info = True)
		interval = [0,2200]
		index = [np.argmin(abs(info['times']-i)) for i in interval]

		#for condition in ['DrTv','DvTr','DvTv']:
		#	slopes = np.zeros((16,2))
		#	for sj, sl in enumerate(all_slopes):
		#		slopes[sj,0] = np.mean(sl[condition]['evoked'][0,index[0]:index[1]])
		#		slopes[sj,1] = np.mean(sl[condition]['total'][0,index[0]:index[1]])

			#np.savetxt(self.FolderTrackerCTF('data',filename = '{}_memory_int_whole.csv'.format(condition)),slopes, delimiter=',')
	
		#plot_index = [1,3]
		for cnd_idx, power in enumerate(['evoked','total']):
			
			if plot == 'group':
				slopes = self.bootstrapSlopes(subject_id, all_slopes, power, info, 10000)
				
				plt.figure(figsize= (20,10))
				ax = plt.subplot(1,1,1)
				plt.xlim(info['times'][0], info['times'][-1])
				plt.ylim(-0.3, 0.3)
				plt.title(power + ' power across time')
				plt.axhline(y=0, xmin = 0, xmax = 1, color = 'black')

				for i, cond in enumerate(info['conditions']):
					dat = slopes[cond]['M']
					error = slopes[cond]['SE']
					#plt.plot(info['times'], dat, color = ['g','r','y','b'][i], label = cond)
					plt.plot(info['times'], dat, label = cond)
					#plt.fill_between(info['times'], dat + error, dat - error, alpha = 0.2, color = ['g','r','y','b'][i])
					plt.fill_between(info['times'], dat + error, dat - error, alpha = 0.2)
				
				#if len(info['conditions']) > 1:
				#	dif = slopes[info['conditions'][0]]['slopes'] - slopes[info['conditions'][1]]['slopes']
				#	zmap_thresh = self.clusterPerm(dif)
				#	plt.fill_between(info['times'], -0.002, 0.002, where = zmap_thresh != 0, color = 'grey')
					
				plt.legend(loc='upper right', shadow=True)
				#plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder,self.decoding,'figs'], filename = '{}_{}_slopes_group.png'.format(band,power)))
				plt.savefig(self.FolderTracker(extension = ['ctf',self.channel_folder,self.decoding,'figs'], filename = '{}_{}_slopes_group.pdf'.format(band,power)))
				plt.close()	

			elif plot == 'individual':										
				idx = [1,2,5,6,9,10,13,14]
				for i, sbjct in enumerate(subject_id):
					if i % 8 == 0: 
						plt.figure(figsize= (15,10))
						idx_cntr = 0	
					ax = plt.subplot(7,2,idx[idx_cntr],title = 'SUBJECT ' + str(sbjct), ylabel = 'CTF Slope', xlim = (info['times'][0],info['times'][-1]), ylim = (-0.3,0.3))
					for j, cond in enumerate(info['conditions']): 	
						#plt.plot(info['times'], all_slopes[i][cond][power].squeeze(), color = ['g','r','y','b'][j], label = cond)
						plt.plot(info['times'], all_slopes[i][cond][power].squeeze(), label = cond)		
					plt.legend(loc='upper right', frameon=False)	
					plt.axhline(y=0, xmin=info['times'][0], xmax=info['times'][-1], color = 'black')
					idx_cntr += 1

				nr_plots = int(np.ceil(len(subject_id)/8.0))
				for i in range(nr_plots,0,-1):
					plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'],filename = '{}_{}_slopes_ind_{}.pdf'.format(band,power,str(i))))
					plt.close()	

	def bootstrapSlopes(self, subject_id, data, power, info, b_iter = 10000):
		'''
		bootstrapSlopes uses a bootstrap procedure to calculate standard error of slope estimates.

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		data (list): list of individual subjects slope dictionaries
		power (str): evoked or total power slopes
		info (dict): dictionary with variables set in spatialCTF
		b_iter (int): number of bootstrap iterations

		Returns
		- - - -
		bot(dict): dict of bootstrapped slopes
		'''	

		idx_boot = np.empty((len(info['conditions']), b_iter, len(subject_id)))						# store bootstrap indices for replication	
		slope_boot = np.empty((len(info['conditions']), b_iter, len(subject_id), info['nr_samps']))
		mean_boot =  np.empty((len(info['conditions']),b_iter, info['nr_samps'])) * np.nan

		boot = {}
		for c, cond in enumerate(info['conditions']):
			slopes = np.vstack([data[i][cond][power] for i in range(len(subject_id))])
			for b in range(b_iter):
				idx = np.random.choice(len(subject_id),len(subject_id),replace = True) 				# sample nr subjects observations from the slopes sample (with replacement)
				mean_boot[c,b,:] = np.mean(slopes[idx,:],axis = 0)
				idx_boot[c,b,:] = idx	

			boot.update({cond: {}})
			boot[cond]['SE'] = np.std(mean_boot[c,:],axis = 0)				# save bootstrapped SE
			boot[cond]['M'] = np.mean(slopes,axis = 0)						# save actual mean CTF
			boot[cond]['idx'] = idx_boot[c,:]								# save bootstrap indices 
			boot[cond]['slopes'] = slopes

		return boot										

	def clusterPerm(self, dif, nr_freq = 1, clust_pval = 0.05, voxel_pval = 0.05, test_stat = 'sum', nr_perm = 1000):
		'''
		Doc String clusterPerm
		'''

		# initialize null hypothesis matrices
		perm_vals = np.zeros((nr_perm, nr_freq, dif.shape[-1])) 
		max_clust = np.zeros((nr_perm, 1))

		# permutation loop
		for p in range(nr_perm):

			c_labels = np.array(np.random.randint(2, size = dif.shape[0]), dtype = bool)
			t_labels = np.array(np.random.randint(2, size = dif.shape[0]), dtype = bool)
			f_labels = np.array(np.random.randint(2, size = dif.shape[0]), dtype = bool)

			temp_perm = np.copy(dif)
			temp_perm[c_labels,:] *=  -1

			perm_vals[p,:,:] = np.mean(temp_perm, axis = 0)

		real_mean = np.mean(dif, axis = 0)
		zmap = 	(np.mean(dif, axis = 0) - np.mean(perm_vals, axis = 0)) / np.std(perm_vals, axis = 0)

		thresh_mean = np.copy(real_mean)
		thresh_mean[np.squeeze(abs(zmap) < norm.ppf(1-voxel_pval/2))] = 0

		# now the cluster cor  rection will be done on the permuted data, thus making no assumptions about parameters for p-values
		for p in range(nr_perm):
			#if p % 100 == 0: print '..{}'.format(p)

			# for cluster correction, apply uncorrected threshold and get maximum cluster sizes
			fake_corrs_z = (perm_vals[p,:,:] - np.mean(perm_vals, axis = 0)) / np.std(perm_vals, axis = 0)
			fake_corrs_z[abs(fake_corrs_z) < norm.ppf(1-voxel_pval/2)] = 0


			# get the number of elements in largest supra-threshold cluster
			clust_info, nr_clust = label(fake_corrs_z, np.ones((3,3)))

			if test_stat == 'count':
				try:
					max_clust[p] = max([np.where(clust_info == cl + 1)[0].size for cl in range(nr_clust)])	
				except:
					pass			
			elif test_stat == 'sum':
				if nr_clust > 0:
					temp_clust_sum = np.zeros((nr_clust))
					for cl in range(nr_clust):
						temp_clust_sum[cl] = np.sum(abs(fake_corrs_z[0,np.where(clust_info == cl + 1)[1]]))

					max_clust[p] = max(abs(temp_clust_sum))	
		
		# apply cluster-level corrected threshold
		zmap_thresh = np.copy(zmap)

		# uncorrected pixel-level threshold
		zmap_thresh[abs(zmap_thresh) < norm.ppf(1-voxel_pval/2)] = 0

		# find islands and remove those smaller than cluster size threshold
		clust_info, nr_clust = label(zmap_thresh, np.ones((3,3)))
		if test_stat == 'count':
			clust_len = [np.where(clust_info == cl + 1)[0].size for cl in range(nr_clust)]
		elif test_stat == 'sum':
			clust_len = np.zeros(nr_clust)	
			for cl in range(nr_clust):
				clust_len[cl] = np.sum(abs(zmap_thresh[0,np.where(clust_info == cl + 1)[1]]))

		clust_thresh = np.percentile(max_clust, 100 - clust_pval*100)	
		# identify clusters to remove
		clusters_2_remove = np.where(clust_len < clust_thresh)[0]
		ind_2_remove = np.hstack([np.where(clust_info == cl + 1)[1] for cl in clusters_2_remove])

		# remove clusters
		zmap_thresh[0,ind_2_remove] = 0
		zmap_thresh = zmap_thresh.squeeze()

		return zmap_thresh		

	def plotTimefrequencySlopes(self, subject_id, perm = False, nr_perms = 1000):
		'''
		plotTimefrequencySlopes plots slope estimates across frequencies and samplepoints. Conditions are plot within seperate plots with seperate 
		subplots for evoked (up) and total power (bottem). If perm is set to True, function also returns seperate plots with p-values and significance 
		values for the timefrequency plots.  

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		perm (bool): If True also plot significance maps
		nr_perms (int): nr of permutations used to calculate significance maps 
		'''

		# read in slopes
		real_s = self.readCTF(subject_id, 'all','slopes')
		info = self.readCTF(subject_id, 'all', info = True)
		if perm:
			perm_s = self.readCTF(subject_id, 'all','slopes_perm')

		# set plotting parameters
		freqs = np.linspace(info['freqs'][0][0], info['freqs'][-1][0] + 4,info['nr_freqs'])	
		X = np.tile(freqs,(info['nr_samps'],1)).T
		times = np.linspace(info['times'][0],info['times'][-1],info['nr_samps'])
		Y = np.tile(times,(info['nr_freqs'],1))
		
		dif_e = []
		dif_t = []
		for cnd_idx, cond in enumerate(info['conditions']):
		
			# combine individual slope data
			r_slopes_ev = np.empty((len(subject_id), info['nr_freqs'],info['nr_samps'])) * np.nan
			r_slopes_to = np.copy(r_slopes_ev)
			if perm:
				p_slopes_ev = np.empty((len(subject_id), info['nr_freqs'],info['nr_samps'], nr_perms)) * np.nan
				p_slopes_to = np.copy(p_slopes_ev)	

			for i in range(len(subject_id)):
				r_slopes_ev[i,:,:] = real_s[i][cond]['evoked']
				r_slopes_to[i,:,:] = real_s[i][cond]['total']
				if perm: 
					p_slopes_ev[i,:,:] = perm_s[i][cond]['evoked']
					p_slopes_to[i,:,:] = perm_s[i][cond]['total']

			dif_e.append(r_slopes_ev); 	dif_t.append(r_slopes_to)	

			if perm:		
				plt.figure(figsize= (15,10))
				p_val_e, sig_e = self.permTTest(r_slopes_ev, p_slopes_ev, nr_perms = nr_perms, p_thresh = 0.01)
				p_val_t, sig_t = self.permTTest(r_slopes_to, p_slopes_ev, nr_perms = nr_perms, p_thresh = 0.01)
				
				levels = np.linspace(0,1,1000)
				ax = plt.subplot(2,2,1,title = 'p-values EVOKED', ylabel = 'Frequency')

				plt.contourf(Y,X,p_val_e,levels, cmap = cm.jet_r)
				plt.colorbar(ticks = (levels[0],levels[-1]))

				ax = plt.subplot(2,2,3,title = 'p-values TOTAL', ylabel = 'Frequency')
				plt.contourf(Y,X,p_val_t,levels, cmap = cm.jet_r)
				plt.colorbar(ticks = (levels[0],levels[-1]))

				ax = plt.subplot(2,2,2,title = 'sig EVOKED < 0.01')
				plt.imshow(sig_e, cmap = cm.viridis, interpolation='none', aspect='auto', origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[-1]])

				ax = plt.subplot(2,2,4,title = 'sig TOTAL  < 0.01')
				plt.imshow(sig_t, cmap = cm.viridis, interpolation='none', aspect='auto', origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[-1]])
								
				plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = 'all_sig_' + cond + '.pdf'))
				plt.close()	

			plt.figure(figsize= (20,10))
			levels = np.linspace(-0.2,0.2,1000)
			ax = plt.subplot(2,1,1,title = 'slopes EVOKED (up), slopes TOTAL (under) ', ylabel = 'Frequency')
			ax.set_xticklabels([])
			evoked = np.mean(r_slopes_ev,axis = 0); 
			if perm: 
				evoked[sig_e == 0] = 0
			plt.contourf(Y,X,evoked,levels, cmap = cm.viridis)
			plt.colorbar(ticks = (levels[0],levels[-1]))
			
			ax = plt.subplot(2,1,2, ylabel = 'Frequency')
			total = np.mean(r_slopes_to,axis = 0); 
			if perm: 
				total[sig_t == 0] = 0
			plt.contourf(Y,X,total,levels, cmap = cm.viridis)
			plt.colorbar(ticks = (levels[0],levels[-1]))

			plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = 'all_freqs_' + cond + '.pdf'))
			#plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = 'all_freqs_' + cond + '.eps'), format = 'eps',dpi = 1000)
			plt.close()

		if len(dif_e) > 1:	
			
			# TEMPORARILY COMMENTED OUT FOR MATLAB INPUT
			#zmap_thresh_e = self.clusterPerm(dif_e[0] - dif_e[1], 14)
			#zmap_thresh_t = self.clusterPerm(dif_t[0] - dif_t[1], 14)
			zmap_thresh_e = h5py.File(self.FolderTrackerCTF('data', filename = 'evoked_{}.mat'.format(self.decoding)))['evoked']['zmapthresh'][:]
			zmap_thresh_t = h5py.File(self.FolderTrackerCTF('data', filename = 'total_{}.mat'.format(self.decoding)))['total']['zmapthresh'][:]

			zmap_thresh_e[zmap_thresh_e > 0] = 1; zmap_thresh_e[zmap_thresh_e < 0] = -1
			zmap_thresh_t[zmap_thresh_t > 0] = 1; zmap_thresh_t[zmap_thresh_t < 0] = -1


			plt.figure(figsize= (20,10))

			ax = plt.subplot(2,2,2,title = 'Significant cluster Evoked (up), Total (down)')
			#plt.imshow(zmap_thresh_e.T, cmap = cm.viridis, interpolation='none', aspect='auto', origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[-1]])
			plt.contourf(Y,X,zmap_thresh_e.T, levels = np.linspace(-1,0,3))
			plt.colorbar(ticks = (-1,0))
			ax = plt.subplot(2,2,4)
			#plt.imshow(zmap_thresh_t.T, cmap = cm.viridis, interpolation='none', aspect='auto', origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[-1]])
			plt.contourf(Y,X,zmap_thresh_t.T, levels = np.linspace(-1,0,3))
			plt.colorbar(ticks = (-1,0))
			plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = 'all_freqs_clusters_right.eps'), format = 'eps',dpi = 1000)
			plt.close()


	def permTTest(self, real, perm, nr_perms = 1000, p_thresh = 0.01):
		'''
		permTTest calculates p-values for the one-sample t-stat for each sample point across frequencies using a surrogate distribution generated with 
		permuted data. The p-value is calculated by comparing the t distribution of the real and the permuted slope data across sample points. 
		The t-stats for both distribution is calculated with

		t = (m - 0)/SEm

		, where m is the sample mean slope and SEm is the standard error of the mean slope (i.e. stddev/sqrt(n)). The p value is then derived by dividing 
		the number of instances where the surrogate T value across permutations is larger then the real T value by the number of permutations.  

		Arguments
		- - - - - 
		real(array):  (nr freqs, nr samps, nr chans) 
		perm(array): n
		nr_perms (int): number of permutations 
		p_thresh (float): threshold for significance. All p values below this value are considered to be significant

		Returns
		- - - -
		p_val (array): array with p_values across frequencies and sample points
		sig (array): array with significance indices (i.e. 0 or 1) across frequencies and sample points
		'''

		# preallocate arrays
		p_val = np.empty((real.shape[1],real.shape[2])) * np.nan
		sig = np.zeros((real.shape[1],real.shape[2])) 	# will be filled with 0s (non-significant) and 1s (significant)

		# calculate the real and the surrogate one-sample t-stats
		r_M = np.mean(real, axis = 0); p_M = np.mean(perm, axis = 0)
		r_SE = np.std(real, axis = 0)/sqrt(len(subject_id)); p_SE = np.std(perm, axis = 0)/sqrt(len(subject_id))
		r_T = r_M/r_SE; p_T = p_M/p_SE

		# calculate p-values
		for f in range(real.shape[1]):
			for s in range(real.shape[2]):
				surr_T = p_T[f,s,:]
				p_val[f,s] = len(surr_T[surr_T>r_T[f,s]])/float(nr_perms)
				if p_val[f,s] < p_thresh:
					sig[f,s] = 1

		return p_val, sig

	def readCTF(self, subject_id, cnd_name, band,  dicts= 'ctf', info = False):
		'''
		readCTF helper function to read in saved CTF or slope dictionaries.

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		band (str): frequency band
		info (boolean): If True reads in the info dictionary for the specified frequency band 

		Returns
		- - - -
		ctf(dict): dict of CTF data or dict of info data
		'''

		if not info:
			ctf = []
			for sbjct in subject_id:
				print (sbjct)
				with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_{}_{}_{}.pickle'.format(cnd_name,str(sbjct),dicts,band)),'rb') as handle:
					ctf.append(pickle.load(handle))
		else:
			with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_info.pickle').format(band),'rb') as handle:
				ctf = pickle.load(handle)		
	
		return ctf	

	def selectChannelId(self, EEG, channels = 'all'):
		'''


		Arguments
		- - - - - 
		subject_id (int): subject id
		channels (list | str): list of channels used for analysis. If 'all' all channels are used for analysis 

		Returns
		- - - -
		
		''' 


		if 'all' in channels:
			picks = mne.pick_types(EEG.info, eeg=True, exclude='bads')
		elif channels == 'posterior_channels':
			to_select = ['TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
						'TP8','P7','P5','P3','P1','Pz','P2','P4','P6',
						'P8','P9','PO7','PO3','POz','PO4','PO8','P10','O1',
						'Oz','O2','Iz']
			picks = np.array([EEG.ch_names.index(elec) for elec in to_select])	

		return	picks


	def positionHeog(self, subject_id, interval, filter_time):
		'''

		'''

		conditions = ['single_task','dual_task']

		time = np.arange(interval[0] - filter_time,interval[1] +filter_time,1000.0/self.sfreq)
		idx_time = [np.argmin(abs(time - i)) for i in [-300, 2200]]
		idx_base = [np.argmin(abs(time - i)) for i in [-300, -3]]
		hg = np.zeros((2,16,8,idx_time[1] - idx_time[0]))


		plt.figure(figsize= (20,20))
		plt_idx = 1

		for sj in subject_id:

			# select HEOG data
			file = self.FolderTrackerCTF('data', filename = str(sj) + '_EEG.mat')
			EEG = h5py.File(file)
			heog = EEG['erp']['trial']['data'][:,-2,:]
			idx_art = np.squeeze(EEG['erp']['arf']['artifactIndCleaned']) 


			# get position info and filter out artifact trials
			pos_bins, condition = self.behaviorCTF(sj, 64, conditions, 'location_bin_search', 'block_type')
			heog = heog[:,idx_art == 0] 
			heog = np.swapaxes(heog,0,1)									
			pos_bins = pos_bins[idx_art == 0]
			condition = condition[idx_art == 0]

			# baseline correct heog	
			base_mean = heog[:,idx_base[0]:idx_base[1]].mean(axis = 1).reshape(heog.shape[0],-1)
			base_heog = heog[:,idx_time[0]:idx_time[1]] - base_mean

			for c, cnd in enumerate(conditions):
				for b in range(8):
					hg[c,i,b,:] = np.mean(base_heog[np.logical_and(pos_bins == b, condition == c),:], axis = 0)

			for i, cnd in enumerate(conditions):
				if sj == 1:
					plt.subplot(16,2, plt_idx,title = cnd)
				else:		
					plt.subplot(16,2, plt_idx)

				for line in range(8):
					plt.plot(time[idx_time[0]:idx_time[1]],np.mean(base_heog[np.logical_and(pos_bins == line,condition == i),:], axis = 0), color = ['red','green','blue','pink','orange','black','yellow','purple'][line])	
					plt.ylim(-15,15)
					
				if plt_idx % 2 == 1 and sj == 8:
					plt.ylabel('VEOG amplitude (microvolts)')
				if sj == 16:	
					plt.xlabel('Time (ms)')
				#plt.legend(loc = 'best')	
				plt.xlim(-300, 2200)

				plt_idx += 1

		plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = 'veog_ind_mem.pdf'))		
		plt.close()	

		plt.figure(figsize= (20,10))			

		for i, cnd in enumerate(conditions):
			plt.subplot(1,2, i + 1,title = cnd)	
			for line in range(8):
				plt.plot(time[idx_time[0]:idx_time[1]],hg[i].mean(axis = 0)[line], label = str(line),color = ['red','green','blue','pink','orange','black','yellow','purple'][line])

			plt.legend(loc = 'best')
			plt.xlim(-300, 2200)
			plt.ylim(-15,15)
			plt.xlabel('Time (ms)')
			plt.ylabel('HEOG amplitude (microvolts)')

		plt.savefig(self.FolderTracker(['ctf',self.channel_folder,self.decoding,'figs'], filename = 'heog_search.pdf'))		
		plt.close()

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '4'
	os.environ['NUMEXP_NUM_THREADS'] = '4'
	os.environ['OMP_NUM_THREADS'] = '4'
	project_folder = '/home/dvmoors1/BB/Dist_suppression'
	os.chdir(project_folder) 
	subject_id = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]	

	### INITIATE SESSION ###
	header = 'target_loc'
	if header == 'target_loc':
		conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
	else:
		conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']

	session = SpatialEM('all_channels_no-eye', header, nr_iter = 10, nr_blocks = 3, nr_bins = 6, nr_chans = 6, delta = False)
	
	### CTF and SLOPES ###
	for sj in subject_id:
		session.CTF(sj, [-300, 800], conditions, method = 'fold', downsample = 4)
	#	session.crosstrainCTF(sj, [-300, 800], ['DvTv_0','DvTv_3'], ['DrTv_0', 'DrTv_3'], tgm = False, name = 'V-R')
	#	print subject
	#	session.spatialCTF(sj,[-300,800],500, conditions = conditions, freqs = dict(alpha = [8,12]))
	#	session.spatialCTF(subject, [-300,800],500, conditions = 'all', freqs = dict(all=[4,30]), downsample = 4)
	#	session.permuteSpatialCTF(subject_id = subject, nr_perms = 500)
	
	#session.CTFSlopes(subject_id, conditions = conditions,cnd_name = 'cnds', band = 'alpha', perm = False)
	#session.CTFSlopes(subject_id, conditions = ['all'], cnd_name = 'all', band = 'all', perm = False)
	#session.CTFSlopes(subject_id, conditions = ['all'], cnd_name = 'all', band = 'all', perm = True)


	#subject_id = [2,3,4,6,7,9]
	#session.topoWeights(subject_id)
	### PLOTTING ###
	#for plot in ['group','individual']:
	#	session.plotCTF(subject_id = subject_id, band = 'alpha',plot = plot)
	#	session.plotCTFSlopes(subject_id,'alpha', plot = plot)

	#session.plotTimefrequencySlopes(subject_id, perm = True, nr_perms = 500)
	#session.positionHeog(subject_id, [-300, 2200], 500)


