"""
analyze EEG data

Created by Dirk van Moorselaar on 13-06-2018.
Copyright (c) 2018 DvM. All rights reserved.
"""
import os
import mne
import pickle
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Generic, Union, Tuple, Any
from mne.filter import filter_data
from mne.time_frequency import tfr_array_morlet
from mne.baseline import rescale
from scipy.signal import hilbert
from numpy.fft import fft, ifft,rfft, irfft

from eeg_analyses.ERP import ERP
from support.FolderStructure import *
from support.support import *
from signals.signal_processing import *
from IPython import embed


class TF(FolderStructure):

	def __init__(self, sj: int, beh: pd.DataFrame, epochs: mne.Epochs, 
				min_freq: int = 4, max_freq: int = 40, num_frex: int = 25,
				cycle_range: tuple = (3,10), freq_scaling: str = 'log',
				baseline: tuple = None, base_method: str = 'conspec',
				method: str = 'wavelet', downsample: int = 1, 
				laplacian: bool = True):
		"""
		_summary_

		Args:
			sj (int): 
			beh (pd.DataFrame): Dataframe with behavioral parameters per epoch 
			(see epochs)
			epochs (mne.Epochs): epoched eeg data (linked to beh)
			min_freq (int, optional): lower bound of frequency range
			used in TF analysis. Defaults to 4.
			max_freq (int, optional):upper bound of frequency range
			used in TF analysis. Can not be higher than Nyquist frequency. 
			Defaults to 40.
			num_frex (int, optional): Number of frequencies used in TF analysis
			(i.e., in between min and max freq). In general 20-30 frequencies 
			provide a reasnable number to cover a broad frequency range 
			(e.g., 4-60 Hz), while also making nice-looking plots. 
			Defaults to 25.
			cycle_range (tuple, optional): The number of cycles of the Gaussian
			taper define its width, and thus the width of the wavelet. This 
			parameter controls the trade-off between temporal and frequency 
			precision. Specifying a range (i.e., a tuple) makes sure that that 
			the cycles increase in the same number of steps as the frequency of
			the wavelets. Defaults to (3,10).
			freq_scaling (str, optional): specifies how frequencies are spaced.
			Supports logarithmic (log) and linear (lin) scaling. If main 
			results are expected in lower frequency bands logarithmic scale 
			is adviced, whereas linear scale is advised for expected results 
			in higher frequency bands. Defaults to 'log'.
			baseline (tuple, optional): Time window used for baselining using 
			Decibel Conversion. Defaults to no baseline correction.
			base_method (str, optional): specifies whether DB conversion is 
			condition specific ('cnd_spec') or averaged across conditions 
			('cnd_avg'). Defaults to condition specific baselining. 
			method (str, optional): _description_. Defaults to 'wavelet'.
			downsample (int, optional): factor used for downsampling 
			(aplied after filtering). Defaults to 1 (i.e., no downsampling)
			laplacian (bool, optional): _description_. Defaults to True.
		"""

		self.sj = sj
		self.beh = beh
		self.epochs = epochs
		self.min_freq = min_freq
		self.max_freq = max_freq
		self.method = method
		self.baseline = baseline
		self.base_method = base_method
		self.downsample = downsample
		self.num_frex = num_frex
		self.cycle_range = cycle_range
		self.freq_scaling = freq_scaling
		self.laplacian = laplacian
		# set params
		if self.method == 'wavelet':
			s_freq = epochs.info['sfreq']
			nr_time = epochs.times.size
			wavelets, frex = self.create_morlet(min_freq, max_freq, num_frex, 
												cycle_range, freq_scaling, 
												nr_time, s_freq)
			self.wavelets = wavelets
			self.frex = frex
		elif self.method == 'hilbert':
			print('Double check')

	def select_tf_data(self, excl_factor: dict, 
					   	topo_flip: dict) -> Tuple[pd.DataFrame, mne.Epochs]:

		# get local copy of data
		beh = self.beh.copy()
		epochs = self.epochs.copy()

		# if specified remove trials matching specified criteria
		if excl_factor is not None:
			beh, epochs = trial_exclusion(beh, epochs, excl_factor)

		# apply laplacian filter using mne defaults
		if self.laplacian:
			epochs = mne.preprocessing.compute_current_source_density(epochs)

		# check whether left stimuli should be 
		# artificially transferred to left hemifield
		if topo_flip is not None:
			(header, left), = topo_flip.items()
			epochs = ERP.flip_topography(epochs, beh,  left,  header)
		else:
			print('No topography info specified. It is assumed as if all '
				'stimuli of interest are presented right '
				'(i.e., left  hemifield')		
	
		return beh, epochs

	def create_morlet(self, min_freq: int, max_freq: int, num_frex: int, 
					  cycle_range: tuple, freq_scaling: str, nr_time:int, 
					  s_freq: float) -> Tuple[np.array, np.array]:
		"""
		Creates Morlet wavelets for TF decomposition (based on Ch 12, 13 of 
		Mike X Cohen, Analyzing neural time series data)

		Args:
			min_freq (int): lower bound of frequency range
			max_freq (int): upper bound of frequency range
			num_frex (int): number of frequencies in between lower and upper
			bound
			cycle_range (tuple): _description_
			freq_scaling (str): specifies how frequencies are spaced. 
			Supports logarithmic (log) and linear (lin) scaling. 
			nr_time (int): wavelets should be long enough such that the lowest 
			frequency wavelets tapers to zero. As a general rule, nr_time can 
			be equivalent to nr of timepoints in epoched data
			s_freq (float): sampling frequency in Hz

		Raises:
			ValueError: If incorrect frequency option is specified

		Returns:
			wavelets (np.array): Morlet wavelets per frequency (frequencies are
			stored within frex)
			frex (np.array): frequencies used in TF analysis
		"""

		# setup wavelet parameters (gaussian width and time)
		if freq_scaling == 'log':
			frex = np.logspace(np.log10(min_freq), np.log10(max_freq), 
							  num_frex)
			s = np.logspace(np.log10(cycle_range[0]), 
									np.log10(cycle_range[1]),num_frex) 
			s /= (2*math.pi*frex)
		elif freq_scaling == 'linear':
			frex = np.linspace(min_freq, max_freq, num_frex)
			s = np.linspace(cycle_range[0], cycle_range[1], num_frex) 
			s /= (2*math.pi*frex)
		else:
			raise ValueError('Unknown frequency scaling option')

		t = np.arange(-nr_time/s_freq/2, nr_time/s_freq/2, 1/s_freq)	
		 	
		# create wavelets
		wavelets = np.zeros((num_frex, len(t)), dtype = complex)
		for fi in range(num_frex):
			wavelets[fi,:] = (np.exp(2j * math.pi * frex[fi] * t) * 
							 np.exp(-t**2/(2*s[fi]**2)))
		
		return wavelets, frex	

	def wavelet_convolution(self, X, wavelet, l_conv, nr_time, nr_epochs):

		# new code
		m = ifft(X * fft(wavelet, l_conv), l_conv)
		m = m[:nr_time * nr_epochs + nr_time - 1]
		m = np.reshape(m[math.ceil((nr_time-1)/2 - 1):int(-(nr_time-1)/2-1)], 
									  (nr_time, -1), order = 'F').T 

		return m

	def tf_loop(self, epochs: mne.Epochs, picks: np.array) -> np.array:
		"""
		Generates time-frequency matrix per channel

		Args:
			epochs (mne.Epochs): Data used for time-frequency decomposition
			picks (np.array): indices of electrodes of interest

		Returns:
			raw_conv (np.array): Time frequency decomposition per trial
			(nr_epochs X nr_frequencies X nr_elec X nr_time)
		"""

		# initialize convolution array
		nr_time = epochs.times.size
		nr_epochs = len(epochs)
		l_conv = 2**self.nextpow2(nr_time * nr_epochs + nr_time - 1)
		raw_conv = np.zeros((nr_epochs, self.num_frex, picks.size, 
							nr_time), dtype = complex)

		# loop over channels					
		for i, pick in enumerate(picks):

			x = epochs._data[:, pick].ravel()	
			if self.method == 'wavelet':
				# fft decomposition
				x_fft = fft(x, l_conv)

			for f in range(self.num_frex):
				# convolve and get analytic signal
				if self.method == 'wavelet':
					m = self.wavelet_convolution(x_fft, self.wavelets[f], 
												l_conv, nr_time, nr_epochs)
				elif self.method == 'hilbert':
					print('method not yet implemented')
					pass

				raw_conv[:,f,i] = m
		
		return raw_conv

	def baseline_tf(self, tf: dict, base: dict, method: str, 
					elec_oi: str = 'all') -> dict:
		"""
		Apply baseline correction via decibel conversion

		Args:
			tf (dict): TF power per condition (epochs X nr_freq X nr_ch X 
			nr_time)
			base (dict): mean baseline TF power averaged across trials (nr_freq 
			X nr_chan)
			method (str): method for baseline correction
			elec_oi (str): Necessary when baselining depends on the topographic
			distribution of electrodes (i.e., when method is 'norm' or 'Z')

		Returns:
			tf (dict): normalized time frequency power
		"""

		tf_base = {}
		cnds = list(tf['power'].keys())
		if method == 'cnd_avg':
			cnd_avg = np.mean(np.stack([base[cnd] for cnd in cnds]), axis = 0)

		for cnd in cnds:
			power = tf['power'][cnd]
			if method == 'cnd_spec':	
				tf['power'][cnd] = self.db_convert(power, base[cnd])
			elif method == 'cnd_avg':
				tf['power'][cnd] = self.db_convert(power, cnd_avg)
			elif method == 'norm':
				print('For normalization procedure it is assumed that it is as'
				 	 ' if all stimuli of interest are presented right')
				tf['power'][cnd], info = self.normalize_power(power, elec_oi) 
				tf.update(dict(norm_info = info))
			
			# power values can now safely be averaged
			tf['power'][cnd] = np.mean(tf['power'][cnd], axis = 0)			

		return tf
				
	def db_convert(self, power: np.array, base_power: np.array) -> np.array:
		"""
		Decibel (dB) is the ratio between frequency-band specific power and 
		baseline level of power in that same frequency band. 

		Args:
			power (np.array): TF power (epochs X nr_freq X nr_ch X nr_time)
			base_power (np.array): baseline power (nr_freq X nr_chan)

		Returns:
			norm_power (np.array): baseline normalized power(dB)
		"""

		nr_time = power.shape[-1]
		base_power = np.repeat(base_power[...,np.newaxis],nr_time,axis = 2)
		norm_power = 10*np.log10(power/base_power)

		return power

	def normalize_power(self, power: np.array, elec_oi: list):

		# set params
		_, num_frex, _, nr_time = power.shape

		# ipsi_contra pairs
		contra_ipsi_pair = [('Fp1','Fp2'),('AF7','AF8'),('AF3','AF4'),
							('F7','F8'),('F5','F6'),('F3','F4'),
							('F1','F2'),('FT7','FT8'),('FC5','FC6'),
							('FC3','FC4'),('FC1','FC2'),('T7','T8'),
							('C5','C6'),('C3','C4'),('C1','C2'),('TP7','TP8'),
							('CP5','CP6'),('CP3','CP4'),('CP1','CP2'),
							('P9','P10'),('P7','P8'),('P5','P6'),('P3','P4'),
							('P1','P2'),('PO7','PO8'),('PO3','PO4'),
							('O1','O2')]

		pair_idx =  [i for i, pair in enumerate(contra_ipsi_pair) 
					if pair[0] in elec_oi]			
		elec_pairs = np.array(contra_ipsi_pair)[pair_idx] 

		# initiate array
		norm = np.zeros((elec_pairs.shape[0], num_frex, nr_time))
		norm_elec = []

		# loop over contra_ipsi pairs
		for i, (contra, ipsi) in enumerate(elec_pairs):
			norm_elec.append((contra, ipsi))	
			# get indices of electrode pair
			contra_idx = elec_oi.index(contra)
			ipsi_idx = elec_oi.index(ipsi)
			subtr = power[:,:,contra_idx] - power[:,:,ipsi_idx]
			add = power[:,:,contra_idx] + power[:,:,ipsi_idx]
			norm[i] = np.mean(subtr/add, axis = 0)	

		return norm, norm_elec

	def lateralized_tf(self,pos_labels, cnds: dict = None, 
					   elec_oi: list = 'all',midline: dict = None, 
					   topo_flip: dict = None, time_oi: tuple = None, 
					   excl_factor: dict = None, name : str = 'main'):
		
		# get data
		beh, epochs = self.select_tf_data(excl_factor, topo_flip)

		# limit analysis to electrodes of interest
		if elec_oi == 'all':
			picks = mne.pick_types(epochs.info, eeg=True, csd = True)
			elec_oi = np.array(epochs.ch_names)[picks]
		else:
			picks = mne.pick_channels(epochs.ch_names, elec_oi)
		
		# select trials of interest (i.e., lateralized stimuli)
		idx = ERP.select_lateralization_idx(beh, pos_labels, midline)

		# get baseline index
		times = epochs.times
		nr_time = times.size
		if type(self.baseline) is not None:
			base = {}
			s, e = self.baseline
			base_idx = get_time_slice(times, s, e)
		
		time_idx = np.where((times >= time_oi[0]) * (times <= time_oi[1]))[0] 
		idx_2_save = np.array([idx for i, idx in enumerate(time_idx) if 
							  i % self.downsample == 0]) 

		# initiate output dicts and loop over conditions
		tf = {'ch_names':np.array(epochs.ch_names)[picks], 
		 	  'times':times[idx_2_save], 
			  'frex': self.frex, 'power': {}}
		if cnds is None:
			cnds = ['all_trials']
		else:
			(cnd_header, cnds), = cnds.items()

		for cnd in cnds:
			# set tf name
			tf_name = f'sj_{self.sj}_{name}'

			# slice condition trials
			if cnd == 'all_trials':
				idx_c = idx
			else:
				idx_c = np.where(beh[cnd_header] == cnd)[0]
				idx_c = np.intersect1d(idx, idx_c)

			# TF decomposition 
			raw_conv = self.tf_loop(epochs, picks)
			# get baseline power (correction is done after condition loop)
			if type(self.baseline) is not None:
				base[cnd] = np.mean(abs(raw_conv[...,base_idx])**2, 
									axis = (0,3))
			# populate tf
			tf['power'][cnd] = abs(raw_conv[..., idx_2_save])**2

		# baseline correction
		tf = self.baseline_tf(tf, base, self.base_method, elec_oi)
		
		# save output
		with open(self.FolderTracker(['tf',self.method],
				f'{tf_name}.pickle') ,'wb') as handle:
			pickle.dump(tf, handle)	
				
	def TFanalysis(self, sj, cnds, cnd_header, time_period, tf_name, base_period = None, elec_oi = 'all',factor = None, method = 'hilbert', flip = None, base_type = 'conspec', downsample = 1, min_freq = 5, max_freq = 40, num_frex = 25, cycle_range = (3,12), freq_scaling = 'log'):
		'''
		Time frequency analysis using either morlet waveforms or filter-hilbert method for time frequency decomposition

		Add option to subtract ERP to get evoked power
		Add option to match trial number

		Arguments
		- - - - - 
		sj (int): subject number
		cnds (list): list of conditions as stored in behavior file
		cnd_header (str): key in behavior file that contains condition info
		base_period (tuple | list): time window used for baseline correction. 
		time_period (tuple | list): time window of interest
		tf_name (str): name of analysis. Used to create unique file location
		elec_oi (str | list): If not all, analysis are limited to specified electrodes 
		factor (dict): limit analysis to a subset of trials. Key(s) specifies column header
		method (str): specifies whether hilbert or wavelet convolution is used for time-frequency decomposition
		flip (dict): flips a subset of trials. Key of dictionary specifies header in beh that contains flip info 
		List in dict contains variables that need to be flipped. Note: flipping is done from right to left hemifield
		base_type (str): specifies whether DB conversion is condition specific ('conspec') or averaged across conditions ('conavg').
						If Z power is Z-transformed (condition specific). 
		downsample (int): factor used for downsampling (aplied after filtering). Default is no downsampling
		min_freq (int): minimum frequency for TF analysis
		max_freq (int): maximum frequency for TF analysis
		num_frex (int): number of frequencies in TF analysis
		cycle_range (tuple): number of cycles increases in the same number of steps used for scaling
		freq_scaling (str): specify whether frequencies are linearly or logarithmically spaced. 
							If main results are expected in lower frequency bands logarithmic scale 
							is adviced, whereas linear scale is advised for expected results in higher
							frequency bands
		Returns
		- - - 
		
		wavelets(array): 


	
		'''

		# read in data
		eegs, beh = self.selectTFData(self.laplacian, factor)
		times = self.EEG.times
		if elec_oi == 'all':
			picks = mne.pick_types(self.EEG.info, eeg=True, exclude='bads')
			ch_names = list(np.array(self.EEG.ch_names)[picks])
		else:
			ch_names = elec_oi	

		# flip subset of trials (allows for lateralization indices)
		if flip != None:
			key = list(flip.keys())[0]
			eegs = self.topoFlip(eegs, beh[key], self.EEG.ch_names, left = flip.get(key))

		# get parameters
		nr_time = eegs.shape[-1]
		nr_chan = eegs.shape[1] if elec_oi == 'all' else len(elec_oi)
		if method == 'wavelet':
			wavelets, frex = self.createMorlet(min_freq = min_freq, max_freq = max_freq, num_frex = num_frex, 
									cycle_range = cycle_range, freq_scaling = freq_scaling, 
									nr_time = nr_time, s_freq = self.EEG.info['sfreq'])
		
		elif method == 'hilbert':
			frex = [(i,i + 4) for i in range(min_freq, max_freq, 2)]
			num_frex = len(frex)	

		if type(base_period) in [tuple,list]:
			base_s, base_e = [np.argmin(abs(times - b)) for b in base_period]
		idx_time = np.where((times >= time_period[0]) * (times <= time_period[1]))[0]  
		idx_2_save = np.array([idx for i, idx in enumerate(idx_time) if i % downsample == 0])

		# initiate dicts
		tf = {'ch_names':ch_names, 'times':times[idx_2_save], 'frex': frex}
		tf_base = {'ch_names':ch_names, 'times':times[idx_2_save], 'frex': frex}
		base = {}
		plot_dict = {}

		# loop over conditions
		for c, cnd in enumerate(cnds):
			print(cnd)
			tf.update({cnd: {}})
			tf_base.update({cnd: {}})
			base.update({cnd: np.zeros((num_frex, nr_chan))})

			if cnd != 'all':
				cnd_idx = np.where(beh[cnd_header] == cnd)[0]
			else:
				cnd_idx = np.arange(beh[cnd_header].size)	

			l_conv = 2**self.nextpow2(nr_time * cnd_idx.size + nr_time - 1)
			raw_conv = np.zeros((cnd_idx.size, num_frex, nr_chan, idx_2_save.size), dtype = complex) 

			# loop over channels
			for idx, ch in enumerate(ch_names[:nr_chan]):
				# find ch_idx
				ch_idx = self.EEG.ch_names.index(ch)

				print('Decomposed {0:.0f}% of channels ({1} out {2} conditions)'.format((float(idx)/nr_chan)*100, c + 1, len(cnds)), end='\r')

				# fft decomposition
				if method == 'wavelet':
					eeg_fft = fft(eegs[cnd_idx,ch_idx].ravel(), l_conv)    # eeg is concatenation of trials after ravel

				# loop over frequencies
				for f in range(num_frex):

					if method == 'wavelet':
						# convolve and get analytic signal (OUTPUT DIFFERS SLIGHTLY FROM MATLAB!!! DOUBLE CHECK)
						m = ifft(eeg_fft * fft(wavelets[f], l_conv), l_conv)
						m = m[:nr_time * cnd_idx.size + nr_time - 1]
						m = np.reshape(m[math.ceil((nr_time-1)/2 - 1):int(-(nr_time-1)/2-1)], 
									  (nr_time, -1), order = 'F').T 

					elif method == 'hilbert': # NEEDS EXTRA CHECK
						X = eegs[cnd_idx,ch_idx].ravel()
						m = self.hilbertMethod(X, frex[f][0], frex[f][1], s_freq)
						m = np.reshape(m, (-1, times.size))	

					# populate
					raw_conv[:,f,idx] = m[:,idx_2_save]
					
					# baseline correction (actual correction is done after condition loop)
					if type(base_period) in [tuple,list]:
						base[cnd][f,idx] = np.mean(abs(m[:,base_s:base_e])**2)

			# update cnd dict with phase values (averaged across trials) and power values
			tf[cnd]['power'] = abs(raw_conv)**2
			tf[cnd]['phase'] = abs(np.mean(np.exp(np.angle(raw_conv) * 1j), axis = 0))

		# baseline normalization
		for cnd in cnds:
			if base_type == 'conspec': #db convert: condition specific baseline
				tf_base[cnd]['base_power'] = 10*np.log10(tf[cnd]['power']/np.repeat(base[cnd][:,:,np.newaxis],idx_2_save.size,axis = 2))
			elif base_type == 'conavg':	
				con_avg = np.mean(np.stack([base[cnd] for cnd in cnds]), axis = 0)
				tf_base[cnd]['base_power'] = 10*np.log10(tf[cnd]['power']/np.repeat(con_avg[:,:,np.newaxis],idx_2_save.size,axis = 2))
			elif base_type == 'Z':
				print('For permutation procedure it is assumed that it is as if all stimuli of interest are presented right')
				tf_base[cnd]['Z_power'], z_info = self.permuted_Z(tf[cnd]['power'],ch_names, num_frex, idx_2_save.size) 
				tf_base.update(dict(z_info = z_info))
			elif base_type == 'norm':
				print('For normalization procedure it is assumed that it is as if all stimuli of interest are presented right')
				tf_base[cnd]['norm_power'], norm_info = self.normalizePower(tf[cnd]['power'],ch_names, num_frex, idx_2_save.size) 
				tf_base.update(dict(norm_info = norm_info))
			if base_type in ['conspec','conavg']:
				tf[cnd]['base_power'] = np.mean(tf_base[cnd]['base_power'], axis = 0)

			# power values can now safely be averaged
			tf[cnd]['power'] = np.mean(tf[cnd]['power'], axis = 0)

		# save TF matrices
		with open(self.FolderTracker(['tf',method,tf_name],'{}-tf.pickle'.format(sj)) ,'wb') as handle:
			pickle.dump(tf, handle)		

		with open(self.FolderTracker(['tf',method,tf_name],'{}-tf_base.pickle'.format(sj)) ,'wb') as handle:
			pickle.dump(tf_base, handle)	
	
		


	@staticmethod	
	def nextpow2(i):
		'''
		Gives the exponent of the next higher power of 2
		'''

		n = 1
		while 2**n < i: 
			n += 1
		
		return n




	def hilbertMethod(self, X, l_freq, h_freq, s_freq = 512):
		'''
		Apply filter-Hilbert method for time-frequency decomposition. 
		Data is bandpass filtered before a Hilbert transform is applied

		Arguments
		- - - - - 
		X (array): eeg signal
		l_freq (int): lower border of frequency band
		h_freq (int): upper border of frequency band
		s_freq (int): sampling frequency
		
		Returns
		- - - 

		X (array): filtered eeg signal
		
		'''	

		X = hilbert(filter_data(X, s_freq, l_freq, h_freq))

		return X

	def TFanalysisMNE(self, sj, cnds, cnd_header, base_period, time_period, method = 'hilbert', flip = None, base_type = 'conspec', downsample = 1, min_freq = 5, max_freq = 40, num_frex = 25, cycle_range = (3,12), freq_scaling = 'log'):
		'''
		Time frequency analysis using either morlet waveforms or filter-hilbertmethod for time frequency decomposition

		Add option to subtract ERP to get evoked power
		Add option to match trial number

		Arguments
		- - - - - 
		sj (int): subject number
		cnds (list): list of conditions as stored in behavior file
		cnd_header (str): key in behavior file that contains condition info
		base_period (tuple | list): time window used for baseline correction
		time_period (tuple | list): time window of interest
		method (str): specifies whether hilbert or wavelet convolution is used for time-frequency decomposition
		flip (dict): flips a subset of trials. Key of dictionary specifies header in beh that contains flip info 
		List in dict contains variables that need to be flipped. Note: flipping is done from right to left hemifield
		base_type (str): specifies whether DB conversion is condition specific ('conspec') or averaged across conditions ('conavg')
		downsample (int): factor used for downsampling (aplied after filtering). Default is no downsampling
		min_freq (int): minimum frequency for TF analysis
		max_freq (int): maximum frequency for TF analysis
		num_frex (int): number of frequencies in TF analysis
		cycle_range (tuple): number of cycles increases in the same number of steps used for scaling
		freq_scaling (str): specify whether frequencies are linearly or logarithmically spaced. 
							If main results are expected in lower frequency bands logarithmic scale 
							is adviced, whereas linear scale is advised for expected results in higher
							frequency bands
		Returns
		- - - 
		
		wavelets(array): 


	
		'''

		# read in data
		eegs, beh, times, s_freq, ch_names = self.selectTFData(sj)

		# flip subset of trials (allows for lateralization indices)
		if flip != None:
			key = flip.keys()[0]
			eegs = self.topoFlip(eegs, beh[key], ch_names, left = [flip.get(key)])

		# get parameters
		nr_time = eegs.shape[-1]
		nr_chan = eegs.shape[1]
		
		freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frex)
		nr_cycles = np.logspace(np.log10(cycle_range[0]), np.log10(cycle_range[1]),num_frex)

		base_s, base_e = [np.argmin(abs(times - b)) for b in base_period]
		idx_time = np.where((times >= time_period[0]) * (times <= time_period[1]))[0]  
		idx_2_save = np.array([idx for i, idx in enumerate(idx_time) if i % downsample == 0])

		# initiate dict
		tf = {}
		base = {}

		# loop over conditions
		for c, cnd in enumerate(cnds):
			tf.update({cnd: {}})
			base.update({cnd: np.zeros((num_frex, nr_chan))})

			cnd_idx = np.where(beh['block_type'] == cnd)[0]

			power = tfr_array_morlet(eegs[cnd_idx], sfreq= s_freq,
					freqs=freqs, n_cycles=nr_cycles,
					output='avg_power')		

			# update cnd dict with power values
			tf[cnd]['power'] = np.swapaxes(power, 0,1)
			tf[cnd]['base_power'] = rescale(np.swapaxes(power, 0,1), times, base_period, mode = 'logratio')
			tf[cnd]['phase'] = '?'

		# save TF matrices
		with open(self.FolderTracker(['tf',method],'{}-tf-mne.pickle'.format(sj)) ,'wb') as handle:
			pickle.dump(tf, handle)		

		# store dictionary with variables for plotting
		plot_dict = {'ch_names': ch_names, 'times':times[idx_2_save], 'frex': freqs}

		with open(self.FolderTracker(['tf', method], filename = 'plot_dict.pickle'),'wb') as handle:
			pickle.dump(plot_dict, handle)




	def permuted_Z(self, raw_power, ch_names, num_frex, nr_time, nr_perm = 1000):


		
		# ipsi_contra pairs
		contra_ipsi_pair = [('Fp1','Fp2'),('AF7','AF8'),('AF3','AF4'),('F7','F8'),('F5','F6'),('F3','F4'),\
					('F1','F2'),('FT7','FT8'),('FC5','FC6'),('FC3','FC4'),('FC1','FC2'),('T7','T8'),\
					('C5','C6'),('C3','C4'),('C1','C2'),('TP7','TP8'),('CP5','CP6'),('CP3','CP4'),\
					('CP1','CP2'),('P9','P10'),('P7','P8'),('P5','P6'),('P3','P4'),('P1','P2'),\
					('PO7','PO8'),('PO3','PO4'),('O1','O2')]
		pair_idx =  [i for i, pair in enumerate(contra_ipsi_pair) if pair[0] in ch_names]			
		contra_ipsi_pair = np.array(contra_ipsi_pair)[pair_idx] 

		# initiate Z array
		Z = np.zeros((contra_ipsi_pair.shape[0], num_frex, nr_time))
		Z_elec = []
		# loop over contra_ipsi pairs
		pair_idx = 0

		for (contra_elec, ipsi_elec) in contra_ipsi_pair:
			
			Z_elec.append(contra_elec)	
			
			# get indices of electrode pair
			contra_idx = ch_names.index(contra_elec)
			ipsi_idx = ch_names.index(ipsi_elec)
			# contra_ipsi_norm = (raw_power[:,:,contra_idx] - raw_power[:,:,ipsi_idx])/(raw_power[:,:,contra_idx] + raw_power[:,:,ipsi_idx])
			# norm[pair_idx] = contra_ipsi_norm.mean(axis = 0)

			# # get the real difference
			real_diff = raw_power[:,:,contra_idx] - raw_power[:,:,ipsi_idx]

			# create distribution of fake differences
			fake_diff = np.zeros((nr_perm,) + real_diff.shape)

			for p in range(nr_perm):
			 	# randomly flip ipsi and contra
				signed = np.sign(np.random.normal(size = real_diff.shape[0]))
				permuter = np.tile(signed[:,np.newaxis, np.newaxis], ((1,) + real_diff.shape[-2:]))
				fake_diff[p] = real_diff * permuter

			# # Z scoring
			print('Z scoring pair {}-{}'.format(contra_elec, ipsi_elec))
			Z[pair_idx] = (np.mean(real_diff, axis = 0) - np.mean(fake_diff, axis = (0,1))) / np.std(np.mean(fake_diff, axis = 1), axis = 0, ddof = 1)	
			pair_idx += 1

		return Z, Z_elec




 	