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


class TFR(FolderStructure):
	"""
	The TFR class supports functionality for time-frequency 
	decomposition (TFR) analysis on EEG data.
	It provides methods for ....
    TODO: UPDATE DOCSTRING!!!!!

    This class inherits from FolderStructure, which provides 
    functionality for managing file paths and saving outputs. 

	Args:
        sj (int): Subject identifier.
        df (pd.DataFrame): Behavioral data associated with the 
            EEG epochs.
        epochs (mne.Epochs): Preprocessed EEG data segmented 
            into epochs.
        min_freq (int, optional): Minimum frequency (in Hz) for the 
            time-frequency analysis. Defaults to 4 Hz.
        max_freq (int, optional): Maximum frequency (in Hz) for the 
            time-frequency analysis. Can not be higher than Nyquist 
			frequency.Defaults to 40 Hz.
        num_frex (int, optional): Number of frequencies to analyze 
            between `min_freq` and `max_freq`. In general 20-30 
			frequencies provide a reasonable number to cover a broad 
			frequency range (e.g., 4-60 Hz), while also generating 
			nice-looking plots. Defaults to 25.
        cycle_range (tuple, optional): Range of cycles to use for the 
            wavelet analysis. The tuple specifies the minimum and 
            maximum number of cycles. The number of cycles of the 
			Gaussiantaper define its width, and thus the width of the 
			wavelet. This parameter controls the trade-off between 
			temporal and frequency precision. Specifying a range 
			(i.e., a tuple) makes sure that that the cycles increase in 
			the same number of steps as the frequency of the wavelets. 
			Defaults to (3,10).
        freq_scaling (str, optional): Scaling method for the frequency 
            axis. Supported values are `'log'` (logarithmic scaling) 
            and `'linear'`. If main results are expected in lower 
			frequency bands logarithmic scale is adviced, whereas linear 
			scale is advised for expected results in higher frequency 
			bands. Defaults to 'log'.Defaults to `'log'`.
        baseline (tuple, optional): Time range (start, end) in seconds 
            for baseline correction. If None, no baseline correction is 
            applied. Defaults to None.
        base_method (str, optional): Method for baseline correction. 
			Specifies whether DB conversion is condition specific 
			('cnd_spec') or averaged across conditions ('cnd_avg'). 
			Defaults to condition specific baselining. 
        method (str, optional): Method for time-frequency analysis. 
            Supported values are `'wavelet'` and `'morlet'`. 
            Defaults to `'wavelet'`.
            TODO: Check MORLET functionality
        downsample (int, optional): Factor by which to downsample the
            time-frequency data. Defaults to 1 (no downsampling).
        laplacian (bool, optional): If True, applies a Laplacian spatial 
            filter to the EEG data before time-frequency analysis. 
            Defaults to False.

	Attributes:
        sj (int): Subject identifier.
        df (pd.DataFrame): Behavioral data associated with the 
			EEG epochs.
        epochs (mne.Epochs): Preprocessed EEG data segmented into 
			epochs.
        min_freq (int): Minimum frequency (in Hz) for the time-frequency 
			analysis.
        max_freq (int): Maximum frequency (in Hz) for the time-frequency 
			analysis.
        num_frex (int): Number of frequencies to analyze between
		 	`min_freq` and `max_freq`.
        cycle_range (tuple): Range of cycles to use for the wavelet 
			analysis.
        freq_scaling (str): Scaling method for the frequency axis 
			(`'log'` or `'linear'`).
        baseline (tuple): Time range (start, end) in seconds for 
			baseline correction.
        base_method (str): Method for baseline correction 
			(condition specific or condition averaged).
        method (str): Method for time-frequency analysis
			(`'wavelet'` or `'hilbert'`).
        downsample (int): Factor by which to downsample the 	
			time-frequency data.
        laplacian (bool): Indicates whether a Laplacian spatial filter 
			is applied.
        wavelets (np.ndarray): Morlet wavelets used for time-frequency 
			analysis (if `method='wavelet'`).
        frex (np.ndarray): Array of frequencies corresponding to the 
			wavelets (if `method='wavelet'`).

    Returns:
        None: This is the constructor method and does not 
        return a value.

	"""

	def __init__(
		self, 
		sj: int, 
		epochs: mne.Epochs, 
		df: pd.DataFrame, 
		min_freq: int = 4, 
		max_freq: int = 40, 
		num_frex: int = 25, 
		cycle_range: tuple = (3, 10), 
		freq_scaling: str = 'log', 
		baseline: tuple = None, 
		base_method: str = 'cnd_spec', 
		method: str = 'wavelet', 
		downsample: int = 1, 
		laplacian: bool = False,
		report: bool = False
	):
		"""class constructor"""

		self.sj = sj
		self.df = df
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
		self.report = report
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

	def select_tfr_data(
		self, 
		elec_oi: Union[str, list], 
		excl_factor: dict = None, 
		topo_flip: dict = None
	) -> Tuple[mne.Epochs, pd.DataFrame]:
		"""
		Selects time-frequency (TF) data and applies initial data 
		transformation steps.

		This function performs the following optional preprocessing 
		steps in order:
			1. Excludes trials based on behavioral data.
			2. Applies a Laplacian spatial filter to the EEG data.
			3. Flips the topography of trials based on 
				specified criteria.
			4. Selects electrodes of interest for further analysis.

		Args:
			elec_oi (Union[str, list]): Electrodes of interest. 
				Can be specified as:
					- A list of electrode names (e.g., `['O1', 'O2']`).
					- A string referring to a subset of electrodes 
						(e.g., `'all'`, `'frontal'`).
			excl_factor (dict, optional): Dictionary specifying 
				behavioral factors to exclude.Keys are column names in 
				the behavioral DataFrame, and values are lists of 
				values to exclude. For example, `{'cnd': [1, 2]}` 
				excludes trials where the condition is 1 or 2. 
				Defaults to None.
			topo_flip (dict, optional): Dictionary specifying 
				criteria for flipping the topography of trials. 
				Keys are column names in the behavioral DataFrame, and 
				values specify the conditions to flip. For example, 
				`{'cnd': 'left'}` flips the topography of all trials 
				where the condition is `'left'`. Defaults to None.

		Returns:
			Tuple[mne.Epochs, pd.DataFrame]: 
				- `epochs`: Epoched EEG data after applying the 
					specified transformations.
				- `df`: Behavioral DataFrame with parameters 
					corresponding to each epoch.
		"""	
		
		# get local copy of data
		df = self.df.copy()
		epochs = self.epochs.copy()

		# if specified remove trials matching specified criteria
		if excl_factor is not None:
			df, epochs, _ = trial_exclusion(df, epochs, excl_factor)

		# apply laplacian filter using mne defaults
		if self.laplacian:
			epochs = mne.preprocessing.compute_current_source_density(epochs)

		# check whether left stimuli should be 
		# artificially transferred to left hemifield
		if topo_flip is not None:
			(header, left), = topo_flip.items()
			epochs = ERP.flip_topography(epochs, df,  left,  header)
		else:
			print('No topography info specified. For lateralization analysis,'
	 			' it is assumed as if all stimuli of interest are presented '
				'right (i.e., left  hemifield')		
			
		# limit analysis to electrodes of interest
		if elec_oi == 'all':
			picks = mne.pick_types(epochs.info, eeg=True, csd = True)
			elec_oi = np.array(epochs.ch_names)[picks]
		epochs.pick_channels(elec_oi)		
	
		return epochs, df
	
	def create_morlet(
		self, 
		min_freq: int, 
		max_freq: int, 
		num_frex: int, 
		cycle_range: tuple, 
		freq_scaling: str, 
		nr_time: int, 
		s_freq: float
	) -> Tuple[np.array, np.array]:
		"""
		Creates Morlet wavelets for time-frequency (TF) decomposition.
		(based on Ch 12, 13 of Mike X Cohen, Analyzing neural time 
		series data)

		This function generates a set of Morlet wavelets for 
		time-frequency analysis based on the specified frequency range, 
		scaling method, and sampling frequency. 
		The wavelets are constructed to ensure that the lowest frequency 
		wavelet tapers to zero, making them suitable 
		for TF decomposition.

		Args:
			min_freq (int): Lower bound of the frequency range (in Hz).
			max_freq (int): Upper bound of the frequency range (in Hz).
			num_frex (int): Number of frequencies to generate between 
				`min_freq` and `max_freq`.
			cycle_range (tuple): Range of cycles for the wavelets. The 
				tuple specifies the minimum and maximum number of cycles 
				(e.g., `(3, 10)`).
			freq_scaling (str): Specifies how frequencies are spaced. 
				Supported values are:
					- `'log'`: Logarithmic scaling.
					- `'linear'`: Linear scaling.
			nr_time (int): Number of time points for the wavelets. This 
				should be long enough to ensure that the lowest 
				frequency wavelet tapers to zero. Typically, this 
				can be equivalent to the number of time points in the 
				epoched data.
			s_freq (float): Sampling frequency (in Hz).

		Raises:
			ValueError: If an unsupported frequency scaling option is 
			specified.

		Returns:
			Tuple[np.array, np.array]: 
				- `wavelets` (np.array): Complex Morlet wavelets for 
					each frequency. The shape is `(num_frex, nr_time)`.
				- `frex` (np.array): Array of frequencies corresponding 
					to the wavelets.
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
	
	def wavelet_convolution(
			self, 
			X: np.ndarray, 
			wavelet: np.ndarray, 
			l_conv: int, 
			nr_time: int, 
			nr_epochs: int
		) -> np.ndarray:
		"""
		Performs wavelet convolution for time-frequency analysis.

		This function convolves the input data (`X`) with a Morlet 
		wavelet in the frequency domain using the Fast Fourier Transform 
		(FFT). The convolution is performed efficiently by leveraging 
		the FFT and Inverse FFT (IFFT), and the result is reshaped to 
		match the time-frequency structure of the input data.

		Args:
			X (np.ndarray): Input data in the frequency domain 
				(FFT of the signal).
			wavelet (np.ndarray): The Morlet wavelet to convolve with 
				the input data.
			l_conv (int): Length of the convolution (in samples).
			nr_time (int): Number of time points in the output.
			nr_epochs (int): Number of epochs in the input data.

		Returns:
			np.ndarray: The result of the wavelet convolution, 
				reshaped into a 2D array of shape `(nr_epochs, nr_time)`, 
				where each row corresponds to an epoch and each 
				column corresponds to a time point.
		"""


		# new code
		m = ifft(X * fft(wavelet, l_conv), l_conv)
		m = m[:nr_time * nr_epochs + nr_time - 1]
		m = np.reshape(m[math.ceil((nr_time-1)/2 - 1):int(-(nr_time-1)/2-1)], 
									  (nr_time, -1), order = 'F').T 

		return m

	def generate_tfr_report(self,tfr:dict,info:mne.Info,
						 report_name: str):
		#TODO add docstring

		report_name = self.folder_tracker(['tfr', 'report'],
										f'{report_name}.h5')
        
		report = mne.Report(title='Single subject tfr overview')
		for cnd in tfr['power'].keys():
			# select a subset of frequencies for topo plots
			freqs = tfr['frex']
			if len(freqs)>3:
				freqs_oi = np.linspace(freqs[0], freqs[-1], 3).astype(int)
			else:
				freqs_oi = freqs
			# get time points for topo plots
			idx = int(np.argmax(tfr['power'][cnd], axis=2).mean())
			time_oi = tfr['times'][idx]
			time_freqs = tuple(((time_oi,f) for f in freqs_oi))

			# create mne object for visualization
			x = tfr['power'][cnd]

			# change data into mne format (..., n_ch, n_freq, n_time)
			x = np.swapaxes(x, 0, 1) if x.ndim == 3 else np.swapaxes(x, 1, 2)
				
			tfr_ = mne.time_frequency.AverageTFR(info,x,tfr['times'],self.frex,
				       						tfr['cnd_cnt'][cnd], 
											method = self.method,
											comment = cnd)

			section = "Condition: " + cnd	

			#TODO: add section after update mne (so that cnd info is displayed)
			report.add_figure(tfr_.plot_joint(timefreqs = time_freqs),
						title = '2D TFR & Topos (collapsed over electrodes)')

			report.add_figure(fig=tfr_.plot(), 
						title="Individual electrodes", 
						caption=tfr['ch_names'])
							
		report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html', overwrite=True)

	def lateralized_tfr(
		self, 
		pos_labels: dict, 
		cnds: dict = None, 
		elec_oi: list = 'all', 
		midline: dict = None, 
		topo_flip: dict = None, 
		window_oi: tuple = None, 
		excl_factor: dict = None, 
		name: str = 'main'
	):
		"""
		Performs lateralized time-frequency (TFR) decomposition for 
		specified conditions.

		This function computes lateralized time-frequency 
		representations (TFRs) for specified conditions and electrodes 
		of interest. It supports baseline correction, downsampling, and 
		condition-specific trial selection. The output is saved in a 
		format compatible with MNE-Python.

		Args:
			pos_labels (dict): Dictionary specifying the position labels 
				for lateralized stimuli. For example, 
				`{'target_loc': [2, 6]}` specifies the column name 
				(`'target_loc'`) and the values corresponding to 
				left and right hemifield stimuli.
			cnds (dict, optional): Dictionary specifying conditions for 
				TFR decomposition. The key should be the column name in 
				the behavioral data, and the value should be a list of 
				condition labels. For example, `{'cnd': ['A', 'B']}` 
				processes conditions `'A'` and `'B'`. If None, all 
				trials are processed. Defaults to None.
			elec_oi (list, optional): Electrodes of interest. Can be 
				`'all'` to include all electrodes or a list of electrode 
				names (e.g., `['O1', 'O2']`). Defaults to `'all'`.
			midline (dict, optional): Dictionary specifying trials where 
				another stimulus of interest is presented on the 
				vertical midline. The key should be the column name, and 
				the value should be a list of labels. Defaults to None.
			topo_flip (dict, optional): Dictionary specifying criteria 
				for flipping the topography of certain trials. The key 
				should be the column name in the behavioral data, and 
				the value should be a list of labels indicating trials 
				to flip. Defaults to None.
			window_oi (tuple, optional): Time window of interest 
				(start, end) in seconds. If specified, the TFR is 
				cropped to this window. Defaults to None.
			excl_factor (dict, optional): Dictionary specifying criteria 
				for excluding trials from the analysis. For example, 
				`{'cnd': ['exclude']}` excludes all trials where the 
				condition is `'exclude'`. Defaults to None.
			name (str, optional): Name used for saving the TFR output. 
				Defaults to `'main'`.

		Returns:
			None: The function saves the computed TFR to disk in
				MNE-compatible format.
		"""
		
		# get data
		epochs, df = self.select_tfr_data(elec_oi,excl_factor,topo_flip)
		
		# select trials of interest (i.e., lateralized stimuli)
		idx = ERP.select_lateralization_idx(df, pos_labels, midline)

		# get baseline index
		times = epochs.times
		nr_time = times.size
		if self.baseline is not None:
			base = {}
			s, e = self.baseline
			base_idx = get_time_slice(times, s, e)
		
		time_idx = np.where((times >= window_oi[0]) * 
					  		(times <= window_oi[1]))[0] 
		idx_2_save = np.array([idx for i, idx in enumerate(time_idx) if 
							  i % self.downsample == 0]) 

		# initiate output dicts and loop over conditions
		tfr = {'ch_names':np.array(epochs.ch_names), 
		 	  'times':times[idx_2_save], 
			  'frex': self.frex, 'power': {},
			  'cnd_cnt':{}}
		
		if cnds is None:
			cnds = ['all_data']
		else:
			(cnd_header, cnds), = cnds.items()

		for c, cnd in enumerate(cnds):
			counter = c + 1 
			print(f'Decomposing condition {counter}: {cnd}')
			# set tfr name
			tfr_name = f'sj_{self.sj}_{name}'

			# slice condition trials
			if cnd == 'all_data':
				idx_c = idx
			else:
				idx_c = np.where(df[cnd_header] == cnd)[0]
				idx_c = np.intersect1d(idx, idx_c)

			# TF decomposition 
			raw_conv = self.tfr_loop(epochs[idx_c])

			# get baseline power (correction is done after condition loop)
			if self.baseline is not None:
				#TODO: speed up
				base[cnd] = np.mean(abs(raw_conv[...,base_idx])**2, 
									axis = (0,3))
			else:
				base = {}
			# populate tf
			tfr['power'][cnd] = abs(raw_conv[..., idx_2_save])**2

		# baseline correction
		tfr = self.baseline_tfr(tfr,base,self.base_method,elec_oi)

		if self.report:
			self.generate_tfr_report(tfr,
							epochs.info,f'sj_{self.sj}_{name}')

		# save output
		self.save_to_mne_format(tfr,epochs,tfr_name)

	def tfr_loop(self, epochs: mne.Epochs) -> np.array:
		"""
		TODO: implement hilbert convolution
		Generates a time-frequency (TF) matrix for each channel.

		This function performs time-frequency decomposition for each 
		channel in the provided EEG epochs. It supports wavelet-based 
		decomposition and computes the analytic signal for each 
		frequency and channel. The resulting TF matrix contains the 
		time-frequency representation for each trial, frequency, 
		channel, and time point.

		Args:
			epochs (mne.Epochs): Preprocessed EEG data segmented into 
				epochs. The data is used for time-frequency 
				decomposition.

		Returns:
			np.array: A 4D NumPy array containing the time-frequency 
			decomposition per trial. The shape of the array is 
			`(nr_epochs, nr_frequencies, nr_channels, nr_time)`, where:
				- `nr_epochs`: Number of trials in the input data.
				- `nr_frequencies`: Number of frequencies used for 
					decomposition.
				- `nr_channels`: Number of EEG channels.
				- `nr_time`: Number of time points in each trial.
		"""

		# initialize convolution array
		nr_time = epochs.times.size
		nr_ch = len(epochs.ch_names)
		nr_epochs = len(epochs)
		l_conv = 2**self.nextpow2(nr_time * nr_epochs + nr_time - 1)
		raw_conv = np.zeros((nr_epochs, self.num_frex, nr_ch, 
							nr_time), dtype = complex)

		# loop over channels			
		for ch_idx in range(nr_ch):
			print(f'Decomposing channel {ch_idx+1} out of {nr_ch} channels', 
				  end='\r')

			x = epochs._data[:, ch_idx].ravel()	
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

				raw_conv[:,f,ch_idx] = m
		
		return raw_conv
	
	def save_to_mne_format(self,tfr:dict,epochs:mne.Epochs,
						tfr_name:str):
		"""
		convert tfr data to mne container for time-frequency data
		(i.e, epochsTFR or AvereageTFR)

		Args:
			tfr (dict): dictionary with tfr data
			epochs (mne.Epochs): epoched eeg data (linked to beh)
			tfr_name (str): name of tfr analysis
		"""

		# set output parameters
		times =tfr['times']

		for cnd in tfr['power'].keys():
			x = tfr['power'][cnd]

			# change data into mne format (..., n_ch, n_freq, n_time)
			x = np.swapaxes(x, 0, 1) if x.ndim == 3 else np.swapaxes(x, 1, 2)
				
			# create mne object
			tfr_ = mne.time_frequency.AverageTFR(epochs.info,x,times,self.frex,
				       						tfr['cnd_cnt'][cnd], 
											method = self.method,
											comment = cnd)
			
			# save TFR object
			f_name = self.folder_tracker(['tfr',self.method],
									f'{tfr_name}_{cnd}-tfr.h5')
			tfr_.save(f_name, overwrite = True)
				

	def baseline_tfr(self,tfr:dict,base:dict,method:str,
		 			elec_oi:str='all') -> dict:
		"""
		Apply baseline correction via decibel conversion. 

		Args:
			tfr (dict): TF power per condition (epochs X nr_freq X 
				nr_ch X nr_time)
			base (dict): mean baseline TF power averaged across 
				trials (nr_freq X nr_chan)
			method (str): method for baseline correction
				elec_oi (str): Necessary when baselining depends on the 
				topographic distribution of electrodes 
				(i.e., when method is 'norm' or 'Z')

		Raises:
			ValueError: In case incorrect baselining method is specified

		Returns:
			tf (dict): normalized time frequency power
		"""

		cnds = list(tfr['power'].keys())
		
		if method == 'cnd_avg':
			cnd_avg = np.mean(np.stack([base[cnd] for cnd in cnds]), axis = 0)

		for cnd in cnds:
			if method != 'Z':
				#TODO: implement Z scoring baseline
				tfr['cnd_cnt'][cnd] = tfr['power'][cnd].shape[0]
				power = np.mean(tfr['power'][cnd], axis = 0)
			if method == 'cnd_spec':	
				tfr['power'][cnd] = self.db_convert(power, base[cnd])
			elif method == 'cnd_avg':
				tfr['power'][cnd] = self.db_convert(power, cnd_avg)
			elif method == 'norm':
				print('For normalization procedure it is assumed that it is as'
				 	 ' if all stimuli of interest are presented right')
				tfr['power'][cnd], info = self.normalize_power(power, 
															 list(elec_oi)) 
				tfr.update({'norm_info':info})
			elif method is None or not base:
				tfr['power'][cnd] = power
			else:
				raise ValueError('Invalid method specified')

			# # power values can now safely be averaged
			# if method != 'norm':
			# 	tf['power'][cnd] = np.mean(tf['power'][cnd], axis = 0)			

		return tfr
				
	def db_convert(self, power: np.array, base_power: np.array) -> np.array:
		"""
		Decibel (dB) is the ratio between frequency-band specific 
		power and baseline level of power in that same frequency band. 

		Args:
			power (np.array): TF power (epochs X nr_freq X nr_ch 
				X nr_time)
			base_power (np.array): baseline power (nr_freq X nr_chan)

		Returns:
			norm_power (np.array): baseline normalized power(dB)
		"""

		nr_time = power.shape[-1]
		base_power = np.repeat(base_power[...,np.newaxis],nr_time,axis = 2)
		norm_power = 10*np.log10(power/base_power)

		return norm_power

	@staticmethod
	def lateralization_index(tfr:dict,elec_oi:list='all',
		elec_pairs:list=None) -> dict:	

		"""
		Computes lateralization index (LI) for each frequency band and 
		sample per condition. It is assumed that all stimuli of interest 
		are presented with the same lateralization (i.e., all left or 
		all	right), ot this is set artificially via topo_flip.

		The lateralization index is computed as follows:

		lx = (contra - ipsi) / (contra + ipsi)

		In case elec_oi is a subset of electrodes, the lateralization 
		index is computed as the average lateralization index across 
		electrodes.					 

		Args:
			tfr (dict): dictionary as returned by read_tfr with 
	        time-frequencypower per condition (n_ch, nr_freq, n_times)
			elec_oi (list): list of electrodes of interest
			elec_pairs (dict): dictionary with ipsi- and contra-lateral		
			electrode pairs. Within this list, each unique electrode of 
			interest needs to be coupled once to its mirror electrode in 
			the other hemifield in a list. In case, the analysis is run 
	        to create topoplots, midline electrodes are unique and 
	        should be coupled to themselves (i.e., lx = 0). If not the 
	        function will assume a biosemi64 configuration.

			For example: [['Fp1':'Fp2']] will create a lateralization 
			index based on the difference between Fp1 and Fp2, where Fp1
			is assumed to be the contralateral electrode.


		Returns:
			tfr (dict): modified instance of tfr with normalized power 
			values	
		"""

		# make a copy of the original tfr
		tfr_ = copy.deepcopy(tfr)

		# set function parameters
		temp_tfr = list(tfr.values())[0]
		nr_sj = len(temp_tfr)
		channels = temp_tfr[0].ch_names
		nr_freqs = temp_tfr[0].freqs.size
		nr_times = temp_tfr[0].times.size

		# set electrode pairs
		if elec_pairs is None:
			ch_pairs = [['Fp1','Fp2'],['AF7','AF8'],['AF3','AF4'],
						['F7','F8'],['F5','F6'],['F3','F4'],
						['F1','F2'],('FT7','FT8'),['FC5','FC6'],
						['FC3','FC4'],['FC1','FC2'],['T7','T8'],
						['C5','C6'],['C3','C4'],['C1','C2'],['TP7','TP8'],
						['CP5','CP6'],['CP3','CP4'],['CP1','CP2'],
						['P9','P10'],['P7','P8'],['P5','P6'],['P3','P4'],
						['P1','P2'],['PO7','PO8'],['PO3','PO4'],
						['O1','O2'],('Fpz','Fpz'),['AFz','AFz'],
						['Fz','Fz'],['FCz','FCz'],['Cz','Cz'],['CPz','CPz'],
						['Pz','Pz'],['POz','POz'],['Oz','Oz'],['Iz','Iz']]
	
		if isinstance(elec_oi, list):
			pair_idx = [i for i, p in enumerate(ch_pairs) if p[0] in elec_oi]
			contra_idx = [channels.index(ch_pairs[idx][0]) for idx in pair_idx]
			ipsi_idx = [channels.index(ch_pairs[idx][1]) for idx in pair_idx]
			output = np.zeros((nr_sj,nr_freqs,nr_times))

		# frequency and condition loop
		for cnd in tfr_.keys():
			for f in range(nr_freqs):
				X =	np.stack([t._data[:,f,:] for t in tfr_[cnd]])
				if elec_oi == 'all':
					for ch in channels:
						if len([p for p in ch_pairs if p[0] == ch]) == 1:
							pair = [p for p in ch_pairs if p[0] == ch][0]
						elif len([p for p in ch_pairs if p[1] == ch]) == 1:
							pair = [p for p in ch_pairs if p[1] == ch][0][::-1]
						else:
							raise ValueError('Channel not found in ch_pairs.'
											f'the following pairs {ch_pairs} '
											' were found.')
						
						contra_idx = channels.index(pair[0])
						ipsi_idx = channels.index(pair[1])
			
						# modify trf in place by looping over all sjs
						for sj in range(nr_sj):
							contra = tfr[cnd][sj]._data[contra_idx,f]
							ipsi = tfr[cnd][sj]._data[ipsi_idx,f]
							lx = (contra - ipsi) / (contra + ipsi)
							tfr_[cnd][sj]._data[contra_idx,f] = lx

				else:
					contra = X[:,contra_idx].mean(axis = 1)
					ipsi = X[:,ipsi_idx].mean(axis = 1)

					lx = (contra - ipsi) / (contra + ipsi)
					output[:,f,:] = lx

			if isinstance(elec_oi, list):
				tfr_[cnd] = output

		return tfr_

	@staticmethod	
	def nextpow2(i):
		'''
		Gives the exponent of the next higher power of 2
		'''

		n = 1
		while 2**n < i: 
			n += 1
		
		return n

	def apply_hilbert(self, 
		X: np.ndarray, 
		l_freq: float, 
		h_freq: float, 
		s_freq: float = 512) -> np.ndarray:
		"""
		Apply the filter-Hilbert method for time-frequency 
		decomposition.

		This method first bandpass filters the input EEG signal 
		between `l_freq` and `h_freq`using the specified sampling 
		frequency (`s_freq`). It then applies the Hilbert transform
		to obtain the analytic signal, which can be used to extract 
		instantaneous amplitude and phase.

		Args:
			X (np.ndarray): array containing the EEG signal. 
			l_freq (float): Lower bound of the frequency band (Hz).
			h_freq (float): Upper bound of the frequency band (Hz).
			s_freq (float, optional): Sampling frequency of the signal 
				(Hz). Defaults to 512.

		Returns:
			np.ndarray: The analytic signal after bandpass filtering and 
				Hilbert transform. Same shape as input `X`.
		"""

		X = hilbert(filter_data(X, s_freq, l_freq, h_freq))

		return X





 	