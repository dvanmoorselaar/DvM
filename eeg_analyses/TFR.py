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
		normalize_wavelets (bool, optional): Whether to normalize
			wavelets to unit energy for consistent amplitude scaling 
			across frequencies. Recommended for most applications. 
			Defaults to True.

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
		base_method: str = 'trial_spec', 
		method: str = 'wavelet', 
		power: str = 'total',
		downsample: int = 1, 
		laplacian: bool = False,
		normalize_wavelets: bool = True,
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
		self.normalize_wavelets = normalize_wavelets
		self.report = report
		self.power = power
		
		# Initialize wavelets as None - will be generated when needed
		self.wavelets = None
		self.frex = None
		self._wavelet_params_hash = None  # Track when wavelets need regeneration
		
		if self.method == 'hilbert':
			self.freq_bands = self.create_freq_bands()
			self.frex = [(low + high)/2 for low, high in self.freq_bands]

	def _get_wavelet_params_hash(self):
		"""Generate a hash of current wavelet parameters to 
		detect changes."""
		import hashlib
		params = (self.min_freq, self.max_freq, self.num_frex, 
				 self.cycle_range, self.freq_scaling, self.normalize_wavelets,
				 self.epochs.info['sfreq'], self.epochs.times.size)
		return hashlib.md5(str(params).encode()).hexdigest()
	
	def _ensure_wavelets(self):
		"""Ensure wavelets are generated and up-to-date 
		with current parameters."""
		if self.method != 'wavelet':
			return
			
		current_hash = self._get_wavelet_params_hash()
		
		if (self.wavelets is None or self.frex is None or 
			self._wavelet_params_hash != current_hash):
			
			s_freq = self.epochs.info['sfreq']
			nr_time = self.epochs.times.size
			self.wavelets, self.frex = self.create_morlet(
				self.min_freq, self.max_freq, self.num_frex, 
				self.cycle_range, self.freq_scaling, 
				nr_time, s_freq, self.normalize_wavelets)
			self._wavelet_params_hash = current_hash

	def select_tfr_data(
		self, 
		elec_oi: Union[str, list], 
		excl_factor: dict = None, 
		topo_flip: dict = None,
		cnds: dict = None
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
			cnds (dict, optional): Conditions for condition-specific
				evoked subtraction when calculating induced power.
				Format: {column_name: [cond1, cond2]}.

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
	
	def create_freq_bands(self) -> list:
		"""Creates frequency bands for Hilbert transform.
		
		Uses same frequency spacing as wavelet method (linear or log) 
		to create overlapping frequency bands. Band edges are placed 
		halfway between center frequencies.
		
		Returns:
			list: List of tuples containing (low_freq, high_freq) for 
			each band
		"""
		# Get center frequencies using same spacing as wavelets
		if self.freq_scaling == 'log':
			center_freqs = np.logspace(np.log10(self.min_freq), 
									np.log10(self.max_freq), 
									self.num_frex)
		else:  # linear
			center_freqs = np.linspace(self.min_freq, 
									self.max_freq, 
									self.num_frex)
		
		# Create band edges halfway between center frequencies
		freq_bands = []
		for i in range(len(center_freqs)):
			if i == 0:  # First band
				low_freq = center_freqs[0] - (center_freqs[1] - center_freqs[0])/2
				low_freq = max(0.1, low_freq)  # Avoid too low frequencies
			else:
				low_freq = (center_freqs[i-1] + center_freqs[i])/2
				
			if i == len(center_freqs)-1:  # Last band
				high_freq = center_freqs[-1] + (center_freqs[-1] - center_freqs[-2])/2
			else:
				high_freq = (center_freqs[i] + center_freqs[i+1])/2
				
			freq_bands.append((low_freq, high_freq))
		
		return freq_bands
	
	def create_morlet(
		self, 
		min_freq: int, 
		max_freq: int, 
		num_frex: int, 
		cycle_range: tuple, 
		freq_scaling: str, 
		nr_time: int, 
		s_freq: float,
		normalize_wavelets: bool = True
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
			normalize_wavelets (bool, optional): Whether to normalize 
				wavelets to unit energy (Cohen-style normalization). 
				This ensures consistent amplitude scaling across 
				frequencies and is recommended for most applications. 
				Defaults to True.

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
			wavelet = (np.exp(2j * math.pi * frex[fi] * t) * 
					   np.exp(-t**2/(2*s[fi]**2)))
			
			# Optional normalization to unit energy (Cohen-style)
			if normalize_wavelets:
				wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
			
			wavelets[fi,:] = wavelet
		
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


		# Perform convolution in frequency domain
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

	def compute_tfrs(self, epochs: mne.Epochs, output:str = 'power'):
		
		# get data
		raw_conv = self.tfr_loop(epochs)
		raw_conv = raw_conv.astype(np.complex64, copy=False)
		if output == 'power':
			X = raw_conv.real**2
			X += raw_conv.imag**2
		elif output == 'phase':
			#TODO: check difference between sin and cos
			X = np.cos(np.angle(raw_conv))

		# apply baseline (trial specific)
		if output == 'power' and self.baseline is not None:
			print('Applying TFR baseline correction')
			s,e = self.baseline
			base_idx = get_time_slice(epochs.times,s,e)
			base = X[..., base_idx].mean(axis=-1, keepdims=True)
			X = 10*(np.log10(X+1e-12)-np.log(base+1e-12))

		# make sure freqs are the first dim
		X = np.swapaxes(X, 0, 1)

		return X

	def condition_tfrs(
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
		Performs time-frequency (TFR) decomposition for specified 
		conditions with optional lateralization.

		This function computes time-frequency representations (TFRs) 
		for specified conditions and electrodes of interest. It supports 
		lateralization handling via topography flipping (`topo_flip`). 
		The function also supports baseline correction, downsampling, and 
		condition-specific trial selection. The output is saved in a 
		format compatible with MNE-Python.

		Args:
			pos_labels (dict): Dictionary specifying the position labels 
				for stimuli of interest. For example, 
				`{'target_loc': [2, 6]}` specifies the column name 
				(`'target_loc'`) and the values corresponding to 
				left and right hemifield stimuli. Note: lateralized 
				TFRs are optional when `topo_flip` is defined, as 
				topography flipping can handle lateralization.
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

		# Ensure wavelets are generated BEFORE creating output dict
		self._ensure_wavelets()
		
		# select trials of interest (e.g., lateralized stimuli)
		if isinstance(pos_labels, dict):
			idx = ERP.select_lateralization_idx(df, pos_labels, midline)
		elif pos_labels is None:
			idx = np.arange(len(df))
		else:
			raise TypeError(f"pos_labels must be a dict or None, \
							got {type(pos_labels).__name__}")

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
			  'base':'baseline' if self.baseline is not None else 'no_base', 
		 	  'times':times[idx_2_save], 
			  'frex': self.frex, 'power': {},
			  'cnd_cnt':{}}
		
		if cnds is None:
			cnds = ['all_data']
		else:
			(cnd_header, cnds), = cnds.items()

		for c, cnd in enumerate(cnds):
			counter = c + 1 
			print(f'Decomposing condition {counter}: {cnd} \n')
			# set tfr name
			tfr_name = f'sj_{self.sj}_{name}'

			# slice condition trials
			if cnd == 'all_data':
				idx_c = idx
			else:
				idx_c = np.where(df[cnd_header] == cnd)[0]
				idx_c = np.intersect1d(idx, idx_c)

			# calculate induced power
			if self.power == 'induced':
				print('Calculating induced power (i.e., subtracting evoked ' \
				'response)')
				evoked = epochs[idx_c].average()
				tfr_epochs = epochs[idx_c].subtract_evoked(evoked)
			elif self.power == 'evoked':
				print('Calculating evoked power (i.e., average of ' \
				'epochs)')
				tfr_epochs = epochs[idx_c].average()	
			else: # total power
				tfr_epochs = epochs[idx_c]

			# TF decomposition 
			raw_conv = self.tfr_loop(tfr_epochs)

			# get baseline power (correction is done after condition loop)
			if self.baseline is not None:
				base_power = raw_conv[..., base_idx].real**2
				base_power += raw_conv[..., base_idx].imag**2
				if self.base_method == 'trial_spec':
					base[cnd] = np.mean(base_power, 
									axis = -1)
				else: # cnd_avg or cnd_spec
					base[cnd] = np.mean(base_power, 
									axis = (0,3))
			else:
				base = {}
			# populate tf
			power = raw_conv[..., idx_2_save].real**2
			power += raw_conv[..., idx_2_save].imag**2
			tfr['power'][cnd] = power

		# baseline correction
		tfr = self.baseline_tfr(tfr,base,self.base_method,elec_oi)

		if self.report:
			self.generate_tfr_report(tfr,
							epochs.info,f'sj_{self.sj}_{name}')

		# save output
		self.save_to_mne_format(tfr,epochs,tfr_name)

	def tfr_loop(self, epochs: mne.Epochs) -> np.array:
		"""
		TODO: check hilbert convolution
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

		# # Ensure wavelets are up-to-date before processing
		# self._ensure_wavelets()

		# initialize convolution array
		s_freq = epochs.info['sfreq']
		nr_time = epochs.times.size
		nr_ch = len(epochs.ch_names)
		if isinstance(epochs, mne.epochs.EpochsFIF):
			nr_epochs = len(epochs.events)
		else:  # mne.epochs.EpochsArray
			nr_epochs = 1
			epochs._data = epochs._data[np.newaxis, ...]  # Add epoch dimension

		l_conv = 2**self.nextpow2(nr_time * nr_epochs + nr_time - 1)
		raw_conv = np.zeros((nr_epochs, self.num_frex, nr_ch, 
							nr_time), dtype = complex)

		# loop over channels			
		for ch_idx in range(nr_ch):
			print(f'Decomposing channel {ch_idx+1} out of {nr_ch} channels', 
				  end='\r')

			x = epochs._data[:, ch_idx]
			if self.method == 'wavelet':
				# fft decomposition
				x_fft = fft(x.ravel(), l_conv)
				for f in range(self.num_frex):
					# convolve and get analytic signal
					m = self.wavelet_convolution(x_fft, self.wavelets[f], 
													l_conv, nr_time, nr_epochs)
					# populate output array
					raw_conv[:,f,ch_idx] = m
			elif self.method == 'hilbert':
				# Loop through frequency bands
				for f, (l_freq, h_freq) in enumerate(self.freq_bands):
					# Bandpass filter across all epochs at once
					x_filt = mne.filter.filter_data(x,s_freq, l_freq, h_freq, 
												method='fir', 
												phase='zero-double',
												verbose=False)
	
					# Apply Hilbert transform
					analytic = hilbert(x_filt, axis=-1)
					
					# Store result
					raw_conv[:,f,ch_idx] = analytic

		# for memory efficiency
		raw_conv = raw_conv.astype(np.complex64, copy=False)
		
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
											comment = tfr['base'])
			
			# save TFR object
			f_name = self.folder_tracker(['tfr',self.method],
									f'{tfr_name}_{cnd}-tfr.h5')
			tfr_.save(f_name, overwrite = True)
				
	def baseline_tfr(self,tfr:dict,base:dict,method:str,
		 			elec_oi:str='all') -> dict:
		"""
		Apply baseline correction via decibel conversion. 

		For 'trial_spec': applies baseline correction to individual trials 
		then averages
		For 'cnd_spec'/'cnd_avg': averages trials first then applies baseline 
		correction. 

		Args:
			tfr (dict): TF power per condition (epochs X nr_freq X 
				nr_ch X nr_time)
			base (dict): baseline TF power. Format depends on method:
				- For 'cnd_spec'/'cnd_avg': mean baseline across trials 
				(nr_freq X nr_chan)
				- For 'trial_specific': baseline per trial 
				(epochs X nr_freq X nr_chan)
			method (str): method for baseline correction. Options:
				- 'cnd_spec': condition-specific baseline 
					(each condition uses its own baseline)
				- 'cnd_avg': condition-averaged baseline (all conditions 
				use same baseline)
				- 'trial_spec': trial-specific baseline correction
			elec_oi (str): Necessary when baselining depends on the 
				topographic distribution of electrodes 
				(i.e., when method is 'norm' or 'Z')

		Raises:
			ValueError: In case incorrect baselining method is specified

		Returns:
			tfr (dict): baseline corrected time frequency power
		"""

		cnds = list(tfr['power'].keys())
		
		if method == 'cnd_avg':
			cnd_avg = np.mean(np.stack([base[cnd] for cnd in cnds]), axis = 0)

		for cnd in cnds:
			power = tfr['power'][cnd]  # Shape: (epochs X nr_freq X nr_ch X nr_time)
			tfr['cnd_cnt'][cnd] = power.shape[0]

			if method == 'trial_spec':
				#  baseline correct individual trials, then average
				power = self.db_convert(power, base[cnd])
				tfr['power'][cnd] = np.mean(power, axis = 0)
			elif method == 'cnd_spec' or method == 'cnd_avg':
				# average first, then baseline correct
				avg_power = np.mean(power, axis=0)  
				if method == 'cnd_spec':
					tfr['power'][cnd] = self.db_convert(avg_power, base[cnd])
				else:
					tfr['power'][cnd] = self.db_convert(avg_power,cnd_avg)
			elif method == 'norm':
				print('For normalization procedure it is assumed that it is as'
				 	 ' if all stimuli of interest are presented right')
				# Average first for normalization
				avg_power = np.mean(power, axis=0)  # Shape: (nr_freq X nr_ch X nr_time)
				tfr['power'][cnd], info = self.normalize_power(avg_power, 
															 list(elec_oi)) 
				tfr.update({'norm_info':info})
			elif method is None or not base:
				# Simply average across trials when no baseline correction
				tfr['power'][cnd] = np.mean(power, axis = 0)
			else:
				raise ValueError(f'Invalid method specified: {method}')

		return tfr	
	
	def db_convert(self, power: np.array, base_power: np.array) -> np.array:
		"""
		Decibel (dB) conversion with automatic detection of 
		baseline type.
		
		Handles both trial-specific and condition-averaged baseline correction 
		based on the dimensionality of both power and base_power inputs.

		Args:
			power (np.array): TF power. Can be:
				- 4D (epochs X nr_freq X nr_ch X nr_time): individual trials
				- 3D (nr_freq X nr_ch X nr_time): trial-averaged power
			base_power (np.array): baseline power. Can be:
				- 2D (nr_freq X nr_chan): condition-averaged baseline
				- 3D (epochs X nr_freq X nr_chan): trial-specific baseline

		Returns:
			norm_power (np.array): baseline normalized power (dB)
				- Same shape as input power
		"""

		nr_time = power.shape[-1]
		if base_power.ndim == 2:
			# trial generic baseline (condition specific or averaged)
			base_power = np.repeat(base_power[...,np.newaxis],nr_time,axis = 2)
		elif base_power.ndim == 3:
			# trial specific baseline
			base_power = np.repeat(base_power[...,np.newaxis],nr_time,axis=-1)
		else:
			raise ValueError('base_power should be either 2D or 3D array')
		
		norm_power = 10*(np.log10(power+1e-12)-np.log10(base_power+1e-12))

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





 	