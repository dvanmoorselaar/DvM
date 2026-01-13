"""
Time-Frequency Representation (TFR) Analysis for EEG Data.

This module provides comprehensive functionality for time-frequency 
decomposition analysis of EEG data. It implements wavelet-based and 
Hilbert transform-based methods for extracting spectral power and phase 
information across time and frequency domains.

The module supports multiple time-frequency analysis methods including:
- Morlet wavelet convolution with customizable parameters
- Hilbert transform for broadband analysis  
- Baseline correction and normalization procedures
- Condition-based analysis with statistical testing
- Integration with MNE-Python data structures

Key Features
------------
- Flexible frequency specification (linear/logarithmic scaling)
- Customizable wavelet parameters for temporal-spectral resolution 
  trade-offs
- Multiple baseline correction methods (condition-specific or averaged)
- Spatial filtering options (Laplacian filtering)
- Automated report generation with visualizations
- Memory-efficient processing with optional downsampling

Classes
-------
TFR : Main time-frequency analysis class
    Provides complete pipeline for TFR analysis from raw epochs to 
    statistical results with visualization support.

Notes
-----
This implementation follows established practices from Cohen (2014) 
"Analyzing Neural Time Series Data" and integrates with the MNE-Python 
ecosystem for comprehensive EEG analysis workflows.

References
----------
Cohen, M. X. (2014). Analyzing neural time series data: theory and 
practice. MIT press.

Examples
--------
Basic time-frequency analysis:

>>> tfr = TFR(sj=1, epochs=epochs, df=behavioral_data, 
...           min_freq=4, max_freq=40, num_frex=25)
>>> tfr_results = tfr.compute_tfrs(conditions={'condition': ['A', 'B']})

Custom wavelet parameters:

>>> tfr = TFR(sj=1, epochs=epochs, df=behavioral_data,
...           cycle_range=(4, 12), freq_scaling='linear')
>>> tfr.condition_tfrs(conditions=['encoding', 'retrieval'])

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
from support.preprocessing_utils import *
from IPython import embed


class TFR(FolderStructure):
	"""
	Time-frequency representation analysis for EEG data.
	
	Provides comprehensive functionality for time-frequency 
	decomposition of EEG data using wavelet convolution or Hilbert 
	transform methods. Supports flexible frequency specification, 
	baseline correction, condition-based analysis, and automated 
	visualization.
	
	This class inherits from FolderStructure for file management and 
	output organization capabilities.

	Parameters
	----------
	sj : int
		Subject identifier for file naming and organization.
	epochs : mne.Epochs
		Preprocessed EEG data segmented into epochs. Should contain 
		clean, artifact-free data ready for time-frequency analysis.
	df : pd.DataFrame
		Behavioral data corresponding to the EEG epochs. Must have 
		same number of rows as epochs.
	min_freq : int, default=4
		Minimum frequency in Hz for time-frequency analysis. 
		Should be above 1 Hz for stable wavelet analysis.
	max_freq : int, default=40
		Maximum frequency in Hz for time-frequency analysis. 
		Cannot exceed Nyquist frequency (sampling_rate / 2).
	num_frex : int, default=25
		Number of frequencies between min_freq and max_freq. 
		Typical range is 20-30 for good frequency resolution 
		without excessive computation time.
	cycle_range : tuple of int, default=(3, 10)
		Range of cycles for wavelet construction as (min_cycles,
		max_cycles). Controls temporal-frequency resolution trade-off. 
		Lower cycles give better temporal resolution, higher cycles give 
		better frequency resolution.
	freq_scaling : {'log', 'linear'}, default='log'
		Frequency axis scaling method:
		- 'log': Logarithmic spacing (recommended for lower frequencies)
		- 'linear': Linear spacing (recommended for higher frequencies)
	baseline : tuple of float, optional
		Time range (start, end) in seconds for baseline correction.
		If None, no baseline correction is applied.
	base_method : {'trial_spec', 'cnd_avg'}, default='trial_spec'
		Baseline correction method:
		- 'cnd_spec': Condition-specific baseline correction 
		  (recommended by Cohen 2014)
		- 'trial_spec': Trial-specific baseline correction
		- 'cnd_avg': Condition-averaged baseline correction
	method : {'wavelet', 'hilbert'}, default='wavelet'
		Time-frequency decomposition method:
		- 'wavelet': Morlet wavelet convolution
		- 'hilbert': Hilbert transform (broadband analysis)
	power : {'total', 'evoked', 'induced'}, default='total'
		Type of power to compute:
		- 'total': Total power (evoked + induced)
		- 'evoked': Evoked power (phase-locked activity)
		- 'induced': Induced power (non-phase-locked activity)
	downsample : int, default=1
		Temporal downsampling factor. Values > 1 reduce temporal 
		resolution but speed up analysis and reduce memory usage.
	laplacian : bool, default=False
		TODO: check at which point during preprocessing to apply
		Whether to apply Laplacian spatial filtering before analysis.
		Helps reduce volume conduction effects.
	normalize_wavelets : bool, default=True
		Whether to normalize wavelets to unit energy. Recommended 
		for consistent amplitude scaling across frequencies.
	report : bool, default=False
		Whether to generate automated HTML reports with visualizations.
	
	Examples
	--------
	Basic time-frequency analysis:
	
	>>> tfr = TFR(sj=1, epochs=epochs, df=df, min_freq=4, max_freq=40)
	>>> results = tfr.compute_tfrs(conditions=['condition_A', 
	...   'condition_B'])
	
	Custom wavelet parameters with baseline correction:
	
	>>> tfr = TFR(sj=1, epochs=epochs, df=df, 
	...           cycle_range=(4, 12), baseline=(-0.2, 0))
	>>> tfr.condition_tfrs(conditions=['encoding', 
	...   'retrieval'])
	
	High-frequency analysis with linear scaling:
	
	>>> tfr = TFR(sj=1, epochs=epochs, df=df, 
	...           min_freq=20, max_freq=80, freq_scaling='linear')
	>>> results = tfr.compute_tfrs()
	
	Notes
	-----
	Time-frequency analysis reveals how spectral power changes over 
	time, providing insights into neural oscillations and their 
	functional roles. The choice of wavelet parameters (cycles) 
	determines the temporal-frequency resolution trade-off: more cycles 
	give better frequency resolution but poorer temporal resolution.
	
	References
	----------
	Cohen, M. X. (2014). Analyzing neural time series data: theory and 
	practice. MIT press.
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
		"""
		Initialize TFR analysis object.
		
		Sets up time-frequency analysis parameters and validates input 
		data. Creates wavelets if using wavelet method, or prepares 
		frequency bands for Hilbert transform method.
		
		Parameters are documented in the class docstring above.
		"""
		
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
		# Track when wavelets need regeneration
		self._wavelet_params_hash = None  
		
		if self.method == 'hilbert':
			self.freq_bands = self.create_freq_bands()
			self.frex = [(low + high)/2 for low, high in self.freq_bands]

	def _get_wavelet_params_hash(self):
		"""
		Generate hash of current wavelet parameters for change 
		detection.
		
		Creates MD5 hash from all parameters that affect wavelet 
		generation to detect when wavelets need to be regenerated.
		
		Returns
		-------
		str
			MD5 hash string representing current parameter state.
		"""
		import hashlib
		params = (self.min_freq, self.max_freq, self.num_frex, 
				 self.cycle_range, self.freq_scaling, self.normalize_wavelets,
				 self.epochs.info['sfreq'], self.epochs.times.size)
		return hashlib.md5(str(params).encode()).hexdigest()
	
	def _ensure_wavelets(self):
		"""
		Ensure wavelets are generated and up-to-date with current 
		parameters.
		
		Checks if wavelets need regeneration based on parameter changes 
		and creates new wavelets/frequency bands as needed. Handles both 
		waveletand Hilbert transform methods.
		"""
		if self.method != 'wavelet':
			return
			
		current_hash = self._get_wavelet_params_hash()
		
		if self.method == 'wavelet':
			if (self.wavelets is None or self.frex is None or 
				self._wavelet_params_hash != current_hash):
				
				s_freq = self.epochs.info['sfreq']
				nr_time = self.epochs.times.size
				self.wavelets, self.frex = self.create_morlet(
					self.min_freq, self.max_freq, self.num_frex, 
					self.cycle_range, self.freq_scaling, 
					nr_time, s_freq, self.normalize_wavelets)
				self._wavelet_params_hash = current_hash
		elif self.method == 'hilbert':
			if (self.freq_bands is None or self.frex is None or 
				self._wavelet_params_hash != current_hash):

				self.freq_bands = self.create_freq_bands()
				self.frex = [(low + high)/2 for low, high in self.freq_bands]
				self._wavelet_params_hash = current_hash

	def select_tfr_data(
		self, 
		elec_oi: Union[str, list], 
		excl_factor: dict = None, 
		topo_flip: dict = None,
		cnds: dict = None
	) -> Tuple[mne.Epochs, pd.DataFrame]:
		"""
		Select and preprocess data for time-frequency analysis.

		Applies optional preprocessing steps including trial exclusion,
		spatial filtering, topography flipping, and electrode selection
		in preparation for TFR analysis.

		Parameters
		----------
		elec_oi : str or list
			Electrode selection criteria:
			- 'all': Use all available electrodes in ch_names
			- 'posterior': Posterior electrodes (parietal, occipital 
			regions)
			- 'frontal': Frontal electrodes (prefrontal, frontal 
			   regions)
			- 'central': Central electrodes (motor, somatosensory 
			   regions)
			- list: Specific electrode names to select
		excl_factor : dict, optional
			Trial exclusion criteria as 
			{column_name: [values_to_exclude]}. Trials matching any 
			specified values will be removed.
		topo_flip : dict, optional
			Topography flipping criteria as 
			{column_name: condition_value}. Flips electrode positions 
			for specified trials (e.g., for lateralization analysis).
		cnds : dict, optional
			#TODO: Document condition specification for induced power 
			# analysis
			
		Returns
		-------
		epochs : mne.Epochs
			Preprocessed EEG epochs after applying transformations.
		df : pd.DataFrame
			Behavioral data frame corresponding to remaining epochs.
			
		Notes
		-----
		Processing steps applied in order:
		1. Trial exclusion based on behavioral criteria
		2. Laplacian spatial filtering (if self.laplacian=True)
		3. Topography flipping for lateralization analysis
		4. Electrode selection and data reduction
		
		For lateralization analysis without explicit topography 
		specification, assumes all stimuli are presented in the right 
		visual field.
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
			epochs = ERP.flip_topography(epochs, df, topo_flip)
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
		"""
		Create frequency bands for Hilbert transform analysis.
		
		Generates overlapping frequency bands with the same frequency 
		spacing as the wavelet method (linear or logarithmic). Band 
		edges are positioned halfway between adjacent center frequencies 
		to ensure consistent frequency coverage.

		Returns
		-------
		list of tuple
			List of (low_freq, high_freq) tuples defining each frequency 
			band. Length equals self.num_frex.
			
		Notes
		-----
		This method is used when self.method='hilbert' to create 
		frequency bands that match the center frequencies used in 
		wavelet analysis, ensuring comparable results between methods.
		
		Edge handling:
		- First band: Extended downward by half the frequency step
		- Last band: Extended upward by half the frequency step
		- Minimum frequency capped at 0.1 Hz to avoid filtering issues
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
				freq_step = center_freqs[1] - center_freqs[0]
				low_freq = center_freqs[0] - freq_step / 2
				low_freq = max(0.1, low_freq)  # Avoid too low frequencies
			else:
				low_freq = (center_freqs[i-1] + center_freqs[i])/2
				
			if i == len(center_freqs)-1:  # Last band
				freq_step = center_freqs[-1] - center_freqs[-2]
				high_freq = center_freqs[-1] + freq_step / 2
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
		Create Morlet wavelets for time-frequency decomposition.
		
		Generates a bank of complex Morlet wavelets optimized for EEG 
		time-frequency analysis. Implementation follows Cohen (2014) 
		methodology with proper normalization and frequency scaling.
		
		Parameters
		----------
		min_freq : int
			Minimum frequency in Hz for wavelet bank.
		max_freq : int  
			Maximum frequency in Hz for wavelet bank.
		num_frex : int
			Number of wavelets to create between min_freq and max_freq.
		cycle_range : tuple of int
			Range of cycles as (min_cycles, max_cycles). Controls 
			temporal-frequency resolution trade-off.
		freq_scaling : {'log', 'linear'}
			Frequency spacing method:
			- 'log': Logarithmic spacing (better for lower frequencies)
			- 'linear': Linear spacing (uniform frequency resolution)
		nr_time : int
			Number of time points for wavelet construction. Should match
			or exceed epoch length for proper convolution.
		s_freq : float
			Sampling frequency in Hz.
		normalize_wavelets : bool, default=True
			Whether to normalize wavelets to unit energy for consistent 
			amplitude scaling across frequencies.
			
		Returns
		-------
		wavelets : np.ndarray
			Complex Morlet wavelets with shape 
			(n_frequencies, n_timepoints). Each row contains one wavelet 
			in the frequency domain.
		frex : np.ndarray  
			Center frequencies corresponding to each wavelet.
			
		Raises
		------
		ValueError
			If freq_scaling is not 'log' or 'linear'.
			
		Notes
		-----
		Morlet wavelets are Gaussian-windowed complex exponentials that 
		provide optimal time-frequency resolution for oscillatory 
		signals. The number of cycles determines the temporal-frequency 
		precision:
		
		- Low cycles (e.g., 3): Good temporal resolution, poor frequency 
		  resolution
		- High cycles (e.g., 10): Poor temporal resolution, 
		  good frequency resolution
		
		The wavelets are generated in the frequency domain and 
		normalized  according to Cohen (2014) standards when 
		normalize_wavelets=True.
		
		References
		----------
		Cohen, M. X. (2014). Analyzing neural time series data: theory
		and practice. MIT press.
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

	def compute_tfrs(
		self, 
		epochs: mne.Epochs, 
		output: str = 'power',
		for_decoding: bool = False,
		cnd_idx: Optional[list] = None
	) -> np.ndarray:
		
		"""
		Compute time-frequency representations with optional baseline 
		correction and induced power calculation.

		This function performs time-frequency decomposition on EEG 
		epochs with support for different power types (total, evoked, 
		induced) and applies appropriate baseline correction depending 
		on the intended use case. For decoding analyses, it uses percent 
		change from baseline (or raw power if no baseline), while for 
		visualization/statistics it uses decibel conversion with log 
		baseline correction.

		Args:
			epochs (mne.Epochs): Preprocessed EEG data segmented into 
				epochs.
				The data is used for time-frequency decomposition.
			output (str, optional): Type of output to compute. 
				Supported values:
				- 'power': Compute instantaneous power 
				  (magnitude squared)
				- 'phase': Compute instantaneous phase (cosine of 
				  phase angle)
				Defaults to 'power'.
			for_decoding (bool, optional): Whether the output will be 
			    used for classification/decoding analysis. When True, 
				applies percent change baseline correction (or raw power 
				if no baseline) which preserves linear relationships 
				needed for most classifiers. When False, applies 
				log-based decibel conversion for
				visualization/statistics. Defaults to False.
			cnd_idx (Optional[list], optional): List of trial indices 
			    for each condition when computing induced power 
				(self.power='induced'). Each element should be an array 
				of trial indices belonging to the same condition. 
				Required when self.power='induced' to enable 
				condition-specific evoked response subtraction. 
				Defaults to None.

		Returns:
			np.ndarray: Time-frequency representation with shape 
				(nr_frequencies, nr_epochs, nr_channels, nr_time) for 
				epoched data or (nr_frequencies, nr_channels, nr_time) 
				for averaged data. Shape depends on input epochs
				structure and the power type (evoked vs. epoched data).

		Notes:
			- Power types (controlled by self.power):
				* 'total': Standard power computation without evoked 
				   subtraction
				* 'induced': Subtracts condition-specific evoked 
				   response from each trial before TFR computation 
				   (requires cnd_idx parameter)
				* 'evoked': Computes power of the averaged evoked 
				   response
			- For decoding (for_decoding=True): Uses percent change from 
			baseline when self.baseline is specified, or raw power when 
			no baseline. This preserves the linear feature space needed 
			for classification.
			- For statistics/visualization (for_decoding=False): Uses 
			decibel conversion (10*log10(power/baseline)) when baseline 
			is specified.
			- The baseline period is defined by self.baseline tuple 
			(start, end) in seconds.
			- Power computation: real²+ imag² of the complex analytic 
			signal.
			- Phase computation: cosine of the phase angle of the 
			analytic signal.

		Raises:
			ValueError: If self.power='induced' but cnd_idx is not 
			provided, or if an unsupported output type is specified.	
		"""

		if self.power == 'induced':
			if cnd_idx is None:
				raise ValueError("cnd_idx must be provided when "
					 "power is 'induced'")

		if self.power == 'induced':
			# subtract condition specific evoked from each trial
			print('Calculating induced power: subtracting condition specific '\
			'evoked from each trial')
			for idx in cnd_idx:
				evoked = epochs[idx].average()
				epochs[idx]._data = epochs[idx].subtract_evoked(evoked)._data

		# Ensure wavelets are generated TFR decomposition
		self._ensure_wavelets()
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
			s,e = self.baseline
			base_idx = get_time_slice(epochs.times,s,e)
			base = X[..., base_idx].mean(axis=-1, keepdims=True)

			if for_decoding:
				print('Applying percent change baseline for decoding')
				X = (X - base) / base
			else:
				print('Applying TFR baseline correction')
				X = 10*(np.log10(X+1e-12)-np.log10(base+1e-12))
		elif output == 'power' and for_decoding:
			print('Using raw power for decoding (no baseline)')	
		
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
		Compute condition-specific time-frequency representations with 
		optional lateralization analysis.

		This is the primary analysis method for computing time-frequency 
		decompositions across experimental conditions. It supports 
		sophisticated preprocessing including lateralization handling, 
		spatial filtering, trial exclusion, and condition-specific 
		baseline correction. Results are automatically saved in 
		MNE-compatible format for further analysis and visualization.

		The method implements a complete TFR analysis pipeline:
		1. Data preprocessing (trial exclusion, spatial filtering)
		2. Condition-specific trial selection and lateralization
		3. Time-frequency decomposition (wavelet or Hilbert)
		4. Baseline correction (trial- or condition-specific)
		5. Temporal downsampling and windowing
		6. Output formatting and file saving

		Parameters
		----------
		pos_labels : dict
			Position labels for lateralized stimuli analysis. Dictionary 
			mapping column names to position values, e.g., 
			{'target_loc': [2, 6]} where values correspond to left and 
			right hemifield stimulus positions. Used to select trials 
			for lateralization analysis. Can be None if no 
			position-based selection is needed.
		cnds : dict, optional
			Experimental conditions for TFR analysis. Dictionary with 
			column name as key and list of condition labels as values, 
			e.g., {'condition': ['encoding', 'retrieval']}. Each 
			condition will be processed separately. If None, all trials 
			are analyzed as a single condition. Default is None.
		elec_oi : {'all', list}, default='all'
		elec_oi : str or list
			Electrode selection criteria:
			- 'all': Use all available electrodes in ch_names
			- 'posterior': Posterior electrodes (parietal, occipital 
			regions)
			- 'frontal': Frontal electrodes (prefrontal, frontal 
			   regions)
			- 'central': Central electrodes (motor, somatosensory 
			   regions)
			- list: Specific electrode names to select
		midline : dict, optional
			Trials with stimuli on the vertical midline for 
			lateralization analysis. Dictionary specifying column name 
			and values for midline trials, e.g., {'target_loc': [0]}. 
			These trials are typically excluded from lateralization 
			comparisons. Default is None.
		topo_flip : dict, optional
			Topography flipping specification for lateralization 
			analysis. Dictionary mapping column name to condition 
			values that require electrode position flipping, e.g., 
			{'stimulus_side': ['left']}. Enables combining left and 
			right stimuli by flipping left-stimulus topographies to 
			match right-stimulus electrode positions. Default is None.
		window_oi : tuple of float, optional
			Time window of interest in seconds as (start, end). If 
			specified, TFR output is cropped to this temporal window 
			to reduce memory usage and focus analysis. E.g., (0.0, 1.0) 
			for 0-1 second post-stimulus. Default is None (full epoch).
		excl_factor : dict, optional
			Trial exclusion criteria for data cleaning. Dictionary 
			mapping column names to lists of values to exclude, e.g., 
			{'response_accuracy': ['incorrect'], 'artifact': [True]}. 
			Trials matching any exclusion criteria are removed before 
			TFR analysis. Default is None.
		name : str, default='main'
			Analysis name for output file identification. Used in 
			filename generation for saved TFR objects. Should be 
			descriptive of the analysis (e.g., 'alpha_lateralization', 
			'gamma_encoding').

		Returns
		-------
		None
			This method does not return data directly. Instead, it saves 
			condition-specific TFR results to disk in MNE-Python 
			AverageTFR format. Files are saved in the TFR analysis 
			directory with names following the pattern: 
			'sj_{subject}_{name}_{condition}-tfr.h5'.

		Notes
		-----
		**Processing Pipeline:**
		
		1. **Data Selection**: Applies electrode selection,
		   trial exclusion, and topography flipping as specified
		2. **Lateralization**: If pos_labels provided, selects trials 
		   based on stimulus positions for lateralization analysis
		3. **Condition Processing**: For each condition, performs TFR 
		   decomposition using the specified method (wavelet or Hilbert)
		4. **Baseline Correction**: Applies baseline normalization using 
		   the method specified in self.base_method
		5. **Output**: Saves results as MNE AverageTFR objects for 
		   compatibility with MNE analysis workflows

		**Lateralization Analysis:**
		When pos_labels is provided, the method assumes a standard 
		lateralization paradigm where stimuli are presented in left 
		and right visual fields. Combined with topo_flip, this enables 
		analysis of lateralized neural responses by aligning electrode 
		positions relative to stimulus location.

		**Memory Management:**
		Large datasets can be reduced in memory usage by:
		- Using window_oi to crop temporal dimension
		- Setting downsample > 1 to reduce temporal resolution  
		- Selecting specific electrodes with elec_oi

		**File Organization:**
		Output files are organized in the project's TFR directory 
		structure. Each condition generates a separate file to enable 
		flexible downstream analysis and comparison.

		Examples
		--------
		Basic condition comparison:

		>>> tfr.condition_tfrs(
		...     pos_labels=None,
		...     cnds={'task': ['encoding', 'retrieval']},
		...     window_oi=(0.0, 2.0)
		... )

		Lateralization analysis:

		>>> tfr.condition_tfrs(
		...     pos_labels={'target_location': [1, 2]},  # left, right
		...     topo_flip={'target_location': [1]},   # flip left trials
		...     cnds={'condition': ['attend', 'ignore']},
		...     elec_oi=['P7', 'P8', 'PO7', 'PO8']
		... )

		High-frequency analysis with artifact exclusion:

		>>> tfr.condition_tfrs(
		...     pos_labels=None,
		...     cnds={'stimulus': ['faces', 'houses']},
		...     excl_factor={'muscle_artifact': [True], 
		...                  'response_time': ['too_fast']},
		...     name='gamma_categorization'
		... )

		See Also
		--------
		compute_tfrs : Lower-level TFR computation method
		select_tfr_data : Data preprocessing and selection
		baseline_tfr : Baseline correction implementation
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
			tfr_name = f'sub_{self.sj}_{cnd}_{name}'

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
		# TODO: check hilbert method
		Perform time-frequency decomposition across all channels.

		Helper method that applies the specified time-frequency method 
		(wavelet or Hilbert) to each EEG channel independently. Handles 
		both individual epochs and averaged evoked responses.

		Parameters
		----------
		epochs : mne.Epochs
			Preprocessed EEG data segmented into epochs. Can be either 
			individual trial data or averaged evoked responses.

		Returns
		-------
		np.ndarray
			Complex-valued time-frequency representation with shape 
			(n_epochs, n_frequencies, n_channels, n_times). Contains 
			the analytic signal for each trial, frequency, channel, 
			and time point.

		Notes
		-----
		The method supports two decomposition approaches:
		
		- **Wavelet convolution**: Uses precomputed Morlet wavelets 
		  for frequency-domain convolution
		- **Hilbert transform**: Applies bandpass filtering followed 
		  by Hilbert transform for each frequency band
		
		Memory efficiency is optimized by converting output to 
		complex64 precision and processing channels sequentially.
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
		Save TFR results to MNE-Python compatible format.

		Converts internal TFR dictionary format to MNE AverageTFR 
		objects and saves each condition as a separate .h5 file. 
		Handles proper axis ordering and metadata preservation for 
		MNE compatibility.

		Parameters
		----------
		tfr : dict
			TFR results dictionary containing 'power', 'times', 
			'frex', 'ch_names', and 'cnd_cnt' keys with 
			condition-specific power arrays.
		epochs : mne.Epochs
			Original EEG epochs object containing channel information 
			and metadata for constructing MNE objects.
		tfr_name : str
			Base filename for saved TFR files. Each condition will be 
			saved as '{tfr_name}_{condition}-tfr.h5'.

		Notes
		-----
		Files are saved in the project's TFR directory structure 
		using the folder_tracker system. Each condition generates 
		a separate MNE AverageTFR file for independent loading and 
		analysis.

		The method handles axis reordering to match MNE's expected 
		format: (..., n_channels, n_frequencies, n_times).
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
								f'{tfr_name}-tfr.h5')
			tfr_.save(f_name, overwrite = True)	def baseline_tfr(self,tfr:dict,base:dict,method:str,
		 			elec_oi:str='all') -> dict:
		"""
		Apply baseline correction to time-frequency power data.

		Performs baseline normalization using various correction 
		methods. Supports both trial-specific and condition-averaged 
		baseline correction strategies with decibel conversion for 
		statistical analysis and visualization.

		Parameters
		----------
		tfr : dict
			TFR results dictionary containing condition-specific power 
			arrays with shape (n_epochs, n_frequencies, n_channels, 
			n_times) and associated metadata (times, channel names, 
			etc.).
		base : dict
			Baseline power values for each condition. Format varies by 
			correction method:
			
			- 'trial_spec': (n_epochs, n_frequencies, n_channels) 
			   per condition
			- 'cnd_spec'/'cnd_avg': (n_frequencies, n_channels) 
			   per condition
		method : {'trial_spec', 'cnd_spec', 'cnd_avg', 'norm', None}
			Baseline correction strategy:
			
			- 'trial_spec': Apply baseline correction to individual 
			  trials, then average across trials
			- 'cnd_spec': Average trials first, then apply 
			  condition-specific baseline correction
			- 'cnd_avg': Average trials first, then apply grand-average 
			  baseline correction across all conditions
			- 'norm': Apply normalization procedure 
			  (electrode-dependent)
			- None: Simple trial averaging without baseline correction
		elec_oi : str or list, default='all'
		    # TODO: check this
			Electrode specification. Required for normalization methods 
			that depend on electrode topography ('norm', 'Z'). Usually 
			matches the electrode selection used in analysis.

		Returns
		-------
		dict
			Modified TFR dictionary with baseline-corrected power 
			values.  Trial dimension is averaged out, resulting in power 
			arrays with shape (n_frequencies, n_channels, n_times) 
			per condition.

		Raises
		------
		ValueError
			If an unsupported baseline correction method is specified.

		Notes
		-----
		**Correction Methods:**
		
		- **Condition-specific ('cnd_spec')**: Standard approach 
		  recommended by Cohen (2014). Averages trials within each 
		  condition first, then applies baseline correction using that 
		  condition's own baseline period. Balances statistical validity 
		  with computational efficiency.
		- **Trial-specific ('trial_spec')**: Alternative approach that 
		  corrects each individual trial before averaging. May introduce 
		  artifacts and is generally not recommended as the primary 
		  method.
		- **Condition-averaged ('cnd_avg')**: Uses grand-average 
		  baseline across all conditions. Useful for between-condition 
		  comparisons when conditions have similar baseline 
		  characteristics.
		- **Normalization ('norm')**: Specialized electrode-dependent 
		  normalization (implementation-specific).

		**Output Format:**
		The method modifies the TFR dictionary in-place and adds 
		trial count information ('cnd_cnt') for each condition to 
		track statistical degrees of freedom.

		See Also
		--------
		db_convert : Decibel conversion implementation
		"""

		cnds = list(tfr['power'].keys())
		
		if method == 'cnd_avg':
			cnd_avg = np.mean(np.stack([base[cnd] for cnd in cnds]), axis = 0)

		for cnd in cnds:
			# Shape: (epochs X nr_freq X nr_ch X nr_time)
			power = tfr['power'][cnd]  
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
				avg_power = np.mean(power, axis=0)  
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
		Convert power to decibels relative to baseline.
		
		Performs decibel transformation [10 * log10(power / baseline)] 
		with automatic detection of baseline type based on input 
		dimensions. Handles both trial-specific and condition-averaged 
		baselines.
		
		Parameters
		----------
		power : np.ndarray
			Time-frequency power data with shapes:
			- 4D (n_epochs, n_frequencies, n_channels, n_times): 
			  individual trials
			- 3D (n_frequencies, n_channels, n_times): 
			  trial-averaged power
		base_power : np.ndarray  
			Baseline power data with shapes:
			- 2D (n_frequencies, n_channels): condition-averaged 
			  baseline
			- 3D (n_epochs, n_frequencies, n_channels): trial-specific 
			  baseline
			
		Returns
		-------
		np.ndarray
			Baseline-normalized power in decibels with same shape as 
			input power.Values represent power change relative to 
			baseline:
			- 0 dB: power equals baseline
			- Positive dB: power exceeds baseline  
			- Negative dB: power below baseline
			
		Raises
		------
		ValueError
			If base_power dimensions are not 2D or 3D.
			
		Notes
		-----
		The conversion uses the formula: 10 * log10(power / baseline)
		
		Small epsilon values (1e-12) are added to prevent log(0) errors.
		Baseline arrays are automatically expanded along the time 
		dimension to match the power array shape.
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

	#TODO: update fucnction (make it more general)
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
		"""
		Calculate the exponent of the next higher power of 2.

		Determines the smallest integer n such that 2^n >= i. This is 
		commonly used in FFT algorithms to find the optimal buffer size 
		for frequency-domain convolution operations.

		Parameters
		----------
		i : int
			Input value for which to find the next power of 2 exponent.

		Returns
		-------
		int
			Exponent n such that 2^n >= i and 2^(n-1) < i.

		Notes
		-----
		This function is used internally for determining optimal FFT 
		buffer sizes during wavelet convolution. Powers of 2 are 
		preferred for FFT efficiency.
		"""

		n = 1
		while 2**n < i: 
			n += 1
		
		return n





 	