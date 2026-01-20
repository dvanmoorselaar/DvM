"""
Channel Tuning Function (CTF) Analysis for EEG Data.

This module implements spatial encoding analysis methods based on the 
inverted encoding model (IEM) framework for analyzing neural population 
responses. The CTF approach reconstructs spatial tuning functions from 
multivariate EEG patterns, enabling investigation of spatial working 
memory representations and attentional mechanisms.

The implementation is modeled after Foster et al. (2015) "The topography 
of alpha-band activity tracks the content of spatial working memory" and 
extends the BDM (Brain Decoding Multivariate) framework with spatial 
encoding capabilities.

Classes
-------
CTF : 
    Main class for channel tuning function analysis, inheriting from 
	BDM. Provides methods for spatial encoding analysis, basis function 
    reconstruction, and topographical visualization of spatial tuning.

Notes
-----
The CTF analysis pipeline includes:
1. Spatial basis function construction (von Mises functions)
2. Encoding model training on spatial location data  
3. Channel response reconstruction and tuning curve estimation
4. Statistical analysis of spatial selectivity and precision

References
----------
Foster, J. J., Sutterer, D. W., Serences, J. T., Vogel, E. K., & Awh, E. 
(2015). The topography of alpha-band activity tracks the content of 
spatial working memory. Journal of Neurophysiology, 115(1), 168-177.

Examples
--------
Basic CTF analysis:

>>> from eeg_analyses.CTFtemp import CTF
>>> ctf = CTF(sj=1, epochs=epochs, df=df, 
...           to_decode='target_position', nr_bins=8, nr_chans=8)
>>> ctf.spatial_ctf(cnds={'condition': ['spatial_task']})

Created by Dirk van Moorselaar on 30-03-2016.
Copyright (c) 2016 DvM. All rights reserved.
"""

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
from analysis.BDM import *  
from support.FolderStructure import *
from matplotlib import cm
from math import pi, sqrt
from mne.filter import filter_data
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from visualization.plot import plot_ctf_timecourse

from support.preprocessing_utils import select_electrodes, trial_exclusion, \
							get_time_slice,baseline_correction, format_subject_id 
from IPython import embed

class CTF(BDM):
	"""
	Channel Tuning Function (CTF) analysis for spatial encoding in 
	EEG data.
	
	Implements inverted encoding models (IEM) to reconstruct spatial
	tuning functions from multivariate EEG patterns. This class extends 
	the BDM framework to analyze how neural populations encode spatial
	information, enabling investigation of spatial working memory and 
	attention mechanisms.
	
	The analysis follows the approach of Foster et al. (2015), using 
	basis functions to model hypothetical spatial channels and 
	reconstructing channel responses from observed neural activity 
	patterns.

	Parameters
	----------
	sj : int
		Subject identifier for analysis organization.
	epochs : mne.Epochs
		Epoched EEG data containing the neural responses to be analyzed.
	df : pd.DataFrame
		Behavioral data containing trial information, including the 
		spatial locations to decode (specified in `to_decode` column).
	to_decode : str
		Column name in behavioral DataFrame containing spatial location 
		information for each trial (e.g., 'target_location', 
		'cue_position').
	nr_bins : int
		Number of spatial location bins used in the experiment (e.g., 8 
		for 8 possible target positions around a circle).
	nr_chans : int
		Number of hypothetical underlying spatial channels in the 
		encoding model. Determines the resolution of spatial 
		reconstruction.
	shift_bins : int, default=0
		Circular shift applied to align tuning curves around a specific 
		location. Useful for centering analysis around locations with 
		specific experimental manipulations (e.g., higher target 
		probability) that vary across subjects. For example, if the 
		"special" location is at position 2 in one subject and position 
		6 in another, shift_bins allows alignment relative to this
		location for group-level analysis.
	nr_iter : int, default=10
		Number of iterations for forward model application and 
		averaging.
	nr_folds : int, default=3
		Number of cross-validation folds for model training and testing.
	elec_oi : str or list, default='all'
		Electrode selection criteria:
		- 'all': Use all available electrodes in ch_names
		- 'posterior': Posterior electrodes (parietal, occipital 
		   regions)
		- 'frontal': Frontal electrodes (prefrontal, frontal regions)
		- 'central': Central electrodes (motor, somatosensory regions)
		- list: Specific electrode names to select
	downsample : int, default=128
		Target sampling frequency for EEG data in Hz. Data will be 
		downsampled to this rate if necessary.
	sin_power : int, default=7
		Power parameter for von Mises basis functions, controlling the
		width of spatial tuning curves. Higher values create narrower
		tuning.
	delta : bool, default=False
		Whether to use delta functions (True) or shaped basis functions
		(False) for spatial channels. Delta functions assume no specific
		tuning shape.
	method : str, default='Foster'
		#TODO: implement cross-validation method
		Analysis method to use. 'Foster' implements the Foster et al.
		(2015) approach.
	avg_ch : bool, default=True
		Whether to average across reconstructed channels for summary
		statistics.
	ctf_param : str or bool, default='slopes'
		Parameter extraction method for reconstructed CTFs. Options:
		- 'slopes': Extract only slope parameters (fastest)
		- 'von_mises': Extract slopes + von Mises fitting parameters 
		  (amplitude, baseline, concentration, mean location)
		- 'gaussian': Extract slopes + Gaussian fitting parameters
		  (amplitude, mean, sigma, with derived concentration measure)
		- False: No parameter extraction
	min_freq : int, default=4
		Minimum frequency for time-frequency analysis (Hz).
	max_freq : int, default=40
		Maximum frequency for time-frequency analysis (Hz).
	num_frex : int, default=25
		Number of frequency steps for time-frequency decomposition.
	freq_scaling : str, default='log'
		Frequency spacing: 'log' for logarithmic, 'linear' for linear.
	slide_window : int, default=0
		Sliding window size for temporal smoothing (samples). 0 disables
		smoothing.
	laplacian : bool, default=False
		Whether to apply Laplacian spatial filtering to enhance spatial
		specificity of EEG signals.
	pca_cmp : int, default=0
		Number of PCA components for dimensionality reduction. 
		0 disables PCA.
	filter : int, optional
		Lowpass filter cutoff frequency (Hz) applied to broadband EEG 
		data. Only used when analysis is run on broadband signals 
		(freqs='broadband' in spatial_ctf). Recommended range: 6-8 Hz 
		for optimal CTF results.None disables filtering.
	report : bool, default=False
		Whether to generate automated analysis reports with
		visualizations.
	baseline : tuple, optional
		Time window for baseline correction (start, end) in seconds.
		None disables baseline correction.
	seed : int or bool, default=42213
		Random seed for reproducible results. False disables seeding.

	Attributes
	----------
	sfreq : float
		Actual sampling frequency extracted from epochs.info['sfreq'].
	basisset : np.ndarray
		Computed spatial basis functions, created during analysis.

	Notes
	-----
	The CTF analysis pipeline involves several key steps:
	
	1. **Basis Function Construction**: Create spatial basis functions
	   (von Mises or delta) representing hypothetical neural channels
	   
	2. **Forward Model Training**: Learn mapping from channel responses
	   to observed EEG patterns using training data
	   
	3. **Channel Response Reconstruction**: Apply inverted model to 
	   reconstruct channel responses from test data
	   
	4. **Spatial Tuning Analysis**: Extract tuning parameters and
	   assess spatial selectivity
	
	The method assumes that EEG activity reflects weighted sums of
	underlying spatial channel responses, enabling reconstruction of
	population-level spatial representations.

	References
	----------
	Foster, J. J., Sutterer, D. W., Serences, J. T., Vogel, E. K., & 
	Awh, E. (2015). The topography of alpha-band activity tracks the 
	content of spatial working memory. Journal of Neurophysiology, 
	115(1), 168-177.

	Examples
	--------
	Basic spatial CTF analysis:
	
	>>> from eeg_analyses.CTFtemp import CTF
	>>> ctf = CTF(sj=1, epochs=epochs, df=df, 
	...           to_decode='target_position', 
	...           nr_bins=8, nr_chans=8)
	>>> ctf.spatial_ctf(cnds={'condition': ['spatial_task']})
	
	Time-frequency CTF analysis:
	
	>>> ctf_tf = CTF(sj=1, epochs=epochs, df=df,
	...              to_decode='cue_location',
	...              nr_bins=6, nr_chans=6,
	...              power='band', min_freq=8, max_freq=30)
	>>> ctf_tf.spatial_ctf(cnds={'task': ['working_memory']})
	"""

	def __init__(self,sj:int,epochs:mne.Epochs,df:pd.DataFrame, 
				to_decode:str,nr_bins:int,nr_chans:int,shift_bins:int=0,
				nr_iter:int=10,nr_folds:int=3,elec_oi:Union[str,list]='all',
				downsample:int=128,sin_power:int=7,delta:bool=False,
				method:str='Foster',avg_ch:bool=True,
				ctf_param:Union[str,bool]='slopes',power:str='band',
				min_freq:int=4,max_freq:int=40,num_frex:int=25,
				freq_scaling:str='log',slide_window:int=0,
				laplacian:bool=False,pca_cmp:int=0,
				filter:int=None,VEP:bool=False,report:bool=False,
				baseline:Optional[tuple]=None,seed:Union[int, bool] = 42213):
		"""
		Initialize CTF analysis object with experimental parameters.
		
		Sets up the channel tuning function analysis framework with 
		specified spatial encoding parameters, preprocessing options,
		and analysis configuration.

		Notes
		-----
		The initialization configures all analysis parameters and 
		validates sampling rate compatibility. The object inherits BDM 
		functionality while adding CTF-specific spatial encoding 
		capabilities.
		
		All parameters are documented in the class docstring above.
		"""

		# TODO: change moment to filter

		self.sj = sj
		self.df = df
		self.epochs = epochs
		self.to_decode = to_decode
		self.elec_oi = elec_oi
		self.ctf_param = ctf_param
		self.avg_ch = avg_ch
		self.cross = False
		self.baseline = baseline
		self.seed = seed
		self.shift_bins = shift_bins		# shift of channel position

		# specify model parameters
		self.sfreq = epochs.info['sfreq']
		if self.sfreq % downsample != 0:
			warnings.warn(f"Warning: sfreq ({self.sfreq}) is not evenly "
				 f"divisible by downsample ({downsample}). This will result in"
                 f" inexact downsampling. Consider using a downsample value "
                 f"that evenly divides {self.sfreq} ")
		self.downsample = downsample
		self.method = method
		self.nr_bins = nr_bins													
		self.nr_chans = nr_chans 												
		self.nr_iter = nr_iter														
		self.nr_folds = nr_folds												
		self.sin_power = sin_power
		self.delta = delta												
		self.min_freq = min_freq
		self.max_freq = max_freq
		self.num_frex = num_frex
		self.freq_scaling = freq_scaling
		self.slide_wind = slide_window
		self.laplacian = laplacian
		self.pca=pca_cmp
		self.filter = filter	
		self.report = report
		
	def calculate_basis_set(self,nr_bins:int,nr_chans:int, 
			 				sin_power:int=7,delta:bool=False)->np.array:
		"""
		Calculate spatial basis functions for channel tuning function 
		analysis.
		
		Creates a set of hypothetical spatial channel response functions 
		that serve as basis functions for the inverted encoding model. 
		Each basis function represents the response profile of a spatial 
		channel across different angular locations.

		Parameters
		----------
		nr_bins : int
			Number of spatial location bins in the experiment (e.g., 
			8 for 8 target positions arranged in a circle).
		nr_chans : int
			Number of hypothetical spatial channels to model. Typically
			equal to nr_bins for full spatial coverage.
		sin_power : int, default=7
			Power parameter for sinusoidal basis functions. Higher 
			values create narrower spatial tuning curves. Used when 
			delta=False.
		delta : bool, default=False
			Type of basis function to use:
			- False: Shaped functions (half sinusoid raised to 
			sin_power)
			- True: Delta functions (single bin = 1, others = 0)

		Returns
		-------
		np.ndarray
			Basis function matrix with shape (nr_chans, nr_bins).
			Each row represents one spatial channel's response profile
			across all spatial locations.

		Notes
		-----
		The basis functions model hypothetical neural populations with
		spatial selectivity. The choice between delta and shaped 
		functions affects the assumptions about spatial tuning:
		
		- **Shaped functions**: Assume smooth, overlapping tuning curves
		  following R = sin(θ/2)^sin_power, where θ is angular distance
		  from preferred location.
		  
		- **Delta functions**: Assume sharp, non-overlapping tuning with
		  channels responding only to their preferred location.
		
		Each basis function is circularly shifted so that channels have
		different preferred locations evenly distributed across space.

		Examples
		--------
		Create smooth basis functions for 8 locations and 8 channels:
		
		>>> ctf = CTF(...)
		>>> basis = ctf.calculate_basis_set(nr_bins=8, nr_chans=8, 
		...                                sin_power=7, delta=False)
		>>> basis.shape  # (8, 8)
		
		Create delta function basis set:
		
		>>> basis_delta = ctf.calculate_basis_set(nr_bins=6, nr_chans=6,
		...                                      delta=True)
		>>> np.sum(basis_delta, axis=1)  # Each channel sums to 1
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
	
	def generate_ctf_report(self,ctf:dict,ctf_param,freqs:Union[str, dict],
						 info:mne.Info,
						 report_name: str):
		"""
		Generate comprehensive HTML report for CTF analysis results.
		
		Creates an automated report containing visualizations of channel
		tuning functions, slope parameters over time, and frequency-specific
		analyses. The report includes both time-domain and time-frequency
		representations of spatial encoding results.

		Parameters
		----------
		ctf : dict
			Dictionary containing CTF reconstruction results for each
			condition. Keys are condition names, values contain channel
			response matrices and metadata.
		ctf_param : dict
			#TODO: Clarify structure and contents of ctf_param
			Dictionary containing extracted CTF parameters (e.g., slopes,
			peaks) for statistical analysis.
		freqs : str or dict
			Frequency specification for analysis:
			- str: Frequency band name (e.g., 'alpha')  
			- dict: Custom frequency definitions
		info : mne.Info
			MNE info object containing channel and sampling information
			for proper visualization setup.
		report_name : str
			Base name for the output HTML report file. Will be saved
			in the CTF report folder with .html extension.

		Notes
		-----
		The report automatically includes:
		
		1. **CTF Slope Timecourses**: Parameter evolution over time
		2. **Channel Response Functions**: Reconstructed spatial tuning
		3. **Time-Frequency Plots**: If multiple frequencies analyzed
		4. **Condition Comparisons**: Separate sections per condition
		
		Report files are saved in HTML format for easy viewing and sharing.
		Figures are embedded directly in the report for self-contained
		documentation.

		Examples
		--------
		Generate report after CTF analysis:
		
		>>> # Run CTF analysis
		>>> ctf_results, ctf_params = ctf.spatial_ctf(...)
		
		>>> # Create comprehensive report  
		>>> ctf.generate_ctf_report(ctf_results, ctf_params, 
		...                        freqs={'alpha': [8, 12]},
		...                        info=epochs.info,
		...                        report_name='subject_01_spatial')
		"""
		
		report_name = self.folder_tracker(['ctf', 'report'],
										f'{report_name}.h5')
		
		report = mne.Report(title='Single subject ctf overview')

		for cnd in ctf.keys():
			if cnd == 'info':
				continue

			section = "Condition: " + cnd	
			#TODO: add section after update mne (so that cnd info is displayed)
			# get ctf slope
			output = [f'{d}_slopes' for d in ['envelope','voltage','T','E'] 
			 						if f'{d}_slopes' in ctf_param[cnd].keys()]

			if len(freqs) == 1:
				fig, ax = plt.subplots()
				plot_ctf_timecourse(ctf_param,cnds = [cnd],
											colors=['black']
											,timecourse = '1d',
											output=output,stats=False)
				report.add_figure(fig,
							title = f'CTF slope over time: {cnd}',
							caption = f'CTF slope for {cnd} condition')
				plt.close()
			else:
				for out in output:
					fig, ax = plt.subplots()
					plot_ctf_timecourse(ctf_param,cnds = [cnd],
											colors=['black']
											,timecourse = '2d_tfr',
											output=out,stats=False)
					report.add_figure(fig,
						title = f'CTF slope over time: {cnd}, {out}',
						caption = f'CTF slope for {cnd} condition')
					plt.close()

			# get ctf tuning functions
			#TODO: add that multiple frequencies can be plotted in a list
			for to in ['E','T','voltage','envelope']:
				if f'C2_{to}' in ctf[cnd].keys():
					fig, ax = plt.subplots()
					plot_ctf_timecourse(ctf,cnds = [cnd],
										colors=['black'],timecourse = '2d_ctf',
										output=f'C2_{to}',stats=False)
					report.add_figure(fig,
						title = f'Channel response {to}: {cnd}',
						caption = f'CTF tuning function over time')
					plt.close()
			
		report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html', overwrite=True)

	def spatial_ctf(self, pos_labels: dict = 'all', cnds: dict = None,
					excl_factor: dict = None, window_oi: tuple = (None, None),
					freqs: dict = 'main_param', GAT: bool = False, 
					nr_perm: int = 0, collapse: bool = False, name: str = 'main'):
		"""
		Perform spatial channel tuning function analysis across 
		conditions.
		
		This is the main analysis function that implements the complete 
		CTF pipeline using inverted encoding models to reconstruct 
		spatial channel responses from multivariate EEG patterns. The 
		method follows Foster et al. (2015) and supports both 
		within-condition and cross-condition training/testing paradigms
		with comprehensive parameter extraction capabilities.

		Parameters
		----------
		pos_labels : dict or str, default='all'
			Spatial position labels for CTF analysis:
			- 'all': Use all unique values from the to_decode column
			- dict: {column_name: [list_of_positions]} to specify subset
			  of positions for analysis
		cnds : dict, optional
			Condition specifications for decoding analysis. Structure 
			determines analysis type:
			
			Within-condition decoding:
				``{'condition_column': ['cond1', 'cond2']}``
			
			Cross-condition decoding (train on first, test on second):
				``{'condition_column': [['train_cond'], 'test_cond']}``
			
			Multiple training/testing conditions:
				``{'condition_column': [['train1', 'train2'], 
				                        ['test1', 'test2']]}``
			
			If None, decoding performed on all data.
		excl_factor : dict, optional
			Exclusion criteria for trial selection. Format:
			{column_name: [values_to_exclude]}. Multiple columns and
			values can be specified to filter out unwanted trials.
		window_oi : tuple, default=(None, None)
			Time window of interest (start_time, end_time) in seconds. 
			Data will be cropped to this window after preprocessing. 
			None values preserve the full epoch duration.
		freqs : dict or str, default='main_param'
			Frequency specification for time-frequency analysis:
			- 'main_param': Use frequency settings from class 
			   initialization
			- dict: {band_name: [low_freq, high_freq]} for custom 
			  frequency bands
			- 'broadband': Analyze unfiltered broadband signals only
		GAT : bool, default=False
			Whether to perform Generalization Across Time (GAT) 
			analysis. When True, creates encoding matrices for all 
			possible train/test time combinations. 
			Warning: Computationally intensive.
		nr_perm : int, default=0
			Number of permutation tests for statistical validation.
			Set to 0 to disable permutation testing and speed up
			analysis.
		collapse : bool, default=False
			Whether to collapse across specific conditions or factors.
			Note: This feature is currently under development.
		name : str, default='main'
			Analysis name identifier used for file naming and 
			organization. Results will be saved with this name as 
			suffix.

		Returns
		-------
		None
		Function performs analysis and saves results to pickle 
		files:
		- 'ctfs_{name}.pickle': Reconstructed channel tuning 
		   functions. For filtered data: contains 'C2_E' 
		   (evoked power) and 'C2_T' (total power). For broadband: 
		   contains 'C2_envelope' (amplitude) and 'C2_voltage' 
		   (raw ERPs).
		- 'ctf_param_{name}.pickle': Extracted tuning parameters 
		  (slopes, von Mises fits, Gaussian fits)
		- 'ctf_info_{name}.pickle': Analysis metadata and indices
		
		If self.report=True, also generates an HTML report with 
		visualizations of the results.		Notes
		-----
		The spatial CTF analysis pipeline implements the following 
		steps:
		
		1. **Data Preparation**: 
		   - Select electrodes and apply preprocessing
		   - Filter trials by conditions and exclusion criteria
		   
		2. **Basis Function Setup**: 
		   - Create spatial channel models using von Mises functions
		   - Configure cross-validation folds for robust estimation
		   
		3. **Time-Frequency Analysis**:
			- For filtered data: Apply bandpass filtering, extract 
			evoked power (phase-locked) and total power 
			(non-phase-locked)
			- For broadband: Extract raw voltages (ERPs) and amplitude 
			envelope via Hilbert transform
		
		4. **Cross-Validation Loop**:
			- Partition data into training and testing sets
			- Train encoding models on spatial location data
			- Test model performance on held-out trials		
		
		5. **Channel Reconstruction**: 
			- Use forward modeling to reconstruct channel responses
			- Generate tuning functions across spatial positions
		   
		6. **Parameter Extraction**:
		   - Extract tuning curve slopes using linear regression
		   - Optionally fit von Mises functions to estimate peak tuning
		   - Optionally fit Gaussian functions for comparison
		   
		7. **Output Generation**:
		   - Save reconstructed tuning functions and parameters
		   - Generate analysis report with visualizations
		
		The analysis supports multiple frequency bands, cross-condition 
		decoding, and comprehensive statistical validation through 
		permutation testing.
		
		References
		----------
		Foster, J. J., Sutterer, D. W., Serences, J. T., Vogel, E. K., 
		& Awh, E. (2015). The topography of alpha-band activity tracks 
		the content of spatial working memory. Journal of 
		Neurophysiology, 115(1), 168-177.
		
		Examples
		--------
		Basic analysis with default settings:
		
		>>> ctf = CTF(sj=1, to_decode='target_loc')
		>>> ctf.spatial_ctf()
		
		Cross-condition analysis:
		
		>>> conditions = {'condition': [['easy'], 'hard']}
		>>> ctf.spatial_ctf(cnds=conditions, name='cross_difficulty')
		
		Multi-frequency analysis with custom bands:
		
		>>> freq_bands = {'alpha': [8, 12], 'beta': [13, 30]}
		>>> ctf.spatial_ctf(freqs=freq_bands, name='multi_band')
		"""
		
		# Input validation and setup

		# hypothesized set tuning functions underlying power measured 
		# across electrodes
		print('Creating bassiset with sin_power ', self.sin_power )
		self.basisset = self.calculate_basis_set(self.nr_bins, self.nr_chans,
													self.sin_power,self.delta)
		# read in data
		if cnds is None:
			(cnd_head,_) = (None,['all_data'])
		else:
			(cnd_head,_), = cnds.items()
		headers = [cnd_head]
		epochs = self.epochs.copy()
		df = self.df.copy()
		freqs, nr_freqs, bands = self.set_frequencies(freqs)
		data_type = 'power' if freqs != ['broadband'] else 'broadband'
		epochs, df = self.select_ctf_data(epochs, df,self.elec_oi, 
											headers, excl_factor, data_type)

		# set params
		sfreq = epochs.info['sfreq']
		downsample = int(sfreq // self.downsample)
		actual_sfreq = sfreq / downsample
		if abs(actual_sfreq - self.downsample) > 1:
			print(f"Warning: Actual sampling frequency ({actual_sfreq}) "
		 	"is not equal to desired downsample ({self.downsample}).")	

		nr_itr = self.nr_iter * self.nr_folds
		ctf_name = f'sub_{self.sj}_{name}'
		nr_perm += 1
		nr_elec = len(epochs.ch_names)
		tois = get_time_slice(epochs.times, window_oi[0], window_oi[1])
		times_oi = epochs.times[tois]
		nr_samples = epochs.times[tois][::downsample].size
		ctf, info = {}, {}
		
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
			train_cnds, test_cnds = self.check_cnds_input(cnds)

			if test_cnds is not None:
				# check for overlap between train and test conditions
				overlap = set(train_cnds) & set(test_cnds)
				if overlap:
					print(f"Warning: Found overlapping conditions between "
						  f"train and test: {overlap}. Training will be done "
						  f"on a subset of the data to match trial counts.")
					self.cross = 'cross_cv'
				else:
					self.cross = True
				nr_itr = self.nr_iter
			else:
				self.cross = False
		else:
			train_cnds = ['all_data']
			test_cnds = None
			
		# based on conditions get position bins
		(pos_bins, 
		cnds, 
		epochs, 
		max_tr) = self.select_ctf_labels(epochs, df, pos_labels, cnds)

		if GAT:
			print('Creating generalization across time matrix.',
	 			'This may take some time and is generally not ',
				'recommended')
			
		# Frequency loop (ensures that data is only filtered once)
		cnd_combos = []
		for fr in range(nr_freqs):
			print('Frequency {} out of {}'.format(str(fr + 1), str(nr_freqs)))

			# Time-Frequency Analysis
			E, T = self.tfr_decomposition(epochs.copy(),freqs[fr],
								        tois,downsample)

			# Loop over conditions
			for c, cnd in enumerate(train_cnds):
				for te_cnd in (test_cnds if test_cnds is not None else [None]):
					# set condition info
					cnd_inf = str(cnd)
					if self.cross:
						cnd_inf += f'_{te_cnd}'
					cnd_combos.append(cnd_inf)
					print(f'Running ctf for {cnd_inf} condition')

					# preallocate arrays
					if GAT:
						C2_E = np.zeros((nr_perm,nr_freqs, nr_itr,
										nr_samples-self.slide_wind,
										nr_samples-self.slide_wind,
										self.nr_bins, self.nr_chans))
						W_E = np.zeros((nr_perm,nr_freqs, nr_itr,
										nr_samples-self.slide_wind, 
										nr_samples-self.slide_wind,
										self.nr_chans, 
										nr_elec))						
					else:
						C2_E = np.zeros((nr_perm,nr_freqs, nr_itr,
										nr_samples-self.slide_wind,
										self.nr_bins, self.nr_chans))
						W_E = np.zeros((nr_perm,nr_freqs, nr_itr,
										nr_samples-self.slide_wind, 
										self.nr_chans, nr_elec))							 
					C2_T, W_T  = C2_E.copy(), W_E.copy()				 
	
					# partition data into training and testing sets
					# is done once to ensure each frequency has the same sets
					if fr == 0:
						# update ctf dicts to keep track of output	
						info.update({cnd_inf:{}})
						ctf.update({cnd_inf:{'C2_E':C2_E,'C2_T':C2_T,
										'W_E':W_E,'W_T':W_T}})
						# get condition indices
						cnd_idx = cnds == cnd
						if self.cross:
							test_idx = cnds == te_cnd
							test_bins = np.unique(pos_bins[test_idx])
							if self.cross == 'cross_cv':
								trial_limit = max_tr
							else:
								trial_limit = None
							(train_idx, 
							test_idx) = self.train_test_cross(pos_bins, 
										  					cnd_idx,test_idx, 
															self.nr_iter,
															trial_limit)
						else:
							(train_idx, 
							test_idx) = self.train_test_split(pos_bins,
															cnd_idx,max_tr)
							test_bins = np.unique(pos_bins)

						info[cnd_inf]['train_idx'] = train_idx
						info[cnd_inf]['test_idx'] = test_idx
						if self.method == 'Foster':
							C1 = np.empty((self.nr_bins * (self.nr_folds - 1), 
											self.nr_chans)) * np.nan
						else:
							C1 = self.basisset

					# iteration loop
					for itr in range(nr_itr):

						# TODO: insert permutation loop
						p = 0
						train_idx = info[cnd_inf]['train_idx'][itr]

						# initialize evoked and total power arrays
						bin_te_E = np.zeros((self.nr_bins, nr_elec, 
						                                        nr_samples)) 
						bin_te_T = bin_te_E.copy()
						if self.method == 'k-fold':
							pass
							#TODO: implement 
						elif self.method == 'Foster':
							nr_itr_tr = self.nr_bins * (self.nr_folds - 1)
							bin_tr_E = np.zeros((nr_itr_tr, nr_elec, 
							                                    nr_samples)) 
							bin_tr_T = bin_tr_E.copy()
							
						# position bin loop
						bin_cnt = 0
						for bin in range(self.nr_bins):
							if bin in test_bins:
								condition_info = info[cnd_inf]['test_idx']
								iteration_data = condition_info[itr]
								bin_indices = iteration_data[bin]
								test_idx = np.squeeze(bin_indices)
								bin_te_T[bin] = np.mean(T[test_idx], axis = 0)
								evoked_mean = np.mean(E[test_idx], axis=0)
								bin_te_E[bin] = self.extract_power(evoked_mean, 
																   freqs[fr])

							if self.method == 'Foster':
								for j in range(self.nr_folds - 1):
									evoked = self.extract_power(np.mean(\
												E[train_idx[bin][j]],
												axis = 0), freqs[fr])
									bin_tr_E[bin_cnt] = evoked
									total = np.mean(T[train_idx[bin][j]], 
						                                            axis = 0)
									bin_tr_T[bin_cnt] = total
									C1[bin_cnt] = self.basisset[bin]
									bin_cnt += 1
							elif self.method == 'k-fold':
								pass
								#TODO: implement
						
						(ctf[cnd_inf]['C2_E'][p,fr,itr], 
						ctf[cnd_inf]['W_E'][p,fr,itr],
						ctf[cnd_inf]['C2_T'][p,fr,itr],
						ctf[cnd_inf]['W_T'][p,fr,itr]) = (
							self.forward_model_loop(
								bin_tr_E, 
								bin_te_E,
								bin_tr_T, 
								bin_te_T,C1,
								GAT))
		
		# take the average across model iterations
		for cnd in np.unique(cnd_combos):
			for key in ['C2_E','C2_T','W_E','W_T']:
				ctf[cnd][key] = ctf[cnd][key].mean(axis = 2)

			if freqs == ['broadband']:
				ctf[cnd]['C2_envelope'] = ctf[cnd].pop('C2_E')
				ctf[cnd]['W_envelope'] = ctf[cnd].pop('W_E')
				ctf[cnd]['C2_voltage'] = ctf[cnd].pop('C2_T')
				ctf[cnd]['W_voltage'] = ctf[cnd].pop('W_T')

		# save output
		times_oi = epochs.times[tois][::downsample]
		if self.slide_wind > 0:
			times_oi = times_oi[:-self.slide_wind]
		
		print('get ctf tuning params')
		ctf_param = self.get_ctf_tuning_params(ctf,self.ctf_param,
											GAT=GAT,avg_ch=self.avg_ch,
											test_bins=test_bins)
		
		ctf_param.update({'info':{'times':times_oi,
							'freqs':freqs,'bands':bands}})

		if self.ctf_param:
			with open(self.folder_tracker(['ctf',self.to_decode], 
					fname=f'{ctf_name}_param.pickle'),'wb') as handle:
				print('saving ctf params')
				pickle.dump(ctf_param, handle)

		# save ctf data
		print('saving ctf data')
		ctfs = {}
		for cnd in ctf.keys():
			ctfs[cnd] = {}
			for key in ['C2_E','C2_T','C2_envelope','C2_voltage']:
				if key in ctf[cnd].keys():
					# extract non-permuted data
					data = ctf[cnd][key][0]

					if self.avg_ch:
						# average across channels
						data = data.mean(axis = -2)

					ctfs[cnd][key] = data	

		ctfs.update({'info':{'times':times_oi, 'freqs':freqs}})

		# generate report
		if self.report:
			self.generate_ctf_report(ctfs,ctf_param,freqs,info = epochs.info,
									report_name = ctf_name)

		with open(self.folder_tracker(['ctf',self.to_decode], 
				fname = f'{ctf_name}_ctf.pickle'),'wb') as handle:
			print('saving ctfs')
			pickle.dump(ctfs, handle)

		# TODO: add saving of weights and permutations and add to report
		with open(self.folder_tracker(['ctf',self.to_decode], 
				fname = f'{ctf_name}_info.pickle'),'wb') as handle:
			pickle.dump(info, handle)
		
	def select_ctf_data(self,epochs:mne.Epochs,df:pd.DataFrame,
						elec_oi:Union[list, str]= 'all',headers:list = [],
						excl_factor:dict = None,
						data_type:str = 'broadband') -> Tuple[mne.Epochs,
															pd.DataFrame]:
		"""
		Select and preprocess EEG data for CTF analysis.
		
		Filters epochs and behavioral data based on specified criteria,
		applies preprocessing steps (Laplacian filtering if requested),
		and selects electrodes of interest for spatial encoding 
		analysis.

		Parameters
		----------
		epochs : mne.Epochs
			Input epoched EEG data to be processed.
		df : pd.DataFrame
			Behavioral data frame containing trial information that
			corresponds to the epochs.
		elec_oi : list or str, default='all'
			Electrodes of interest for analysis:
			- 'all': Use all available electrodes
			- list: Specific electrode names or selection criteria
		headers : list, default=[]
			#TODO: Clarify headers parameter usage
			Additional column headers for processing.
		excl_factor : dict, optional
			Exclusion criteria for trial selection. Format:
			{column_name: [values_to_exclude]}. Trials matching any
			specified criteria will be removed.
		data_type : str, default='broadband'
			Type of data analysis. When 'broadband', applies lowpass 
			filtering if self.filter is specified. Other values skip 
			the filtering step.

		Returns
		-------
		tuple[mne.Epochs, pd.DataFrame]
			epochs : mne.Epochs
				Processed epochs with selected electrodes and filtered
				trials.
			df : pd.DataFrame
				Corresponding behavioral data with matching trial 
				indices.

		Notes
		-----
		The processing pipeline includes:
		
		1. **Trial Exclusion**: Remove trials based on excl_factor
		   criteria
		2. **Index Reset**: Ensure proper alignment between epochs and 
		   behavioral data
		3. **Lowpass Filtering**: Apply temporal filtering 
		   (if data_type='broadband') to remove high-frequency noise 
		   before spatial processing
		4. **Laplacian Filtering**: Apply current source density if 
		   requested
		5. **Electrode Selection**: Pick specified electrodes for 
		   analysis
		
        Temporal filtering precedes Laplacian 
		transform to ensure clean spatial gradient calculations. 
		Laplacian filtering  enhances spatial specificity by computing 
		current source density, which can improve CTF spatial 
		selectivity.

		#TODO: check whether trial averaging makes sense for broadband 
		ctf

		Examples
		--------
		Basic data selection:
		
		>>> epochs_clean, df_clean = ctf.select_ctf_data(
		...     epochs, df, elec_oi='posterior'
		... )
		
		With trial exclusion:
		
		>>> epochs_filt, df_filt = ctf.select_ctf_data(
		...     epochs, df, 
		...     elec_oi=['Pz', 'POz', 'Oz'],
		...     excl_factor={'accuracy': [0], 'rt': ['>2000']}
		... )
		"""

		# if specified remove trials matching specified criteria
		if excl_factor is not None:
			df, epochs,_ = trial_exclusion(df, epochs, excl_factor)

		# if not already done reset index (to properly align beh and epochs)
		df.reset_index(inplace = True, drop = True)

		# Apply lowpass filtering before Laplacian to remove high-freq noise
		if data_type == 'broadband':
			if self.filter is not None:
				epochs.filter(l_freq = None, h_freq = self.filter)

		if self.laplacian:
			epochs = mne.preprocessing.compute_current_source_density(epochs)

		# if specified # average across trials
		#TODO: check whether this makes sense for ctf		
		# (epochs, 
		# df) = self.average_trials(epochs,df,[self.to_decode] + headers) 

		# limit analysis to electrodes of interest
		picks = select_electrodes(epochs, elec_oi) 
		epochs.pick(picks)

		return epochs, df

	def check_cnds_input(self, cnds:dict
					  )-> Tuple[list, Union[list, str, None]]:
		"""
		Parse conditions dictionary for within vs cross-condition 
		analysis.

		Parameters
		----------
		cnds : dict
			Conditions specification. 
			   Format: {column_name: condition_values}
			- For within-condition: condition_values is list of 
			   conditions
			- For cross-condition: condition_values is 
			   [train_conditions, test_conditions]

		Returns
		-------
		train_cnds : list
			Conditions for model training.
		test_cnds : list, str, or None
			Conditions for model testing. None indicates 
			within-condition analysis.
		"""
		
		(_, cnds_oi), = cnds.items()
		if type(cnds_oi[0]) == list:
			train_cnds, test_cnds = cnds_oi
		else:
			train_cnds, test_cnds = cnds_oi, None

		return train_cnds, test_cnds

	def select_ctf_labels(self, epochs: mne.Epochs, df: pd.DataFrame, 
						pos_labels: Union[dict, str], 
						cnds: Optional[dict]
						) -> Tuple[np.ndarray, np.ndarray,mne.Epochs, int]:
		"""
		Select and filter epochs and behavioral data based on position 
		and condition criteria.
		
		Filters trials to include only specified spatial positions and 
		experimental conditions, ensuring compatibility with CTF 
		analysis requirements. Also enforces that position bins don't 
		exceed the number specified during class initialization.

		Parameters
		----------
		epochs : mne.Epochs
			Input epochs object containing EEG data.
		df : pd.DataFrame
			Behavioral dataframe with trial information and labels.
		pos_labels : dict or str
			Position specification for CTF analysis:
			- dict: {column_name: [position_values]} to select specific 
			   positions
			- 'all': Use all positions from self.to_decode column
		cnds : dict or None
			Condition specification:
			- dict: {column_name: [condition_values]} to select specific 
			   conditions  
			- None: Include all trials regardless of condition

		Returns
		-------
		pos_bins : np.ndarray
			Array of position labels for selected trials.
		cnds : np.ndarray  
			Array of condition labels for selected trials.
		epochs : mne.Epochs
			Filtered epochs object containing only selected trials.
		max_tr : int
			Maximum number of trials per position bin that can be used
			for balanced cross-validation across conditions.
		"""

		# extract conditions of interest
		if cnds is None:
			cnds = np.array(['all_data']*df.shape[0])
			cnd_idx = np.arange(df.shape[0])
		else:
			(cnd_header, cnds_oi), = cnds.items()
			cnd_idx = np.where(df[cnd_header].isin(np.hstack(cnds_oi)))[0]
			cnds = df[cnd_header]
		
		# extract position labels of interest
		if pos_labels != 'all':
			(pos_header, pos_labels), = pos_labels.items()
			pos_idx =  np.where(df[pos_header].isin(pos_labels))[0]
		else:
			pos_header = self.to_decode
			pos_idx = np.arange(df.shape[0]) 
		
		# get data of interest
		idx = np.intersect1d(cnd_idx, pos_idx)
		pos_bins = df[pos_header].values[idx]
		cnds = cnds[idx]
		epochs = epochs[idx]
		pos_bins, cnds, epochs = self.select_bins_oi(pos_bins, cnds, epochs)
		max_tr = self.set_max_trial(cnds, pos_bins, self.method)

		return pos_bins, cnds, epochs, max_tr

	def select_bins_oi(self, pos_bins: np.ndarray, cnds: np.ndarray, 
					  epochs: mne.Epochs) -> Tuple[np.ndarray, 
					  							  np.ndarray, mne.Epochs]:
		"""
		Filter out position bins that exceed the valid range for 
		CTF analysis.
		
		Ensures that all position labels are within the valid range 
		[0, nr_bins) as specified during class initialization, 
		removing any invalid trials.

		Parameters
		----------
		pos_bins : np.ndarray
			Array of position labels for trials.
		cnds : np.ndarray
			Array of condition labels for trials.
		epochs : mne.Epochs
			EEG epochs object containing trial data.

		Returns
		-------
		pos_bins : np.ndarray
			Filtered position labels with only valid bins.
		cnds : np.ndarray
			Filtered condition labels for corresponding trials.
		epochs : mne.Epochs
			Filtered epochs containing only trials with valid position
			  bins.
		"""

		mask = np.logical_and(pos_bins >= 0,pos_bins < self.nr_bins)
		pos_bins = pos_bins[mask]
		cnds = cnds[mask]
		epochs = epochs[np.where(mask)[0]]

		return pos_bins, cnds, epochs

	def set_max_trial(self, cnds: np.ndarray, pos_bins: np.ndarray, 
					 method: str) -> int:
		"""
		Determine maximum trials per bin for balanced cross-validation.
		
		Calculates the maximum number of trials that can be used per 
		position bin to ensure balanced data across conditions and 
		cross-validation folds.

		#TODO: implement k-fold method properly

		Parameters
		----------
		cnds : np.ndarray
			Array of condition labels for trials.
		pos_bins : np.ndarray  
			Array of position labels for trials.
		method : str
			Cross-validation method to use:
			- 'Foster': Standard k-fold approach using self.nr_folds
			- 'k-fold': #TODO: Clarify k-fold method implementation

		Returns
		-------
		int
			Maximum number of trials per position bin that ensures
			balanced data across conditions and cross-validation folds.
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

	def extract_power(self, x: np.ndarray, 
				   band: Union[str, list, tuple, None] = None) -> np.ndarray:
		"""
		Extract power or envelope from signal data.
		
		Computes power by squaring the complex magnitude of the analytic 
		signal for filtered data, computes envelope via Hilbert 
		transform for broadband data, or returns the original signal 
		unchanged.

		Parameters
		----------
		x : np.ndarray
			Complex analytic signal from Hilbert transform 
			(filtered data) or raw voltage data (broadband).
		band : str, list, tuple, or None, default=None
			Frequency band specification:
			- list/tuple: Apply power extraction (|x|²) for filtered 
			  data
			- 'broadband': Compute envelope |Hilbert(x)| after trial 
			  averaging for phase-locked amplitude
			- str or None: Return original signal without transformation

		Returns
		-------
		np.ndarray
			Power signal (|x|²) if band is list/tuple,
			envelope if band is 'broadband',
			otherwise original signal unchanged.
		"""

		if isinstance(band, (list,tuple)):
			power = abs(x)**2
		elif band == 'broadband':
			# Compute envelope of averaged signal (after averaging)
			power = np.abs(hilbert(x, axis=-1))
		else:	
			power = x

		return power

	def tfr_decomposition(self, epochs: mne.Epochs, band: Union[tuple, str], 
					   tois: slice, downsample: int
					   ) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Perform time-frequency decomposition for CTF analysis.
		
		Isolates frequency-specific activity using Butterworth filtering 
		followed by Hilbert transform, or processes broadband signals. 
		Extracts both evoked and total power representations for 
		subsequent encoding model analysis.

		Parameters
		----------
		epochs : mne.Epochs
			Epochs object containing EEG data for time-frequency 
			analysis.
		band : tuple or str
			Frequency specification:
			- tuple: (low_freq, high_freq) for bandpass filtering
			- 'broadband': Use unfiltered broadband signal
		tois : slice
			Time window of interest slice object for data extraction.
		downsample : int
			Downsampling factor applied after filtering to reduce data 
			size.

		Returns
		-------
		E : np.ndarray
			For filtered data: Complex analytic signal from Hilbert 
			transform, shape (trials, channels, time). Used for evoked 
			power calculation: activity phase-locked to stimulus onset 
			(computed by averaging complex signal across trials, 
			then squaring magnitude).
			For broadband: Raw baseline-corrected voltages, 
			shape (trials, channels, time). Envelope will be computed 
			AFTER trial averaging via extract_power() for cleaner 
			phase-locked amplitude estimation.
		T : np.ndarray
			For filtered data: Total power signal, shape (trials, channels, 
			time). Represents ongoing activity irrespective of phase 
			relationship to stimulus onset (computed by squaring 
			magnitude of complex signal, then averaging across trials).
			For broadband: Raw baseline-corrected voltages, 
			shape (trials, channels, time). Preserves phase-locked 
			activity and polarity information for ERP analysis.		
			
		Notes
		-----
		The decomposition pipeline includes:
		
		**For filtered data (band = tuple):**
		1. **Frequency Filtering**: 5th-order Butterworth bandpass 
		filter
		2. **Hilbert Transform**: Extract complex analytic signal for 
		phase information
		3. **Power Extraction**: Compute |signal|² for total power
		4. **Time Windowing**: Extract specified time window of interest
		5. **Downsampling**: Reduce temporal resolution to target 
		sampling rate
		
		**For broadband analysis (band = 'broadband'):**
		1. **Baseline Correction**: Apply baseline normalization
		2. **Data Storage**: 
		- Both T and E contain raw voltages initially
		- Envelope for E is computed AFTER trial averaging 
			(in extract_power) to capture phase-locked amplitude
		3. **Time Windowing**: Extract specified time window of interest
		4. **Downsampling**: Reduce temporal resolution to target 
		sampling rate
		"""		
	
		# initiate arrays for evoked and total power
		_, nr_chan, nr_time = epochs._data.shape

		# extract power using hilbert or wavelet convolution
		if isinstance(band, (list,tuple)):
			epochs.filter(band[0], band[1], method = 'iir', 
						iir_params = dict(ftype = 'butterworth', order = 5))
			epochs.apply_hilbert()
			E = epochs._data
			T = self.extract_power(E,band)
		elif band == 'broadband':
			epochs.apply_baseline(baseline = self.baseline)
			E = epochs._data
			T = epochs._data

		# trim filtered data (after filtering to avoid artifacts)
		E = E[:,:,tois]
		T = T[:,:,tois]

		# downsample 
		if band != 'broadband':
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
		Apply inverted encoding model (IEM) to reconstruct spatial 
		channels.
		
		Implements a two-stage process to learn and apply spatial 
		encoding models: first training weights to map from hypothetical 
		channel responses to observed electrode patterns, then inverting 
		the model to reconstruct channel responses from test data.

		Parameters
		----------
		train_X : np.ndarray
			Training EEG data with shape (epochs, electrodes, samples).
			If 3D, samples are averaged to create (epochs, electrodes).
		test_X : np.ndarray  
			Test EEG data with same structure as train_X.
		C1 : np.ndarray
			Basis function matrix with shape (channels, spatial_bins).
			Represents hypothetical channel responses for training.

		Returns
		-------
		tuple[np.ndarray, np.ndarray]
			C2s : np.ndarray
				Reconstructed channel responses aligned to common 
				spatial reference frame, shape (epochs, channels).
			W : np.ndarray
				Learned weight matrix mapping channels to electrodes,
				shape (channels, electrodes).

		Notes
		-----
		The IEM procedure follows two stages:
		
		**Training Stage:**
		Estimates weights W using: B1 = W @ C1
		- B1: Training electrode data (electrodes x epochs)  
		- C1: Predicted channel responses (channels x spatial_bins)
		- W: Weight matrix (channels x electrodes)

		**Testing Stage:**
		Reconstructs channel responses: C2 = (W^T)^-1 @ B2
		- B2: Test electrode data
		- C2: Reconstructed channel responses
		
		**Spatial Alignment:**
		Channel responses are circularly shifted to align with a common
		spatial reference frame, enabling averaging across trials with
		different target locations.
		
		Optional PCA dimensionality reduction is applied if specified
		during initialization to reduce electrode space complexity.

		Examples
		--------
		Apply forward model to single time point:
		
		>>> # Prepare data and basis functions
		>>> C1 = ctf.calculate_basis_set(8, 8) # 8 channels, 8 positions
		>>> 
		>>> # Apply encoding model
		>>> C2_reconstructed, weights = ctf.forward_model(
		...     train_data, test_data, C1)
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

	def forward_model_loop(self, E_train: np.ndarray, E_test: np.ndarray,
							T_train: np.ndarray, T_test: np.ndarray, 
							C1: np.ndarray, GAT: bool
							) -> Tuple[np.ndarray, np.ndarray,
											   np.ndarray, np.ndarray]:
		"""
		Apply inverted encoding model across all time points.
		
		Performs forward model analysis for each time sample, with 
		option for Generalization Across Time (GAT) analysis that tests 
		all train/test time combinations or standard diagonal analysis.

		Parameters
		----------
		E_train : np.ndarray
			Training data with evoked power, shape (trials, channels, 
			time).
		E_test : np.ndarray
			Test data with evoked power, shape (trials, channels, time).
		T_train : np.ndarray
			Training data with total power, shape (trials, channels, 
			time).
		T_test : np.ndarray
			Test data with total power, shape (trials, channels, time).
		C1 : np.ndarray
			Spatial basis function matrix, shape (channels, 
			spatial_bins).
		GAT : bool
			Whether to perform Generalization Across Time analysis:
			- True: Test all train/test time combinations (full matrix)
			- False: Test only matching time points (diagonal only)

		Returns
		-------
		C2_E : np.ndarray
			Reconstructed channel responses from evoked power analysis.
			Shape depends on GAT: (time, bins, channels) or 
			(time, time, bins, channels).
		W_E : np.ndarray
			Weight matrices from evoked power analysis.
			Shape depends on GAT: (time, bins, electrodes) or 
			(time, time, bins, electrodes).
		C2_T : np.ndarray
			Reconstructed channel responses from total power analysis.
			Same shape structure as C2_E.
		W_T : np.ndarray
			Weight matrices from total power analysis.
			Same shape structure as W_E.

		Notes
		-----
		This function applies the forward model (see `forward_model` 
		method) to every time point in the analysis window. For GAT 
		analysis, it creates a full time x time matrix showing how well 
		models trained at each timepoint generalize to every other time 
		point.
		
		#TODO: Consider parallelizing the nested time loop for
        #  performance
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

	def train_test_cross(self, pos_bins: np.ndarray, train_idx: np.ndarray,
						test_idx: np.ndarray, nr_iter: int,
						trial_limit: Optional[int] = None
						) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Create balanced train/test indices for cross-condition analysis.
		
		Generates balanced trial indices for training and testing across 
		different experimental conditions, ensuring equal representation 
		of position bins in both training and test sets.

		Parameters
		----------
		pos_bins : np.ndarray
			Array of position bin labels for all trials.
		train_idx : np.ndarray
			Boolean or integer indices specifying trials available for 
			training.
		test_idx : np.ndarray
			Boolean or integer indices specifying trials available for 
			testing.
		nr_iter : int
			Number of cross-validation iterations to generate.
		trial_limit : int, optional
			Maximum number of trials per bin to use. Applied when 
			train/test conditions overlap to prevent data leakage.

		Returns
		-------
		train_idx : np.ndarray
			Balanced training indices, 
			shape (iterations, bins, folds, trials_per_bin).
		test_idx : np.ndarray
			Balanced test indices, 
			shape (iterations, bins, 1, trials_per_bin).

		Notes
		-----
		This function ensures statistical validity by:
		
		1. **Balanced Sampling**: Equal trial counts across position 
		   bins
		2. **Cross-Validation**: Splits training data into multiple 
		   folds
		3. **Data Leakage Prevention**: Removes overlapping trials when 
           trial_limit specified
		4. **Random Sampling**: Uses class seed for reproducible trial 
		   selection
		"""

		if self.seed:
			np.random.seed(self.seed) # set seed 

		# select training data (ensure that number of bins is balanced)
		train_bins = pos_bins[train_idx]
		train_idx = np.arange(pos_bins.size)[train_idx]
		bins,  counts = np.unique(train_bins, return_counts=True)
		min_obs = min(counts)
        # Apply trial limit if specified (for overlapping train/test cnds)
		if trial_limit is not None:
			min_obs = min(min_obs, trial_limit)

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

		# now create test set per iteration
		test_bins = pos_bins[test_idx]
		idx = np.arange(pos_bins.size)[test_idx]
		test_idx = []
		for itr in range(nr_iter):
			if trial_limit is not None:
				# get selected train indices and remove from test set
				used_train_indices = train_idx[itr].flatten()
				available_idx = np.setdiff1d(idx, used_train_indices)
				available_test_bins = pos_bins[available_idx]
			else:
				available_idx = idx
				available_test_bins = test_bins

			bins,  counts = np.unique(available_test_bins, return_counts=True)
			min_obs_test = min(counts)

			# handle case where test bins do not match all train bins
			train_bins_unique = np.unique(train_bins)
			if bins.size == train_bins_unique.size:
				# all bins present - simple case
				selected_test_idx = [np.random.choice(
								available_idx[available_test_bins==b], 
                                min_obs_test, False) for b in bins]
				selected_test_idx = np.array(selected_test_idx)
			else:
				# some bins missing - need to pad with zeros
				selected_test_idx = np.zeros((train_bins_unique.size, 
								               min_obs_test), dtype=int)
				bin_cnt = 0
				for bin_idx, bin_val in enumerate(train_bins_unique):
					if bin_val in bins:
						mask = available_test_bins == bin_val
						selected_test_idx[bin_idx] = np.random.choice(
							available_idx[mask], min_obs_test, False)
						bin_cnt += 1
	
			test_idx.append(np.array(selected_test_idx)[:, None, :])
		test_idx = np.stack(test_idx)

		return train_idx, test_idx	

	def train_test_split(self, pos_bins: np.ndarray, cnd_idx: np.ndarray, 
						max_tr: int) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Create random block assignment for within-condition 
		cross-validation.
		
		Generates randomized train/test splits within the same 
		experimental condition using k-fold cross-validation approach 
		with balanced position bins.

		Parameters
		----------
		pos_bins : np.ndarray
			Array of position bin labels for all trials.
		cnd_idx : np.ndarray
			Boolean or integer indices specifying trials from the target 
			condition.
		max_tr : int
			Maximum number of trials per position bin to use in 
			analysis.

		Returns
		-------
		train_idx : np.ndarray
			Training indices with 
			shape (iterations x folds, bins, folds-1, trials_per_block).
			Contains trial indices for training across all 
			cross-validation folds.
		test_idx : np.ndarray
			Test indices with 
			shape (iterations x folds, bins, trials_per_block).
			Contains trial indices for testing in each cross-validation 
			fold.

		Notes
		-----
		This function implements k-fold cross-validation within a single 
        condition:
		
		1. **Random Block Assignment**: Trials randomly assigned to k 
		   folds
		2. **Balanced Bins**: Equal trials per position bin across folds
		3. **Cross-Validation**: Each fold serves as test set once
		4. **Multiple Iterations**: Process repeated for robust 
		   estimates
		
		The block assignment ensures that each position bin has equal 
	    representation in training and test sets across all 
		cross-validation iterations.
		"""

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
			blocks = np.full(nr_tr, np.nan)
			shuf_blocks = np.full(nr_tr, np.nan)

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

	def set_frequencies(self, freqs: Union[str, dict]) -> Tuple[list, int]:
		"""
		Configure frequency bands for CTF analysis.
		
		Processes frequency specification to create standardized 
		frequency list for time-frequency decomposition or broadband 
		analysis.

		Parameters
		----------
		freqs : str or dict
			Frequency specification for analysis:
			- 'main_param': Use class initialization parameters 
			  (min_freq, max_freq, num_frex, freq_scaling) to create 
			  frequency bands
			- 'broadband': Use unfiltered broadband signal  
			- dict: Custom frequency bands as 
			   {band_name: [low_freq, high_freq]}

		Returns
		-------
		frex : list
			List of frequency specifications:
			- For frequency bands: list of sorted (low_freq, high_freq)
			  tuples
			- For broadband: list containing single 'broadband' string
		nr_frex : int
			Number of frequency bands to process.
		bands : list or None
			List of sorted band names if custom dict provided, 
			otherwise None.
		"""

		bands = None
		if freqs == 'main_param':
			if self.freq_scaling == 'log':
				frex = np.logspace(np.log10(self.min_freq), 
								np.log10(self.max_freq), 
							  	self.num_frex)
			elif self.freq_scaling == 'linear':
				frex = np.linspace(self.min_freq,self.max_freq,self.num_frex)
			frex = [(frex[i], frex[i+1]) for i in range(self.num_frex -1)]
		elif type(freqs) == dict:
			frex = [(freqs[band][0],freqs[band][1]) for band 
		   					in sorted(freqs.keys(), key=lambda k: freqs[k][0])]
			bands = sorted(freqs.keys(), key=lambda k: freqs[k][0])
		elif freqs == 'broadband':
			frex = [freqs]
		nr_frex = len(frex)

		return frex, nr_frex, bands

	def localizer_spatial_ctf(self,pos_labels_tr:dict ='all',
							pos_labels_te:dict ='all',
							freqs:dict='main_param',te_cnds:dict=None,
							te_header:str=None,
							window_oi_tr:tuple=None,window_oi_te:tuple=None,
							excl_factor_tr:dict=None,excl_factor_te:dict=None,
							downsample:int = 1,nr_perm:int=0,GAT:bool=False,
							name:str='loc_ctf'):

		# set train and test data
		epochs_tr, df_tr = self.select_ctf_data(self.epochs[0], self.df[0],
												self.elec_oi,excl_factor_tr)

		epochs_te, df_te = self.select_ctf_data(self.epochs[1], self.df[1],
												self.elec_oi, excl_factor_te)

		if window_oi_tr is None:
			window_oi_tr = (epochs_tr.tmin, epochs_tr.tmax)

		if window_oi_te is None:
			window_oi_te = window_oi_tr

		nr_itr = 1
		ctf_name = f'sub_{self.sj}_{name}'
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
		data_type = 'power' if freqs != ['broadband'] else 'broadband'

		if type(te_cnds) == dict:
			(cnd_header, test_cnds), = te_cnds.items()
		else:
			test_cnds = ['all_data']
			cnd_header = None

		# set train and test data
		(pos_bins_tr, 
		_,
		epochs_tr, 
		_) = self.select_ctf_labels(epochs_tr, df_tr, pos_labels_tr, None)

		(pos_bins_te, 
		cnds, 
		epochs_te, 
		_) = self.select_ctf_labels(epochs_te, df_te, pos_labels_te, te_cnds)

		# Frequency loop (ensures that data is only filtered once)
		for fr in range(nr_freqs):
			print('Frequency {} out of {}'.format(str(fr + 1), str(nr_freqs)))

			# Time-Frequency Analysis 
			(E_tr, 
			T_tr) = self.tfr_decomposition(epochs_tr.copy(),freqs[fr],
									 tois_tr,downsample)	
			nr_samp_tr = 1 if avg_tr else E_tr.shape[-1]
			(E_te, 
			T_te) = self.tfr_decomposition(epochs_te.copy(),freqs[fr],
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

			if freqs == ['broadband']:
				# Rename to clarify what each represents in broadband
				ctf[cnd]['C2_envelope'] = ctf[cnd].pop('C2_E')
				ctf[cnd]['W_envelope'] = ctf[cnd].pop('W_E')
				ctf[cnd]['C2_voltage'] = ctf[cnd].pop('C2_T')
				ctf[cnd]['W_voltage'] = ctf[cnd].pop('W_T')		# save output
				
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
					  							GAT=GAT,avg_ch=self.avg_ch)
			
			with open(self.folder_tracker(['ctf',self.to_decode], 
					fname=f'ctf_param_{ctf_name}.pickle'),'wb') as handle:
				print('saving ctf params')
				pickle.dump(ctf_param, handle)

	def extract_slopes(self,X:np.array)->float:
		"""
		Calculate slope parameter from channel tuning function.
		
		Extracts slope by collapsing symmetric points in the tuning 
		curve and fitting a linear trend from the periphery to the peak. 
		This provides a summary statistic of spatial selectivity 
		strength.

		Parameters
		----------
		X : np.ndarray
			Reconstructed channel tuning function with shape 
			(nr_chans,). Represents channel responses across spatial 
			locations.

		Returns
		-------
		float
			Slope coefficient indicating tuning curve steepness. Higher
			values indicate stronger spatial selectivity.

		Notes
		-----
		The slope extraction process:
		
		1. **Symmetric Collapse**: Average responses at symmetric 
		   spatial positions (e.g., position 1 with position 7 in 
		   8-position task)
		   
		2. **Linear Fit**: Fit first-order polynomial to collapsed data
		   from tails to peak of tuning curve
		   
		3. **Slope Extraction**: Return linear coefficient as 
		   selectivity measure
		
		This approach reduces noise by leveraging symmetry and provides
		a single parameter characterizing spatial tuning strength.

		Examples
		--------
		Extract slope from reconstructed CTF:
		
		>>> # Calculate selectivity slope
		>>> slope = ctf.extract_slopes(channel_response)
		>>> print(f"Spatial selectivity: {slope:.3f}")
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
						  perm_idx:int,ch_idx:int=None,
						  fitting_method:str=None)->dict:
		"""
		Extract tuning parameters from CTF reconstructions.
		
		Helper function that processes reconstructed channel tuning 
		functions to extract slope and optional cosine fitting 
		parameters.

		Parameters
		----------
		ctfs : np.ndarray
			Reconstructed channel tuning functions with shape:
			- 3D: (n_freqs, n_samples, n_spatial_bins) for standard 
			   analysis
			- 4D: (n_freqs, n_train_samples, n_test_samples, 
			   n_spatial_bins) for Generalization Across Time (GAT) 
			   analysis
		params : dict
			Pre-allocated parameter arrays to store results. Keys follow 
			pattern '{signal}_paramname' (e.g., 'E_slopes', 'T_amps').
		signal : str
			Signal type identifier: 'E', 'T' for filtered data; 
			'envelope', 'voltage' for broadband data.
		perm_idx : int
			Permutation iteration index for parameter storage.
		ch_idx : int, optional
			Spatial channel index when processing individual channels 
			(avg_ch=False). If None, processes averaged channels.
		fitting_method : str, optional
			Method for curve fitting. Options:
			- 'von_mises': Von Mises (circular cosine) fitting (default)
			- 'gaussian': Gaussian bell curve fitting
			Default is 'von_mises'.		
		
		Returns
		-------
		dict
			Updated params dictionary with extracted tuning parameters.

		Notes
		-----
		Extracts slope parameters via `extract_slopes()` and optional 
		curve fitting parameters (von Mises or Gaussian) when amplitude 
		keys are present in params dict.
		"""
		
		# Determine if input is GAT (4D) or standard (3D)
		if ctfs.ndim == 3:
			nr_freqs, nr_samples_tr, nr_chan = ctfs.shape
			nr_samples_te = 1
			GAT = False
			# insert new dimension so that indexing does not crash
			ctfs = ctfs[...,np.newaxis,:]
		else:
			nr_freqs, nr_samples_tr, nr_samples_te, nr_chan = ctfs.shape 
			GAT = True

		# Extract CTF parameters for each frequency and time point
		for freq_idx in range(nr_freqs):
			for train_sample in range(nr_samples_tr):
				for test_sample in range(nr_samples_te):
					
					# Extract current CTF for this time/frequency combination
					current_ctf = ctfs[freq_idx, train_sample, test_sample]
					
					# Calculate slope parameter
					slope = self.extract_slopes(current_ctf)
					if ch_idx is None:
						params[f'{signal}_slopes'][perm_idx, freq_idx, 
						                            train_sample, 
													test_sample] = slope
					else:
						params[f'{signal}_slopes'][perm_idx, freq_idx, 
						                          train_sample, test_sample, 
						                          ch_idx] = slope
					
					# Calculate curve fitting parameters 
                    # if requested AND fitting method provided
					if (any([k for k in params.keys() if 'amps' in k]) and 
						fitting_method):
						if fitting_method == 'von_mises':
							(amplitude, baseline, conc, 
							 mean_loc, _) = self.fit_cos_to_ctf(current_ctf)
						elif fitting_method == 'gaussian':
							gaussian_params = self.fit_gaussian(current_ctf)
							amplitude, mean_loc, sigma = gaussian_params
							# Gaussian doesn't have baseline offset
							baseline = 0  
							# Convert sigma to concentration-like measure
							conc = 1.0 / sigma  
						else:
							raise ValueError(f"Unknown fitting method: "
											f"{fitting_method}")
						
						if ch_idx is None:
							params[f'{signal}_amps'][perm_idx, freq_idx, 
							                        train_sample, 
							                        test_sample] = amplitude
							params[f'{signal}_base'][perm_idx, freq_idx, 
							                        train_sample, 
							                        test_sample] = baseline
							params[f'{signal}_conc'][perm_idx, freq_idx, 
							                        train_sample, 
							                        test_sample] = conc
							params[f'{signal}_means'][perm_idx, freq_idx, 
							                         train_sample, 
							                         test_sample] = mean_loc
						else:
							params[f'{signal}_amps'][perm_idx, freq_idx, 
							                        train_sample, test_sample, 
							                        ch_idx] = amplitude
							params[f'{signal}_base'][perm_idx, freq_idx, 
							                        train_sample, test_sample, 
							                        ch_idx] = baseline
							params[f'{signal}_conc'][perm_idx, freq_idx, 
							                        train_sample, test_sample, 
							                        ch_idx] = conc
							params[f'{signal}_means'][perm_idx, freq_idx, 
							                         train_sample, test_sample, 
							                         ch_idx] = mean_loc	

		return params

	def summarize_ctfs(self,ctfs:dict,params:dict,nr_samples:Union[int,tuple],
					   nr_freqs:int,test_bins:np.array,nr_perm:int=1,
					   avg_ch:bool=True,fitting_method:str=None)->dict:
		"""
		Extract tuning parameters from CTF reconstructions across 
		conditions and permutations.
		
		Orchestrates the extraction of tuning parameters (slopes, 
		amplitudes, etc.) from channel tuning functions for multiple 
		signals, permutations, and spatial channels.Handles both
		standard CTF analysis and Generalization Across Time (GAT) 
		analysis.

		Parameters
		----------
		ctfs : dict
			Condition-specific CTF reconstructions with keys like 
			'C2_T', 'C2_E' containing arrays of shape (n_perm, n_freqs, 
			n_samples, n_channels) for standard analysis
			or (n_perm, n_freqs, n_train_samples, n_test_samples, 
			n_channels) for GAT.
		params : dict
			Pre-initialized output dictionary with arrays for storing 
			tuning parameters. Keys follow pattern 
			'{signal}_{parameter}' (e.g., 'T_slopes', 'E_amps').
		nr_samples : Union[int, tuple]
			Number of time samples. Int for standard analysis, 
			tuple (n_train, n_test) for GAT analysis.
		nr_freqs : int
			Number of independent frequencies in the CTF analysis.
		test_bins : np.array
			Spatial bin indices actually used for CTF fitting. Allows 
			analysis of subset of spatial locations.
		nr_perm : int, optional
			Number of permutations. Default is 1 (no permutation).
		avg_ch : bool, optional
			Whether to average CTFs across channels (True) or analyze 
			each channel individually (False). Default is True.
		fitting_method : str, optional
			Method for curve fitting. Options:
			- 'von_mises': Von Mises (circular cosine) fitting (default)
			- 'gaussian': Gaussian bell curve fitting
			Default is 'von_mises'.

		Returns
		-------
		dict
			Updated params dictionary with extracted tuning parameters 
			filled in.
		"""

		# check whether ctfs contains matrix of train and test samples
		GAT = True if isinstance(nr_samples,tuple) else False
	
		# infer datatypes to extract
		signals = np.unique([key.split('_')[0] for key in params.keys()])

		for signal in signals:		
			# get params for each permutation
			for perm_idx in range(nr_perm):
				signal_ctfs = ctfs[f'C2_{signal}'][perm_idx]
				if avg_ch:
					# check whether ctfs contains unfitted bins
					if test_bins.size < self.nr_bins:
						signal_ctfs = signal_ctfs[:,:,test_bins.astype(int)]
					signal_ctfs = signal_ctfs.mean(axis = -2)
					params = self.extract_ctf_params(signal_ctfs, params,
					 							signal, perm_idx,
												fitting_method=fitting_method)
				else:
					nr_chans = signal_ctfs.shape[-2]
					
					for channel_idx in range(nr_chans):
						if GAT:
							ch_ctfs = signal_ctfs[:,:,:,channel_idx]
						else:
							ch_ctfs = signal_ctfs[:,:,channel_idx]
						if np.all(ch_ctfs == 0):
							continue
						params = self.extract_ctf_params(ch_ctfs,params, 
														signal, perm_idx, 
														channel_idx,
														fitting_method)
		
		return params

	def get_ctf_tuning_params(self,ctfs:dict,ctf_param:str='slopes',
			   				GAT:bool=False,avg_ch:bool=True,
							test_bins:np.array=None)->dict:
		"""
		Extract tuning parameters from channel tuning functions across 
		experimental conditions.
		
		Main interface for quantifying CTF reconstructions by extracting 
		tuning parameters such as slopes and curve fitting parameters 
		(amplitude, baseline, concentration, mean location). 
		Orchestrates the analysis pipeline through summarize_ctfs() and 
		extract_ctf_params() helper functions. Supports both von Mises 
		and Gaussian curve fitting. Automatically detects data type 
		(power vs broadband) from available CTF keys.

		Parameters
		----------
		ctfs : dict
			CTF reconstructions per condition as returned by 
			spatial_ctf() and localizer_spatial_ctf(). Keys are 
			condition names, values contain 'C2_T', 'C2_E' 
			(for filtered power data) or 'C2_envelope'/'C2_voltage' 
			(for broadband data).
			Data type is automatically inferred from available keys.
		ctf_param : str, optional
			Parameter extraction method. Options:
			- 'slopes': Extract only slope parameters (fastest)
			- 'von_mises': Extract slopes + von Mises fitting parameters 
			  (amplitude, baseline, concentration, mean location)
			- 'gaussian': Extract slopes + Gaussian fitting parameters
			  (amplitude, mean, sigma, with derived measures)
			Default is 'slopes'.
		GAT : bool, optional
			Whether analysis uses Generalization Across Time with 
			independent training and testing timepoints. When True, 
			expects 4D arrays(n_freqs, n_train_samples, n_test_samples, 
			n_channels). Default is False.
		avg_ch : bool, optional
			Whether to average CTFs across spatial channels (True) or
			analyze each channel individually (False). Default is True.
		test_bins : np.array, optional
			Spatial bin indices to include in analysis. If None, uses 
			all available spatial bins. Default is None.

		Returns
		-------
		dict
			Nested dictionary with structure 
			{condition: {parameter: values}}. Parameter keys follow 
			pattern '{signal}_{metric}' where:
			- signal: 'T'/'E' (filtered power), 'envelope'/'voltage' 
			(broadband)
			- metric: 'slopes', 'amps', 'base', 'conc', 'means'
			Values are arrays with dimensions squeezed to remove 
			singleton axes (except for frequency ).
		"""		
	
		# determine the output parameters
		if test_bins is None:
			test_bins = np.arange(self.nr_bins)
		
		# infer data type from available keys
		sample_ctf = ctfs[list(ctfs.keys())[0]]
		if 'C2_envelope' in sample_ctf:
			signals = ['envelope', 'voltage']
			data_key = 'C2_envelope'
		else:
			signals = ['T', 'E']
			data_key = 'C2_E'
		
		output_params = []
		output_params += [f'{signal}_slopes' for signal in signals]
		
		# Determine fitting method and whether to extract curve fitting params
		if ctf_param in ['von_mises', 'gaussian']:
			fitting_method = ctf_param
			tuning_params = ['amps', 'base','conc','means']
			output_params += [f'{signal}_{param}' for param in tuning_params
		     										for signal in signals]
		else:
			fitting_method = None  # Only slopes, no curve fitting
			
		# initiate output dict
		ctf_param = {}
		data = sample_ctf[data_key]

		if GAT:
			(nr_perm, nr_freqs, 
			nr_train_samples, nr_test_samples) = list(data.shape)[:4]
			output = np.zeros((nr_perm,nr_freqs, nr_train_samples,
					                                        nr_test_samples))
			nr_samples = (nr_train_samples,nr_test_samples)
		else:
			nr_perm, nr_freqs, nr_samples = list(data.shape)[:3]
			output = np.zeros((nr_perm,nr_freqs,nr_samples,1))		

		if not avg_ch:
			output = output[..., np.newaxis] * np.zeros(self.nr_chans)

		# loop over all conditions
		for cnd in ctfs.keys():
			ctf_param.update({cnd:{}})
			# initiate output data
			for param in output_params:
				ctf_param[cnd].update({param:output.copy()})

			# get parameters
			ctf_param[cnd] = self.summarize_ctfs(ctfs[cnd],ctf_param[cnd],
				     						nr_samples,nr_freqs,test_bins,
											nr_perm,avg_ch,fitting_method)
			
		
			# get rid of unnecessary dimensions
			ctf_param[cnd] = {k: np.squeeze(v) 
										for k, v in ctf_param[cnd].items()}
		
			# restore frequency dimension if it was squeezed out
			if nr_freqs == 1:
				ctf_param[cnd] = {k: np.expand_dims(v, axis=0) 
									for k, v in ctf_param[cnd].items()}

		return ctf_param	

	def fit_cos_to_ctf(self, ctf_data: np.array, conc_step: float = 0.1, 
					   estimate_center: bool = False):
		"""
		Fit a von Mises (circular cosine) function to channel tuning 
		function data.
		
		Estimates parameters of a von Mises distribution by fitting a 
		cosine-shaped tuning curve to CTF reconstruction data. Uses grid 
		search over concentration parameters combined with least squares 
		estimation of amplitude and baseline.Supports both fixed-center 
		spatial working memory) and free-center (spatial attention) 
		approaches.

		Parameters
		----------
		ctf_data : np.array
			1D array of CTF reconstruction values across spatial bins.
			Represents channel response as a function of spatial 
			location.
		conc_step : float, optional
			Step size for concentration parameter grid search. Smaller 
			values provide finer resolution but slower fitting.
			Default is 0.1.
		estimate_center : bool, optional
			Whether to estimate the mean location (True) or fix it at 
			center (False). Default is False. 

		Returns
		-------
		tuple
			Five-element tuple containing:
			- amplitude (float): Peak response amplitude of the tuning 
			  curve
			- baseline (float): Baseline offset level  
			- concentration (float): Concentration parameter (tuning 
			  width)
			- mean_location (float): Mean location of tuning curve (in 
			  radians)
			- rmse (float): Root mean squared error of the fit
		"""


		num_bins = ctf_data.size

		# possible concentration parameters to consider
		# step size of 0.1 is usually good, but smaller makes for better fits
		concentration_range = np.arange(3, 40 + conc_step, conc_step)

		# allocate storage arrays for fitting results
		(sse, 
        baseline_estimates, 
		amplitude_estimates) = np.zeros((3, concentration_range.size))

		# define spatial coordinates for von Mises function
		spatial_coords = np.linspace(0, np.pi - np.pi/num_bins, num_bins)
		
		if estimate_center:
			# estimate optimal mean location by testing multiple positions
			mean_locations = spatial_coords
		else:
			# use center as fixed mean location 
			mean_locations = [spatial_coords[int(num_bins/2)]]

		# storage for best parameters across all mean locations
		best_overall_sse = np.inf
		best_params = None

		# loop over mean locations (single center point or all positions)
		for mean_location in mean_locations:
			# loop over all concentration parameters
			# estimate best amplitude and baseline offset 
			# and find combination that minimizes sum of squared errors
			initial_amplitude, initial_baseline = 1, 0
			for conc_idx in range(concentration_range.size):
				# create the von Mises function with current concentration
				current_concentration = concentration_range[conc_idx]
				predicted_response = (initial_amplitude * 
								np.exp(current_concentration * 
								(np.cos(mean_location - spatial_coords) - 1)) + 
								initial_baseline)
				
				# build design matrix and use GLM to estimate ampl and baseline
				design_matrix = np.zeros((num_bins, 2))
				design_matrix[:,0] = predicted_response
				design_matrix[:,1] = np.ones(num_bins)
				
				# solve for optimal amplitude and baseline
				betas, _, _, _ = np.linalg.lstsq(design_matrix, ctf_data, 
									                                rcond=None)
				(amplitude_estimates[conc_idx], 
				 baseline_estimates[conc_idx]) = betas
				
				# calculate fitted response and sum of squared errors
				fitted_response = predicted_response * betas[0] + betas[1]
				sse[conc_idx] = sum((fitted_response - ctf_data)**2)

			# find best parameters for this mean location
			best_conc_idx = np.argmin(sse)
			current_sse = sse[best_conc_idx]
			
			# update overall best if this mean location is better
			if current_sse < best_overall_sse:
				best_overall_sse = current_sse
				best_params = {
					'amplitude': amplitude_estimates[best_conc_idx],
					'baseline': baseline_estimates[best_conc_idx], 
					'concentration': concentration_range[best_conc_idx],
					'mean_location': mean_location
				}

		# extract best parameters
		best_amplitude = best_params['amplitude']
		best_baseline = best_params['baseline']
		best_concentration = best_params['concentration']
		best_mean_location = best_params['mean_location']
		
		# calculate final fitted response and root mean squared error
		final_prediction = (best_amplitude * 
						 np.exp(best_concentration * 
						(np.cos(best_mean_location - spatial_coords) - 1)) + 
						best_baseline)
		rmse = np.sqrt(sum((final_prediction - ctf_data)**2) / 
					   final_prediction.size)

		return (best_amplitude, best_baseline, best_concentration, 
				best_mean_location, rmse)

	def gaussian(self, x, amp, mu, sig):
		"""
		Standard Gaussian function for curve fitting.
		
		Computes Gaussian probability density values for given
		parameters, commonly used for fitting bell-shaped tuning curves 
		in CTF analysis.

		Parameters
		----------
		x : np.ndarray
			Input coordinates for Gaussian evaluation.
		amp : float
			Gaussian amplitude (peak height).
		mu : float
			Gaussian mean (center location).
		sig : float
			Gaussian standard deviation (width parameter).

		Returns
		-------
		np.ndarray
			Gaussian function values at input coordinates.

		Notes
		-----
		Uses the standard Gaussian formula:
		y = amp * exp(-(x - mu)² / (2 * sig²))
		"""

		y = amp * np.exp(-(x - mu)**2/(2*sig**2))

		return y
		
	def fit_gaussian(self, y):
		"""
		Fit a Gaussian function to input data using least squares.
		
		Estimates optimal Gaussian parameters (amplitude, mean, width)
		that best fit the provided data, commonly used for 
		characterizing spatial tuning curve shapes in CTF analysis.

		Parameters
		----------
		y : np.ndarray
			Data points to fit with Gaussian function. Should represent
			a bell-shaped distribution (e.g., CTF tuning curve).

		Returns
		-------
		np.ndarray
			Array containing fitted parameters [amplitude, mean, sigma]
			where:
			- amplitude: Peak height of fitted Gaussian
			- mean: Center location of fitted Gaussian  
			- sigma: Width (standard deviation) of fitted Gaussian

		Notes
		-----
		Initial parameter estimates are computed from data statistics:
		- Amplitude: Maximum value in y
		- Mean: Weighted center of mass
		- Sigma: Estimated from data spread
		
		Uses scipy.optimize.curve_fit for parameter optimization.

		Examples
		--------
		Fit Gaussian to CTF tuning curve:
		
		>>> # Reconstruct channel responses
		>>> channel_response = ctf_reconstructed[0, :]  # Single trial
		>>> 
		>>> # Fit Gaussian to characterize tuning
		>>> amp, mu, sigma = ctf.fit_gaussian(channel_response)
		>>> print(f"Tuning width: {sigma:.2f} spatial bins")
		"""

		x = np.arange(y.size)
		mu = sum(x * y) / sum(y)
		sig = np.sqrt(sum(y * mu)**2) / sum(y)

		popt, pcov = curve_fit(self.gaussian, x, y, p0=[max(y), mu, sig] )

		return popt 
