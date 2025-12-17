"""
Support utilities for EEG analysis.

This module provides core utility functions for EEG data manipulation 
andexperimental design handling. 

Functions
---------
Electrode selection:
    select_electrodes : Select electrode subsets by region or name
    get_diff_pairs : Create contra-ipsi electrode pairs

Data manipulation:
    baseline_correction : Apply baseline correction to EEG data
    match_epochs_times : Synchronize time axes across objects

Experimental design:
    trial_exclusion : Exclude trials based on conditions
    get_time_slice : Convert time windows to array indices
    create_cnd_loop : Generate factorial condition combinations
    log_preproc : Log preprocessing parameters to CSV

"""

import os
import mne
import warnings
import itertools

import numpy as np
import pandas as pd

from typing import Optional, Union, Tuple

def select_electrodes(
	epochs: mne.Epochs,
	elec_oi: Union[list, str] = 'all'
) -> np.ndarray:
	"""
	Select subset of electrodes based on available channels and 
	electrode groupings.

	Parameters
	----------
	epochs : mne.Epochs
		Epoched EEG data containing the neural responses to be analyzed
		and the electrode names.
	elec_oi : list or str, default='all'
		Electrode selection criteria:
		- 'all': Use all available electrodes in ch_names
		- 'posterior': Posterior electrodes (parietal, occipital 
		   regions)
		- 'frontal': Frontal electrodes (prefrontal, frontal regions)  
		- 'central': Central electrodes (motor, somatosensory regions)
		- list: Specific electrode names to select

	Returns
	-------
	np.ndarray
		Indices of selected electrodes that exist in ch_names.

	Notes
	-----
	The function automatically detects which electrodes are available in 
	the dataset and only selects those that exist. Regional groupings 
	are defined based on standard 10-20 electrode naming conventions and 
	will work across different EEG systems (32, 64, 128+ channels).
	
	For unknown electrode names, the function will issue a warning but
	continue with available electrodes.
	"""

	ch_names = epochs.ch_names

	if isinstance(elec_oi, str):
		if elec_oi == 'all':
			eeg_picks = mne.pick_types(epochs.info, eeg=True)
			# Use all available electrodes
			elec_oi = list(np.array(ch_names)[eeg_picks])
			
		elif elec_oi == 'posterior':
			# Posterior electrodes: parietal, occipital, and posterior temporal
			posterior_electrodes = [
				# Occipital
				'Oz', 'O1', 'O2', 'Iz',
				# Parietal-Occipital  
				'POz', 'PO3', 'PO4', 'PO7', 'PO8', 'PO1', 'PO2', 'PO5', 'PO6',
				# Parietal
				'Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
				'P7', 'P8', 'P9', 'P10',
				# Central-Parietal
				'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
				# Posterior temporal
				'TP7', 'TP8', 'TP9', 'TP10'
			]
			elec_oi = posterior_electrodes
			
		elif elec_oi == 'frontal':
			# Frontal electrodes: prefrontal, frontal, and frontal-central
			frontal_electrodes = [
				# Prefrontal
				'Fpz', 'Fp1', 'Fp2', 
				# Anterior frontal
				'AFz', 'AF3', 'AF4', 'AF7', 'AF8', 'AF1', 'AF2', 'AF5', 'AF6',
				# Frontal
				'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 
				'F10',
				# Frontal-Central
				'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
				# Frontal-Temporal
				'FT7', 'FT8', 'FT9', 'FT10'
			]
			elec_oi = frontal_electrodes
			
		elif elec_oi == 'central':
			# Central electrodes: motor and somatosensory regions
			central_electrodes = [
				'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
				'T7', 'T8', 'T9', 'T10'  # Temporal electrodes
			]
			elec_oi = central_electrodes
			
			
	# Filter elec_oi to only include electrodes that exist in ch_names
	available_electrodes = [e for e in elec_oi if e in ch_names]
	
	if not available_electrodes:
		warnings.warn(f'None of the specified electrodes found in channel \
					names. 'f'Available channels: {list(ch_names)[:10]}...')
		return np.array([], dtype=int)
		
	# Log how many electrodes were found vs requested
	if len(available_electrodes) != len(elec_oi) and isinstance(elec_oi, list):
		missing = set(elec_oi) - set(available_electrodes)
		if len(missing) <= 5:  # Only show if reasonable number
			print(f'Note: {len(missing)} requested electrodes not found: '
				  f'{list(missing)}')

		else:
			print(f'Note: {len(missing)} requested electrodes not found '
				  'in dataset')

	picks = mne.pick_channels(ch_names, include=available_electrodes)

	return picks	 

def baseline_correction(
	X: np.ndarray,
	times: np.ndarray,
	baseline: tuple
) -> np.ndarray:
	"""Perform baseline correction on 3D EEG data.

	Subtracts the mean activity during a baseline period from all time
	points in the data. Baseline correction is applied independently to
	each epoch and electrode.

	Parameters
	----------
	X : np.ndarray
		Input EEG data of shape (n_epochs, n_electrodes, n_times).
	times : np.ndarray
		Time points in seconds corresponding to the last dimension of X.
	baseline : tuple
		Start and end times (in seconds) defining the baseline period.
		For example, (-0.2, 0) for 200ms pre-stimulus baseline.

	Returns
	-------
	corrected_data : np.ndarray
		Baseline-corrected data with same shape as input.

	Raises
	------
	ValueError
		If input data is not 3-dimensional.

	Notes
	-----
	The baseline mean is computed by averaging across the time dimension
	within the specified baseline window, separately for each epoch and
	electrode. This mean is then subtracted from all time points.

	Examples
	--------
	>>> # Baseline correct using -200 to 0 ms pre-stimulus
	>>> corrected = baseline_correction(
	...     X=epochs_data,
	...     times=time_vector,
	...     baseline=(-0.2, 0)
	... )

	See Also
	--------
	get_time_slice : Convert time window to array indices
	"""
	# Check if the input data is 3D
	if X.ndim != 3:	
		raise ValueError(
			f"Input data must be 3D (epochs x electrodes x time), "
			f"got {X.ndim}D"
		)
	
	# find indices of baseline period
	base_idx = get_time_slice(times, baseline[0], baseline[1])

	# Extract the baseline period
	baseline_data = X[:, :, base_idx]

	# Compute the mean over the baseline period (axis=-1 is time)
	baseline_mean = np.mean(baseline_data, axis=-1, keepdims=True)

	# Subtract the baseline mean from the entire data
	corrected_data = X - baseline_mean

	return corrected_data

def match_epochs_times(erps: list) -> list:
	"""
	Truncate epochs to match shortest time axis.

	Finds the evoked object with the fewest time samples and truncates
	all other objects to match this length. Useful when combining data
	across sessions or subjects with slightly different timing.

	Parameters
	----------
	erps : list of mne.Evoked
		List of evoked response objects to synchronize.

	Returns
	-------
	erps : list of mne.Evoked
		List of evoked objects with identical time axes. All objects
		are truncated to match the shortest time series in the input.

	Warnings
	--------
	This function modifies the input list in-place. Data is truncated
	from the end of longer time series.

	Notes
	-----
	The function finds the object with minimum samples, then loops
	through all objects and truncates any that exceed this length by
	copying the shortest object's structure and replacing its data.

	Examples
	--------
	>>> # Synchronize ERPs with mismatched time points
	>>> erp1 = mne.read_evokeds('subj1-ave.fif')[0]  # 500 timepoints
	>>> erp2 = mne.read_evokeds('subj2-ave.fif')[0]  # 505 timepoints
	>>> synchronized = match_epochs_times([erp1, erp2])
	>>> # Both now have 500 timepoints

	See Also
	--------
	support.FolderStructure.read_erps : Automatically applies matching
	"""
	# get min nr of samples
	min_samp_idx = np.argmin([e.times.size for e in erps])
	base_epoch = erps[min_samp_idx]
	min_samp = base_epoch.times.size
	
	# loop over all objects
	for i, e in enumerate(erps):
		if e.times.size > min_samp:
			upd_epoch = base_epoch.copy()
			upd_epoch._data = e._data[..., :min_samp]
			erps[i] = upd_epoch

	return erps

def get_diff_pairs(ch_names: list) -> dict:
	"""
	Create electrode pairing for contralateral-ipsilateral analysis.

	Returns a dictionary mapping each electrode to its contralateral
	pair, enabling computation of lateralized difference topographies
	(contra - ipsi). Left hemisphere electrodes are assumed 
	contralateral to the effect of interest. Midline electrodes map to 
	themselves (resulting in zero difference).

	The function automatically detects available electrodes from the
	channel names and creates pairs based on standard 10-20 system
	naming conventions (odd numbers = left, even numbers = right).

	Parameters
	----------
	ch_names : list of str
		List of channel names present in the data. Used to detect
		available electrodes and convert names to indices.

	Returns
	-------
	idx_pairs : dict
		Dictionary where keys are electrode names and values are tuples
		of (contra_idx, ipsi_idx). Only includes electrodes present in
		ch_names. For midline electrodes, both indices are identical.

	Notes
	-----
	The function creates bidirectional mappings. For example, if 'P7'
	maps to 'P8', then 'P8' also maps to 'P7' (with reversed indices).

	The pairing is based on standard 10-20 nomenclature:
	- Odd numbers (1, 3, 5, 7, 9) indicate left hemisphere
	- Even numbers (2, 4, 6, 8, 10) indicate right hemisphere
	- 'z' suffix indicates midline electrodes
	
	Only electrode pairs where BOTH electrodes exist in ch_names are
	included in the output dictionary.

	The resulting difference maps show:
	- Left hemisphere: contra - ipsi effect
	- Right hemisphere: ipsi - contra effect (inverse)
	- Midline: zero (electrode minus itself)

	Examples
	--------
	>>> ch_names = epochs.ch_names
	>>> pairs = get_diff_pairs(ch_names)
	>>> # For each epoch, compute lateralized differences
	>>> contra_idx, ipsi_idx = pairs['P7']
	>>> diff = data[:, contra_idx] - data[:, ipsi_idx]

	See Also
	--------
	select_electrodes : Select electrode subsets by region
	"""
	# Comprehensive electrode pairs based on 10-20 system
	# This covers 32, 64, 128+ channel systems
	all_pairs = {
		# Frontal polar
		'Fp1': 'Fp2', 'Fpz': 'Fpz',
		# Anterior frontal
		'AF9': 'AF10', 'AF7': 'AF8', 'AF5': 'AF6', 'AF3': 'AF4',
		'AF1': 'AF2', 'AFz': 'AFz',
		# Frontal
		'F9': 'F10', 'F7': 'F8', 'F5': 'F6', 'F3': 'F4',
		'F1': 'F2', 'Fz': 'Fz',
		# Frontal-temporal
		'FT9': 'FT10', 'FT7': 'FT8',
		# Frontal-central
		'FC5': 'FC6', 'FC3': 'FC4', 'FC1': 'FC2', 'FCz': 'FCz',
		# Temporal
		'T9': 'T10', 'T7': 'T8',
		# Central
		'C5': 'C6', 'C3': 'C4', 'C1': 'C2', 'Cz': 'Cz',
		# Temporal-parietal
		'TP9': 'TP10', 'TP7': 'TP8',
		# Central-parietal
		'CP5': 'CP6', 'CP3': 'CP4', 'CP1': 'CP2', 'CPz': 'CPz',
		# Parietal
		'P9': 'P10', 'P7': 'P8', 'P5': 'P6', 'P3': 'P4',
		'P1': 'P2', 'Pz': 'Pz',
		# Parietal-occipital
		'PO9': 'PO10', 'PO7': 'PO8', 'PO5': 'PO6', 'PO3': 'PO4',
		'PO1': 'PO2', 'POz': 'POz',
		# Occipital
		'O1': 'O2', 'Oz': 'Oz',
		# Inion
		'I1': 'I2', 'Iz': 'Iz'
	}

	# Filter to only include pairs where both electrodes exist
	idx_pairs = {}
	for contra, ipsi in all_pairs.items():
		if contra in ch_names and ipsi in ch_names:
			idx_pairs[contra] = (ch_names.index(contra), ch_names.index(ipsi))
			if contra != ipsi:  # Don't duplicate midline electrodes
				idx_pairs[ipsi] = (
					ch_names.index(ipsi), ch_names.index(contra)
				)

	if not idx_pairs:
		warnings.warn(
			'No standard electrode pairs found in channel names. '
			'Ensure channel names follow 10-20 naming conventions.'
		)

	return idx_pairs

def trial_exclusion(
	df: pd.DataFrame,
	epochs: mne.Epochs,
	excl_factor: dict
) -> Tuple[pd.DataFrame, mne.Epochs, np.ndarray]:
	"""
	Exclude trials based on experimental condition criteria.

	Removes trials matching specified conditions from both behavioral
	data and EEG epochs. Useful for excluding specific experimental
	conditions or error trials.

	Parameters
	----------
	df : pd.DataFrame
		Behavioral data with one row per trial. Modified in-place.
	epochs : mne.Epochs
		EEG epochs object. Modified in-place.
	excl_factor : dict
		Exclusion criteria as {column_name: [values_to_exclude]}.
		Multiple columns and values can be specified. All matching
		trials are removed (OR logic within each key, OR across keys).

	Returns
	-------
	df : pd.DataFrame
		Behavioral data with excluded trials removed, index reset.
	epochs : mne.Epochs
		EEG epochs with excluded trials removed.
	idx : np.ndarray
		Indices of excluded trials (before removal).

	Warnings
	--------
	Modifies df and epochs in-place. Data cannot be recovered after
	exclusion. Reload from file if needed again.

	Prints warning if no trials match exclusion criteria.

	Notes
	-----
	The function uses OR logic: a trial is excluded if it matches ANY
	of the specified values in ANY of the specified columns.

	Examples
	--------
	>>> # Exclude error trials
	>>> excl = {'correct': [0]}
	>>> df, epochs, idx = trial_exclusion(df, epochs, excl)
	>>> 
	>>> # Exclude multiple conditions
	>>> excl = {
	...     'cue_side': ['right'],
	...     'trial_type': ['practice', 'catch']
	... }
	>>> df, epochs, idx = trial_exclusion(df, epochs, excl)
	>>> print(f'Excluded {len(idx)} trials')

	See Also
	--------
	exclude_eye : Exclude trials based on eye movements
	support.FolderStructure.load_processed_epochs : Load with exclusion
	"""


	mask = [(df[key] == f).values for key in excl_factor.keys() 
		 							for f in excl_factor[key]]
	for m in mask: 
		mask[0] = np.logical_or(mask[0],m)
	mask = mask[0]

	idx = np.where(mask)[0]
	if mask.sum() > 0:
		epochs.drop(idx, reason='trial exclusion')
		df.drop(idx, inplace = True)
		df.reset_index(inplace = True, drop = True)	
		print(f'Dropped {sum(mask)} trials after specifying excl_factor')
		print('NOTE DROPPING IS DONE IN PLACE. ' \
		'PLEASE REREAD DATA IF THAT CONDITION IS NECESSARY AGAIN')
	else:
		print('Trial exclusion: no trials selected ' \
		'that matched specified criteria')

	return df, epochs, idx

def get_time_slice(
	times: np.ndarray,
	start_time: Optional[float],
	end_time: Optional[float],
	include_final: bool = True,
	step: Optional[int] = None
) -> slice:
	"""
	Convert time window to array slice indices.

	Finds array indices corresponding to specified start and end times,
	returning a slice object for indexing time-series data.

	Parameters
	----------
	times : np.ndarray
		Time points array in seconds.
	start_time : float or None
		Window start time in seconds. If None, uses first time point.
	end_time : float or None
		Window end time in seconds. If None, uses last time point.
	include_final : bool, default=True
		If True, includes the end time point in the slice. If False,
		excludes it (Python standard half-open interval).
	step : int, optional
		Step size for the slice. None means step=1. Can be used for
		downsampling. Default is None.

	Returns
	-------
	time_slice : slice
		Slice object for indexing: data[:, :, time_slice]

	Notes
	-----
	Uses np.argmin to find closest time points, so works even if
	exact times don't exist in the array.

	Examples
	--------
	>>> times = np.linspace(-0.2, 1.0, 601)  # -200 to 1000ms
	>>> # Get baseline window
	>>> baseline_slice = get_time_slice(times, -0.2, 0)
	>>> baseline_data = epochs_data[:, :, baseline_slice]
	>>> 
	>>> # Get post-stimulus window
	>>> post_slice = get_time_slice(times, 0, 0.5)
	>>> 
	>>> # Downsample by 2
	>>> downsampled_slice = get_time_slice(times, 0, 1.0, step=2)
	"""

	# get start and end index
	idx = [np.argmin(abs(times - t)) 
				for t in (start_time, end_time) if t is not None]
	if len(idx) == 0:
		idx = [0, times.size - 1]
	elif len(idx) == 1:
		if start_time is None:
			idx.insert(0,0)
		else:
			idx.insert(1,times.size - 1)

	s, e = idx
	if include_final:
		e += 1
	time_slice = slice(s, e, step)

	return time_slice

def create_cnd_loop(cnds: dict) -> list:
	"""
	Generate condition combinations for factorial experimental designs.

	Creates all possible combinations of experimental conditions and
	generates pandas query strings for selecting each combination from
	a DataFrame. Useful for factorial designs with multiple factors.

	Parameters
	----------
	cnds : dict
		Dictionary where keys are column names in behavioral DataFrame
		and values are lists of possible values for that factor.
		Example: {'cue_side': ['left', 'right'],
		          'validity': ['valid', 'invalid']}

	Returns
	-------
	filters : list of tuple
		List of (query_string, condition_name) tuples. Each tuple 
		contains:
			- query_string : str for DataFrame.query() or eval()
			- condition_name : str combining all factor values with '_'

	Notes
	-----
	Uses itertools.product to generate all factorial combinations.
	String values are automatically quoted in query strings.

	Examples
	--------
	>>> # 2x2 factorial design
	>>> cnds = {
	...     'cue_side': ['left', 'right'],
	...     'validity': ['valid', 'invalid']
	... }
	>>> filters = create_cnd_loop(cnds)
	>>> for query, name in filters:
	...     print(f"{name}: {query}")
	left_valid: cue_side == 'left' and validity == 'valid'
	left_invalid: cue_side == 'left' and validity == 'invalid'
	right_valid: cue_side == 'right' and validity == 'valid'
	right_invalid: cue_side == 'right' and validity == 'invalid'
	>>> 
	>>> # Use in analysis
	>>> for query, cnd_name in filters:
	...     cnd_data = df.query(query)
	...     # Analyze this condition...

	See Also
	--------
	trial_exclusion : Exclude trials based on conditions
	"""
	
	filters = []
	keys, values = zip(*cnds.items())
	for var_combo in [dict(zip(keys, v)) for v in itertools.product(*values)]:
		name = []
		for i, (k, v) in enumerate(var_combo.items()):
			name += [str(v)]
			if i == 0:
				if isinstance(v,str):
					df_filt = f'{k} == \'{v}\'' 
				else:
					df_filt = f'{k} == {v}'
			else:
				if isinstance(v,str):
					df_filt += f' and {k} == \'{v}\''
				else:
					df_filt += f' and {k} == {v}'
		
		filters.append((df_filt,'_'.join(name)))
	
	return filters

def log_preproc(
	idx: tuple,
	file: str,
	nr_sj: int = 1,
	nr_sessions: int = 1,
	to_update: Optional[dict] = None
) -> None:
	"""
	Log preprocessing parameters to multi-indexed CSV file.

	Creates or updates a preprocessing log file with multi-level 
	indexing(subject_id, session). Useful for tracking preprocessing 
	parameters across subjects and sessions.

	Parameters
	----------
	idx : tuple
		Multi-index tuple (subject_id, session) for the row to update.
		Example: (1, 1) for subject 1, session 1.
	file : str
		Path to CSV file for logging. Created if doesn't exist.
	nr_sj : int, default=1
		Number of subjects (used when creating new file).
	nr_sessions : int, default=1
		Number of sessions per subject (used when creating new file).
	to_update : dict, optional
		Dictionary of {column_name: value} pairs to log. If column
		doesn't exist, it's created. List values are converted to
		strings. Default is None (no updates).

	Returns
	-------
	None
		File is saved to disk.

	Warnings
	--------
	Modifies file in-place. All values are converted to strings.

	Notes
	-----
	The function:
	1. Loads existing file or creates new multi-indexed DataFrame
	2. Adds missing columns with NaN
	3. Updates specified row with new values
	4. Saves back to CSV

	Examples
	--------
	>>> # Log preprocessing parameters for subject 1, session 1
	>>> params = {
	...     'high_pass': 0.1,
	...     'low_pass': 30,
	...     'bad_channels': ['Fp1', 'Fp2'],
	...     'n_epochs': 450
	... }
	>>> log_preproc(
	...     idx=(1, 1),
	...     file='preprocessing/preproc_log.csv',
	...     to_update=params
	... )

	See Also
	--------
	support.FolderStructure.folder_tracker : Generate file paths
	"""

	# check whether file exists
	if os.path.isfile(file):
		df = pd.read_csv(file, index_col=[0,1])
	else:
		arrays = [np.arange(nr_sj) + 1, list(np.arange(1)+1)*nr_sj]
		multi_idx = pd.MultiIndex.from_arrays(
			arrays, names=('subject_id', 'session')
		)
		df = pd.DataFrame(None, multi_idx, to_update.keys())

	# do actual updating
	if to_update != None:
		for key, value in to_update.items():
			if key not in df:
				df[key] = np.nan 
			if type(value) == list:
				value = str(value) 
			df.loc[idx,key] = str(value)

	# save datafile
	df.to_csv(file)
