import os
import mne
import warnings
import itertools

import numpy as np
import pandas as pd

from numpy.lib.npyio import NpzFile
from typing import Optional, Generic, Union, Tuple, Any
from IPython import embed 
from math import sqrt
from sklearn.feature_extraction.image import grid_to_graph
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
from scipy.stats import t, ttest_rel

def select_electrodes(epochs:mne.Epochs,
						elec_oi: Union[list, str] = 'all') -> np.ndarray:
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
		- 'posterior': Posterior electrodes (parietal, occipital regions)
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


def baseline_correction(X:np.array,times:np.array,baseline:tuple) -> np.array:
	"""
	Perform baseline correction on 3D EEG data (epochs x electrodes x time).

	Parameters:
		data (np.ndarray): Input data of shape (epochs, electrodes, time).
		baseline_indices (tuple): Start and end indices for the baseline period.

	Returns:
		np.ndarray: Baseline-corrected data.
	"""
	# Check if the input data is 3D
	if X.ndim != 3:	
		raise ValueError("Input data must be 3D (epochs x electrodes x time).")
	
	# find indices of baseline period
	base_idx = get_time_slice(times, baseline[0], baseline[1])

	# Extract the baseline period
	baseline_data = X[:,:,base_idx]

	# Compute the mean over the baseline period (axis=-1 is time)
	baseline_mean = np.mean(baseline_data, axis=-1, keepdims=True)

	# Subtract the baseline mean from the entire data
	corrected_data = X - baseline_mean

	return corrected_data

def match_epochs_times(erps:list)->list:
	"""Finds all objects where the number of samples exceed those of the
	object with the lowest number of samples and truncates those objects
	""

	Args:
		erps (list): list of evoked objects

	Returns:
		list: list of evoked objects with same time axis
	"""

	# get min nr of samples
	min_samp_idx = np.argmin([e.times.size for e in erps])
	base_epoch = erps[min_samp_idx]
	min_samp = base_epoch.times.size
	
	# loop over all objects
	for i, e in enumerate(erps):
		if e.times.size > min_samp:
			upd_epoch = base_epoch.copy()
			upd_epoch._data = e._data[...,:min_samp]
			erps[i] = upd_epoch

	return erps

def get_diff_pairs(montage:str, ch_names:list)->dict:
	"""Returns a dictionary that allows for the creation of a 
	contralateral vs. ipsilateral topography, where left 
	electrodes depict contra - ipsi, and right electrodes depict 
	the inverse (it is assumed that left electrodes are contralateral
	to the effect of interest). Scores on the vertical midline will be 
	automatically set to zero.

	Args:
		montage (str): montage used during recording
		ch_names (list): list of channel names

	Returns:
		idx_pairs (dict): Dictionary containing electrode pair indices per
		electrode
	"""

	if montage == 'biosemi64':
		pairs = {'Fp1':'Fp2','Fpz':'Fpz','AF7':'AF8','AF3':'AF4','AFz':'AFz',
				'F7':'F8','F5':'F6','F3':'F4','F1':'F2','Fz':'Fz','FT7':'FT8',
				'FC5':'FC6','FC3':'FC4','FC1':'FC2','FCz':'FCz','T7':'T8',
				'C5':'C6','C3':'C4','C1':'C2','Cz':'Cz','TP7':'TP8',
				'CP5':'CP6','CP3':'CP4','CP1':'CP2','CPz':'CPz','P9':'P10',
				'P7':'P8','P5':'P6','P3':'P4','P1':'P2','Pz':'Pz',
				'PO7':'PO8','PO3':'PO4','POz':'POz','O1':'O2','Oz':'Oz',
				'Iz':'Iz'}

	idx_pairs = {}
	for contra, ipsi in pairs.items():
		idx_pairs[contra] = (ch_names.index(contra), ch_names.index(ipsi))
		if contra != ipsi:
			idx_pairs[ipsi] = (ch_names.index(ipsi), ch_names.index(contra))

	return idx_pairs


def exclude_eye(sj:int,beh:pd.DataFrame,epochs:mne.Epochs,
				eye_dict:dict,eye: Optional[NpzFile] = None,preproc_file:str=None)->\
				Tuple[pd.DataFrame,mne.Epochs]:
	"""
	Filters out eye movements based on either a step algorhytm or 
	deviation from fixation as measured with the eyetracker. In the 
	latter case it is required that eye tracker data is linked to the 
	epochs object during preprocessing (see EEG.link_eye)

	Args:
		sj (int): subject identifier
		beh (pd.DataFrame): behavioral data
		epochs (mne.Epochs): epoched data
		eye_dict (dict): parameters used to mark eye movement artefacts.
		Supports the following keys:
			eye_window (tuple): window used to search for eye movements.
			Defaults to entire window
            eye_ch (str): channel to search for eye movements. Defaults
			to 'HEOG'
            angle_thresh (float): threshold in degrees of visual angles
            step_param (dict): tuple, containing window size, window 
			step and thershold used to detect artefacts using sliding 
			window approach. Defaults to (200, 10, 20) 
			use_tracker (bool): should eye tracker data be used to 
            search eye movements. If not exclusion will be based on 
            step algorhytm applied to eog channel as specified in eye_ch
		preproc_file (str): name specified for specific 
            preprocessing pipeline. Used to update the preprocessing 
			document
	Returns:
        beh (pd.DataFrame): behavioral data alligned to eeg data
        epochs (mne.Epochs): preprocessed eeg data
	"""

	# initialize some parameters
	if 'drift_correct' in eye_dict:
		drift_correct = eye_dict['drift_correct'] 
	else:
		drift_correct = False
	if 'window_oi' not in eye_dict:
		eye_dict['window_oi'] = (epochs.tmin, epochs.tmax)
	if 'eye_ch' not in eye_dict:
		print('Eye channel is not specified in eyedict, using HEOG as default')
		eye_dict['eye_ch'] = 'HEOG'
	if 'use_eog' not in eye_dict:
		eye_dict['use_eog'] = True

	# specify window of interest
	s, e = eye_dict['window_oi']	
	if drift_correct:
		if drift_correct[0] < s:
			s = drift_correct[0]	

	# check whether selection should be based on eyetracker data
	if 'use_tracker' not in eye_dict or not eye_dict['use_tracker']:
		tracker_bins = np.full(beh.shape[0], np.nan)
		perc_tracker = 'no tracker'
		window_idx = get_time_slice(epochs.times,s,e)
	else:
		if isinstance(eye,NpzFile) or ('x' in epochs.ch_names):
			if eye is not None:
				x, y, times = eye['x'], eye['y'], eye['times']
				sfreq = int(eye['sfreq'])
				window_idx = get_time_slice(times,s,e)
				x = x[:,window_idx]
				y = y[:,window_idx]
				times = times[window_idx]
			else:
				window_idx = get_time_slice(epochs.times,s,e)
				x = epochs._data[:, epochs.ch_names.index('x'), window_idx]
				y = epochs._data[:, epochs.ch_names.index('y'), window_idx]
				times = epochs.times[window_idx]
				sfreq = epochs.info['sfreq']

			from eeg_analyses.EYE import EYE, SaccadeGlissadeDetection
			EO = EYE(sfreq = sfreq,
					viewing_dist = eye_dict['viewing_dist'],
					screen_res = eye_dict['screen_res'],
					screen_h = eye_dict['screen_h'])
			angles = EO.angles_from_xy(x.copy(), y.copy(), times, drift_correct)
			if eye_dict['window_oi'][0] > times[0]:
				window_idx = get_time_slice(times, eye_dict['window_oi'][0], e)
				angles_oi = np.array(angles)[:,window_idx]
			else:
				angles_oi = np.array(angles)
			min_samples = 40 * epochs.info['sfreq'] / 1000  # 40 ms
			tracker_bins = bin_tracker_angles(angles_oi, eye_dict['angle_thresh'], min_samples)
			perc_tracker = np.round(sum(tracker_bins == 1) / tracker_bins.size * 100, 1)
		else:
			warnings.warn('No eye tracker data found, \n'
						'skipping eye tracker based exclusion')
			perc_tracker = 'no tracker data found'
			tracker_bins = np.full(beh.shape[0], np.nan)
		# temp code for docky				
		# elif 'eye_bins' in beh:
		# 	tracker_bins = beh.eye_bins.values
		# 	tracker_bins[tracker_bins <= eye_dict['angle_thresh']] = 0
		# 	tracker_bins[tracker_bins > eye_dict['angle_thresh']] = 1
		# 	perc_tracker = np.round(sum(tracker_bins == 1)/ 
		# 			sum(tracker_bins < 2)*100,1)

	# apply step algorhytm to trials with missing data
	nan_idx = np.where(np.isnan(tracker_bins) > 0)[0]
	if nan_idx.size > 0 and eye_dict['use_eog']:
		eye_ch = eye_dict['eye_ch']
		eog = epochs._data[nan_idx,epochs.ch_names.index(eye_ch),window_idx]
		if 'step_param' not in eye_dict:
			size, step, thresh = (200, 10, 15e-6)
		else:
			size, step, thresh = eye_dict['step_param']
		idx_art = eog_filt(eog,sfreq = epochs.info['sfreq'], windowsize = size, 
								windowstep = step, thresh = thresh)
		tracker_bins[nan_idx[idx_art]] = 2
		perc_eog = np.round(sum(tracker_bins == 2)/ tracker_bins.size*100,1)
		print('{} trials missing eyetracking'.format(len(nan_idx)))
		print('data (used eog instead)')
	else:
		perc_eog = 'eog not used for exclusion'

	perc_eye = np.round(sum(tracker_bins >= 1)/ tracker_bins.size*100,1)
	# if it exists update preprocessing information
	if os.path.isfile(preproc_file):
		print('Eye exclusion info saved in preprocessing file (at session 1')
		idx = (sj, 1)
		df = pd.read_csv(preproc_file, index_col=[0,1],on_bad_lines='skip')
		df.loc[idx,'% tracker'] = f'{perc_tracker}%' 
		df.loc[idx,'% eog'] = f'{perc_eog}% (N = {nan_idx.size})'
		df.loc[idx,'eye_excl'] = f'{perc_eye}%' 

		# save datafile
		df.to_csv(preproc_file)

	# remove trials from behavior and eeg
	if 'level_0' not in beh:
		beh.reset_index(inplace = True)
	else:
		beh.drop('level_0', axis=1, inplace=True)
		beh.reset_index(inplace = True, drop = True)
	to_drop = np.where(tracker_bins >= 1 )[0]	
	epochs.drop(to_drop, reason='eye detection')
	beh.drop(to_drop, inplace = True, axis = 0)
	beh.reset_index(inplace = True, drop = True)

	return beh, epochs

def bin_tracker_angles(angles:np.array,thresh:float,min_samp:float)->np.array:
	"""
	Summarizes eye tracker data per trial with a single value that 
	indicates whether that trial had a segment of data that did exceed 
	the specified threshold in visual angle (1) or not (0). 

	Args:
		angles (np.array): deviation from fixation in visual angle
		thresh (float): max deviation threshold to mark for exclusion
		min_samp (float): minumum number of consecutive samples where 
		the threshold should be exceeded (controls for eyetracker noise)

	Returns:
		tracker_bins (np.array): array that indicates per trial whether 
		or not that trial should be excluded. Trials with missing eye 
		tracker data are indicated via nan
	"""

	#TODO: how to deal with trials without data
	tracker_bins = []
	for i, angle in enumerate(angles):
		# get data where deviation from fix is larger than thresh
		binned = np.where(angle > thresh)[0]
		segments = np.split(binned, np.where(np.diff(binned) != 1)[0]+1)

		# check whether a segment exceeds min duration
		if np.where(np.array([s.size for s in segments])>min_samp)[0].size > 0:
			tracker_bins.append(1)
		elif np.any(np.isnan(angle)):
			tracker_bins.append(np.nan)
		else:
			tracker_bins.append(0)

	return np.array(tracker_bins)

def eog_filt(eog:np.array,sfreq:float,windowsize:int=200,
			windowstep:int=10,thresh:int=30)->np.array:
	"""
	Split-half sliding window approach. This function slides a window 
	in prespecified steps over specified data. If the change in voltage 
	from the first half to the second half of the window is greater 
	than the specified threshold, this trial is marked for rejection

	Args:
		eog (np.array): nr epochs X times. Data used for trial rejection
		sfreq (float): digitizing frequency
		windowsize (int, optional): size of the sliding window (in ms). 
		Defaults to 200.
		windowstep (int, optional): step size to slide window over the 
		trial. Defaults to 10.
		thresh (int, optional): threshold in microvolt. Defaults to 30.

	Returns:
		eye_trials (np.array): indices of marked epochs
	"""

	# shift miliseconds to samples
	windowstep /= 1000.0 / sfreq
	windowsize /= 1000.0 / sfreq
	s, e = 0, eog.shape[-1]

	# create multiple windows based on window parameters 
	# (accept that final samples of epoch may not be included) 
	window_idx = [(i, i + int(windowsize)) 
					for i in range(s, e, int(windowstep)) 
					if i + int(windowsize) < e]

	# loop over all epochs and store all eye events into a list
	eye_trials = []
	for i, x in enumerate(eog):
		
		for idx in window_idx:
			window = x[idx[0]:idx[1]]

			w1 = np.mean(window[:int(window.size/2)])
			w2 = np.mean(window[int(window.size/2):])

			if abs(w1 - w2) > thresh:
				eye_trials.append(i)
				break

	eye_trials = np.array(eye_trials, dtype = int)			

	return eye_trials

def trial_exclusion(df, epochs, excl_factor):


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

def get_time_slice(times, start_time, end_time, include_final = True, step = None):

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

def filter_eye(beh, eeg, eye_window, eye_ch = 'HEOG', eye_thresh = 1, eye_dict = None, use_tracker = True):
	"""Filters out data based on eye movements. Either based on eye tracker data 
	as stored in the beh file or using a step like algorhythm

	
	Arguments:
		beh {dataframe}  -- behavioral info after preprocessing
		eeg {epochs object} -- preprocessed eeg data
		eye_window {tuple | list} -- Time window used for step algorhythm
		eye_ch {str} -- Name of channel used to detect eye movements
		eye_thresh {array} -- threshold in visual degrees. Used for eye tracking data
		eye_dict (dict) -- dictionry with three parameters (specified as keys) for step algorhytm: 
						windowsize (in ms), windowstep (in ms), threshold (in microV)
		use_tracker (bool) -- specifies whether eye tracker data should be used (i.e., is reliable)				
	
	Returns:
		beh {dataframe}  -- behavioral info with trials with eye movements removed
		eeg {epochs object} -- preprocessed eeg data with eye movements removed
		"""

	if not use_tracker or 'eye_bins' not in beh:
		beh['eye_bins'] = np.nan
	nan_idx = np.where(np.isnan(beh['eye_bins']) > 0)[0]
	print('Trials without reliable eyetracking data {} out of {} clean trials ({}%)'.format(nan_idx.size, beh['eye_bins'].size, nan_idx.size/float(beh['eye_bins'].size)*100))

	# limit step algorhytm to trials without eye tracking data
	if nan_idx.size > 0:
		s,e = [np.argmin(abs(eeg.times - t)) for t in eye_window]
		eog = eeg._data[nan_idx,eeg.ch_names.index(eye_ch),s:e]

		if eye_dict != None:

			idx_eye = eog_filt(eog, sfreq = eeg.info['sfreq'], windowsize = eye_dict['windowsize'], 
								windowstep = eye_dict['windowstep'], threshold = eye_dict['threshold'])
			beh['eye_bins'][nan_idx[idx_eye]] = 99	
	
	# remove trials from beh and eeg objects
	beh.reset_index(inplace = True)
	to_drop = np.where(beh['eye_bins'] > eye_thresh)[0]	
	eeg.drop(to_drop, reason='eye detection')
	print('Dropped {} trials based on threshold criteria ({})%'.format(to_drop.size, to_drop.size/float(beh['eye_bins'].size)*100))
	beh.drop(to_drop, inplace = True)
	beh.reset_index(inplace = True, drop = True)

	return beh, eeg


def create_cnd_loop(cnds):
	
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



def log_preproc(idx, file, nr_sj = 1, nr_sessions = 1, to_update = None):

	# check whether file exists
	if os.path.isfile(file):
		df = pd.read_csv(file, index_col=[0,1])
	else:
		arrays = [np.arange(nr_sj) + 1, list(np.arange(1)+1)*nr_sj]
		multi_idx = pd.MultiIndex.from_arrays(arrays, names = ('subject_id', 'session'))
		df = pd.DataFrame(None,multi_idx, to_update.keys())

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




def cnd_baseline(EEG, beh, cnds, cnd_header, base_period = (-0.1, 0), nr_elec = 64):
	"""Does baselining per condition

	Arguments:

	
	Returns:
		EEG {object} -- baseline corrected MNE epochs object
	"""

	# select indices baseline period
	start, end = [np.argmin(abs(EEG.times - b)) for b in base_period]

	# loop over conditions
	for cnd in cnds:

		# get indices of interest
		idx = np.where(beh[cnd_header] == cnd)[0]

		# get data
		X = EEG._data[idx,:nr_elec]
		# get base_mean (per electrode)
		X_base = X[:,:,start:end].mean(axis = (0,2))

		#do baselining per electrode
		for i in range(nr_elec):
			X[:,i,:] -= X_base[i] 

		EEG._data[idx,:nr_elec] = X	

	return EEG

def cnd_time_shift(EEG, beh, cnd_info, cnd_header):
	"""This function shifts the timings of all epochs that meet a specific criteria. 
	   Can be usefull when events of interest in different conditions are not aligned

	Arguments:
		EEG {object} -- MNE Epochs object
		beh {dataframe} -- Dataframe with behavior info
		cnd_info {dict} -- For each key in cnd_info data will be shifted according to 
							the specified time (in seconds). E.G., {neutral: 0.1}
		cnd_header {str} -- column in beh that contains the keys specified in cnd_info
	
	Returns:
		EEG {object} -- MNE Epochs object with shifted timings
	"""

	print('Data will be artificially shifted. Be carefull in selecting the window of interest for further analysis')
	print('Original timings range from {} to {}'.format(EEG.tmin, EEG.tmax))
	# loop over all conditions
	for cnd in cnd_info.keys():
		# set how much data needs to be shifted
		to_shift = cnd_info[cnd]
		to_shift = int(np.diff([np.argmin(abs(EEG.times - t)) for t in (0,to_shift)]))
		if to_shift < 0:
			print('EEG data is shifted backward in time for all {} trials'.format(cnd))
		elif to_shift > 0:
			print('EEG data is shifted forward in time for all {} trials'.format(cnd))	

		# find indices of epochs to shift
		mask = (beh[cnd_header] == cnd).values

		# do actual shifting
		EEG._data[mask] = np.roll(EEG._data[mask], to_shift, axis = 2)

	return EEG







	


def confidence_int(data, p_value = .05, tail='two', morey=True):
	'''

	Cousineau's method (2005) for calculating within-subject confidence intervals
	If needed, Morey's correction (2008) can be applied (recommended).

	Arguments
	----------
	data (array): Data for which CIs should be calculated
	p_value (float): p-value for determining t-value (the default is .05).
	tail (string): Two-tailed ('two') or one-tailed t-value.
	morey (bool), Apply Morey correction (the default is True)

	Returns
	-------
	CI (array): Confidence intervals for each condition
	'''

	if tail=='two':
		p_value = p_value/2
	elif tail not in ['two','one']:
		p_value = p_value/2        
		warnings.warn('Incorrect argument for tail: using default')
	
	# normalize the data by subtracting the participants mean performance from each observation, 
	# and then add the grand mean to each observation
	ind_mean = np.nanmean(data, axis=1).reshape(data.shape[0],1)
	grand_mean = np.nanmean(data, axis=1).mean()
	data = data - ind_mean + grand_mean
	# Look up t-value and caluclate CIs
	t_value = abs(t.ppf([p_value], data.shape[0]-1)[0])
	CI = np.nanstd(data, axis=0, ddof=1)/sqrt(data.shape[0])*t_value

	# correct CIs according to Morey (2008)
	if morey:
		CI = CI*(data.shape[1]/float((data.shape[1] - 1))) 

	return CI 

def curvefitting(x, y, bounds, func = lambda x, a, d: d + (1 - d) * a**x):
	'''

	'''

	from scipy.optimize import curve_fit
	popt, pcov = curve_fit(func, x, y, bounds = bounds)

	return popt, pcov

def permTestMask1D(diff, p_value = 0.05):
	'''

	'''

	T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)	
	print(cluster_pv)
	mask = np.zeros(diff[0].shape[1],dtype = bool)
	sig_clusters = []
	for cl in np.array(clusters)[np.where(cluster_pv < p_value)[0]]:
		mask[cl[0]] = True	
		sig_clusters.append(cl)

	return mask, sig_clusters	

def permTestMask2D(diff, p_value = 0.05):
	'''

	'''

	a = np.arange(diff[0].shape[1] * diff[0].shape[2]).reshape((diff[0].shape[1], diff[0].shape[2]))
	adj = connected_adjacency(a, '8').toarray()
	b = grid_to_graph(diff[0].shape[1], diff[0].shape[2]).toarray()
	conn = np.array(np.array(np.add(adj,b), dtype = bool),dtype = int)
	conn = sparse.csr_matrix(conn)

	#T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t, connectivity = conn)	
	T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)	
	#T_obs, clusters, cluster_pv, HO = spatio_temporal_cluster_test(diff, stat_fun = paired_t, connectivity = conn)	
	#T_obs, clusters, cluster_pv, HO = spatio_temporal_cluster_test(diff, stat_fun = paired_t)
	T_obs_plot = np.nan * np.ones_like(T_obs)
	print(cluster_pv)
	for c, p_val in zip(clusters, cluster_pv):
		if p_val <= p_value:
			print(c.sum())
			T_obs_plot[c] = T_obs[c]

	return T_obs_plot	

def paired_t(*args):
	'''
	Call scipy.stats.ttest_rel, but return only f-value
	'''
	
	return ttest_rel(*args)[0]

def bootstrap(X, b_iter = 1000):
	'''
	bootstrapSlopes uses a bootstrap procedure to calculate standard error of slope estimates.

	Arguments
	- - - - - 
	test

	Returns
	- - - -

	'''	

	nr_obs = X.shape[0]
	bootstrapped = np.zeros((b_iter,X.shape[1]))

	for b in range(b_iter):
		idx = np.random.choice(nr_obs,nr_obs,replace = True) 				# sample nr subjects observations from the slopes sample (with replacement)
		bootstrapped[b,:] = np.mean(X[idx,:],axis = 0)

	error = np.std(bootstrapped, axis = 0)
	mean = X.mean(axis = 0)

	return error, mean

def permTTest(real, perm, nr_perms = 1000, p_thresh = 0.01):
	'''
	permTTest calculates p-values for the one-sample t-stat for each sample point across frequencies using a surrogate distribution generated with 
	permuted data. The p-value is calculated by comparing the t distribution of the real and the permuted slope data across sample points. 
	The t-stats for both distribution is calculated with

	t = (m - 0)/SEm

	, where m is the sample mean slope and SEm is the standard error of the mean slope (i.e. stddev/sqrt(n)). The p value is then derived by dividing 
	the number of instances where the surrogate T value across permutations is larger then the real T value by the number of permutations.  

	Arguments
	- - - - - 
	real(array):  
	perm(array): 
	p_thresh (float): threshold for significance. All p values below this value are considered to be significant

	Returns
	- - - -
	p_val (array): array with p_values across frequencies and sample points
	sig (array): array with significance indices (i.e. 0 or 1) across frequencies and sample points
	'''

	# get number observations
	nr_obs = real.shape[0]
	nr_perms = perm.shape[-1]

	# preallocate arrays
	p_val = np.empty((real.shape[1],real.shape[2])) * np.nan
	sig = np.zeros((real.shape[1],real.shape[2])) 	# will be filled with 0s (non-significant) and 1s (significant)

	# calculate the real and the surrogate one-sample t-stats
	r_M = np.mean(real, axis = 0); p_M = np.mean(perm, axis = 0)
	r_SE = np.std(real, axis = 0)/sqrt(nr_obs); p_SE = np.std(perm, axis = 0)/sqrt(nr_obs)
	r_T = r_M/r_SE; p_T = p_M/p_SE

	# calculate p-values
	for f in range(real.shape[1]):
		for s in range(real.shape[2]):
			surr_T = p_T[f,s,:]
			p_val[f,s] = len(surr_T[surr_T>r_T[f,s]])/float(nr_perms)
			if p_val[f,s] <= p_thresh:
				sig[f,s] = 1

	return p_val, sig


def connected_adjacency(image, connect, patch_size=(1, 1)):
	"""
	Creates an adjacency matrix from an image where nodes are considered adjacent 
	based on 4-connected or 8-connected pixel neighborhoods.

	:param image: 2 or 3 dim array
	:param connect: string, either '4' or '8'
	:param patch_size: tuple (n,m) used if the image will be decomposed into 
	               contiguous, non-overlapping patches of size n x m. The 
	               adjacency matrix will be formed from the smaller sized array
	               e.g. original image size = 256 x 256, patch_size=(8, 8), 
	               then the image under consideration is of size 32 x 32 and 
	               the adjacency matrix will be of size 
	               32**2 x 32**2 = 1024 x 1024
	:return: adjacency matrix as a sparse matrix (type=scipy.sparse.csr.csr_matrix)
	"""	
	

	r, c = image.shape[:2]

	r = r / patch_size[0]
	c = c / patch_size[1]

	if connect == '4':
		# constructed from 2 diagonals above the main diagonal
		d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
		d2 = np.ones(c*(r-1))
		upper_diags = s.diags([d1, d2], [1, c])
		return upper_diags + upper_diags.T

	elif connect == '8':
		# constructed from 4 diagonals above the main diagonal
		d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
		d2 = np.append([0], d1[:c*(r-1)])
		d3 = np.ones(c*(r-1))
		d4 = d2[1:-1]
		upper_diags = sparse.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
		return upper_diags + upper_diags.T
	else:
		raise ValueError('Invalid parameter \'connect\'={connect}, must be "4" or "8".'
				.format(connect=repr(connect)))