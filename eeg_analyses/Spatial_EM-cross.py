"""
analyze EEG data

Created by Dirk van Moorselaar on 30-03-2016.
Copyright (c) 2016 DvM. All rights reserved.
"""

from helperFunctions import *
from FolderStructure import FolderStructure
import random
import glob, os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXP_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import h5py
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from math import pi, sqrt
import pandas as pd
import mne
from scipy.signal import hilbert
from scipy.stats import norm
from scipy.ndimage.measurements import label
import pickle
import json
import time
from IPython import embed as shell

class SpatialEM(FolderStructure):
	'''
	Spatial encoding scripts modeled after: "The topography of alpha-band activity tracks the content of spatial working memory" by Foster et al. (2015).
	Scipts based on Matlab scripts published on open science Framework (https://osf.io/bwzjj/) and lab visit to Chicago University (spring quarter 2016).
	'''

	def __init__(self, channel_folder, decoding, nr_iter, nr_blocks, nr_bins, nr_chans, delta):
		''' 
		Init function needs to be adjusted to match python pipeline input. At the moment only works with preprocessed matlab data (EEG.mat)
		Arguments
		- - - - - 
		channel_folder (str): folder specifying which electrodes are used for CTF analysis (e.g. posterior channels)
		decoding (str)| Default ('memory'): String specifying what to decode. Defaults to decoding of spatial memory location, 
		but can be changed to different locations (e.g. decoding of the location of an intervening stimulus).
		nr_iter (int):  number iterations to apply forward model 	
		nr_blocks (str): number of blocks to apply forward model
		nr_bins (int): number of location bins used in experiment
		nr_chans (int): number of hypothesized underlying channels

		Returns
		- - - -
		self (object): SpatialEM object
		'''

		self.channel_folder = channel_folder
		self.decoding = decoding
		self.project_folder = '/home/dvmoors1/big_brother/Spatial_EM_Ch' # Here specify the project folder
		#os.chdir(os.path.join(os.getcwd(), 'ctf', channel_folder, decoding))
		
		# specify model parameters
		self.nr_bins = nr_bins													# nr of spatial locations 
		self.nr_chans = nr_chans 												# underlying channel functions coding for spatial location
		self.nr_iter = nr_iter													# nr iterations to apply forward model 		
		self.nr_blocks = nr_blocks												# nr blocks to split up training and test data with leave one out test procedure
		self.sfreq = 512
		self.basisset = self.calculateBasisset(self.nr_bins, self.nr_chans)		# hypothesized set tuning functions underlying power measured across electrodes

	def behaviorCTF(self, subject_id, nr_practice, conditions, bin_header, condition_header):
		'''
		behaviorCTF reads in position bins and condition info from a csv file. Function assumes that memory location bins are specified in degrees.
		
		Arguments
		- - - - - 
		subject_id (int): subject_id
		nr_practice(int): nr of practice trials that need to be ignored in the csv file
		conditions (list): list of condition names that are specified in the csv file
		bin_header (str): column name in csv file containing the location bins (in degrees)
		condition_header (str): column name in csv file containing the conditions (str)

		Returns
		- - - -
		pos_bins (array): array of position bins
		condition (array): array of conditions
		'''

		try:
			data = pd.read_csv(self.FolderTrackerCTF('behavior', filename = 'subject-' + str(subject_id) + '.csv'))
		except:
			print(file + ' cannot be read in into data frame. Check fileformat')

		pos_bins = np.array(data[bin_header][nr_practice:]) 		# read in experimental trials only and adjust degrees to bins
		if 'bin_mem' in bin_header:
			pos_bins /= 45

		condition = np.array(data[condition_header])[nr_practice:]					
		for i, cond in enumerate(condition):
			condition[i] = conditions.index(cond)	
		
		return pos_bins, condition

	def readData(self, subject_id, conditions, thresh_bin = 1, eye_window = (-0.3, 0.8)):
		'''
		behaviorCTF reads in position bins and condition info from a csv file. Function assumes that memory location bins are specified in degrees.
		
		Arguments
		- - - - - 
		subject_id (int): subject_id
		conditions (list): list of condition names that are specified in the pickle file with key condition

		Returns
		- - - -
		pos_bins (array): array of position bins
		condition (array): array of conditions
		eegs (array): array with eeg data
		thresh_bin (int):
		eye_window
		'''

		# read in processed behavior from pickle file
		with open(self.FolderTracker(extension = ['beh','processed'], filename = 'subject-{}_all.pickle'.format(subject_id)),'rb') as handle:
			beh = pickle.load(handle)

		# read in eeg data 
		EEG = mne.read_epochs(self.FolderTracker(extension = ['processed'], filename = 'subject-{}_all-epo.fif'.format(subject_id)))

		# select electrodes
		picks = self.selectChannelId(EEG, self.channel_folder)
		eegs = EEG._data[:,picks,:]

		# select conditions from pickle file	
		if conditions == 'all':
			cnd_mask = np.ones(beh['condition'].size, dtype = bool)
		else:	
			cnd_mask = np.array([cnd in conditions for cnd in beh['condition']])

		# exclude trials contaminated by unstable eye position
		nan_idx = np.where(np.isnan(beh['eye_bins']) > 0)[0]
		s,e = [np.argmin(abs(EEG.times - t)) for t in eye_window]
		heog = EEG._data[:,EEG.ch_names.index('HEOG'),s:e]

		eye_trials = eog_filt(beh, EEG, heog, sfreq = EEG.info['sfreq'], windowsize = 50, windowstep = 25, threshold = 30)
		beh['eye_bins'][eye_trials] = 99

		# use mask to select conditions and position bins (flip array for nans)
		eye_mask = ~(beh['eye_bins'] > thresh_bin)

		cnds = beh['condition'][cnd_mask * eye_mask] 
		pos_bins = beh[self.decoding][cnd_mask * eye_mask]

		eegs = eegs[cnd_mask * eye_mask ,:,:]

		return pos_bins, cnds, eegs, EEG

	def calculateBasisset(self, nr_bins = 8, nr_chans = 8, sin_power = 7):
		'''
		calculateBasisset returns a basisset that is used to reconstruct location-selective CTFs from the topographic distribution 
		of oscillatory power across electrodes. It is assumed that power measured at each electrode reflects the weighted sum of 
		a specific number of spatial channels (i.e. neural populations), each tuned for a different angular location. The response 
		profile of each channel across angular locations is modeled as a half sinusid raised to the sin_power, given by: 

		R = sin(0.50)**sin_power

		,where 0 is angular location and R is response of the spatial channel in arbitrary units. This R is then circularly shifted 
		for each channel such that the peak response of each channel is centered over one of the location bins.
		
		Arguments
		- - - - - 
		nr_bins (int): nr of spatial locations to decode
		nr_chans(int): assumed nr of spatial channels 
		sin_power (int): power for sinussoidal function

		Returns
		- - - -
		bassisset(array): set of basis functions used to predict channel responses in forward model
		'''

		# specify basis set
		if nr_bins % 2 == 0:
			x = np.linspace(0, 2*pi - 2*pi/nr_bins, nr_bins)
		else:
			x = np.linspace(0, 2*pi - 2*pi/nr_bins, nr_bins) + np.deg2rad(180/nr_bins)

		c_centers = np.linspace(0, 2*pi - 2*pi/nr_chans, nr_chans)			
		c_centers = np.rad2deg(c_centers)

		from math import radians
		pred = np.sin(0.5*x)**sin_power									# hypothetical channel responses
		
		if nr_bins % 2 == 0:											# shift the initial basis function	
			pred = np.roll(pred,nr_chans - (np.argmax(pred) + 1))
		else:
			pred = np.roll(pred,np.argmax(pred))	

		basisset = np.zeros((nr_chans,nr_bins))
		for c in range(nr_chans):
			basisset[c,:] = np.roll(pred,c+1)
	
		return basisset	

	def powerAnalysis(self, data, tois, sfreq, band, downsample):
		'''
		powerAnalysis bandpass filters raw EEG signal to isolate frequency-specific activity using a 5th order butterworth filter (different then original FIR filter). 
		A Hilbert Transform is then applied to the filtered data to extract the complex analytic signal. To extract the total power this signal is computed by squaring 
		the complex magnitude of the complex signal (is done within SpatialCTF). 

		Arguments
		- - - - - 
		data(array): raw eeg data (epochs x nr_electrodes x nr_samps)
		tois(array): boolean index to remove times that are not of interest after filtering 
		sfreq(int): sampling rate in Hz
		band(list): lower and higher cut-off value for bandpass filter
		downsample(int): factor to downsample data after filtering

		Returns
		- - - -
		data_evoked(array): evoked power (ongoing activity irrespective of phase relation to stimulus onset)
		data_total(array): total power (phase-locked activity)
		'''	

		evoked = np.empty((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex_) * np.nan; 
		total = np.copy(evoked)

		# Filter each electrode seperately
		for tr in range(data.shape[0]): 
			evoked[tr,:,:] = hilbert(mne.filter.filter_data(data[tr,:,:], sfreq,band[0], band[1], method = 'iir', iir_params = dict(ftype = 'butterworth', order = 5)))
			total[tr,:,:] = np.abs(hilbert(mne.filter.filter_data(data[tr,:,:], sfreq,band[0], band[1], method = 'iir', iir_params = dict(ftype = 'butterworth', order = 5))))**2

		# trim filtered data to remove times that are not of interest (after filtering to avoid artifacts)
		evoked = evoked[:,:,tois]
		total = total[:,:,tois]

		# downsample to reduced sample rate (after filtering so that downsampling doesn't affect filtering)	
		evoked = evoked[:,:,np.arange(0,tois.sum(),downsample)]
		total = total[:,:,np.arange(0,tois.sum(),downsample)]

		return evoked, total

	def forwardModel(self, data, labels, c, block_nr, block, bins):
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
		data (array): array of training and test data for each electrode
		labels(): training and test labels 
		c (array): basisset
		block_nr (array): array that contains block numbers used to seperate training and test blocks
		block(int): used to specify test block. The other blocks are automatically used as training blocks
		bins(array): all bins used in IEM

		Returns
		- - - -
		C2(array): unshifted predicted channel response
		C2s(array): shifted predicted channels response
		'''

		idx_train = np.hstack(block_nr != block)
		idx_test = np.hstack(block_nr == block)

		train_label = labels[idx_train]							# training labels
		test_label = labels[idx_test]							# test labels

		###### POWER ANALYSIS ######
		B1 = data[idx_train,:] 									# training data (nr training blocks x nr_electrodes)
		B2 = data[idx_test,:]									# test data (nr test blocks x nr_electrodes)
		C1 = c[idx_train,:]										# predicted channel outputs for training data (nr training blocks x nr_chans)
		W, resid_w, rank_w, s_w = np.linalg.lstsq(C1,B1)		# estimate weight matrix W (nr_chans x nr_electrodes)
		C2, resid_c, rank_c, s_c = np.linalg.lstsq(W.T,B2.T)	# estimate channel response C2 (nr_chans x nr test blocks) 

		# transpose C2 (makes sure that averaging and shifting is done across channels)
		C2 = C2.T
		C2s = np.copy(C2)

		# shift eegs to common center
		nr_2_shift = int(np.ceil(C2.shape[1]/2.0))
		for i in range(C2.shape[0]):
			idx_shift = abs(bins - test_label[i]).argmin()
			shift = idx_shift - nr_2_shift
			if self.nr_bins % 2 == 0:							# CHECK THIS: WORKS FOR 5 AND 8 BINS
				C2s[i,:] = np.roll(C2[i,:], - shift)	
			else:
				C2s[i,:] = np.roll(C2[i,:], - shift - 1)			

		return C2, C2s, W 											# return the unshifted and the shifted channel response

	def forwardModelCross(self, train_data, test_data, c, bins):
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
		data (array): array of training and test data for each electrode
		labels(): training and test labels 
		c (array): basisset
		bins(array): all bins used in IEM

		Returns
		- - - -
		C2(array): unshifted predicted channel response
		C2s(array): shifted predicted channels response
		'''

		###### POWER ANALYSIS ######
		B1 = train_data											# training data 
		B2 = test_data											# test data 
		C1 = c													# predicted channel outputs for training data (nr training blocks x nr_chans)
		W, resid_w, rank_w, s_w = np.linalg.lstsq(C1,B1)		# estimate weight matrix W (nr_chans x nr_electrodes)
		C2, resid_c, rank_c, s_c = np.linalg.lstsq(W.T,B2.T)	# estimate channel response C2 (nr_chans x nr test blocks) 

		# transpose C2 (makes sure that averaging and shifting is done across channels)
		C2 = C2.T
		C2s = np.copy(C2)

		# shift eegs to common center
		nr_2_shift = int(np.ceil(C2.shape[1]/2.0))
		for i in range(C2.shape[0]):
			idx_shift = abs(bins - i).argmin()
			shift = idx_shift - nr_2_shift
			if self.nr_bins % 2 == 0:							# CHECK THIS: WORKS FOR 5 AND 8 BINS
				C2s[i,:] = np.roll(C2[i,:], - shift)	
			else:
				C2s[i,:] = np.roll(C2[i,:], - shift - 1)			

		return C2, C2s, W 	

	def maxTrial(self, conditions, condition, pos_bins):
		'''
		minTrial calculates the maximum number of trials that can be used in the block assignment of the forward model such that the number of trials used 
		in the IEM is equated across conditions

		Arguments
		- - - - - 
		conditions (list): condition names
		condition (array): array with integers corresponding to the different conditions 
		pos_bins (array): array with position bins

		Returns
		- - - -
		nr_per_bin(int): maximum number of trials 
		'''

		bin_count = np.zeros((len(conditions),self.nr_bins))
		for cond in range(len(conditions)):
			temp_bin = pos_bins[condition == cond]
			for b in range(self.nr_bins):
				bin_count[cond,b] = sum(temp_bin == b) 

		min_count = np.min(bin_count)
		nr_per_bin = int(np.floor(min_count/self.nr_blocks))						# max number of trials per bin

		return nr_per_bin

	def assignBlocks(self, pos_bin, nr_per_bin):
		'''
		assignBlocks creates a random block assignment (train and test blocks)

		Arguments
		- - - - - 
		pos_bin (array): array with position bins
		nr_per_bin(int): maximum number of trials to be used per location bin

		Returns
		- - - -
		blocks(array): randomly assigned block indices
		'''
		
		nr_trials = len(pos_bin)

		#preallocate arrays
		blocks = np.empty(nr_trials) * np.nan
		shuf_blocks = np.empty(nr_trials) * np.nan

		idx_shuf = np.random.permutation(nr_trials) 			# create shuffle index
		shuf_bin = pos_bin[idx_shuf]							# shuffle trial order

		# take the 1st nr_per_bin x nr_blocks trials for each position bin
		for b in range(self.nr_bins):
			idx = np.where(shuf_bin == b)[0] 					# get index of trials belonging to the current bin
			idx = idx[:nr_per_bin * self.nr_blocks] 			# drop excess trials
			x = np.tile(np.arange(self.nr_blocks),(nr_per_bin,1))
			shuf_blocks.flat[idx] = x 							# assign randomly order trial to blocks

		# unshuffle block assignment and save to CTF
		blocks[idx_shuf] = shuf_blocks	
		trials_per_block = sum(blocks == 0)

		return blocks, trials_per_block	

	def spatialCTFCrossTrain(self, subject_id, interval, filt_art, train = 'memory', test = 'search', freqs = dict(alpha = [8,12]), downsample = 1):
		'''
		Calculates spatial CTFs across subjects and conditions using the filter-Hilbert method. This function trains and tests on independent datasets

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		interval (list): time interval in ms used for ctf calculation
		filt_art (int | list): Time in ms added to start and end interval to correct for filter artifacts. To add differential timing to start and end
		specify a list of values.  
		freqs (dict) | Default is the alpha band: Frequency band used for bandpass filter. If key set to all, analyses will be performed in increments 
		from 1Hz
		downsample (int): factor to downsample the bandpass filtered data

		Returns
		- - - -
		self(object): dict of CTF data
		'''	


		self.ctf_name = freqs.keys()[0]
		self.subject_id = subject_id
		conditions = ['single_task','dual_task']

		CTF_cross = {}

		# Extra parameters for inverted encoding model
		stime = 1000.0/self.sfreq						
		time = np.arange(interval[0],interval[1],stime)		

		if 'all' in freqs.keys():
			freqs = np.vstack([[i, i + 4] for i in range(freqs['all'][0],freqs['all'][1] + 1,2)])
		else:	
			bands = freqs.keys()								
			freqs = np.vstack([[freqs[band][0],freqs[band][1]] for band in bands])
		nr_freqs = freqs.shape[0]

		# save standard parameters in info dict
 		#self.info = dict(stime = stime, time = time, downsample = downsample, freqs = freqs, nr_freqs = nr_freqs, conditions = conditions)
		self.info = dict(stime = stime, time = time, downsample = downsample, freqs = freqs, nr_freqs = nr_freqs)

		# get behavioral data (location bin and condition)
		#train_idx = 1 # 0 is single task, 2 is dual task
		#test_idx = 1

		
		
		if self.decoding == 'target_loc':
			cnds = [['DvTr_0'],['DvTr_3']]
		elif self.decoding == 'dist_loc':
			cnds = [['DrTv_0'],['DrTv_3']]

		slopes = {}
		for cnd in cnds:
			if '3' in cnd[0]:
				train_pos_bins, train_cnds, train_eeg, EEG = self.readData(subject_id, ['DvTv_3'])
			elif '0' in cnd[0]:
				train_pos_bins, train_cnds, train_eeg, EEG = self.readData(subject_id, ['DvTv_0'])	
			test_pos_bins, test_cnds, test_eeg, EEG = self.readData(subject_id, ['DvTr_3'])

			#train_pos_bins, condition = self.behaviorCTF(subject_id, 64, conditions, 'memory_orient', 'block_type')
			#train_pos_bins = train_pos_bins/45
			#train_pos_bins[train_pos_bins < 0] = 7
			#test_pos_bins, condition = self.behaviorCTF(subject_id, 64, conditions, 'location_bin_search', 'block_type')
			
			#get EEG data
			#file = self.FolderTrackerCTF('data', filename = str(subject_id) + '_EEG.mat')
			#EEG = h5py.File(file) 

			#idx_channel, nr_selected, nr_total, all_channels = self.selectChannelId(subject_id)
			
			#eegs = EEG['erp']['trial']['data'][:,idx_channel,:]						# read in eeg data (drop scalp elecrodes)
			#eegs = np.swapaxes(eegs,0,2)											# eeg needs to be a (trials x electrodes x timepoints) array

			nr_electrodes = train_eeg.shape[1]

			#idx_art = np.squeeze(EEG['erp']['arf']['artifactIndCleaned']) 			# grab artifact rejection index
			
			if isinstance(filt_art,int):
				time_filt = np.arange(interval[0] - filt_art,interval[1] + filt_art,stime)
			elif isinstance(filt_art,list):
				time_filt = np.arange(interval[0] - filt_art[0],interval[1] + filt_art[1],stime)
			else:
				raise ValueError('time_filt needs to be a list or an int')	

			tois = np.logical_and(time_filt >= interval[0], time_filt <= interval[1]) 	# index time points for analysis
			nr_times = len(tois)
			nr_samps = tois.sum()
			nr_total = 64

			# Remove rejected trials for eeg and behavior
			#eegs = eegs[idx_art == 0,:,:] 									
			#train_pos_bins = train_pos_bins[idx_art == 0]
			#test_pos_bins = test_pos_bins[idx_art == 0]
			#condition = condition[idx_art == 0]

			# save total nr trials to CTF dict (update info dict)
			if downsample > 1:
				nr_samps = np.arange(0,tois.sum(),downsample).size
			self.info.update({'times':time_filt[tois],'tois':tois, 'nr_samps': nr_samps, 'nr_times': nr_times, 'sel_electr': nr_electrodes,'all_electr': nr_total,'nr_trials': len(train_pos_bins)})

			# Determine the number of trials that can be used for each position bin, matched across conditions		
			nr_per_bin_train = []
			#temp_bin = train_pos_bins[train_cnds == train_idx]
			for b in range(self.nr_bins):
				nr_per_bin_train.append(sum(train_pos_bins == b))

			nr_per_bin_train = np.min(nr_per_bin_train)

			nr_per_bin_test = []
			#temp_bin = test_pos_bins[condition == test_idx]
			for b in range(self.nr_bins):
				nr_per_bin_test.append(sum(test_pos_bins == b))

			nr_per_bin_test = np.min(nr_per_bin_test)

			#train_eeg = eegs[condition == train_idx,:,:nr_times]
			#test_eeg = eegs[condition == test_idx,:,:nr_times]
			#train_pos_bins = train_pos_bins[condition == train_idx]
			#test_pos_bins = test_pos_bins[condition == test_idx]

			# adjust nr_of train and test bins
			for i in range(self.nr_bins):
				if sum(train_pos_bins == i) > nr_per_bin_train:
					idx_rem = random.sample(np.where(train_pos_bins == i)[0], sum(train_pos_bins == i) - nr_per_bin_train)
					train_pos_bins = np.delete(train_pos_bins, idx_rem)
					train_eeg = np.delete(train_eeg, idx_rem, axis = 0)
				if sum(test_pos_bins == i) > nr_per_bin_test:
					idx_rem = random.sample(np.where(test_pos_bins == i)[0], sum(test_pos_bins == i) - nr_per_bin_test)
					test_pos_bins = np.delete(test_pos_bins, idx_rem)
					test_eeg = np.delete(test_eeg, idx_rem, axis = 0)	
		
			# save extra variables for permuteSpatialCTF
			CTF_cross['perm'] = dict(train_pos_bins = train_pos_bins, test_pos_bins = test_pos_bins, train_eeg = train_eeg, test_eeg = test_eeg, nr_per_bin_train = nr_per_bin_train, nr_per_bin_test = nr_per_bin_test)

			# Preallocate arrays
			tf_evoked = np.empty((nr_freqs,nr_samps, nr_samps, self.nr_chans)) * np.nan; tf_total = np.copy(tf_evoked)
			C2_evoked = np.empty((nr_freqs,nr_samps, nr_samps, self.nr_bins, self.nr_chans)) * np.nan; C2_total = np.copy(C2_evoked)
			W_evoked = np.empty((nr_freqs,nr_samps, nr_samps, self.nr_chans, nr_electrodes)) * np.nan; W_total = np.copy(W_evoked)

			# Time-Frequency Analysis
			filt_data_evoked_train, filt_data_total_train = self.powerAnalysis(train_eeg, tois, self.sfreq, [freqs[0][0],freqs[0][1]], downsample)
			filt_data_evoked_test, filt_data_total_test = self.powerAnalysis(test_eeg, tois, self.sfreq, [freqs[0][0],freqs[0][1]], downsample)
			
			train_data_evoked = np.empty((self.nr_bins, nr_electrodes, nr_samps)) * np.nan 		# averaged evoked data
			train_data_total = np.copy(train_data_evoked) 										# averaged total data
			test_data_evoked = np.empty((self.nr_bins, nr_electrodes, nr_samps)) * np.nan 		# averaged evoked data
			test_data_total = np.copy(test_data_evoked) 										# averaged total data
			c = np.empty((self.nr_bins,self.nr_chans)) * np.nan 								# predicted channel responses for averaged data

			for i in range(self.nr_bins):
				idx_train = train_pos_bins == i		
				idx_test = test_pos_bins == i
				train_data_evoked[i,:,:] = abs(np.mean(filt_data_evoked_train[idx_train,:,:],axis = 0))**2		# evoked power is averaged over trials before calculating power
				test_data_evoked[i,:,:] = abs(np.mean(filt_data_evoked_test[idx_test,:,:],axis = 0))**2			# evoked power is averaged over trials before calculating power
				train_data_total[i,:,:] = np.mean(filt_data_total_train[idx_train,:,:],axis = 0)					# total power is calculated before average over trials takes place (see frequency loop)
				test_data_total[i,:,:] = np.mean(filt_data_total_test[idx_test,:,:],axis = 0)					# total power is calculated before average over trials takes place (see frequency loop)
				c[i,:] = self.basisset[i,:]

			if self.nr_chans % 2 == 0:
				steps = self.nr_chans / 2 + 1 
			else:
				steps = int(np.ceil(self.nr_chans / 2.0))

			slope_matrix_evoked = np.empty((nr_samps, nr_samps)) * np.nan	
			slope_matrix_total = np.copy(slope_matrix_evoked)

			# Loop over all samples 
			freq = 0
			for smpl_train in range(nr_samps):
				for smpl_test in range(nr_samps):

					sample_evoked_train = train_data_evoked[:,:,smpl_train]
					sample_total_train = train_data_total[:,:,smpl_train]

					sample_evoked_test = test_data_evoked[:,:,smpl_test]
					sample_total_test = test_data_total[:,:,smpl_test]

					###### ANALYSIS ON EVOKED POWER ######
					C2, C2s, We = self.forwardModelCross(sample_evoked_train, sample_evoked_test, c, np.arange(self.nr_bins))
					C2_evoked[freq,smpl_train,smpl_test,:,:] = C2 					# save the unshifted channel response
					tf_evoked[freq,smpl_train,smpl_test,:] = np.mean(C2s,0)			# save average of shifted channel response
					W_evoked[freq, smpl_train,smpl_test,:,:] = We 					# save estimated weights per channel and electrode
					#C2s = np.mean(C2s,0)

					#if self.nr_chans % 2 == 0:
					#	d = np.array([C2s[0]] + [np.mean((C2s[i],C2s[-i])) for i in range(1,steps)])
					#else:
					#	d = np.array([np.mean((C2s[i],C2s[i+(-1-2*i)])) for i in range(steps)])
					#slope_matrix_evoked[smpl_train,smpl_test] = np.polyfit(range(1, len(d) + 1), d, 1)[0]

					###### ANALYSIS ON TOTAL POWER ######
					
					C2, C2s, Wt = self.forwardModelCross(sample_total_train, sample_total_test, c, np.arange(self.nr_bins))
					C2_total[freq,smpl_train,smpl_test,:,:] = C2 					# save the unshifted channel response
					tf_total[freq,smpl_train,smpl_test,:] = np.mean(C2s,0)			# save average of shifted channel response
					W_total[freq, smpl_train,smpl_test,:,:] = We 	
					#C2s = np.mean(C2s,0)
					#if self.nr_chans % 2 == 0:
					#	d = np.array([C2s[0]] + [np.mean((C2s[i],C2s[-i])) for i in range(1,steps)])
					#else:
					#	d = np.array([np.mean((C2s[i],C2s[i+(-1-2*i)])) for i in range(steps)])
					#slope_matrix_total[smpl_train,smpl_test] = np.polyfit(range(1, len(d) + 1), d, 1)[0]

			CTF_cross['C2'] = {}
			CTF_cross['C2'].update({'evoked': C2_evoked,'total': C2_total})
			CTF_cross['ctf'] = {}
			CTF_cross['ctf'] = {'evoked': tf_evoked,'total': tf_total}	

			data = tf_total
			cross_slopes = np.zeros((nr_samps, nr_samps))
			for smpl_train in range(nr_samps):
				cross_slopes[smpl_train,:] = self.calculateSlopes(data[:,smpl_train,:,:], 1, nr_samps)

			slopes.update({cnd[0]:{}})
			slopes[cnd[0]]['cross'] = cross_slopes	
		
		# save data
		#with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_ctf_{}.pickle'.format(str(subject_id),'cross')),'wb') as handle:
		#	print('saving CTF dict')
		#	#self.CTF = CTF
		#	pickle.dump(CTF_cross, handle)

		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_info.pickle'.format('cross')),'wb') as handle:
			print('saving info dict')
			pickle.dump(self.info, handle)	

		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_slopes-sub_{}.pickle'.format(str(subject_id),'cross')),'wb') as handle:
			print('saving slopes dict')
			#self.CTF = CTF
			pickle.dump(slopes, handle)

		#np.savetxt(self.FolderTrackerCTF('ctf', [self.channel_folder, 'cross_corr'], filename = 'slope_matrix_evoked_{}.csv'.format(subject_id)),slope_matrix_evoked, delimiter=',')
		#np.savetxt(self.FolderTrackerCTF('ctf', [self.channel_folder, 'cross_corr'], filename = 'slope_matrix_total_{}.csv'.format(subject_id)),slope_matrix_total, delimiter=',')		
		# save data

	def permuteSpatialCTFCrossTrain(self, subject_id, nr_perms = 1000):
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
		
		info = self.readCTF(subject_id, 'alpha', info = True)
		CTF = self.readCTF([subject_id], 'alpha', 'ctf')[0]
	
		CTFp_cross = {}	
		
		#get EEG data and position pin info
		eeg_train = CTF['perm']['train_eeg']
		eeg_test = CTF['perm']['test_eeg']
		train_pos_bin = CTF['perm']['train_pos_bins']
		test_pos_bin = CTF['perm']['test_pos_bins']
		nr_electrodes = eeg_train.shape[1]

		# Preallocate arrays
		#nr_trials_block = CTF[curr_cond]['nr_trials_block']			# SHOULD BE SAME PER CONDITION SO CAN BE CHANGED
		tf_evoked = np.empty((info['nr_freqs'], nr_perms,info['nr_samps'], info['nr_samps'],self.nr_chans)) * np.nan; tf_total = np.copy(tf_evoked)
		idx_perm = np.empty((info['nr_freqs'],nr_perms, train_pos_bin.size))			

		# loop through each frequency
		for freq in range(info['nr_freqs']):
			print('Frequency {} out of {}'.format(str(freq + 1), str(info['nr_freqs'])))
			
			# Time-Frequency Analysis
			filt_data_evoked_train, filt_data_total_train = self.powerAnalysis(eeg_train, info['tois'], self.sfreq, [info['freqs'][freq][0],info['freqs'][freq][1]], info['downsample'])
			filt_data_evoked_test, filt_data_total_test = self.powerAnalysis(eeg_test, info['tois'], self.sfreq, [info['freqs'][freq][0],info['freqs'][freq][1]], info['downsample'])

			# loop through permutations
			for p in range(nr_perms):
				if p % 10 == 0:
					print(p)

				# Permute trial assignment (only in training blocks)
				perm_train_pos_bin = np.empty((train_pos_bin.size)) * np.nan
				idx_p = np.random.permutation(train_pos_bin.size)		# create a permutation index
				perm_train_pos_bin[idx_p] = train_pos_bin 
				idx_perm[freq, p,:] = idx_p						# store the permutation NOT YET SAVED TO DICTIONARY!!!!!

				# average data for each position bin across blocks
				train_data_evoked = np.empty((self.nr_bins, nr_electrodes, info['nr_samps'])) * np.nan 		# averaged evoked data
				train_data_total = np.copy(train_data_evoked) 										# averaged total data
				test_data_evoked = np.empty((self.nr_bins, nr_electrodes, info['nr_samps'])) * np.nan 		# averaged evoked data
				test_data_total = np.copy(test_data_evoked) 										# averaged total data
				c = np.empty((self.nr_bins,self.nr_chans)) * np.nan 								# predicted channel responses for averaged data

				for i in range(self.nr_bins):
					idx_train = perm_train_pos_bin == i		
					idx_test = test_pos_bin == i
					train_data_evoked[i,:,:] = abs(np.mean(filt_data_evoked_train[idx_train,:,:],axis = 0))**2		# evoked power is averaged over trials before calculating power
					test_data_evoked[i,:,:] = abs(np.mean(filt_data_evoked_test[idx_test,:,:],axis = 0))**2			# evoked power is averaged over trials before calculating power
					train_data_total[i,:,:] = np.mean(filt_data_total_train[idx_train,:,:],axis = 0)					# total power is calculated before average over trials takes place (see frequency loop)
					test_data_total[i,:,:] = np.mean(filt_data_total_test[idx_test,:,:],axis = 0)					# total power is calculated before average over trials takes place (see frequency loop)
					c[i,:] = self.basisset[i,:]

					
				# Loop over all samples CHECK WHETHER THIS LOOP CAN BE PARALLIZED 
				for smpl_train in range(info['nr_samps']):
					for smpl_test in range(info['nr_samps']):

						sample_evoked_train = train_data_evoked[:,:,smpl_train]
						sample_total_train = train_data_total[:,:,smpl_train]

						sample_evoked_test = test_data_evoked[:,:,smpl_test]
						sample_total_test = test_data_total[:,:,smpl_test]


						###### ANALYSIS ON EVOKED POWER ######
						C2, C2s, We = self.forwardModelCross(sample_evoked_train, sample_evoked_test, c, np.arange(self.nr_bins))
						tf_evoked[freq,p,smpl_train,smpl_test,:] = np.mean(C2s,0)			# save average of shifted channel response

						###### ANALYSIS ON TOTAL POWER ######
						C2, C2s, We = self.forwardModelCross(sample_total_train, sample_total_test, c, np.arange(self.nr_bins))
						tf_total[freq,p,smpl_train,smpl_test,:] = np.mean(C2s,0)	

		# save relevant data to CTF dictionary
		CTFp_cross['ctf'] = {}
		CTFp_cross['ctf'] = {'evoked': tf_evoked,'total': tf_total}

		# save data
		with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_ctfperm_all.pickle'.format(str(subject_id))),'wb') as handle:
			print('saving CTF perm dict')
			pickle.dump(CTFp_cross, handle)	

	def spatialCTF(self, subject_id, interval, filt_art, conditions = ['single'], freqs = dict(alpha = [8,12]), downsample = 1):
		'''
		Calculates spatial CTFs across subjects and conditions using the filter-Hilbert method. 

		Arguments
		- - - - - 
		subject_id (list): list of subject id's 
		interval (list): time interval in ms used for ctf calculation
		filt_art (int | list): Time in ms added to start and end interval to correct for filter artifacts. To add differential timing to start and end
		specify a list of values.  
		condition (list)| Default ('single'): List of conditions. Defaults to decoding during simple memory task with no extra instructions,
		but can be extended to different locations
		freqs (dict) | Default is the alpha band: Frequency band used for bandpass filter. If key set to all, analyses will be performed in increments 
		from 1Hz
		downsample (int): factor to downsample the bandpass filtered data

		Returns
		- - - -
		self(object): dict of CTF data
		'''	

		self.ctf_name = list(freqs.keys())[0]
		self.subject_id = subject_id
		CTF = {} 							# initialize dict to save CTFs
		for cond in conditions:													
			CTF.update({cond:{}})			# update CTF dict such that condition data can be saved later
			
		# Extra parameters for inverted encoding model
		stime = 1000.0/self.sfreq						
		time = np.arange(interval[0],interval[1],stime)		

		if 'all' in freqs.keys():
			freqs = np.vstack([[i, i + 4] for i in range(freqs['all'][0],freqs['all'][1] + 1,2)])
		else:	
			bands = freqs.keys()								
			freqs = np.vstack([[freqs[band][0],freqs[band][1]] for band in bands])
		nr_freqs = freqs.shape[0]

		# save standard parameters in info dict
		self.info = dict(stime = stime, time = time, downsample = downsample, freqs = freqs, nr_freqs = nr_freqs, conditions = conditions)
				
		# get behavioral data (location bin and condition)
		if self.decoding == 'memory':
			bin_header = 'location_bin_mem'
		elif self.decoding == 'search':
			bin_header = 'location_bin_search'
		pos_bins, condition = self.behaviorCTF(subject_id, 64, conditions, bin_header, 'block_type')
		
		#get EEG data
		file = self.FolderTrackerCTF('data', filename = str(subject_id) + '_EEG.mat')
		EEG = h5py.File(file) 

		idx_channel, nr_selected, nr_total, all_channels = self.selectChannelId(subject_id, channels = 'all')
		
		eegs = EEG['erp']['trial']['data'][:,idx_channel,:]						# read in eeg data (drop scalp elecrodes)
		eegs = np.swapaxes(eegs,0,2)											# eeg needs to be a (trials x electrodes x timepoints) array

		nr_electrodes = eegs.shape[1]

		idx_art = np.squeeze(EEG['erp']['arf']['artifactIndCleaned']) 			# grab artifact rejection index
		
		if isinstance(filt_art,int):
			time_filt = np.arange(interval[0] - filt_art,interval[1] + filt_art,stime)
		elif isinstance(filt_art,list):
			time_filt = np.arange(interval[0] - filt_art[0],interval[1] + filt_art[1],stime)
		else:
			raise ValueError('time_filt needs to be a list or an int')	

		tois = np.logical_and(time_filt >= interval[0], time_filt <= interval[1]) 	# index time points for analysis
		nr_times = len(tois)
		nr_samps = tois.sum()

		# Remove rejected trials for eeg and behavior
		eegs = eegs[idx_art == 0,:,:] 									
		pos_bins = pos_bins[idx_art == 0]
		condition = condition[idx_art == 0]

		# save total nr trials to CTF dict (update info dict)
		if downsample > 1:
			nr_samps = np.arange(0,tois.sum(),downsample).size
		self.info.update({'times':time_filt[tois],'tois':tois, 'nr_samps': nr_samps, 'nr_times': nr_times, 'sel_electr': nr_selected,'all_electr': nr_total,'nr_trials': len(pos_bins)})

		# Determine the number of trials that can be used for each position bin, matched across conditions
		nr_per_bin = self.maxTrial(conditions, condition, pos_bins)
		
		# save extra variables for permuteSpatialCTF
		CTF['perm'] = dict(pos_bins = pos_bins, condition = condition, idx_art = idx_art, max_trial_per_bin = nr_per_bin)

		# Loop over conditions
		for cond, curr_cond in enumerate(conditions):
			eeg = eegs[condition == cond,:,:nr_times] 						# selected condition eeg data equated for nr of time points 
			pos_bin = pos_bins[condition == cond]
			nr_trials = len(pos_bin)

			# Preallocate arrays
			tf_evoked = np.empty((nr_freqs,self.nr_iter,nr_samps,self.nr_blocks, self.nr_chans)) * np.nan; tf_total = np.copy(tf_evoked)
			C2_evoked = np.empty((nr_freqs,self.nr_iter,nr_samps,self.nr_blocks, self.nr_bins, self.nr_chans)) * np.nan; C2_total = np.copy(C2_evoked)
			W_evoked = np.empty((nr_freqs,self.nr_iter,nr_samps,self.nr_blocks, self.nr_chans, nr_electrodes)) * np.nan; W_total = np.copy(W_evoked)
			blocks = np.zeros((nr_trials, self.nr_iter))

			# Create block assignment for each iteration (trials are assigned to blocks so that nr of trials per bin are equated within blocks)
			# This is done before the frequency loop so that the same block assignments are used for all freqs
			for itr in range(self.nr_iter):

				CTF[curr_cond]['blocks_' + str(itr)], CTF[curr_cond]['nr_trials_block'] = self.assignBlocks(pos_bin, nr_per_bin)
					
			# Loop over frequency bands
			for freq in range(nr_freqs):
				print('Frequency {} out of {}'.format(str(freq + 1), str(nr_freqs)))
				
				# Time-Frequency Analysis
				filt_data_evoked, filt_data_total = self.powerAnalysis(eeg, tois, self.sfreq, [freqs[freq][0],freqs[freq][1]], downsample)

				# Loop through each iteration
				for itr in range(self.nr_iter):

					# grab block assignment for current iteration
					blocks = CTF[curr_cond]['blocks_' + str(itr)]

					# average data for each position bin across blocks
					all_bins = np.arange(self.nr_bins)
					block_data_evoked = np.empty((self.nr_bins*self.nr_blocks, nr_electrodes, nr_samps)) * np.nan 	# averaged evoked data
					block_data_total = np.copy(block_data_evoked) 								  					# averaged total data
					labels = np.empty((self.nr_bins*self.nr_blocks,1)) * np.nan 									# bin labels for averaged data
					block_nr = np.copy(labels)																		# block numbers for averaged data
					c = np.empty((self.nr_bins*self.nr_blocks,self.nr_chans)) * np.nan 								# predicted channel responses for averaged data

					bin_cntr = 0
					for i in range(self.nr_bins):
						for j in range(self.nr_blocks):
							idx = np.logical_and(pos_bin == all_bins[i],blocks == j)
							evoked_2_mean = filt_data_evoked[idx,:,:][:,:,:] 										
							block_data_evoked[bin_cntr,:,:] = abs(np.mean(evoked_2_mean,axis = 0))**2	# evoked power is averaged over trials before calculating power
							total_2_mean = filt_data_total[idx,:,:][:,:,:]								# CHECK WARNING: NUMBERS SHOULD NO LONGER BE COMPLEX
							block_data_total[bin_cntr,:,:] = np.mean(total_2_mean,axis = 0)				# total power is calculated before average over trials takes place (see frequency loop)
							labels[bin_cntr] = i
							block_nr[bin_cntr] = j
							c[bin_cntr,:] = self.basisset[i,:]
							bin_cntr += 1

					# Loop over all samples CHECK WHETHER THIS LOOP CAN BE PARALLIZED 
					for smpl in range(nr_samps):

						data_evoked = block_data_evoked[:,:,smpl] # CHECK WHY I DO NOT AVERAGE HERE!!!!!!
						data_total = block_data_total[:,:,smpl]

						# FORWARD MODEL
						for blck in range(self.nr_blocks):

							###### ANALYSIS ON EVOKED POWER ######
							C2, C2s, We = self.forwardModel(data_evoked, labels, c, block_nr, blck, all_bins)
							C2_evoked[freq,itr,smpl,blck,:,:] = C2 					# save the unshifted channel response
							tf_evoked[freq,itr,smpl,blck,:] = np.mean(C2s,0)		# save average of shifted channel response
							W_evoked[freq, itr, smpl,blck,:,:] = We 				# save estimated weights per channel and electrode

							###### ANALYSIS ON TOTAL POWER ######
							
							C2, C2s, Wt = self.forwardModel(data_total, labels, c, block_nr, blck, all_bins)
							C2_total[freq,itr,smpl,blck,:,:] = C2 					# save the unshifted channel response
							tf_total[freq,itr,smpl,blck,:] = np.mean(C2s,0)			# save average of shifted channel response
							W_total[freq, itr, smpl,blck,:,:] = Wt 					# save estimated weights per channel and electrode
			
			# save relevant data to CTF dictionary
			CTF[curr_cond]['C2'] = {}
			CTF[curr_cond]['C2'].update({'evoked': C2_evoked,'total': C2_total})
			CTF[curr_cond]['ctf'] = {}
			CTF[curr_cond]['ctf'] = {'evoked': tf_evoked,'total': tf_total}
			#CTF[curr_cond]['W'] = {}
			#CTF[curr_cond]['W'] = {'evoked': W_evoked,'total': W_total}

		# save data
		#with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_ctf_{}.pickle'.format(str(subject_id),self.ctf_name)),'wb') as handle:
		with open('/home/dvmoors1/ctf_output/{}_ctf_{}.pickle'.format(str(subject_id),self.ctf_name),'wb') as handle:
			print('saving CTF dict')
			self.CTF = CTF
			pickle.dump(CTF, handle)

		#with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_info.pickle'.format(self.ctf_name)),'wb') as handle:
		with open('/home/dvmoors1/ctf_output/{}_info.pickle'.format(self.ctf_name),'wb') as handle:
			print('saving info dict')
			pickle.dump(self.info, handle)

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
		
		# load CTF parameters
		#info = self.readCTF(subject_id, 'all', info = True)
		#CTF = self.readCTF([subject_id], 'all', 'ctf')[0]

		with open('/home/dvmoors1/ctf_output/{}_ctf_all.pickle'.format(subject_id),'rb') as handle:
			CTF = pickle.load(handle)

		with open('/home/dvmoors1/ctf_output/all_info.pickle'.format(subject_id),'rb') as handle:
			info = pickle.load(handle)
	
		# initialize dict to save CTFs
		CTFp = {}
		for cond in info['conditions']:													
			CTFp.update({cond:{}})			# update CTF dict such that condition data can be saved later			

		#get EEG data
		file = self.FolderTrackerCTF('data', filename = str(subject_id) + '_EEG.mat')
		EEG = h5py.File(file)

		idx_channel, nr_selected, nr_total, all_channels = self.selectChannelId(subject_id, channels = 'all')
		eegs = EEG['erp']['trial']['data'][:,idx_channel,:]						# read in eeg data (drop scalp elecrodes)
		eegs = np.swapaxes(eegs,0,2)											# eeg needs to be a (trials x electrodes x timepoints) array
		eegs = eegs[CTF['perm']['idx_art'] == 0,:,:]
		nr_electrodes = eegs.shape[1]

		# loop through conditions
		for cond, curr_cond in enumerate(info['conditions']):
			eeg = eegs[CTF['perm']['condition'] == cond,:,:len(info['tois'])] # selected condition eeg data equated for nr of time points 
			pos_bin = CTF['perm']['pos_bins'][CTF['perm']['condition'] == cond]
			nr_trials = len(pos_bin)	

			# Preallocate arrays
			nr_trials_block = CTF[curr_cond]['nr_trials_block']			# SHOULD BE SAME PER CONDITION SO CAN BE CHANGED
			tf_evoked = np.empty((info['nr_freqs'],self.nr_iter, nr_perms,info['nr_samps'], self.nr_chans)) * np.nan; tf_total = np.copy(tf_evoked)
			idx_perm = np.empty((info['nr_freqs'],self.nr_iter,nr_perms, self.nr_blocks,nr_trials_block))			

			# loop through each frequency
			for freq in range(info['nr_freqs']):
				print('Frequency {} out of {}'.format(str(freq + 1), str(info['nr_freqs'])))
				
				# Time-Frequency Analysis
				filt_data_evoked, filt_data_total = self.powerAnalysis(eeg, info['tois'], self.sfreq, [info['freqs'][freq][0],info['freqs'][freq][1]], info['downsample'])

				# Loop through each 
				for itr in range(self.nr_iter):

					# grab block assignment for current iteration
					blocks = CTF[curr_cond]['blocks_' + str(itr)]
						
					# loop through permutations
					for p in range(nr_perms):

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
		with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_ctfperm_all.pickle'.format(str(subject_id))),'wb') as handle:
			print('saving CTF perm dict')
			pickle.dump(CTFp, handle)	
					
	def calculateSlopes(self, data, nr_freqs, nr_samps):
		'''
		calculateSlopes calculates slopes of CTF data by collapsing across symmetric data points in the tuning curve. The collapsed data create a linear 
		increasing vector array from the tails to the peak of the tuning curve. A first order polynomial is then fitted to this array to estimate the slope
		of the tuning curve. 

		Arguments
		- - - - - 
		data (array): CTF data across frequencies and sample points (nr freqs, nr samps, nr chans) 
		nr_freqs (int): number of frequencies in CTF data
		nr_samps (int): number of sample points in CTF data

		Returns
		- - - -
		slopes(array): slope values across frequencies and sample points
		'''

		if self.nr_chans % 2 == 0:
			steps = self.nr_chans / 2 + 1 
		else:
			steps = int(np.ceil(self.nr_chans / 2.0))	

		slopes = np.empty((nr_freqs, nr_samps)) * np.nan

		for f in range(nr_freqs):
			for s in range(nr_samps):
				d = data[f,s,:] 
				# collapse symmetric slope positions, eg (0, (45,315), (90,270), (135,225), 180)
				if self.nr_chans % 2 == 0:
					d = np.array([d[0]] + [np.mean((d[i],d[-i])) for i in range(1,steps)])
				else:
					d = np.array([np.mean((d[i],d[i+(-1-2*i)])) for i in range(steps)])
				slopes[f,s] = np.polyfit(range(1, len(d) + 1), d, 1)[0]

		return slopes	

	def CTFSlopesCrossTrain(self, subject_id, band = 'alpha', perm = False):
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
			ctfs = self.readCTF(subject_id, band, 'ctfperm')
			sname = 'slopes_perm'
		else:	
			sname = 'slopes'
			ctfs = self.readCTF(subject_id, band)
	
		info = self.readCTF(subject_id, band, dicts = 'ctf', info = True)

		for i, sbjct in enumerate(subject_id):
			slopes = {}

			for power in ['total']:
				if perm: 
					data = ctfs[i]['ctf'][power]											# average across iterations
					nr_perms = data.shape[1]
					p_slopes = np.empty((data.shape[2],data.shape[3],nr_perms)) * np.nan
					for p in range(nr_perms):											
						for smpl_train in range(info['nr_samps']):												
							p_slopes[smpl_train,:,p] = self.calculateSlopes(data[:,p,smpl_train,:,:], info['nr_freqs'], info['nr_samps'])	
					slopes.update({power: p_slopes})
				else:	
					data = ctfs[i]['ctf'][power] 
					cross_slopes = np.zeros((info['nr_samps'], info['nr_samps']))
					for smpl_train in range(info['nr_samps']):
						cross_slopes[smpl_train,:] = self.calculateSlopes(data[:,smpl_train,:,:], info['nr_freqs'], info['nr_samps'])

					slopes.update({power: cross_slopes})

			with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_{}_{}.pickle'.format(str(sbjct),sname,band)),'wb') as handle:
				print('saving slopes dict')
				pickle.dump(slopes, handle)		

	def CTFSlopes(self, subject_id, band = 'alpha', perm = False):
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
			ctfs = self.readCTF(subject_id, band, 'ctfperm')
			sname = 'slopes_perm'
		else:	
			sname = 'slopes'
			ctfs = self.readCTF(subject_id, band)
	
		info = self.readCTF(subject_id, band, dicts = 'ctf', info = True)

		for i, sbjct in enumerate(subject_id):
			slopes = {}

			for cond in info['conditions']:
				slopes.update({cond:{}})
				for power in ['evoked','total']:
					if perm: 
						data = np.mean(ctfs[i][cond]['ctf'][power],axis = 1)					# average across iterations
						nr_perms = data.shape[1]
						p_slopes = np.empty((data.shape[0],data.shape[2],nr_perms)) * np.nan
						for p in range(nr_perms):												# loop over permutations
							p_slopes[:,:,p] = self.calculateSlopes(data[:,p,:,:], info['nr_freqs'], info['nr_samps'])
						slopes[cond].update({power: p_slopes})
					else:	
						data = np.mean(np.mean(ctfs[i][cond]['ctf'][power],axis = 1),axis = 2) 	# average across iteration and cross-validation blocks
						sl = self.calculateSlopes(data, info['nr_freqs'], info['nr_samps'])
						slopes[cond].update({power: sl})

			with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_{}_{}.pickle'.format(str(sbjct),sname,band)),'wb') as handle:
				print 'saving slopes dict'
				pickle.dump(slopes, handle)	

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

			#ctf = np.mean(t_ctf[:,idx_win[0]:idx_win[1],:], axis = (0,1))	
			#plt.plot(ctf - ctf.min(), label = cond)	

		#plt.legend(loc = 'best')
	

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
					
					levels = np.linspace(0,0.75,1000)
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
					plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = '{}_ctfs_ind_{}_{}.pdf'.format(band,cond,i)))
					plt.close()

			elif plot == 'group':
				ectf = np.nanmean(e_ctf,0) 
				tctf = np.nanmean(t_ctf,0) 
				if self.nr_chans % 2 == 0:
					ectf = np.hstack((ectf,ectf[:,0].reshape(-1,1)))
					tctf = np.hstack((tctf,tctf[:,0].reshape(-1,1)))
				
				levels = np.linspace(0,0.6,1000)
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
				plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = '{}_ctfs_group_{}.pdf'.format(band,cond)))
				plt.close()	

	def corrCTFSlopes(self, subject_id):
		'''

		'''

		from scipy.stats.stats import pearsonr

		info = self.readCTF(subject_id, 'alpha', info = True)
		ctf_mem, ctf_search = [],[]
		for sbjct in subject_id:
			with open(self.FolderTrackerCTF('ctf', [self.channel_folder,'memory'], filename = '{}_{}_{}.pickle'.format(str(sbjct),'slopes','alpha')),'rb') as handle:
				ctf_mem.append(pickle.load(handle))		
			with open(self.FolderTrackerCTF('ctf', [self.channel_folder,'search'], filename = '{}_{}_{}.pickle'.format(str(sbjct),'slopes','alpha')),'rb') as handle:
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

				plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'correction_ind.pdf'))
				plt.close()		

			if plot == 'group':
				plt.figure(figsize= (20,10))

				plt.plot(info['times'],np.mean(slopes_mem_cor, axis = 0), color = 'red', label = 'memory')
				plt.plot(info['times'],np.mean(slopes_search_cor, axis = 0), color = 'green', label = 'search')
				plt.xlim(-300,2200)
				plt.legend(loc = 'best')

				plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'correction_group.pdf'))
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
		
				plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'WM-search_slopeselectivity_corrected.pdf'))
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
				
			plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'WM-search_slopeselectivity_ind.pdf'))
			plt.close()	


		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])	
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.xlabel('Memory CTF slope')
		plt.ylabel('Search CTF slope')
		plt.xlim(0, 0.14)
		plt.ylim(0, 0.14)

		plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'scatter_group.pdf'))
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

		plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'scatter_intervalslope_group_newint.pdf'))
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

	
		plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'scatter_individual.pdf'))
		plt.close()
	
		cor_matrix[p_matrix > 0.05] = 0		
		plt.imshow(cor_matrix, cmap = cm.viridis, interpolation='none', aspect='auto', origin = 'lower', extent=[-300,2200,-300,2200])
		plt.colorbar(ticks = (-1,1))
		plt.xlabel('memory')
		plt.ylabel('search')

		plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder], filename = 'slope_corr_matrix.pdf'))
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

		for condition in ['single_task','dual_task']:
			slopes = np.zeros((16,1281))
			for sj, sl in enumerate(all_slopes):
				slopes[sj,:] = sl[condition]['total']
				
			np.savetxt(self.FolderTrackerCTF('data',filename = '{}_{}_slope_values_total.csv'.format(condition, self.decoding)),slopes, delimiter=',')
	
		#plot_index = [1,3]
		for cnd_idx, power in enumerate(['evoked','total']):
			
			if plot == 'group':
				slopes = self.bootstrapSlopes(subject_id, all_slopes, power, info, 10000)
				
				plt.figure(figsize= (20,10))
				ax = plt.subplot(1,1,1)
				plt.xlim(info['times'][0], info['times'][-1])
				plt.ylim(-0.1, 0.2)
				plt.title(power + ' power across time')
				plt.axhline(y=0, xmin = 0, xmax = 1, color = 'black')

				for i, cond in enumerate(info['conditions']):
					dat = slopes[cond]['M']
					error = slopes[cond]['SE']
					plt.plot(info['times'], dat, color = ['g','r','y','b'][i], label = cond)
					plt.fill_between(info['times'], dat + error, dat - error, alpha = 0.2, color = ['g','r','y','b'][i])
				
				#if len(info['conditions']) > 1:
				#	dif = slopes[info['conditions'][0]]['slopes'] - slopes[info['conditions'][1]]['slopes']
				#	zmap_thresh = self.clusterPerm(dif)
				#	plt.fill_between(info['times'], -0.002, 0.002, where = zmap_thresh != 0, color = 'grey')
					
				plt.legend(loc='upper right', shadow=True)
				plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder,self.decoding,'figs'], filename = '{}_{}_slopes_group.pdf'.format(band,power)))
				plt.close()	

			elif plot == 'individual':										
				idx = [1,2,5,6,9,10,13,14]
				for i, sbjct in enumerate(subject_id):
					if i % 8 == 0: 
						plt.figure(figsize= (15,10))
						idx_cntr = 0	
					ax = plt.subplot(7,2,idx[idx_cntr],title = 'SUBJECT ' + str(sbjct), ylabel = 'CTF Slope', xlim = (info['times'][0],info['times'][-1]), ylim = (-0.1,0.2))
					for j, cond in enumerate(info['conditions']): 	
						plt.plot(info['times'], all_slopes[i][cond][power].squeeze(), color = ['g','r','y','b'][j], label = cond)	
					plt.legend(loc='upper right', frameon=False)	
					plt.axhline(y=0, xmin=info['times'][0], xmax=info['times'][-1], color = 'black')
					idx_cntr += 1

				nr_plots = int(np.ceil(len(subject_id)/8.0))
				for i in range(nr_plots,0,-1):
					plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'],filename = '{}_{}_slopes_ind_{}.pdf'.format(band,power,str(i))))
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
		Doc String clusterPerm. Script based on Matlab script. Results in paper are based on Matlab output!!!
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
				p_val_t, sig_t = self.permTTest(r_slopes_to, p_slopes_to, nr_perms = nr_perms, p_thresh = 0.01)
				
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
							
				plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = 'all_sig_' + cond + '.pdf'))
				plt.close()	

			plt.figure(figsize= (20,10))
			levels = np.linspace(0,0.2,1000)
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

			plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = 'all_freqs_' + cond + '.pdf'))
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
			plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = 'all_freqs_clusters_right.eps'), format = 'eps',dpi = 1000)
			plt.close()

	def plotCrossTraining(self, subject_id, perm = False, nr_perms = 1000):
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
		real_s = self.readCTF(subject_id, 'alpha','slopes')
		info = self.readCTF(subject_id, 'alpha', info = True)
		if perm:
			perm_s = self.readCTF(subject_id, 'alpha','slopes_perm')

		# set plotting parameters
		times = np.linspace(info['times'][0], info['times'][-1],info['nr_samps'])		
		X = np.tile(times,(info['nr_samps'],1)).T
		Y = np.tile(times,(info['nr_samps'],1))
				
		# combine individual slope data
		r_slopes_ev = np.empty((len(subject_id), info['nr_samps'],info['nr_samps'])) * np.nan
		r_slopes_to = np.copy(r_slopes_ev)
		if perm:
			p_slopes_ev = np.empty((len(subject_id), info['nr_samps'],info['nr_samps'], nr_perms)) * np.nan
			p_slopes_to = np.copy(p_slopes_ev)	

		for i in range(len(subject_id)):
			r_slopes_to[i,:,:] = real_s[i]['total']
			if perm: 
				p_slopes_to[i,:,:] = perm_s[i]['total']

		if perm:		
			plt.figure(figsize= (15,10))
			p_val_t, sig_t = self.permTTest(r_slopes_to, p_slopes_to, nr_perms = nr_perms, p_thresh = 0.01)
			
			levels = np.linspace(0,1,1000)
			ax = plt.subplot(2,2,1,title = 'p-values EVOKED', ylabel = 'Frequency')

			ax = plt.subplot(2,2,3,title = 'p-values TOTAL', ylabel = 'Frequency')
			plt.contourf(Y,X,p_val_t,levels, cmap = cm.jet_r)
			plt.colorbar(ticks = (levels[0],levels[-1]))

			ax = plt.subplot(2,2,4,title = 'sig TOTAL  < 0.01')
			plt.imshow(sig_t, cmap = cm.viridis, interpolation='none', aspect='auto', origin = 'lower', extent=[-300,2200,-300,2200])
							
			plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = 'all_sig_cross_training.pdf'))
			plt.close()	

		plt.figure(figsize= (20,10))
		levels = np.linspace(0,0.2,100)
		
		ax = plt.subplot(1,1,1, ylabel = 'memory dual task',xlabel = 'search dual task')
		total = np.mean(r_slopes_to,axis = 0); 
		# TEMP CODE 
		if perm: 
			total[sig_t == 0] = 0
		plt.imshow(total, aspect='auto', origin = 'lower', extent=[-300,2200,-300,2200], cmap = cm.viridis)
		plt.colorbar(ticks = (0,0.15))

		plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = 'cross_training.pdf'))
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

	def readCTF(self, subject_id, band,  dicts= 'ctf', info = False):
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
				with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_{}_{}.pickle'.format(str(sbjct),dicts,band)),'rb') as handle:
					ctf.append(pickle.load(handle))
		else:
			with open(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding], filename = '{}_info.pickle').format(band),'rb') as handle:
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

	def selectChannelIdold(self, subject_id, channels = ['A16','A17','A18','A19','A32','B24','B23','B22','B21','A24','A23','A22','A21','A20','A31','B25','B26','B27','B28','B29','A25','A26','A30','B31','B30','A27','A28','A29','B32']):
		'''
		selectChannelId. Function needs to be adjusted. At the moment works only with matlab matrix input where bad channels are already removed.
		Requires that bad channels are set for each participant.
		Function selects cahnnel indices used for SpatialCTF

		Arguments
		- - - - - 
		subject_id (int): subject id
		channels (list | str): list of channels used for analysis. If 'all' all channels are used for analysis 

		Returns
		- - - -
		idx_channel (list): list of selected (good) channel indices
		nr_selected_channels (int): nr of seleced channels
		nr_all_channels (int): nr of all channels (only good channels)
		all_channels (list): list of all channels (only good channels)

		all_channels
		
		''' 

		bad_channels = {'subject-1': ['A1','A15','A17','A18','B1','B2','B3','B11','B12','B20'],
						'subject-2': ['A1','A6','A7','A8','A16','A19','A20','A25','A30','B2','B5','B11','B12','B16','B18','B19','B20'],
						'subject-3': ['A15','A22','A24','B7','B8','B9','B11','B20','B21','B24','B27','B28','B29','B30'],
						'subject-4': ['A7','A9','A17', 'A23','B11', 'B20'],
						'subject-5': ['A8','A10','A14','A15','A16','A24','A25','A28', 'B10', 'B11', 'B17', 'B19', 'B23'],
						'subject-6': ['B2','B9', 'B10','B11','B12','B19'],
						'subject-7': ['A8','A15','A16','B10','B11'],
						'subject-8': ['A7','A8', 'A9', 'A14','A15','A17','B11','B12','B20'],
						'subject-9': ['A7','A16','A22','A24','B11','B12','B13','B20','B24'],
						'subject-10': ['A9','A15','A16','B1','B2','B9','B10','B11'],
						'subject-11': ['A7','A8','A9','A20','B10','B19','B20','B28','B29'],
						'subject-12': ['A7','A13','A14','A16','A23','B11','B12','B31'],
						'subject-13': ['A8','A25','A27','A28','B8','B11','B20'],
						'subject-14': ['A2','A15','A25','B9','B10','B20'],
						'subject-15': ['A6','A7','A8','A9','A14','A15','A16','A24','B2','B11'],
						'subject-16': ['A8','A23','A24','A28','B10','B11','B20'],
						}
			
		all_channels = []
		channel_labels = ['A','B']
		for label in channel_labels:
			all_channels += [label + str(i) for i in range(1,33)]

		# first drop bad electrodes
		for chnnl in bad_channels['subject-' + str(subject_id)]:
			all_channels.remove(chnnl)

		# get indices of channels to analyze
		if channels == 'all':
			idx_channel = range(len(all_channels))
		else:	
			idx_channel = []
			for chn in channels:
				if chn in all_channels:
					idx_channel.append(all_channels.index(chn))
			idx_channel.sort()		

		return	idx_channel, len(idx_channel), len(all_channels), all_channels


	def positionHeog(self, subject_id, interval, filter_time):
		'''
		positionHEOG. Function checks for effectiveness of ocular artifacts was effective. At the moment function is still hardcoded for JoCN project. Needs to be adjusted

		Arguments
		- - - - - 
		subject_id (int): subject id
		interval (list): start and endpoint in ms for analysis
		filter_time (int): time padded for filtering

		Returns
		- - - -
		
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

		plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = 'veog_ind_mem.pdf'))		
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

		plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'], filename = 'heog_search.pdf'))		
		plt.close()

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '4'
	os.environ['NUMEXP_NUM_THREADS'] = '4'
	os.environ['OMP_NUM_THREADS'] = '4'
	project_folder = '/home/dvmoors1/BB/Dist_suppression'
	os.chdir(project_folder) 
	subject_id = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]	

	### INITIATE SESSION ###
	header = 'dist_loc'
	if header == 'target_loc':
		conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
	else:
		conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']

	session = SpatialEM('all_channels_no-eye', header, nr_iter = 5, nr_blocks = 3, nr_bins = 6, nr_chans = 6, delta = False)
	
	#session.corrCTFSlopes(subject_id)
	### CTF and SLOPES ###
	for subject in subject_id:
		session.spatialCTFCrossTrain(subject, [-300,800],500, downsample = 4)
		#session.permuteSpatialCTFCrossTrain(subject, nr_perms = 500)

		#session.spatialCTF(subject,[-300,2200],500, ['single_task','dual_task'])
		#session.spatialCTF(subject,[-300,2200],500, ['single_task','dual_task'], freqs = dict(all=[4,34]), downsample = 4)
		#session.permuteSpatialCTF(subject_id = subject, nr_perms = 500)
	
	#session.CTFSlopesCrossTrain(subject_id, band = 'alpha', perm = False)
	#session.CTFSlopesCrossTrain(subject_id, band = 'alpha', perm = True)
	#session.CTFSlopes(subject_id, band = 'all', perm = False)
	#session.CTFSlopes(subject_id, band = 'all', perm = True)

	#session.topoWeights(subject_id)
	### PLOTTING ###
	#for plot in ['group','individual']:
	#	session.plotCTF(subject_id = subject_id, plot = plot)
	#	session.plotCTFSlopes(subject_id,'alpha', plot = plot)

	#session.plotCrossTraining(subject_id, perm = True, nr_perms = 500)	
	#session.plotTimefrequencySlopes(subject_id, perm = True, nr_perms = 500)
	#session.positionHeog(subject_id, [-300, 2200], 500)


