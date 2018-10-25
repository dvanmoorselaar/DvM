"""
analyze EEG data

Created by Dirk van Moorselaar on 30-03-2016.
Copyright (c) 2016 DvM. All rights reserved.
"""

#from session import *
import json
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

from helperFunctions import *
from BDM import BDM
from FolderStructure import FolderStructure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import pi, sqrt
from mne.filter import filter_data
from scipy.signal import hilbert
from scipy.stats import norm
from scipy.ndimage.measurements import label
from IPython import embed 

class CTF(BDM):
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
		delta (bool): should basisset assume a shape of CTF or be a delta function

		Returns
		- - - -
		self (object): SpatialEM object
		'''

		self.channel_folder = channel_folder
		self.decoding = decoding

		# specify model parameters
		self.nr_bins = nr_bins													# nr of spatial locations 
		self.nr_chans = nr_chans 												# underlying channel functions coding for spatial location
		self.nr_iter = nr_iter													# nr iterations to apply forward model 		
		self.nr_blocks = nr_blocks												# nr blocks to split up training and test data with leave one out test procedure
		self.sfreq = 512														# shift of channel position
		self.basisset = self.calculateBasisset(self.nr_bins, self.nr_chans, delta = delta)		# hypothesized set tuning functions underlying power measured across electrodes


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


	def calculateBasisset(self, nr_bins = 8, nr_chans = 8, sin_power = 7, delta = False):
		'''
		calculateBasisset returns a basisset that is used to reconstruct location-selective CTFs from the topographic distribution 
		of oscillatory power across electrodes. It is assumed that power measured at each electrode reflects the weighted sum of 
		a specific number of spatial channels (i.e. neural populations), each tuned for a different angular location. 

		The basisset either assumes a particular shape of the CTF. In this case the profile of each channel across angular locations 
		is modeled as a half sinusoid raised to the sin_power, given by: 

		R = sin(0.50)**sin_power

		, where 0 is angular location and R is response of the spatial channel in arbitrary units. If no shape is assumed the channel 
		response for each target position is set to 1, while setting all the other channel response to 0 (delta function).

		The R is then circularly shifted for each channel such that the peak response of each channel is centered over one of the location bins.
		
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

		_, nr_chan, nr_time = data.shape
		T = np.empty(data.shape, dtype= np.complex_) * np.nan
		E = np.copy(T)

		# loop over channels 
		for ch in range(nr_chan):
			# concatenate trials to speed up processing time
			x = np.ravel(data[:,ch,:])
			x_hilb = hilbert(filter_data(x, sfreq,band[0], band[1], method = 'iir', iir_params = dict(ftype = 'butterworth', order = 5)))
			x_hilb = np.reshape(x_hilb, (data.shape[0],-1))
			T[:,ch] = np.abs(x_hilb)**2
			E[:,ch] = x_hilb
			
		# trim filtered data to remove times that are not of interest (after filtering to avoid artifacts)
		E = E[:,:,tois]
		T = T[:,:,tois]

		# downsample to reduced sample rate (after filtering so that downsampling doesn't affect filtering)
		E = E[:,:,::downsample]
		T = T[:,:,::downsample]
		#E = mne.filter.resample(E, down = downsample, npad = 'auto')
		#T = mne.filter.resample(T, down = downsample, npad = 'auto')

		return E, T


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
		W, resid_w, rank_w, s_w = np.linalg.lstsq(C1,B1)		# estimate weight matrix W (nr_chans x nr_electrodes)
		C2, resid_c, rank_c, s_c = np.linalg.lstsq(W.T,B2.T)	# estimate channel response C2 (nr_chans x nr test blocks) 

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


	def maxTrial(self, conditions, pos_bins, of_interest, method = 'k-fold'):
		'''
		maxTrial calculates the maximum number of trials that can be used in the block/fold assignment 
		of the forward model such that the number of trials used in the IEM is equated across conditions

		Arguments
		- - - - - 
		conditions (array): array with conditions 
		pos_bins (array): array with position bins
		of_interest (array | str): condition(s) that will be analyzed
		method (str): method used for block assignment

		Returns
		- - - -
		nr_per_bin(int): maximum number of trials 
		'''

		if type(of_interest) == list and len(of_interest) > 2:

			bin_count = np.zeros((len(of_interest),self.nr_bins))
			for i, cond in enumerate(of_interest):
				temp_bin = pos_bins[conditions == cond]
				for b in range(self.nr_bins):
					bin_count[i,b] = sum(temp_bin == b) 

		elif of_interest == 'all':
			_, bin_count = np.unique(pos_bins, return_counts = True)

		else:
			_, bin_count = np.unique(pos_bins[cnds == of_interest], return_counts = True)			

		min_count = np.min(bin_count)
		if method == 'Foster':
			nr_per_bin = int(np.floor(min_count/self.nr_blocks))						# max number of trials per bin
		elif method == 'k-fold':
			nr_per_bin = int(np.floor(min_count/self.nr_iter)*self.nr_iter)

		return nr_per_bin


	def assignBlocks(self, pos_bin, idx_tr, nr_per_bin,  nr_iter):
		'''
		assignBlocks creates a random block assignment (train and test blocks)

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
		
		nr_trials = len(pos_bin)
		bl_assign = np.zeros((nr_iter, nr_trials))

		for i in range(nr_iter):
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
			bl_assign[i] = blocks
			tr_per_block = sum(blocks == 0)/self.nr_bins

		# split up data according to number of blocks in one test and remaining training sets
		tr_idx = np.zeros((nr_iter * self.nr_blocks, self.nr_bins, self.nr_blocks - 1, tr_per_block), dtype = int)
		te_idx = np.zeros((nr_iter * self.nr_blocks, self.nr_bins, tr_per_block), dtype = int)

		# make sure that data is splitted such that test contains indices of test trials
		# and train contains indices of rows for all seperate train trials
		idx = 0
		for itr in range(nr_iter):
			for bl in range(self.nr_blocks):
				for b in range(self.nr_bins): 
					# select trials where bin is b from test block trials 
					te_idx[idx,b] = idx_tr[np.where((bl_assign[itr] == bl) * (pos_bin == b))[0]]
					# select trials where bin is b from no test block trials
					train = idx_tr[np.where((~np.isnan(bl_assign[itr])) * (bl_assign[itr] != bl) * (pos_bin == b))[0]]
					random.shuffle(train)
					# split data in number of training blocks (i.e. nr test blocks - 1) and save data seperately
					train = np.array_split(train, self.nr_blocks - 1)
					for j in range(self.nr_blocks - 1):
						tr_idx[idx, b, j] = train[j]
				idx += 1

		return tr_idx, te_idx

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
		pos_bins, cnds, eegs, EEG = self.readData(sj, train_cnds + test_cnds)
		samples = np.logical_and(EEG.times >= window[0], EEG.times <= window[1])
		nr_samples = EEG.times[samples][::downsample].size
	
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
					print "\r{0:.2f}% of permutations".format((float(p)/nr_perm)*100),

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

	def spatialCTF(self, sj, window, conditions = 'all', freqs = dict(alpha = [8,12]), downsample = 1, method = 'Foster', plot = True, nr_perm = 0):
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
		plot (bool): show plots of total power slopes on subject level across conditions
		nr_perm (int): number of times model is run with permuted data

		Returns
		- - - -
		self(object): dict of CTF data
		'''	

		# set number of permutations
		nr_perm += 1
		
		ctf = {}
		ctf_info = {}

		# set nr_blocks (only relevant when Foster method is used)
		if method == 'k-fold':
			self.nr_blocks = 1
		
		# set condition for loop
		if type(conditions) == str:
			cnd_name = 'all'
			conditions = [conditions]
		else:
			cnd_name = 'cnds'													
			
		if 'all' in freqs.keys():
			frqs = np.vstack([[i, i + 4] for i in range(freqs['all'][0],freqs['all'][1] + 1,2)])
		else:									
			frqs = np.vstack([[freqs[band][0],freqs[band][1]] for band in freqs.keys()])
		nr_freqs = frqs.shape[0]

		# read in all data 
		pos_bins, cnds, eegs, EEG = self.readData(sj, conditions)
		samples = np.logical_and(EEG.times >= window[0], EEG.times <= window[1])	
		ctf_info['times'] = EEG.times[samples][::downsample]
		nr_samples = ctf_info['times'].size
		
		# Determine the number of trials that can be used for each position bin, matched across conditions
		nr_per_bin = self.maxTrial(cnds, pos_bins, conditions, method) # NEEDS TO BE FIXED

		# Loop over frequency bands (such that all data is filtered only once)
		for fr in range(nr_freqs):
			print('Frequency {} out of {}'.format(str(fr + 1), str(nr_freqs)))

			# Time-Frequency Analysis 
			E, T = self.powerAnalysis(eegs, samples, self.sfreq, [frqs[fr][0],frqs[fr][1]], downsample)

			# Loop over conditions
			for c, cnd in enumerate(conditions):

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
					# ADD CODE FOR NEW TRAINING PROCEDURE
						pass
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
						print "\r{0:.2f}% of permutations".format((float(p)/nr_perm)*100),
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
		slopes = {}
		for cnd in ctf.keys():
			e_slopes = np.zeros((nr_perm,nr_freqs, nr_samples))
			t_slopes = np.zeros((nr_perm,nr_freqs, nr_samples))
			slopes.update({cnd:{}})
			for p_ in range(nr_perm):
				e_slopes[p_,:,:] = self.calculateSlopes(np.mean(ctf[cnd]['tf_E'][p_], axis = 1),nr_freqs,nr_samples)
				t_slopes[p_,:,:] = self.calculateSlopes(np.mean(ctf[cnd]['tf_T'][p_], axis = 1),nr_freqs,nr_samples)
			
			if nr_perm == 1:
				slopes[cnd]['T_slopes'] = t_slopes[0]
				slopes[cnd]['E_slopes'] = e_slopes[0]
			else:
				slopes[cnd] = {'T_slopes_p': t_slopes[1:], 'T_slopes': t_slopes[0], 
							   'E_slopes_p': e_slopes[1:], 'E_slopes': e_slopes[0]}

		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_{}_slopes-{}_{}.pickle'.format(cnd_name,str(sj),method, freqs.keys()[0])),'wb') as handle:
			print('saving slopes dict')
			pickle.dump(slopes, handle)
	
		with open(self.FolderTracker(['ctf',self.channel_folder,self.decoding], filename = '{}_info.pickle'.format(freqs.keys()[0])),'wb') as handle:
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
						print '\r{0}% of permutations ({1} out of {2} conditions; {3} out of {4} frequencies); iter {5}'.format((float(p)/nr_perms)*100, cond + 1, len(info['conditions']),str(freq + 1),str(info['nr_freqs']),itr + 1),

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
			steps = int(self.nr_chans / 2 + 1) 
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
						for p in range(nr_perms):
							print p												# loop over permutations
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
				print sbjct
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



