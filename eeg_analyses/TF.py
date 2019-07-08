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

from mne.filter import filter_data
from mne.time_frequency import tfr_array_morlet
from mne.baseline import rescale
from scipy.signal import hilbert
from numpy.fft import fft, ifft,rfft, irfft
from support.FolderStructure import *
from support.support import trial_exclusion
from signals.signal_processing import *
from IPython import embed


class TF(FolderStructure):

	def __init__(self, beh, eeg, laplacian=True):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		self.beh = beh
		self.EEG = eeg
		self.laplacian = laplacian

	def selectTFData(self, laplacian, excl_factor):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		# load processed behavior and eeg
		beh = self.beh
		EEG = self.EEG

		# check whether trials need to be excluded
		if type(excl_factor) == dict: # remove unwanted trials from beh
			beh, EEG = trial_exclusion(beh, EEG, excl_factor)

		# select electrodes of interest
		picks = mne.pick_types(EEG.info, eeg=True, exclude='bads')
		eegs = EEG._data[:,picks,:]

		if laplacian:
			x,y,z = np.vstack([EEG.info['chs'][i]['loc'][:3] for i in picks]).T
			leg_order = 10 if picks.size <=100 else 12
			eegs = laplacian_filter(eegs, x, y, z, leg_order = leg_order, smoothing = 1e-5)
	
		return eegs, beh

	def RESS(self, sfreq, time_oi, peakwidth = .5, neighfreq = 1, neighwidt = 1, peak_freqs = [6, 7.5], elec_oi = ['Oz','O2']):

		# set FFT parameters
		sfreq = eeg.info['sfreq']
		nfft = np.ceil(sfreq/.1 ) # .1 Hz resolution
		t_idx_s, t_idx_e  = [np.argmin(abs(eeg.times - t)) for t in time_oi]
		hz = np.linspace(0,sfreq,nfft)

		# extract eeg data (should be implemented in cnd loop)
		data = eeg._data[cnd_mask,:,:]
		#dataX = np.mean(abs(fft(data[:,t_idx_s:t_idx_e, nfft, axis = 2)/ (t_idx_e - t_idx_s)), axis = 0) # This needs to be checked!!!!!
		
	def FGFilter(self, X, sfreq, f, fwhm):
		"""[summary]
		
		Arguments:
			X {[type]} -- [description]
			sfreq { [type]} -- [description]
			f {[type]} -- [description]
			fwhm {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		"""


		
		# compute and apply filter

		# frequencies
		hz = np.linspace(0,sfreq,X.shape[1])

		# create Gaussian (CHECK THIS)
		s  = fwhm*(2 * np.pi-1)/(4*np.pi) # normalized width
		x  = hz-f                 		  # shifted frequencies
		fx = np.exp(-.5*(x/s)**2)    	  # gaussian
		fx = fx/np.max(fx)          	  # gain-normalized

		# filter data
		filtX = 2 * np.real(ifft( fft(data, [],2)* fx, [],2))
		#filtdat = 2*real( ifft( bsxfun(@times,fft(data,[],2),fx) ,[],2) );
		
		# compute empirical frequency and standard deviation
		#idx = dsearchn(hz',f);
		#emp_vals[1] = hz[idx]

		# find values closest to .5 after MINUS before the peak
		#emp_vals[2] = hz(idx-1+dsearchn(fx(idx:end)',.5)) - hz(dsearchn(fx(1:idx)',.5))

		return filtdat, emp_vals





	@staticmethod	
	def nextpow2(i):
		'''
		Gives the exponent of the next higher power of 2
		'''

		n = 1
		while 2**n < i: 
			n += 1
		
		return n

	def topoFlip(self, eegs, var, ch_names, left = []):
		''' 
		Flips the topography of trials where the stimuli of interest was presented 
		on the left (i.e. right hemifield). After running this function it is as if 
		all stimuli are presented right (i.e. the left hemifield)

		Arguments
		- - - - - 
		eegs(array): eeg data
		var (array|list): location info per trial
		ch_names (list): list of channel names
		left (list): list containing stimulus labels indicating spatial position 

		Returns
		- - - -
		inst (instance of ERP): The modified instance 

		'''	
		
		# dictionary to flip topographic layout
		flip_dict = {'Fp1':'Fp2','AF7':'AF8','AF3':'AF4','F7':'F8','F5':'F6','F3':'F4',\
					'F1':'F2','FT7':'FT8','FC5':'FC6','FC3':'FC4','FC1':'FC2','T7':'T8',\
					'C5':'C6','C3':'C4','C1':'C2','TP7':'TP8','CP5':'CP6','CP3':'CP4',\
					'CP1':'CP2','P9':'P10','P7':'P8','P5':'P6','P3':'P4','P1':'P2',\
					'PO7':'PO8','PO3':'PO4','O1':'O2','Fpz':'Fpz','AFz':'AFz','Fz':'Fz',\
					'FCz':'FCz','Cz':'Cz','CPz':'CPz','Pz':'Pz','POz':'POz','Oz':'Oz',\
					'Fp2':'Fp1','AF8':'AF7','AF4':'AF3','F8':'F7','F6':'F5','F4':'F3',\
					'F2':'F1','FT8':'FT7','FC6':'FC5','FC4':'FC3','FC2':'FC1','T8':'T7',\
					'C6':'C5','C4':'C3','C2':'C1','TP7':'TP8','CP5':'CP6','CP3':'CP4',\
					'CP1':'CP2','P9':'P10','P7':'P8','P5':'P6','P3':'P4','P1':'P2','PO8':'PO7',\
					'PO4':'PO3','O2':'O1'}

		idx_l = np.sort(np.hstack([np.where(var == l)[0] for l in left]))

		# left stimuli are flipped as if presented right
		pre_flip = eegs[idx_l,:,:]
		flipped = np.zeros(pre_flip.shape)

		# do actual flipping
		for key in flip_dict.keys():
			flipped[:,ch_names.index(flip_dict[key]),:] = pre_flip[:,ch_names.index(key),:]

		eegs[idx_l,:,:] = flipped
		
		return eegs


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
		Z = np.zeros((contra_ipsi_pair.size, num_frex, nr_time))
		norm = np.zeros((contra_ipsi_pair.size, num_frex, nr_time))
		Z_elec = []
		# loop over contra_ipsi pairs
		pair_idx = 0
		for (contra_elec, ipsi_elec) in contra_ipsi_pair:
			
			Z_elec.append(contra_elec)	
			
			# get indices of electrode pair
			contra_idx = ch_names.index(contra_elec)
			ipsi_idx = ch_names.index(ipsi_elec)
			contra_ipsi_norm = (raw_power[:,:,contra_idx] - raw_power[:,:,ipsi_idx])/(raw_power[:,:,contra_idx] + raw_power[:,:,ipsi_idx])
			norm[pair_idx] = contra_ipsi_norm.mean(axis = 0)

			# # get the real difference
			#real_diff = raw_power[:,:,contra_idx] - raw_power[:,:,ipsi_idx]

			# # create distribution of fake differences
			#try:
			# 	fake_diff = np.zeros((nr_perm,) + real_diff.shape)
			#except:
			#	embed()
			# 	print 'Data of real_diff is averaged to reduce the size of the array of two becuase of memory problems'
			# 	if real_diff.shape[0] % 2 == 1:
			# 		real_diff = real_diff[:-1]
			# 	to_split = np.random.permutation(real_diff.shape[0])%2
			# 	real_diff = np.mean((real_diff[to_split == 0],real_diff[to_split == 1]), axis = 0)
			# 	fake_diff = np.zeros((nr_perm,) + real_diff.shape)

			#for p in range(nr_perm):
			 	# randomly flip ipsi and contra
			# 	signed = np.sign(np.random.normal(size = real_diff.shape[0]))
			# 	permuter = np.tile(signed[:,np.newaxis, np.newaxis], ((1,) + real_diff.shape[-2:]))
			# 	fake_diff[p] = real_diff * permuter

			# # Z scoring
			#print('Z scoring pair {}-{}'.format(contra_elec, ipsi_elec))
			#Z[pair_idx] = (np.mean(real_diff, axis = 0) - np.mean(fake_diff, axis = (0,1))) / np.std(np.mean(fake_diff, axis = 1), axis = 0, ddof = 1)	
			pair_idx += 1

		return norm, Z_elec

	def TFanalysis(self, sj, cnds, cnd_header, time_period, base_period = None, elec_oi = 'all',factor = None, method = 'hilbert', flip = None, base_type = 'conspec', downsample = 1, min_freq = 5, max_freq = 40, num_frex = 25, cycle_range = (3,12), freq_scaling = 'log'):
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
			key = flip.keys()[0]
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
		base = {}
		plot_dict = {}

		# loop over conditions
		for c, cnd in enumerate(cnds):
			tf.update({cnd: {}})
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

				print '\r Decomposed {0:.0f}% of channels ({1} out {2} conditions)'.format((float(idx)/nr_chan)*100, c + 1, len(cnds)),

				# fft decomposition
				if method == 'wavelet':
					eeg_fft = fft(eegs[cnd_idx,ch_idx].ravel(), l_conv)    # eeg is concatenation of trials after ravel

				# loop over frequencies
				for f in range(num_frex):

					if method == 'wavelet':
						# convolve and get analytic signal (OUTPUT DIFFERS SLIGHTLY FROM MATLAB!!! DOUBLE CHECK)
						m = ifft(eeg_fft * fft(wavelets[f], l_conv), l_conv)
						m = m[:nr_time * cnd_idx.size + nr_time - 1]
						m = np.reshape(m[(nr_time-1)/2 - 1:-(nr_time-1)/2-1], (nr_time, -1), order = 'F').T 
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
				tf[cnd]['base_power'] = 10*np.log10(tf[cnd]['power']/np.repeat(base[cnd][:,:,np.newaxis],idx_2_save.size,axis = 2))
			elif base_type == 'conavg':	
				con_avg = np.mean(np.stack([base[cnd] for cnd in cnds]), axis = 0)
				tf[cnd]['base_power'] = 10*np.log10(tf[cnd]['power']/np.repeat(con_avg[:,:,np.newaxis],idx_2_save.size,axis = 2))
			elif base_type == 'Z':
				print('For permutation procedure it is assumed that it is as if all stimuli of interest are presented right')
				tf[cnd]['Z_power'], z_info = self.permuted_Z(tf[cnd]['power'],ch_names, num_frex, idx_2_save.size) 
				tf.update(dict(z_info = z_info))

			# power values can now safely be averaged
			tf[cnd]['power'] = np.mean(tf[cnd]['power'], axis = 0)

		# save TF matrices
		with open(self.FolderTracker(['tf',method,'proactive'],'{}-tf.pickle'.format(sj)) ,'wb') as handle:
			pickle.dump(tf, handle)		
	
	def createMorlet(self, min_freq, max_freq, num_frex, cycle_range, freq_scaling, nr_time, s_freq):
		''' 

		Creates Morlet wavelets for TF decomposition (based on Ch 12, 13 of Mike X Cohen, Analyzing neural time series data)

		Arguments
		- - - - - 
		min_freq (int): minimum frequency for TF analysis
		max_freq (int): maximum frequency for TF analysis
		num_frex (int): number of frequencies in TF analysis
		cycle_range (tuple): number of cycles increases in the same number of steps used for scaling
		freq_scaling (str): specify whether frequencies are linearly or logarithmically spaced. 
							If main results are expected in lower frequency bands logarithmic scale 
							is adviced, whereas linear scale is advised for expected results in higher
							frequency bands
		nr_time (int): wavelets should be long enough such that the lowest frequency wavelets tapers to zero. As
					   a general rule, nr_time can be equivalent to nr of timepoints in epoched eeg data
		s_freq (float): sampling frequency in Hz

		Returns
		- - - 
		
		wavelets(array): 
		frex (array):

		'''

		# setup wavelet parameters (gaussian width and time)
		if freq_scaling == 'log':
			frex = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frex)
			s = np.logspace(np.log10(cycle_range[0]), np.log10(cycle_range[1]),num_frex) / (2*math.pi*frex)
		elif freq_scaling == 'linear':
			frex = np.linspace(min_freq, max_freq, num_frex)
			s = np.linspace(cycle_range[0], cycle_range[1], num_frex) / (2*math.pi*frex)
		else:
			raise ValueError('Unknown frequency scaling option')

		t = np.arange(-nr_time/s_freq/2, nr_time/s_freq/2, 1/s_freq)	
		 	
		# create wavelets
		wavelets = np.zeros((num_frex, len(t)), dtype = complex)
		for fi in range(num_frex):
			wavelets[fi,:] = np.exp(2j * math.pi * frex[fi] * t) * np.exp(-t**2/(2*s[fi]**2))	
		
		return wavelets, frex	

 	