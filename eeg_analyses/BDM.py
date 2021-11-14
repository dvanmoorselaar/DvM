import os
import mne
import pickle
import random
import copy
import itertools

#import matplotlib
#matplotlib.use('agg') # now it works via ssh connection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Generic, Union
from support.FolderStructure import *
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from support.support import select_electrodes, trial_exclusion
from scipy.stats import rankdata
from scipy.signal import hilbert

from IPython import embed


class BDM(FolderStructure):

	"""The BDM object supports multivariate decoding functionality to predict 
	an experimental variable (condition) given an observed pattern of brain activity.
	By default the BDM class employs Linear Discriminant Analysis (LDA) to perform decoding.

	The BDM class makes use of k-fold cross validation, in which the trials are split up into k equally sized folds. 
	The model is trained on k-1 folds, and testing is done on the remaining fold that was not used for training. This procedure 
	is repeated k times until each fold (all data) has been tested exactly once, while on any given iteration the trials used 
	for training are independent from the trials that were used for testing.

	In class assignment, BDM applies event balancing by default through undersampling so that each class has the same number of observations.
	(i.e., if one class has 200 observations, and the other class has 100 observations, the number of observations in the first class would 
	be artificially lowered by randomly removing 100 observations). Consequently, if the design is heavily unbalanced between classes you may 
	loose a lot of data. 

	Area under the curve (AUC; Bradley, 1997), a metric derived from signal detection theory, is the default performance measure that BDM computes

	Args:
		FolderStructure (object): Class that creates file paths to load raw eeg/ behavior and save decoding ouput
	"""

	def __init__(self, beh: pd.DataFrame, epochs: mne.Epochs, to_decode: str, nr_folds: int, 
				method: str = 'auc', elec_oi: Union[str, list] = 'all', downsample: int = 128, 
				avg_runs: int = 1, bdm_filter: Optional[dict] = None, baseline: Optional[tuple] = None):
		"""set decoding parameters that will be used in BDM class

		Args:
			beh (pd.DataFrame): Dataframe with behavioral parameters per epoch (see eeg)
			eeg (mne.Epochs): epoched eeg data (linked to beh)
			to_decode (str): column in beh that contains classes used for decoding
			nr_folds (int): specifies how many folds will be used for k-fold cross validation
			method (str, optional): [description]. Defaults to 'auc'.
			elec_oi (Optional[str, list], optional): [description]. Defaults to 'all'.
			downsample (int, optional): [description]. Defaults to 128.
			bdm_filter (Optional[dict], optional): [description]. Defaults to None.
			baseline (Optional[tuple], optional): [description]. Defaults to None.
		"""	
							
		self.beh = beh
		self.epochs = epochs.apply_baseline(baseline = baseline)
		self.to_decode = to_decode
		self.nr_folds = nr_folds
		self.elec_oi = elec_oi
		self.downsample = downsample
		self.bdm_filter = bdm_filter
		self.method = method
		self.avg_runs = avg_runs
		if bdm_filter != None:
			self.bdm_type, self.bdm_band = list(bdm_filter.items())[0]
		else:	 
			self.bdm_type = 'broad'

	def selectBDMData(self, epochs, beh, time, excl_factor = None):
		''' 
		Arguments
		- - - - - 
		EEG (object):
		beh (dataFrame):
		time (tuple | list): time samples (start to end) for decoding
		excl_factor (dict): see Classify documentation
		Returns
		- - - -
		eegs (array): eeg data (trials X electrodes X time)
		beh (dict): contains variables of interest for decoding
		'''

		# check whether trials need to be excluded
		if type(excl_factor) == dict: # remove unwanted trials from beh
			beh, epochs = trial_exclusion(beh, epochs, excl_factor)

		# apply filtering and downsampling (if specified)
		if self.bdm_type != 'broad':
			print('filtering raw eeg data')
			T = np.empty(epochs._data.shape, dtype= np.complex_) * np.nan
			# selecting eeg electrodes
			for ch in range(epochs._data.shape[1]):
				x = np.ravel(epochs._data[:,ch,:])
				x_hilb = hilbert(filter_data(x, epochs.info['sfreq'],self.bdm_band[0], self.bdm_band[1], 
								method = 'iir', iir_params = dict(ftype = 'butterworth', order = 5)))
				x_hilb = np.reshape(x_hilb, (epochs._data.shape[0],-1))
				T[:,ch] = np.abs(x_hilb)**2

			epochs._data= T

		if self.downsample < int(epochs.info['sfreq']):
			print('downsampling data')
			epochs.resample(self.downsample, npad='auto')

		# select time window and EEG electrodes
		s, e = [np.argmin(abs(epochs.times - t)) for t in time]
		picks = mne.pick_types(epochs.info, eeg=True, eog= True, exclude='bads')
		picks = select_electrodes(np.array(epochs.ch_names)[picks], self.elec_oi)
		eegs = epochs._data[:,picks,s:e]
		times = epochs.times[s:e]

		# if specified average over trials 
		#beh, eegs = self.averageTrials(beh, eegs, self.to_decode, cnd_header, 4)

		return 	eegs, beh, times

	def averageTrials(self, X, Y, trial_avg):


		# DOWNSIDE OF ONLY AVERAGING AT THIS STAGE AND WHIT PRSENT METHOD IS THAT TRIALS ARE LOSSED
		# initiate arrays
		x_, y_ = [], []

		# loop over each label in Y
		for label in np.unique(Y):
			idx = np.where(Y[0] == label)[0]
			# note that trial order is shuffled 
			list_of_groups = zip(*(iter(idx),) * trial_avg)
			for sub in list_of_groups:
				x_.append(X[:,np.array(sub)].mean(axis = 1))
				y_.append(label)

		# create averaged x and y arrays
		x_ = np.swapaxes(np.stack(x_),0,1)
		y_ = np.array([y_,]*Y.shape[0])

		return x_, y_

	def Classify(self,sj, cnds, cnd_header, time, collapse = False, bdm_labels = 'all', excl_factor = None, nr_perm = 0, gat_matrix = False, downscale = False, save = True):
		''' 
		Arguments
		- - - - - 
		sj(int): subject number
		cnds (list): list of condition labels (as stored in beh dict). 
		cnd_header (str): variable name containing conditions of interest
		bdm_labels (list | str): Specifies whether all labels or only a subset of labels should be decoded
		excl_factor (dict | None): This gives the option to exclude specific conditions from analysis. 
								For example, to only include trials where the cue was pointed to the left and not to the right specify
								the following: factor = dict('cue_direc': ['right']). Mutiple column headers and multiple variables per 
								header can be specified 
		time (tuple | list): time samples (start to end) for decoding
		collapse (boolean): If True also run analysis collapsed across all conditions
		nr_perm (int): If perm = 0, run standard decoding analysis. 
					If perm > 0, the decoding is performed on permuted labels. 
					The number sets the number of permutations
		gat_matrix (bool): If True, train X test decoding analysis is performed
		downscale (bool): If True, decoding is repeated with increasingly less trials. Set to True if you are 
						interested in the minumum number of trials that support classification
		save (bool): sets whether output is saved (via standard file organization) or returned 	
		Returns
		- - - -
		'''	

		# first round of classification is always done on non-permuted labels
		nr_perm += 1

		# read in data 
		eegs, beh, times = self.selectBDMData(self.epochs, self.beh, time, excl_factor)	
		
		# select minumum number of trials given the specified conditions
		max_tr = [self.selectMaxTrials(beh, cnds, bdm_labels,cnd_header)]

		if downscale:
			max_tr = [(i+1)*self.nr_folds for i in range(max_tr[0]/self.nr_folds)][::-1]

		# create dictionary to save classification accuracy
		classification = {'info': {'elec': self.elec_oi, 'times':times}}

		if collapse:
			beh['collapsed'] = 'no'
			cnds += ['collapsed']
	
		# loop over conditions
		for cnd in cnds:

			# reset selected trials
			bdm_info = {}

			if cnd != 'collapsed':
				cnd_idx = np.where(beh[cnd_header] == cnd)[0]
				if collapse:
					beh['collapsed'][cnd_idx] = 'yes'
			else:
				# reset max_tr again such that analysis is not underpowered
				max_tr = [self.selectMaxTrials(beh, ['yes'], bdm_labels,'collapsed')]
				cnd_idx =  np.where(np.sum(
					[(beh[cnd_header] == c).values for c in cnds], 
					axis = 0))[0]
			cnd_labels = beh[self.to_decode][cnd_idx].values

			if bdm_labels != 'all':
				sub_idx = [i for i,l in enumerate(cnd_labels) if l in bdm_labels]	
				cnd_idx = cnd_idx[sub_idx]
				cnd_labels = cnd_labels[sub_idx]

			labels = np.unique(cnd_labels)
			print ('\nYou are decoding {}. The nr of trials used for folding is set to {}'.format(cnd, max_tr[0]))
			
			# initiate decoding array
			if gat_matrix:
				class_acc = np.empty((self.avg_runs, nr_perm, eegs.shape[2], eegs.shape[2])) * np.nan
				label_info = np.empty((self.avg_runs, nr_perm, eegs.shape[2], eegs.shape[2], labels.size)) * np.nan
			else:	
				class_acc = np.empty((self.avg_runs, nr_perm, eegs.shape[2])) * np.nan	
				label_info = np.empty((self.avg_runs,nr_perm, eegs.shape[2], labels.size)) * np.nan

			# permutation loop (if perm is 1, train labels are not shuffled)
			for p in range(nr_perm):

				if p > 0: # shuffle condition labels
					np.random.shuffle(cnd_labels)
			
				for i, n in enumerate(max_tr):
					if i > 0:
						print('Minimum condition label downsampled to {}'.format(n))
						bdm_info = {}

					# select train and test trials
					for run in range(self.avg_runs):
						train_tr, test_tr, bdm_info = self.trainTestSplit(cnd_idx, cnd_labels, n, {}) #  labels not saved
						Xtr, Xte, Ytr, Yte = self.trainTestSelect(beh[self.to_decode], eegs, train_tr, test_tr)
						# TRIAL AVERAGING NEEDS UPDATING
						#self.trial_avg = 3
						#if self.trial_avg > 1:
						#	Xtr, Ytr = self.averageTrials(Xtr, Ytr, 2)
						#	Xte, Yte = self.averageTrials(Xte, Yte, 2)
						# do actual classification
						#class_acc[p], label_info[p] = self.linearClassification(eegs, train_tr, test_tr, n, cnd_labels, gat_matrix)
						
						class_acc[run, p], label_info[run,p] = self.crossTimeDecoding(Xtr, Xte, Ytr, Yte, labels, gat_matrix)

					class_acc = class_acc.mean(axis = 0)	
					if i == 0:
						classification.update({cnd:{'standard': copy.copy(class_acc[0])}, 'bdm_info': bdm_info})
					else:
						classification[cnd]['{}-nrlabels'.format(n)] = copy.copy(class_acc[0])
								
			if nr_perm > 1:
				classification[cnd].update({'perm': class_acc[1:]})
	
		# store classification dict	
		if save: 
			with open(self.FolderTracker(['bdm',self.elec_oi, self.to_decode], filename = 'class_{}-{}.pickle'.format(sj,self.bdm_type)) ,'wb') as handle:
				pickle.dump(classification, handle)
		else:
			return classification	

	def localizerClassify(self, sj, loc_beh, loc_eeg, cnds, cnd_header, time, tr_header, te_header, collapse = False, loc_excl = None, test_excl = None, gat_matrix = False, save = True):
		"""Training and testing is done on seperate/independent data files
		
		Arguments:
			sj {int} -- Subject number
			loc_beh {DataFrame} -- DataFrame that contains labels necessary for training the model
			loc_eeg {object} -- EEG data used to train the model (MNE Epochs object)
			cnds {list} -- List of conditions. Decoding is done for each condition seperately
			cnd_header {str} -- Name of column that contains condition info in test behavior file
			time {tuple} -- Time window used for decoding
			tr_header {str} -- Name of column that contains training labels
			te_header {[type]} -- Name of column that contains testing labels
		
		Keyword Arguments:
			collapse {bool} -- If True also run analysis collapsed across all conditions
			loc_excl {dict| None} -- Option to exclude trials from localizer. See Classify for more info (default: {None})
			test_excl {[type]} -- Option to exclude trials from (test) analysis. See Classify for more info (default: {None})
			gat_matrix {bool} -- If set to True, a generalization across time matrix is created (default: {False})
			save {bool} -- Determines whether output is saved (via standard file organization) or returned (default: {True})
		
		Returns:
			classification {dict} -- Decoding output (for each condition seperately)
		"""

		# set up localizer data 
		tr_eegs, tr_beh, times = self.selectBDMData(loc_eeg, loc_beh, time, loc_excl)
		# set up test data
		te_eegs, te_beh, times = self.selectBDMData(self.EEG, self.beh, time, test_excl)
		
		# create dictionary to save classification accuracy
		classification = {'info': {'elec': self.elec_oi, 'times':times}}

		# specify training parameters (fixed for all testing conditions)
		tr_labels = tr_beh[tr_header].values
		min_nr_tr_labels = min(np.unique(tr_labels, return_counts = True)[1])
		# make sure training is not biased towards a label
		tr_idx = np.hstack([random.sample(np.where(tr_beh[tr_header] == label )[0], 
							k = min_nr_tr_labels) for label in np.unique(tr_labels)])
		Ytr = tr_beh[tr_header][tr_idx].values.reshape(1,-1)
		Xtr = tr_eegs[tr_idx,:,:][np.newaxis, ...]

		if collapse:
			cnds += ['collapsed']

		# loop over all conditions
		for cnd in cnds:

			# set condition mask
			if cnd != 'collapsed':
				test_mask = (te_beh[cnd_header] == cnd).values
			else:
				test_mask =  np.array(np.sum(
					[(beh[cnd_header] == c).values for c in cnds], 
					axis = 0), dtype = bool)	
			# specify testing parameters
			Yte = te_beh[te_header][test_mask].values.reshape(1,-1)
			Xte = te_eegs[test_mask,:,:][np.newaxis, ...]
	
			# do actual classification
			class_acc, label_info = self.crossTimeDecoding(Xtr, Xte, Ytr, Yte, np.unique(Ytr), gat_matrix)

			classification.update({cnd:{'standard': copy.copy(class_acc)}})
		
		# store classification dict	
		if save: 
			with open(self.FolderTracker(['bdm',self.elec_oi, 'cross'], filename = 'class_{}-{}.pickle'.format(sj,te_header)) ,'wb') as handle:
				pickle.dump(classification, handle)
		
		return classification


	def crossClassify(self, sj, cnds, cnd_header, time, tr_header, te_header, tr_te_rel = 'ind', excl_factor = None, tr_factor = None, te_factor = None, bdm_labels = 'all', gat_matrix = False, save = True, bdm_name = 'cross'):	
		'''
		UPdate function but it does the trick
		'''

		# read in data 
		print ('NR OF TRAIN LABELS DIFFER PER CONDITION!!!!')
		print ('DOES NOT YET CONTAIN FACTOR SELECTION FOR DEPENDENT DATA')

		eegs, beh, times = self.selectBDMData(self.EEG, self.beh, time, excl_factor)		
		nr_time = times.size
		
		if cnds == 'all':
			cnds = [cnds]

		if tr_te_rel == 'ind':	
			# use train and test factor to select independent trials!!!	
			tr_mask = [(beh[key] == f).values for  key in tr_factor.keys() for f in tr_factor[key]]
			for m in tr_mask: 
				tr_mask[0] = np.logical_or(tr_mask[0],m)
			tr_eegs = eegs[tr_mask[0]]
			tr_beh = beh.drop(np.where(~tr_mask[0])[0])
			tr_beh.reset_index(inplace = True, drop = True)
			
			te_mask = [(beh[key] == f).values for  key in te_factor.keys() for f in te_factor[key]]
			for m in te_mask: 
				te_mask[0] = np.logical_or(te_mask[0],m)
			te_eegs = eegs[te_mask[0]]
			te_beh = beh.drop(np.where(~te_mask[0])[0])
			te_beh.reset_index(inplace = True, drop = True)

		# create dictionary to save classification accuracy
		classification = {'info': {'elec': self.elec_oi, 'times': times}}

		if cnds == 'all':
			cnds = [cnds]
	
		# loop over conditions
		for cnd in cnds:
			if type(cnd) == tuple:
				tr_cnd, te_cnd = cnd
			else:
				tr_cnd = te_cnd = cnd	

			#print ('You are decoding {} with the following labels {}'.format(cnd, np.unique(tr_beh[self.decoding], return_counts = True)))
			if tr_te_rel == 'ind':
				tr_mask = (tr_beh[cnd_header] == tr_cnd).values
				Ytr = tr_beh[tr_header][tr_mask].values.reshape(1,-1)
				Xtr = tr_eegs[tr_mask,:,:][np.newaxis, ...]

				te_mask = (te_beh[cnd_header] == te_cnd).values
				Yte = te_beh[te_header][te_mask].values.reshape(1,-1)
				Xte = te_eegs[te_mask,:,:][np.newaxis, ...]
			else:
				if cnd != 'all':
					cnd_idx = np.where(beh[cnd_header] == cnd)[0]
					cnd_labels = beh[self.to_decode][cnd_idx].values
				else:
					cnd_idx = np.arange(beh[cnd_header].size)
					cnd_labels = beh[self.to_decode].values

				# select train and test trials	
				train_tr, test_tr, bdm_info = self.trainTestSplit(cnd_idx, cnd_labels, max_tr, {})
				Xtr, Xte, Ytr, Yte = self.trainTestSelect(beh[tr_header], eegs, train_tr, test_tr)
	
			# do actual classification
			class_acc, label_info = self.crossTimeDecoding(Xtr, Xte, Ytr, Yte, np.unique(Ytr), gat_matrix)
	
			classification.update({tr_cnd:{'standard': copy.copy(class_acc)}})
		# store classification dict	
		if save: 
			with open(self.FolderTracker(['bdm', self.elec_oi, 'cross', bdm_name], filename = 'class_{}-{}.pickle'.format(sj,self.bdm_type)) ,'wb') as handle:
				pickle.dump(classification, handle)
		else:
			return classification	

	def crossTimeDecoding(self, Xtr, Xte, Ytr, Yte, labels, gat_matrix = False):
		'''
		At the moment only supports linear classification as implemented in sklearn. Decoding is done 
		across all time points. 
		Arguments
		- - - - - 
		Xtr (array): 
		xte (array): 
		Ytr (array):
		Yte (array): 
		labels (array | list):
		gat_matrix (bool):
		
		Returns
		- - - -
		class_acc (array): classification accuracies (nr train time X nr test time). If Decoding is only done across diagonal nr test time equals 1 
		label_info (array): Shows how frequent a specific label is selected (nr train time X nr test time X nr unique labels).   
		'''

		# set necessary parameters
		nr_labels = len(labels)
		N = self.nr_folds
		nr_elec, nr_time = Xtr.shape[-2], Xtr.shape[-1]
		if gat_matrix:
			nr_test_time = nr_time
		else:
			nr_test_time = 1	

		# initiate linear classifier
		lda = LinearDiscriminantAnalysis()

		# inititate decoding arrays
		class_acc = np.zeros((N,nr_time, nr_test_time))
		label_info = np.zeros((N, nr_time, nr_test_time, nr_labels))

		for n in range(N):
			print('\r Fold {} out of {} folds'.format(n + 1,N),end='')
			Ytr_ = Ytr[n]
			Yte_ = Yte[n]
			
			for tr_t in range(nr_time):
				for te_t in range(nr_test_time):
					if not gat_matrix:
						te_t = tr_t

					Xtr_ = Xtr[n,:,:,tr_t]
					Xte_ = Xte[n,:,:,te_t]

					# train model and predict
					lda.fit(Xtr_,Ytr_)
					scores = lda.predict_proba(Xte_) # get posteriar probability estimates
					predict = lda.predict(Xte_)
					class_perf = self.computeClassPerf(scores, Yte_, np.unique(Ytr_), predict) # 

					if not gat_matrix:
						#class_acc[n,tr_t, :] = sum(predict == Yte_)/float(Yte_.size)
						label_info[n, tr_t, :] = [sum(predict == l) for l in labels]
						class_acc[n,tr_t,:] = class_perf # 

					else:
						#class_acc[n,tr_t, te_t] = sum(predict == Yte_)/float(Yte_.size)
						label_info[n, tr_t, te_t] = [sum(predict == l) for l in labels]	
						class_acc[n,tr_t, te_t] = class_perf

		class_acc = np.squeeze(np.mean(class_acc, axis = 0))
		label_info = np.squeeze(np.mean(label_info, axis = 0))

		return class_acc, label_info

	def computeClassPerf(self, scores, true_labels, label_order, predict):
		'''
		
		Computes classifier performance, using the test scores of the classifier and the true labels of
		the test set.
		Arguments
		- - - - - 
		scores (array): confidences scores of the classifier to the trials in the test set
		true_labels (array): true labels of the trials in the test set
		label_order (list): order of columns in scores
		predict (array): predicted labels
		Returns
		- - - -
		class_perf (float): classification accuracy as calculated with specified method
 
		'''

		if self.method == 'auc':
			
			# shift true_scores to indices
			true_labels = np.array([list(label_order).index(l) for l in true_labels])
			# check whether it is a more than two class problem
			if scores.ndim > 1:
				nr_class = scores.shape[1]
			else:
				scores = np.reshape(scores, (-1,1)) 
				nr_class = 2	

			# select all pairwise combinations of classes
			pairs = list(itertools.combinations(range(nr_class), 2))
			if len(pairs) > 1: # do this both ways in case of multi class problem
				pairs += [p[::-1] for p in pairs]

			# initiate AUC
			auc = np.zeros(len(pairs))	

			# loop over all pairwise combinations
			for i, comp in enumerate(pairs):
				pair_idx = np.logical_or(true_labels == comp[0], true_labels == comp[1]) 	# grab two classes
				bool_labels = np.zeros(true_labels.size, dtype = bool) 	# set all labels to false
				bool_labels[true_labels == comp[0]] = True 				# set only positive class to True
				labels_2_use = bool_labels[pair_idx]					# select pairwise labels
				scores_2_use = scores[pair_idx,comp[0]]					# select pairwisescores
				auc[i] = self.scoreAUC(labels_2_use, scores_2_use)		# compute AUC

			class_perf = np.mean(auc)

		elif self.method == 'acc':
			#predict = np.argmin(scores, axis =1)
			class_perf = np.sum(predict == true_labels)/float(true_labels.size)
				
		return class_perf
		

	def scoreAUC(self, labels, scores):
		'''
		Calculates the AUC - area under the curve.
		Besides being the area under the ROC curve, AUC has a slightly less known interpretation:
		If you choose a random pair of samples which is one positive and one negative - AUC is the probabilty 
		that the positive-sample score is above the negative-sample score.
		
		Here we compute the AUC by counting these pairs.
		function modified after the ADAM toolbox and http://www.springerlink.com/content/nn141j42838n7u21/fulltext.pdf
		Arguments
		- - - - - 
		labels (array): Boolen labels of size N
		scores (array): scores of size N
		Returns
		- - - -
		auc (float): area under the curve
		'''

		num_pos = np.sum(labels)
		num_neg = labels.size - num_pos

		assert num_pos != 0,'no positive labels entered in AUC calculation'
		assert num_neg != 0,'no negative labels entered in AUC calculation'

		ranks = rankdata(scores) 
		auc = (np.sum(ranks[labels]) - num_pos * (num_pos + 1)/2)/ (num_pos * num_neg)

		return auc


	def selectMaxTrials(self,beh, cnds, bdm_labels = 'all', cnds_header = 'condition'):
		''' 
		
		For each condition the maximum number of trials per decoding label are determined
		such that data can be split up in equally sized subsets. This ensures that across 
		conditions each unique decoding label is selected equally often
		Arguments
		- - - - - 
		beh (dict): contains all logged variables of interest
		cnds (list): list of conditions for decoding analysis
		bdm_labels(list|str): which labels will be used for decoding
		cnds_header (str): variable name containing conditions of interest
		Returns
		- - - -
		max_trials (int): max number unique labels
		'''

		# make sure selection is based on corrrect trials
		if bdm_labels == 'all':
			bdm_labels = np.unique(beh[self.to_decode]) 

		N = self.nr_folds
		cnd_min = []

		# trials for decoding
		if cnds != 'all':
			for cnd in cnds:
		
				# select condition trials and get their decoding labels
				trials = np.where(beh[cnds_header] == cnd)[0]
				labels = [l for l in beh[self.to_decode][trials] if l in bdm_labels]

				# select the minimum number of trials per label for BDM procedure
				# NOW NR OF TRIALS PER CODE IS BALANCED (ADD OPTION FOR UNBALANCING)
				min_tr = np.unique(labels, return_counts = True)[1]
				min_tr = int(np.floor(min(min_tr)/N)*N)	

				cnd_min.append(min_tr)

			max_trials = min(cnd_min)
		elif cnds == 'all':
			labels = [l for l in beh[self.to_decode] if l in bdm_labels]
			min_tr = np.unique(labels, return_counts = True)[1]
			max_trials = int(np.floor(min(min_tr)/N)*N)	

		if max_trials == 0:
			print('At least one condition does not contain sufficient info for current nr of folds')

		return max_trials

	def trainTestSelect(self, tr_labels, eegs, train_tr, test_tr, te_labels = None):
		'''
		Arguments
		- - - - - 
		tr_labels (array): decoding labels used for training
		eegs (array): eeg data (epochs X electrodes X timepoints)
		train_tr (array): indices of train trials (nr of folds X nr unique train labels X nr train trials)
		test_tr (array): indices of test trials (nr of folds X nr unique test labels X nr test trials)
		te_labels (array): only specify if train and test labels differ (e.g. in cross decoding analysis)
		Returns
		- - - -
		Xtr (array): data that serves as training input (nr folds X epochs X elecs X timepoints) 
		Xte (array): data that serves to evaluate model
		Ytr (array): training labels. Training label for each epoch in Xtr
		Yte (array): test labels. Test label for each epoch in Xte
		'''

		# check test labels
		if te_labels is None:
			te_labels = tr_labels

		# initiate train and test label arrays
		Ytr = np.zeros(train_tr.shape, dtype = tr_labels.dtype).reshape(self.nr_folds, -1)
		Yte = np.zeros(test_tr.shape, dtype = tr_labels.dtype).reshape(self.nr_folds, -1)

		# initiate train and test data arrays
		Xtr = np.zeros((self.nr_folds, np.product(train_tr.shape[-2:]), eegs.shape[1],eegs.shape[2]))
		Xte = np.zeros((self.nr_folds, np.product(test_tr.shape[-2:]), eegs.shape[1],eegs.shape[2]))

		# select train data and train labels
		for n in range(train_tr.shape[0]):
			Xtr[n] = eegs[np.hstack(train_tr[n])]
			Xte[n] = eegs[np.hstack(test_tr[n])]
			Ytr[n] = tr_labels[np.hstack(train_tr[n])]
			Yte[n] = te_labels[np.hstack(test_tr[n])]

		return Xtr, Xte, Ytr, Yte	


	def trainTestSplit(self, idx, labels, max_tr, bdm_info):
		''' 
		Splits up data into training and test sets. The number of training and test sets is 
		equal to the number of folds. Splitting is done such that all data is tested exactly once.
		Number of folds determines the ratio between training and test trials. With 10 folds, 90%
		of the data is used for training and 10% for testing. 
		
		Arguments
		- - - - - 
		idx (array): trial indices of decoding labels
		labels (array): decoding labels
		max_tr (int): max number unique labels
		bdm_info (dict): dictionary with selected trials per label. If {}, a random subset of trials
		will be selected
		Returns
		- - - -
		train_tr (array): trial indices per fold and unique label (folds X labels X trials)
		test_tr (array): trial indices per fold and unique label (folds X labels X trials)
		'''

		N = self.nr_folds
		nr_labels = np.unique(labels).size
		steps = int(max_tr/N)

		# select final sample for BDM and store those trials in dict so that they can be saved
		if bdm_info == {}:
			for i, l in enumerate(np.unique(labels)):
				bdm_info.update({l:idx[random.sample(list(np.where(labels==l)[0]),max_tr)]})	

		# initiate train and test arrays	
		train_tr = np.zeros((N,nr_labels, steps*(N-1)),dtype = int)
		test_tr = np.zeros((N,nr_labels,steps),dtype = int)

		# split up the dataset into N equally sized subsets for cross validation 
		for i, b in enumerate(np.arange(0,max_tr,steps)):
			
			idx_train = np.ones(max_tr,dtype = bool)
			idx_test = np.zeros(max_tr, dtype = bool)

			idx_train[b:b + steps] = False
			idx_test[b:b + steps] = True

			for j, key in enumerate(bdm_info.keys()):
				train_tr[i,j,:] = np.sort(bdm_info[key][idx_train])
				test_tr[i,j,:] = np.sort(bdm_info[key][idx_test])

		return train_tr, test_tr, bdm_info	


	def linearClassification(self, X, train_tr, test_tr, max_tr, labels, gat_matrix = False):
		''' 
		Arguments
		- - - - - 
		X (array): eeg data (trials X electrodes X time)
		train_tr (array): trial indices per fold and unique label (folds X labels X trials)
		test_tr (array): trial indices per fold and unique label (folds X labels X trials)
		max_tr (int): max number unique labels
		labels (array): decoding labels 
		bdm_matrix (bool): If True, return an train X test time decoding matrix. Otherwise only
							return the diagoanl of the matrix (standard decoding)
		Returns
		- - - -
		class_acc
		'''

		N = self.nr_folds
		nr_labels = np.unique(labels).size
		steps = int(max_tr/N)

		nr_elec, nr_time = X.shape[1], X.shape[2]
		if gat_matrix:
			nr_test_time = nr_time
		else:
			nr_test_time = 1	

		lda = LinearDiscriminantAnalysis()

		# set training and test labels
		Ytr = np.hstack([[i] * (steps*(N-1)) for i in np.unique(labels)])
		Yte = np.hstack([[i] * (steps) for i in np.unique(labels)])

		class_acc = np.zeros((N,nr_time, nr_test_time))
		label_info = np.zeros((N, nr_time, nr_test_time, nr_labels))

		for n in range(N):
			print('\r Fold {} out of {} folds'.format(n + 1,N),)
			
			for tr_t in range(nr_time):
				for te_t in range(nr_test_time):
					if not gat_matrix:
						te_t = tr_t

					Xtr = np.array([X[train_tr[n,l,:],:,tr_t] for l in range(nr_labels)]).reshape(-1,nr_elec) 
					Xte = np.vstack([X[test_tr[n,l,:],:,te_t].reshape(-1,nr_elec) for l, lbl in enumerate(np.unique(labels))])

					lda.fit(Xtr,Ytr)
					predict = lda.predict(Xte)
					
					if not gat_matrix:
						class_acc[n,tr_t, :] = sum(predict == Yte)/float(Yte.size)
						label_info[n, tr_t, :] = [sum(predict == l) for l in np.unique(labels)]	
					else:
						class_acc[n,tr_t, te_t] = sum(predict == Yte)/float(Yte.size)
						label_info[n, tr_t, te_t] = [sum(predict == l) for l in np.unique(labels)]	
						#class_acc[n,t] = clf.fit(X = Xtr, y = Ytr).score(Xte,Yte)

		class_acc = np.squeeze(np.mean(class_acc, axis = 0))
		label_info = np.squeeze(np.mean(label_info, axis = 0))

		return class_acc, label_info

	def mneClassify(self, sj, to_decode, conditions, time = [-0.3, 0.8]):
		'''
		'''

		clf = make_pipeline(StandardScaler(), LogisticRegression())
		time_decod = SlidingEstimator(clf, n_jobs=1)	

		# get eeg data
		eeg = []
		for session in range(2):
			eeg.append(mne.read_epochs('/Users/dirk/Desktop/suppression/processed/subject-{}_ses-{}-epo.fif'.format(sj,session + 1)))


		times = eeg[0].times
		# select time window and electrodes	
		s_idx, e_idx = eeg[0].time_as_index(time)	
		picks = mne.pick_types(eeg[0].info, eeg=True, exclude='bads')

		eeg = np.vstack((eeg[0]._data,eeg[1]._data))[:,picks,:][:,:,s_idx:e_idx]


		# get behavior data
		with open('/Users/dirk/Desktop/suppression/beh/processed/subject-{}_all.pickle'.format(sj),'rb') as handle:
			beh = pickle.load(handle)


		plt.figure(figsize = (20,20))
		for i,cnd in enumerate(conditions):
		
			X = eeg[beh['condition'] == cnd]	
			y = beh[to_decode][beh['condition'] == cnd]

			scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)
			plt.plot(times[s_idx:e_idx],scores.mean(axis = 0), color = ['r','g','b','y'][i], label = cnd)

		plt.legend(loc = 'best')
		plt.savefig('/Users/dirk/Desktop/suppression/bdm/figs/{}_{}_bdm.pdf'.format(to_decode,sj))	
		plt.close()	

if __name__ == '__main__':

	project_folder = '/home/dvmoors1/big_brother/Dist_suppression'
	os.chdir(project_folder) 

	subject_id = [1,2,5,6,7,8,10,12,13,14,15,18,19,21,22,23,24]	
	subject_id = [16]		
	to_decode = 'target_loc'
	if to_decode == 'target_loc':
		conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
	else:
		conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']

	session = BDM('all_channels', to_decode, nr_folds = 10)

	for sj in subject_id:
		print(sj)
		session.Classify(sj, conditions = conditions, bdm_matrix = True)
		#session.Classify(sj, conditions = conditions, nr_perm = 500, bdm_matrix = True)