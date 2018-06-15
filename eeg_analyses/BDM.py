import os
import mne
import pickle
import random
#import matplotlib
#matplotlib.use('agg') # now it works via ssh connection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helperFunctions import *
from FolderStructure import FolderStructure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from IPython import embed


class BDM(FolderStructure):


	def __init__(self, channel_folder, decoding, nr_folds, eye ):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		self.channel_folder = channel_folder
		self.decoding = decoding
		self.nr_folds = nr_folds
		self.eye = eye

	def selectBDMData(self, sj, time = (-0.3, 0.8), thresh_bin = 1, downsample = 128):
		''' 

		Arguments
		- - - - - 

		sj(int): subject number
		time (tuple | list): time samples (start to end) for decoding
		thresh_bin (int): exclude trials with a deviation larger than 1. If 0 all trials are use for decoding analysis
		downsample (int): downsample the data to this sampling range (save computational time)


		Returns
		- - - -

		eegs (array): eeg data (trials X electrodes X time)
		beh (dict): contains variables of interest for decoding

		'''

		# read in processed behavior from pickle file
		with open(self.FolderTracker(extension = ['beh','processed'], filename = 'subject-{}_all.pickle'.format(sj)),'rb') as handle:
			beh = pickle.load(handle)

		# read in eeg data 
		EEG = mne.read_epochs(self.FolderTracker(extension = ['processed'], filename = 'subject-{}_all-epo.fif'.format(sj)))
		if downsample != int(EEG.info['sfreq']):
			print 'downsampling data'
			EEG.resample(downsample)

		# select time window and EEG electrodes
		s, e = [np.argmin(abs(EEG.times - t)) for t in time]
		picks = mne.pick_types(EEG.info, eeg=True, exclude='bads')
		eegs = EEG._data[:,picks,s:e]
		times = EEG.times[s:e]

		# exclude trials contaminated by unstable eye position
		nan_idx = np.where(np.isnan(beh['eye_bins']) > 0)[0]
		heog = EEG._data[:,EEG.ch_names.index('HEOG'),s:e]

		eye_trials = eog_filt(beh, EEG, heog, sfreq = EEG.info['sfreq'], windowsize = 50, windowstep = 25, threshold = 30)
		beh['eye_bins'][eye_trials] = 99
	
		# use mask to select conditions and position bins (flip array for nans)
		eye_mask = ~(beh['eye_bins'] > thresh_bin)	
		if self.eye:
			eegs = eegs[eye_mask,:,:]

			for key in beh.keys():
				if key not in ['clean_idx']:
					beh[key] = beh[key][eye_mask]

		# store dictionary with variables for plotting
		plot_dict = {'ch_names': EEG.ch_names, 'times':times, 'info':EEG.info}

		with open(self.FolderTracker(['bdm',self.decoding], filename = 'plot_dict.pickle'),'wb') as handle:
			pickle.dump(plot_dict, handle)						

		return 	eegs, beh


	def Classify(self,sj, cnds, cnd_header, subset = None, time = (-0.3, 0.8), nr_perm = 0, bdm_matrix = False):
		''' 

		Arguments
		- - - - - 

		sj(int): subject number
		cnds (list): list of condition labels (as stored in beh dict). 
		cnd_header (str): variable name containing conditions of interest
		subset (list | None): Specifies whether all labels or only a subset of labels should be decoded
		time (tuple | list): time samples (start to end) for decoding
		nr_perm (int): If perm = 0, run standard decoding analysis. 
					If perm > 0, the decoding is performed on permuted labels. 
					The number sets the number of permutations
		bdm_matrix (bool): If True, train X test decoding analysis is performed

		Returns
		- - - -
		'''	

		# first round of classification is always done on non-permuted labels
		nr_perm += 1

		# read in data 
		eegs, beh = self.selectBDMData(sj, time = time)

		# select minumum number of trials given the specified conditions
		max_tr = self.selectMaxTrials(beh, cnds, cnd_header)

		# create dictionary to save classification accuracy
		classification = {}

		if cnds == 'all':
			cnds = [cnds]

		# loop over conditions
		for cnd in cnds:

			# reset selected trials
			bdm_info = {}

			# initiate decoding array
			if bdm_matrix:
				class_acc = np.empty((nr_perm, eegs.shape[2], eegs.shape[2])) * np.nan
			else:	
				class_acc = np.empty((nr_perm, eegs.shape[2])) * np.nan	

			if cnd != 'all':
				cnd_idx = np.where(beh[cnd_header] == cnd)[0]
				cnd_labels = beh[self.decoding][cnd_idx]
			else:
				cnd_idx = np.arange(beh[cnd_header].size)
				cnd_labels = beh[self.decoding]	

			if subset != None:
				sub_idx =  [i for i,l in enumerate(cnd_labels) if l in subset]	
				cnd_idx = cnd_idx[sub_idx]
				cnd_labels = cnd_labels[sub_idx]

			# permutation loop (if perm is 1, train labels are not shuffled)
			for p in range(nr_perm):

				if p > 0: # shuffle condition labels
					np.random.shuffle(cnd_labels)
			
				# select train and test trials 	
				train_tr, test_tr, bdm_info = self.bdmTrialSelection(cnd_idx, cnd_labels, max_tr, bdm_info)

				# do actual classification
				class_acc[p,:] = self.linearClassification(eegs, train_tr, test_tr, max_tr, cnd_labels, bdm_matrix)

			classification.update({cnd:{'standard': class_acc[0]}, 'info': bdm_info})
			if nr_perm > 1:
				classification[cnd].update({'perm': class_acc[1:]})

		# store classification dict	
		with open(self.FolderTracker(['bdm',self.decoding], filename = 'class_{}_perm-{}.pickle'.format(sj,bool(nr_perm -1))) ,'wb') as handle:
			pickle.dump(classification, handle)

	def selectMaxTrials(self,beh, cnds, cnds_header = 'condition'):
		''' 
		
		For each condition the maximum number of trials per decoding label are determined
		such that data can be split up in equally sized subsets. This ensures that across 
		conditions each unique decoding label is selected equally often

		Arguments
		- - - - - 
		beh (dict): contains all logged variables of interest
		cnds (list): list of conditions for decoding analysis
		cnds_header (str): variable name containing conditions of interest

		Returns
		- - - -
		max_trials (int): max number unique labels
		'''

		N = self.nr_folds

		cnd_min = []
		# trials for decoding
		if cnds != 'all':
			for cnd in cnds:
		
				# select condition trials and get their decoding labels
				trials = np.where(beh[cnds_header] == cnd)[0]
				labels = beh[self.decoding][trials]

				# select the minimum number of trials per label for BDM procedure
				# NOW NR OF TRIALS PER CODE IS BALANCED (ADD OPTION FOR UNBALANCING)
				min_tr = np.unique(labels, return_counts = True)[1]
				min_tr = int(np.floor(min(min_tr)/N)*N)	

				cnd_min.append(min_tr)

			max_trials = min(cnd_min)
		elif cnds == 'all':
			labels = beh[self.decoding]
			min_tr = np.unique(labels, return_counts = True)[1]
			max_trials = int(np.floor(min(min_tr)/N)*N)	

		return max_trials

	def bdmTrialSelection(self, idx, labels, max_tr, bdm_info):
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

	def linearClassification(self, X, train_tr, test_tr, max_tr, labels, bdm_matrix = False):
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
		if bdm_matrix:
			nr_test_time = nr_time
		else:
			nr_test_time = 1	

		lda = LinearDiscriminantAnalysis()

		# set training and test labels
		Ytr = np.hstack([[i] * (steps*(N-1)) for i in np.unique(labels)])
		Yte = np.hstack([[i] * (steps) for i in np.unique(labels)])

		class_acc = np.zeros((N,nr_time, nr_test_time))

		for n in range(N):
			print ('Fold {} out of {} folds'.format(n + 1,N))
			
			for tr_t in range(nr_time):
				for te_t in range(nr_test_time):
					if not bdm_matrix:
						te_t = tr_t

					Xtr = np.array([X[train_tr[n,l,:],:,tr_t] for l in range(nr_labels)]).reshape(-1,nr_elec) 
					Xte = np.vstack([X[test_tr[n,l,:],:,te_t].reshape(-1,nr_elec) for l, lbl in enumerate(np.unique(labels))])

					lda.fit(Xtr,Ytr)
					predict = lda.predict(Xte)

					if not bdm_matrix:
						class_acc[n,tr_t, :] = sum(predict == Yte)/float(Yte.size)
					else:
						class_acc[n,tr_t, te_t] = sum(predict == Yte)/float(Yte.size)	
						#class_acc[n,t] = clf.fit(X = Xtr, y = Ytr).score(Xte,Yte)

		class_acc = np.squeeze(np.mean(class_acc, axis = 0))

		return class_acc



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
		print sj
		session.Classify(sj, conditions = conditions, bdm_matrix = True)
		#session.Classify(sj, conditions = conditions, nr_perm = 500, bdm_matrix = True)
		









	







