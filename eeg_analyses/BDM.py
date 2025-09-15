import os
from xml.dom import NotFoundErr
import mne
import pickle
import random
import copy
import itertools
import warnings

import numpy as np
from numpy.fft import ifft2
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Generic, Union, Tuple, Any
from support.FolderStructure import *
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
 
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from support.support import select_electrodes, trial_exclusion, get_time_slice
from scipy.stats import rankdata
from scipy.signal import hilbert
from eeg_analyses.TFR import TFR

from IPython import embed

warnings.simplefilter('default')

class BDM(FolderStructure):

	"""
	The BDM object supports multivariate decoding functionality to 
	predict an experimental variable (class) given an observed 
	pattern of brain activity.
	By default the BDM class employs Linear Discriminant Analysis (LDA) 
	to perform decoding.

	The BDM class makes use of k-fold cross validation, in which the 
	trials are split up into k equally sized folds, such that on any 
	given iteration the training daa are independent from the test data.
	In an iterative procedure, the model is trained on k-1 folds, 
	and testing is done on the remaining fold that was not used for 
	training, until each trial served as a test set once. It is ensured 
	that each class has the same number of observations 
	(i.e., balanced classes) in both the training and the testing set. 

	In class assignment, BDM applies event balancing by default through 
	undersampling so that each class has the same number of 
	observations(i.e., if one class has 200 observations, and the other 
	class has 100 observations, the number of observations in the first 
	class is artificially lowered by randomly removing 100 
	observations). Consequently, if the design is heavily unbalanced 
	between classes you may loose a lot of data. 

	Area under the curve (AUC; Bradley, 1997), a metric derived from 
	signal detection theory, is the default performance measure that BDM 
	computes.

	Args:
		FolderStructure (object): Class that creates file paths to 
		load raw eeg/ behavior and save decoding ouput
	"""

	def __init__(self,sj:int,epochs:Union[mne.Epochs,list],
				df:Union[pd.DataFrame,list],to_decode:str, 
				nr_folds:int=10,classifier:str='LDA',data_type:str='raw',
				tfr:Optional[TFR]=None,metric:str='auc',
				elec_oi:Union[str,list]='all',downsample:int=128,
				avg_runs:int=1,avg_trials:int=1,
				sliding_window:tuple=(1,True,False),
				scale:dict={'standardize':False,'scale':False}, 
				pca_components:tuple=(0,'across'),montage:str='biosemi64',
				output_params:Optional[bool]=False,
				baseline:Optional[tuple]=None, 
				seed:Union[int, bool] = 42213):
		"""set decoding parameters that will be used in BDM class

		Args:
			sj (int): subject identifier. Used to save output files.
			beh (pd.DataFrame | list): Dataframe with behavioral 
			parameters. Each row represents an epoch in epochs object.
			epochs (mne.Epochs | list): Epoched eeg data (linked to 
			beh). Can be a list of two mne.Epochs objects, in which case
			the first will serve as the training set and the second as 
			the test set in a localizer based decoding regime (see
			localizer_classif())
			to_decode (str): column name in beh that contains classes
			information. By default all classes are used for 
			classification, but a subset can be specified when calling
			decoding functions (i.e., classify or localizer_classify)
			nr_folds (int): Number of folds that are used for 
			k-fold cross validation
			classifier (str, optional): Sets which classifier is used 
			for decoding. Supports 'LDA' (linear discriminant analysis),
			'svm' (support vector machine), 'GNB' (Gaussian Naive Bayes)
			data_type (str, optional): Specifies whether decoding should 
			be performed on raw data ('raw') or on power after 
			time-frequency decomposition ('tfr'). In the latter case, 
			decoding can be done on a single band (e.g., alpha) or a 
			range of frequency bands. By default decoding is performed
			on the alpha band, unless specified otherwise. See tf_bands. 
			tf_bands (dict | list): [description]
			metric (str, optional): Metric used to quantify decoding 
			performance. Defaults to 'auc'.
			elec_oi (Optional[str, list], optional): Electrodes used as
			features during decoding. Can be a string of a predefined 
			subset of electrodes (e.g., post, see documentation) or a 
			list of electrode names. Defaults to 'all'.
			downsample (int, optional): Sampling frequency used
			during decoding. Can be used to save computation time.
			Defaults to 128.
			avg_runs (int, optional): Specifies how often (random) 
			cross-validation procedure is performed. Decoding output 
			reflects the average of all cross-validation runs.
			avg_trials (int, Optional): Specifies the number of trials 
			that are averaged together before cross-validation. 
			Averaging is done across each unique combination 
			of condition and decoding label. Trial averaging is always 
			done before subsequent optional data transformation steps
			(except tfr decomposition).Defaults to  1 
			(i.e., no trial averaging). 
			sliding_window (tuple, optional): Increases the  number of 
			features used for decoding by a factor of the size of the 
			sliding_window by giving the classifier access to all time 
			points in the window (see Grootswagers et al. 2017, JoCN). 
			Second argument in tuple specifies whether (True) or not 
			(False) the activity in each sliding window is demeaned 
			(see Hajonides et al. 2021, NeuroImage). If the third \
			argument is set to True rather than increasing the number 
			of features, each  time point reflects the average within 
			the sliding window. Defaults to (1,True,False) meaning that 
			no data transformation will be applied.
			scale (dict): Dictinary with two keys specifying whether 
			data should be standardized (True) or not (False). The scale 
			argument specifies whether or not data should also be scaled 
			to unit variance (or equivalently, unit standard deviation). 
			This step is always performed before PCA. Defaults to 
			{'standardize': False, 'scale': False}, no standardization
			pca_components (tuple, optional): Apply dimensionality 
			reduction before decoding. The first arguments specifies how 
			features should reduce to N principal components,
            if N < 1 it indicates the % of explained variance 
			(and the number of components is inferred). The second 
			argument specifies whether transformation is estimated
			on both training and test data ('all') or estimated on 
			training data only and applied to the test data in each 
			cross validation step. Defaults to (0, 'across') (i.e., no 
			PCA reduction)
			montage (Optional[str]): Montage used during recording. 
			Is used to plot weigts in the bdm report.
			bdm_filter (Optional[dict], optional): [description]. 
			Defaults to None.
			baseline (Optional[tuple], optional): [description]. 
			Defaults to None.
			seed (Optional[int]): Sets a random seed such that 
			cross-validation procedure can be repeated. In case of False 
			, no seed is applied before cross validation. In case 
			avg_runs > 1, seed will be increased by 1 for each run.
			Defaults to 42213 (A1Z26 cipher of DvM)
		"""	

		self.sj = sj
		self.epochs = epochs					
		self.df = df
		self.classifier = classifier
		self.baseline = baseline
		self.to_decode = to_decode
		self.data_type = data_type
		self.tfr = tfr
		self.nr_folds = nr_folds
		self.elec_oi = elec_oi
		self.downsample = downsample
		self.window_size = sliding_window
		self.scale = scale
		self.montage = montage
		self.pca_components = pca_components
		self.metric = metric
		self.avg_runs = avg_runs
		self.avg_trials = avg_trials
		self.baseline = baseline
		self.seed = seed
		self.output_params = output_params
		self.cross = False

	def select_classifier(self) -> Any:
		"""
		Function that initialises the classifier

		Raises:
			ValueError: In case incorrect classifier is selected

		Returns:
			clf: sklearn classifier class
		"""
		
		if self.classifier == 'LDA':
			clf = LinearDiscriminantAnalysis()
		elif self.classifier == 'GNB':
			clf = GaussianNB()
		elif self.classifier == 'svm':
			clf = CalibratedClassifierCV(LinearSVC())
		else:
			raise ValueError('Classifier not correctly defined.')
		
		return clf

	def get_classifier_weights(self, clf, Xtr_):
		"""Get classifier weights regardless of classifier type."""
		if hasattr(clf, 'coef_'):
			# For LDA, LogisticRegression
			weights = clf.coef_[0]
		elif hasattr(clf, 'calibrated_classifiers_'):
			# For CalibratedClassifierCV (SVM)
			base_clf = clf.calibrated_classifiers_[0].base_estimator
			weights = base_clf.coef_[0]
		else:
			weights = np.zeros(Xtr_.shape[1])
			warnings.warn('Classifier weights not available for this \n' +
			'classifier type')
		return weights

	def classify(self,cnds:dict=None,window_oi:tuple=None,
				labels_oi:Union[str,list]='all',collapse:bool=False,
				excl_factor:dict=None,nr_perm:int=0,GAT:bool=False, 
				downscale:bool=False,split_fact:dict=None,
				save:bool=True,bdm_name:str='main')->dict:
		"""
		Multivariate decoding across time of the classes specified upon 
		class initialization. Decoding can either be condition specific,
		or a cross decoding analysis, where the model is trained on 
		classes within one condition to decode another condition 
		(see cnds argument on how to initialize different kinds of
		decoding analyses)

		Args:
			cnds (dict, optional): Condition information used in 
			decoding analysis. For condition specific decoding specify 
			a dictionary, where the key is the column that contains 
			condition labels in beh. Values should be a list of all 
			conditions to be included. Defaults to None --> decoding is
			performed on all data.
			
			Example:
			cnds = dict(condition = ['present','absent'])). 

			For a cross condition decoding analysis, training 
			conditions are specified within a list at the first position
			of the condition list. 

			Example:
			cnds = dict(condition = [['present'],'absent']))

			In this case the model is trained using data from the 
			present condition to predict data within the absent 
			condition. Multiple testing conditions can also be 
			specified. To use multiple training and testing conditions 
			use the following syntax:

			cnds = dict(condition = [['train_1','train_2'],
										['test_1','test_2']))

			window_oi (tuple, optional): time window of interest. 
			Defaults to None --> use all samples in epochs
			labels_oi (Union[str,list], optional): can be used to limit 
			decoding to a subset of classes. 
			Defaults to 'all' (i.e., use all classes for decoding).
			collapse (bool, optional): In addition to condition 
			specific decoding, also perform decoding collapsed
			across all conditions. Defaults to False.
			excl_factor (dict, optional): This gives the option to 
			exclude specific conditions from analysis. 
			
			For example, to only include trials where the cue was 
			pointed to the left and not to the right 
			specify the following: 

			excl_factor = dict(cue_direc = ['right']). 
			
			Mutiple column headers and multiple variables per header can 
			be specified. Defaults to None (i.e., no trial exclusion).
			nr_perm (int, optional): Can be used to obtain a data driven
			chance baseline. If perm > 0, decoding is additionaly 
			performed on permuted labels, where the number sets the 
			number of permutations. Defaults to 0.
			GAT (bool, optional): perform decosing across
			all combinations of timepoints (i.e., generate a 
			generalization across time decoding matrix). 
			Defaults to False.
			downscale (bool, optional): Allows decoding to be repeatedly 
			run with increasingly less trials. Can be used to examine 
			the minimum number of trials that support reliable 
			classification. Defaults to False.
			save (bool, optional): save decoding output to disk.
			Defaults to True.
			bdm_name (str, optional): name of decoding analysis used 
			during saving. Defaults to 'main'.

		Returns:
			bdm_scores: decoding output
		"""

		# select condition specific data
		if cnds is None:
			(cnd_head,cnds) = (None,['all_data'])
		else:
			(cnd_head,cnds), = cnds.items()

		# select the data of interest
		if labels_oi != 'all':
			all_labels = np.unique(self.df[self.to_decode])
			to_exclude = [v for v in all_labels if v not in labels_oi]
			if len(to_exclude) > 0:
				if excl_factor is None:
					excl_factor = {self.to_decode:to_exclude}
				else:
					excl_factor[self.to_decode] = to_exclude

		headers = [cnd_head]
		if split_fact is not None:
			headers += list(split_fact.keys())

		(X,
		y,
		df, 
		times) = self.select_bdm_data(self.epochs.copy(), self.df.copy(),
									window_oi, excl_factor, headers)

		(nr_labels,
		tr_max, 
		tr_cnds, 
		te_cnds) = self.set_bdm_param(y,df,cnds,cnd_head,labels_oi,downscale)

		if collapse:
			df['collapsed'] = 'no'
			cnds += ['collapsed']

		# set bdm_name
		bdm_name = f'sj_{self.sj}_{bdm_name}'				
		
		# set up dict to save decoding scores
		bdm_params = {}
		if split_fact is None:
			(bdm_scores, 
			bdm_params, 
			bdm_info) = self.classify_(X,y,df,tr_cnds,te_cnds,cnd_head,tr_max,
										labels_oi, collapse,GAT,nr_perm)
		else:
			(bdm_scores, 
			bdm_params, 
			bdm_info) = self.iter_classify_(split_fact,X,y,df,tr_cnds,te_cnds,
								   			cnd_head,tr_max,labels_oi, 
											collapse,GAT,nr_perm)	
					
		bdm_scores.update({'info':{'elec':self.elec_oi,'times':times}})
		if self.data_type == 'tfr':
			bdm_scores['info'].update({'freqs':self.tfr.frex})

		# create report (specific to unpermuted data)
		if not GAT:
			pass
			#self.report_bdm(bdm_scores, cnds, bdm_name)

		# store classification dict	
		if save: 
			ext = self.set_folder_path()
			with open(self.folder_tracker(ext, fname = 
					f'{bdm_name}.pickle') ,'wb') as handle:
				pickle.dump(bdm_scores, handle)
			if self.output_params:
				with open(self.folder_tracker(ext, fname = 
						f'{bdm_name}_params.pickle') ,'wb') as handle:
					pickle.dump(bdm_params, handle)				

		return bdm_scores, bdm_params	
	
	def iter_classify_(self,split_fact:dict,X:np.array,y:np.array,
					beh:pd.DataFrame,tr_cnds:list,te_cnds:list,cnd_head:str,
					tr_max:list,labels_oi:Union[str,list],collapse:bool,
					GAT:bool,nr_perm:int):
		"""
		helper function of classify that does decoding 
		per condition in an iterative procedure such that decoding 
		output reflects the mean of the seperate decoding regimes
		"""

		# split the selected data into subsets and apply decoding 
		# within those subsets
		dec_scores, dec_params = [],[]
		for key, value in split_fact.items():
			for v in value:
				mask = beh[key] == v
				X_ = X[mask]
				y_ = y[mask].reset_index(drop = True)
				beh_ = beh[mask].reset_index(drop = True)
				# reset max trials for folding
				tr_max = [self.selectMaxTrials(beh_,tr_cnds,labels_oi,
								   			cnd_head)]
				
				(bdm_scores, 
				bdm_params, 
				bdm_info) = self.classify_(X_,y_,beh_,tr_cnds,te_cnds,cnd_head,
							   			tr_max,labels_oi, collapse,GAT,nr_perm)
				
				dec_scores.append(bdm_scores)
				dec_params.append(bdm_params)

		# create averaged output dictionary
		#TODO: 
		for key in (k for k in bdm_scores if k != 'bdm_info'):
			output = np.mean([scores[key]['dec_scores'] for scores 
						 							in dec_scores], axis = 0)
			bdm_scores[key]['dec_scores'] = output
			#filtyhy hack for now
			if dec_params[0] != {}:
				W = np.mean([params[key]['W'] for params in dec_params], axis = 0)
				bdm_params[key]['W'] = W
			else:
				bdm_params = {key: {}}
			
		return bdm_scores, bdm_params, bdm_info

	def classify_(self,X:np.array,y:np.array,beh:pd.DataFrame,tr_cnds:list,
			   		te_cnds:list,cnd_head:str,tr_max:list,
					labels_oi:Union[str,list],collapse:bool,GAT:bool,
					nr_perm:int):
		"""
		helper function of classify that does actual decoding 
		per condition
		"""

		bdm_scores, bdm_params = {}, {}

		# set decoding parameters
		if X.ndim == 3:
			nr_epochs, nr_elec, nr_time = X.shape
			nr_freq = 1
		elif X.ndim == 4:
			nr_freq, nr_epochs, nr_elec, nr_time = X.shape

		nr_perm += 1

		# loop over (training and testing) conditions
		for tr_cnd in tr_cnds:
			for te_cnd in (te_cnds if te_cnds is not None else [None]):

				# reset selected trials
				bdm_info = {}

				# get condition indices and labels
				(beh, cnd_idx, 
				cnd_labels, labels,
				tr_max) = self.get_condition_labels(beh, cnd_head,tr_cnd,
													tr_max, labels_oi,collapse)

				# initiate decoding arrays for current condition
				if GAT:
					class_acc = np.empty((self.avg_runs, nr_perm,nr_freq,
									nr_time, nr_time)) * np.nan
					weights = np.empty((self.avg_runs, nr_perm, nr_freq,
										nr_time, nr_time, nr_elec))
					conf_matrix = np.empty((self.avg_runs, nr_perm,nr_freq,
										nr_time, nr_time, labels.size,labels.size))									
				else:	
					class_acc = np.empty((self.avg_runs,nr_perm,nr_freq,
									nr_time,1)) * np.nan	
					weights = np.empty((self.avg_runs, nr_perm,nr_freq,
										nr_time, 1,nr_elec))
					conf_matrix = np.empty((self.avg_runs, nr_perm,nr_freq,
										nr_time, 1,labels.size,labels.size))	

				# permutation loop (if perm is 1, train labels are not shuffled)
				#TODO: check position of permutation loop
				for p in range(nr_perm):

					if p > 0: # shuffle condition labels
						np.random.shuffle(cnd_labels)
				
					for i, n in enumerate(tr_max):
						if i > 0:
							print(f'Minimum condition label downsampled to {n}')
							bdm_info = {}

						# select train and test trials
						self.run_info = 1
						for run in range(self.avg_runs):
							bdm_info.update({'run_' +str(self.run_info): {}})
							if self.cross:
								# TODO1: make sure that multiple test conditions can be classified
								# TODO2: make sure that bdm_info is saved 
								test_idx = np.where(beh[cnd_head] == te_cnd)[0]
								(Xtr, Xte, 
								Ytr, Yte) = self.train_test_cross(X, y, 
																cnd_idx,test_idx)
								
							else:
								(train_tr, test_tr, 
								bdm_info) = self.train_test_split(cnd_idx, 
														cnd_labels, n, bdm_info) 
								(Xtr, Xte, 
								Ytr, Yte) = self.train_test_select(X, y,
																train_tr,test_tr)
							
							(class_acc[run, p], 
							weights[run, p],
							conf_matrix[run,p]) = self.cross_time_decoding(Xtr,Xte, 
																		Ytr, Yte, 
																		labels, 
																		GAT, X)

							self.seed += 1 # update seed used for cross validation
							self.run_info += 1

						mean_class = class_acc.mean(axis = 0)	
						#TODO check weights!!!!!!			
						W = weights.mean(axis = 0)[0]
						conf_M = conf_matrix.mean(axis = 0)

						if i == 0:
							# get standard dec scores
							cnd_inf = str(tr_cnd)
							if te_cnd is not None:
								cnd_inf += f'_{te_cnd}'
							#W = self.set_bdm_weights(W, Xtr, nr_elec, nr_time)
							dec_scores = copy.copy(np.squeeze(mean_class[0]))
							bdm_scores.update({cnd_inf:{'dec_scores': 
								   			dec_scores}, 
												'bdm_info': bdm_info})
							if self.output_params:
								bdm_params.update({cnd_inf:{'W':W, 
												'conf_matrix':copy.copy(conf_M[0])
												}})

						else:
							bdm_scores[cnd_inf]['{}-nrlabels'.format(n)] = \
											copy.copy(mean_class[0])

				if nr_perm > 1:
					bdm_scores[cnd_inf].update({'perm_scores': mean_class[1:]})

		return bdm_scores, bdm_params, bdm_info
	
	def localizer_classify_(self,X_tr:np.array,y_tr:np.array,X_te:np.array,
						 	y_te:np.array,labels_oi_tr:list,labels_oi_te:list,
							cnd_header_te:str,cnds_te:list,cnd_idx_tr:list,
							beh_te:pd.DataFrame,max_tr:list,GAT:bool,
							nr_perm:int):
		"""
		helper function of localizer_classify that does actual decoding
		per condition
		"""
		
		# set decoding parameters
		nr_epochs_tr,nr_elec,nr_time_tr = X_tr.shape
		nr_epochs_te,nr_elec,nr_time_te = X_te.shape

		# set up dict to save decoding scores
		bdm_scores = {'info': {'elec':self.elec_oi,
					 'times_oi':(nr_time_tr,nr_time_te)}}
		bdm_params = {}

		_, label_counts = np.unique(y_te, return_counts = True)
		nr_tests = min(label_counts) * np.unique(y_te).size

		nr_elec = X_tr.shape[1]
		
		# first round of classification is always done on non-permuted labels
		nr_perm += 1

		for cnd in cnds_te:

			# get condition indices and labels
			(beh_te, cnd_idx_te, 
			cnd_labels, labels,
			max_tr) = self.get_condition_labels(beh_te,cnd_header_te,cnd,max_tr, 
												labels_oi_te)

			# initiate decoding arrays
			if GAT:
				class_acc = np.empty((self.avg_runs, nr_perm,
								nr_time_tr, nr_time_te)) * np.nan
				conf_matrix = np.empty((self.avg_runs, nr_perm, 
								nr_time_tr, nr_time_te, 
								len(labels_oi_te), len(labels_oi_tr))) * np.nan
				weights = np.empty((self.avg_runs, nr_perm,
									nr_time_tr, nr_time_te, 
									nr_elec)) * np.nan
			else:	
				#TODO: check whether time assignment works
				class_acc = np.empty((self.avg_runs,nr_perm,
								nr_time_te)) * np.nan	
				conf_matrix = np.empty((self.avg_runs, nr_perm, 
								nr_time_tr, len(labels_oi_te), 
								len(labels_oi_tr))) * np.nan
				weights = np.empty((self.avg_runs, nr_perm,
									nr_time_te, nr_elec)) * np.nan

			# permutation loop (if perm is 1, train labels are not shuffled)
			for p in range(nr_perm):

				if p > 0: # shuffle condition labels
					np.random.shuffle(cnd_labels)
			
				for i, n in enumerate([1]):
					if i > 0:
						print(f'Minimum condition label downsampled to {n}')
						bdm_info = {}

					# select train and test trials
					self.run_info = 1
					for run in range(self.avg_runs):
						#bdm_info.update({'run_' +str(self.run_info): {}})
						# TODO2: make sure that bdm_info is saved 

						# split independent training and test data
						(Xtr, _, 
						Ytr, _) = self.train_test_cross(X_tr, y_tr, 
														cnd_idx_tr,False)
												
						(Xte, _, 
						Yte, _) = self.train_test_cross(X_te, y_te, 
														cnd_idx_te,False,
														max_tr)

						(class_acc[run, p], 
						weights[run, p],
						conf_matrix[run,p]) = self.cross_time_decoding(Xtr,Xte, 
																Ytr, Yte, 
																(labels_oi_tr,
		 														labels_oi_te),
																GAT, Xtr)
												
						self.seed += 1 # update seed used for cross validation
						self.run_info += 1

					mean_class = class_acc.mean(axis = 0)
					conf_M = conf_matrix.mean(axis = 0)
					W = weights.mean(axis = 0)[0]

					if i == 0:
						# get standard dec scores
						#W = self.set_bdm_weights(W, Xtr, nr_elec, nr_time)
						bdm_scores.update({cnd:{'dec_scores': 
										copy.copy(np.squeeze(mean_class[0]))}})
						if self.output_params:
							bdm_params.update({cnd:{'W':W, 
											'conf_matrix':copy.copy(conf_M[0])
											}})

					else:
						bdm_scores[cnd]['{}-nrlabels'.format(n)] = \
										copy.copy(mean_class[0])

				# bdm_scores.update({cnd:{'dec_scores': 
				# 						copy.copy(np.squeeze(mean_class[0]))},
				# 						'label_inf':copy.copy(label_inf)})

				bdm_scores.update({cnd:{'dec_scores': 
										copy.copy(np.squeeze(mean_class[0]))}})
				
		return bdm_scores, bdm_params

	
	def localizer_classify(self,te_cnds:dict=None,tr_window_oi:tuple=None,
						te_window_oi:tuple=None,tr_excl_factor:dict=None,
						te_excl_factor:dict=None,
						tr_labels_oi:Union[str,list]='all',
						te_labels_oi:Union[str,list]='all',te_header:str=None,
						avg_window:bool=False,GAT:bool=False,nr_perm:int=0,
						save:bool=True, bdm_name:str='loc_dec'):
		
		# set bdm name
		bdm_name = f'sj_{self.sj}_{bdm_name}'

		# set parameters
		self.cross = True
		if self.nr_folds != 1:
			print('Nr folds is reset: ')
			print('Cross decoding is always done on a single fold')
			self.nr_folds = 1
		max_tr = [1]

		# select condition specific data
		if te_cnds is None:
			# TODO: Make sure that trial averaging also works without condition info
			cnds = ['all_data']
			cnd_header = None
		else:
			(cnd_header, cnds), = te_cnds.items()

		if te_header is None:
			te_header = self.to_decode

		# set train data
		if isinstance(self.avg_trials, list):
			self.avg_trials += [self.avg_trials[0]]
		(X_tr, 
   		y_tr, 
		times_tr,
		cnd_idx_tr) = self.get_train_X(tr_window_oi,tr_excl_factor,
				 	tr_labels_oi, avg_window)
		tr_labels_oi = np.unique(y_tr)
		if avg_window:
			GAT = True

		# set testing data
		if isinstance(self.avg_trials, list):
			self.avg_trials += [self.avg_trials[1]]
		print('prepare testing data')
		(X_te, 
		y_te,	
		beh_te, 
		times_te) = self.select_bdm_data(self.epochs[1].copy(),
										self.beh[1].copy(),te_window_oi,
										te_excl_factor,[cnd_header])

		max_tr = self.selectMaxTrials(beh_te,cnds,te_labels_oi,cnd_header)

		# check labels
		if te_labels_oi == 'all':
			te_labels_oi = np.unique(y_te)

		(bdm_scores, 
   		bdm_params) = self.localizer_classify_(X_tr,y_tr,X_te,y_te,
												tr_labels_oi,te_labels_oi,
												cnd_header,cnds,cnd_idx_tr,
												beh_te,max_tr,GAT,nr_perm)

		# store classification dict	
		if save: 
			ext = self.set_folder_path()
			with open(self.folder_tracker(ext, fname = 
					f'{bdm_name}.pickle') ,'wb') as handle:
				pickle.dump(bdm_scores, handle)

			if self.output_params:
				with open(self.folder_tracker(ext, fname = 
						f'{bdm_name}_params.pickle') ,'wb') as handle:
					pickle.dump(bdm_params, handle)	

		return bdm_scores	
	
	def sliding_window_base(self,epochs:mne.Epochs,
						window_size:int=100)->mne.Epochs:
		"""applies a sliding window baseline procedure to the data,
		where the activity at each time point in the window of interest 
		is demeaned by the average activity in the preceding window. 
		Operates in place

		Args:
			epochs (mne.Epochs): epochs object
			window_size (int, optional): size of the widnow in ms.
			  Defaults to 100.
		"""

		# get data
		raw_data = epochs._data.copy()

		# get time points
		window_size /= 1000
		times = epochs.times

		# find index of time point that can be demeaned
		start_idx = np.argmin(abs(times - (times[0] + window_size)))

		# loop over time points
		for i in range(start_idx, raw_data.shape[-1]):
			# get sliding window index
			idx = get_time_slice(times, times[i]-window_size, times[i])
			epochs._data[:,:,i] -= raw_data[:,:,idx].mean(axis = -1)

		return epochs

	def get_train_X(self,window_oi:tuple,excl_factor:dict,
				 	labels_oi:Union[str,list],avg_window:bool)->Tuple[np.array, 
                                                            np.array,np.array,
															np.array]:
		"""selects data that serves as an independent pattern estimator
		in a localizer based decoding analysis

		Args:
			window_oi (tuple): time window of interest
			excl_factor (dict): see docstring of localizer_classify
			labels_oi (Union[str,list]): decoding labels of interest
			avg_window (bool): if True, training data will be averaged 
			within the window of interest

		Returns:
			X (np.array): training data
			y(np.array): training labels
			times (np.array): time samples
			cnd_idx (np.array): indices of interest
		"""

		print('prepare training data')
		# select the data
		(X, 
		_,
		beh, 
		times) = self.select_bdm_data(self.epochs[0].copy(),self.beh[0].copy(),
									window_oi, excl_factor, [])
		
		# limit to labels of interest
		if labels_oi == 'all':
			labels_oi = np.unique(beh[self.to_decode])		
		mask = np.in1d(beh[self.to_decode], labels_oi)
		y = beh[self.to_decode][mask].reset_index(drop = True)
		X = X[mask]
		cnd_idx = np.arange(y.size)

		if avg_window:
			X = X.mean(axis=-1)[..., np.newaxis]
			times = times.mean()

		return X, y, times, cnd_idx	

	def select_bdm_data(self,epochs:mne.Epochs,df:pd.DataFrame,
						window_oi:tuple,excl_factor:dict=None,
						headers:list=None)-> \
						Tuple[np.array, pd.DataFrame, np.array]:
		"""
		Selects bdm data and applies initial data transformation steps 
		in the following order (all optional).

		1. Slicing data by excluding specific trials (set excl_factor)
		2. Data reduction via trial averaging (see class parameters)
		3. Time-frequency decomposition (see class parameters)
		4. Baseline correction (see class parameters)
		5. Downsampling (see class parameters)
		6. Slicing time window (window_oi) and electrodes (see class 
		parameters) of interest
		7. Sliding window based data transformation

		Args:
			epochs (mne.Epochs): epochs objects
			beh (pd.DataFrame): Dataframe with behavioral parameters
			window_oi (tuple): time window of interest
			excl_factor (dict, optional): exclude specific conditions 
			(see classify or localizer_classify)). Defaults to None.
			headers (list, optional): column that contains condition 
			info. Makes sure that trial averaging is condition 
			(and label) specific. Defaults to None.

		Returns:
			X (np.array): data [nr_epochs, nr_elecs, nr_samples]
			beh (pd.DataFrame): behavior parameters 
			times (np.array): sampling times of decoding data
		"""

		# remove a subset of trials 
		if type(excl_factor) == dict: 
			df, epochs,_ = trial_exclusion(df, epochs, excl_factor)

		# reset index(to properly align beh and epochs)
		df.reset_index(inplace = True, drop = True)

		# limit epochs object to electrodes of interest
		picks = mne.pick_types(epochs.info,eeg=True, eog= True, misc = True)
		picks = select_electrodes(np.array(epochs.ch_names)[picks], 
								self.elec_oi)
		epochs = epochs.pick_channels(np.array(epochs.ch_names)[picks])

		# apply filtering and downsampling (if specified)
		if self.data_type == 'tfr':
			print('start tfr decomposition')
			self.tfr.epochs = epochs
			self.tfr.df = df
			X = self.tfr.compute_tfr(epochs)

		# apply baseline correction
		if isinstance(self.baseline, tuple) and self.data_type == 'raw':
			epochs.apply_baseline(baseline = self.baseline)

		# average across trials
		# TODO: make sure trial averaging also works for tfr data
		(epochs, 
		df) = self.average_trials(epochs,df,[self.to_decode] + headers) 
		y = df[self.to_decode]			

		# downsample data
		if self.downsample < int(epochs.info['sfreq']):
			self.down_factor = int(epochs.info['sfreq'])/self.downsample
			if self.window_size[0] == 1:
				print('downsampling data')
				epochs.resample(self.downsample, npad='auto')					
				if self.data_type == 'tfr':
					#print('data is downsampled via subsampling')
					#X = X[:,:,:,::self.down_factor]
					X = mne.filter.resample(X.astype(float), 
							 		down=self.down_factor, 
									npad='auto', pad='edge')
			else:
				print('downsampling will be done after data averaging')
				print('in sliding window approach')

		# select time window and EEG electrodes
		if window_oi is None:
			window_oi = (epochs.tmin, epochs.tmax)
		elif isinstance(window_oi, (int, float)):
			# limit decoding to a single timepoint
			step = np.diff(epochs.times)[0] * self.window_size[0]
			window_oi = (window_oi, window_oi + step)
		idx = get_time_slice(epochs.times, window_oi[0], window_oi[1])
		if self.data_type == 'raw':
			X = epochs._data[:,:,idx]
		else:
			X = X[...,idx]
		times = epochs.times[idx]

		# transform eeg data in case of sliding window approach
		if self.window_size[0] > 1:
			X = self.sliding_window(X, self.window_size[0], 
									self.window_size[1],self.window_size[2])
			X = X[:,:,self.window_size[0]-1:]
			end_time = times[-self.window_size[0]]
			times = np.linspace(times[0], end_time, X.shape[-1])
			s_freq = epochs.info['sfreq']
			time_red = 1/s_freq * 1000 * self.window_size[0]
			warnings.warn(('Final timepoint in analysis is reduced by'
							f'{time_red} ms as each timepoint in analysis now'
							f'reflects {self.window_size[0]} data samples at a'
							f'sampling rate of {s_freq} Hz'), UserWarning)
		
		return 	X,y,df, times

	def set_bdm_param(self,y:np.array,beh:pd.DataFrame,cnds:list,cnd_head:str,
					labels_oi:Union[str,list],downscale:bool)-> \
						Tuple[int,list,list,str]:
		"""
		Based on classification input set the parameters for the 
		decoding analysis

		Args:	
			y (np.array): decoding labels
			beh (pd.DataFrame): behavioral parameters per epoch
			cnds (list): list with condition info.
			cnd_head (str): column name with conditon info
			labels_oi (str | list): labels used for decoding
			downscale (bool): if True decoding is run on increasingly 
			smaller trial numbers in an iterative procedure
			
		Returns:
			nr_labels (int): number of classes
			max_tr (list): maximum number of trials to be used for 
			balanced class sampling
			cnds (list): list of (training) conditions
			test_cnd (str): test condition
		"""

		if labels_oi == 'all':
			nr_labels = np.unique(y).size
		else:
			nr_labels = len(labels_oi)

		if isinstance(cnds[0], list):
			# split train and test conditions for cross train analysis
			tr_cnds, te_cnds = cnds
			self.cross = True
			self.nr_folds = 1
			tr_max = [self.selectMaxTrials(beh,tr_cnds,labels_oi,cnd_head)]
		else:
			self.cross = False
			if self.nr_folds == 1:
				self.nr_folds = 10
				warnings.warn('Nr folds is set to default as only one fold ' +
				'was specified. Please check whether this was intentional')
			tr_max = [self.selectMaxTrials(beh,cnds,labels_oi,cnd_head)] 
			if downscale:
				tr_max = [(i+1)*self.nr_folds 
							for i in range(int(tr_max[0]/self.nr_folds))][::-1]
			tr_cnds, te_cnds = cnds, None
			
		if isinstance(te_cnds, str):
			te_cnds = [te_cnds]

		return nr_labels,tr_max,tr_cnds,te_cnds

	def plot_bdm(self,bdm_scores:dict,cnds:list):

		times = bdm_scores['info']['times']
		fig, ax = plt.subplots(1)
		plt.ylabel(self.metric)
		plt.xlabel('Time (ms')
		# loop over all specified conditins
		for cnd in cnds:
			X = bdm_scores[cnd]['dec_scores']
			plt.plot(times, X, label = cnd)
		plt.legend(loc='best')

		return fig   

	def report_bdm(self,bdm_scores:dict,cnds:list,bdm_name:str):

		pass
		# set report and condition name
		# if self.elec_oi != 'all':
		# 	return None

		# name_info = bdm_name.split('_')
		# report_name = name_info[-1]
		# report_path = self.set_folder_path()
		# report_name = self.folder_tracker(report_path,
		# 								f'report_{report_name}.h5')

		# # create fake info object (so that weights can be plotted in report)
		# montage = mne.channels.make_standard_montage(self.montage)
		# n_elec = len(montage.ch_names)
		# info = mne.create_info(ch_names=montage.ch_names,sfreq=self.downsample,
        #                        ch_types='eeg')
		# t_min = bdm_scores['info']['times'][0]

		# # loop over all specified conditins
		# for cnd in cnds:

		# 	# set condition name
		# 	cnd_name = '_'.join(map(str, name_info[:-1] + [cnd]))
	
		# 	# create fake ekoked array
		# 	W = bdm_scores[cnd]['W']
		# 	W_evoked = mne.EvokedArray(W.T, info, tmin = t_min)
		# 	W_evoked.set_montage(montage)

		# 	# check whether report exists
		# 	if os.path.isfile(report_name):
		# 		with mne.open_report(report_name) as report:
		# 			# if section exists delete it first
		# 			report.remove(title=cnd_name)
		# 			report.add_evokeds(evokeds=W_evoked,titles=cnd_name,	
		# 		 				  n_time_points=30)
		# 		report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html', 
		# 				overwrite = True)
		# 	else:
		# 		report = mne.Report(title='Single subject evoked overview')
		# 		report.add_evokeds(evokeds=W_evoked,titles=cnd_name,	
		# 		 				n_time_points=30)
		# 		report.save(report_name)
		# 		report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html')

		# name = '_'.join(map(str, name_info[:-1]))

		# report.add_figure(self.plot_bdm(bdm_scores, cnds), 
        #                         title = name, section = 'bdm_scores')
		# report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html', 
		# 				overwrite = True)
			
	def set_folder_path(self) -> list:
		"""
		sets the folder path for the current analysis based on 
		input parameters

		Returns:
			list: folder path (used as extension to set file location)
		"""

		base = ['bdm']
		base += [self.to_decode, f'{self.elec_oi}_elecs']

		if self.cross:
			base += ['cross']

		# if self.bdm_type != 'broad':
		# 	base += [self.classifier]

		if self.classifier != 'LDA':
			base += [self.classifier]

		return base
		
	def sliding_window(self,X:np.array,window_size:int=20,demean:bool=True, 
						avg_window:bool=False)->np.array:
		"""	
		Copied from temp_dec developed by @author: jasperhajonides 
		(github.com/jasperhajonides/temp_dec)
			
		Reformat array so that time point t includes all information 
		from features up to t-n where n is the size of the 
		predefined window.

		Allows input to be either:
    	- 3D array [trial repeats x electrodes x time points]
    	- 4D array [frequencies x trial repeats x electrodes x time points]

		example:
			
		100, 60, 240 = X.shape
		data_out = sliding_window(X, window_size=5)
		100, 300, 240 = output.shape

		Args:
			X (np.array):Input array (3D or 4D)
			window_size (int, optional): number of time points to 
				include in the sliding window. Defaults to 20.
			demean (bool, optional): subtract mean from each feature 
				within the specified sliding window. Defaults to True.
			avg_window (bool, optional): If True rather than increasing 
				number of features, each timepoint reflects the average 
				activity within that time window and subsequent timepoint 
				as defined by the size of the window. Defaults to False 

		Raises:
			ValueError: In case data has incorrect format

		Returns:
			output (np.array): Reshaped array with increased feature 
				dimension based on window_size, or averaged within 
				window if avg_window=True
		"""

		# Handle both 3D and 4D inputs
		if X.ndim == 3:
			n_obs, n_elec, n_time = X.shape
			n_freq = 1
			X = X[np.newaxis, ...]  # Add frequency dimension
		elif X.ndim == 4:
			n_freq, n_obs, n_elec, n_time = X.shape
		else:
			raise ValueError("Input must be 3D or 4D array")
		
		if window_size <= 1 or n_time < window_size:
			print('Input data not suitable. Data will be returned')
			return X

		# predefine variables
		if avg_window:
			output = np.zeros(X.shape)
		else:
			output = np.zeros((n_freq, n_obs, n_elec*window_size, n_time))


		# Loop over frequencies and time points
		for freq in range(n_freq):
			for t in range(window_size-1, n_time):
				# Get window data for current frequency
				window_data = X[freq, :, :, (t-window_size+1):(t+1)]
				mean_value = window_data.mean(2)
				
				if avg_window:
					output[freq, :, :, t] = mean_value
				else:
					# Reshape and demean if selected
					x_window = window_data.reshape(n_obs, n_elec*window_size)
					if demean:
						x_window -= np.tile(mean_value, window_size).reshape(
							n_obs, n_elec*window_size)
					output[freq, :, :, t] = x_window
		
		if self.window_size[2]:
			print('downsampling data')
			output = mne.filter.resample(output, down=self.down_factor, 
									npad='auto', pad='edge')
			
		# Remove singleton frequency dimension for 3D input
		if X.shape[0] == 1:
			output = output[0]

		return output

	def average_trials(self,epochs:mne.Epochs,beh:pd.DataFrame,
				beh_headers:list) -> Tuple[np.array, pd.DataFrame]:
		"""
		Reduces shape of eeg data by averaging across trials. 
		The number of trials used for averaging is set as a BDM 
		parameter. Averaging is done across all unique labels and 
		conditions as specified in the behavior info. Remaining trials 
		after grouping will averaged together.

		example 1 (data contains two labels, each with 
		four observations):
			
		8, 64, 240 = epochs._data.shape
		self.tr_avg = 4
		epochs, beh = self.average_trials(epochs, beh) 
		2, 64, 240 = epochs._data.shape

		example 2 (data contains two labels, each with 
		four observations):

		8, 64, 240 = epochs._data.shape
		self.tr_avg = 3
		epochs, beh = self.average_trials(epochs, beh) 
		4, 64, 240 = epochs._data.shape

		Args:
			epochs (mne.Epochs): epoched data [trial repeats by 
			electrodes by time points].
			beh (pd.DataFrame): behavior dataframe with two 
			columns (conditions and labels)
			beh_headers (list): Header of decoding labels and condition 
			info in behavior. If no condition info is specified, 
			data will be averaged across all trials

		Returns:
			epochs (mne.Epochs):epoched data data after trial averaging
			beh (pd.DataFrame): updated behavior dataframe 
		"""

		#TODO: make sure trial averaging is always done on the same subset of data

		# get averaging info
		if isinstance(self.avg_trials, list):
			avg_trials = self.avg_trials[-1]
		else:
			avg_trials = self.avg_trials

		if avg_trials == 1:
			return epochs, beh

		print(f'Averaging across {avg_trials} trials')
		# initiate condition and label list
		cnds, labels, X = [], [], []

		# slice beh
		beh = beh.loc[:,[h for h in beh_headers if h is not None]]
		if beh.shape[-1] == 1:
			cnd_header = 'condition'
			beh['condition'] = 'all_data'
		else:
			cnd_header = beh_headers[1]	
			if beh.shape[-1] == 3:
				split_header = beh_headers[-1]
				split = []

		# loop over each label and condition pair
		options = dict(beh.apply(lambda col: col.unique()))
		keys, values = zip(*options.items())
		for var_combo in [dict(zip(keys, v)) \
			for v in itertools.product(*values)]:
			for i, (k, v) in enumerate(var_combo.items()):
				if i == 0:
					df_filt = f'{k} == \'{v}\'' if isinstance(v,str) \
					else f'{k} == {v}'
				else:
					if isinstance(v,str):
						df_filt += f' and {k} == \'{v}\''
					else:
						df_filt += f' and {k} == {v}'

			# select subset of data and average across random selection
			avg_idx = beh.query(df_filt).index.values
			random.shuffle(avg_idx)
			avg_idx = [avg_idx[i:i+avg_trials] 
						for i in np.arange(0,avg_idx.size, avg_trials)]
			X += [epochs._data[idx].mean(axis = 0) for idx in avg_idx]
			labels  += [var_combo[self.to_decode]] * len(avg_idx)
			cnds += [var_combo[cnd_header]] * len(avg_idx)
			if beh.shape[-1] == 3:
				split += [var_combo[split_header]] * len(avg_idx)

		# set data
		epochs._data = np.stack(X)
		if beh.shape[-1] == 3:
			beh = pd.DataFrame.from_dict({cnd_header:cnds,
								self.to_decode:labels, split_header:split})
		else:
			beh = pd.DataFrame.from_dict({cnd_header:cnds,
								self.to_decode:labels})

		return epochs, beh

	def get_condition_labels(self,beh:pd.DataFrame,cnd_header:str,
				cnd:str,max_tr:list,labels:Union[str,list]='all', 
				collapse:bool=False)->\
				Tuple[pd.DataFrame,np.array,np.array,np.array,list]:
		"""
		Based on input data selects the condition indices and labels. 
		These serve as input for train_test_split (and train_test_cross)
		to create training and test data

		Args:
			beh (pd.DataFrame): behavior dataframe with variables of 
			interest
			cnd_header (str): column name that contains condition 
			information
			cnd (str): current condition used for decoding. 
			In a cross training analysis this is the train condition
			max_tr (list): contains the maximum number of observations 
			that allows for balanced class assignment. 
			Is only adjusted in case collapse is set to True
			labels (str, list): unique decoding labels. 
			Defaults to 'all'.
			collapse (bool, optional): specifies wheter or not  
			decoding is performed on collapsed condition data. 
			Defaults to False.

		Returns:
			beh (pd.DataFrame): behavior dataframe with variables of 
			interest
			cnd_idx (np.array): trial indices of current condition
			cnd_labels (np.array): condition labels
			labels (np.array): unique decoding labels
			max_tr (list): contains the maximum number of observations 
			that allows for balanced class assignment
		"""

		# get condition indices
		if cnd == 'all_data':
			cnd_idx = np.arange(beh.shape[0])
		elif cnd != 'collapsed':
			cnd_idx = np.where(beh[cnd_header] == cnd)[0]
			if collapse:
				beh.loc[cnd_idx,'collapsed'] = 'yes'
		else:
			# reset max_tr again such that analysis is not underpowered
			max_tr = [self.selectMaxTrials(beh, ['yes'], labels,'collapsed')]
			cnd_idx = np.where(beh.collapsed == 'yes')[0]
		cnd_labels = beh[self.to_decode][cnd_idx].values

		# make sure that labels that should not be in analysis are excluded
		# if not already done so
		if not isinstance(labels, str):
			sub_idx = [i for i,l in enumerate(cnd_labels) if l in labels]	
			cnd_idx = cnd_idx[sub_idx]
			cnd_labels = cnd_labels[sub_idx]

		# print decoding update
		labels, counts = np.unique(cnd_labels, return_counts = True)
		if not self.cross:
			print (f'\nYou are decoding {cnd}. The nr of trials used for') 
			print('folding is set to {}'.format(max_tr[0]))
		diff = max(counts) - min(counts)
		print ('\nThe difference between the highest and the lowest')
		print (f'number of observations per class is {diff}')

		return beh, cnd_idx, cnd_labels, labels, max_tr

	def train_test_split(self,idx:np.array,labels:np.array,max_tr:int, 
						bdm_info: dict) -> Tuple[np.array, np.array, dict]:
		"""
		Splits up data into training and test sets. The number of 
		training and test sets is equal to the number of folds. 
		Splitting is done such that all data is tested exactly once.
		Number of folds determines the ratio between training and test 
		trials. With 10 folds, 90% of the data is used for training and 
		10% for testing. Ensures that the number of observations per 
		class is balanced both in the training and the testing set
 
		Args:
			idx (np.array): trial indices of decoding labels
			labels (np.array): array of decoding labels
			max_tr (int): max number unique labels
			bdm_info (dict): dictionary with selected trials per label. 
			If the value of the current run is {}, 
			a random subset of trials will be selected

		Returns:
			train_tr (np.array): trial indices per fold 
			and unique label [folds, labels, trials]
			test_tr (np.array): trial indices per fold and 
			unique label [folds, labels, trials]
			bdm_info (dict): cross-validation info per decoding run 
			(can be reported for replication purposes)
		"""

		# set up params
		N = self.nr_folds
		nr_labels = np.unique(labels).size
		steps = int(max_tr/N)

		# select final sample for BDM and store those trials in 
		# dict so that they can be saved
		if self.seed:
			random.seed(self.seed) # set seed 
		if bdm_info['run_' + str(self.run_info)] == {}:
			for i, l in enumerate(np.unique(labels)):
				label_dct = {l:idx[random.sample(list(np.where(labels==l)[0]),
							max_tr)]}
				bdm_info['run_' + str(self.run_info)].update(label_dct)	

		# initiate train and test arrays	
		train_tr = np.zeros((N,nr_labels, steps*(N-1)),dtype = int)
		test_tr = np.zeros((N,nr_labels,steps),dtype = int)

		# split dataset into N equally sized subsets for cross validation 
		for i, b in enumerate(np.arange(0,max_tr,steps)):
			
			idx_train = np.ones(max_tr,dtype = bool)
			idx_test = np.zeros(max_tr, dtype = bool)

			idx_train[b:b + steps] = False
			idx_test[b:b + steps] = True

			for j, key in enumerate(bdm_info['run_' + \
							str(self.run_info)].keys()):
				train = bdm_info['run_' + str(self.run_info)][key][idx_train]
				test = bdm_info['run_' + str(self.run_info)][key][idx_test]
				train_tr[i,j,:] = np.sort(train)
				test_tr[i,j,:] = np.sort(test)

		return train_tr, test_tr, bdm_info

	def train_test_cross(self,X:np.array,y:np.array,train_idx:np.array, 
				test_idx:Union[np.array,bool],max_tr:int=None)-> \
				Tuple[np.array, np.array, np.array, np.array]:
		"""
		Creates independent training and test sets (based on training 
		and test conditions). Ensures that the number of observations 
		per classis balanced both in the training and the testing set

		Args:
			X (np.array): Input data [trials x channels x time] or 
           		[freq x trials x channels x time]
			y (np.array): Decoding labels 
			train_idx (np.array): Indices of train trials 
				(i.e., condition specific data as selected in classify
				or localizer_classify)
			test_idx (np.array | bool): Indices of test trials. If False
				only training data is selected

		Returns:
			Xtr (array): Training data [folds x (freq) x trials x 
				channels x time]
			Xte (array): Test data [folds x (freq) x trials x channels x time]
			Ytr (array): Training labels. Training label for trial epoch in Xtr
			Yte (array): Test labels. Test label for each trial in Xte
		"""

		if self.seed:
			random.seed(self.seed) # set seed 

		# make sure that train label counts is balanced
		tr_labels = y[train_idx].values
		labels, label_counts = np.unique(tr_labels, return_counts = True)
		if not isinstance(max_tr, int):
			max_tr = min(label_counts)	

		# select train data and labels
		tr_idx = np.hstack([random.sample(list(np.where(tr_labels == l)[0]),  
									k = max_tr) for l in labels])
		Ytr = tr_labels[tr_idx]

		# Handle 3D vs 4D input
		if X.ndim == 3:  # [trials × channels × time]
			Xtr = X[train_idx[tr_idx]]
		else:  # [freq × trials × channels × time]
			Xtr = X[:, train_idx[tr_idx]]

		# add new (empty) axis to data so that cross time decoding can 
		# index (arteficial) folds
		Xtr = Xtr[np.newaxis, ...]
		Ytr = Ytr[np.newaxis, ...]

		# match test and train labels (if not already the case)
		if isinstance(test_idx, np.ndarray): 
			test_idx = [idx for idx in test_idx if y[idx] in tr_labels]

			# make sure that test label counts is balanced
			test_labels = y[test_idx].values
			labels, label_counts = np.unique(test_labels, return_counts = True)
			if not isinstance(max_tr, int):
				max_tr = min(label_counts)	

			# select test data and labels
			te_idx = [random.sample(list(np.where(test_labels == l)[0]),  
											k = max_tr) for l in labels]
			te_idx = np.hstack(te_idx)
			Yte = test_labels[te_idx]

			# Select test data based on dimensionality
			if X.ndim == 3:
				Xte = X[np.array(test_idx)[te_idx]]
			else:
				Xte = X[:, np.array(test_idx)[te_idx]]			

			Xte = Xte[np.newaxis, ...]
			Yte = Yte[np.newaxis, ...]
		else:
			Xte = None
			Yte = None

		return Xtr, Xte, Ytr, Yte

	def train_test_select(self,X:np.array,Y:np.array,train_tr:np.array, 
						test_tr:np.array) -> \
						Tuple[np.array, np.array, np.array, np.array]:
		"""
		Based on training and test data 
		(as returned by train_test_split) splits data into training and 
		test data

		Args:
			X: Input data [trials x channels x time] 
				or [freq x trials x channels x time]
			Y (np.array): decoding labels 
			train_tr (np.array): indices of train trials per fold and 
				unique label [folds, train labels, train trials]
			test_tr (np.array): indices of test trials per fold and 
			unique label [folds, test labels, test trials]

		Returns:
			Xtr (array): Training data 
			[nr folds x (freqs) x nr trials, elecs,time] 
			Xte (array): Test data [nr folds x (freqs) x nr trials, elecs,time]
			Ytr (array): training labels. Training label for trial epoch 
				in Xtr [folds × trials]
			Yte (array): test labels. Test label for each trial in Xte
				[folds × trials]
		"""

		# initialize train and test label arrays
		Ytr = np.zeros(train_tr.shape, dtype = Y.dtype).reshape(self.nr_folds, -1)
		Yte = np.zeros(test_tr.shape, dtype = Y.dtype).reshape(self.nr_folds, -1)

		# Initialize data arrays with proper dimensions
		if X.ndim == 3:  # [trials × channels × time]		
			Xtr = np.zeros((self.nr_folds, np.product(train_tr.shape[-2:]), 
				   			X.shape[1],X.shape[2]))
			Xte = np.zeros((self.nr_folds, np.product(test_tr.shape[-2:]), 
				   			X.shape[1],X.shape[2]))
		else:
			Xtr = np.zeros((self.nr_folds, X.shape[0], 
						np.product(train_tr.shape[-2:]), 
						X.shape[2], X.shape[3]))
			Xte = np.zeros((self.nr_folds, X.shape[0], 
						np.product(test_tr.shape[-2:]), 
						X.shape[2], X.shape[3]))			

		# select data for each fold
		for n in range(train_tr.shape[0]):
			if X.ndim == 3:
				Xtr[n] = X[np.hstack(train_tr[n])]
				Xte[n] = X[np.hstack(test_tr[n])]
			else:
				Xtr[n] = X[:,np.hstack(train_tr[n])]
				Xte[n] = X[:,np.hstack(test_tr[n])]
			Ytr[n] = Y[np.hstack(train_tr[n])]
			Yte[n] = Y[np.hstack(test_tr[n])]

		return Xtr, Xte, Ytr, Yte	

	def set_bdm_weights(self, W, Xtr, nr_elec, nr_time):
		#TODO: add docstring
		#TODO: make it work with GAT
		# stack all training data
		if  W.ndim > 2:
			W =  False
		else:
			Xtr = Xtr.reshape(-1,nr_elec, nr_time)
			W = np.stack([np.matmul(np.cov(Xtr[...,i].T),W[i]) 
					for i in range(nr_time)])

		return W

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
			with open(self.FolderTracker(['bdm',self.elec_oi, 'cross'], fname = 'class_{}-{}.pickle'.format(sj,te_header)) ,'wb') as handle:
				pickle.dump(classification, handle)
		
		return classification


	def crossClassify(self, sj, cnds, cnd_header, time, tr_header, te_header, tr_te_rel = 'ind', excl_factor = None, tr_factor = None, te_factor = None, bdm_labels = 'all', gat_matrix = False, save = True, bdm_name = 'cross'):	
		'''
		Update function but it does the trick
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

	def cross_time_decoding(self, Xtr, Xte, Ytr, Yte, labels, GAT= False, X=[]):
		'''
		Decoding across all time points, supporting both 3D and 4D 
		input.

		Args:
			Xtr: Training data [folds x trials x features x time] or 
				[folds x freq x trials x features x time]
			Xte: Test data (same format as Xtr)
			Ytr: Training labels [folds x trials]
			Yte: Test labels [folds x trials]
			labels: Label set(s) for training and testing
			GAT: If True, compute generalization across time
			X: Optional raw data for PCA computation
		
		Returns:
			class_acc: Classification accuracies
			weights: Classifier weights
			conf_matrix: Confusion matrix
		'''

		# set necessary parameters
		N = self.nr_folds
		nr_time_tr = Xtr.shape[-1]
		nr_time_te = Xte.shape[-1] if GAT else 1

		# Handle 3D vs 4D input	
		if Xtr.ndim == 4:  # 3D data: [folds × trials × features × time]
			nr_freq = 1
			# Insert freq dimension at position 1
			Xtr = np.expand_dims(Xtr, axis=1)  
			Xte = np.expand_dims(Xte, axis=1)
		elif Xtr.ndim == 5:  # 4D data: [folds × freq × trials × features × time]
			nr_freq = Xtr.shape[1]
		else:
			raise ValueError("Input must be 4D or 5D array")

		# Get dimensions
		_, _, _, nr_elec, nr_time_tr = Xtr.shape
		nr_time_te = Xte.shape[-1] if GAT else 1

		# initiate classifier
		clf = self.select_classifier()

		 # Handle label sets for training/testing
		if isinstance(labels, np.ndarray):
			labels = (labels,labels)

		 # Initialize arrays for results
		class_acc = np.zeros((N, nr_freq,nr_time_tr, nr_time_te))
		weights = np.zeros((N,nr_freq,nr_time_tr, nr_time_te, nr_elec))
		conf_matrix = np.zeros((N,nr_freq,nr_time_tr, nr_time_te,
			  					len(labels[1]),len(labels[0])))


		for n in range(N):
			print(f'\rFold {n+1} out of {N} folds in run {self.run_info}', end='')
			Ytr_ = Ytr[n]
			Yte_ = Yte[n]

			for freq in range(nr_freq):	
				for tr_t in range(nr_time_tr):
					# When GAT=False, we only need one iteration
					te_time_points = range(nr_time_te) if GAT else [0]
					for te_t in te_time_points:

						Xtr_ = Xtr[n,freq,:,:,tr_t]
						Xte_ = Xte[n,freq,:,:,te_t if GAT else tr_t]

						# Apply standardization if specified
						if self.scale['standardize']:
							scaler = StandardScaler(with_std=self.scale['scale'])
							Xtr_ = scaler.fit_transform(Xtr_)
							Xte_ = scaler.transform(Xte_)

						if self.pca_components[0]:
							# Show warning only once if data is not standardized
							if not self.scale['standardize'] and tr_t == 0 and n == 0 and self.run_info == 1:
								warnings.warn('It is recommended to standardize the data before '
											'applying PCA correction')
							
							if self.pca_components[1] == 'across':
								pca = PCA(n_components=self.pca_components[0])
								Xtr_ = pca.fit_transform(Xtr_)
								Xte_ = pca.transform(Xte_)
							elif self.pca_components[1] == 'all':
								if X != []:
									X_ = X[..., tr_t] if X.ndim == 3 else X[freq, ..., tr_t]
									if self.scale['standardize']:
										X_ = StandardScaler().fit_transform(X_)
									pca = PCA(n_components=self.pca_components[0])
									pca.fit(X_)
									Xtr_ = pca.fit_transform(Xtr_)
									Xte_ = pca.transform(Xte_)

										
						# Train and test classifier
						clf.fit(Xtr_,Ytr_)
						scores = clf.predict_proba(Xte_) # get posteriar probability estimates
						predict = clf.predict(Xte_)

						# Compute performance metrics
						if bool(set(Ytr_)& set(Yte_)):
							class_perf = self.computeClassPerf(scores, Yte_, 
										  np.unique(Ytr_), predict) #
							conf_m = confusion_matrix(Yte_, predict,labels=labels[0])
						else:
							class_perf = 0
							conf_m = self.get_fake_confusion_matrix(Yte_, predict) 

						# store results
						# TODO: create interpretable weights
						class_acc[n, freq, tr_t, te_t] = class_perf
						conf_matrix[n, freq, tr_t, te_t] = conf_m
						if not self.pca_components[0]:
							weights[n, freq, tr_t, te_t] = self.get_classifier_weights(
							clf, Xtr_)
						# pairs = list(zip(Yte_, predict))

		weights = np.mean(weights, axis=0)
		conf_matrix = np.sum(conf_matrix, axis=0)
		class_acc = np.mean(class_acc, axis=0)

		return class_acc, weights, conf_matrix

	def get_fake_confusion_matrix(self,y_true:np.array,
			       				y_pred:np.array)->np.array:

		row_values = np.unique(y_true)
		col_values = np.unique(y_pred)
		output = np.zeros((row_values.size, col_values.size)) 

		for r,r_value in enumerate(row_values):
			for c,c_value in enumerate(col_values):
				output[r,c] = sum(y_pred[y_true == r_value] == c_value)

		return output

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

		if self.metric == 'auc':
			
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

		elif self.metric == 'acc':
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
		if cnds != ['all_data']:
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
		elif cnds == ['all_data']:
			labels = [l for l in beh[self.to_decode] if l in bdm_labels]
			min_tr = np.unique(labels, return_counts = True)[1]
			max_trials = int(np.floor(min(min_tr)/N)*N)	

		if max_trials == 0:
			print('At least one condition does not contain sufficient info for current nr of folds')

		return max_trials




	


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