"""
Brain Decoding Multivariate (BDM) Analysis Module.

This module provides comprehensive multivariate decoding functionality 
for EEG dataanalysis. The BDM class implements various machine learning 
classifiers to predict experimental variables (classes) from observed 
patterns of brain activity, primarily using Linear Discriminant Analysis 
(LDA) by default.

Key Features
------------
- Multiple classifier support (LDA, SVM, Gaussian Naive Bayes)
- K-fold cross-validation with balanced class sampling
- Time-frequency domain decoding support
- Generalization Across Time (GAT) analysis
- Independent training/testing sets (localizer paradigm)
- Sliding window feature enhancement
- PCA dimensionality reduction
- Comprehensive performance metrics (AUC, accuracy)

Classes
-------
BDM : Main class for brain decoding analysis
    Inherits from FolderStructure for file management

Notes
-----
The module uses event balancing through undersampling to ensure equal 
class representation. Area under the curve (AUC) is the default 
performance metric.

Examples
--------
>>> from eeg_analyses.BDM import BDM
>>> bdm = BDM(sj=1, epochs=epochs, df=behavior_df, 
				to_decode='target_loc')
>>> results = bdm.classify(window_oi=(0.1, 0.5))

See Also
--------
eeg_analyses.TFR : Time-frequency analysis for use with BDM
support.FolderStructure : Base class for file organization
"""

import os
import mne
import pickle
import random
import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Generic, Union, Tuple, Any, List
from support.FolderStructure import FolderStructure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from support.support import select_electrodes, trial_exclusion, get_time_slice
from scipy.stats import rankdata
from eeg_analyses.TFR import TFR

warnings.simplefilter('default')

class BDM(FolderStructure):
	"""
	Brain Decoding Multivariate analysis class for EEG data 
	classification.

	The BDM class provides comprehensive multivariate decoding 
	functionality to predict experimental variables (classes) from 
	observed patterns of brain activity. It employs Linear Discriminant
	Analysis (LDA) by default and supports various other classifiers.

	The class implements k-fold cross validation with balanced class 
	sampling, ensuring that each class has equal representation in both 
	training and testing sets. Event balancing is achieved through 
	undersampling.

	Parameters
	----------
	sj : int
		Subject identifier used for file naming and organization.
	epochs : mne.Epochs or list of mne.Epochs
		Epoched EEG data. If list of two Epochs objects, the first 
		serves as training set and second as test set for
		localizer-based decoding.
	df : pd.DataFrame or list of pd.DataFrame  
		Behavioral parameters dataframe. Each row corresponds to an 
		epoch.If list, should match epochs structure for localizer 
		paradigm.
	to_decode : str
		Column name in df containing class labels for classification.
	nr_folds : int, default=10
		Number of folds for k-fold cross validation.
	classifier : {'LDA', 'svm', 'GNB'}, default='LDA'
		Classifier type:
		- 'LDA': Linear Discriminant Analysis
		- 'svm': Support Vector Machine (with calibration)
		- 'GNB': Gaussian Naive Bayes
	data_type : {'raw', 'tfr'}, default='raw'
		Data domain for decoding:
		- 'raw': Time domain EEG data
		- 'tfr': Time-frequency domain power
	tfr : TFR, optional
		Pre-computed TFR object for time-frequency decoding. Required if
		data_type='tfr' and min_freq/max_freq not provided.
	metric : {'auc', 'acc'}, default='auc'
		Performance metric:
		- 'auc': Area Under Curve (ROC)
		- 'acc': Classification accuracy
	elec_oi : str or list, default='all'
		Electrodes of interest. Can be predefined subset name or list
		of electrode names.
	downsample : int, default=128
		Target sampling frequency for computational efficiency.
	avg_runs : int, default=1
		Number of cross-validation repetitions to average.
	avg_trials : int, default=1
		Number of trials to average within each condition/label 
		combination before cross-validation.
	sliding_window : tuple, default=(1, True, False)
		Sliding window parameters (size, demean, average):
		- size: Window size in samples
		- demean: Whether to demean within window
		- average: Whether to average within window vs. 
			concatenate features
	scale : bool, default=False
		Whether to apply standardization and unit variance scaling.
	pca_components : tuple, default=(0, 'across')
		PCA parameters (n_components, mode):
		- n_components: Number of components (or fraction if <1)
		#TODO: check whether this is correct
		- mode: 'across' (fit on train+test) or 'all' 
			(fit on train only)
	montage : str, default='biosemi64'
		EEG montage for visualization in reports.
	output_params : bool, default=False
		Whether to save classifier weights and confusion matrices.
	baseline : tuple, optional
		Baseline correction window (start, end) in seconds.
	seed : int or bool, default=42213
		Random seed for reproducibility. If False, no seed applied.
	min_freq : float, optional
		Minimum frequency for auto-generated TFR (when data_type='tfr').
	max_freq : float, optional  
		Maximum frequency for auto-generated TFR (when data_type='tfr').
	**tfr_kwargs
		Additional arguments passed to TFR constructor.

	Attributes
	----------
	data_type : str
		Immutable data type set during initialization.
	cross : bool
		Whether using cross-condition decoding paradigm.

	Raises
	------
	ValueError
		If data_type='tfr' but neither tfr object nor frequency 
			parameters provided.
		If invalid classifier or data_type specified.
	Warning
		If TFR parameters provided when data_type='raw'.

	Examples
	--------
	Standard within-condition decoding:

	>>> bdm = BDM(sj=1, epochs=epochs, df=behavior, 
					to_decode='stimulus_type')
	>>> results = bdm.classify(window_oi=(0.1, 0.5))

	Cross-condition decoding:

	>>> cnds = {'condition': [['train_cond'], 'test_cond']}
	>>> results = bdm.classify(cnds=cnds)

	Time-frequency decoding:

	>>> bdm = BDM(sj=1, epochs=epochs, df=behavior, 
					to_decode='response',data_type='tfr',
					min_freq=8, max_freq=12)
	>>> results = bdm.classify()

	Generalization Across Time:

	>>> results = bdm.classify(GAT=True)

	Notes
	-----
	- Class balancing is performed through undersampling
	- Default metric (AUC) is robust to class imbalance
	- Trial averaging reduces noise but decreases sample size
	- PCA should be combined with data standardization
	- Cross-validation ensures generalizability of results

	See Also
	--------
	TFR : Time-frequency analysis
	support.FolderStructure : Base class for file management
	"""

	def __init__(self,sj:int,epochs:Union[mne.Epochs,list],
				df:Union[pd.DataFrame,list],to_decode:str, 
				nr_folds:int=10,classifier:str='LDA',data_type:str='raw',
				tfr:Optional[TFR]=None,metric:str='auc',
				elec_oi:Union[str,list]='all',downsample:int=128,
				avg_runs:int=1,avg_trials:int=1,
				sliding_window:tuple=(1,True,False),
				scale:bool=False, 
				pca_components:tuple=(0,'across'),montage:str='biosemi64',
				output_params:Optional[bool]=False,
				baseline:Optional[tuple]=None, 
				seed:Union[int, bool] = 42213,min_freq:Optional[float]=None,
				max_freq:Optional[float]=None,**tfr_kwargs):
		"""
		Initialize BDM decoding analysis.

		Sets up all parameters for multivariate decoding analysis
		including classifier selection, cross-validation strategy, and
		data preprocessing options.

		Parameters
		----------
		sj : int
			Subject identifier used for file naming and organization.
		epochs : mne.Epochs or list of mne.Epochs
			Epoched EEG data. If list of two Epochs objects, first 
			serves as training set and second as test set for 
			localizer-based decoding.
		df : pd.DataFrame or list of pd.DataFrame
			Behavioral parameters dataframe where each row represents an 
			epoch. If list, should match epochs structure.
		to_decode : str
			Column name in df containing class labels for 
			classification.
		nr_folds : int, default=10
			Number of folds for k-fold cross validation.
		classifier : {'LDA', 'svm', 'GNB'}, default='LDA'
			Classifier type to use for decoding.
		data_type : {'raw', 'tfr'}, default='raw'
			Whether to perform decoding on raw EEG or time-frequency 
			data.
		tfr : TFR, optional
			Pre-computed TFR object for time-frequency decoding.
		metric : {'auc', 'acc'}, default='auc'
			Performance metric for classification evaluation.
		elec_oi : str or list, default='all'
			Electrodes of interest for decoding features.
		downsample : int, default=128
			Target sampling frequency for computational efficiency.
		avg_runs : int, default=1
			Number of cross-validation repetitions to average.
		avg_trials : int, default=1
			Number of trials to average within each condition/label 
			before CV.
		sliding_window : tuple, default=(1, True, False)
			Sliding window parameters (size, demean, average).
		scale : bool, default=False
			Whether to apply standardization before decoding.
		pca_components : tuple, default=(0, 'across')
			PCA dimensionality reduction parameters.
		montage : str, default='biosemi64'
			EEG montage for visualization.
		output_params : bool, default=False
			Whether to save classifier weights and confusion matrices.
		baseline : tuple, optional
			Baseline correction window in seconds.
		seed : int or bool, default=42213
			Random seed for reproducibility.
		min_freq : float, optional
			Minimum frequency for auto-generated TFR.
		max_freq : float, optional
			Maximum frequency for auto-generated TFR.
		**tfr_kwargs
			Additional arguments for TFR constructor.

		Raises
		------
		ValueError
			If data_type='tfr' but required TFR parameters missing.
			If invalid classifier or data_type specified.
		Warning
			If TFR parameters provided when data_type='raw'.

		Notes
		-----
		The data_type parameter becomes immutable after initialization to
		prevent inconsistent analysis configurations.
		"""	

		self.sj = sj
		self.epochs = epochs					
		self.df = df
		self.classifier = classifier
		self.baseline = baseline
		self.to_decode = to_decode
		self._data_type = data_type  # Use private attribute during init
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

		# data type is immutable after initialization
		self._data_type = data_type

		# Set up TFR processing: use provided object or
		#  auto-create from parameters
		if self._data_type == 'tfr':
			if tfr is not None:
				# Use explicit TFR object
				self.tfr = tfr
			elif min_freq is not None and max_freq is not None:
				# Auto-create TFR object
				self.tfr = TFR(
					sj=sj, epochs=epochs, df=df,
					min_freq=min_freq, max_freq=max_freq, 
					baseline=baseline, **tfr_kwargs
				)
			else:
				raise ValueError("When data_type='tfr', must provide either " \
							"'tfr' object or frequency parameters (min_freq" \
							", max_freq)")
		elif self._data_type == 'raw':
			if tfr is not None or min_freq is not None:
				raise Warning("TFR parameters ignored when data_type='raw'")
			self.tfr = None
		else:
			raise ValueError(f"data_type must be 'raw' or 'tfr', "
					f"got '{self._data_type}'")

	@property
	def data_type(self):
		"""
		Get the data type for decoding analysis.
		
		Returns
		-------
		str
			The data type ('raw' or 'tfr') set during initialization.
			
		Notes
		-----
		This property is immutable after initialization to prevent
		inconsistent analysis configurations.
		"""
		return self._data_type
	
	@data_type.setter
	def data_type(self, value):
		"""
		Prevent changing data_type after initialization.
		
		Parameters
		----------
		value : str
			Attempted new data type value.
			
		Raises
		------
		AttributeError
			Always raised since data_type cannot be changed after 
			initialization.
		"""
		raise AttributeError("data_type cannot be changed after " \
					"initialization. Create a new BDM instance instead.")

	def select_classifier(self) -> Any:
		"""
		Initialize and return the specified classifier.

		Creates an instance of the classifier specified during BDM 
		initialization. Supports Linear Discriminant Analysis, Support 
		Vector Machine (with calibration for probability estimates), 
		and Gaussian Naive Bayes.

		Returns
		-------
		sklearn classifier
			Initialized classifier object ready for training:
			- LinearDiscriminantAnalysis for 'LDA'
			- CalibratedClassifierCV(LinearSVC()) for 'svm' 
			- GaussianNB for 'GNB'

		Raises
		------
		ValueError
			If classifier type is not one of the supported options.

		Notes
		-----
		SVM classifier is wrapped with CalibratedClassifierCV to provide
		probability estimates required for AUC calculation.
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

	def get_classifier_weights(self, clf: Any, Xtr_: np.ndarray) -> np.ndarray:
		"""
		Extract classifier weights regardless of classifier type.

		Retrieves the feature weights from different types of trained 
		classifiers, handling the varied interfaces of sklearn 
		classifiers.

		Parameters
		----------
		clf : sklearn classifier
			Trained classifier object with fitted parameters.
		Xtr_ : np.ndarray
			Training data used to determine weight array dimensions.
			Shape: (n_samples, n_features)

		Returns
		-------
		np.ndarray
			Feature weights with shape (n_features,). Returns zero array
			if weights cannot be extracted from the classifier type.

		Notes
		-----
		Different classifiers store weights in different attributes:
		- LDA: use coef_ attribute
		- CalibratedClassifierCV (SVM): extract from base_estimator.coef_
		- Other classifiers: return zeros with warning

		The weights represent the contribution of each feature 
		(electrode/time) to the classification decision.
		"""
		if hasattr(clf, 'coef_'):
			# For LDA
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
				excl_factor:dict=None,nr_perm:int=0,
				GAT:Union[bool,Tuple[Tuple[float,float],
						 Tuple[float,float]]]=False, 
				downscale:bool=False,split_fact:dict=None,
				save:bool=True,bdm_name:str='main')->dict:
		"""
		Perform multivariate decoding across time on specified classes.

		This is the main decoding function that supports both 
		within-condition and cross-condition decoding analyses. 
		It implements k-fold cross-validation with balanced class 
		sampling and various data preprocessing options.

		Parameters
		----------
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
			
		window_oi : tuple, optional
			Time window of interest in seconds, e.g., (0.1, 0.5).
			If None, uses all time samples in epochs.
		labels_oi : str or list, default='all'
			Subset of class labels to include in decoding. If 'all',
			uses all unique labels in to_decode column.
		collapse : bool, default=False
			Whether to also perform decoding collapsed across all 
			conditions in addition to condition-specific decoding.
		excl_factor : dict, optional
			Criteria for excluding trials from analysis. Format:
			``{'column_name': ['value1', 'value2']}`` excludes trials
			where column_name contains specified values.
		nr_perm : int, default=0
			Number of permutation tests for chance-level baseline.
			If >0, additional decoding performed with shuffled labels.
		GAT : bool or tuple, default=False
			Generalization Across Time analysis:
			
			- False: Standard within-time decoding
			- True: Full GAT matrix (train at each time, 
									test at all times)
			- Tuple: Custom time windows as ((train_start, train_end), 
			  (test_start, test_end))
			  
		downscale : bool, default=False
			Whether to run decoding with progressively fewer trials to
			examine minimum trial requirements for reliable 
			classification.
		split_fact : dict, optional
			Additional factor to split analysis by. Decoding performed
			separately for each level and results averaged.
		save : bool, default=True
			Whether to save results to disk using standard file 
			organization.
		bdm_name : str, default='main'
			Name identifier for saving analysis results.

		Returns
		-------
		tuple
			(bdm_scores, bdm_params) where:
			
			bdm_scores : dict
				Classification results with structure:
				``{'condition': {'dec_scores': array}, 'info': {...}}``
				Contains accuracy/AUC scores and analysis metadata.
				
			bdm_params : dict
				Analysis parameters including classifier weights and
				confusion matrices (if output_params=True).

		Examples
		--------
		Basic within-condition decoding:
		
		>>> results, params = bdm.classify(window_oi=(0.1, 0.5))
		
		Cross-condition decoding:
		
		>>> cnds = {'block_type': [['practice'], 'test']}
		>>> results, params = bdm.classify(cnds=cnds)
		
		Generalization across time:
		
		>>> results, params = bdm.classify(GAT=True)
		
		Permutation testing:
		
		>>> results, params = bdm.classify(nr_perm=100)

		Notes
		-----
		- Results automatically include metadata about electrodes and 
			times
		- Cross-validation ensures independence of training and 
			test data
		- Class balancing prevents bias toward majority classes
		- Progress information printed during execution
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
									window_oi, excl_factor,headers,cnds)

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
										labels_oi, collapse,GAT,nr_perm,times)
		else:
			(bdm_scores, 
			bdm_params, 
			bdm_info) = self.iter_classify_(split_fact,X,y,df,tr_cnds,te_cnds,
								   			cnd_head,tr_max,labels_oi, 
											collapse,GAT,nr_perm,times)	

		bdm_scores.update({'info':{'elec':self.elec_oi,'times':times}})
		if self._data_type == 'tfr':
			bdm_scores['info'].update({'freqs':self.tfr.frex})
		
		if GAT:
			if isinstance(GAT, tuple):
				tr_window, te_window = GAT
				train_times = times[get_time_slice(times, tr_window[0], 
									   						tr_window[1])]
				test_times = times[get_time_slice(times, te_window[0],
									  						te_window[1])]
			else:
				train_times = test_times = times
			bdm_scores['info'].update({'test_times':test_times,})
			bdm_scores['info']['times'] = train_times

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
				df:pd.DataFrame,tr_cnds:list,te_cnds:list,cnd_head:str,
				tr_max:list,labels_oi:Union[str,list],collapse:bool,
				GAT:Union[bool,Tuple[Tuple[float,float],Tuple[float,float]]],
				nr_perm:int,times:np.array):
		"""
		Internal helper for iterative decoding across data subsets.
		
		Splits data by specified factors (e.g., blocks, sessions) and runs
		classify_ on each subset, then averages results across iterations.
		Used when multiple independent decoding runs are needed.
		
		Parameters
		----------
		split_fact : dict
			Factors to split data by (e.g., {'block': [1,2,3]})
		X, y, df : array-like
			EEG data, labels, and behavioral dataframe
		tr_cnds, te_cnds : list
			Training and test conditions
		Other parameters passed through to classify_
		
		Returns
		-------
		tuple
			Averaged bdm_scores, bdm_params, bdm_info across iterations
		"""

		# split the selected data into subsets and apply decoding 
		# within those subsets
		dec_scores, dec_params = [],[]
		for key, value in split_fact.items():
			for v in value:
				mask = df[key] == v
				X_ = X[mask]
				y_ = y[mask].reset_index(drop = True)
				df_ = df[mask].reset_index(drop = True)
				# reset max trials for folding
				tr_max = [self.select_max_trials(df_,tr_cnds,labels_oi,
								   			cnd_head)]
				
				(bdm_scores, 
				bdm_params, 
				bdm_info) = self.classify_(X_,y_,df_,tr_cnds,te_cnds,cnd_head,
							   			tr_max,labels_oi, collapse,GAT,nr_perm,
										times)
				
				dec_scores.append(bdm_scores)
				dec_params.append(bdm_params)

		# create averaged output dictionary
		#TODO: 
		for key in (k for k in bdm_scores if k != 'bdm_info'):
			output = np.mean([scores[key]['dec_scores'] for scores 
						 							in dec_scores], axis = 0)
			bdm_scores[key]['dec_scores'] = output
			#filthy hack for now
			if dec_params[0] != {}:
				W = np.mean([params[key]['W'] for params in dec_params],
																	axis = 0)
				bdm_params[key]['W'] = W
			else:
				bdm_params = {key: {}}
			
		return bdm_scores, bdm_params, bdm_info

	def classify_(self,X:np.array,y:np.array,df:pd.DataFrame,tr_cnds:list,
			   	te_cnds:list,cnd_head:str,tr_max:list,
				labels_oi:Union[str,list],collapse:bool,
				GAT:Union[bool,Tuple[Tuple[float,float],Tuple[float,float]]],
				nr_perm:int,times:np.array):
		"""
		Internal helper that performs the core decoding computation.
		
		Executes the main classification pipeline including 
		cross-validation, permutation testing, and GAT analysis. Handles 
		both within-condition and cross-condition decoding based on 
		tr_cnds and te_cnds.
		
		Parameters
		----------
		X, y : array-like
			EEG data and corresponding labels
		df : pd.DataFrame
			Behavioral data with condition information
		tr_cnds, te_cnds : list
			Training and test condition lists
		GAT : bool or tuple
			Generalization across time settings
		nr_perm : int
			Number of permutation iterations
		times : array
			Time vector for time window selection
		Other parameters : 
			Various classification parameters passed from main classify 
			method
		
		Returns
		-------
		tuple
			bdm_scores, bdm_params, bdm_info for current classification 
			run
		"""

		bdm_scores, bdm_params = {}, {}

		# set decoding parameters
		if X.ndim == 3:
			nr_epochs, nr_elec, nr_time = X.shape
			nr_freq = 1
		elif X.ndim == 4:
			nr_freq, nr_epochs, nr_elec, nr_time = X.shape

		nr_perm += 1

		# set train and test time points based on GAT
		if isinstance(GAT, tuple):
			tr_window, te_window = GAT
			tr_idx = get_time_slice(times, tr_window[0], tr_window[1])
			te_idx = get_time_slice(times, te_window[0], te_window[1])
		else: # full GAT or standard decoding
			window_oi = times[[0,-1]]
			tr_idx = get_time_slice(times, times[0], times[-1])
			te_idx = tr_idx

		nr_train = tr_idx.stop - tr_idx.start
		if GAT == False:
			nr_test = 1
		else:
			nr_test = te_idx.stop - te_idx.start

		# loop over (training and testing) conditions
		for tr_cnd in tr_cnds:
			for te_cnd in (te_cnds if te_cnds is not None else [None]):

				# reset selected trials
				bdm_info = {}

				# get condition indices and labels
				(df, cnd_idx, 
				cnd_labels, labels,
				tr_max) = self.get_condition_labels(df, cnd_head,tr_cnd,
													tr_max, labels_oi,collapse)

				# initiate decoding arrays for current condition
				class_acc = np.empty((self.avg_runs, nr_perm,nr_freq,
								nr_train, nr_test)) * np.nan
				weights = np.empty((self.avg_runs, nr_perm, nr_freq,
									nr_train, nr_test, nr_elec))
				conf_matrix = np.empty((self.avg_runs, nr_perm,nr_freq,
									nr_train, nr_test, labels.size,
									labels.size))

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
								test_idx = np.where(df[cnd_head] == te_cnd)[0]
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
							
							# restrict to training and test time window
							Xtr = Xtr[..., tr_idx]
							Xte = Xte[..., te_idx]
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

		max_tr = self.select_max_trials(beh_te,cnds,te_labels_oi,cnd_header)

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
						headers:list=None,cnds:list=None)-> \
						Tuple[np.array, pd.DataFrame, np.array]:
		"""
		Select and preprocess data for BDM analysis.

		Applies data transformation steps in the following order:
		1. Trial exclusion based on specified criteria
		2. Trial averaging (if specified)
		3. Time-frequency decomposition (if specified)
		4. Baseline correction (if specified)
		5. Downsampling (if specified)
		6. Time window and electrode selection
		7. Sliding window transformation (if specified)

		Parameters
		----------
		epochs : mne.Epochs
			Epoched EEG data to be processed.
		df : pd.DataFrame
			Behavioral dataframe corresponding to epochs.
		window_oi : tuple
			Time window of interest in seconds, e.g., (0.1, 0.5).
		excl_factor : dict, optional
			Trial exclusion criteria. 
			Format: {'column': ['value1', 'value2']} excludes trials 
			where column contains specified values.
		headers : list, optional
			Column names for condition-specific processing.
		cnds : list, optional
			Conditions of interest for trial selection.

		Returns
		-------
		X : np.ndarray
			Preprocessed EEG data with shape:
			- (n_epochs, n_channels, n_times) for raw data
			- (n_freqs, n_epochs, n_channels, n_times) for TFR data
		y : pd.Series
			Decoding labels corresponding to preprocessed data.
		df : pd.DataFrame
			Updated behavioral dataframe after preprocessing.
		times : np.ndarray
			Time samples corresponding to data time dimension.

		Notes
		-----
		This method handles the core data preprocessing pipeline 
		including trial exclusion, averaging, frequency decomposition, 
		and temporal windowing. The specific transformations applied 
		depend on the BDM class initialization parameters.
		"""

		# remove a subset of trials (including conditiions not of interest)
		cnd_header = headers[0]
		if headers is not None and cnds is not None and cnds != ['all_data']:
			# exclude trials based on condition info
			flat_cnds = []
			for item in cnds:
				if isinstance(item, list):
					flat_cnds.extend(item)
				else:
					flat_cnds.append(item)
			to_exclude = [cnd for cnd in df[cnd_header].unique() 
				 									if cnd not in flat_cnds]

			if to_exclude:
				if excl_factor is None:
					excl_factor = {}
				excl_factor.setdefault(cnd_header, []).extend(to_exclude)

		if excl_factor is not None:
			df, epochs,_ = trial_exclusion(df, epochs, excl_factor)

		# cnd selection for optional induced tfr decoding
		if self.tfr is not None:
			if self.tfr.power == 'induced':
				if cnds == ['all_data']:
					cnd_idx = [np.arange(df.shape[0])]
				else:
					cnd_idx = [df[cnd_header].values == cnd for cnd in cnds]
			else:
				cnd_idx = None

		# reset index(to properly align beh and epochs)
		df.reset_index(inplace = True, drop = True)

		# limit epochs object to electrodes of interest
		picks = mne.pick_types(epochs.info,eeg=True, eog= True, misc = True)
		picks = select_electrodes(np.array(epochs.ch_names)[picks], 
								self.elec_oi)
		epochs = epochs.pick_channels(np.array(epochs.ch_names)[picks])

		# apply filtering and downsampling (if specified)
		if self._data_type == 'tfr':
			X = self.tfr.compute_tfrs(epochs, for_decoding = True, 
							 		cnd_idx = cnd_idx)

		# apply baseline correction
		if isinstance(self.baseline, tuple) and self._data_type == 'raw':
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
				if self._data_type == 'tfr':
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
		if self._data_type == 'raw':
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
			tr_max = [self.select_max_trials(beh,tr_cnds,labels_oi,cnd_head)]
		else:
			self.cross = False
			if self.nr_folds == 1:
				self.nr_folds = 10
				warnings.warn('Nr folds is set to default as only one fold ' +
				'was specified. Please check whether this was intentional')
			tr_max = [self.select_max_trials(beh,cnds,labels_oi,cnd_head)] 
			if downscale:
				tr_max = [(i+1)*self.nr_folds 
							for i in range(int(tr_max[0]/self.nr_folds))][::-1]
			tr_cnds, te_cnds = cnds, None
			
		if isinstance(te_cnds, str):
			te_cnds = [te_cnds]

		return nr_labels,tr_max,tr_cnds,te_cnds

	def plot_bdm(self, bdm_scores: dict, cnds: list) -> None:

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

	def report_bdm(self, bdm_scores: dict, cnds: list, bdm_name: str) -> None:

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
		Generate standardized folder path components for analysis 
		organization.
		
		Creates a systematic folder structure based on analysis 
		parameters to ensure consistent file organization and easy 
		retrieval of results. The path components reflect key analysis 
		settings for identification.

		Returns
		-------
		list[str]
			Folder path components that will be joined to create the 
			full directory path for saving analysis results. 
			Components include:
			- 'bdm': Base identifier for BDM analyses
			- Decoding target (e.g., 'stimulus', 'response')
			- Electrode selection (e.g., 'all_elecs', 'post_elecs')
			- 'cross': Added if cross-condition analysis
			- Classifier name: Added if non-default (not 'LDA')

		Notes
		-----
		The folder structure enables:
		1. Systematic organization of analysis results
		2. Easy identification of analysis parameters from path
		3. Separation of different analysis types and configurations
		4. Consistent naming across different experiments

		The path is used by the parent FolderStructure class to
		determine where to save output files and figures.

		Examples
		--------
		Standard within-condition LDA analysis:
		
		>>> bdm.to_decode = 'stimulus'
		>>> bdm.elec_oi = 'all'
		>>> bdm.cross = False
		>>> bdm.classifier = 'LDA'
		>>> path_components = bdm.set_folder_path()
		>>> path_components  # ['bdm', 'stimulus', 'all_elecs']

		Cross-condition SVM analysis:
		
		>>> bdm.cross = True
		>>> bdm.classifier = 'SVM'
		>>> path_components = bdm.set_folder_path()
		>>> path_components  # ['bdm', 'stimulus', 'all_elecs', 'cross', 
		...						'SVM']

		Posterior electrode selection:
		
		>>> bdm.elec_oi = 'post'
		>>> path_components = bdm.set_folder_path()
		>>> 'post_elecs' in path_components  # True
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
		Apply sliding window transformation to increase feature space.
		
		Reformats data so that time point t includes information from
		features up to t-n where n is the window size. This increases
		the number of features by a factor of window_size, giving the
		classifier access to temporal context (Grootswagers et al. 2017)

		Parameters
		----------
		X : np.ndarray
			Input EEG data with shape:
			- (n_trials, n_channels, n_times) for 3D input
			- (n_freqs, n_trials, n_channels, n_times) for 4D input
		window_size : int, default=20
			Number of time points to include in sliding window.
		demean : bool, default=True
			Whether to subtract mean from each feature within the 
			sliding window (Hajonides et al. 2021).
		avg_window : bool, default=False
			If True, each time point reflects average activity within 
			the window rather than concatenated features.

		Returns
		-------
		np.ndarray
			Transformed data with shape:
			- (n_trials, n_channels*window_size, n_times) 
				if avg_window=False
			- (n_trials, n_channels, n_times) if avg_window=True
			For 4D input, frequency dimension is preserved.

		Notes
		-----
		Based on temporal decoding implementation by @jasperhajonides
		(github.com/jasperhajonides/temp_dec). The sliding window 
		approach improves decoding by providing temporal context around 
		each time point.

		Examples
		--------
		Increase feature dimension:
		
		>>> X_shape = (100, 64, 500)  # trials, channels, times
		>>> X_windowed = bdm.sliding_window(X, window_size=5)
		>>> X_windowed.shape  # (100, 320, 500)

		Average within window:
		
		>>> X_avg = bdm.sliding_window(X, window_size=5, 
										avg_window=True)
		>>> X_avg.shape  # (100, 64, 500)
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
		Reduce data dimensionality by averaging trials within 
		condition groups.
		
		Groups trials by unique combinations of labels and conditions, 
		then averages within each group. The number of trials per 
		average is controlled by the `tr_avg` parameter. This 
		preprocessing step can improve signal-to-noise ratio while 
		reducing computational load.

		Parameters
		----------
		epochs : mne.Epochs
			Epoched EEG data with shape (n_trials, n_channels, n_times).
		beh : pd.DataFrame
			Behavioral data containing condition and label information
			for each trial.
		beh_headers : list
			Column names in behavioral dataframe specifying:
			- [0]: decoding labels (target variable)
			- [1]: condition information (grouping variable, optional)
			If only one header provided, averaging is done across all 
			trials within each label.

		Returns
		-------
		tuple[mne.Epochs, pd.DataFrame]
			epochs : mne.Epochs
				Averaged epoched data with reduced trial count.
			beh : pd.DataFrame
				Updated behavioral dataframe corresponding to averaged 
				trials.

		Notes
		-----
		The averaging process:
		1. Groups trials by unique label-condition combinations
		2. Within each group, creates subgroups of size `self.tr_avg`
		3. Averages trials within each subgroup
		4. Updates behavioral dataframe to match new trial structure

		Examples
		--------
		Two labels with 4 observations each, average every 4 trials:
		
		>>> epochs.get_data().shape  # (8, 64, 240)
		>>> bdm.tr_avg = 4
		>>> epochs_avg, beh_avg = bdm.average_trials(epochs, beh, 
														['label'])
		>>> epochs_avg.get_data().shape  # (2, 64, 240)

		Two labels with 4 observations each, average every 3 trials:
		
		>>> epochs.get_data().shape  # (8, 64, 240) 
		>>> bdm.tr_avg = 3
		>>> epochs_avg, beh_avg = bdm.average_trials(epochs, beh, 
		...													['label'])
		>>> epochs_avg.get_data().shape  # (4, 64, 240)
		
		Warnings
		--------
		#TODO: Ensure trial averaging uses consistent subsets 
		# across runs.
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
		Extract condition-specific trial indices and labels for 
		classification.
		
		This method selects trials based on condition criteria and 
		prepares the data structure needed for train-test splitting. 
		It handles special cases like collapsed conditions and ensures 
		balanced class assignment.

		Parameters
		----------
		beh : pd.DataFrame
			Behavioral dataframe containing trial information and 
			labels.
		cnd_header : str
			Column name containing condition information.
		cnd : str
			Current condition for decoding. Special values:
			- 'all_data': Use all available trials
			- 'collapsed': Use previously collapsed conditions
			- Other: Specific condition name to filter by
		max_tr : list
			Maximum number of trials per class for balanced assignment.
			Updated when collapse=True.
		labels : str or list, default='all'
			Decoding labels to include. If 'all', uses all available 
			labels. If list, filters trials to only include specified 
			labels.
		collapse : bool, default=False
			Whether to mark current condition trials for later collapsed
			analysis. When True, adds 'collapsed' column to behavioral 
			data.

		Returns
		-------
		tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list]
			beh : pd.DataFrame
				Updated behavioral dataframe (may include 'collapsed' 
				column).
			cnd_idx : np.ndarray
				Trial indices for the selected condition.
			cnd_labels : np.ndarray
				Decoding labels for the selected trials.
			labels : np.ndarray
				Unique labels present in the selected data.
			max_tr : list
				Updated maximum trials per class (modified if 
				collapsed).

		Notes
		-----
		This method is a core component of the cross-validation pipeline
		, preparing data for `train_test_split` and `train_test_cross` 
		methods.
		
		The collapse functionality allows for:
		1. Marking trials from multiple conditions during initial passes
		2. Later analysis of the combined 'collapsed' condition set
		3. Automatic rebalancing of trial counts for collapsed 
			conditions

		Examples
		--------
		Select trials from specific condition:
		
		>>> (beh, cnd_idx, cnd_labels, 
		...     labels, max_tr) = bdm.get_condition_labels(
		...     beh, 'condition', 'cond_A', [50], labels='all')
		>>> len(cnd_idx)  # Number of trials in condition A

		Mark trials for collapsed analysis:
		
		>>> (beh, cnd_idx, cnd_labels, 
		... labels, max_tr = bdm.get_condition_labels(
		...     beh, 'condition', 'cond_A', [50], collapse=True)
		>>> 'collapsed' in beh.columns  # True

		Use previously collapsed trials:
		
		>>> (beh, cnd_idx, cnd_labels, 
		...     labels, max_tr) = bdm.get_condition_labels(
		...     beh, 'condition', 'collapsed', [50])
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
			max_tr = [self.select_max_trials(beh, ['yes'], labels,'collapsed')]
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
		Create balanced train-test splits for k-fold cross-validation.
		
		Generates stratified k-fold splits ensuring balanced class 
		representation in both training and test sets. Each fold uses a 
		different subset for testing while maintaining equal class 
		distribution across folds.

		Parameters
		----------
		idx : np.ndarray
			Trial indices corresponding to the decoding labels.
		labels : np.ndarray
			Array of class labels for each trial in idx.
		max_tr : int
			Maximum number of trials per class to ensure balanced 
			sampling.
		bdm_info : dict
			Cross-validation information dictionary. If empty ({}), 
			random trial selection is performed. If populated, 
			uses specified trial assignments for replication.

		Returns
		-------
		tuple[np.ndarray, np.ndarray, dict]
			train_tr : np.ndarray
				Training trial indices with shape (n_folds, n_classes,
				n_train_trials).
				Each fold contains balanced training sets for each 
				class.
			test_tr : np.ndarray
				Test trial indices with shape (n_folds, n_classes, 
				n_test_trials).
				Each fold contains balanced test sets for each class.
			bdm_info : dict
				Updated cross-validation information containing trial 
				assignments for each class and fold. Can be saved for 
				exact replication.

		Notes
		-----
		The splitting strategy:
		1. Ensures balanced class representation in all folds
		2. Tests each trial exactly once across all folds
		3. Maintains consistent train/test ratios (e.g., 90%/10% for 
			10-fold CV)
		4. Supports reproducible splits via bdm_info parameter

		The number of folds is controlled by the `self.nr_folds` 
		parameter, with more folds providing larger training sets but 
		smaller test sets.

		Examples
		--------
		Create 10-fold cross-validation splits:
		
		>>> self.nr_folds = 10
		>>> idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		>>> labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
		>>> (train_tr, test_tr, 
		... bdm_info) = bdm.train_test_split(idx, labels, 5, {})
		>>> train_tr.shape  # (10, 2, 4) - 10 folds, 2 classes, 
											4 training trials
		>>> test_tr.shape   # (10, 2, 1) - 10 folds, 2 classes, 
											1 test trial

		Replicate previous split:
		
		>>> # Use bdm_info from previous run for exact replication
		>>> train_tr_rep, test_tr_rep, _ = bdm.train_test_split(
		...     idx, labels, 5, bdm_info)
		>>> np.array_equal(train_tr, train_tr_rep)  # True
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
		Create independent training and test sets for cross-condition 
		analysis.
		
		This method implements cross-condition decoding by creating 
		balanced training and test sets from different experimental 
		conditions. Unlike standard cross-validation, training and 
		testing are performed on completely independent condition sets.

		Parameters
		----------
		X : np.ndarray
			Input EEG data with shape:
			- (n_trials, n_channels, n_times) for 3D data
			- (n_freqs, n_trials, n_channels, n_times) for 4D data
		y : np.ndarray
			Decoding labels corresponding to each trial in X.
		train_idx : np.ndarray
			Trial indices for training set (condition-specific subset).
		test_idx : np.ndarray or bool
			Trial indices for test set. If False, only training data
			is prepared (test arrays will be empty).
		max_tr : int, optional
			Maximum trials per class for balanced sampling. If None,
			determined automatically from minimum class count.

		Returns
		-------
		tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
			Xtr : np.ndarray
				Training data with shape:
				- (n_folds, n_trials_train, n_channels, n_times) for 3D 
					input
				- (n_folds, n_freqs, n_trials_train, n_channels, 
					n_times) for 4D input
			Xte : np.ndarray
				Test data with same structure as Xtr but for 
					test trials. Empty if test_idx=False.
			Ytr : np.ndarray
				Training labels with shape (n_folds, n_trials_train).
			Yte : np.ndarray
				Test labels with shape (n_folds, n_trials_test).
				Empty if test_idx=False.

		Notes
		-----
		This method is essential for cross-condition generalization 
		analysis, where the classifier is trained on one condition and 
		tested on another. This approach tests whether learned patterns 
		generalize across different experimental contexts.

		The balancing strategy:
		1. Determines minimum class count across both train and test 
			sets
		2. Randomly samples equal numbers of trials from each class
		3. Creates multiple balanced folds for robust estimation
		4. Ensures no overlap between training and test conditions

		Examples
		--------
		Cross-condition decoding (train on condition A, 
		test on condition B):

		>>> train_idx = np.where(beh['condition'] == 'A')[0]
		>>> test_idx = np.where(beh['condition'] == 'B')[0] 
		>>> Xtr, Xte, Ytr, Yte = bdm.train_test_cross(
		...     X, y, train_idx, test_idx)
		>>> Xtr.shape  # (n_folds, n_trials_per_class*n_classes,
		...  				n_channels, n_times)

		Training-only preparation:

		>>> Xtr, _, Ytr, _ = bdm.train_test_cross(X, y, train_idx, 
		...     												False)
		>>> Xte.size == 0  # True - no test data prepared
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
		Extract training and test data based on cross-validation fold 
		indices.
		
		This method takes the trial indices generated by 
		train_test_split anduses them to extract the corresponding data 
		and labels for each fold. It properly handles both 3D and 4D 
		data formats and organizes themfor efficient cross-validation 
		training.

		Parameters
		----------
		X : np.ndarray
			Input EEG data with shape:
			- (n_trials, n_channels, n_times) for 3D data  
			- (n_freqs, n_trials, n_channels, n_times) for 4D data
		Y : np.ndarray
			Decoding labels corresponding to each trial in X.
		train_tr : np.ndarray
			Training trial indices with shape (n_folds, n_classes, 
			n_train_trials).Output from train_test_split method.
		test_tr : np.ndarray
			Test trial indices with shape (n_folds, n_classes, 
			n_test_trials). Output from train_test_split method.

		Returns
		-------
		tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
			Xtr : np.ndarray
				Training data organized by folds:
				- (n_folds, n_train_total, n_channels, n_times)
				 	for 3D input
				- (n_folds, n_freqs, n_train_total, n_channels, n_times) 
					for 4D input where 
					n_train_total = n_classes * n_train_trials
			Xte : np.ndarray
				Test data with same structure as Xtr but for
				  test trials.
			Ytr : np.ndarray
				Training labels with shape (n_folds, n_train_total).
			Yte : np.ndarray
				Test labels with shape (n_folds, n_test_total).

		Notes
		-----
		This method bridges the gap between cross-validation index 
		generation(train_test_split) and actual data organization 
		for machine learning.
		
		Key operations:
		1. Flattens class-wise trial indices into fold-wise arrays
		2. Extracts corresponding data trials using advanced indexing
		3. Organizes labels to match the flattened data structure
		4. Handles both 3D (time-domain) and 4D (time-frequency) data

		The output arrays are ready for direct use with scikit-learn
		classifiers in the cross-validation loop.

		Examples
		--------
		Standard 3D data processing:
		
		>>> X.shape  # (200, 64, 500) - trials, channels, times  
		>>> train_tr.shape  # (10, 2, 18) - folds, classes, train trials
		>>> test_tr.shape   # (10, 2, 2) - folds, classes, test trials
		>>> (Xtr, Xte, 
		... Ytr, Yte) = bdm.train_test_select(X, Y, train_tr, test_tr)
		>>> Xtr.shape  # (10, 36, 64, 500) - folds, total_train_trials, 
		... 								channels, times
		>>> Ytr.shape  # (10, 36) - folds, total_train_trials

		Time-frequency data (4D):
		
		>>> X.shape  # (5, 200, 64, 500) - freqs, trials, 
		... 								channels, times
		>>> Xtr, Xte, Ytr, Yte = bdm.train_test_select(X, Y, train_tr, 
		... 													test_tr)
		>>> Xtr.shape  # (10, 5, 36, 64, 500) - folds, freqs, trials, 
		... 											channels, times
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

	def cross_time_decoding(self, Xtr: np.ndarray, Xte: np.ndarray, 
		Ytr: np.ndarray, Yte: np.ndarray, 
		labels: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
		GAT: Union[bool, Tuple[Tuple[float, float], 
						 Tuple[float, float]]] = False, 
		X: Union[List, np.ndarray] = []
		) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Perform multivariate decoding across time with optional 
		GAT analysis.
		
		Executes the core classification pipeline for each time point 
		and fold,supporting both within-time decoding and 
		Generalization Across Time (GAT).Handles standardization, 
		PCA dimensionality reduction, and classifier training/testing 
		for comprehensive temporal decoding analysis.

		Parameters
		----------
		Xtr : np.ndarray
			Training data with shape:
			- (n_folds, n_trials, n_features, n_times) for 3D input
			- (n_folds, n_freqs, n_trials, n_features, n_times) for 
				4D input
			Data should be pre-sliced to desired time windows.
		Xte : np.ndarray
			Test data with same format as Xtr. Should be pre-sliced to
			desired time windows for testing.
		Ytr : np.ndarray
			Training labels with shape (n_folds, n_trials).
		Yte : np.ndarray  
			Test labels with shape (n_folds, n_trials).
		labels : np.ndarray or tuple of np.ndarray
			Label set(s) for training and testing:
			- Single array: same labels for both training and testing
			- Tuple: (train_labels, test_labels) for different label 
				sets
		GAT : bool or tuple, default=False
			Generalization Across Time configuration:
			- False: Within-time decoding (train and test at same time 
				points)
			- True: Full GAT (train at each time, test at all times)
			- Tuple: Custom time windows - data should be pre-sliced
		X : list or np.ndarray, default=[]
			Optional raw data for PCA computation when using 'across' 
			mode. Should match the dimensionality of Xtr/Xte.

		Returns
		-------
		tuple[np.ndarray, np.ndarray, np.ndarray]
			class_acc : np.ndarray
				Classification performance with shape (n_folds, n_freqs, 
				n_time_train, n_time_test). For GAT=False, n_time_test=1.
			weights : np.ndarray
				Classifier weights with shape (n_folds, n_freqs, 
				n_time_train,n_time_test, n_features).
			conf_matrix : np.ndarray
				Confusion matrices with shape (n_folds, n_freqs, 
				n_time_train, n_time_test, n_labels_test, 
				n_labels_train).

		Notes
		-----
		**Preprocessing Pipeline:**
		1. Automatic dimensionality detection (3D vs 4D input)
		2. Optional standardization (if self.scale=True)
		3. Optional PCA dimensionality reduction (if self.pca_components
		 	set)
		4. Classifier training and testing

		**GAT Analysis Types:**
		- **Within-time**: Classifier trained and tested at same time 
			points
		- **Full GAT**: Train at time t, test at all time points
		- **Custom GAT**: Train/test on pre-specified time windows

		**Performance Metrics:**
		Uses the metric specified in self.metric ('auc' or 'acc').
		For 'acc', implements balanced accuracy for robust class 
			evaluation.

		**Memory Considerations:**
		For large datasets, consider using PCA to reduce feature 
		dimensionalityand prevent memory overflow during GAT analysis.

		Examples
		--------
		Within-time decoding:
		
		>>> class_acc, weights, conf_mat = bdm.cross_time_decoding(
		...     Xtr, Xte, Ytr, Yte, labels, GAT=False)
		>>> class_acc.shape  # (n_folds, n_freqs, n_times, 1)

		Full GAT analysis:
		
		>>> class_acc, weights, conf_mat = bdm.cross_time_decoding(
		...     Xtr, Xte, Ytr, Yte, labels, GAT=True)  
		>>> class_acc.shape  # (n_folds, n_freqs, n_times, n_times)

		Cross-condition decoding with different label sets:
		
		>>> train_labels = np.array(['A', 'B'])
		>>> test_labels = np.array(['C', 'D']) 
		>>> class_acc, weights, conf_mat = bdm.cross_time_decoding(
		...     Xtr, Xte, Ytr, Yte, (train_labels, test_labels))

		Time-frequency analysis (4D input):
		
		>>> # Input: (folds, freqs, trials, channels, times)
		>>> class_acc, weights, conf_mat = bdm.cross_time_decoding(
		...     Xtr_4d, Xte_4d, Ytr, Yte, labels)
		>>> class_acc.shape  # (n_folds, n_freqs, n_times, n_test_times)

		Raises
		------
		ValueError
			If input arrays are not 4D or 5D, or if dimensions are 
			incompatible.

		See Also
		--------
		select_classifier : Initialize the classifier
		compute_class_perf : Calculate classification performance  
		"""

		# set necessary parameters
		N = self.nr_folds
		nr_time_tr = Xtr.shape[-1]
		nr_time_te = Xte.shape[-1] if GAT else 1

		# Handle 3D vs 4D input	
		if Xtr.ndim == 4: # 3D data:[folds × trials × features × time]
			nr_freq = 1
			# Insert freq dimension at position 1
			Xtr = np.expand_dims(Xtr, axis=1)  
			Xte = np.expand_dims(Xte, axis=1)
		elif Xtr.ndim == 5: # 4D data:[folds × freq × trials × features × time]
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
			print(f'\rFold {n+1} out of {N} folds '
				f'in run {self.run_info}', end='')
			Ytr_ = Ytr[n]
			Yte_ = Yte[n]

			for freq in range(nr_freq):	
				if self.tfr is not None:
					print(f'\rFrequency {freq+1} out of {nr_freq} '
						f'in run {self.run_info}', end='')
				for tr_t in range(nr_time_tr):
					# When GAT=False, we only need one iteration
					te_time_points = range(nr_time_te) if GAT else [0]
					for te_t in te_time_points:

						Xtr_ = Xtr[n,freq,:,:,tr_t]
						Xte_ = Xte[n, freq, :, :, te_t if GAT else tr_t]

						# Apply standardization if specified
						if self.scale:
							scaler = StandardScaler(
												with_std=True)
							Xtr_ = scaler.fit_transform(Xtr_)
							Xte_ = scaler.transform(Xte_)

						if self.pca_components[0]:
							# Show warning only once if data is not standardized
							first_iter= (tr_t == 0 and n == 0 and 
							 								self.run_info == 1)
							if not self.scale and first_iter:
								warnings.warn('It is recommended to ' \
								'standardize the data before applying PCA ' \
								'correction')
							
							if self.pca_components[1] == 'across':
								pca = PCA(n_components=self.pca_components[0])
								Xtr_ = pca.fit_transform(Xtr_)
								Xte_ = pca.transform(Xte_)
							elif self.pca_components[1] == 'all':
								if X != []:
									if X.ndim == 3:
										X_ = X[..., tr_t]
									else:
										X_ = X[freq, ..., tr_t]

									if self.scale:
										X_ = StandardScaler(
											with_std=True).fit_transform(X_)
									pca = PCA(
										n_components=self.pca_components[0])
									pca.fit(X_)
									Xtr_ = pca.fit_transform(Xtr_)
									Xte_ = pca.transform(Xte_)

										
						# Train and test classifier
						clf.fit(Xtr_,Ytr_)
						# Get posterior probability estimates
						scores = clf.predict_proba(Xte_) 
						predict = clf.predict(Xte_)

						# Compute performance metrics
						if bool(set(Ytr_)& set(Yte_)):
							class_perf = self.compute_class_perf(scores, Yte_, 
										  np.unique(Ytr_), predict) #
							conf_m = confusion_matrix(Yte_, predict,
								 							labels=labels[0])
						else:
							class_perf = 0
							conf_m = self.get_fake_confusion_matrix(Yte_,
											   						 predict) 

						# store results
						# TODO: create interpretable weights
						class_acc[n, freq, tr_t, te_t] = class_perf
						conf_matrix[n, freq, tr_t, te_t] = conf_m
						if not self.pca_components[0]:
							weights[n, freq, tr_t, te_t] = (
								self.get_classifier_weights(clf, Xtr_))
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

	def compute_class_perf(self, scores: np.ndarray, true_labels: np.ndarray, 
						 label_order: list, predict: np.ndarray) -> float:
		"""
		Compute classification performance using specified metric.
		Calculates classifier performance based on test scores and true
		labels. Supports both accuracy and AUC metrics, with special
		handling for multi-class problems using pairwise AUC 
		comparisons.

		Parameters
		----------
		scores : array-like
			Classifier confidence scores with shape:
			- (n_samples,) for binary classification
			- (n_samples, n_classes) for multi-class classification
		true_labels : array-like
			True class labels for test samples with shape (n_samples,).
		label_order : list
			Ordered list of class labels corresponding to score columns.
			Defines the mapping from score indices to actual labels.
		predict : array-like
			Predicted class labels with shape (n_samples,). Used when
			metric is 'acc' for accuracy calculation.

		Returns
		-------
		float
			Classification performance score:
			- For 'auc': Average pairwise AUC (0.0 to 1.0)
			- For 'acc': Balanced accuracy (0.0 to 1.0)

		Notes
		-----
		**AUC Calculation:**
		For multi-class problems, computes all pairwise AUC values 
		between classes and returns the average. This provides a robust 
		performance measure that is threshold-independent.

		The algorithm:
		1. Convert true labels to indices based on label_order
		2. Generate all pairwise class combinations
		3. For each pair, compute binary AUC using relevant scores
		4. Return mean AUC across all pairs

		**Balanced Accuracy Calculation:**
		Computes accuracy for each class separately, then averages 
		across classes. This provides robust performance assessment that 
		is not biased by class imbalances or classifier preferences:

		balanced_accuracy = mean([acc_class1, acc_class2, ...])

		Where acc_classi = (correct_predictions_classi) / (total_classi)

		This measure:
		- Returns ~0.5 for chance performance regardless of class 
			distribution
		- Detects classifier bias toward specific classes
		- Provides fair comparison across datasets with different class 
			ratios

		Examples
		--------
		Binary classification with AUC:
		
		>>> scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
		>>> true_labels = np.array(['A', 'B', 'A'])
		>>> label_order = ['A', 'B']
		>>> predict = np.array(['A', 'B', 'A'])
		>>> bdm.metric = 'auc'
		>>> perf = bdm.compute_class_perf(scores, true_labels, 
		... 	label_order, predict)
		>>> perf  # AUC value between 0.5 and 1.0

		Multi-class classification with accuracy:
		
		>>> scores = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
		>>> true_labels = np.array(['A', 'B'])
		>>> predict = np.array(['A', 'B'])
		>>> bdm.metric = 'acc'
		>>> perf = bdm.compute_class_perf(scores, true_labels, 
		... 	label_order, predict)
		>>> perf  # 1.0 (perfect accuracy)

		Binary classification (1D scores):
		
		>>> scores = np.array([0.8, 0.3, 0.9])  # Single column
		>>> true_labels = np.array([1, 0, 1])
		>>> label_order = [0, 1]
		>>> bdm.metric = 'auc'
		>>> perf = bdm.compute_class_perf(scores, true_labels, 
		... 	label_order, predict)
		>>> perf  # Automatically handles 1D to 2D conversion

		See Also
		--------
		score_auc : Computes AUC for binary classification
		"""

		if self.metric == 'auc':
			
			# shift true_scores to indices
			true_labels = np.array([list(label_order).index(l) 
						   								for l in true_labels])
			# check whether it is a more than two class problem
			if scores.ndim > 1:
				nr_class = scores.shape[1]
			else:
				scores = np.reshape(scores, (-1,1)) 
				nr_class = 2	

			# select all pairwise combinations of classes
			pairs = list(itertools.combinations(range(nr_class), 2))
			# do this both ways in case of multi class problem
			if len(pairs) > 1: 
				pairs += [p[::-1] for p in pairs]

			# initiate AUC
			auc = np.zeros(len(pairs))	

			# loop over all pairwise combinations
			for i, comp in enumerate(pairs):
				# grab two classes and set labels to fals
				pair_idx = np.logical_or(true_labels == comp[0], 
							 			true_labels == comp[1]) 	
				bool_labels = np.zeros(true_labels.size, dtype = bool) 	
				# set labels of positive class to True
				bool_labels[true_labels == comp[0]] = True 	
				# compute AUC for this pair			
				labels_2_use = bool_labels[pair_idx]					
				scores_2_use = scores[pair_idx,comp[0]]					
				auc[i] = self.score_auc(labels_2_use, scores_2_use)		

			class_perf = np.mean(auc)

		elif self.metric == 'acc':
			# Balanced accuracy: average per-class accuracy
			unique_labels = np.unique(true_labels)
			class_accuracies = []
			
			for label in unique_labels:
				class_mask = true_labels == label
				if np.sum(class_mask) > 0:  # Avoid division by zero
					class_correct = np.sum(predict[class_mask] == 
													true_labels[class_mask])
					class_total = np.sum(class_mask)
					class_acc = class_correct / class_total
					class_accuracies.append(class_acc)
			
			class_perf = np.mean(class_accuracies)  # Balanced accuracy
				
		return class_perf
		

	def score_auc(self, labels: np.ndarray, scores: np.ndarray) -> float:
		"""
		Calculate Area Under the Curve (AUC) for binary classification.
		
		Computes AUC using a rank-based approach that counts 
		positive-negative pairs. AUC represents the probability that a 
		randomly chosen positive sample scores higher than a randomly 
		chosen negative sample, providing a threshold-independent 
		measure of classifier performance.

		Parameters
		----------
		labels : array-like
			Binary labels with shape (n_samples,). Should contain 
			boolean values or be convertible to boolean 
			(0/1, True/False).
		scores : array-like
			Classification scores with shape (n_samples,). Higher scores
			should indicate higher probability of positive class.

		Returns
		-------
		float
			Area under the ROC curve, ranging from 0.0 to 1.0:
			- 0.5: Random performance (no discrimination)
			- 1.0: Perfect classification 
			- 0.0: Perfect but inverted classification

		Raises
		------
		AssertionError
			If all labels are of the same class (no positive or no 
			negative labels), making AUC calculation impossible.

		Notes
		-----
		This implementation uses the Wilcoxon-Mann-Whitney U statistic
		approach, which is equivalent to AUC but computed efficiently
		using rank statistics:

		AUC = (∑ranks[positives] - n_pos(n_pos + 1)/2) / (n_pos × n_neg)

		where ranks are computed across all scores. This method is
		computationally efficient and numerically stable.

		The function is based on the ADAM toolbox implementation and
		follows the approach described in Fawcett (2006).

		Examples
		--------
		Perfect classification:
		
		>>> labels = np.array([0, 0, 1, 1])
		>>> scores = np.array([0.1, 0.2, 0.8, 0.9])
		>>> auc = bdm.score_auc(labels, scores)
		>>> auc  # 1.0

		Random performance:
		
		>>> labels = np.array([0, 1, 0, 1])
		>>> scores = np.array([0.4, 0.3, 0.6, 0.7])
		>>> auc = bdm.score_auc(labels, scores)
		>>> auc  # ≈ 0.5

		Boolean labels:
		
		>>> labels = np.array([False, False, True, True])
		>>> scores = np.array([0.2, 0.3, 0.7, 0.8])
		>>> auc = bdm.score_auc(labels, scores)
		>>> auc  # 1.0

		References
		----------
		Fawcett, T. (2006). An introduction to ROC analysis. Pattern 
		Recognition Letters, 27(8), 861-874.
		"""

		num_pos = np.sum(labels)
		num_neg = labels.size - num_pos

		assert num_pos != 0,'no positive labels entered in AUC calculation'
		assert num_neg != 0,'no negative labels entered in AUC calculation'

		rank = rankdata(scores) 
		sum_positive_ranks = np.sum(rank[labels]) - num_pos * (num_pos + 1) / 2
		auc = sum_positive_ranks / (num_pos * num_neg)

		return auc


	def select_max_trials(self, df: pd.DataFrame, cnds: list, 
					   	bdm_labels: Union[str, list] = 'all', 
						cnds_header: str = 'condition') -> int:
		"""
		Determine maximum balanced trial count for cross-validation.
		
		Calculates the maximum number of trials per class that allows 
		for balanced sampling across all conditions and k-fold 
		cross-validation. Ensures equal representation of each decoding 
		label across conditions by finding the minimum available trials 
		and making them divisible by the number of folds.

		Parameters
		----------
		df : pd.DataFrame
			Behavioral dataframe containing trial information and 
			labels.
		cnds : list
			List of conditions for decoding analysis. Special value
			['all_data'] uses all available trials regardless of 
			condition.
		bdm_labels : list or str, default='all'
			Decoding labels to include in analysis. If 'all', uses all
			unique labels found in the target column.
		cnds_header : str, default='condition'
			Column name containing condition information in behavioral 
			data.

		Returns
		-------
		int
			Maximum number of trials per class that ensures:
			- Balanced representation across all conditions
			- Divisible by number of folds for clean cross-validation
			- Equal class representation within each condition

		Notes
		-----
		The algorithm:
		1. For each condition, counts trials per decoding label
		2. Finds minimum trial count across labels within each condition
		3. Makes count divisible by n_folds (floor division)
		4. Returns minimum across all conditions for balanced sampling

		This ensures that cross-validation can proceed with equal class
		representation in every fold, which is critical for unbiased
		performance estimation.

		Examples
		--------
		Balance across multiple conditions:
		
		>>> beh = pd.DataFrame({
		...     'condition': ['A', 'A', 'A', 'B', 'B', 'B'],
		...     'target': ['left', 'left', 'right', 'left', 'right', 
		...     'right']})
		>>> bdm.nr_folds = 2
		>>> max_trials = bdm.select_max_trials(beh, ['A', 'B'], ['left', 
		...		'right'])
		>>> max_trials  # 2 (limited by condition A having only 2
		...	  'left' trials)

		Use all available data:
		
		>>> max_trials = bdm.select_max_trials(beh, ['all_data'])
		>>> max_trials  # Uses minimum class count across entire dataset

		Custom label subset:
		
		>>> max_trials = bdm.select_max_trials(beh, ['A'], ['left'])
		>>> max_trials  # Only considers 'left' trials in condition A

		Warnings
		--------
		If max_trials returns 0, indicates insufficient data for the 
		current number of folds. Consider reducing n_folds or collecting 
		more data.
		"""

		# make sure selection is based on corrrect trials
		if bdm_labels == 'all':
			bdm_labels = np.unique(df[self.to_decode]) 

		N = self.nr_folds
		cnd_min = []

		# trials for decoding
		if cnds != ['all_data']:
			for cnd in cnds:
		
				# select condition trials and get their decoding labels
				trials = np.where(df[cnds_header] == cnd)[0]
				labels = [l for l in df[self.to_decode][trials] 
			  											if l in bdm_labels]

				# select the minimum number of trials per label for 
				# BDM procedure
				# TODO: ADD OPTION FOR UNBALANCING
				min_tr = np.unique(labels, return_counts = True)[1]
				min_tr = int(np.floor(min(min_tr)/N)*N)	

				cnd_min.append(min_tr)

			max_trials = min(cnd_min)
		elif cnds == ['all_data']:
			labels = [l for l in beh[self.to_decode] if l in bdm_labels]
			min_tr = np.unique(labels, return_counts = True)[1]
			max_trials = int(np.floor(min(min_tr)/N)*N)	

		if max_trials == 0:
			print('At least one condition does not contain sufficient ' \
			'info for current nr of folds')

		return max_trials

