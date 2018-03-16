"""
analyze EEG data

Created by Dirk van Moorselaar on 25-08-2017.
Copyright (c) 2015 DvM. All rights reserved.
"""

import mne
import pickle
import numpy as np

from IPython import embed as shell
from scipy.stats import ttest_rel

def eeg_reader(subject_id, sessions = 2):
	'''

	'''

	# get eeg data
	eeg = []
	for session in range(sessions):
		eeg.append(mne.read_epochs('/Users/dirk/Desktop/suppression/processed/subject-{}_ses-{}-epo.fif'.format(subject_id,session + 1)))

		# extract other parameters
		if session == 0:
			times, ch_names, info = eeg[0].times, eeg[0].ch_names, eeg[0].info

	eeg = np.vstack(eeg)

	# get behavior data
	with open('/Users/dirk/Desktop/suppression/beh/processed/subject-{}_all.pickle'.format(subject_id),'rb') as handle:
		beh = pickle.load(handle)
	
	return eeg, beh, times, ch_names, info	

def paired_t(*args):
	'''
	Call scipy.stats.ttest_rel, but return only f-value
	'''
	
	return ttest_rel(*args)[0]

