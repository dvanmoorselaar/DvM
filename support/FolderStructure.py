import os
import mne
import pickle

import pandas as pd
from support import *
from IPython import embed



class FolderStructure(object):
	'''
	Creates the folder structure
	'''
	 
	def __init__(self):
		pass 	

	def FolderTracker(self, extension = [], filename = '', overwrite = True):
		'''
		FolderTracker creates folder address. At the same time it 
		checks whether the specific folder already exists (if not it is created)

		Arguments
		- - - - - 
		extension (list): list of subfolders that are attached to current working directory
		filename (str): name of file
		overwrite (bool): if overwrite is False, an * is added to the filename 

		Returns
		- - - -
		folder (str): file adress

		'''				

		# create folder adress
		folder = os.getcwd()
		if extension != []:
			folder = os.path.join(folder,*extension)	

		# check whether folder exists
		if not os.path.isdir(folder):
			os.makedirs(folder)

		if filename != '':	
			if not overwrite:
				while os.path.isfile(os.path.join(folder,filename)):
					filename = filename[:-4] + '*' + filename[-4:]
			folder = os.path.join(folder,filename)
			
		return folder	


	def loadData(self, sj, eye_window, eyefilter, eye_ch = 'HEOG'):
		'''
		loads EEG and behavior data

		Arguments
		- - - - - 
		sj (int): subject number
		eye_window (tuple|list): timings to scan for eye movements
		eyefilter (bool): in or exclude eye movements based on step like algorythm
		eye_ch (str): name of channel to scan for eye movements

		Returns
		- - - -
		beh (Dataframe): behavior file
		eeg (mne object): preprocessed eeg data

		'''

		# read in processed behavior from pickle file
		beh = pickle.load(open(self.FolderTracker(extension = ['beh','processed'], 
							filename = 'subject-{}_all.pickle'.format(sj)),'rb'))
		beh = pd.DataFrame.from_dict(beh)

		# read in processed EEG data
		eeg = mne.read_epochs(self.FolderTracker(extension = ['processed'], 
							filename = 'subject-{}_all-epo.fif'.format(sj)))

		if eyefilter:
			filter_eye(beh, eeg, eye_window, eye_ch)



		return beh, eeg


