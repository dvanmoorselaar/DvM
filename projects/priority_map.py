import os
import mne
import sys
import glob
import pickle
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
sys.path.append('/Users/dvm/DvM')

import numpy as np
import seaborn as sns

from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import * 

from support.FolderStructure import *
from support.support import *

# subject specific info
sj_info = {'1': {'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			'2': {'replace':{}},
			'3': {'replace':{}},
			'4': {'replace':{}},
			'5': {'replace':{}},
			'6': {'replace':{}},
			'7': {'replace':{}},
			'8': {'replace':{}},
			'9': {'replace':{}},
			'10': {'replace':{}},
			'11': {'replace':{}},
			'12': {'replace':{}},
			'13': {'replace':{}},
			'14': {'replace':{}},
			'15': {'replace':{}},
			'16': {'replace':{}},
			'17': {'replace':{}},
			'18': {'replace':{}},
			'19': {'replace':{}},
			'20': {'replace':{}},
			'21': {'replace':{}},
			'22': {'replace':{}},
			'23': {'replace':{}},
			'24': {'replace':{}},
							}

# project specific info
project = 'Ping'
factors = []
labels = []
to_filter = ['RT'] 
project_param = ['nr_trials','trigger','RT', 'subject_nr', 'block_cnt', 'practice',
				'block_type', 'correct','dist_high','dist_loc','dist_shape', 
				'dist_color', 'high_prob_loc', 'target_high','target_loc','target_shape',
				'target_color','ping','ping_trigger','ping_type','block_cnt','trial_type']


# eeg info (event_id specified below)
ping = 'no_ping' # 'ping'
part = 'beh'
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
eeg_runs = [1]
# ping parameters
t_min = -0.7 
t_max = 0.6
#event_id = [9, 100, 102, 104, 106]
# search parameters
#t_min = -0.1 
#t_max = 0.6
#event_id = [12,13,14,15,16,17,18,19,21,23,24,25,26,27,28,29,31,32,34,35,36,37,38,39,41,42,43,45,46,\
#			47,48,49,51,52,53,54,56,57,58,59,61,62,63,64,65,67,68,69,71,72,73,74,75,76,78,79,81,82,83,84,85,86,87,89]

flt_pad = 0.5
binary =  0

# eye tracker info
tracker_ext = 'asc'
eye_freq = 1000
start_event = 'Onset ping' # start_event = 'Onset search'  
tracker_shift = 0
viewing_dist = 60 
screen_res = (1680, 1050) 
screen_h = 29

class priorityMap(FolderStructure):

	def __init__(self): pass


	def prepareBEH(self, project, part, factors, labels, project_param, to_filter):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and correct == 1', cnd_sel = False, save = True)
		PP.exclude_outliers(criteria = dict(RT = 'RT_filter == True', correct = ''))
		PP.save_data_file()

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, event_id, project_param, project_folder, binary, channel_plots, inspect):
		'''
		EEG preprocessing as preregistred @
		'''

		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']
		log_file = self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocessing_param.csv')

		# start logging
		logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename= self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocess_sj{}_ses{}.log'.format(
                        sj, session), overwrite = False),
                    filemode='w')

		# READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
		EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
		                                  preload=True, eog=eog) for run in eeg_runs])

		#EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['V_do','H_l','Ref_r','Ref_l','EXG7','EXG8'])
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEGica = EEG.copy()
		EEGica.filter(h_freq=None, l_freq=1,
		                   fir_design='firwin', skip_by_annotation='edge')
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		            skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(event_id, binary=binary, min_duration=0)
		beh, missing = EEG.matchBeh(sj, session, events, event_id, 
		                             headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=event_id,
		        tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, reject_by_annotation = True) 

		# AUTMATED ARTIFACT DETECTION
		epochs.selectBadChannels(run_ransac = True, channel_plots = False, inspect = True, RT = None)  
		z = epochs.artifactDetection(z_thresh=4, band_pass=[110, 140], plot=True, inspect=True)

		# ICA
		epochs.applyICA(EEGica, method='picard', fit_params = dict(ortho=False, extended=True), inspect = True)
		del EEGica

		# EYE MOVEMENTS
		epochs.detectEye(missing, events, beh.shape[0], time_window=(t_min*1000, t_max*1000), 
						tracker_shift = tracker_shift, start_event = start_event, 
						extension = tracker_ext, eye_freq = eye_freq, 
						screen_res = screen_res, viewing_dist = viewing_dist, 
						screen_h = screen_h)

		# INTERPOLATE BADS
		bads = epochs.info['bads']   
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, event_id)

		logPreproc((sj, session), log_file, nr_sj = len(sj_info.keys()), nr_sessions = nr_sessions, 
					to_update = dict(nr_clean = len(epochs), z_value = z, nr_bads = len(bads), bad_el = bads))

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/Users/dvm/Desktop/Ping'
	os.chdir(project_folder)

	# initiate current project
	PO = priorityMap()

	#Run preprocessing 
	PO.prepareBEH(project, part, factors, labels, project_param, to_filter)

	#Run preprocessing EEG
	PO.prepareEEG(sj = 2, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
			t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
			event_id = event_id, project_param = project_param, 
			project_folder = project_folder, binary = binary, 
			channel_plots = True, inspect = True)
			