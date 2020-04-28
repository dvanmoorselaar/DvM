#import matplotlib
#matplotlib.use('agg') # now it works via ssh connection

import os
import mne
import sys
import glob
import pickle
import logging
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM_3')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed
from beh_analyses.PreProcessing import *
#from eeg_analyses.TF import * 
from eeg_analyses.EEG import * 
#from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info

sj_info = {'1': {'tracker': (False, '...', 0, '',0), 'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
		   '2': {'tracker': (False, '...', 0, '',0), 'replace':{}},}


# project specific info
project = 'color_decoding'
part = 'beh'
factors = []
labels = []
to_filter = [] 
project_param = ['practice','nr_trials','trigger','RT', 'subject_nr',
				'block_type', 'correct','dist_loc','dist_orient','target_loc','target_orient', 
				'block_cnt', 'dist_color', 'target_color']

montage = mne.channels.read_montage(kind='biosemi64')
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
trigger = [12,13,14,15,16,21,23,24,25,26,31,32,34,35,36,41,42,43,45,46,51,52,53,54,56,61,62,63,64,65,101,102,103,104,105,106]
t_min = -0.75
t_max = 0.55
flt_pad = 0.5
eeg_runs = [1]
binary =  0

# eye tracker info
tracker_ext = 'asc'
eye_freq = 500
start_event = 'Onset search'
tracker_shift = 0
viewing_dist = 100 
screen_res = (1680, 1050) 
screen_h = 29

class color_decoding(FolderStructure):

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
		PP.prep_JASP(agg_func = 'mean', voi = 'RT', data_filter = 'RT_filter == True', save = True)
		PP.save_data_file()

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, channel_plots, inspect):
		'''
		EEG preprocessing as preregistred @ https://osf.io/b2ndy/register/5771ca429ad5a1020de2872e
		'''

		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']
		tracker, ext, t_freq, start_event, shift = sj_info[str(sj)]['tracker']

		# start logging
		logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename= self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocess_sj{}_ses{}.log'.format(
                        sj, session), overwrite = False),
                    filemode='w+')

		logging.info('Started preprocessing subject {}, session {}'.format(sj, session))
		# READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
		EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
		                                   montage=None, preload=True, eog=eog) for run in eeg_runs])

		#EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['V_do','H_l','Ref_r','Ref_l','EXG7','EXG8'])
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEGica = EEG.filter(h_freq=None, l_freq=1,
		                    fir_design='firwin', skip_by_annotation='edge')
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		             skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
		if sj == 1:
			events = np.delete(events, -1, 0)   

		beh, missing = EEG.matchBeh(sj, session, events, trigger, 
		                             headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		epochs.selectBadChannels(channel_plots = False, inspect = True, RT = None)    
		epochs.artifactDetection(inspect=True, run = True)

		# ICA
		epochs.applyICA(EEG, method='picard')

		# EYE MOVEMENTS
		epochs.detectEye(missing, events, beh.shape[0], time_window=(t_min*1000, t_max*1000), 
						tracker_shift = tracker_shift, start_event = start_event, 
						extension = tracker_ext, eye_freq = eye_freq, 
						screen_res = screen_res, viewing_dist = viewing_dist, 
						screen_h = screen_h)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, trigger)

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/color_decoding'
	os.chdir(project_folder)

	# initiate current project
	PO = color_decoding()
	sj = 2

	# PO.prepareEEG(sj = sj, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 			t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 			trigger = trigger, project_param = project_param, 
	# 			project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)


	#location decoding
	for sj in [1,2]:
		beh, eeg = PO.loadData(sj, False, (-1,0.6),'HEOG', 1,
				 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = False) # don't remove trials 

		#bdm = BDM(beh, eeg, to_decode = 'target_color', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc')
		#bdm.Classify(sj, cnds = ['target_pred'], cnd_header = 'block_type', 
		#			bdm_labels = ['#64009a','#fb0300','#649a00','#00cb33','#64009a'], time = (-0.75, 0.5), nr_perm = 0, gat_matrix = False)

		bdm = BDM(beh, eeg, to_decode = 'dist_color', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc')
		bdm.Classify(sj, cnds = ['target_pred'], cnd_header = 'block_type', 
					bdm_labels = ['#64009a','#fb0300','#649a00','#00cb33','#64009a'], time = (-0.75, 0.5), nr_perm = 0, gat_matrix = False)

