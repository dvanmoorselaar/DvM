import os
import mne
import sys
import glob
import pickle
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM_3')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#plt.switch_backend('Qt4Agg') # run next two lines to interactively scroll through plots
#import matplotlib

#import matplotlib          # run these lines only when running sript via ssh connection
#matplotlib.use('agg')

from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.TF import * 
from eeg_analyses.BDM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
#from stats.nonparametric import *

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
project = 'stat_pred'
factors = ['trial_type','block_type','dist_high']
labels = [['distractor','single_target'],['unpred','shape_pred','color_pred', 'pred'],['yes','no']]
to_filter = ['RT'] 
project_param = ['practice','nr_trials','trigger','RT', 'subject_nr','selected_high_prob',
				'block_type', 'correct','dist_high','dist_loc','dist_shape',
                'high_loc', 'target_high','target_loc','target_shape', 'target_color','trial_type','block_cnt']


# eeg info (event_id specified below)
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
eeg_runs = [1]
t_min = -1
t_max = 0.6
flt_pad = 0.5
binary =  0

# eye tracker info
tracker_ext = 'asc'
eye_freq = 500
start_event = 'Onset search'
tracker_shift = 0
viewing_dist = 100 
screen_res = (1680, 1050) 
screen_h = 29

class StatPred(FolderStructure):

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
		EEG preprocessing as preregistred @ https://osf.io/n35xa/registrations
		'''

		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']

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
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		             skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(event_id, binary=binary, min_duration=0)

		beh, missing = EEG.matchBeh(sj, session, events, event_id, 
		                             headers = project_param)
		beh = self.updateBEH(beh, session)

		# # EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=event_id,
		         tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		epochs.selectBadChannels(channel_plots = False, inspect = True, RT = None)    
		epochs.artifactDetection(inspect=False, run = True)

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
		epochs.linkBeh(beh, events, event_id)


	def updateBEH(self, beh, session):
		'''
		Adds columns for beh that can be used to select trials during eeg analysis
		'''

		beh['session'] = session	
		#beh['spatial_bias'] = 'no_bias'

		#if beh['subject_nr'].iloc[0] in [0,1,2,3,5,6]:
		#	beh['spatial_bias'][beh['block_cnt'] >= 9] = 'bias'
		#elif beh['subject_nr'].iloc[0] in [7,8,9,10,11,13,14,15]:
		#	beh['spatial_bias'][beh['block_cnt'] >= 11] = 'bias'
		#else:
		#	beh['spatial_bias'][beh['block_cnt'] >= 10] = 'bias'

		# add no bias vs bias info (control for programing error)
		beh['spatial_bias'] = 'no bias'
		beh.loc[(beh['subject_nr'].isin([4,10,12,16,17,18,19,20,21,22,23,24])) &\
       			(beh['block_cnt'] >= 10), 'spatial_bias'] = 'bias'
		beh.loc[(beh['subject_nr'].isin([1,2,3,5,6])) &\
       			(beh['block_cnt'] >= 9), 'spatial_bias'] = 'bias'
		beh.loc[(beh['subject_nr'].isin([7,8,9,11,13,14,15])) &\
       			(beh['block_cnt'] >= 11), 'spatial_bias'] = 'bias'

		cnd_info = []
		for i in range(beh.shape[0]):
			cnd_info.append('{}-{}'.format(beh.iloc[i]['block_type'], beh.iloc[i]['spatial_bias']))
		beh['condition'] = cnd_info

		return beh

	def cndRTsplit(self, beh, conditions, method = 'median'):
		'''
		Splits condition data in half based on median/mean
		'''

		for cnd in conditions:
			mask = beh.condition == cnd
			RTs = beh.RT[mask]

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/stat_pred'
	os.chdir(project_folder)

	# initiate current project
	PO = StatPred()

	# analyze behavior
	# behavioral experiment 1 
	#PO.prepareBEH('stat_pred', 'exp1', factors, labels, project_param, to_filter)
	
	# behavioral experiment 2  
	#PO.prepareBEH('stat_pred', 'exp2', factors, labels, project_param, to_filter)

	#behavioral experiment 3 (eeg)
	#PO.prepareBEH('stat_pred', 'beh', ['trial_type','block_type','dist_high'], [['distractor','single_target'],['unpred', 'pred'],['yes','no']], project_param, to_filter)

	# Run preprocessing behavior
	# for sj in [10]:
	# 	print('starting subject {}'.format(sj))
	# 	session = 2
	# 	if (sj % 2 == 0 and session == 2) or (sj % 2 == 1 and session == 1):
	# 		event_id = [10,20,30,40,12,13,14,21,23,24,31,32,34,41,42,43]
	# 	else:
	# 		event_id = [110,120,130,140,112,113,114,121,123,124,131,132,134,141,142,143]

	# 	PO.prepareEEG(sj = sj, session = session, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 			t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 			event_id = event_id, project_param = project_param, 
	# 			project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)

 
	# get preprocessing descriptives
	# for sj in range(1,25):	
	# 	beh, eeg = PO.loadData(sj, False, (-1,0.6),'HEOG', 1,
	# 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = False)
	# 	s,e = [np.argmin(abs(eeg.times - t)) for t in (-1,0.6)]
	# 	eog = eeg._data[:,eeg.ch_names.index('HEOG'),s:e]
	# 	idx_eye = eog_filt(eog, sfreq = eeg.info['sfreq'], windowsize = 200, windowstep = 10, threshold = 20)
	# 	print('Total percentage dropped based on step for sj {} is {}%'.format(sj, idx_eye.size/eeg._data.shape[0]*100 ))

	output_auc, output_acc = [], []
	for sj in range(1,25):	

		if sj in [14,24]:
			use_tracker = False
		else:
			use_tracker = True	
		
		# read in preprocessed data for main ERP analysis
		beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
				 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20) , use_tracker = use_tracker)

		# ERP ANALYSIS pipeline (flip all electrodes as if all stimuli are presented right)
		# distractor tuned (no spatial bias)
		# erp = ERP(eeg, beh, 'dist_loc', (-0.1,0))
		# erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = None)
		# erp.topoFlip(left = ['2'], header = 'dist_loc')
		# erp.ipsiContra(sj = sj, left = ['2'], right = ['6'], l_elec = ['PO7'], conditions = ['pred-no_bias','unpred-no_bias'], cnd_header = 'condition',
  #                      r_elec = ['PO8'], midline = {'target_loc': [0,4]}, erp_name = 'dist', RT_split = True, permute = 1000)

		# # target tuned (no spatial bias)
		# erp = ERP(eeg, beh, 'target_loc', (-0.1,0))
		# erp.selectERPData(time = [-0.1, 0.6], l_filter = False, excl_factor = None) # data is already filtered
		# erp.topoFlip(left = [2], header = 'target_loc')
		# erp.ipsiContra(sj = sj, left = [2], right = [6], l_elec = ['PO7'], conditions = ['pred-no_bias','unpred-no_bias'], cnd_header = 'condition',
  #                      r_elec = ['PO8'], midline = {'dist_loc': ['0','4','None']}, erp_name = 'target', RT_split = True, permute = 1000)

		# # read in preprocessed data for main ERP analysis
		# beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
		# 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)

		# # distractor tuned (distractor at high probability location)
		# erp = ERP(eeg, beh, 'dist_loc', (-0.1,0))
		# erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = dict(dist_high = ['no'])) # exclude all trials with distractor at low probability location (or without distractor)
		# erp.topoFlip(left = [2], header = 'high_loc') # as if all high probabilty locations are on the right
		# erp.ipsiContra(sj = sj, left = ['2'], right = ['6'], l_elec = ['PO7'], conditions = ['pred-bias','unpred-bias'], cnd_header = 'condition',
  #                      r_elec = ['PO8'], midline = {'target_loc': [0,4]}, erp_name = 'dist', RT_split = True, permute = 1000)

		# # read in preprocessed data
		# beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
		# 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)
		
		# # target tuned (target at high probability location)
		# erp = ERP(eeg, beh, 'target_loc', (-0.1,0))
		# erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = dict(target_high = ['no']))
		# erp.topoFlip(left = [2], header = 'high_loc')
		# erp.ipsiContra(sj = sj, left = [2], right = [6], l_elec = ['PO7'], conditions = ['pred-bias','unpred-bias'], cnd_header = 'condition',
  #                      r_elec = ['PO8'], midline = {'dist_loc': ['0','4','None']}, erp_name = 'target', RT_split = True, permute = 1000)

		# # read in preprocessed data
		# beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
		# 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)
		
		# # neutral tuned (target and distractor at midline or absent)
		# erp = ERP(eeg, beh, 'dist_loc', (-0.1,0))
		# erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = None)
		# erp.topoFlip(left = [2], header = 'high_loc')
		# erp.ipsiContra(sj = sj, left = ['0', '4', 'None'], right = [], l_elec = ['PO7'], 
		# 			  conditions = ['pred-bias','unpred-bias', 'pred-no_bias','unpred-no_bias'], cnd_header = 'condition',
  #                      r_elec = ['PO8'], midline = {'target_loc': [0,4]}, erp_name = 'neutral', RT_split = True, permute = 1000)



		# read in preprocessed data
		# do analysis seperately for first and second half of the block (without a spatial bias)
		beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
		 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)

		beh['nr_trials'] = np.mod(beh['nr_trials'], 60) 
		beh.loc[beh['nr_trials'] == 0, 'nr_trials'] = 60  

		erp = ERP(eeg, beh, 'dist_loc', (-0.1,0))
		erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = {'nr_trials': list(range(1,21))})
		erp.topoFlip(left = ['2'], header = 'dist_loc')
		erp.ipsiContra(sj = sj, left = ['2'], right = ['6'], l_elec = ['PO7'], conditions = ['pred-no_bias','unpred-no_bias'], cnd_header = 'condition',
                        r_elec = ['PO8'], midline = {'target_loc': [0,4]}, erp_name = 'dist_2ndhalf', RT_split = False)


		beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
				 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)

		beh['nr_trials'] = np.mod(beh['nr_trials'], 60) 
		beh.loc[beh['nr_trials'] == 0, 'nr_trials'] = 60  

		erp = ERP(eeg, beh, 'dist_loc', (-0.1,0))
		erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = {'nr_trials': list(range(21,61))})
		erp.topoFlip(left = ['2'], header = 'dist_loc')
		erp.ipsiContra(sj = sj, left = ['2'], right = ['6'], l_elec = ['PO7'], conditions = ['pred-no_bias','unpred-no_bias'], cnd_header = 'condition',
                       r_elec = ['PO8'], midline = {'target_loc': [0,4]}, erp_name = 'dist_1sthalf', RT_split = False)


		# do analysis seperately for first and second half of the block (with a spatial bias)
		beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
		 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)

		beh['nr_trials'] = np.mod(beh['nr_trials'], 60) 
		beh.loc[beh['nr_trials'] == 0, 'nr_trials'] = 60  

		erp = ERP(eeg, beh, 'dist_loc', (-0.1,0))
		erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = {'nr_trials': list(range(1,21)), 'dist_high' : ['no']}) 
		erp.topoFlip(left = [2], header = 'high_loc')
		erp.ipsiContra(sj = sj, left = ['2'], right = ['6'], l_elec = ['PO7'], conditions = ['pred-bias','unpred-bias'], cnd_header = 'condition',
                        r_elec = ['PO8'], midline = {'target_loc': [0,4]}, erp_name = 'dist_2ndhalf', RT_split = False)


		beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
				 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)

		beh['nr_trials'] = np.mod(beh['nr_trials'], 60) 
		beh.loc[beh['nr_trials'] == 0, 'nr_trials'] = 60  

		erp = ERP(eeg, beh, 'dist_loc', (-0.1,0))
		erp.selectERPData(time = [-0.1, 0.6], l_filter = 30, excl_factor = {'nr_trials': list(range(21,61)), 'dist_high' : ['no']})
		erp.topoFlip(left = [2], header = 'high_loc')
		erp.ipsiContra(sj = sj, left = ['2'], right = ['6'], l_elec = ['PO7'], conditions = ['pred-bias','unpred-bias'], cnd_header = 'condition',
                       r_elec = ['PO8'], midline = {'target_loc': [0,4]}, erp_name = 'dist_1sthalf', RT_split = False)

	# 	# read in preprocessed data
		# beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
		# 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)
		# # do TF analysis
		# tf = TF(beh, eeg, laplacian = False)
		# tf.TFanalysis(sj, cnds = ['pred-no_bias','unpred-no_bias', 'pred-bias','unpred-bias'], cnd_header = 'condition', base_type = 'Z', min_freq = 1,
		# 			time_period = (-1,0), base_period = None, elec_oi = 'all', method = 'wavelet', flip = dict(high_loc = [2]), tf_name = 'no_lapl_1-40')

		# #location decoding
		# beh, eeg = PO.loadData(sj, True, (-1,0.6),'HEOG', 1,
		# 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = use_tracker)
		
		# bdm = BDM(beh, eeg, to_decode = 'target_loc', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc')
		# bdm.Classify(sj, cnds = ['pred','unpred'], cnd_header = 'block_type', 
		# 			bdm_labels = [0,2,4,6], time = (-0.2, 0.6), nr_perm = 0, gat_matrix = False)

		# bdm = BDM(beh, eeg, to_decode = 'dist_loc', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc')
		# bdm.Classify(sj, cnds = ['pred','unpred'], cnd_header = 'block_type', 
		# 			bdm_labels = ['0','2','4','6'], time = (-0.2, 0.6), nr_perm = 0, gat_matrix = False)

		# #feature decoding (shape)
		# bdm = BDM(beh, eeg, to_decode = 'target_shape', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc')
		# bdm.Classify(sj, cnds = ['pred','unpred'], cnd_header = 'block_type', 
		# 			bdm_labels = ['circle','diamond'], time = (-1, 0.6), nr_perm = 0, gat_matrix = False)

		# #feature decoding (color)
		# bdm = BDM(beh, eeg, to_decode = 'target_color', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc')
		# bdm.Classify(sj, cnds = ['pred','unpred'], cnd_header = 'block_type', 
		# 			bdm_labels = ['red','green'], time = (-1, 0.6), nr_perm = 0, gat_matrix = False)