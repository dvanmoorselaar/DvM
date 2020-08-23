import os
import mne
import sys
import glob
import pickle
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM_3')
#sys.path.append('/Users/Maxi/Desktop/Internship/DvM')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg') # run next two lines to interactively scroll through plots
import matplotlib

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
project = 'templates_targets_distractors'
factors = ['this_block']
labels = ['distractor','target']
to_filter = ['RT'] 
project_param = ['nr_trials','trigger','RT_search', 'subject_nr','block_nr',
				'distractor_color_code', 'distractor_position','log_memory_condition','log_search_condition','position_colors_M',
                'random_template_color_index', 'search_response','template_position','this_block', 'codes_colors_M','template_color_code']


# eeg info (event_id specified below)
eog =  ['EXG1','EXG2','EXG3','EXG4']
ref =  ['EXG5','EXG6']
session = 1
eeg_runs = [1]
t_min = - 2.6
t_max = 0.6
flt_pad = 0.5
binary =  0
event_id = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45,
            51, 52, 53, 54, 55, 61,62, 63, 64, 65, 71, 72, 73, 74, 75, 81, 82, 83, 84, 85]


# eye tracker info
tracker_ext = 'asc'
eye_freq = 500
start_event = 'Onset search'
tracker_shift = 0
viewing_dist = 100 
screen_res = (1680, 1050) 
screen_h = 29

class Templates(FolderStructure):

	def __init__(self): pass


	def prepareBEH(self, project, part, factors, labels, project_param, to_filter):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and search_response == 1', cnd_sel = False, save = True)
		PP.exclude_outliers(criteria = dict(RT_search = 'RT_search_filter == True', search_response = ''))
		#PP.prep_JASP(agg_func = 'mean', voi = 'RT', data_filter = 'RT_filter == True', save = True)
		PP.save_data_file()

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, event_id, project_param, project_folder, binary, channel_plots, inspect):
		'''
		EEG preprocessing as preregistred @ https://osf.io/n35xa/registrations
		'''

		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']
		self.adjustBeh(sj)
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
		                                  preload=True, eog=eog) for run in eeg_runs])

		#EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['EXG2','EXG4','EXG5','EXG6','EXG7','EXG8'])
		montage = EEG.set_montage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
						skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(event_id, binary=binary, min_duration=0)

		beh, missing = EEG.matchBeh(sj, session, events, event_id,
									headers = project_param)

		# # EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=event_id,
		         tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 
		
		# ARTIFACT DETECTION
		epochs.selectBadChannels(run_ransac = True, channel_plots = False, inspect = True, RT = None)    
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


	def adjustBeh(self, sj):
		'''
		
		'''
		beh_file = self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_session_{}.csv'.format(sj, session))

		# get triggers logged in beh file
		beh = pd.read_csv(beh_file)
		if 'nr_trials' not in beh.columns:
			beh = beh.rename(columns={'trial_nr': 'nr_trials'})
			beh.to_csv(beh_file)

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/Negative_template'
	#project_folder = '/Users/Maxi/Desktop/Internship/templates'
	os.chdir(project_folder)

	# initiate current project
	PO = Templates()

	# behavioral analysis
	# PO.prepareBEH('Negative_template', 'beh', ['this_block'], 
	# 			  [['positive','negative']], project_param, ['RT_search'] )

	# Run preprocessing eeg
	# 'good' sjs
	#subjects = [3,4,7,9,10,11,12,15,16,17,19,20,21,22,23]

	# 'bad' sjs
	#subjects = [2,5,6,8,13,14,18,24]

	subjects = list(range(2,25))
	
	for sj in subjects:
		print('starting subject {}'.format(sj))
		# PO.prepareEEG(sj = sj, session = session, eog = eog, ref = ref, eeg_runs = eeg_runs, 
		# 		t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
		# 		event_id = event_id, project_param = project_param, 
		# 		project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)
		
		# Start ERP analysis
		# read in preprocessed data for main ERP analysis
		beh, eeg = PO.loadData(sj, False, (-0.2,0.55),'HEOG', 1,
				 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = False)

		
		temp_list = []
		for idx in beh['trigger']:
			idx = int(idx/10)*10
			if idx in [10,50]:
				temp_list.append('right')
			if idx in [20,60]:
				temp_list.append('down')
			if idx in [30,70]:
				temp_list.append('left')
			if idx in [40,80]:
				temp_list.append('up')
		beh['memory_position'] = temp_list

		# ERP ANALYSIS pipeline (flip all electrodes as if all stimuli are presented right)
		# target tuned waveform 
		# erp = ERP(eeg, beh, 'template_position', (-0.2,0))
		# erp.selectERPData(time = [-0.2, 0.55], l_filter = 1.5, h_filter = 30, excl_factor = None)
		# erp.topoFlip(left = [4,5], header = 'template_position')
		# erp.ipsiContra(sj = sj, left = [4,5], right = [1,2], l_elec = ['PO7'], conditions = ['positive','negative'], cnd_header = 'this_block',
  #                     r_elec = ['PO8'], midline = {'distractor_position': [0,3]}, erp_name = 'target_waveform_bandpass', RT_split = False, permute = False)

		# # # distractor tuned waveform
		# erp = ERP(eeg, beh, 'distractor_position', (-0.2,0))
		# erp.selectERPData(time = [-0.2, 0.55], l_filter = 1.5, h_filter = 30, excl_factor = None)
		# erp.topoFlip(left = [4,5], header = 'distractor_position')
		# erp.ipsiContra(sj = sj, left = [4,5], right = [1,2], l_elec = ['PO7'], conditions = ['positive','negative'], cnd_header = 'this_block',
  #                      r_elec = ['PO8'], midline = {'template_position': [0,3]}, erp_name = 'distractor_waveform_bandpass', RT_split = False, permute = False)


   		# EXPLORATORY LOCATION DECODING ANALYSIS
		# bdm = BDM(beh, eeg, to_decode = 'template_position', nr_folds = 10, method = 'auc', elec_oi = 'all', downsample = 128, baseline = (-0.2, 0))
		# bdm.Classify(sj, cnds =  ['positive','negative'], cnd_header = 'this_block', bdm_labels = [0,1,2,3,4,5], time = (-0.2, 0.55), gat_matrix = False)

		# bdm = BDM(beh, eeg, to_decode = 'distractor_position', nr_folds = 10, method = 'auc', elec_oi = 'all', downsample = 128, baseline = (-0.2, 0))
		# bdm.Classify(sj, cnds =  ['positive','negative'], cnd_header = 'this_block', bdm_labels = [0,1,2,3,4,5],time = (-0.2, 0.55), gat_matrix = False)	

		# COLOR DECODING TEMPLATE
		bdm = BDM(beh, eeg, to_decode = 'template_color_code', nr_folds = 8, method = 'auc', elec_oi = 'post', 
				 sliding_window = 5, downsample = 200, baseline = None, use_pca = 0.95)
		bdm.Classify(sj, cnds =  ['positive','negative'], cnd_header = 'this_block', bdm_labels = ['A','B','C','D','E','F','G','H','I','J','K','L'], time = (-2.6, 0.5), gat_matrix = False)
		
		# LOCATION DECODING OF THE TEMPLATE IN MEMORY DISPLAY 	
		# bdm = BDM(beh, eeg, to_decode = 'memory_position', nr_folds = 10, method = 'auc', elec_oi = 'all', downsample = 128, baseline = (-2.8, -2.6))
		# bdm.Classify(sj, cnds =  ['positive','negative'], cnd_header = 'this_block', bdm_labels = ['right', 'left', 'down', 'up'], time = (-2.8, 0.2), gat_matrix = False)


		# # do TF analysis
		# tf = TF(beh, eeg, laplacian = False)
		# tf.TFanalysis(sj, cnds = ['positive','negative'], cnd_header = 'this_block', base_type = 'Z', min_freq = 5, factor = dict(memory_position = ['up', 'down']),
		#  			time_period = (-2.8,0), base_period = None, elec_oi = ['PO7','PO8','PO3','PO4','O1','O2'], method = 'wavelet', 
		#  			flip = dict(memory_position = ['left']), tf_name = 'no_lapl_5-40_Z', downsample = 4)






















