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
project = 'priority_map'
factors = ['block_type','target_high']
labels = [['bias','neutral'],['yes','no','None']]
to_filter = ['RT'] 
project_param = ['nr_trials','trigger','RT', 'subject_nr',
				'block_type', 'correct','dist_high','dist_loc','dist_shape', 'dist_color',
                'high_prob_loc', 'target_high','target_loc','target_shape','target_color','ping_type','block_cnt','trial_type']


# eeg info (event_id specified below)
ping = 'no_ping' # 'ping'
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
eeg_runs = [1]
t_min = -0.8
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

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, event_id, project_param, project_folder, binary, channel_plots, inspect, ping = 'ping'):
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

		if ping == 'ping':
			beh, missing = self.matchBehPriority(sj, session, events, event_id, 
		                             headers = project_param)
		else:
			beh, missing = EEG.matchBeh(sj, session, events, event_id, 
		                             headers = project_param)			

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


	def matchBehPriority(self, sj, session, events, event_id, headers):
		'''
		Alligns bdf file with csv file with experimental variables

		Arguments
		- - - - -
		raw (object): raw mne eeg object
		sj (int): sj number 
		session(int): session number
		events(array): event file from eventSelection (last column contains trigger values)
		trigger(list|array): trigger values used for epoching
		headers (list): relevant column names from behavior file

		Returns
		- - - -
		beh (object): panda object with behavioral data (triggers are alligned)
		missing (araray): array of missing trials (can be used when selecting eyetracking data)
		'''

		# read in data file
		beh_file = self.FolderTracker(extension=[
		            'beh', 'raw'], filename='subject-{}_session_{}.csv'.format(sj, session))

		# get triggers logged in beh file
		beh = pd.read_csv(beh_file)
		beh = beh[headers]
		# control for sparse events
		
		beh['trigger'][beh['high_prob_loc']== '0'] = 100 
		beh['trigger'][beh['high_prob_loc']== '2'] = 102 
		beh['trigger'][beh['high_prob_loc']== '4'] = 104  
		beh['trigger'][beh['high_prob_loc']== '6'] = 106 
		beh['trigger'][beh['ping_type']== 'None'] = 200 

		if 'practice' in headers:
			beh = beh[beh['practice'] == 'no']
			beh = beh.drop(['practice'], axis=1)
		beh_triggers = beh['trigger'].values  

		# get triggers bdf file
		if type(event_id) == dict:
			event_id = [event_id[key] for key in event_id.keys()]
		idx_trigger = [idx for idx, tr in enumerate(events[:,2]) if tr in event_id] 
		bdf_triggers = events[idx_trigger,2] 

		# log number of unique triggers
		unique = np.unique(bdf_triggers)
		logging.info('{} detected unique triggers (min = {}, max = {})'.
		                format(unique.size, unique.min(), unique.max()))

		# make sure trigger info between beh and bdf data matches
		missing_trials = []
		nr_miss = beh_triggers.size - bdf_triggers.size
		logging.info('{} trials will be removed from beh file'.format(nr_miss))

		# check whether trial info is present in beh file
		if nr_miss > 0 and 'nr_trials' not in beh.columns:
			raise ValueError('Behavior file does not contain a column with trial info named nr_trials. Please adjust')

		while nr_miss > 0:
			stop = True
			# continue to remove beh trials until data files are lined up
			for i, tr in enumerate(bdf_triggers):
				if tr != beh_triggers[i]: # remove trigger from beh_file
					miss = beh['nr_trials'].iloc[i]
					#print miss
					missing_trials.append(miss)
					logging.info('Removed trial {} from beh file,because no matching trigger exists in bdf file'.format(miss))
					beh.drop(beh.index[i], inplace=True)
					beh_triggers = np.delete(beh_triggers, i, axis = 0)
					nr_miss -= 1
					stop = False
					break
		
		# check whether there are missing trials at end of beh file        
			if beh_triggers.size > bdf_triggers.size and stop:

				# drop the last items from the beh file
				missing_trials = np.hstack((missing_trials, beh['nr_trials'].iloc[-nr_miss:].values))
				beh.drop(beh.index[-nr_miss], inplace=True)
				logging.info('Removed last {} trials because no matches detected'.format(nr_miss))         
				nr_miss = 0

		# keep track of missing trials to allign eye tracking data (if available)   
		missing = np.array(missing_trials)      
		# log number of matches between beh and bdf    
		logging.info('{} matches between beh and epoched data out of {}'.
		    format(sum(beh['trigger'].values == bdf_triggers), bdf_triggers.size))           

		return beh, missing

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/priority_map'
	os.chdir(project_folder)

	# initiate current project
	PO = priorityMap()


	#Run preprocessing behavior
	for sj in []:
		print('starting subject {}'.format(sj))
		if ping == 'ping':
			event_id = [100,102, 104, 106, 200]
		else:
			event_id = [12,13,14,15,16,17,18,19,21,23,24,25,26,27,28,29,31,32,34,35,36,37,38,39,41,42,43,45,46,\
						47,48,49,51,52,53,54,56,57,58,59,61,62,63,64,65,67,68,69,71,72,73,74,75,76,78,79,81,82,83,84,85,86,87,89]

		PO.prepareEEG(sj = sj, session = 2, eog = eog, ref = ref, eeg_runs = eeg_runs, 
			t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
				event_id = event_id, project_param = project_param, 
				project_folder = project_folder, binary = binary, channel_plots = True, inspect = True, ping = ping)
			

	sj = 3
	# RUN LOCATION DECODING 
	# beh, eeg = PO.loadData(sj, name = 'ping', eyefilter = True,  eye_window = (-0.5,0.5),eye_ch = 'HEOG',
	#  		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20) , use_tracker = False) # only scan for eye movements in anticipatry period

	# bdm = BDM(beh, eeg, to_decode = 'high_prob_loc', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc',
	# 			baseline = (-0.1, 0))
	# beh['cnd'] = 'all'
	# bdm.Classify(sj, cnds = ['all'], cnd_header = 'cnd', bdm_labels = ['0','4','6'], excl_factor = dict(ping_type = ['high','low'], block_type = ['neutral']),
	#  				time = (-0.1, 0.6), nr_perm = 0, gat_matrix = False, save = True)

	# RUN LOCATION DECODING TARGET (all trials)
	# beh, eeg = PO.loadData(sj, name = 'all', eyefilter = True,  eye_window = (-1,0),eye_ch = 'HEOG', 
	# 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20) , use_tracker = False) # only scan for eye movements in anticipatry period

	# bdm = BDM(beh, eeg, to_decode = 'target_loc', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc',
	# 		baseline = (-0.1, 0))
	# beh['cnd'] = 'all'
	# bdm.Classify(1, cnds = ['all'], cnd_header = 'cnd', bdm_labels = [0,1,2,3,4,5,6,7],
	# 				time = (-0.1, 0.6), nr_perm = 0, gat_matrix = False, save = True)	

	# RUN TF analysis (non-ping trials)
	# beh, eeg = PO.loadData(sj, name = 'ping', eyefilter = True,  eye_window = (-0.5,0.5),eye_ch = 'HEOG',
	#   		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20) , use_tracker = False) # only scan for eye movements in anticipatry period

	# tf = TF(beh, eeg, laplacian = False)
	# beh['cnd'] = 'all'
	# tf.TFanalysis(sj, cnds = ['all'], cnd_header = 'cnd', base_type = 'Z', min_freq = 1, factor = dict(ping_type = ['high','low'], block_type = ['neutral'], high_prob_loc = ['0','4']),
	#  				time_period = (-0.5,0.5), base_period = None, elec_oi = ['PO7','PO8'], method = 'wavelet', flip = dict(high_prob_loc = ['2']), tf_name = 'no_lapl_1-40')


	# RUN LOCATION DECODING (PING TRIALS)
	# beh, eeg = PO.loadData(sj, name = 'ping', eyefilter = True,  eye_window = (-0.5,0.5),eye_ch = 'HEOG',
	#   		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20) , use_tracker = False) # only scan for eye movements in anticipatry period

	# bdm = BDM(beh, eeg, to_decode = 'high_prob_loc', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc',
	#   		baseline = (-0.1, 0))
	# beh['cnd'] = 'all'
	# bdm.Classify(sj, cnds = ['all'], cnd_header = 'cnd', bdm_labels = ['2','6'], excl_factor = dict(ping_type = ['None'], block_type = ['neutral']),
	#   				time = (-0.1, 0.6), nr_perm = 0, gat_matrix = False, save = True, downscale = False)

	# # HIGH vs Low Ping trials
	# beh, eeg = PO.loadData(sj, name = 'ping', eyefilter = True,  eye_window = (-0.5,0.5),eye_ch = 'HEOG',
	#  		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20) , use_tracker = False) # only scan for eye movements in anticipatry period
	# bdm = BDM(beh, eeg, to_decode = 'high_prob_loc', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc',
	#  		baseline = (-0.1, 0))
	# bdm.Classify(sj, cnds = ['high','low'], cnd_header = 'ping_type', bdm_labels = ['0','2','4','6'],  excl_factor = dict(ping_type = ['None'], block_type = ['neutral']),
	#  				time = (-0.1, 0.6), nr_perm = 0, gat_matrix = False, save = True, downscale = False)

	# ERP (N2pc) ping trials

	beh, eeg = PO.loadData(sj, name = 'ping', eyefilter = True,  eye_window = (-0.5,0.5),eye_ch = 'HEOG',
	  		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20) , use_tracker = False) # only scan for eye movements in anticipatry period
	beh['cnd'] = 'all'
	erp = ERP(eeg, beh, 'high_prob_loc', (-0.1,0))
	erp.selectERPData(time = (-0.1, 0.6), h_filter = 30, excl_factor = dict(ping_type = ['None'], block_type = ['neutral']))
	erp.topoFlip(left = ['2'], header = 'high_prob_loc')
	erp.ipsiContra(sj = sj, left = ['2'], right = ['6'], l_elec = ['PO7'], conditions = ['all'], cnd_header = 'cnd',
                      r_elec = ['PO8'], erp_name = 'ping', RT_split = False)

