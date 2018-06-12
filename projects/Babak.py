import sys
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import seaborn as sns

from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import *
from eeg_analyses.BDM import *  
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info
sj_info = {'1': {'tracker': (False, '', '','',0),  'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			'2': {'tracker': (False, '', '','',0),  'replace':{}},	
			'3': {'tracker': (False, '', '','',0),  'replace':{}}} 
# project specific info
# Behavior
#project = 'wholevspartial'
#part = 'beh'
#factors = ['block_type','cue']
#labels = [['whole','partial'],['cue','no']]
#to_filter = [] 
project_param = ['trigger','condition']

# EEG
eog =  ['EXG2','EXG3','EXG4']
ref =  ['EXG5','EXG6']
trigger = [83, 82, 81]
t_min = 0
t_max = 1.8
flt_pad = 0.5
eeg_runs = [1,2]
binary = 61440

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class Audio(FolderStructure):

	def __init__(self): pass


	def preprocessingEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, channel_plots, inspect):
		'''
		PIPELINE FOR THE FLEUR PROJECT
		'''
		
		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']
		tracker, ext, t_freq, start_event, shift = sj_info[str(sj)]['tracker']

		# start logging
		logging.basicConfig(level=logging.DEBUG,
		                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
		                    datefmt='%m-%d %H:%M',
		                    filename='processed/info/preprocess_sj{}_ses{}.log'.format(
		                        sj, session),
		                    filemode='w')

		# READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
		EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
		                                   montage=None, preload=True, eog=eog) for run in eeg_runs])
		EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=['EXG2'], hEOG=['EXG3','EXG4'], changevoltage=True, 
						to_remove = ['EXG1','EXG4','EXG5','EXG6','EXG7','EXG8','GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp'])
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEGica = EEG.filter(h_freq=None, l_freq=1,
		                   fir_design='firwin', skip_by_annotation='edge')
		EEG.filter(h_freq=None, l_freq=0.01, fir_design='firwin',
		           skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
		beh, missing = EEG.matchBeh(sj, session, events, trigger, 
		                            headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
				tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		if 'RT' in beh.keys():
			epochs.selectBadChannels(channel_plots = False, inspect=True, RT = beh['RT']/1000)
		else:
			epochs.selectBadChannels(channel_plots = False, inspect=True, RT = None)    
		epochs.artifactDetection(plot = True, inspect=False)

		# ICA
		epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = inspect)

		# EYE MOVEMENTS
		epochs.detectEye(missing, time_window=(t_min*1000, t_max*1000), tracker = tracker, tracker_shift = shift, start_event = start_event, extension = ext, eye_freq = t_freq)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, trigger)

if __name__ == '__main__':
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/Audio'
	os.chdir(project_folder)

	# behavior analysis
	PO =  Audio()
	sj = 2
	# run preprocessing
	# PO.preprocessingEEG(sj = sj, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 			  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 			  trigger = trigger, project_param = project_param, 
	# 			  project_folder = project_folder, binary = binary, channel_plots = False, inspect = True)

	session = BDM('all_channels', 'condition', nr_folds = 10, eye = False)
	session.Classify(sj, cnds = 'all', cnd_header = 'condition', subset = [1,3], time = (0, 1.8), nr_perm = 0, bdm_matrix = False)


