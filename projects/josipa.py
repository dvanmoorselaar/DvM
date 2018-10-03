import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import os
import mne
import sys
import glob
import pickle
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import argrelextrema
from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.TF import * 
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
from eeg_analyses.CTF import * 
from eeg_analyses.Spatial_EM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info
sj_info = {'1': {'tracker': (False, '', '','',''),  'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			}

# project specific info
project = 'Josipa'
part = 'beh'
factors = []
labels = []
to_filter = [] 
project_param = ['nr_trials','trigger','condition','trialtype', 'subjects',
				't1_num', 't2_num','t3_num','t1_time','t2_time','t3_time','t1_det','t1_ide'
				,'t2_det','t2_ide','t3_det','t3_ide','t1_ide_any','t2_ide_any','t3_ide_any']

montage = mne.channels.read_montage(kind='biosemi64')
eog =  ['EXG1','EXG2','EXG5','EXG6']
ref =  ['EXG3','EXG4']
trigger = [51]
t_min = -3.4
t_max = 0
flt_pad = 0.5
eeg_runs = [1]
binary =  61440

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class Josipa(FolderStructure):

	def __init__(self): pass

	def updateBeh(self, sj):
		'''
		add missing column info to behavior file
		'''

		# read in data file
		beh_file = self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_ses_1.csv'.format(sj))


        # get triggers logged in beh file
		beh = pd.read_csv(beh_file)
		beh['condition'] = None
		beh['trigger'] = 50
		beh['nr_trials'] = range(1, beh.shape[0] + 1)

		# update condition column
		for i,cnd in [(1,'TDDDT'),(2,'TTDDD'),(3,'TDTDD'),(4,'TDDTD'),
			(5,'TTTDD'),(6,'TTDTD'),(7,'TTDDD'),(8,'singleT')]:
			mask = beh['trialtype'] == i
			beh['condition'][mask] = cnd

		# save data file (.csv)
		beh.to_csv(self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_ses_1-upd.csv'.format(sj)))

	def matchBeh(self, sj, events, trigger, headers):
		'''
		make sure info in behavior files lines up with detected events
		'''

		# read in data file
		beh_file = self.FolderTracker(extension=[
		            'beh', 'raw'], filename='subject-{}_ses_1-upd.csv'.format(sj))

		embed()
        # get triggers logged in beh file
		beh = pd.read_csv(beh_file)
		beh = beh[headers]

		# make sure trigger info between beh and bdf data matches
		idx_trigger = [idx for idx, tr in enumerate(events[:,2]) if tr in trigger] 
		nr_miss = beh.shape[0] - len(idx_trigger)
		missing_trials = []
		



		return beh, missing

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, channel_plots, inspect):
		'''
		EEG preprocessing
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
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True)
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEGica = EEG.filter(h_freq=None, l_freq=1,
		                  fir_design='firwin', skip_by_annotation='edge')
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		           skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
		beh, missing = self.matchBeh(sj, events, trigger, 
		                             headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		epochs.selectBadChannels(channel_plots = channel_plots, inspect=inspect, RT = None)    
		epochs.artifactDetection(inspect=inspect)

		# ICA
		epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = inspect)

		# EYE MOVEMENTS
		epochs.detectEye(missing, time_window=(t_min*1000, t_max*1000), tracker = tracker, tracker_shift = shift, start_event = start_event, extension = ext, eye_freq = t_freq)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, trigger)

	def prepareBEH(self, project, part, factors, labels, project_param):
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

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '2' 
	os.environ['NUMEXP_NUM_THREADS'] = '2'
	os.environ['OMP_NUM_THREADS'] = '2'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/Josipa'
	os.chdir(project_folder)

	# initiate current project
	PO = Josipa()

	# run preprocessing
	for sj in [1]:
	
		PO.updateBeh(sj = sj)
		PO.prepareEEG(sj = sj, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
				  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
				  trigger = trigger, project_param = project_param, 
				  project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)
