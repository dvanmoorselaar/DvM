#import matplotlib
#matplotlib.use('agg') # now it works via ssh connection

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
		   '2': {'tracker': (False, '', '','',''),  'replace':{}},
		   '3': {'tracker': (False, '', '','',''),  'replace':{}},
		   '4': {'tracker': (False, '', '','',''),  'replace':{}},
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
#trigger = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18] # for localizer
trigger = [1002,1003,1004,1005,1006,1007,1008,1009] # for AB (start response interval)
#t_min = -0.2 # for localizer
#t_max = 1 # for localizer
t_min = -0.4   # for AB
t_max = 1.2
flt_pad = 0.5
eeg_runs = [2]
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

	def matchBehnew(self, sj, events, trigger, task):
		'''

		'''

		missing = np.array([])
		if task == 'localizer':

			idx_trigger = np.sort(np.hstack([np.where(events[:,2] == i)[0] for i in trigger]))
			triggers = events[idx_trigger,2]
			beh = pd.DataFrame(index = range(triggers.size),columns=['condition', 'nr_trials', 'label'])
			beh['digit'] = 0
			beh['digit'][triggers < 10] = triggers[triggers < 10]
			beh['letter'] = 0
			beh['letter'][triggers > 10] = triggers[triggers > 10]
			beh['condition'] = 'digit'
			beh['condition'][triggers > 10] = 'letter'
			beh['nr_trials'] = range(triggers.size)

			print 'detected {} epochs'.format(triggers.size)
			
		elif task == 'AB':
			# find indices of beginning of fixation and change T1 triggers
			idx_end = np.where(events[:,2] == 51)[0]
			events[idx_end - 14,2] += 1000
			beh = pd.DataFrame(index = range(idx_end.size),columns=['condition', 'nr_trials', 'T1','T2','T3','D1','D2','D3'])

			cnd_idx = np.where((events[:,2] > 60) *( events[:,2] < 70))[0]
			cnds = events[cnd_idx,2]
			# save condition info
			beh['nr_trials'] = range(1,cnd_idx.size + 1) 
			beh['condition'][cnds == 61] = 'T..DDDT'
			beh['condition'][cnds == 62] = 'TTDDD'
			beh['condition'][cnds == 63] = 'TDTDD'
			beh['condition'][cnds == 64] = 'TDDTD'
			beh['condition'][cnds == 65] = 'TTTDD'
			beh['condition'][cnds == 66] = 'TTDTD'
			beh['condition'][cnds == 67] = 'TDTTD'
			# save T1 info per trial
			beh['T1'] = events[idx_end - 14,2] - 1000
			# save T2 info per trial (61 missing)
			beh['T2'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 7,2]
			beh['T2'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 8,2]
			beh['T2'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 9,2]
			beh['T2'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 7,2]
			beh['T2'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 7,2]
			beh['T2'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 8,2]

			# save T3 info per trial
			beh['T3'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 8,2]
			beh['T3'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 9,2]
			beh['T3'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 9,2]

			# save D1,D2,D3 info (61 missing)
			beh['D1'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 8,2]
			beh['D2'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 9,2]
			beh['D3'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 10,2]
			beh['D1'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 7,2]
			beh['D2'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 9,2]
			beh['D3'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 10,2]
			beh['D1'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 7,2]
			beh['D2'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 8,2]
			beh['D3'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 10,2]
			beh['D1'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 9,2]
			beh['D2'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 10,2]
			beh['D1'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 8,2]
			beh['D2'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 10,2]
			beh['D1'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 7,2]
			beh['D2'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 10,2]			

			print 'detected {} epochs'.format(idx_end.size)

		return beh, missing

	def matchBeh(self, sj, events, trigger, headers):
		'''
		make sure info in behavior files lines up with detected events
		'''

		embed()
		# read in data file
		beh_file = self.FolderTracker(extension=[
		            'beh', 'raw'], filename='subject-{}_ses_1-upd.csv'.format(sj))

        # get triggers logged in beh file
		beh = pd.read_csv(beh_file)
		beh = beh[headers]
		beh['timing'] = None

		# make sure trigger info between beh and bdf data matches
		idx_trigger = np.array([idx for idx, tr in enumerate(events[:,2]) if tr in trigger]) 
		nr_miss = beh.shape[0] - idx_trigger.size
		missing_trials = np.array([])

		# store timing info for each condition (now only logs pos 5 - 9)
		t_factor = 1000/512.0
		for i, idx in enumerate(idx_trigger):
			print "\r{0}% of trial timings updated".format((float(i)/idx_trigger.size)*100),
			beh['timing'][i] = {'p5':(events[idx,0] - events[idx-14,0]) * t_factor,
					  'p6':(events[idx,0] - events[idx-13,0]) * t_factor,
					  'p7':(events[idx,0] - events[idx-12,0]) * t_factor,
					  'p8':(events[idx,0] - events[idx-11,0]) * t_factor,
					  'p9':(events[idx,0] - events[idx-10,0]) * t_factor}
		
		print 'The number of missing trials is {}'.format(nr_miss)

		return beh, missing

	def splitEpochs(self):
		'''

		'''

		pass

	def plotBDM(self):
		'''
		Plots GAT matrix and diagonal for digits (top) and letters (bottom)

		'''

		with open(self.FolderTracker(['bdm','digit'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	
 
		# plot conditions
		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/8.0)
		for idx, header in enumerate(['digit','letter']):
			bdm = []
			files = glob.glob(self.FolderTracker(['bdm',header], filename = 'class_*_perm-False-broad.pickle'))
			for file in files:
				print file
				with open(file ,'rb') as handle:
					bdm.append(pickle.load(handle))

			
			X = np.stack([bdm[i][header]['standard'] for i in range(len(bdm))])
			print X.shape
			X = X.mean(axis = 0)
			
			# plot diagonal
			ax = plt.subplot(2,2, idx + 3, title = 'Diagonal-{}'.format(header), ylabel = 'dec acc (%)', xlabel = 'time (ms)')
			plt.plot(times, np.diag(X))
			plt.axhline(y = 1/8.0, color = 'black', ls = '--')
			sns.despine(offset=50, trim = False)

			# plot GAT
			X[X <1/8.0] = 1/8.0
			ax = plt.subplot(2,2, idx + 1, title = 'GAT-{}'.format(header), ylabel = 'train time (ms)', xlabel = 'test time (ms)')
			plt.imshow(X, norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
						cmap = cm.bwr, interpolation = None, vmin = 1/8.0, vmax = 0.16)
			plt.colorbar()
			sns.despine(offset=50, trim = False)

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['bdm'], filename = 'localizer.pdf'))
		plt.close()

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
		#beh, missing = self.matchBeh(sj, events, trigger, 
		#                             headers = project_param)
		beh, missing = self.matchBehnew(sj, events, trigger, task = 'AB')

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
	#PO.plotBDM()

	# run preprocessing
	for sj in [1,2,3,4]:
	
		for session in range(1,2):
		   	# PO.updateBeh(sj = sj)
			PO.prepareEEG(sj = sj, session = session, eog = eog, ref = ref, eeg_runs = eeg_runs, 
					  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
					  trigger = trigger, project_param = project_param, 
					  project_folder = project_folder, binary = binary, channel_plots = False, inspect = True)

		#bdm = BDM('digit', nr_folds = 10, eye = False)
		#bdm.Classify(sj, cnds = ['digit'], cnd_header = 'condition', bdm_labels = [2,3,4,5,6,7,8,9], factor = dict(condition = 'digit'), time = (-0.2, 1.2), nr_perm = 0, bdm_matrix = True)
		#bdm = BDM('letter', nr_folds = 10, eye = False)
		#bdm.Classify(sj, cnds = ['letter'], cnd_header = 'condition', bdm_labels = [11,12,13,14,15,16,17,18], factor = dict(condition = 'letter'), time = (-0.2, 1.2), nr_perm = 0, bdm_matrix = True)
