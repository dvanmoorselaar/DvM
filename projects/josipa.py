import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import os
import mne
import sys
import glob
import pickle
#sys.path.append('/home/jalilov1/DvM')
#sys.path.append('/home/jalilov1/BB/AB_R/DvM')
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM_3')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import argrelextrema
from IPython import embed
from beh_analyses.PreProcessing import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from eeg_analyses.TF import * 
from eeg_analyses.EEG import * 
#from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
#from eeg_analyses.CTF import * 
#from eeg_analyses.Spatial_EM import * 
from visuals.visuals import MidpointNormalize
#from support.FolderStructure import *
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

	def updatePickleSeenUnseen(self):
		'''

		'''

		for sj in [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35]:
			behs = []
			for session in [2,3]:
				# read in behavior
				beh_p = pickle.load(open(self.FolderTracker(extension=[
                    'AB', 'beh','processed'], filename='subject-{}_ses-{}-new.pickle'.format(sj, session)), "rb" ), encoding='latin1')
				beh_p = pd.DataFrame.from_dict(beh_p)
				beh_m = pd.read_csv(self.FolderTracker(extension=['beh_matlab'], filename='abr_datamat_{}_sb{}.csv'.format(session - 1,sj)))

				# find matching trials between behavior in python and matlab
				nr_trials = beh_m['trial'].values
				split_trials = np.split(nr_trials, np.where(np.diff(nr_trials) < 0)[0]+ 1)  
				to_add = [split[-1] for split in split_trials]
				for i in range(1, len(to_add)):
					split_trials[i] += sum(to_add[:i]) 
				nr_trials = np.hstack(split_trials)
				trial_idx = np.array([list(nr_trials).index(tr) for tr in beh_p['nr_trials']])
				beh_p['T2_seen'] = beh_m['t2_ide_correct'].values[trial_idx]
				behs.append(beh_p)
			behs = pd.concat(behs, ignore_index = True).to_dict(orient = 'series') 
			# save combined pickle file
			pickle.dump(behs, open(self.FolderTracker(extension=['AB','beh', 'processed'],
            	filename='subject-{}_all.pickle'.format(sj)), 'wb' ) )

	def matchBehnew(self, sj, events, trigger, task):
		'''

		'''

		missing = np.array([])
		if task == 'localizer':

			idx_trigger = np.sort(np.hstack([np.where(events[:,2] == i)[0] for i in trigger]))
			triggers = events[idx_trigger,2]
			beh = pd.DataFrame(index = range(triggers.size),columns=['condition', 'nr_trials', 'label'])
			#beh['digit'] = 0
			#beh['digit'][triggers < 10] = triggers[triggers < 10]
			#beh['letter'] = 0
			#beh['letter'][triggers > 10] = triggers[triggers > 10]
			#beh['condition'] = 'digit'
			#beh['condition'][triggers > 10] = 'letter'
			#beh['nr_trials'] = range(triggers.size)

			beh['digit'] = 0 
			beh['digit'][(triggers > 21) * (triggers < 30)] = triggers[(triggers > 21) * (triggers < 30)]
			#beh['digit'][triggers == 22 | 23 | 24 | 25 | 26 | 27 | 28| 29 ] = triggers[triggers == 22 | 23 | 24 | 25 | 26 | 27 | 28| 29 ]
			beh['letter'] = 0
			#beh['letter'][triggers == 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 ] = triggers[triggers == 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 ]
			beh['letter'][(triggers > 30) * (triggers < 39)] = triggers[(triggers > 30) * (triggers < 39)]
			beh['condition'] = 'digit'
			#beh['condition'][triggers == 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37] = 'letter'
			beh['condition'][(triggers > 30) * (triggers < 39)] = 'letter'
			beh['nr_trials'] = range(triggers.size)
		
			print('detected {} epochs'.format(triggers.size))
			
		elif task == 'AB':

			# find indices of beginning of fixation and change T1 triggers
			idx_end = np.where(events[:,2] == 51)[0]
			events[idx_end - 14,2] += 1000
			#beh = pd.DataFrame(index = range(idx_end.size),columns=['condition', 'nr_trials', 'T1','T2','T3','D1','D2','D3'])
			cnd_idx = np.where((events[:,2] > 60) *( events[:,2] < 68))[0]
			beh = pd.DataFrame(index = range(cnd_idx.size),columns=['condition', 'nr_trials', 'T1','T2','T3','D1','D2','D3'])
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
			#beh['T1'] = events[idx_end - 14,2] - 1000
			t1 = events[idx_end - 14,2] - 1000
			beh['T1'] = t1[t1 < 10]
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

			print('detected {} epochs'.format(idx_end.size))

		return beh, missing

	def matchBeh(self, sj, events, trigger, headers):
		'''
		make sure info in behavior files lines up with detected events
		'''

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
			print("\r{0}% of trial timings updated".format((float(i)/idx_trigger.size)*100),)
			beh['timing'][i] = {'p5':(events[idx,0] - events[idx-14,0]) * t_factor,
					  'p6':(events[idx,0] - events[idx-13,0]) * t_factor,
					  'p7':(events[idx,0] - events[idx-12,0]) * t_factor,
					  'p8':(events[idx,0] - events[idx-11,0]) * t_factor,
					  'p9':(events[idx,0] - events[idx-10,0]) * t_factor}
		
		print('The number of missing trials is {}'.format(nr_miss))

		return beh, missing

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, channel_plots, inspect, task):
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
		                    filename=self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocess_sj{}_ses{}.log'.format(
                        sj, session), overwrite = False),
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
		beh, missing = self.matchBehnew(sj, events, trigger, task = task)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		epochs.selectBadChannels(channel_plots = True, inspect=False, RT = None)    
		epochs.artifactDetection(inspect=False, run = True)

		# ICA
		epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = inspect)

		# EYE MOVEMENTS
		epochs.detectEye(missing, time_window=(t_min*1000, t_max*1000), tracker = tracker, tracker_shift = shift, start_event = start_event, extension = ext, eye_freq = t_freq)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		self.savePreprocessing(beh, epochs, events, trigger, task)

	def savePreprocessing(self, beh, eeg, events, trigger,task):
		'''
		saves the preprocessed localizer and AB sessions in seperate folders
		Data of two AB sessions will be combined into a single session
		'''	

		# check which trials were excluded
		events[[i for i, idx in enumerate(events[:,2]) if idx in trigger],2] = range(beh.shape[0])
		sel_tr = events[eeg.selection, 2]

		# create behavior dictionary (only include clean trials after preprocessing)
		# also include eye binned data
		eye_bins = np.loadtxt(self.FolderTracker(extension=[
		                    'preprocessing', 'subject-{}'.format(eeg.sj), eeg.session], 
		                    filename='eye_bins.txt'))

		beh_dict = {'clean_idx': sel_tr, 'eye_bins': eye_bins}
		for header in beh.columns:
			beh_dict.update({header: beh[header].values[sel_tr]})

		# save behavior
		if task == 'AB':
			with open(self.FolderTracker(extension=[task,'beh', 'processed'],
			filename='subject-{}_ses-{}.pickle'.format(eeg.sj, eeg.session)), 'wb') as handle:
				pickle.dump(beh_dict, handle)
		elif task == 'localizer':
			with open(self.FolderTracker(extension=[task,'beh', 'processed'],
			filename='subject-{}_all.pickle'.format(eeg.sj)), 'wb') as handle:
				pickle.dump(beh_dict, handle)		

		# save eeg
		if task == 'localizer':
			eeg.save(self.FolderTracker(extension=[
		      task,'processed'], filename='subject-{}_all-epo.fif'.format(eeg.sj, eeg.session)), split_size='2GB')
		elif task == 'AB':	
			eeg.save(self.FolderTracker(extension=[
		      task,'processed'], filename='subject-{}_ses-{}-epo.fif'.format(eeg.sj, eeg.session)), split_size='2GB')

		# update preprocessing information
		logging.info('Nr clean trials is {0} ({1:.0f}%)'.format(
		sel_tr.size, float(sel_tr.size) / beh.shape[0] * 100))

		try:
			cnd = beh['condition'].values
			min_cnd, cnd = min([sum(cnd == c) for c in np.unique(cnd)]), np.unique(cnd)[
				np.argmin([sum(cnd == c) for c in np.unique(cnd)])]

			logging.info(
				'Minimum condition ({}) number after cleaning is {}'.format(cnd, min_cnd))
		except:
			logging.info('no condition found in beh file')		

		logging.info('EEG data linked to behavior file')

		#if task == 'AB' and int(self.session) == 3:
		if task == 'AB' and int(eeg.session) == 3:

			# combine eeg and beh files of seperate sessions
			all_beh = []
			all_eeg = []
			nr_events = []
			for i in range(2,4):
				with open(self.FolderTracker(extension=[task, 'beh', 'processed'],
					# filename='subject-{}_ses-{}.pickle'.format(eeg.sj, i + 1)), 'rb') as handle:
					# all_beh.append(pickle.load(handle))
			        filename='subject-{}_ses-{}.pickle'.format(eeg.sj, i)), 'rb') as handle:
					all_beh.append(pickle.load(handle))
			        # if i > 0:
			        #     all_beh[i]['clean_idx'] += sum(nr_events)
			        # nr_events.append(all_beh[i]['condition'].size)

				#all_eeg.append(mne.read_epochs(self.FolderTracker(extension=[
			                   #'processed', task], filename='subject-{}_ses-{}-epo.fif'.format(eeg.sj, i + 1))))
				all_eeg.append(mne.read_epochs(self.FolderTracker(extension=[  task,
			                   'processed'], filename='subject-{}_ses-{}-epo.fif'.format(eeg.sj, i))))
			# do actual combining 
			for key in beh_dict.keys():
				beh_dict.update(
		        	{key: np.hstack([beh[key] for beh in all_beh])})

			with open(self.FolderTracker(extension=[task, 'beh', 'processed'],
				filename='subject-{}_all.pickle'.format(eeg.sj)), 'wb') as handle:
				pickle.dump(beh_dict, handle)

			all_eeg = mne.concatenate_epochs(all_eeg)
			all_eeg.save(self.FolderTracker(extension=[task,
		             'processed'], filename='subject-{}_all-epo.fif'.format(eeg.sj)), split_size='2GB')

			logging.info('EEG sessions combined')

	def loadDataTask(self, sj, eye_window, eyefilter, eye_ch = 'HEOG', eye_thresh = 1, task = 'AB'):
		'''
		loads EEG and behavior data

		Arguments
		- - - - - 
		sj (int): subject number
		eye_window (tuple|list): timings to scan for eye movements
		eyefilter (bool): in or exclude eye movements based on step like algorythm
		eye_ch (str): name of channel to scan for eye movements
		eye_thresh (int): exclude trials with an saccades exceeding threshold (in visual degrees)

		Returns
		- - - -
		beh (Dataframe): behavior file
		eeg (mne object): preprocessed eeg data

		'''

		# read in processed behavior from pickle file
		beh = pickle.load(open(self.FolderTracker(extension = [task,'beh','processed'], 
							filename = 'subject-{}_all.pickle'.format(sj)),'rb'))
		beh = pd.DataFrame.from_dict(beh)

		# read in processed EEG data
		eeg = mne.read_epochs(self.FolderTracker(extension = [task,'processed'], 
							filename = 'subject-{}_all-epo.fif'.format(sj)))

		if eyefilter:
			beh, eeg = filter_eye(beh, eeg, eye_window, eye_ch, eye_thresh)

		return beh, eeg

	def crossTaskBDM(self, sj, cnds = 'all', window = (-0.2,0.8), to_decode_tr = 'digit', to_decode_te = 'T1',gat_matrix = True):
		'''
		function that decoding across localizer and AB task
		'''

		# STEP 1: reading data from localizer task and AB task (EEG and behavior)
		locEEG = mne.read_epochs(self.FolderTracker(extension = ['localizer','processed'], filename = 'subject-{}_all-epo.fif'.format(sj)))
		abEEG = mne.read_epochs(self.FolderTracker(extension = ['AB','processed'], filename = 'subject-{}_all-epo.fif'.format(sj)))
		beh_loc = pickle.load(open(self.FolderTracker(extension = ['localizer','beh','processed'], filename = 'subject-{}_all.pickle'.format(sj)),'rb'))
		beh_ab = pickle.load(open(self.FolderTracker(extension = ['AB','beh','processed'], filename = 'subject-{}_all.pickle'.format(sj)),'rb'))

		# STEP 2: downsample data
		locEEG.resample(128)
		abEEG.resample(128)

		# set general parameters
		s_loc, e_loc = [np.argmin(abs(locEEG.times - t)) for t in window]
		s_ab, e_ab = [np.argmin(abs(abEEG.times - t)) for t in window]
		picks = mne.pick_types(abEEG.info, eeg=True, exclude='bads') # 64 good electrodes in both tasks (interpolation)
		eegs_loc = locEEG._data[:,picks,s_loc:e_loc]
		eegs_ab = abEEG._data[:,picks,s_ab:e_ab]
		nr_time = eegs_loc.shape[-1]
		if gat_matrix:
			nr_test_time = eegs_loc.shape[-1]
		else:
			nr_test_time = 1

		# STEP 3: get training and test info
		identity_idx = np.where(beh_loc[to_decode_tr] > 0)[0] # digits are 0 in case of letters and vice versa
		train_labels = beh_loc[to_decode_tr][identity_idx] # select the labels used for training
		nr_tr_labels = np.unique(train_labels).size 
		min_tr_labels = min(np.unique(train_labels, return_counts= True)[1])
		print('You are using {}s to train, with {} as unique labels'.format(to_decode_tr, np.unique(train_labels)))
		train_idx = np.sort(np.hstack([random.sample(np.where(beh_loc[to_decode_tr] == l)[0],min_tr_labels) for l in np.unique(train_labels)]))

		# set test labels
		#test_idx = np.where(np.array(beh_ab['condition']) == cnd)[0] # number test labels is not yet counterbalanced
		test_idx = range(np.array(beh_ab[to_decode_te]).size)

		# STEP 4: do classification
		lda = LinearDiscriminantAnalysis()

		# set training and test labels
		Ytr = beh_loc[to_decode_tr][train_idx] % 10 # double check whether this also works for letters
		Yte = np.array(beh_ab[to_decode_te]) #[test_idx]

		class_acc = np.zeros((nr_time, nr_test_time))
		label_info = np.zeros((nr_time, nr_test_time, nr_tr_labels))
	
		for tr_t in range(nr_time):
			print(tr_t)
			for te_t in range(nr_test_time):
				if not gat_matrix:
					te_t = tr_t

				Xtr = eegs_loc[train_idx,:,tr_t].reshape(-1, picks.size)
				Xte = eegs_ab[test_idx,:,te_t].reshape(-1, picks.size)

				lda.fit(Xtr,Ytr)
				predict = lda.predict(Xte)
				
				if not gat_matrix:
					#class_acc[tr_t, :] = sum(predict == Yte)/float(Yte.size)
					class_acc[tr_t, :] = np.mean([sum(predict[Yte == y] == y)/ float(sum(Yte == y)) for y in np.unique(Yte)])
					label_info[tr_t, :] = [sum(predict == l) for l in np.unique(Ytr)]	
				else:
					#class_acc[tr_t, te_t] = sum(predict == Yte)/float(Yte.size)
					class_acc[tr_t, te_t] = np.mean([sum(predict[Yte == y] == y)/ float(sum(Yte == y)) for y in np.unique(Yte)])
					label_info[tr_t, te_t] = [sum(predict == l) for l in np.unique(Ytr)]	
						
		pickle.dump(class_acc, open(self.FolderTracker(extension = ['cross_task','bdm'], filename = 'subject-{}_bdm.pickle'.format(sj)),'wb'))

	def splitEpochs(self):
		'''

		'''

		pass

	def plotBDMwithin(self, to_plot = 'T1'):
		'''	
		THIS FUNCTION PLOTS WITHIN TASK DECODING
		'''

		# set plotting parameters
		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/8.0)
		with open(self.FolderTracker(['bdm','identity'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	

		for task in ['localizer', 'AB']:
			files = glob.glob(self.FolderTracker([task, 'bdm'], filename = '*_*_dec.pickle'))
			bdm = [pickle.load(open(file,'rb')) for file in files]
			# collapse data
			if task == 'localizer':
				for idx, cnd in enumerate(['digit','letter']):
					X = np.stack([bdm[i][cnd]['standard'] for i in range(len(bdm))])			
					# plot diagonal
					ax = plt.subplot(3,2, [2,4][idx], title = 'Diagonal-{}'.format(cnd), ylabel = 'dec acc (%)', xlabel = 'time (ms)')
					diag = np.diag(np.mean(X,0))
					plt.plot(times, diag)
					plt.axhline(y = 1/8.0, color = 'black', ls = '--')
					sns.despine(offset=50, trim = False)

					# plot gat matrix
					ax = plt.subplot(3,2, [1,3][idx], title = 'GAT-{}'.format(cnd), ylabel = 'train time (ms)', xlabel = 'test time (ms)')
					plt.imshow(X.mean(0), norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
						cmap = cm.bwr, interpolation = None, vmin = 0.09, vmax = 0.15)
					plt.colorbar()
					sns.despine(offset=50, trim = False)

			elif task == 'AB':
				X = np.stack([bdm[i]['all']['standard'] for i in range(len(bdm))])	
				# plot diagonal
				ax = plt.subplot(3,2, 6, title = 'Diagonal-{}'.format(to_plot), ylabel = 'dec acc (%)', xlabel = 'time (ms)')
				diag = np.diag(np.mean(X,0))
				plt.plot(times, diag)
				plt.axhline(y = 1/8.0, color = 'black', ls = '--')
				sns.despine(offset=50, trim = False)

				# plot gat matrix
				ax = plt.subplot(3,2, 5, title = 'GAT-{}'.format(to_plot), ylabel = 'train time (ms)', xlabel = 'test time (ms)')
				plt.imshow(X.mean(0), norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
					cmap = cm.bwr, interpolation = None, vmin = 0.09, vmax = 0.15)
				plt.colorbar()
				sns.despine(offset=50, trim = False)

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['bdm'], filename = 'within_dec_{}.pdf'.format(to_plot)))
		plt.close()
	
	def plotcrossBDM(self):
		'''

		'''

		files = glob.glob(self.FolderTracker(['cross_task','bdm'], filename = 'subject-*_bdm.pickle'))
		bdm = np.stack([pickle.load(open(file,'rb')) for file in files])
		#with open(self.FolderTracker(['bdm','identity'], filename = 'plot_dict.pickle') ,'rb') as handle:
		#	info = pickle.load(handle)
		times = np.linspace(-0.2,0.9,128)
		norm = MidpointNormalize(midpoint=1/8.0)	
		plt.figure(figsize = (20,10))

		# plot diagonal
		ax = plt.subplot(1,2, 1, title = 'Diagonal', ylabel = 'dec acc (%)', xlabel = 'time (ms)')
		diag = np.diag(np.mean(bdm,0))
		plt.plot(times, diag)
		plt.axhline(y = 1/8.0, color = 'black', ls = '--')
		sns.despine(offset=50, trim = False)

		# plot gat matrix
		ax = plt.subplot(1,2, 2, title = 'GAT', ylabel = 'train time (ms)', xlabel = 'test time (ms)')
		plt.imshow(bdm.mean(0), norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
			cmap = cm.bwr, interpolation = None, vmin = 0.09, vmax = 0.15)
		plt.colorbar()
		sns.despine(offset=50, trim = False)

	
		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['cross_task','bdm'], filename = 'cross-task.pdf'))
		plt.close()


	def plotBDM(self, elec_oi = 'all', task = 'AB'):
		'''
		Plots GAT matrix and diagonal for digits (top) and letters (bottom)

		'''
		embed()
		with open(self.FolderTracker([task,'bdm','T1'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	
		# plot conditions
		# plt.figure(figsize = (30,20))
		# norm = MidpointNormalize(midpoint=1/8.0)
		# for idx, header in enumerate(['digit','letter']):
		# 	bdm = []
		# 	files = glob.glob(self.FolderTracker(['bdm',header], filename = 'class_*_perm-False-broad.pickle'))
		# 	for file in files:
		# 		print file
		# 		with open(file ,'rb') as handle:
		# 			bdm.append(pickle.load(handle))

			
		# 	X = np.stack([bdm[i][header]['standard'] for i in range(len(bdm))])
		# 	print X.shape
		# 	X = X.mean(axis = 0)
			
		# 	# plot diagonal
		# 	ax = plt.subplot(2,2, idx + 3, title = 'Diagonal-{}'.format(header), ylabel = 'dec acc (%)', xlabel = 'time (ms)')
		# 	plt.plot(times, np.diag(X))
		# 	plt.axhline(y = 1/8.0, color = 'black', ls = '--')
		# 	sns.despine(offset=50, trim = False)

		# 	# plot GAT
		# 	X[X <1/8.0] = 1/8.0
		# 	ax = plt.subplot(2,2, idx + 1, title = 'GAT-{}'.format(header), ylabel = 'train time (ms)', xlabel = 'test time (ms)')
		# 	plt.imshow(X, norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
		# 				cmap = cm.bwr, interpolation = None, vmin = 1/8.0, vmax = 0.16)
		# 	plt.colorbar()
		# 	sns.despine(offset=50, trim = False)

		# plt.tight_layout()	
		# plt.savefig(self.FolderTracker(['bdm'], filename = 'localizer.pdf'))
		# plt.close()

		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/8.0)
		for idx, header in enumerate(['digit','letter']):
			bdm = []
			
			files = glob.glob(self.FolderTracker(['bdm',header], filename = 'class_*_perm-False-broad.pickle'))
			for file in files:
				with open(file ,'rb') as handle:
					bdm.append(pickle.load(handle))
			

			###### plot diagonal - downscaling #######
			# ax = plt.subplot(1,2, idx + 1, title = 'Diagonal-{}'.format(header), ylabel = 'dec acc (%)', xlabel = 'time (ms)')
			# if header == 'digit':
			# 	label_list = ['standard','110-nrlabels','100-nrlabels','90-nrlabels','80-nrlabels','70-nrlabels']
			# 	label_list = ['standard','100-nrlabels']
			# else:
			# 	label_list = ['standard','100-nrlabels','90-nrlabels','80-nrlabels','70-nrlabels']
			# 	label_list = ['standard','100-nrlabels']
		
			# for label in label_list:	% use this for plotting when downscaling; loop through labels
		 # 	#for label in np.sort(bdm[0][header].keys())[::3]:			
		 # 		X = np.stack([bdm[i][header][label] for i in range(len(bdm))])			

		 # 		x = X.mean(axis = 0)
		 # 		plt.plot(times, x, label = label)
			
		 # 	plt.legend(loc = 'best')
		 # 	plt.axhline(y = 1/8.0, color = 'black', ls = '--')
		 # 	sns.despine(offset=50, trim = False)

		 # plt.tight_layout()	
		 # plt.savefig(self.FolderTracker(['bdm'], filename = 'localizer-posterior-chans.pdf'))
		 # plt.close()

			X = np.stack([bdm[i][header]['standard'] for i in range(len(bdm))])
			print(X.shape)
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


if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '2' 
	os.environ['NUMEXP_NUM_THREADS'] = '2'
	os.environ['OMP_NUM_THREADS'] = '2'
	
	# Specify project parameters
	#project_folder = '/home/jalilov1/data/loc_pilot2' 
	#project_folder = '/home/jalilov1/BB/AB_R/data' #change the paths for within-task decoding!!!!!!
	project_folder = '/home/dvmoors1/BB/Josipa'
	os.chdir(project_folder)

	# initiate current project
	PO = Josipa()

	# seen vs unseen decoding Python test
	#PO.updatePickleSeenUnseen()

	for sj in [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35]:
		beh = pickle.load(open(PO.FolderTracker(extension=[
                    'AB', 'beh','processed'], filename='subject-{}_all.pickle'.format(sj)), "rb" ))
		beh = pd.DataFrame.from_dict(beh)
		eeg = mne.read_epochs(PO.FolderTracker(extension = ['AB','processed'], 
							filename = 'subject-{}_all-epo.fif'.format(sj)))
		# shift timings
		eeg =  cnd_time_shift(eeg, beh, cnd_info = {'TDTDD': -0.083*2,'TDDTD': -0.083*3}, cnd_header = 'condition')
		bdm = BDM(beh, eeg, to_decode = 'T2_seen', nr_folds = 10, elec_oi = 'all', downsample = 128, method = 'auc')
		bdm.Classify(sj, cnds = ['TDDTD', 'TDTDD'], cnd_header = 'condition', collapse = True,
		 			bdm_labels = [1,2], time = (-0.2, 0.8), nr_perm = 0, gat_matrix = False)




	# STEP 1: run task specific preprocessing 
	# for sj in [3]:
	
	# 	for session in range(1,4): #1,4
	# 		if session == 1:
	# 			task = 'localizer'
	# 			trigger = [22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38] # for localizer, 22-29 target numbers; 31-38 target letters
	# 			t_min = -0.4 
	# 			t_max = 1 

	# 		else:
	# 			task = 'AB'
	# 			trigger = [1002,1003,1004,1005,1006,1007,1008,1009] # for AB (start response interval)
	# 			t_min = -0.4   
	# 			t_max = 2  

	# 	   	# PO.updateBeh(sj = sj)
	# 	   	pass
	# 	 	PO.prepareEEG(sj = sj, session = session, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 	 			  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 	 			  trigger = trigger, project_param = project_param, 
	# 	 			  project_folder = project_folder, binary = binary, channel_plots = False, inspect = True, task = task)
	

	# STEP 2: run task specific analysis

	# first run within task decoding
	for sj in [1,3]:
		for task in ['localizer','AB']:
			beh, eeg = PO.loadDataTask(sj,(-0.2, 0.9), False, task = task)
			if task == 'localizer':
				decoding = 'identity'
				beh['identity'] = beh['digit'] + beh['letter'] # collapse letters and digits so that decoding can be run in one go
				cnds = ['letter','digit']
			elif task == 'AB':
				decoding = 'T1' 
				cnds = 'all' # or later on a list of conditions ['DDTTTT','TTTD']
				
			bdm = BDM(beh = beh, eeg= eeg, decoding=decoding, nr_folds = 10)
			bdm_acc = bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', bdm_labels = 'all', 
			 		factor = None, time = (-0.2, 0.9), nr_perm = 0, gat_matrix = True, downscale = False, save = False)
			pickle.dump(bdm_acc, open( PO.FolderTracker([task, 'bdm'],filename = '{}_{}_dec.pickle'.format(sj,decoding)), 'wb'))	
 
		#next run across task decoding
		PO.crossTaskBDM(sj=sj, gat_matrix = True)		


	# STEP 3: plot results
	PO.plotBDMwithin(to_plot = 'T1')	
	PO.plotcrossBDM()

