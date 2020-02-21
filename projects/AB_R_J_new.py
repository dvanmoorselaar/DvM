import matplotlib # comment out if running on tux
matplotlib.use('agg') # now it works via ssh connection; # comment out if running on tux

import os
import mne
import sys
import glob
import pickle
import math
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
from eeg_analyses.TF import * 
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
#from eeg_analyses.CTF import * 
#from eeg_analyses.Spatial_EM import * 
from visuals.visuals import MidpointNormalize
#from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *


# subject specific info
sj_info = {'1': {'tracker': (False, '', '','',''),  'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
		   '3': {'tracker': (False, '', '','',''),  'replace':{}},
		   '4': {'tracker': (False, '', '','',''),  'replace':{}},
		   '5': {'tracker': (False, '', '','',''),  'replace':{}},
		   '6': {'tracker': (False, '', '','',''),  'replace':{}},
		   '7': {'tracker': (False, '', '','',''),  'replace':{}},
		   '8': {'tracker': (False, '', '','',''),  'replace':{}},
		   '9': {'tracker': (False, '', '','',''),  'replace':{}},		  
		   '11': {'tracker': (False, '', '','',''),  'replace':{}},
		   '12': {'tracker': (False, '', '','',''),  'replace':{}},
		   '13': {'tracker': (False, '', '','',''),  'replace':{}},
		   '14': {'tracker': (False, '', '','',''),  'replace':{}},
		   '15': {'tracker': (False, '', '','',''),  'replace':{}},
		   '16': {'tracker': (False, '', '','',''),  'replace':{}},
		   '17': {'tracker': (False, '', '','',''),  'replace':{}},
		   '18': {'tracker': (False, '', '','',''),  'replace':{}},
		   '19': {'tracker': (False, '', '','',''),  'replace':{}}, 
		   '20': {'tracker': (False, '', '','',''),  'replace':{}},
		   '21': {'tracker': (False, '', '','',''),  'replace':{}},
		   '22': {'tracker': (False, '', '','',''),  'replace':{}},
		   '23': {'tracker': (False, '', '','',''),  'replace':{}},
		   '24': {'tracker': (False, '', '','',''),  'replace':{}},
		   '25': {'tracker': (False, '', '','',''),  'replace':{}},
		   '26': {'tracker': (False, '', '','',''),  'replace':{}},
		   '27': {'tracker': (False, '', '','',''),  'replace':{}},
		   '29': {'tracker': (False, '', '','',''),  'replace':{}}, 
		   '30': {'tracker': (False, '', '','',''),  'replace':{}}, 
		   '31': {'tracker': (False, '', '','',''),  'replace':{}}, 
		   '32': {'tracker': (False, '', '','',''),  'replace':{}}, # preprocess
		   '33': {'tracker': (False, '', '','',''),  'replace':{}}, # preprocess
		   '34': {'tracker': (False, '', '','',''),  'replace':{}}, # preprocess
		   '35': {'tracker': (False, '', '','',''),  'replace':{}}, # preprocess


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

	def matchBehnew(self, sj, events, trigger, task):
		'''

		'''

		missing = np.array([])
		if task == 'localizer':

			idx_trigger = np.sort(np.hstack([np.where(events[:,2] == i)[0] for i in trigger]))
			triggers = events[idx_trigger,2]
			beh = pd.DataFrame(index = range(triggers.size),columns=['condition', 'nr_trials', 'label'])

			beh['digit'] = 0 
			beh['digit'][(triggers > 21) * (triggers < 30)] = triggers[(triggers > 21) * (triggers < 30)]
			beh['letter'] = 0
			beh['letter'][(triggers > 30) * (triggers < 39)] = triggers[(triggers > 30) * (triggers < 39)]
			beh['condition'] = 'digit'
			beh['condition'][(triggers > 30) * (triggers < 39)] = 'letter'
			beh['nr_trials'] = range(triggers.size)
		
			print('detected {} epochs'.format(triggers.size))
			
		elif task == 'AB':
			
			# Find indices of beginning of fixation and change T1 triggers; only 4 events after the trigger has been sent and no target, so remove the trigger
			idx_end = np.where(events[:,2] == 51)[0]

			# Find all conditions of interest
			cnd_idx = np.where((events[:,2] > 60) *( events[:,2] < 69))[0] 
			
			# if sj == 18 and session == 3: # this subjects in session 3 had an extra trigger 64 at the very end - something went wrong with recording 
			if sj == 29 and session == 2:	# one extra condition code (61) at the very end; most of the last block was lost for this subject
				cnd_idx = cnd_idx[:len(idx_end)]

			cnds = events[cnd_idx,2]

			# check where the mismatch is for subject 18 session 2 run 1 // and subject 12 session 2
			#idx_end - position of the 51 trigger; check what is 20 places before that - should be a condition code that should aline perfectly with cnds
			#cnds = 803; idx_end = 804
		
			# for i in range(1,len(cnds)):
			# 	print(i) # the idx where the mismatch is + 1 is the missmatch
			# 	if (cnds [i] != events[:,2][idx_end-20][i]):
			# 		break
			
			# subject 12 in session 2 has an extra trigger 218 among condition codes at position 804
			# subject 18, session 2 has an extra condition trigger 191 
			if sj == 18 and session == 2:
				#delete idx_end[490] - 191 trigger 
				idx_end = np.delete(idx_end,490, axis=0)
			elif sj==12 and session ==2:
				#delete idx_end[804] - 218 trigger
				idx_end = np.delete(idx_end,804, axis=0)
			elif sj ==16 and session==3:
				idx_end = np.delete(idx_end,554, axis=0) # weird trigger 255
			elif sj==34 and session ==2:
				idx_end  = np.delete(idx_end,268, axis=0) #events[:,2][idx_end-20][268]) was 25

			# add a 1000 to T1 triggers (these are used for epoching; now only for condition 1 -7)
			cnd_1_7_mask = np.logical_and(cnds >= 61, cnds <= 67) # condition 8 will be added later
			events[(idx_end - 14)[cnd_1_7_mask],2] += 1000 # original code
				
			# create a dataframe that per trial contains all variables of interest
			beh = pd.DataFrame(index = range(cnd_idx.size),columns=['condition', 'nr_trials', 'T1','T2','T3','D1','D2','D3','D4','D5','Cnd_8-T1_pos','Cnd_1-T2_pos'])

			# populate beh
			# save condition info
			beh['nr_trials'] = range(1,cnd_idx.size + 1) 
			beh['condition'][cnds == 61] = 'T..DDDT'
			beh['condition'][cnds == 62] = 'TTDDD'
			beh['condition'][cnds == 63] = 'TDTDD'
			beh['condition'][cnds == 64] = 'TDDTD'
			beh['condition'][cnds == 65] = 'TTTDD'
			beh['condition'][cnds == 66] = 'TTDTD'
			beh['condition'][cnds == 67] = 'TDTTD'
			beh['condition'][cnds == 68] = 'DDTDD' #T is at variable positions 13-16
			
			# save T1 info per trial
			#beh['T1'] = events[idx_end - 14,2] - 1000
			t1 = events[idx_end - 14,2] - 1000
			# find T1 t1for cnds 1-7
			beh['T1'][cnd_1_7_mask] = t1[t1 < 10]
			# find T1 for cnd 8 and also its position in the stream
			idx_8 = np.where(events[:,2] == 68)[0]
			T1_list = []
			T1_pos_list = []
			for idx in idx_8:
				# loop over al possible positions
				for pos_idx in [14,15,16,17]: 
					t1_trigger = events[idx + pos_idx,2]
					if (t1_trigger < 10) * (t1_trigger > 1):
						# add a 1000 to T1 triggers (for cnd 8)
						#events[idx + pos_idx,2] += 1000
						T1_list.append(t1_trigger)
						T1_pos_list.append(pos_idx - 1)
						# add 1000 to 5th position in that condition
						events[idx + pos_idx - (pos_idx - 6),2] += 1000 
						break
	
			# correct!
			id_null = np.where(beh['condition']=='DDTDD') #indices of NaN values in condition 68
			beh['T1'][id_null[0]] = T1_list
			beh['Cnd_8-T1_pos'][id_null[0]]  = T1_pos_list 

			#this code is not adding the values in the correct order!
			#beh['T1'][cnds == 68] = T1_list	# this is not added in the correct order		
			#beh['Cnd_8-T1_pos'][cnds == 68] = T1_pos_list # also not added in the correct order	

			beh['T2'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 7,2]
			beh['T2'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 8,2]
			if sj == 18 and session ==3:
				beh['T2'][cnds == 64] = events[np.where(events[:,2] == 64)[0][:-1] + 9,2] # get the all except the last 64 trigger
			else:
				beh['T2'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 9,2]
			beh['T2'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 7,2]
			beh['T2'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 7,2]
			beh['T2'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 8,2]

			#find T2 in condiiton 61
			idx_1 = np.where(events[:,2] == 61)[0] # remove the last 61 for subjects 29 session 2 from this list
			
			if sj == 29 and session ==2:
				idx_1 = idx_1[:len(idx_1)-1]

			T2_list = []
			T2_pos_list = []

			for idx in idx_1:
			    # loop over al possible positions
				for pos_idx in [14,15,16,17]: 
					t2_trigger = events[idx + pos_idx,2]
					if (t2_trigger < 10) * (t2_trigger > 1):
						# add a 1000 to T1 triggers (for cnd 8)
						#events[idx + pos_idx,2] += 1000
						T2_list.append(t2_trigger)
						T2_pos_list.append(pos_idx - 1)
				
			id2_null = np.where(beh['condition']=='T..DDDT') #indices of NaN values in condition 61
			beh['T2'][id2_null[0]] = T2_list
			beh['Cnd_1-T2_pos'][id2_null[0]]  = T2_pos_list

			# save T3 info per trial
			beh['T3'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 8,2]
			beh['T3'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 9,2]
			beh['T3'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 9,2]			
				
			# save D1,D2,D3 info
			if sj ==29 and session ==2:
				temp=np.where(events[:,2] == 61)[0]
				temp_events =temp[:len(temp)-1] # The last 61 is excess
				beh['D1'][cnds == 61] = events[temp_events+ 7,2] # 61
				beh['D2'][cnds == 61] = events[temp_events + 8,2]
				beh['D3'][cnds == 61] = events[temp_events+ 9,2]
				beh['D4'][cnds == 61] = events[temp_events+ 10,2]
			else:
				beh['D1'][cnds == 61] = events[np.where(events[:,2] == 61)[0] + 7,2] # 61
				beh['D2'][cnds == 61] = events[np.where(events[:,2] == 61)[0] + 8,2]
				beh['D3'][cnds == 61] = events[np.where(events[:,2] == 61)[0] + 9,2]
				beh['D4'][cnds == 61] = events[np.where(events[:,2] == 61)[0] + 10,2]


			beh['D1'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 8,2] #62
			beh['D2'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 9,2]
			beh['D3'][cnds == 62] = events[np.where(events[:,2] == 62)[0] + 10,2]
			beh['D1'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 7,2] #63
			beh['D2'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 9,2]
			beh['D3'][cnds == 63] = events[np.where(events[:,2] == 63)[0] + 10,2]
			
			if sj == 18 and session ==3: # there was an extra trigger 64 at the very end but no events afterwards, so just remove 
				beh['D1'][cnds == 64] = events[np.where(events[:,2] == 64)[0][:-1] + 7,2] # get the all except the last 64 trigger
				beh['D2'][cnds == 64] = events[np.where(events[:,2] == 64)[0][:-1] + 8,2]
				beh['D3'][cnds == 64] = events[np.where(events[:,2] == 64)[0][:-1] + 10,2]
			else:
				beh['D1'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 7,2] #64
				beh['D2'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 8,2]
				beh['D3'][cnds == 64] = events[np.where(events[:,2] == 64)[0] + 10,2]

			beh['D1'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 9,2] #65
			beh['D2'][cnds == 65] = events[np.where(events[:,2] == 65)[0] + 10,2]
			beh['D1'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 8,2] #66
			beh['D2'][cnds == 66] = events[np.where(events[:,2] == 66)[0] + 10,2]
			beh['D1'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 7,2] #67
			beh['D2'][cnds == 67] = events[np.where(events[:,2] == 67)[0] + 10,2]
			beh['D1'][cnds == 68] = events[np.where(events[:,2] == 68)[0] + 6,2] #68
			beh['D2'][cnds == 68] = events[np.where(events[:,2] == 68)[0] + 7,2]
			beh['D3'][cnds == 68] = events[np.where(events[:,2] == 68)[0] + 8,2]
			beh['D4'][cnds == 68] = events[np.where(events[:,2] == 68)[0] + 9,2]
			beh['D5'][cnds == 68] = events[np.where(events[:,2] == 68)[0] + 10,2]	

			print('detected {} epochs'.format(idx_end.size))

		return beh, missing, events

	#self.saveBehnew(sj, task, session)


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

		# set subject specific parameter
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
		if sj == 30 and session == 3:
			eeg_runs = [1,2]

		EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
		                                   montage=None, preload=True, eog=eog) for run in eeg_runs])
		EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True)
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		           skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
		beh, missing, events = self.matchBehnew(sj, events, trigger, task = task)
		# EPOCH DATA (twice, once for ICA)
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(-0.2, 0), flt_pad = flt_pad) # baseline 200 ms before T1 trigger
		#print 'Josipa check whether the terminal prints again that bad channels have been removed after this statement' 
		EEGica = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(None,None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION

		epochs.selectBadChannels(channel_plots = True, inspect=True, RT = None)    
		epochs.artifactDetection(inspect=False, run = True)

		# make sure that bad channels are also removed from data used for ICA
		EEGica.selectBadChannels(channel_plots = False, inspect=False)

		# ICA (now applied on epoched data)
		epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = inspect)

		# EYE MOVEMENTS
		epochs.detectEye(missing, time_window=(t_min*1000, t_max*1000), tracker = tracker, tracker_shift = shift, start_event = start_event, extension = ext, eye_freq = t_freq)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		self.savePreprocessing(beh, epochs, events, trigger, task)
	
	

	def savePreprocessing(self, beh, eeg, events, trigger, task):
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
							filename = 'subject-{}_all-new.pickle'.format(sj)),'rb'), encoding='latin1') # for AB task, add -new to the file name
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
		beh_ab = pickle.load(open(self.FolderTracker(extension = ['AB','beh','processed'], filename = 'subject-{}_all-new.pickle'.format(sj)),'rb'))
		# STEP 2: downsample data
		locEEG.resample(128)
		abEEG.resample(128)

		# set general parameters
		s_loc, e_loc = [np.argmin(abs(locEEG.times - t)) for t in window] #defines the time interval for classification
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


		### Original  code:
		train_idx = np.sort(np.hstack([random.sample(np.where(beh_loc[to_decode_tr] == l)[0],min_tr_labels) for l in np.unique(train_labels)]))
		#Why is this sorted again?
		
		# set test labels
		#test_idx = np.where(np.array(beh_ab['condition']) == cnd)[0] # number test labels is not yet counterbalanced
		test_idx = range(np.array(beh_ab[to_decode_te]).size)

		# STEP 4: do classification
		lda = LinearDiscriminantAnalysis()

		# set training and test labels
		Ytr = beh_loc[to_decode_tr][train_idx] % 10 # double check whether this also works for letters; 
	
		
		# Problem for T2s, not all T2 were defined, so define T2 conditions (have to be temproally aligned!)
		if to_decode_te == 'T2':
			to_decode_te_conditions = ['TTDDD']
			#to_decode_te_conditions = ['TTDDD','TTTDD','TTDTD']
			idx_Yte = np.where(np.array([beh_ab['condition'] == c for c in to_decode_te_conditions]))[1] #finds indices of conditions
			Yte = beh_ab[to_decode_te][idx_Yte]
		else:
			Yte = np.array(beh_ab[to_decode_te]) #[test_idx]

		class_acc = np.zeros((nr_time, nr_test_time))
		label_info = np.zeros((nr_time, nr_test_time, nr_tr_labels))
	
		for tr_t in range(nr_time):
			print(tr_t)
			for te_t in range(nr_test_time):
				if not gat_matrix:
					te_t = tr_t
					
				# Triggers are retrieved from beh files, but these eeg epochs have been locked to T1 (position 5 actually), so how can you decode T2 based on this?
				Xtr = eegs_loc[train_idx,:,tr_t].reshape(-1, picks.size)
				if to_decode_te == 'T2':
					Xte = eegs_ab[idx_Yte,:,te_t].reshape(-1, picks.size)
				else:
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
					
		pickle.dump(class_acc, open(self.FolderTracker(extension = ['cross_task','bdm'], filename = 'subject-{}_{}_bdm-onecond.pickle'.format(sj,to_decode_te)),'wb'))
					

##### Average over 4 trials:


		# train_idx = np.sort(np.hstack([random.sample(np.where(beh_loc[to_decode_tr] == l)[0],min_tr_labels) for l in np.unique(train_labels)]))

		#Randomized indices of the category to decode
		#rand_class=np.hstack([random.sample(np.where(beh_loc[to_decode_tr] == l)[0],min_tr_labels) for l in np.unique(train_labels)])
		
		#set test labels
		#test_idx = np.where(np.array(beh_ab['condition']) == cnd)[0] # number test labels is not yet counterbalanced
		# test_idx = range(np.array(beh_ab[to_decode_te]).size)

		# STEP 4: do classification
		# lda = LinearDiscriminantAnalysis()

		# class_acc = np.zeros((nr_time, nr_test_time))
		# label_info = np.zeros((nr_time, nr_test_time, nr_tr_labels))
	
		# for tr_t in range(nr_time):
		# 	print tr_te
		#  	for te_t in range(nr_test_time):
		#  		if not gat_matrix:
		#  			te_t = tr_t

				#Average over 4 random trials in the training set to increase the signal to noise of decoding and then create an array
				# Xtr_r4 = eegs_loc[train_idx,:,tr_t].reshape(-1,picks.size)
				# Xtr = np.array([np.mean(Xtr_r4[4*j:4*(j+1),:] , axis=0) for j in range(Xtr_r4.shape[0]/4)]).squeeze()

				#Testing set, average over 4 trials in the testing set
				# Xte_r4 = eegs_ab[train_idx,:,te_t].reshape(-1, picks.size)
				# Xte = np.array([np.mean(Xte_r4[4*j:4*(j+1),:] , axis=0) for j in range(Xte_r4.shape[0]/4)]).squeeze()
				#Xte = eegs_ab[test_idx,:,te_t].reshape(-1, picks.size)

		  		# set training and test labels
				# Ytr = beh_loc[to_decode_tr][train_idx[0:Xtr.shape[0]]] % 10 # double check whether this also works for letters; 0:158 because there are 158 points after averaging in Xtr
				# Yte = np.array(beh_ab[to_decode_te])[0:Xtr.shape[0]] #[test_idx]

		# 		lda.fit(Xtr,Ytr)
		# 		predict = lda.predict(Xte) # maybe there is no need to average over Xte, these are the labels you are classifying 

		# 		if not gat_matrix:
		# 			#class_acc[tr_t, :] = sum(predict == Yte)/float(Yte.size)
		# 			class_acc[tr_t, :] = np.mean([sum(predict[Yte == y] == y)/ float(sum(Yte == y)) for y in np.unique(Yte)])
		# 			label_info[tr_t, :] = [sum(predict == l) for l in np.unique(Ytr)]	
		# 		else:
		# 			#class_acc[tr_t, te_t] = sum(predict == Yte)/float(Yte.size)
		# 			class_acc[tr_t, te_t] = np.mean([sum(predict[Yte == y] == y)/ float(sum(Yte == y)) for y in np.unique(Yte)])
		# 			label_info[tr_t, te_t] = [sum(predict == l) for l in np.unique(Ytr)]


		
	
	def splitEpochs(self):
		'''		

		'''

		pass

	def plotBDMwithin(self, to_plot = 'T1', task ='localizer'):
		'''	
		THIS FUNCTION PLOTS WITHIN TASK DECODING - diagonal
		'''

		# set plotting parameters
		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/2.0)
		vmin, vmax = 0.45, 0.6
		ylim = (0.45,0.6)


		with open(self.FolderTracker(['bdm','target_event'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	
		
		#for task in ['localizer''AB']:
		# files = glob.glob(self.FolderTracker([task, 'bdm'], filename = '*_*_dec.pickle'))
		# bdm = [pickle.load(open(file,'rb')) for file in files]
			# collapse data
		if task == 'localizer':
			files = glob.glob(self.FolderTracker(['bdm','post','identity','baseline'], filename = 'class_*-broad.pickle')) #'class_*-broad.pickle'
			bdm = [pickle.load(open(file,'rb')) for file in files]
			for idx, cnd in enumerate(['digit','letter']):
				X = np.stack([bdm[i][cnd]['standard'] for i in range(len(bdm))])			
					# plot diagonal
				ax = plt.subplot(1,2, [1,2][idx], title = 'Diagonal-{}_post'.format(cnd), ylabel = 'AUC', xlabel = 'time (ms)')
				diag = np.diag(np.mean(X,0))

				plt.plot(times, diag)
				plt.axhline(y = 1/2.0, color = 'black', ls = '--')
				sns.despine(offset=50, trim = False)
				plt.ylim(ylim)
				plt.yticks([ylim[0],1/2.0,ylim[1]])
				plt.xticks(np.arange(0,200,600))

					# plot gat matrix
				# ax = plt.subplot(3,2, [1,3][idx], title = 'GAT-{}'.format(cnd), ylabel = 'train time (ms)', xlabel = 'test time (ms)')
				# plt.imshow(X.mean(0), norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
				# 	cmap = cm.bwr, interpolation = None, vmin = 0.09, vmax = 0.15)
				# plt.colorbar()
				# sns.despine(offset=50, trim = False)

		elif task == 'AB':
			
			files = glob.glob(self.FolderTracker(['pre-T1-baseline','bdm', 'within', 'all', 'target_event_targets'], filename = 'class_*-target_event.pickle'))
			bdm = [pickle.load(open(file,'rb')) for file in files]

			X = np.stack([bdm[i]['all']['standard'] for i in range(len(bdm))])			
			# plot diagonal
			ax = plt.subplot(1,2,1, title = 'Boost versus bounce targets', ylabel = 'AUC', xlabel = 'time (ms)')
			diag = X.mean(0)
			plt.plot(times, diag)
			plt.axhline(y = 1/2.0, color = 'black', ls = '--')
			sns.despine(offset=10, trim = False)
			plt.ylim(ylim)
			plt.yticks([ylim[0],1/2.0,ylim[1]])
			plt.xticks(np.arange(0,200,600))
		
		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['within_task', 'bdm'], filename = 'within_dec_AB_targets_bb.pdf'))
		plt.close()

		# if ab:
		
		# 	X_var = [X_dec_ab ,X_dec_noab ]
			
		# 	for j in range (0,2):
		# 		# plot diagonal - all selected
		# 		ax = plt.subplot(2,2,[2,4][j], title = 'Diagonal-AB_{}'.format(to_plot), ylabel = 'Decoding accuracy (%)', xlabel = 'Time (ms)')
		# 		diag = np.diag(np.mean(X_var[j].mean(0),0)) #average across conditions and across participants
		# 		plt.plot(times, diag)
		# 		plt.axhline(y = 1/8.0, color = 'black', ls = '--')
		# 		sns.despine(offset=50, trim = False)

		# 		# plot gat matrix - all selected
		# 		ax = plt.subplot(2,2,[1,3][j], title = 'GAT-AB_{}'.format(to_plot), ylabel = 'Train time (ms)', xlabel = 'Test time (ms)')
		# 		plt.imshow(np.mean(X_var[j].mean(0),0), norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
		# 			cmap = cm.bwr, interpolation = None, vmin = 0.09, vmax = 0.15)
		# 		plt.colorbar()
		# 		sns.despine(offset=50, trim = False)

		# plt.tight_layout()	
		# plt.savefig(self.FolderTracker(['within_task', 'bdm'], filename = 'within_dec_AB_{}.pdf'.format(to_plot)))
		# plt.close()


	# MEAN OVER ALL SUBJECTS (diagonal and GAT) 

	def plotcrossBDM(self, to_plot='T1'):
		'''

		'''
		files = glob.glob(self.FolderTracker(['cross_task','bdm'], filename = 'subject-*{}_bdm-onecond.pickle'.format(to_plot)))
		bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
		#with open(self.FolderTracker(['bdm','identity'], filename = 'plot_dict.pickle') ,'rb') as handle:
		#	info = pickle.load(handle)
		
		times = np.linspace(-0.2,0.9,128)
		norm = MidpointNormalize(midpoint=1/8.0)	
		plt.figure(figsize = (20,10))
		# plot diagonal
		ax = plt.subplot(1,2,1, title = 'Diagonal', ylabel = 'Decoding accuracy (%)', xlabel = 'Time (ms)')
		diag = np.diag(np.mean(bdm,0))
		plt.plot(times, diag)
		plt.axhline(y = 1/8.0, color = 'black', ls = '--')
		sns.despine(offset=50, trim = False)

		#plot gat matrix
		ax = plt.subplot(1,2,2, title = 'GAT', ylabel = 'Train time (ms)', xlabel = 'Test time (ms)')
		plt.imshow(bdm.mean(0), norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
		cmap = cm.bwr, interpolation = None, vmin = 0.09, vmax = 0.15)
		plt.colorbar()
		sns.despine(offset=50, trim = False)
											
		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['cross_task','bdm'], filename = 'cross-task_{}.pdf'.format(to_plot)))
		plt.close()

	def clusterPlot(self, X1, X2, p_val, times, y, color):

		sig_cl = clusterBasedPermutation(X1, X2, p_val = p_val)
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0] + 1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size)* y, color = color, ls = '--')


	def plotcrossGAT(self, gat_matrix = True, collapsed = True, colorbar=True):
		'''
		# Basic GAT plot for quick inspection of decoding
		'''

		# general plotting parameters
		f = plt.figure(figsize = (40,30))
		times = np.linspace(-200,900,140)
		norm = MidpointNormalize(midpoint=1/2.0)
		vmin, vmax = 0.45, 0.6




		for i, T in enumerate(['T2']):
			files = glob.glob(self.FolderTracker(['pre-T1-baseline', 'bdm','all','cross', 'OrderRev'], filename = 'class_*-{}.pickle'.format(T)))
			 

			for plt_idx, cnd in enumerate(['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD']):
				bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
				if cnd not in bdm[0].keys():
					continue
				X = np.stack([b[cnd]['standard'] for b in bdm])
				
				ax = plt.subplot(3,3,plt_idx + 1, title = cnd, ylabel = 'Time(ms)', xlabel = 'Time (ms)')
				# Compute tresholded data
		
				X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
				contour = np.zeros(X_thresh.shape, dtype = bool)
				contour[X_thresh != 1/2.0] = True

				im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
						origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
						vmin = 0.45, vmax = 0.6)
				# Plot contoures around significant datapoints
				plt.contour(contour, origin = 'lower',extent=[times[0],times[-1],times[0],times[-1]])
				plt.yticks([0,200,600])
				plt.xticks([0,200,600])
				sns.despine(offset=10, trim = False)
				plt.axhline(y = 0, ls = '--',color='k')
				plt.axvline(x = 0,  ls = '--',color='k')

		# add a colorbar for all figures
		cb_ax = f.add_axes([0.99, 0.2, 0.01, 0.61])
		cbar = f.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])

		sns.despine(offset = 50, trim = False)
		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['pre-T1-baseline', 'bdm','all','cross'], filename = 'cross-task_T2_OrderRev.pdf'))
		plt.close()



	def plotDecoding(self, plot = 'GAT'):
		'''

		'''
		 # general plotting parameters
		f = plt.figure(figsize = (40,30))
		times = np.linspace(-200,900,140)
		norm = MidpointNormalize(midpoint=1/2.0)
		vmin, vmax = 0.45, 0.6
		ylim = (0.45,0.6)
	
		conditions = ['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD', 'collapsed'] 
		conditions = ['boost', 'bounce']


		for i, T in enumerate(['targets']):
		# for ib, B in enumerate(['incorr']):
			ax = plt.subplot(1,2, i + 1, title = T, ylabel = 'AUC', xlabel = 'Time (ms)')
			# read in T specific data

			files = glob.glob(self.FolderTracker(['pre-T1-baseline','bdm','cross','all'], filename = 'class_*.pickle'))
			# ['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','collapsed']
			for plt_idx, cnd in enumerate(conditions): #'DDTDD'
					
				ax = plt.subplot(3,3,plt_idx+1, title = cnd, ylabel = ylim, xlabel = 'Time (ms)')
				bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
			

				if cnd not in bdm[0].keys():
					continue

				X = np.stack([b[cnd]['standard'] for b in bdm])
			
				if plot == 'GAT':
					# X_ = np.array(X_all).mean(0) # mean over conditions
					ax = plt.subplot(3,3,plt_idx+1, title = cnd, ylabel = 'Time(ms)', xlabel = 'Time (ms)')
			        # Compute tresholded data
			      
					X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
					contour = np.zeros(X_thresh.shape, dtype = bool)
					contour[X_thresh != 1/2.0] = True

					im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
							origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
							vmin = 0.45, vmax = 0.6)

			        # Plot contoures around significant datapoints
					plt.contour(contour, origin = 'lower',extent=[times[0],times[-1],times[0],times[-1]])
					plt.yticks([0,200,600])
					plt.xticks([0,200,600])
					sns.despine(offset=10, trim = False)
					plt.axhline(y = 0, ls = '--',color='k')
					plt.axvline(x = 0,  ls = '--',color='k')

					# add a colorbar for all figures
					cb_ax = f.add_axes([0.95, 0.2, 0.01, 0.61])  # left, bottom, width, height (range 0 to 1)
					cbar = f.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])
					sns.despine(offset = False, trim = False)
					
				elif plot == 'diagonal':
					
					diag = np.stack([par for par in X])
					clusterPlot(diag, 1/2.0, p_val = 0.05, times = times, y = 0.46 + i * 0.002, color = ['blue','green', 'purple'][i])
					err_t, X_t = bootstrap(diag)
					plt.plot(times, X_t, label = T, color = ['blue','green', 'purple'][i])
					plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = ['blue','green', 'purple'][i])
					plt.legend(loc = 'upper right', frameon = False)
					plt.axhline(y = 0.5, ls = '--',color='k')
					plt.axvline(x = 0,  ls = '--',color='k')

					plt.ylim(ylim)
					plt.yticks([ylim[0],1/2.0,ylim[1]])
					plt.xticks([0,250,500]) 
					PO.beautifyPlot(y = 0, xlabel = 'Time (ms)', ylabel = 'AUC')
					sns.despine(offset = 50, trim = False)

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['pre-T1-baseline','bdm','cross','all'], filename ='Boost_bounce_targets.pdf'))


	def cnd_time_shift_variablepos(self, EEG, beh, cnd_info, cnd_header, pos_info, pos_header):
		"""Function shifts the timings of conditions where T1 or T2 are not on fixed positions

		Arguments:
			EEG {object} -- MNE Epochs object
			beh {dataframe} -- Dataframe with behavior info
			cnd_info {dict} -- For each key in cnd_info data will be shifted according to 
								the specified time (in seconds). E.G., {neutral: 0.1}
			cnd_header {str} -- column in beh that contains the keys specified in cnd_info
		
		Returns:
			EEG {object} -- MNE Epochs object with shifted timings
		"""

		print('Data will be artificially shifted. Be carefull in selecting the window of interest for further analysis')
		print('Original timings range from {} to {}'.format(EEG.tmin, EEG.tmax))
		# loop over all conditions
		
		for cnd in cnd_info.keys():
			if cnd == 'T..DDDT' or cnd == 'DDTDD':
				print('Original timings range from {} to {}'.format(EEG.tmin, EEG.tmax))

				# loop over all conditions
				for pos in pos_info.keys():
					to_shift = pos_info[pos]
					to_shift = int(np.diff([np.argmin(abs(EEG.times - t)) for t in (0,to_shift)]))
					if to_shift < 0:
						print('EEG data is shifted backward in time for all {} trials'.format(pos))
					elif to_shift > 0:
						print('EEG data is shifted forward in time for all {} trials'.format(pos))

					mask = (beh[pos_header] == int(pos[-2:])).values # find indices of epochs to shift
					# do actual shifting
					
					EEG._data[mask] = np.roll(EEG._data[mask], to_shift, axis = 2)
			else:
				# set how much data needs to be shifted
				to_shift = cnd_info[cnd]
				to_shift = int(np.diff([np.argmin(abs(EEG.times - t)) for t in (0,to_shift)]))
				if to_shift < 0:
					print('EEG data is shifted backward in time for all {} trials'.format(cnd))
				elif to_shift > 0:
					print('EEG data is shifted forward in time for all {} trials'.format(cnd))	

				# find indices of epochs to shift
				mask = (beh[cnd_header] == cnd).values
				
				# do actual shifting
				EEG._data[mask] = np.roll(EEG._data[mask], to_shift, axis = 2)

		return EEG

	def cnd_time_shift_boost_bounce(self, EEG, beh, cnd_info, cnd_header):

		for cnd in cnd_info.keys():
			print('Original timings range from {} to {}'.format(EEG.tmin, EEG.tmax))
			for i, event_shift in enumerate(cnd_info[cnd]):
				to_shift = int(np.diff([np.argmin(abs(EEG.times - t)) for t in (0,event_shift)]))
				mask = ((beh[cnd_header] == cnd) &  (beh.event_nr == i + 1)).values
				EEG._data[mask] = np.roll(EEG._data[mask], to_shift, axis = 2)

		return EEG



	def correct_T1_filter_beh(self, sj, beh):
		'''
		# adds a column to beh that indicates whether or not T1 was read in correct
		'''	

		# read in two sessions data (matlab output)
		outputs = []
		for session in [1,2]:
			file = FolderStructure.FolderTracker(extension = ['BEH' ,'results'], 
				filename = 'abr_datamat_{}_sb{}.csv'.format(session,sj))
			outputs.append(pd.read_csv(file))
		outputs = pd.concat(outputs)

		# get the clean trials and control for collapsing
		idx_collapse = np.where(np.diff(beh['nr_trials'])<0)[0][0]
		trial_idx = beh['nr_trials'].values
		trial_idx[idx_collapse+1:] += 1072
		
		# now select the correct t1's
		# beh['T1_correct'] = outputs['t1_ide_any'].values[trial_idx- 1]
		beh['T1_correct'] = outputs['t1_det_correct'].values[trial_idx- 1] 

		return beh 

	def correct_T2_filter_beh(self, sj, beh):
		'''
		# adds a column to beh that indicates whether or not T1 was read in correct
		'''	

		# read in two sessions data (matlab output)
		outputs = []
		for session in [1,2]:
			file = FolderStructure.FolderTracker(extension = ['BEH' ,'results'], 
				filename = 'abr_datamat_{}_sb{}.csv'.format(session,sj))
			outputs.append(pd.read_csv(file))
		outputs = pd.concat(outputs)

		# get the clean trials and control for collapsing
		idx_collapse = np.where(np.diff(beh['nr_trials'])<0)[0][0]
		trial_idx = beh['nr_trials'].values
		trial_idx[idx_collapse+1:] += 1072
		
		# now select the correct t2's
		# beh['T2_correct'] = outputs['t2_ide_correct'].values[trial_idx- 1] # trial index filters csv file so that only trials that match eeg epochs remain in the analysis
		beh['T2_correct'] = outputs['t2_ide_any'].values[trial_idx- 1]
		#beh['T1_correct'] = outputs['t1_det_correct'].values[trial_idx- 1] 

		return beh

	def correct_T3_filter_beh(self, sj, beh):
		'''
		# adds a column to beh that indicates whether or not T1 was read in correct
		'''	

		# read in two sessions data (matlab output)
		outputs = []
		for session in [1,2]:
			file = FolderStructure.FolderTracker(extension = ['BEH' ,'results'], 
				filename = 'abr_datamat_{}_sb{}.csv'.format(session,sj))
			outputs.append(pd.read_csv(file))
		outputs = pd.concat(outputs)

		# get the clean trials and control for collapsing
		idx_collapse = np.where(np.diff(beh['nr_trials'])<0)[0][0]
		trial_idx = beh['nr_trials'].values
		trial_idx[idx_collapse+1:] += 1072
		
		# now select the correct t1's
		beh['T3_correct'] = outputs['t3_ide_any'].values[trial_idx- 1]
		#beh['T1_correct'] = outputs['t1_det_correct'].values[trial_idx- 1] 

		return beh

	
	# INDIVIDUAL DATA PLOTS

	# def plotcrossBDM(self):
	# 	'''

	# 	'''
        
	# 	files = glob.glob(self.FolderTracker(['cross_task','bdm'], filename = 'subject-*_bdm.pickle'))
	# 	bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
	# 	plt.suptitle('Diagonal decoding performance')
	# 	times = np.linspace(-0.2,0.9,128)*1000
	# 	norm = MidpointNormalize(midpoint=1/8.0)	
	# 	plt.figure(figsize = (20,10))

   		# plot individual diagonal
  #  		for s in range(len(bdm)):
  #  			plt.suptitle('Diagonal')
		# 	ax = plt.subplot(2,4,s+1,  ylabel = 'Accuracy (%)', xlabel = 'Time (ms)')
		# 	diag = np.diag(bdm[s])
		# 	plt.plot(times, diag, 'r')
		# 	ax.xaxis.label.set_size(18)
		# 	ax.yaxis.label.set_size(18)
	 # 		plt.xticks(np.arange(0, 1100, step=500))
	 # 		ax.tick_params(top='off', bottom='on', left='on', right='off', labelleft='on', labelbottom='on',labelsize = 18 )
		# 	plt.axhline(y = 1/8.0, color = 'black', ls = '--')
	 # 		sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=30 , trim=False)
		
		# plt.tight_layout()	
		# plt.savefig(self.FolderTracker(['cross_task','bdm'], filename = 'cross-task-persubj_diagonal.pdf'))
		# plt.close()

		#  #plot individual GAT
		# for s in range(len(bdm)):
		# 	plt.suptitle('GAT')
		# 	ax = plt.subplot(2,4,s+1, ylabel = 'Train time (ms)', xlabel = 'Test time (ms)')
		# 	plt.imshow(bdm[s], norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], cmap=cm.bwr, 
		# 		interpolation = None, vmin = 0.09, vmax = 0.15)
		# 	cbar=plt.colorbar()
		# 	#cbar = fig.colorbar(surf, aspect=20, fraction=.12,pad=.02)
		# 	#cbar.set_label('Accuracy', size=18)
		# 	cbar.ax.tick_params(labelsize=14) 
		# 	ax.xaxis.label.set_size(18)
		# 	ax.yaxis.label.set_size(18)
	 # 		plt.xticks(np.arange(0, 1100, step=500))
	 # 		ax.tick_params(top='off', bottom='on', left='on', right='off', labelleft='on', labelbottom='on',labelsize = 18 )
		# 	sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset= 30 , trim=False)

		# plt.tight_layout()	
		# plt.savefig(self.FolderTracker(['cross_task','bdm'], filename = 'cross-task-persubj_GAT.pdf'))
		# plt.close()

		

	def WithinDecoding(self, task='AB', decoding = 'targets', d_type = 'boostbounce'):


		# d_type = 'boostbounce' when decoding stimulus identity within boost and bounce conditions
		# d_type = 'stimulus' when decoding stimulus identity

		# WITHIN TASK DECODING
		#1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35
		for sj in [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35]: 

			for task in ['AB']: 
				embed()
				beh, eeg = PO.loadDataTask(sj,(-0.2, 0.9), False, task = task)

				if task == 'localizer':
					decoding = 'identity'
					beh['identity'] = beh['digit'] + beh['letter'] # collapse letters and digits so that decoding can be run in one go
					cnds = ['letter','digit']
					excl_factor = None

				elif task == 'AB':

					# Applied to all targets 
					beh = PO.correct_T1_filter_beh(sj,beh)
					embed()


					# #add a column to beh data structure

					# beh = PO.correct_T2_filter_beh(sj, beh) # check what you are selecting!
					# # select which labels from that column you want to exclude
					# excl_factor = dict(T2_correct = [3,4])

				
					# # ALL CONDITIONS 
					if decoding == 'T2':	
				
						# excl_factor = None
						# beh = PO.correct_T2_filter_beh(sj,beh)

						cnds = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD']
						eeg =  PO.cnd_time_shift_variablepos(eeg, beh, cnd_info = {'T..DDDT': 0,'TTDDD': -0.083, 'TDTDD': -0.083*2,'TDDTD': -0.083*3,'TTTDD': -0.083,'TTDTD': -0.083,'TDTTD': -0.083*2}, 
												cnd_header = 'condition', pos_info = {'pos_13': -0.664,'pos_14': -0.747,'pos_15': -0.83,'pos_16': -0.913}, pos_header = 'Cnd_1-T2_pos')
						
						# embed()
						beh['T2'] = beh['T2'][~beh['T2'].isnull()].astype(int) 

						# cnds = ['TDTDD','TDDTD','TDTTD'] #just AB conditions
						# eeg =  PO.cnd_time_shift_variablepos(eeg, beh, cnd_info = {'TDTDD': -0.083*2,'TDDTD': -0.083*3,'TDTTD': -0.083*2}, cnd_header = 'condition', pos_info = {'pos_13': -0.664,'pos_14': -0.747,'pos_15': -0.83,'pos_16': -0.913}, pos_header = 'Cnd_1-T2_pos')
						

					elif decoding == 'T3':
						cnds = ['TTTDD','TTDTD','TDTTD']
						eeg =  cnd_time_shift(eeg, beh, cnd_info = {'TTTDD': -0.083*2, 'TTDTD': -0.083*3,'TDTTD': -0.083*3}, cnd_header = 'condition')
						
						beh['T3'] = beh['T3'][~beh['T3'].isnull()].astype(int) 

					elif decoding == 'D1':

						cnds = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD']
						eeg =  cnd_time_shift(eeg, beh, cnd_info = {'T..DDDT': -0.083, 'TTDDD': -0.083*2, 'TDTDD': -0.083,'TDDTD': -0.083, 'TTTDD': -0.083*3, 'TTDTD': -0.083*2,'TDTTD': -0.083}, cnd_header = 'condition')


						for en, i in enumerate(beh.D1):
							if i > 1000:
								beh['D1'][en]= i-1000 # to eliminate high triggers that D1s on position 5 received 

						beh['D1'] = beh['D1'].astype(int)


					elif decoding == 'D2':
						cnds = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD']
					elif decoding == 'D3':
						cnds = ['T..DDDT','TTDDD','TDTDD','TDDTD','DDTDD']
					

					elif decoding == 'T1':
						cnds = 	['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] 
						eeg = PO.cnd_time_shift_variablepos(eeg, beh, cnd_info = {'DDTDD': 0}, cnd_header = 'condition', pos_info = {'pos_13': -0.664,'pos_14': -0.747,'pos_15': -0.83,'pos_16': -0.913}, pos_header = 'Cnd_8-T1_pos') # determine how many positions to shift based on T1 position in beh file
					 	# cnds = 'all'


					elif decoding == 'targets' or decoding == 'distractors' or decoding == 'TD':
						# for boostbounce
						# Update test beh so we can decode boost versus bounce - defines which positions (5-9) were boosted and bounced
						
						embed()
						beh['D_P1'], beh['D_P2'], beh['D_P3'] = None, None, None
						beh['T_P2'], beh['T_P3'] = None, None
						for cat in ['T','D']:
							for cnd in [ 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD']:
								idx = [i for i, letter in enumerate(cnd) if letter == cat]
								for pos_idx, pos in enumerate(['P1','P2','P3'][:len(idx)]):
									cat_idx = idx[pos_idx]
									b_or_b = 'boost' if cnd[cat_idx - 1] == 'T' else 'bounce'
									if cat != 'T' or pos != 'P1':
										beh.loc[beh.condition == cnd, '{}_{}'.format(cat, pos)] = b_or_b	

						# for targets distractors
						# add target / distractor labels so you can decode within that category, but you have to do this per temporal position; then duplicate epochs and then create one long column that has all those labels collapsed
						beh ['ST_P1'], beh ['ST_P2'], beh ['ST_P3'], beh ['ST_P4'], beh ['ST_P5'] = None, None, None, None, None
						beh ['L1'], beh ['L2'], beh ['L3'], beh ['L4'], beh ['L5'] = None, None, None, None, None # store which labels were shown per position
						

						for cat in ['T','D']:
							# for cnd in ['TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD']:
							# for cnd in ['TTDDD','T..DDDT']:
							for cnd in ['TDDTD','T..DDDT']:

								if cnd == 'T..DDDT':
									cnd_temp = 'TDDDT'
									idx = [i for i, letter in enumerate(cnd_temp) if letter == cat]
								else: 
									idx = [i for i, letter in enumerate(cnd) if letter == cat]
							

								for i, j in enumerate (idx):
									pos = ['P1','P2','P3','P4','P5'][idx[i]]
									beh.loc[beh.condition == cnd, 'ST_{}'.format(pos)] = cat


								if  cnd == 'TTDDD':
									beh.loc[beh.condition == cnd, 'L1'] = beh['T1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L2'] = beh['T2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L3'] = beh['D1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L4'] = beh['D2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L5'] = beh['D3'][beh['condition'] == cnd]
								elif cnd == 'TDTDD':
									beh.loc[beh.condition == cnd, 'L1'] = beh['T1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L2'] = beh['D1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L3'] = beh['T2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L4'] = beh['D2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L5'] = beh['D3'][beh['condition'] == cnd]
								elif cnd == 'TDDTD':
									beh.loc[beh.condition == cnd, 'L1'] = beh['T1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L2'] = beh['D1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L3'] = beh['D2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L4'] = beh['T2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L5'] = beh['D3'][beh['condition'] == cnd]
								elif cnd == 'TTTDD':
									beh.loc[beh.condition == cnd, 'L1'] = beh['T1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L2'] = beh['T2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L3'] = beh['T3'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L4'] = beh['D1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L5'] = beh['D2'][beh['condition'] == cnd]
								elif cnd == 'TTDTD':
									beh.loc[beh.condition == cnd, 'L1'] = beh['T1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L2'] = beh['T2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L3'] = beh['D1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L4'] = beh['T3'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L5'] = beh['D2'][beh['condition'] == cnd]
								elif cnd == 'TDTTD':
									beh.loc[beh.condition == cnd, 'L1'] = beh['T1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L2'] = beh['D1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L3'] = beh['T2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L4'] = beh['T3'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L5'] = beh['D2'][beh['condition'] == cnd]
								elif cnd == 'T..DDDT':
									beh.loc[beh.condition == cnd, 'L1'] = beh['T1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L2'] = beh['D1'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L3'] = beh['D2'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L4'] = beh['D3'][beh['condition'] == cnd]
									beh.loc[beh.condition == cnd, 'L5'] = beh['T2'][beh['condition'] == cnd]
										
									

                        # beh= beh.loc[:,~beh.columns.str.startswith('ST')]

                        #  to decode target labels in boost and bounce conditions
						if decoding == 'targets' and d_type == 'boostbounce': 

							dupl_eeg_idx = []
							test_beh['event_nr'] = 1
							for trial_idx in range(test_beh.shape[0]):
								# check whether trial should be duplicated more than once
								if test_beh.T_P3[trial_idx] in ['boost','bounce']:
									test_beh = test_beh.append(test_beh.loc[trial_idx], ignore_index = True) 
									test_beh['event_nr'].iloc[-1] = 2
									dupl_eeg_idx.append(trial_idx)
							test_eeg = mne.concatenate_epochs([test_eeg, test_eeg[dupl_eeg_idx]])


							# shift timings to control for different positions
							eeg = PO.cnd_time_shift_boost_bounce(eeg, beh, cnd_info = {'TTDDD': [-0.083], 'TDTDD': [-0.083*2],'TDDTD': [-0.083*3],
																									'TTTDD': [-0.083, -0.083*2],'TTDTD': [-0.083, -0.083*3],
																									'TDTTD': [-0.083*2, -0.083*3]}, cnd_header = 'condition')
							
							# create target event column (the one that is actually used for decoding) - combines 'T_P2' and 'T_P3' into one column
							beh['target_event'] = np.nan
							beh['target_event'][beh['event_nr'] == 1] = beh['T_P2'][beh['event_nr'] == 1]
							beh['target_event'][beh['event_nr'] == 2] = beh['T_P3'][beh['event_nr'] == 2]

							
						elif decoding == 'distractors' and d_type == 'boostbounce': 

							# first decode boost versus bounce targets (duplicate trials to be able to collapse)
							dupl_eeg_p2 = []
							dupl_eeg_p3 = []
							beh['event_nr'] = 1

							for trial_idx in range(beh.shape[0]):
								# check whether trial should be duplicated more than once
								if beh.D_P2[trial_idx] in ['boost','bounce']: # don't need to duplicate for D_P1
									beh = beh.append(beh.loc[trial_idx], ignore_index = True) 
									beh['event_nr'].iloc[-1] = 2
									# at the moment only data array within mne object is duplicated
									dupl_eeg_p2.append(eeg._data[trial_idx])
							eeg._data = np.append(eeg._data, np.stack(dupl_eeg_p2), axis = 0)

							
							for trial_idx in range(sum(beh.event_nr==1)):	# loop through the original beh structure (without duplicates)	
								if beh.D_P3[trial_idx] in ['boost','bounce']: 
									beh = beh.append(beh.loc[trial_idx], ignore_index = True) 
									beh['event_nr'].iloc[-1] = 3
									# at the moment only data array within mne object is duplicated
									dupl_eeg_p3.append(eeg._data[trial_idx])

							eeg._data = np.append(eeg._data, np.stack(dupl_eeg_p3), axis = 0)


							# shift timings to control for different positions
							eeg = PO.cnd_time_shift_boost_bounce(eeg, beh, cnd_info = {'TTDDD': [-0.083*2, -0.083*3, -0.083*4], 'TDTDD': [-0.083,-0.083*3, -0.083*4],'TDDTD': [-0.083,  -0.083*2, -0.083*4],
																									'TTTDD': [-0.083*3, -0.083*4],'TTDTD': [-0.083*2, -0.083*4],
																									'TDTTD': [-0.083, -0.083*4]}, cnd_header = 'condition')

							# create target event column (the one that is actually used for decoding)
							beh['target_event'] = np.nan
							beh['target_event'][beh['event_nr'] == 1] = beh['D_P1'][beh['event_nr'] == 1]
							beh['target_event'][beh['event_nr'] == 2] = beh['D_P2'][beh['event_nr'] == 2]
							beh['target_event'][beh['event_nr'] == 3] = beh['D_P3'][beh['event_nr'] == 3]


						elif decoding == 'TD': # decoding within target and distractor labels (stimulus identity)
						
						#	# there can be max 3 targets in total, and for T1 there needs to be no duplication
							dupl_eeg_idx_t2 = [] 
							beh['event_nr'] = 1 # to keep track which trial was a duplcate 
							beh['T'] = None
							beh['target_event'] = None

							shape_beh = copy.copy(beh.shape)


							# if you want to decode oper temporal positions and only for some conditions; otherwise, use the full code below
							
							# decode for L2 in conditions TTDDD and T..DDDT
							# beh['T'].iloc[range(shape_beh[0])] = beh.L2[range(shape_beh[0])] # get everythin from position 2; targets and distractors
							# beh['target_event'].iloc[range(shape_beh[0])] = beh.ST_P2[range(shape_beh[0])]
							# # note that time shifting will not be correct for T2 in T..DDDT condition; because of variable positions; but it will be for all other events
							# eeg = PO.cnd_time_shift_boost_bounce(eeg, beh, cnd_info = {'TTDDD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4], 'T..DDDT': [0, -0.083, -0.083*2,-0.083*3, -0.083*4]}, cnd_header = 'condition')

							# decode at L3; TDTDD and T..DDDT
							# beh['T'].iloc[range(shape_beh[0])] = beh.L3[range(shape_beh[0])] # get everythin from position 2; targets and distractors
							# beh['target_event'].iloc[range(shape_beh[0])] = beh.ST_P3[range(shape_beh[0])]
							# # note that time shifting will not be correct for T2 in T..DDDT condition; because of variable positions; but it will be for all other events
							# eeg = PO.cnd_time_shift_boost_bounce(eeg, beh, cnd_info = {'TDTDD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4], 'T..DDDT': [0, -0.083, -0.083*2,-0.083*3, -0.083*4]}, cnd_header = 'condition')


							#beh['T'].iloc[range(shape_beh[0])] = beh.L4[range(shape_beh[0])] # get everythin from position 2; targets and distractors
							#beh['target_event'].iloc[range(shape_beh[0])] = beh.ST_P4[range(shape_beh[0])]
							# note that time shifting will not be correct for T2 in T..DDDT condition; because of variable positions; but it will be for all other events
							#eeg = PO.cnd_time_shift_boost_bounce(eeg, beh, cnd_info = {'TDDTD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4], 'T..DDDT': [0, -0.083, -0.083*2,-0.083*3, -0.083*4]}, cnd_header = 'condition')



							## full ## conditions list
													
							beh['T'].iloc[range(shape_beh[0])] = beh.L1[range(shape_beh[0])] # these trials do not need to be duplicated
							beh['target_event'].iloc[range(shape_beh[0])] = beh.ST_P1[range(shape_beh[0])]

							for c_idx, col in enumerate([beh.ST_P2, beh.ST_P3 ,beh.ST_P4, beh.ST_P5]): # no need to duplicate trials for beh.ST_P1
								
								dupl_eeg_idx_t2 = []
								
								for trial_idx in range(shape_beh[0]):	
									
									if col[trial_idx] in ['T','D']: 

										beh = beh.append(beh.loc[trial_idx], ignore_index = True) 

										beh['event_nr'].iloc[-1] = 2 + c_idx # update whether the trial is a duplicate; 1s are not duplicate
										
										dupl_eeg_idx_t2.append(trial_idx)

										# 	# create target event column (the one that is actually used for decoding) - combines 'T_P2' and 'T_P3' into one column
										beh['target_event'].iloc[-1] = beh['ST_P{}'.format(c_idx+2)][trial_idx] 
										# contains all target and distractor labels 
										beh['T'].iloc[-1] = beh['L{}'.format(c_idx+2)][trial_idx] # take the label

								# concatenate trials
								eeg= mne.concatenate_epochs([eeg, eeg[dupl_eeg_idx_t2]]) # this will concatenate a single trial up to 4 times 


							# shift timings to control for different positions
							eeg = PO.cnd_time_shift_boost_bounce(eeg, beh, cnd_info = {'TTDDD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4], 'TDTDD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4],'TDDTD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4],
																									'TTTDD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4],'TTDTD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4],
																									'TDTTD': [0, -0.083, -0.083*2,-0.083*3, -0.083*4]}, cnd_header = 'condition')

					
							# subtract a 9 from distractor label so you can use only [2,3,4,5,6,7,8,9] as labels for decoding - numbers are in essence meaningless
							beh['T'][np.where (beh['T'] > 10)[0]] =  beh['T'][np.where (beh['T'] > 10)[0]] - 9
							beh['T'] = beh['T'][~beh['T'].isnull()].astype(int) # decoding on data type object does not work so change to integers or strings


					# DISTRACTOR
					# to_decode = 'T1' % get labels from this column
					# to_decode = 'target_event' # specifies what bdm_labels are


					# BOOST BOUNCE
					# to_decode = 'T' # specifies what bdm_labels are
					# beh['boost_or_bounce'] = 'all' # need to specify condition, just takes all' for bdm lables


					# #T2 SEEN UNSEEN
					# to_decode = 'T2_correct'  # specifies what bdm_labels are
					# beh['seen/unseen'] = 'all'
					# for trial_idx in range(beh.shape[0]):
					# 	if math.isnan(beh['T2'][trial_idx]):
					# 		beh['T2_correct'][trial_idx] = np.nan # bdm label
					

					# T2 SEEN UNSEEN - decoding number labels
					# beh['seen/unseen'] = None
					# for trial_idx in range(beh.shape[0]):
					# 	if math.isnan(beh['T2'][trial_idx]):
					# 		beh['T2_correct'][trial_idx] = np.nan # bdm label
					# 		beh['seen/unseen'][trial_idx]  = np.nan # condition
	

					# beh['seen/unseen'][beh['T2_correct'] == 1] = 'seen'
					# beh['seen/unseen'][beh['T2_correct'] == 2] = 'unseen'
				
					# beh['T2'] = beh['T2'][~beh['T2'].isnull()].astype(int) # decoding on data type object does not work so change to integers or strings


					# example of how to call BDM for decoding per temporal postion (T1, T2, etc..)
					# bdm = BDM(beh, eeg, to_decode = decoding,  method = 'auc', nr_folds = 10, elec_oi = 'all', downsample = 128)
					# bdm_acc = bdm.Classify(sj, cnds = cnds, cnd_header = 'condition', time = (-0.2, 0.9), collapse = True, bdm_labels = 'all', 
  			# 						 		excl_factor = excl_factor, gat_matrix = False, downscale = False, save = True)

					# # example within decoding of boost or bounce arross all target labels collapsed; TD

					# change this so to decode targets vs distractors	

					bdm = BDM(beh, eeg, to_decode = 'T', method = 'auc', nr_folds = 10, elec_oi = 'all', downsample = 128)
					bdm.Classify(sj, cnds = ['T', 'D'], cnd_header = 'target_event', time = (-0.2, 0.9), bdm_labels = [2,3,4,5,6,7,8,9],
								collapse = False, excl_factor = excl_factor, gat_matrix = True, save = True)

					# T2 seen unseen
					# bdm = BDM(beh, eeg, to_decode = to_decode, method = 'auc', nr_folds = 10, elec_oi = 'all', downsample = 128)
					# bdm.Classify(sj, cnds ='all', cnd_header = 'seen/unseen', time = (-0.2, 0.9), bdm_labels = [1,2],
					# 			collapse = False, excl_factor = excl_factor, gat_matrix = False, save = True)


	def CrossDecoding(self, relevance = 'target', to_decode='all'):

		# CROSS-TASK DECODING (using localizer)
		for sj in [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35]:
			# train data (localizer)
			train_file_eeg = '/home/jalilov1/BB/AB_R/data/localizer/processed/subject-{}_all-epo.fif'.format(sj)
			train_eeg = mne.read_epochs(train_file_eeg)
			train_file_beh = '/home/jalilov1/BB/AB_R/data/localizer/beh/processed/subject-{}_all.pickle'.format(sj)
			train_beh = pickle.load(open(train_file_beh,'rb'))
			train_beh = pd.DataFrame.from_dict(train_beh)
		

			if relevance == 'target':
				train_beh['digit'] -= 20 #this should be digit when decoding Ts and letters when decoding Ds
			else: 
				train_beh['letter'] -= 20 # subtract this otherwise the labels in the training and testing set are not the same

			# load in test data
			test_file_eeg = '/home/jalilov1/BB/AB_R/data/AB/processed/subject-{}_all-epo.fif'.format(sj)
			test_eeg = mne.read_epochs(test_file_eeg)

			test_file_beh = '/home/jalilov1/BB/AB_R/data/AB/beh/processed/subject-{}_all-new.pickle'.format(sj)
			test_beh = pickle.load(open(test_file_beh,'rb'))
			test_beh = pd.DataFrame.from_dict(test_beh)

			
			# update beh so that we can filter based on T1 accuracy
			# test_beh = PO.correct_T1_filter_beh(sj, test_beh) 
			# test_excl = dict(T1_correct = [2])

	
			if relevance == 'target':

				if to_decode == 'T2':

					test_beh = PO.correct_T2_filter_beh(sj, test_beh) # check what you are selecting from behavioral csv file; this creates a new column in test_bhe file

					# test_excl = dict(T2_correct = [2]) # exclude T2 incorrect trials; T2 seen
					# test_excl = dict(T2_correct = [1]) #exclude trials on which T2 was detected correctly; T2 unseen

					# (re)BASELINE data
					# test_eeg.apply_baseline (baseline = (-0.2, 0))

					# cnds = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD'] # all
					# test_eeg =  PO.cnd_time_shift_variablepos(test_eeg, test_beh, cnd_info = {'T..DDDT': 0,'TTDDD': -0.083, 'TDTDD': -0.083*2,'TDDTD': -0.083*3,'TTTDD': -0.083,'TTDTD': -0.083,'TDTTD': -0.083*2}, cnd_header = 'condition', pos_info = {'pos_13': -0.664,'pos_14': -0.747,'pos_15': -0.83,'pos_16': -0.913}, pos_header = 'Cnd_1-T2_pos')

				# 	cnds= ['TTDDD','TTTDD','TTDTD'] #boost
				# 	test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TTDDD': -0.083, 'TTTDD': -0.083,'TTDTD': -0.083}, cnd_header = 'condition')
					
				#  	cnds = ['TDTDD','TDDTD','TDTTD'] #bounce
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTDD': -0.083*2,'TDDTD': -0.083*3, 'TDTTD': -0.083*2}, cnd_header = 'condition')

					## AB conditions

					cnds = ['TDTDD','TDDTD','TDTTD'] #just AB conditions
					test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTTD': -0.083*2,'TDDTD': -0.083*3, 'TDTTD': -0.083*2}, cnd_header = 'condition')	


				elif to_decode == 'T3':
					# test_beh = PO.correct_T3_filter_beh(sj, test_beh) # T3 ide any trials
					# test_excl = dict(T3_correct = [2])

		 			# cnds = ['TTTDD','TTDTD','TDTTD'] #all
		 			# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TTTDD': -0.083*2, 'TTDTD': -0.083*3,'TDTTD': -0.083*3}, cnd_header = 'condition')
		 			
		 			# reBASELINE data
					# test_eeg.apply_baseline (baseline = (-0.2, 0))
		 			
					# cnds = ['TTTDD','TDTTD'] # boost
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TTTDD': -0.083*2, 'TDTTD': -0.083*3}, cnd_header = 'condition')	

				# 	cnds = ['TTDTD'] # bounce	
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TTDTD': -0.083*3}, cnd_header = 'condition')	

					cnds = ['TDTTD', 'TTDTD'] # boost
					test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTTD': -0.083*3,'TTDTD': -0.083*3}, cnd_header = 'condition')	


				elif to_decode == 'T1': #T1
					# test_beh = PO.correct_T1_filter_beh(sj, test_beh) #T1 det correct
					# test_excl = dict(T1_correct = [2])

					cnds = 	['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] #T1
					test_eeg = PO.cnd_time_shift_variablepos(test_eeg, test_beh, cnd_info = {'DDTDD': 0}, cnd_header = 'condition', pos_info = {'pos_13': -0.664,'pos_14': -0.747,'pos_15': -0.83,'pos_16': -0.913}, pos_header = 'Cnd_8-T1_pos') # determine how many positions to shift based on T1 position in beh file
					# BASELINE data
					# test_eeg.apply_baseline (baseline = (-0.2, 0))
				
				elif to_decode == 'all':

					# Update test beh so we can decode boost versus bounce
					test_beh['T_P2'], test_beh['T_P3'] = None, None

					# for cnd in ['TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD']: # T2s and T3s
					for cnd in ['TTDDD','TDTDD','TDDTD']:	# only T2s
						idx = [i for i, letter in enumerate(cnd) if letter == 'T']
						for pos_idx, pos in enumerate(['P1','P2','P3'][:len(idx)]):
							cat_idx = idx[pos_idx]
							b_or_b = 'boost' if cnd[cat_idx - 1] == 'T' else 'bounce' # if the preceeding item is a target then boost
							if pos != 'P1': # skip saving this; T1 
								test_beh.loc[test_beh.condition == cnd, '{}_{}'.format('T', pos)] = b_or_b

					# first decode boost versus bounce targets (duplicate trials to be able to collapse)
					# dupl_eeg = []
					# test_beh['event_nr'] = 1
					# for trial_idx in range(test_beh.shape[0]):
					# 	# check whether trial should be duplicated more than once
					# 	if test_beh.T_P3[trial_idx] in ['boost','bounce']:
					# 		test_beh = test_beh.append(test_beh.loc[trial_idx], ignore_index = True) 
					# 		test_beh['event_nr'].iloc[-1] = 2
					# 		# at the moment only data array within mne object is duplicated
					# 		dupl_eeg.append(test_eeg._data[trial_idx])
					# test_eeg._data = np.append(test_eeg._data, np.stack(dupl_eeg), axis = 0)

				
					dupl_eeg_idx = []
					test_beh['event_nr'] = 1
					for trial_idx in range(test_beh.shape[0]):
						# check whether trial should be duplicated more than once
						if test_beh.T_P3[trial_idx] in ['boost','bounce']:
							test_beh = test_beh.append(test_beh.loc[trial_idx], ignore_index = True) 
							test_beh['event_nr'].iloc[-1] = 2
							dupl_eeg_idx.append(trial_idx)
					#test_beh.loc[:-len(dupl_eeg_idx),'event_nr'] = 2
					if dupl_eeg_idx:
						test_eeg = mne.concatenate_epochs([test_eeg, test_eeg[dupl_eeg_idx]])


					# shift timings to control for different positions; T2 and T3
					# test_eeg = PO.cnd_time_shift_boost_bounce(test_eeg, test_beh, cnd_info = {'TTDDD': [-0.083], 'TDTDD': [ -0.083*2],'TDDTD': [ -0.083*3],
					# 																		'TTTDD': [ -0.083, -0.083*2],'TTDTD': [ -0.083, -0.083*3],
					# 																		'TDTTD': [ -0.083*2, -0.083*3]}, cnd_header = 'condition')
					

					test_eeg = PO.cnd_time_shift_boost_bounce(test_eeg, test_beh, cnd_info = {'TTDDD': [-0.083], 'TDTDD': [-0.083*2],'TDDTD': [-0.083*3] }, cnd_header = 'condition')



					# create target event column (the one that is actually used for decoding)					
					test_beh['target_event'] = np.nan
					test_beh['target_event'][test_beh['event_nr'] == 1] = test_beh['T_P2'][test_beh['event_nr'] == 1]
					test_beh['target_event'][test_beh['event_nr'] == 2] = test_beh['T_P3'][test_beh['event_nr'] == 2]
					# create target column - which numbers are actually presented - decode these labels
					test_beh['T'] = None
					test_beh['T'][test_beh['event_nr'] == 1] = test_beh['T2'][test_beh['event_nr'] == 1]
					test_beh['T'][test_beh['event_nr'] == 2] = test_beh['T3'][test_beh['event_nr'] == 2]

			else:

				#if len([i for i, letter in enumerate(test_beh.condition[0]) if letter == 'T']) > 2

				if to_decode == 'D1':
			
					test_beh = PO.correct_T2_filter_beh(sj, test_beh) # check what you are selecting!
					test_excl = None


					# test_excl = dict(T2_correct = [2]) # exclude T2 incorrect trials; T2 seen
					# test_excl = dict(T2_correct = [1]) #exclude trials on which T2 was detected correctly; T2 unseen

					# cnds = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD'] #boosted D1
			
					# cnds = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] # all

					# cnds = ['TDTDD','TDDTD','TTDTD','TDTTD'] # AB conditions
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTDD': -0.083,'TDDTD': -0.083, 'TTDTD': -0.083*2,'TDTTD': -0.083}, cnd_header = 'condition')
					# test_excl = None

					cnds = ['TDDTD']
					test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDDTD': -0.083}, cnd_header = 'condition')

					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'T..DDDT': -0.083, 'TTDDD': -0.083*2, 'TDTDD': -0.083,'TDDTD': -0.083,'TTTDD': -0.083*3,'TTDTD': -0.083*2,'TDTTD': -0.083}, cnd_header = 'condition')
				
				# 	# test_eeg.apply_baseline (baseline = (-0.65, -0.45)) # 

					# cnds = ['T..DDDT', 'TDTDD','TDDTD','TDTTD'] # boost-1
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'T..DDDT': -0.083, 'TDTDD': -0.083,'TDDTD': -0.083,'TDTTD': -0.083}, cnd_header = 'condition')

					# cnds = ['TTDDD','TTDTD'] #boost-2
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = { 'TTDDD': -0.083*2, 'TTDTD': -0.083*2}, cnd_header = 'condition')

					# cnds = ['TTTDD'] #boost-3
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = { 'TTTDD': -0.083*3}, cnd_header = 'condition')
				
				elif to_decode == 'D2':

					test_beh = PO.correct_T2_filter_beh(sj, test_beh) # check what you are selecting!
					test_excl = None
					cnds = ['TDTDD']
					test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTDD': -0.083}, cnd_header = 'condition')

					# cnds = ['T..DDDT', 'TTDDD','TDDTD','TTTDD'] #bounced D2s
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'T..DDDT': -0.083*2, 'TTDDD': -0.083*3,'TDDTD': -0.083*2,'TTTDD': -0.083*4}, cnd_header = 'condition')
					
				#	cnds = ['TTDTD','TDTTD'] #boost-3
				#	# Maybe you should not include the first conditions because the third T was blinked
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TTDTD': -0.083*4,'TDTTD': -0.083*4}, cnd_header = 'condition')
					
					# This excludes the AB condition, very unlikely that D2 was boosted 
					# cnds = ['TDTTD'] #boost-4
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTTD': -0.083*4}, cnd_header = 'condition')
					
				# 	cnds = ['TDTDD'] #boost-2 
				## Not sure about this comparison, this distractor is presented after a blinked target
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTDD': -0.083*3}, cnd_header = 'condition')

					# cnds = ['T..DDDT', 'TDDTD'] # bounce-1
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'T..DDDT': -0.083*2, 'TDDTD': -0.083*2}, cnd_header = 'condition')
		 			
					# cnds = ['TTDDD'] # bounce-2
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TTDDD': -0.083*3}, cnd_header = 'condition')
					
					# cnds = ['TTTDD'] # bounce-3
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TTTDD': -0.083*4}, cnd_header = 'condition')
							 			

					# cnds = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] # all
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'T..DDDT': -0.083*2, 'TTDDD': -0.083*3, 'TDTDD': -0.083*3,'TDDTD': -0.083*2,'TTTDD': -0.083*4,'TTDTD': -0.083*4,'TDTTD': -0.083*4, 'DDTDD':-0.083}, cnd_header = 'condition')
					# # test_eeg.apply_baseline (baseline = (-0.65, -0.45))			
				
				elif to_decode == 'D3': #D3
					cnds = 	['T..DDDT','TTDDD','TDTDD','TDDTD','DDTDD'] # all
					test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'T..DDDT': -0.083*3, 'TTDDD': -0.083*4, 'TDTDD': -0.083*4,'TDDTD': -0.083*4, 'DDTDD':-0.083*2}, cnd_header = 'condition') # determine how many positions to shift based on T1 position in beh file
					# test_eeg.apply_baseline (baseline = (-0.65, -0.45))

					# cnds = ['TDDTD'] # boost
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDDTD': -0.083*4}, cnd_header = 'condition')

					# cnds = ['TDTDD'] #bounce
					# test_eeg =  cnd_time_shift(test_eeg, test_beh, cnd_info = {'TDTDD': -0.083*4}, cnd_header = 'condition')

				elif to_decode == 'all': #   boostbounce
					# Update test beh so we can decode boost versus bounce
					
					test_beh['D_P1'], test_beh['D_P2'], test_beh['D_P3'] = None, None, None
					# # test_beh['T_P2'], test_beh['T_P3'] = None, None
					# for cat in ['D']:
					# 	for cnd in [ 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD']:
					# 		idx = [i for i, letter in enumerate(cnd) if letter == cat]
					# 		for pos_idx, pos in enumerate(['P1','P2','P3'][:len(idx)]):
					# 			cat_idx = idx[pos_idx]
					# 			b_or_b = 'boost' if cnd[cat_idx - 1] == 'T' else 'bounce'
					# 			if cat != 'D' or pos != 'P1': # SAVE ALSO P1 for distractors
					# 				test_beh.loc[test_beh.condition == cnd, '{}_{}'.format(cat, pos)] = b_or_b

					for cnd in ['TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD']:
						idx = [i for i, letter in enumerate(cnd) if letter == 'D'] # find where in the string is D
						for pos_idx, pos in enumerate(['P1','P2','P3'][:len(idx)]):
							cat_idx = idx[pos_idx]
							b_or_b = 'boost' if cnd[cat_idx - 1] == 'T' else 'bounce' # if the preceeding item is a target then boost
							test_beh.loc[test_beh.condition == cnd, '{}_{}'.format('D', pos)] = b_or_b

					# first decode boost versus bounce targets (duplicate trials to be able to collapse)
					dupl_eeg_idx_p2 = []
					dupl_eeg_idx_p3 = []

					test_beh['event_nr'] = 1

					# for trial_idx in range(test_beh.shape[0]):
					# 	# check whether trial should be duplicated more than once
					# 	if test_beh.D_P2[trial_idx] in ['boost','bounce']: # don't need to duplicate for D_P1
					# 		test_beh = test_beh.append(test_beh.loc[trial_idx], ignore_index = True) 
					# 		test_beh['event_nr'].iloc[-1] = 2
					# 		# at the moment only data array within mne object is duplicated
					# 		dupl_eeg_p2.append(test_eeg._data[trial_idx])
					# test_eeg._data = np.append(test_eeg._data, np.stack(dupl_eeg_p2), axis = 0)

					
					# for trial_idx in range(sum(test_beh.event_nr==1)):	# loop through the original beh structure (without duplicates)	
					# 	if test_beh.D_P3[trial_idx] in ['boost','bounce']: 
					# 		test_beh = test_beh.append(test_beh.loc[trial_idx], ignore_index = True) 
					# 		test_beh['event_nr'].iloc[-1] = 3
					# 		# at the moment only data array within mne object is duplicated
					# 		dupl_eeg_p3.append(test_eeg._data[trial_idx])
					# test_eeg._data = np.append(test_eeg._data, np.stack(dupl_eeg_p3), axis = 0)

					shape_beh = copy.copy(test_beh.shape)

					for trial_idx in range(shape_beh[0]):
						# check whether trial should be duplicated more than once
						if test_beh.D_P2[trial_idx] in ['boost','bounce']:
							test_beh = test_beh.append(test_beh.loc[trial_idx], ignore_index = True)
							test_beh['event_nr'].iloc[-1] = 2 
							dupl_eeg_idx_p2.append(trial_idx)
					#test_beh.loc[:-len(dupl_eeg_idx_p2),'event_nr'] = 2
					test_eeg = mne.concatenate_epochs([test_eeg, test_eeg[dupl_eeg_idx_p2]])

					for trial_idx in range(shape_beh[0]):
						# check whether trial should be duplicated more than once
						if test_beh.D_P3[trial_idx] in ['boost','bounce']:
							test_beh = test_beh.append(test_beh.loc[trial_idx], ignore_index = True) 
							test_beh['event_nr'].iloc[-1] = 3
							dupl_eeg_idx_p3.append(trial_idx)
					#test_beh.loc[:-len(dupl_eeg_idx_p3),'event_nr'] = 3
					test_eeg = mne.concatenate_epochs([test_eeg, test_eeg[dupl_eeg_idx_p3]])

					
					# shift timings for distractors to control for different positions
					test_eeg = PO.cnd_time_shift_boost_bounce(test_eeg, test_beh, cnd_info = {'TTDDD': [-0.083*2, -0.083*3, -0.083*4], 'TDTDD': [-0.083,-0.083*3, -0.083*4],'TDDTD': [-0.083,  -0.083*2, -0.083*4],
																							'TTTDD': [-0.083*3, -0.083*4],'TTDTD': [-0.083*2, -0.083*4],
																							'TDTTD': [-0.083, -0.083*4]}, cnd_header = 'condition')


					# create target event column (the one that is actually used for decoding)
					test_beh['target_event'] = np.nan
					test_beh['target_event'][test_beh['event_nr'] == 1] = test_beh['D_P1'][test_beh['event_nr'] == 1]
					test_beh['target_event'][test_beh['event_nr'] == 2] = test_beh['D_P2'][test_beh['event_nr'] == 2]
					test_beh['target_event'][test_beh['event_nr'] == 3] = test_beh['D_P3'][test_beh['event_nr'] == 3]
					
					test_beh['T'] = None				
					test_beh['T'][test_beh['event_nr'] == 1] = test_beh['D1'][test_beh['event_nr'] == 1]
					test_beh['T'][test_beh['event_nr'] == 2] = test_beh['D2'][test_beh['event_nr'] == 2]
					test_beh['T'][test_beh['event_nr'] == 3] = test_beh['D3'][test_beh['event_nr'] == 3]

					for en, i in enumerate(test_beh['T']):
						if i > 1000:
							test_beh['T'][en]= i-1000 # to eliminate high triggers that D1s on position 5 received 



			# ORIGINAL - for T1, T2, T3, D1, D2, D3

			# bdm = BDM(test_beh, test_eeg, to_decode = to_decode, method = 'auc', nr_folds = 1, elec_oi = 'all', downsample = 128)
			# bdm.localizerClassify(sj, train_beh, train_eeg, cnds, cnd_header = 'condition', time = (-0.2, 0.9), 
			# 				 tr_header = 'digit', te_header= to_decode, collapse = True, loc_excl = dict(digit = [-20]), 
			# 				 test_excl = test_excl, gat_matrix = True, save = True)
		
			# embed()
			# example decoding of target labels for boost and bounce trials seperately
			# bdm = BDM(test_beh, test_eeg, to_decode = 'T', method = 'auc', nr_folds = 1, elec_oi = 'all', downsample = 128)
			# bdm.localizerClassify(sj, train_beh, train_eeg, cnds = ['boost', 'bounce'] , cnd_header = 'target_event', time = (-0.2, 0.9), 
			# 			 tr_header = 'digit', te_header=  'T', collapse = False, loc_excl = dict(digit = [-20]), # digit when decoding Ts, letter when decoding Ds
			# 			 test_excl = test_excl, gat_matrix = True, save = True)


			# D1s on T2 seen/unseen trials OR T2 seen/unseen
			test_beh ['T2_AB'] = None
			for c in cnds:
				idxx = (test_beh ['condition']==c) * (test_beh ['T2_correct']==1)
				test_beh['T2_AB'][idxx] = 'AB_seen'
				idyy = (test_beh ['condition']==c) * (test_beh ['T2_correct']==2)
				test_beh['T2_AB'][idyy] = 'AB_unseen'

			# test_excl = None
			# test_beh['T2'] = test_beh['T2'][~test_beh['T2'].isnull()].astype(int) 
	

			bdm = BDM(test_beh, test_eeg, to_decode = to_decode, method = 'auc', nr_folds = 1, elec_oi = 'all', downsample = 128)
			bdm.localizerClassify(sj, 	train_beh, train_eeg, cnds=['AB_seen', 'AB_unseen'], cnd_header = 'T2_AB', time = (-0.2, 0.9), 
							tr_header = 'letter', te_header= to_decode, collapse = True, loc_excl = dict(letter = [-20]), 
							test_excl = test_excl, gat_matrix = True, save = True)




if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '2' 
	os.environ['NUMEXP_NUM_THREADS'] = '2'
	os.environ['OMP_NUM_THREADS'] = '2'
	
	# Specify project parameters
	#project_folder = '/home/jalilov1/data/loc_pilot2' 
	#project_folder = '/home/jalilov1/BB/AB_R/data' #change the paths for within-task decoding!
	project_folder = '/home/dvmoors1/BB/Josipa'
	os.chdir(project_folder)
	# initiate current project
	PO = Josipa()

	# DECODING
	# PO.crossTaskBDM(sj=sj, to_decode_tr = 'digit', to_decode_te = 'T2', gat_matrix = True)
	PO.WithinDecoding(task='AB', decoding = 'TD', d_type = 'stimulus')
	#PO.CrossDecoding(relevance = 'distractor', to_decode ='D2')

	# PLOTTING
	
	# PO.plotBDMwithin(to_plot = 'targets', task='AB')	# YES; plots diagonal
	# PO.plotDecoding(plot = 'diagonal') # YES, plots either GAT or diagonal, within and cross decoding
	# PO.plotBDMwithin(to_plot = 'T1', task='localizer')	# YES


	# # STEP 1: run task specific preprocessing 
	# for sj in [1]: 
	# 	for session in range(2,4): 
	# 		if session == 1:
	# 			task = 'localizer'
	# 			trigger = [22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38] # for localizer, 22-29 target numbers; 31-38 target letters
	# 			t_min = -0.4 
	# 			t_max = 1.4 # if target on 5 position; 14 stim max; 10*83ms + 600ms blank period ; this was 1 previously

	# 		else:
	# 			task = 'AB'
	# 			trigger = [1002,1003,1004,1005,1006,1007,1008,1009,1011,1012,1013,1014,1015,1016,1017,1018] # 1002:1009 are T1s in cond 1-7 and 1011-1018 are non-target stimuli at position 5 
	# 			t_min = -0.4   
	# 			t_max = 2 #18 stim max + 800ms blank period; 12*83 = 996; 2 seconds in total 

	# 	   	#pass
	# 	 	PO.prepareEEG(sj = sj, session = session, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	# 	 			  t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	# 	 			  trigger = trigger, project_param = project_param, 
	# 	 			  project_folder = project_folder, binary = binary, channel_plots = False, inspect = True, task = task)
			

	# STEP 2: run task specific analysis

	

	# # GET INDICES OF CLEAN EEG TRIALS
	# for sj in [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35]:
	# 	#1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35
	# 	# train data (localizer)
	# 	# train_file_eeg = '/home/jalilov1/BB/AB_R/data/localizer/processed/subject-{}_all-epo.fif'.format(sj)
	# 	# train_eeg = mne.read_epochs(train_file_eeg)
	# 	# train_file_beh = '/home/jalilov1/BB/AB_R/data/localizer/beh/processed/subject-{}_all.pickle'.format(sj)
	# 	# train_beh = pickle.load(open(train_file_beh,'rb'))
	# 	# train_beh = pd.DataFrame.from_dict(train_beh)
	# 	# train_beh['letter'] -= 20 # this should be digit when decoding Ts and letters when decoding Ds
	# 	# # train_beh['letter'] -= 20 # subtract this otherwise the labels in the training and testing set are not the same

	# 	# load in test data
	# 	# test_file_eeg = '/home/jalilov1/BB/AB_R/data/AB/processed/subject-{}_all-epo.fif'.format(sj)
	# 	# test_eeg = mne.read_epochs(test_file_eeg)

	# 	test_file_beh = '/home/jalilov1/BB/AB_R/data/AB/beh/processed/subject-{}_all-new.pickle'.format(sj)
	# 	test_beh = pickle.load(open(test_file_beh,'rb'))
	# 	test_beh = pd.DataFrame.from_dict(test_beh)
	# 	# update beh so that we can filter based on T1 accuracy
	# 	test_beh = PO.correct_T1_filter_beh(sj, test_beh) # Now it doesn't save it in the beh file

	# 	#save trial number that was kept in EEG analysis (after preprocessing + filtering based on T1 detection)
	# 	clean_trials = test_beh['nr_trials'].values #both sessions

	# 	clean_eeg_trials = pd.DataFrame(clean_trials, columns=['clean_trials']).to_csv('subject-{}-clean-eeg-trials.csv'.format(sj) )
	


	# # examine below chance T2 decoding (train on T1 test on T2)
	# for sj in [20,21,22,23,24,25,26]:	
	# 	# load in test data
	# 	te_file_eeg = '/home/jalilov1/BB/AB_R/data/AB/processed/subject-{}_all-epo.fif'.format(sj)
	# 	eeg = mne.read_epochs(te_file_eeg)
	# 	te_file_beh = '/home/jalilov1/BB/AB_R/data/AB/beh/processed/subject-{}_all-new.pickle'.format(sj)
	# 	beh = pickle.load(open(te_file_beh,'rb'))
	# 	beh = pd.DataFrame.from_dict(beh)	
	# 	bdm = BDM(beh, eeg, to_decode = 'T2', method = 'auc', nr_folds = 10, elec_oi = 'all',downsample = 128)
	# 	bdm.crossClassify(sj, cnds = ['TTDDD','TTTDD','TTDTD'], cnd_header ='condition', time = (-0.2, 0.9)
	# 					, tr_header = 'T1', te_header = 'T2',tr_te_rel = 'dep', excl_factor = None, gat_matrix = True, save = True)
	
	

		