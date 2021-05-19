## a list of standard python packages that we need (add more when needed) 
from statistics import mean
import os 
import mne  
import sys 
import glob 
import pickle 
import logging 
 
#import matplotlib 
 
#matplotlib.use('agg') # now it works via ssh connection - make this live if using TUX 
 
## more standard packages  
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
## here we import packages from my toolbox  
# add more when needed (e.g., decoding scripts) 
#sys.path.append('/Users/Account/Documents/Linguistic/2021/pilot/decodingscript/preprocess/DvM-master') # to import the script needs to know where to look 
sys.path.append('/Users/dvm/DvM')
#This is to incorporate the delays of audio onset in the audio file for each stimuli, Ana Radanovic
#time_delays_df=pd.read_csv('C:/Users/anara/Documents/PM_Lab/Experiment/Scripts/delay_times.csv') #filepath for timing delays csv sheet

from beh_analyses.PreProcessing import * 
from eeg_analyses.EEG import *  
#from eeg_analyses.ERP import *  
from eeg_analyses.BDM import * 
from support.FolderStructure import * 
from support.support import * 
from IPython import embed # this one is important (allows for debugging) 
from mne.viz import plot_evoked_topo 
from mne import epochs
 
 
## First step is to specify all project relevant information  
# subject specific info
sj_info = {'1': {'replace':{}, 'expertise': 'expert', 'order': ['FA','OM','CT']}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			'2': {'replace':{}, 'expertise': 'novice', 'order': ['OM','FA','CT']},
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


# eeg info (event_id specified below)
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
eeg_runs = [1]
t_min = -0.2 # double check that this is ok given trigger time
t_max = 2
flt_pad = 0.5
binary =  0
event_id = {'Neg': 10,  
            'Neu': 20, 
            'Pseudo': 30, 
            'Pure_Tones': 40,} 
 
# project specific info
project = 'Pilot'
project_param = ['practice','nr_trials','trigger','RT', 'subject_nr', 
                'block_type', 'trial_type','correct','dist_high','dist_loc','dist_shape', 
                'target_high','target_loc','target_shape','high_loc', 'block_cnt'] # list of relevant variables in behavior file 
 
#eye_tracker info 
tracker_ext = '' 
eye_freq = 0 
start_event = '' 
tracker_shift = 0 
viewing_dist = 100 
screen_res = (1680, 1050)  
screen_h = 29 
 

# here we create a class that contains all functionality we will be using in this specific project ## this is where we will add all future functions. 
 
class Meditate(FolderStructure): 
    def __init__(self): pass 
 
    def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, event_id, project_param, project_folder, binary, channel_plots, inspect): 
 
        """ 
 
        EEG preprocessing function as preregistred @. This function does preprocessing according to the procedure specified here: 
 
        Arguments: 
 
            sj {int} -- Subject number 
            session {int} -- Number of EEG session 
            eog {list} -- EOG electrodes 
            ref {list} -- Reference electrodes electrodes 
            eeg_runs {list} -- list of seperate eeg runs within a single session 
            t_min {float} -- timing before trigger onset 
            t_max {float} -- timing after trigger onset 
            flt_pad {float} -- extend epoched data by this value (on both sides) 
            sj_info {dict} -- subject specific project info 
            event_id {list | dict} -- list of triggers or dictionary that couples triggers to condition info 
            project_param {list} -- project specific parameters 
            project_folder {str} -- folder of the current project 
            binary {int} -- control for incorrect trigger numbers 
            channel_plots {bool} -- should informative channel plots be created 
            inspect {bool} -- open interactive plots during preprocessing 
 
        """ 
 
		# set subject specific parameters
        missing = np.array([])
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

        EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['V_do','H_l','Ref_r','Ref_l','EXG7','EXG8'])
        EEG.setMontage(montage='biosemi64', ch_remove = ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'])
 
        #FILTER DATA BEFORE EPOCHING 
        EEG.filter(h_freq=None, l_freq=0.01, fir_design='firwin', skip_by_annotation='edge') 
 
        # EPOCH DATA 
        events = EEG.eventSelection(event_id, binary=binary, min_duration=0) 
        epochs = Epochs(sj, session, EEG, events, event_id=event_id, 
                tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad)
        epochs.baseline = None # temp fix for new mne update
        beh = pd.DataFrame({'stimulus_type': epochs.events[:,2],
                            'condition': 'all', 
                            'expertise': sj_info[str(sj)]['expertise'],
                            'med_style': sj_info[str(sj)]['order'][session-1]})

		# ARTIFACT DETECTION
        epochs.selectBadChannels(channel_plots = False, inspect = True, RT = None)    
        #z = epochs.artifactDetection(z_thresh=4, band_pass=[110, 140], plot=True, inspect=True)

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

# now we are actually going to do stuff 
if __name__ == '__main__': 
 
    # Make sure we are in the correct folder 
    project_folder = '/Users/dvm/Desktop/Meditate' 
    os.chdir(project_folder) 
 
    # initiate current project 
    PO = Meditate() 

    ## Run Preprocessing 
    for sj in []: 
        print('start preprocessing subject {}'.format(sj))
        for session in [1,2]: 
            PO.prepareEEG(sj = sj, session = session, eog = eog, ref = ref, eeg_runs = eeg_runs, 
            t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, event_id = event_id, 
            project_param = project_param, project_folder = project_folder, binary = binary,  
            channel_plots = True, inspect = True) 

    ## Decoding analysis 
    #10=neg, 20=neu, 30=pseudo, 40=pure
    # step 1: tones vs. words (collapsed across meditation style)  
    sj = 2
    beh, eeg = PO.loadData(sj, name = 'all', eyefilter=False, eye_window=None)   
    beh['type'] = 'word'
    beh.loc[beh.stimulus_type == 40, 'type'] = 'tone'
    eeg.baseline = None # temp_fix
    bdm = BDM(beh, eeg, 'type', nr_folds= 10, method = 'auc', elec_oi = 'all', baseline = (-0.2,0),downsample = 128, bdm_filter = None)
    bdm.Classify(sj, ['all'], 'condition', time = (-0.5,2), collapse = False, bdm_labels = ['word','tone'],  
                        excl_factor = None, nr_perm = 0, gat_matrix = False, downscale = False) 
   
    
 






