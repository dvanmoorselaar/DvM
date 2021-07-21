#%%
import os
import mne
import re
import sys
import glob
import pickle
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
#sys.path.append('/Users/dockyduncan/documents/GitHub/DvM')
sys.path.append('/research/FGB-ETP-DVM/DvM')


import numpy as np
import seaborn as sns

from scipy.interpolate import interp1d
from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import *
from eeg_analyses.BDM import *  
from support.FolderStructure import *
from support.support import *


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
            '25': {'replace':{}}, # run as subject 1
							}

# project specific info
project = 'Ping'
factors = []
labels = []
to_filter = ['RT'] 
nr_sessions = 1
project_param = ['nr_trials','trigger','RT', 'subject_nr', 'block_cnt', 'practice',
				'block_type', 'correct','dist_high','dist_loc','dist_shape', 
				'dist_color', 'high_prob_loc', 'target_high','target_loc','target_shape',
				'target_color','ping','ping_trigger','ping_type','trial_type']


# eeg info (event_id specified below)
ping = 'ping'
part = 'beh'
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
eeg_runs = [1]
# ping parameters
t_min = -0.2 
t_max = 0.6
event_id = [109, 100, 102, 104, 106, 209, 200, 202, 204, 206]
# search parameters
#t_min = -0.1 
#t_max = 0.6
#event_id = [12,13,14,15,16,17,18,19,21,23,24,25,26,27,28,29,31,32,34,35,36,37,38,39,41,42,43,45,46,\
#			47,48,49,51,52,53,54,56,57,58,59,61,62,63,64,65,67,68,69,71,72,73,74,75,76,78,79,81,82,83,84,85,86,87,89]

flt_pad = 0.5
binary =  0

# eye tracker info
tracker_ext = 'asc'
eye_freq = 1000
start_event = 'Onset ping' # start_event = 'Onset search'  
tracker_shift = 0
viewing_dist = 60 
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

    def eyeTowardness(self, sj):
        '''
        0 = top
        2 = left
        4 = bottum
        6 = right
        '''

        # read in behavior data and eyetracking data
        beh = pd.read_csv(self.FolderTracker(extension=['beh', 'raw'], 
                        filename='subject-{}_session_1.csv'.format(sj)))
        # exclude exit interview
        if sj in [1, 5, 25]:
            beh = beh[:-1]
        eye = read_edf(self.FolderTracker(extension=['eye', 'raw'], 
                        filename='sub_{}.asc'.format(sj)), 'Start trial', stop='Response')

        # remove practice trials
        if sj not in [2]:
            eye = np.array(eye)[beh.practice == 'no']
        else:
            eye = np.array(eye)
        beh = beh[beh.practice == 'no']

        # link behavior and eye data
        if len(eye) != beh.shape[0]:
            print('Eye tracking and behavior could not be linked')
        
        # calculate towardness (limited to spatial bias blocks)
        # step 1: extract x and y data per trial in - window (centered at ping display)
        x, y = np.zeros((eye.size,648)), np.zeros((eye.size,648))
        for i, trial in enumerate(eye):
            # get index time point 0
            for event in trial['events']['msg']:
                if 'Onset ping' in event[1]:
                    idx_onset = np.argmin(abs(trial['trackertime'] - event[0]))

            # step 2: if present interpolate blinks
            x_ = trial['x']
            y_ = trial['y']

            if trial['events']['Eblk'] != []:
                # find start/end blink and zero pad around blink on/offset
                for bl_inf in trial['events']['Eblk']:
                    bl_idx = [np.argmin(abs(trial['trackertime'] - t)) for t in bl_inf[:2]]
                    bl_idx = slice(bl_idx[0]-25, bl_idx[1]+26)
                    x_[bl_idx] = 0
                    y_[bl_idx] = 0

                # linear interpolation
                times = np.arange(x_.size)
                f_x = interp1d(times[np.nonzero(x_)], x_[np.nonzero(x_)], fill_value="extrapolate")
                f_y = interp1d(times[np.nonzero(y_)], y_[np.nonzero(y_)], fill_value="extrapolate")
                x_ = f_x(x_)
                y_ = f_y(y_)
            
            # populate x, y
            x[i] = trial['x'][idx_onset - 350:idx_onset + 298]
            y[i] = trial['y'][idx_onset - 350:idx_onset + 298]

        # step 3: get left right bias, and top-down bias (seperate for ping and no-ping trials)
        towardness = {'ping': {},'no_ping':{}}
        for ping_, ping in zip(['no','yes'], ['no_ping', 'ping']):
            left = np.mean(x[(beh.high_prob_loc == 6) & (beh.ping == ping_)], axis = 0)
            right = np.mean(x[(beh.high_prob_loc == 2) & (beh.ping == ping_)], axis = 0)
            top = np.mean(y[(beh.high_prob_loc == 0) & (beh.ping == ping_)], axis = 0)
            bottem = np.mean(y[(beh.high_prob_loc == 4) & (beh.ping == ping_)], axis = 0)

            towardness[ping]['toward_x'] = (right - left)/2
            towardness[ping]['toward_y'] = (bottem - top)/2

        # save data
        pickle.dump(towardness, open(self.FolderTracker(extension=['eye', 'towardness'], 
                        filename='subject_{}.pickle'.format(sj)), 'wb'))

    def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, event_id, project_param, project_folder, binary, channel_plots, inspect):
        '''
        EEG preprocessing as preregistred @
        '''

        # set subject specific parameters
        file = 'subject_{}_session_{}_'.format(sj, session)
        replace = sj_info[str(sj)]['replace']
        log_file = self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocessing_param.csv')

        # start logging
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename= self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocess_sj{}_ses{}.log'.format(
                        sj, session), overwrite = False),
                    filemode='w')

        # READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
        EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
                                            preload=True, eog=eog) for run in eeg_runs])

        #EEG.replaceChannel(sj, session, replace)
        EEG.reReference(ref_channels=ref, vEOG=eog[
                        :2], hEOG=eog[2:], changevoltage=True, to_remove = ['EXG7','EXG8'])
        EEG.setMontage(montage='biosemi64')

        #FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
        EEGica = EEG.copy()
        EEGica.filter(h_freq=None, l_freq=1.5,
                            fir_design='firwin', skip_by_annotation='edge')
        EEG.filter(h_freq=None, l_freq=0.01, fir_design='firwin',
                    skip_by_annotation='edge')

        # MATCH BEHAVIOR FILE
        events = EEG.eventSelection(event_id, binary=binary, min_duration=0)
        beh, missing = EEG.matchBeh(sj, session, events, event_id, 
                                        headers = project_param)

        # EPOCH DATA
        epochs = Epochs(sj, session, EEG, events, event_id=event_id,
                tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, reject_by_annotation = True) 
        epochs_ica = Epochs(sj, session, EEGica, events, event_id=event_id,
                tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, reject_by_annotation = True) 

        # AUTMATED ARTIFACT DETECTION
        epochs.selectBadChannels(run_ransac = True, channel_plots = False, inspect = True, RT = None)  
        z = epochs.artifactDetection(z_thresh=4, band_pass=[110, 140], plot=True, inspect=True)

        # ICA
        epochs.applyICA(EEG, epochs_ica, method='picard', fit_params = dict(ortho=False, extended=True), inspect = True)
        del EEGica

        # EYE MOVEMENTS
        epochs.detectEye(missing, events, beh.shape[0], time_window=(t_min*1000, t_max*1000), 
                        tracker_shift = tracker_shift, start_event = start_event, 
                        extension = tracker_ext, eye_freq = eye_freq, 
                        screen_res = screen_res, viewing_dist = viewing_dist, 
                        screen_h = screen_h)

        # INTERPOLATE BADS
        bads = epochs.info['bads']   
        epochs.interpolate_bads(reset_bads=True, mode='accurate')

        # LINK BEHAVIOR
        epochs.linkBeh(beh, events, event_id)

        logPreproc((sj, session), log_file, nr_sj = len(sj_info.keys()), nr_sessions = nr_sessions, 
                    to_update = dict(nr_clean = len(epochs), z_value = z, nr_bads = len(bads), bad_el = bads))

if __name__ == '__main__':

    os.environ['MKL_NUM_THREADS'] = '5' 
    os.environ['NUMEXP_NUM_THREADS'] = '5'
    os.environ['OMP_NUM_THREADS'] = '5'

    # Specify project parameters
    #project_folder = '/Users/dockyduncan/Documents/EEG/ping'
    project_folder = '/research/FGB-ETP-DVM/PriorityPing' 
    os.chdir(project_folder)

    # initiate current project
    PO = priorityMap()

    #Run preprocessing 
    #PO.prepareBEH(project, part, factors, labels, project_param, to_filter)

    # Eye movement analyses
    #PO.eyeTowardness(sj = 1)

    #Run preprocessing EEG
    # sj = 25
    # PO.prepareEEG(sj = sj, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
    #         t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
    #         event_id = event_id, project_param = project_param, 
    #         project_folder = project_folder, binary = binary, 
    #         channel_plots = True, inspect = True)

    # Run ping decoding
    # for sj in [25]:
    #     beh, eeg = PO.loadData(sj, name = 'ses-1', eyefilter=False, eye_window=None)  
    #     eeg.baseline = None # temp_fix
    #     bdm = BDM(beh, eeg, 'high_prob_loc', nr_folds= 10, method = 'auc', elec_oi = 'all', baseline = (-0.2,0),downsample = 128, bdm_filter = None)
    #     bdm.Classify(sj, ['yes','no'], 'ping', time = (-0.2,0.6), collapse = False, bdm_labels = [0,2,4,6],  
    #                         excl_factor = None, nr_perm = 0, gat_matrix = False, downscale = False) 


    # Run ping decoding (exclude first blocks regularity)
    for sj in [1,2,3,5,25]:
        beh, eeg = PO.loadData(sj, name = 'ses-1', eyefilter=False, eye_window=None)  
        eeg.baseline = None # temp_fix
        bdm = BDM(beh, eeg, 'high_prob_loc', nr_folds= 10, method = 'auc', elec_oi = 'all', baseline = (-0.2,0),downsample = 128, bdm_filter = None)
        bdm.Classify(sj, ['yes','no'], 'ping', time = (-0.2,0.6), collapse = False, bdm_labels = [0,2,4,6],  
                            excl_factor = dict(block_cnt = [2]), nr_perm = 0, gat_matrix = False, downscale = False) 
# %%
