import os
import mne
import sys
import glob
import pickle
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
sys.path.append('/research/FGB-ETP-DVM/DvM')

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
#from eeg_analyses.ERP import * 
#from eeg_analyses.TF import * 
from eeg_analyses.BDM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
#from stats.nonparametric import *

# subject specific info
sj_info = {'1': {'replace':{}}, # example replace: replace = {'15': {'session_1': {'B1': 'EXG7'}}}
			'3': {'replace':{}},
			'4': {'replace':{}},
			'5': {'replace':{}},
			'6': {'replace':{}},
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
							}

# project specific info
project = 'Benchi'
factors = []
labels = []
to_filter = ['RT'] 
project_param = ['practice','nr_trials','trigger','RT', 'subject_nr', 'correct','dist_high','dist_loc','shape',
                'high_loc', 'target_high','target_loc','shape', 'dist_color','block_cnt']


# eeg info (event_id specified below)
eog =  ['EXG1','EXG2','EXG3','EXG4']
ref =  ['EXG5','EXG6']
event_id = [134,144,131,141,150,120,151,152,153,133,142,130,140]
eeg_runs = [1]
t_min = -0.2
t_max = 0.6
flt_pad = 0.5
binary =  0

# eye tracker info
tracker_ext = None
eye_freq = ''
start_event = ''
tracker_shift = 0
viewing_dist = '?' 
screen_res = '?' 
screen_h = '?'

class singleTrialPd(FolderStructure):

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

    def updateBeh(self, sj, high_loc = True):
        '''
        updates the behavior file such that it can be linked to the raw eeg
        '''

        # read in data file
        beh_file = self.FolderTracker(extension=[
                    'beh', 'raw'], filename='proefpersoon{}new.csv'.format(sj))
        beh = pd.read_csv(beh_file)

        # change column names
        beh.rename(columns = {'prac': 'practice',
                             'subj':'subject_nr',
                             'dist':'dist_color',
                             'runNum':'block_cnt'}, inplace = True)

        # change practice info
        beh.loc[beh.practice == 'practice', 'practice'] = 'yes'
        beh.loc[beh.practice == 'testing', 'practice'] = 'no'
        
        # add column specifying correct respones
        beh['correct'] = 0
        beh.loc[(beh.resp_key == beh.resp_1) ,'correct'] = 1

        # set high probability location info
        beh['dist_high'] = 'low'
        beh.loc[beh.dist_loc.isnull(), 'dist_high'] = 'absent'
        beh['target_high'] = 'low'
        if high_loc:
            locs, counts = np.unique(beh.dist_loc, return_counts= True) 
            high_loc = int(locs[np.argmax(counts)])
            beh['high_loc'] = high_loc
            low_loc = 2 if high_loc == 6 else 2
            beh.loc[beh.dist_loc == high_loc, 'dist_high'] = 'high'
            beh.loc[beh.target_loc == high_loc, 'target_high'] = 'high'
        else:
            beh['high_loc'] = 'None'

        # set trigger info
        beh.loc[(beh.dist_loc == low_loc) & (beh.target_loc == 0),'trigger'] = 134
        beh.loc[(beh.dist_loc == low_loc) & (beh.target_loc == 4),'trigger'] = 144  
        beh.loc[(beh.dist_loc == high_loc) & (beh.target_loc == 0),'trigger'] = 131
        beh.loc[(beh.dist_loc == high_loc) & (beh.target_loc == 4),'trigger'] = 141
        beh.loc[(beh.dist_loc.isnull()) & (beh.target_loc == low_loc),'trigger'] = 150
        beh.loc[(beh.dist_loc.isnull()) & (beh.target_loc == high_loc),'trigger'] = 120
        beh.loc[(beh.dist_loc == high_loc) & (beh.target_loc == low_loc),'trigger'] = 151
        beh.loc[(beh.dist_loc == 0) & (beh.target_loc == low_loc),'trigger'] = 152
        beh.loc[(beh.dist_loc == 4) & (beh.target_loc == low_loc),'trigger'] = 153
        beh.loc[(beh.dist_loc == 4) & (beh.target_loc == 0),'trigger'] = 133
        beh.loc[(beh.dist_loc == 0) & (beh.target_loc == 4),'trigger'] = 142
        beh.loc[(beh.dist_loc.isnull()) & (beh.target_loc == 0),'trigger'] = 130
        beh.loc[(beh.dist_loc.isnull()) & (beh.target_loc == 4),'trigger'] = 140
        
        # add column with priming info
        nr_prime = 0
        prime_info = [0]
        for idx, row in beh[1:].iterrows():

            # check whether dist_loc matches previous trial (and it is not the start of a new block) 
            if row.dist_loc == beh.iloc[idx-1].dist_loc and ~np.isnan(row.dist_loc) and row.block_cnt == beh.iloc[idx-1].block_cnt:
                nr_prime += 1
            else:
                nr_prime = 0

            prime_info += [nr_prime]
        
        beh['prime_info'] = prime_info

        # save file
        beh.to_csv(self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_session_1.csv'.format(sj)))


    def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, event_id, project_param, project_folder, binary, channel_plots, inspect):
        '''
        EEG preprocessing pipeline
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

        EEG.reReference(ref_channels=ref, vEOG=eog[
                        :2], hEOG=eog[2:], changevoltage=True, to_remove = ['EXG7','EXG8'])
        EEG.setMontage(montage='biosemi64')

        #FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
        EEG_ica = EEG.copy()
        EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
                    skip_by_annotation='edge')
        EEG_ica.filter(h_freq=None, l_freq=1.5, fir_design='firwin',
                    skip_by_annotation='edge')

        # MATCH BEHAVIOR FILE
        events = EEG.eventSelection(event_id, binary=binary, min_duration=0)
        self.updateBeh(sj = sj)
        beh, missing = EEG.matchBeh(sj, session, events, event_id, 
                                        headers = project_param)
                                        
        # # EPOCH DATA
        epochs = Epochs(sj, session, EEG, events, event_id=event_id,
                    tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 
        epochs_ica = Epochs(sj, session, EEG_ica, events, event_id=event_id,
                    tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

        
        #epochs.autoRepair()

        # ARTIFACT DETECTION
        epochs.selectBadChannels(channel_plots = False, inspect = True, RT = None)    
        z = epochs.artifactDetection(z_thresh=4, band_pass=[110, 140], plot=True, inspect=True)

        # ICA
        epochs.applyICA(EEG, epochs_ica, method='picard', fit_params = dict(ortho=False, extended=True), inspect = True)

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

if __name__ == '__main__':

    os.environ['MKL_NUM_THREADS'] = '5' 
    os.environ['NUMEXP_NUM_THREADS'] = '5'
    os.environ['OMP_NUM_THREADS'] = '5'

    # Specify project parameters
    project_folder = '/research/FGB-ETP-DVM/single_trial_Pd/JoCn_exp1' 
    os.chdir(project_folder)

    # initiate current project
    PO = singleTrialPd()

    sj = 3
    print('starting subject {}'.format(sj))
    PO.prepareEEG(sj = sj, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
                t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
                event_id = event_id, project_param = project_param, 
                project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)