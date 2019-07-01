
import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import os
import mne
import sys
import glob
import pickle
import ssvepy
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info
sj_info = {'1': {'tracker': (False, '', '', '',0), 'replace':{}},
            } 

# project specific info
project = 'Binocular'
part = 'beh' 
project_param = []

montage = mne.channels.read_montage(kind='biosemi64')

# THIS IS WHAT IT SHOULD BE
#eog =  ['V_up','V_do','H_r','H_l']
#ref =  ['Ref_r','Ref_l']
# THIS IS WHAT IT IS
eog =  ['Ref_r','Ref_l','V_up','V_do']
ref =  ['H_r','H_l']
trigger = dict(resp1=49, resp2=50, resp3 = 51)
t_min = -1
t_max = 5
flt_pad = 0.5
eeg_runs = [1] # 3 runs for subject 15 session 2
binary = 0

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class Binocular(FolderStructure):

        def __init__(self): pass

        def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, channel_plots, inspect):
            '''
            EEG preprocessing as preregistred @ https://osf.io/b2ndy/register/5771ca429ad5a1020de2872e
            '''

            # set subject specific parameters
            file = 'subject_{}_session_{}_'.format(sj, session)
            replace = sj_info[str(sj)]['replace']
            #tracker, ext, t_freq, start_event, shift = sj_info[str(sj)]['tracker']

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
                                                montage=None, preload=True, eog=eog) for run in eeg_runs])

            # temp code to get rid of unneccassary data
            to_remove = ['{}{}'.format(letter,i) for i in range(1,33) for letter in ['C','D','E','F','G','H']] 
            to_remove += ['GSR1','GSR2','Erg1','Erg2','Resp','Temp','Plet']
            for i, elec in enumerate(to_remove):
                if elec in ['F1','F2','F3','F4','F5','F6','F7','F8','C1','C2','C3','C4','C5','C6']:
                    to_remove[i] += '-1'
            EEG.drop_channels(to_remove)                                   

            #EEG.replaceChannel(sj, session, replace)
            EEG.reReference(ref_channels=ref, vEOG=eog[
                            :2], hEOG=eog[2:], changevoltage=True, to_remove = ['EXG7','EXG8'])
            EEG.setMontage(montage='biosemi64')

            #FILTER DATA FOR EPOCHING
            EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
                            skip_by_annotation='edge')

            # MATCH BEHAVIOR FILE
            events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
            #beh, missing, events = self.matchBeh(sj, session, events, trigger, 
            #                             headers = project_param)

            # EPOCH DATA
            epochs = Epochs(sj, session, EEG, events, event_id=trigger,
                    tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

            # ARTIFACT DETECTION
            #epochs.selectBadChannels(channel_plots = channel_plots, inspect=inspect, RT = None)    
            epochs.artifactDetection(inspect=inspect, run = True)

            # INTERPOLATE BADS
            epochs.interpolate_bads(reset_bads=True, mode='accurate')

            # save eeg 
            epochs.save(self.FolderTracker(extension=[
                        'processed'], filename='subject-{}_all-epo.fif'.format(sj, session)), split_size='2GB')


if __name__ == '__main__':

    #os.environ['MKL_NUM_THREADS'] = '5' 
    #os.environ['NUMEXP_NUM_THREADS'] = '5'
    #os.environ['OMP_NUM_THREADS'] = '5'

    # Specify project parameters
    project_folder = '/home/dvmoors1/BB/Binocular'
    os.chdir(project_folder)
    PO = Binocular()

    # run actual preprocessing
    for sj in [1]:
        print 'starting subject {}'.format(sj)
        # do preprocessing
        #PO.prepareEEG(1, 1, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, True, True)
    
        # run decoding 
        beh, eeg = PO.loadData(sj, False,  beh_file = False)
        embed()
        bdm = BDM(beh, eeg, decoding = 'condition', nr_folds = 10, method = 'acc', elec_oi = 'all', downsample = 128)
        bdm.Classify(sj, cnds = 'all', cnd_header = 'condition', time = (-0.2, 1.5), bdm_labels = [49,51],gat_matrix = False)
        
        
        # load data (for ssvp)
        epoch = mne.read_epochs(PO.FolderTracker(extension = ['processed'], 
							filename = 'subject-{}_all-epo.fif'.format(sj)))
        embed()
        ssvep_example = ssvepy.Ssvep(epoch, [6, 7.5], fmin=0.5, fmax=30)
        ssvep_example.plot_psd()
        embed()
        plt.show()


