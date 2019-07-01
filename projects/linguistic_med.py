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
project = 'Linguistic'
part = 'beh' 
project_param = []

montage = mne.channels.read_montage(kind='biosemi64')

# THIS IS WHAT IT SHOULD BE
#eog =  ['V_up','V_do','H_r','H_l']
#ref =  ['Ref_r','Ref_l']
# THIS IS WHAT IT IS
eog =  ['Ref_r','Ref_l','V_up','V_do']
ref =  ['H_r','H_l']
trigger = dict(neutral=10, positive=20, negative = 30)
t_min = 0.2
t_max = 2
flt_pad = 0.5
eeg_runs = [1] # 3 runs for subject 15 session 2
binary = 61440

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class Linguistic(FolderStructure):

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

        def checkN400(self, beh, eeg, time = (-0.2, 0.8), elec = ['Fz']):


            eeg.filter(l_freq = None, h_freq = 30)
            s, e = [np.argmin(abs(eeg.times - t)) for t in time]
            elec_idx = [eeg.ch_names.index(el) for el in elec]
            eegs = eeg._data[:,elec_idx,s:e]
            times = eeg.times[s:e]

            for label, cnd in [(10,'negative'),(20,'neutral'),(30,'positive')]: 
                # read in condition data
                data = eeg[cnd]._data[:,elec_idx,s:e]

                # do baselining (using functionality from toolbox)
                data = ERP.baselineCorrect(data, times, (-0.2,0))

                # create ERP
                erp = data.mean(axis = (0,1))   

                plt.plot(times, erp, label = cnd)

            plt.legend(loc = 'best')
            sns.despine(offset= 0, trim = False)
            plt.savefig(PO.FolderTracker(['erp','figs'], filename = 'N400-test.pdf'))		
            plt.close()



if __name__ == '__main__':

    #os.environ['MKL_NUM_THREADS'] = '5' 
    #os.environ['NUMEXP_NUM_THREADS'] = '5'
    #os.environ['OMP_NUM_THREADS'] = '5'

    # Specify project parameters
    project_folder = '/home/dvmoors1/BB/Linguistic'
    os.chdir(project_folder)
    PO = Linguistic()

    # run actual preprocessing
    for sj in [1]:
        print 'starting subject {}'.format(sj)
        # do preprocessing
        #PO.prepareEEG(1, 1, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, True, True)

        # run decoding 
        beh, eeg = PO.loadData(sj, False,  beh_file = False)
        embed()
        PO.checkN400(beh, eeg)
        
        bdm = BDM(beh, eeg, to_decode = 'condition', nr_folds = 10, method = 'acc', elec_oi = 'all', downsample = 128)
        bdm.Classify(sj, cnds = 'all', cnd_header = 'condition', time = (-0.2, 2), bdm_labels = [10,20],gat_matrix = False)