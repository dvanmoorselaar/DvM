"""
analyze EEG data

Created by Dirk van Moorselaar on 10-03-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

import mne
import os
import logging
import itertools
import pickle
import copy
import glob
import sys
import time
import itertools

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import zscore

from typing import Optional, Generic, Union, Tuple, Any
from termios import tcflush, TCIFLUSH
from eeg_analyses.EYE import *
from math import sqrt
from IPython import embed
from support.FolderStructure import *
from scipy.stats.stats import pearsonr
from mne.viz.epochs import plot_epochs_image
from mne.filter import filter_data
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from math import ceil, floor
from autoreject import Ransac, AutoReject

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

class RawBDF(mne.io.edf.edf.RawEDF, FolderStructure):
    '''
    Child originating from MNE built-in RawEDF, such that new methods can be added to this built in class
    '''

    def __init__(self, input_fname, eog=None, stim_channel=-1,
                exclude=(), preload=True, verbose=None):

        super(RawBDF, self).__init__(input_fname=input_fname, eog=eog,
                                     stim_channel=stim_channel, preload=preload, verbose=verbose)

    def report_raw(self, report, events, event_id):
        '''

        '''

        # report raw
        report.add_raw(self, title='raw EEG', psd=True )
        # and events
        events = events[np.in1d(events[:,2], event_id)]
        report.add_events(events, title = 'detected events', sfreq = self.info['sfreq'])
       

        return report



    def replaceChannel(self, sj, session, replace):
        '''
        Replace bad electrodes by electrodes that were used during recording as a replacement

        Arguments
        - - - - -
        raw (object): raw mne eeg object
        sj (int): subject_nr
        session (int): eeg session number
        replace (dict): dictionary containing to be replaced electrodes per subject and session

        Returns
        - - - -
        self(object): raw object with bad electrodes replaced
        '''

        sj = str(sj)
        session = 'session_{}'.format(session)

        if sj in replace.keys():
            if session in replace[sj].keys():
                to_replace = replace[sj][session].keys()
                for e in to_replace:
                    self._data[self.ch_names.index(e), :] = self._data[
                        self.ch_names.index(replace[sj][session][e]), :]

                    # print('Electrode {0} replaced by
                    # {1}'.format(e,replace[sj][session][e]))

    def reReference(self, ref_channels=['EXG5', 'EXG6'], vEOG=['EXG1', 'EXG2'], hEOG=['EXG3', 'EXG4'], changevoltage=True, to_remove = ['EXG7','EXG8']):
        '''
        Rereference raw data to reference channels. By default data is rereferenced to the mastoids.
        Also EOG data is rerefenced. Subtraction of VEOG and HEOG results in a VEOG and an HEOG channel.
        After rereferencing redundant channels are removed. Functions assumes that there are 2 vEOG and 2
        hEOG channels.

        Arguments
        - - - - -
        self(object): RawBDF object
        ref_channels (list): list with channels for rerefencing
        vEOG (list): list with vEOG channels
        hEOG (list): list with hEOG channels
        changevoltage (bool):
        remove(bool): Specify whether channels need to be removed

        Returns
        - - - -

        self (object): Rereferenced raw eeg data
        '''

        # change data format from Volts to microVolts
        if changevoltage:
            self._data[:-1, :] *= 1e6
            print('Volts changed to microvolts')
            logging.info('Volts changed to microvolts')

        # rereference all EEG channels to reference channels
        self.set_eeg_reference(ref_channels=ref_channels)
        to_remove += ref_channels
        print('EEG data was rereferenced to channels {}'.format(ref_channels))
        logging.info(
            'EEG data was rereferenced to channels {}'.format(ref_channels))

        # select eog channels
        eog = self.copy().pick_types(eeg=False, eog=True)
        
        # # rerefence EOG data (vertical and horizontal)
        # idx_v = [eog.ch_names.index(vert) for vert in vEOG]
        # idx_h = [eog.ch_names.index(hor) for hor in hEOG]

        # if len(idx_v) == 2:
        #     eog._data[idx_v[0]] -= eog._data[idx_v[1]]
        # if len(idx_h) == 2:   
        #     eog._data[idx_h[0]] -= eog._data[idx_h[1]]

        # print(
        #     'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')
        # logging.info(
        #     'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')
        
        # # add rereferenced vEOG and hEOG data to self
        # ch_mapping = {vEOG[0]: 'VEOG', hEOG[0]: 'HEOG'}
        # eog.rename_channels(ch_mapping)
        # eog.drop_channels([vEOG[1], hEOG[1]])
        # #self.add_channels([eog])

        # drop ref chans
        self.drop_channels(to_remove)
        print('Reference channels and empty channels removed')
        logging.info('Reference channels and empty channels removed')

    def setMontage(self, montage='biosemi64', ch_remove = []):
        '''
        Uses mne function to set the specified montage. Also changes channel labels from A, B etc 
        naming scheme to standard naming conventions and removes specified channels.
         At the same time changes the name of EOG electrodes (assumes an EXG naming scheme)

        Arguments
        - - - - -
        raw (object): raw mne eeg object
        montage (str): used montage during recording
        ch_remove (list): channels that you want to exclude from analysis (e.g heart rate)

        Returns
        - - - -
        self(object): raw object with changed channel names following biosemi 64 naming scheme (10 - 20 system)
        '''

        # drop channels and get montage
        self.drop_channels(ch_remove)

        # create mapping dictionary
        idx = 0
        ch_mapping = {}
        if self.ch_names[0] == 'A1':
            for hemi in ['A', 'B']:
                for electr in range(1, 33):
                    ch_mapping.update(
                        {'{}{}'.format(hemi, electr): montage.ch_names[idx]})
                    idx += 1

        self.rename_channels(ch_mapping)
        self.set_montage(montage=montage)

        print('Channels renamed to 10-20 system, and montage added')
        logging.info('Channels renamed to 10-20 system, and montage added')

    def eventSelection(self, trigger, binary=0, consecutive=False, min_duration=0.003):
        '''
        Returns array of events necessary for epoching.

        Arguments
        - - - - -
        raw (object): raw mne eeg object
        binary (int): is subtracted from stim channel to control for spoke triggers  (e.g. subtracts 3840)

        Returns
        - - - -
        events(array): numpy array with trigger events (first column contains the event time in samples and the third column contains the event id)
        '''

        self._data[-1, :] -= binary # Make universal
   
        events = mne.find_events(self, stim_channel=None, consecutive=consecutive, min_duration=min_duration)    

        # Check for consecutive 
        if not consecutive:
            spoke_idx = []
            for i in range(events[:-1,2].size):
                if events[i,2] == events[i + 1,2] and events[i,2] in trigger:
                    spoke_idx.append(i)

            events = np.delete(events,spoke_idx,0)    
            logging.info('{} spoke events removed from event file'.format(len(spoke_idx)))    

        return events

    def matchBeh(self, sj, session, events, event_id, trigger_header = 'trigger', headers = []):
        '''
        Alligns bdf file with csv file with experimental variables

        Arguments
        - - - - -
        raw (object): raw mne eeg object
        sj (int): sj number 
        session(int): session number
        events(array): event file from eventSelection (last column contains trigger values)
        trigger(list|array): trigger values used for epoching
        headers (list): relevant column names from behavior file

        Returns
        - - - -
        beh (object): panda object with behavioral data (triggers are alligned)
        missing (araray): array of missing trials (can be used when selecting eyetracking data)
        '''

        # read in data file
        beh_file = self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_session_{}.csv'.format(sj, session))

        # get triggers logged in beh file
        beh = pd.read_csv(beh_file)
        beh = beh[headers]
        if 'practice' in headers:
            beh = beh[beh['practice'] == 'no']
            beh = beh.drop(['practice'], axis=1)
        beh_triggers = beh[trigger_header].values  

        # get triggers bdf file
        if type(event_id) == dict:
            event_id = [event_id[key] for key in event_id.keys()]
        idx_trigger = [idx for idx, tr in enumerate(events[:,2]) if tr in event_id] 
        bdf_triggers = events[idx_trigger,2] 

        # log number of unique triggers
        unique = np.unique(bdf_triggers)
        logging.info('{} detected unique triggers (min = {}, max = {})'.
                        format(unique.size, unique.min(), unique.max()))

        # make sure trigger info between beh and bdf data matches
        missing_trials = []
        nr_miss = beh_triggers.size - bdf_triggers.size
        logging.info('{} trials will be removed from beh file'.format(nr_miss))

        # check whether trial info is present in beh file
        if nr_miss > 0 and 'nr_trials' not in beh.columns:
            raise ValueError('Behavior file does not contain a column with trial info named nr_trials. Please adjust')

        while nr_miss > 0:
            stop = True
            # continue to remove beh trials until data files are lined up
            for i, tr in enumerate(bdf_triggers):
                if tr != beh_triggers[i]: # remove trigger from beh_file
                    miss = beh['nr_trials'].iloc[i]
                    #print miss
                    missing_trials.append(miss)
                    logging.info('Removed trial {} from beh file,because no matching trigger exists in bdf file'.format(miss))
                    beh.drop(beh.index[i], inplace=True)
                    beh_triggers = np.delete(beh_triggers, i, axis = 0)
                    nr_miss -= 1
                    stop = False
                    break
                 
            # check whether there are missing trials at end of beh file        
            if beh_triggers.size > bdf_triggers.size and stop:

                # drop the last items from the beh file
                missing_trials = np.hstack((missing_trials, beh['nr_trials'].iloc[-nr_miss:].values))
                beh.drop(beh.index[-nr_miss:], inplace=True)
                logging.info('Removed last {} trials because no matches detected'.format(nr_miss))         
                nr_miss = 0
      
        # keep track of missing trials to allign eye tracking data (if available)   
        missing = np.array(missing_trials)  
   
        # log number of matches between beh and bdf    
        logging.info('{} matches between beh and epoched data out of {}'.
            format(sum(beh[trigger_header].values == bdf_triggers), bdf_triggers.size))           

        return beh, missing

class Epochs(mne.Epochs, FolderStructure):
    '''
    Child originating from MNE built-in Epochs, such that new methods can be added to this built in class
    '''

    def __init__(self, sj, session, raw, events, event_id, tmin, tmax, flt_pad=True, baseline=(None, None), picks=None, preload=True,
                 reject=None, flat=None, proj=False, decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                 on_missing='error', reject_by_annotation=False, verbose=None, z_thresh = 4):

        # check whether a preprocessed folder for the current subject exists,
        # if not make onen
        self.sj = sj
        self.session = str(session)
        self.flt_pad = flt_pad
        tmin, tmax = tmin - flt_pad, tmax + flt_pad
        
        super(Epochs, self).__init__(raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                                     baseline=baseline, picks=picks, preload=preload, reject=reject,
                                     flat=flat, proj=proj, decim=decim, reject_tmin=reject_tmin,
                                     reject_tmax=reject_tmax, detrend=detrend, on_missing=on_missing,
                                     reject_by_annotation=reject_by_annotation, verbose=verbose)

        # save number of detected events
        self.nr_events = len(self)
        self.drop_beh = []
        logging.info('{} epochs created'.format(len(self)))

    def report_epochs(self, report, title, missing = None):

        if missing is not None:
            report.add_html(missing, title = 'missing trials in beh')

        report.add_epochs(self, title=title)

        return report



    def align_behavior(self, events: np.array,  trigger_header: str = 'trigger', headers: list = [], bdf_remove: np.array =  None):
        """
        Aligns bdf file with csv file with experimental variables. In case there are more behavioral trials than eeg trials
        (e.g., because trigger was not properly sent/detected), trials are removed from the raw behavioral data such that
        both datasets are aligned. Information about this process can be found in individual preprocessing log files. 

        Args:
            events (np.array): event info as returned by RAW.event_selection
            trigger_header (str, optional): Column in raw behavior that contains trigger values used for epoching. Defaults to 'trigger'.
            headers (list, optional): List of headers that should be linked to eeg data. Defaults to [].
            bdf_remove (np.array, optional): Indices of trigger events that need to be removed. Only specify when to many trials are recorded. 

        Raises:
            ValueError: In case behavior and eeg data do not align (i.e., contain different trial numbers). If there are more behavior trials than epochs,
            raises an error in case there is no column 'nr_trials', which prevents informed alignment of eeg and behavior. Also raises an error if there are
            too many epochs and automatic allignment fails

        Returns:
            beh (pd.DataFrame): Behavior data after aligning to eeg data (index is reset)
            missing (np.array): array with trials that are removed from beh because no matching trigger was detected. Is used 
            when aligning eyetracker data (where it is assumed that eyetracking data and raw behavior contain the same number of trials)
        """

        print('Linking behavior to eeg data')
        report_str = ''

        # read in data file
        beh_file = self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_session_{}.csv'.format(self.sj, self.session))

        # get triggers logged in beh file
        beh = pd.read_csv(beh_file)
        beh = beh[headers]
        if 'practice' in headers:
            print('{} practice trials removed from behavior'.format(beh[beh.practice == 'yes'].shape[0]))
            #logging.info('{} practice trials removed from behavior'.format(beh[beh.practice == 'yes'].shape[0]))
            beh = beh[beh.practice == 'no']
            beh = beh.drop(['practice'], axis=1)
            beh.reset_index(inplace = True, drop = True)
        beh_triggers = beh[trigger_header].values 
        
        # get eeg triggers in epoched order ()
        bdf_triggers = events[self.selection, 2]
        if bdf_remove is not  None:
            self.drop(bdf_remove)
            report_str += '{} bdf triggers removed as specified by the user \n'.format(bdf_remove.size)
            #logging.info('{} bdf triggers and epochs removed as specified by the user'.format(bdf_remove.size))
            bdf_triggers = np.delete(bdf_triggers, bdf_remove)

        # log number of unique triggers
        #unique = np.unique(bdf_triggers)
        #logging.info('{} detected unique triggers (min = {}, max = {})'.
        #                format(unique.size, unique.min(), unique.max()))

        # make sure trigger info between beh and bdf data matches
        missing_trials = []
        nr_miss = beh_triggers.size - bdf_triggers.size

        #logging.info(f'{nr_miss} trials will be removed from beh file')
        if nr_miss > 0:
            report_str += f'Behavior has {nr_miss} more trials than detected events. The following trial numbers \
                          will be removed in attempt to fix this: \n'
        # check whether trial info is present in beh file
        if nr_miss > 0 and 'nr_trials' not in beh.columns:
            raise ValueError('Behavior file does not contain a column with trial info named nr_trials. Please adjust')
        elif nr_miss < 0:
            report_str += 'EEG events are removed in an attempt to align the data. Please inspect your data carefully! \n'
            while nr_miss < 0:
                # continue to remove bdf triggers until data files are lined up 
                for i, tr in enumerate(beh_triggers):
                    if tr != bdf_triggers[i]: # remove trigger from eeg_file
                        bdf_triggers = np.delete(bdf_triggers, i, axis = 0)
                        nr_miss += 1
            # check file sizes
            if sum(beh_triggers == bdf_triggers) < bdf_triggers.size:
                raise ValueError('Behavior and eeg cannot be linked as too many eeg triggers received. Please pass indices of trials to be removed \
                to subject_info dict with key bdf_remove')

        while nr_miss > 0:
            stop = True
            # continue to remove beh trials until data files are lined up
            for i, tr in enumerate(bdf_triggers):
                if tr != beh_triggers[i]: # remove trigger from beh_file
                    miss = beh['nr_trials'].iloc[i]
                    missing_trials.append(i)
                    report_str += f'{miss}, '
                    #logging.info(f'Removed trial {miss} from beh file,because no matching trigger exists in bdf file')
                    beh.drop(beh.index[i], inplace=True)
                    beh_triggers = np.delete(beh_triggers, i, axis = 0)
                    nr_miss -= 1
                    stop = False
                    break
                 
            # check whether there are missing trials at end of beh file        
            if beh_triggers.size > bdf_triggers.size and stop:

                # drop the last items from the beh file
                missing_trials = np.hstack((missing_trials, beh.index[-nr_miss:].values))
                beh.drop(beh.index[-nr_miss:], inplace=True)
                #logging.info('Removed last {} trials because no matches detected'.format(nr_miss))    
                report_str += f'\n Removed final {nr_miss} trials from behavior to allign data. Please inspect your data carefully!'     
                nr_miss = 0
      
        # keep track of missing trials to allign eye tracking data (if available)   
        missing = np.array(missing_trials)  
        beh.reset_index(inplace = True)

        # add behavior to epochs object
        self.metadata = beh

        # log number of matches between beh and bdf    
        #logging.info('{} matches between beh and epoched data out of {}'.
        #    format(sum(beh[trigger_header].values == bdf_triggers), bdf_triggers.size))  
        report_str += '\n {} matches between beh and epoched data out of {}'.format(sum(beh[trigger_header].values == bdf_triggers), bdf_triggers.size) 

        return missing, report_str

    def autoRepair(self):
        '''

        '''
        # select eeg channels
        picks = mne.pick_types(self.info, meg=False, eeg=True, stim=False, eog=False,
                       include=[], exclude=[])

        # initiate parameters p and k
        n_interpolates = np.array([1, 4, 32])
        consensus_percs = np.linspace(0, 1.0, 11)

        ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                thresh_method='random_search', random_state=42)
        
        self, reject_log = ar.fit_transform(self, return_log=True)  

    def select_bad_channels(self, report):

        # step 1: run ransac
        bad_chs = self.apply_ransac()

        # step 2: create report
        self.report_ransac(bad_chs, report)

        # step 3: manual inspection


    def report_ransac(self, bad_chs, report):

        figs = []
        for ch in bad_chs:
            figs += self.plot_image(picks = ch)
        report.add_figure(figs, title = 'Bad channels selected by Ransac')

    def apply_ransac(self):
        '''
        Implements RAndom SAmple Consensus (RANSAC) method to detect bad channels.

        Returns
        - - - -
        self.info['bads']: list with all bad channels detected by the RANSAC algorithm

        '''

        # select channels to display
        picks = mne.pick_types(self.info, eeg=True, exclude='bads')

        # use Ransac, interpolating bads and append bad channels to self.info['bads']
        ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
        epochs_clean = ransac.fit_transform(self)
        print('The following electrodes are selected as bad by Ransac:')
        print('\n'.join(ransac.bad_chs_))

        return ransac.bad_chs_


    def selectBadChannels(self, run_ransac = True, channel_plots=True, inspect=True, n_epochs=10, n_channels=32, RT = None):
        '''

        '''

        logging.info('Start selection of bad channels')
        #matplotlib.style.use('classic')

        # select channels to display
        picks = mne.pick_types(self.info, eeg=True, exclude='bads')

        # plot epoched data
        if channel_plots:

            for ch in picks:
                # plot evoked responses across channels
                try:  # handle bug in mne for plotting undefined x, y coordinates
                    plot_epochs_image(self.copy().crop(self.tmin + self.flt_pad,self.tmax - self.flt_pad), ch, show=False, overlay_times = RT)
                    plt.savefig(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session, 'channel_erps'], filename='{}.pdf'.format(self.ch_names[ch])))
                    plt.close()

                except:
                    plt.savefig(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session,'channel_erps'], filename='{}.pdf'.format(self.ch_names[ch])))
                    plt.close()

                self.plot_psd(picks = [ch], show = False)   
                plt.savefig(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session,'channel_erps'], filename='_psd_{}'.format(self.ch_names[ch])))
                plt.close() 

            # plot power spectra topoplot to detect any clear bad electrodes
            self.plot_psd_topomap(bands=[(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (
                12, 30, 'Beta'), (30, 45, 'Gamma'), (45, 100, 'High')], show=False)
            plt.savefig(self.FolderTracker(extension=[
                        'preprocessing', 'subject-{}'.format(self.sj), self.session], filename='psd_topomap.pdf'))
            plt.close()

        if run_ransac:
            self.applyRansac()

        if inspect:
            # display raw eeg with 50mV range
            self.plot(block=True, n_epochs=n_epochs,
                      n_channels=n_channels, picks=picks, scalings=dict(eeg=50))

            if self.info['bads'] != []:
                with open(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(self.sj),
                	self.session], filename='marked_bads.txt'), 'wb') as handle:
                    pickle.dump(self.info['bads'], handle)

        else:
            try:
                with open(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(self.sj),
                	self.session], filename='marked_bads.txt'), 'rb') as handle:
                    self.info['bads'] = pickle.load(handle)

                print('The following channals were read in as bads from a txt file: {}'.format(
                    self.info['bads']))
            except:
                print('No bad channels selected')

        logging.info('{} channels marked as bad: {}'.format(
            len(self.info['bads']), self.info['bads']))

    def automatic_artifact_detection(self, z_thresh=4, band_pass=[110, 140], plot=True, inspect=True):
        """ Detect artifacts> modification of FieldTrip's automatic artifact detection procedure 
        (https://www.fieldtriptoolbox.org/tutorial/automatic_artifact_rejection/). 
        Artifacts are detected in three steps:
        1. Filtering the data within specified frequency range
        2. Z-transforming the filtered data across channels and normalize it over channels
        3. Threshold the accumulated z-score

        Counter to fieldtrip the z_threshold is ajusted based on the noise level within the data
        Note: all data included for filter padding is now taken into consideration to calculate z values

        Afer running this function, Epochs contains information about epeochs marked as bad (self.marked_epochs)
        
        Arguments:
            
        Keyword Arguments:
            z_thresh {float|int} -- Value that is added to difference between median 
                    and min value of accumulated z-score to obtain z-threshold
            band_pass {list} --  Low and High frequency cutoff for band_pass filter
            plot {bool} -- If True save detection plots (overview of z scores across epochs, 
                    raw signal of channel with highest z score, z distributions, 
                    raw signal of all electrodes)
            inspect {bool} -- If True gives the opportunity to overwrite selected components
        """
        embed()
        # select data for artifact rejection
        sfreq = self.info['sfreq']
        self_copy = self.copy() 
        self_copy.pick_types(eeg=True, exclude='bads')

        #filter data and apply Hilbert
        self_copy.filter(band_pass[0], band_pass[1], fir_design='firwin', pad='reflect_limited') 
        #self_copy.filter(band_pass[0], band_pass[1], method='iir', iir_params=dict(order=6, ftype='butter'))    
        self_copy.apply_hilbert(envelope=True)

        # get the data and apply box smoothing
        data = self_copy.get_data()
        nr_epochs = data.shape[0]
        for i in range(data.shape[0]):
            data[i] = self.boxSmoothing(data[i])

        # get the data and z_score over electrodes
        data = data.swapaxes(0,1).reshape(data.shape[1],-1) 
        z_score = zscore(data, axis = 1) # check whether axis is correct!!!!!!!!!!!

        # normalize z_score
        z_score = z_score.sum(axis = 0)/sqrt(data.shape[0]) 
        #z_score = filter_data(z_score, self.info['sfreq'], None, 4, pad='reflect_limited') 

        # adjust threshold (data driven)    
        z_thresh += np.median(z_score) + abs(z_score.min() - np.median(z_score)) 

        # transform back into epochs
        z_score = z_score.reshape(nr_epochs, -1) 

        # control for filter padding
        if self.flt_pad > 0:
            idx_ep = self.time_as_index([self.tmin + self.flt_pad, self.tmax - self.flt_pad])
            z_score = z_score[:, slice(*idx_ep)]

        # mark bad epochs
        bad_epochs = []
        cnt = 0
        for ep, X in enumerate(z_score):
            noise_smp = np.where(X > z_thresh)[0]
            if noise_smp.size > 0:
                bad_epochs.append(ep)

        if inspect:
            print('This interactive window selectively shows epochs marked as bad. You can overwrite automatic artifact detection by clicking on selected epochs')
            bad_eegs = self[bad_epochs]
            idx_bads = bad_eegs.selection
            # display bad eegs with 50mV range
            bad_eegs.plot(
                n_epochs=5, n_channels=data.shape[1], scalings=dict(eeg = 50))
            plt.show()
            plt.close()
            missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eegs.selection],dtype = int)
            logging.info('Manually ignored {} epochs out of {} automatically selected({}%)'.format(
                            missing.size, len(bad_epochs),100 * round(missing.size / float(len(bad_epochs)), 2)))
            bad_epochs = np.delete(bad_epochs, missing)
        
        if plot:
            plt.figure(figsize=(10, 10))
            with sns.axes_style('dark'):

                plt.subplot(111, xlabel='samples', ylabel='z_value',
                            xlim=(0, z_score.size), ylim=(-20, 40))
                plt.plot(np.arange(0, z_score.size), z_score.flatten(), color='b')
                plt.plot(np.arange(0, z_score.size),
                         np.ma.masked_less(z_score.flatten(), z_thresh), color='r')
                plt.axhline(z_thresh, color='r', ls='--')

                plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                    self.sj), self.session], filename='automatic_artdetect.pdf'))
                plt.close()

        # drop bad epochs and save list of dropped epochs
        self.drop_beh = bad_epochs
        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
            self.sj), self.session], filename='noise_epochs.txt'), bad_epochs)
        print('{} epochs dropped ({}%)'.format(len(bad_epochs),
                                               100 * round(len(bad_epochs) / float(len(self)), 2)))
        logging.info('{} epochs dropped ({}%)'.format(
            len(bad_epochs), 100 * round(len(bad_epochs) / float(len(self)), 2)))
        self.drop(np.array(bad_epochs), reason='art detection ecg')
        logging.info('{} epochs left after artifact detection'.format(len(self)))

        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(self.sj), self.session], filename='automatic_artdetect.txt'),
                   ['Artifact detection z threshold set to {}. \n{} epochs dropped ({}%)'.
                    format(round(z_thresh, 1), len(bad_epochs), 100 * round(len(bad_epochs) / float(len(self)), 2))], fmt='%.100s')

        return z_thresh

    def artifactDetectionOLD(self, z_cutoff=4, band_pass=[110, 140], min_dur = 0.05, min_nr_art = 1, run = True, plot=True, inspect=True):
        """ Detect artifacts based on FieldTrip's automatic artifact detection. 
        Artifacts are detected in three steps:
        1. Filtering the data (6th order butterworth filter)
        2. Z-transforming the filtered data and normalize it over channels
        3. Threshold the accumulated z-score

        False-positive transient peaks are prevented by low-pass filtering the resulting z-score time series at 4 Hz.

        Afer running this function, Epochs contains information about epeochs marked as bad (self.marked_epochs)
        
        Arguments:
            
        Keyword Arguments:
            z_cuttoff {int} -- Value that is added to difference between median 
                    nd min value of accumulated z-score to obtain z-threshold
            band_pass {list} --  Low and High frequency cutoff for band_pass filter
            min_dur {float} -- minimum duration of detected artefects to be considered an artefact
            min_nr_art {int} -- minimum number of artefacts that may be present in an epoch (irrespective of min_dur)
            run {bool} -- specifies whether analysis is run a new or whether bad epochs are read in from memory
            plot {bool} -- If True save detection plots (overview of z scores across epochs, 
                    raw signal of channel with highest z score, z distributions, 
                    raw signal of all electrodes)
            inspect {bool} -- If True gives the opportunity to overwrite selected components
            time {tuple} -- Time window used for decoding
            tr_header {str} -- Name of column that contains training labels
            te_header {[type]} -- Name of column that contains testing labels
        """

        # select channels for artifact detection
        picks = mne.pick_types(self.info, eeg=True, exclude='bads')
        nr_channels = picks.size
        sfreq = self.info['sfreq']

        # control for filter padding
        if self.flt_pad > 0:
            idx_ep = self.time_as_index([self.tmin + self.flt_pad, self.tmax - self.flt_pad])
            timings = self.times[idx_ep[0]:idx_ep[1]]
        ep_data = []

        # STEP 1: filter each epoch data, apply hilbert transform and boxsmooth
        # the resulting data before removing filter padds
        if run:
            print('Started artifact detection')
            logging.info('Started artifact detection')
            for epoch, X in enumerate(self):

                # CHECK IF THIS IS CORRECT ORDER IN FIELDTRIP CODE / ALSO CHANGE
                # FILTER TO MNE 0.14 STANDARD
                X = filter_data(X[picks, :], sfreq, band_pass[0], band_pass[
                                1], method='iir', iir_params=dict(order=6, ftype='butter'))
                X = np.abs(sp.signal.hilbert(X))
                X = self.boxSmoothing(X)

                X = X[:, idx_ep[0]:idx_ep[1]]
                ep_data.append(X)

            # STEP 2: Z-transform data
            epoch = np.hstack(ep_data)
            avg_data = epoch.mean(axis=1).reshape(-1, 1)
            std_data = epoch.std(axis=1).reshape(-1, 1)
            z_data = [(ep - avg_data) / std_data for ep in ep_data]

            # STEP 3 threshold z-score per epoch
            z_accumel = np.hstack(z_data).sum(axis=0) / sqrt(nr_channels)
            z_accumel_ep = [np.array(z.sum(axis=0) / sqrt(nr_channels))
                            for z in z_data]
            z_thresh = np.median(
                z_accumel) + abs(z_accumel.min() - np.median(z_accumel)) + z_cutoff

            # split noise epochs based on start and end time
            # and select bad epochs based on specified criteria
            bad_epochs = []
            for ep, X in enumerate(z_accumel_ep):
                noise_smp = np.where((X > z_thresh) == True)[0]
                noise_smp = np.split(noise_smp, np.where(np.diff(noise_smp) != 1)[0]+1)
                time_inf = [timings[smp[-1]] - timings[smp[0]]  for smp in noise_smp
                            if smp.size > 0]
                if len(time_inf) > 0:
                    if max(time_inf) > min_dur or len(time_inf) > min_nr_art:
                        bad_epochs.append(ep)

            if plot:
                plt.figure(figsize=(10, 10))
                with sns.axes_style('dark'):

                    plt.subplot(111, xlabel='samples', ylabel='z_value',
                                xlim=(0, z_accumel.size), ylim=(-20, 40))
                    plt.plot(np.arange(0, z_accumel.size), z_accumel, color='b')
                    plt.plot(np.arange(0, z_accumel.size),
                             np.ma.masked_less(z_accumel, z_thresh), color='r')
                    plt.axhline(z_thresh, color='r', ls='--')

                    plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                        self.sj), self.session], filename='automatic_artdetect.pdf'))
                    plt.close()

            bad_epochs = np.array(bad_epochs)
        else:
            logging.info('Bad epochs read in from file')
            bad_epochs = np.loadtxt(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session], 
                                filename='noise_epochs.txt'))    

        if inspect:
            print('You can now overwrite automatic artifact detection by clicking on epochs selected as bad')
            bad_eegs = self[bad_epochs]
            idx_bads = bad_eegs.selection
            # display bad eegs with 50mV range
            bad_eegs.plot(
                n_epochs=5, n_channels=picks.size, picks=picks, scalings=dict(eeg = 50))
            plt.show()
            plt.close()
            missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eegs.selection], dtype = int)
            logging.info('Manually ignored {} epochs out of {} automatically selected({}%)'.format(
                            missing.size, bad_epochs.size,100 * round(missing.size / float(bad_epochs.size), 2)))
            bad_epochs = np.delete(bad_epochs, missing)

        # drop bad epochs and save list of dropped epochs
        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
            self.sj), self.session], filename='noise_epochs.txt'), bad_epochs)
        print('{} epochs dropped ({}%)'.format(len(bad_epochs),
                                               100 * round(len(bad_epochs) / float(len(self)), 2)))
        logging.info('{} epochs dropped ({}%)'.format(
            len(bad_epochs), 100 * round(len(bad_epochs) / float(len(self)), 2)))
        self.drop(np.array(bad_epochs), reason='art detection ecg')
        logging.info('{} epochs left after artifact detection'.format(len(self)))

        if run:
            np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(self.sj), self.session], filename='automatic_artdetect.txt'),
                   ['Artifact detection z threshold set to {}. \n{} epochs dropped ({}%)'.
                    format(round(z_thresh, 1), len(bad_epochs), 100 * round(len(bad_epochs) / float(len(self)), 2))], fmt='%.100s')

    def boxSmoothing(self, data, box_car=0.2):
        '''
        doc string boxSmoothing
        '''

        pad = int(round(box_car * self.info['sfreq']))
        if pad % 2 == 0:
            # the kernel should have an odd number of samples
            pad += 1
        kernel = np.ones(pad) / pad
        pad = int(ceil(pad / 2))
        pre_pad = int(min([pad, floor(data.shape[1]) / 2.0]))
        edge_left = data[:, :pre_pad].mean(axis=1)
        edge_right = data[:, -pre_pad:].mean(axis=1)
        data = np.concatenate((np.tile(edge_left.reshape(data.shape[0], 1), pre_pad), data, np.tile(
            edge_right.reshape(data.shape[0], 1), pre_pad)), axis=1)
        data_smooth = sp.signal.convolve2d(
            data, kernel.reshape(1, kernel.shape[0]), 'same')
        data = data_smooth[:, pad:(data_smooth.shape[1] - pad)]
 
        return data

    def link_eye(self, eye_info, missing, vEOG, hEOG):

        # select eog channels
        eog = self.copy().pick_types(eeg=False, eog=True)

        # # rerefence EOG data (vertical and horizontal)
        idx_v = [eog.ch_names.index(vert) for vert in vEOG]
        idx_h = [eog.ch_names.index(hor) for hor in hEOG]

        if len(idx_v) == 2:
            eog._data[idx_v[0]] -= eog._data[idx_v[1]]
        if len(idx_h) == 2:   
            eog._data[idx_h[0]] -= eog._data[idx_h[1]]

        print(
            'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')
        logging.info(
            'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')
        
        # # add rereferenced vEOG and hEOG data to self
        ch_mapping = {vEOG[0]: 'VEOG', hEOG[0]: 'HEOG'}
        eog.rename_channels(ch_mapping)
        eog.drop_channels([vEOG[1], hEOG[1]])
        self.add_channels([eog])

        # CODE FOR EYETRACKER DATA 

        EO = EYE(sfreq = eye_info['sfreq'], viewing_dist = eye_info['viewing_dist'], 
                 screen_res = eye_info['screen_res'], screen_h = eye_info['screen_h'])
        # do binning based on eye-tracking data (if eyetracker data exists)
        eye_bins, window_bins, trial_nrs = EO.eyeBinEEG(self.sj, int(self.session), 
                                int((self.tmin + self.flt_pad + eye_info['tracker_shift'])*1000), int((self.tmax - self.flt_pad + eye_info['tracker_shift'])*1000),
                                drift_correct = (-200,0), start_event = eye_info['start_event'], extension = eye_info['tracker_ext'])
        
        if missing.size > 0:
            eye_bins = np.delete(eye_bins, np.array(missing, dtype = int))

        self.metadata['eye_bins'] = eye_bins

    def detectEye(self, missing, events, nr_events, time_window, threshold=20, windowsize=100, windowstep=10, channel='HEOG', tracker_shift = 0, start_event = '', extension = 'asc', eye_freq = 500, screen_res = (1680, 1050), viewing_dist = 60, screen_h = 29):
        '''
        Marking epochs containing step-like activity that is greater than a given threshold

        Arguments
        - - - - -
        self(object): Epochs object
        missing
        events (array):
        nr_events (int):
        time_window (tuple): start and end time in seconds
        threshold (int): range of amplitude in microVolt
        windowsize (int): total moving window width in ms. So each window's width is half this value
        windowsstep (int): moving window step in ms
        channel (str): name of HEOG channel
        tracker_shift (float): specifies difference in ms between onset trigger and event in eyetracker data
        start_event (str): marking onset of trial in eyetracker data
        extension (str): type of eyetracker file (now supports .asc/ .tsv)
        eye_freq (int): sampling rate of the eyetracker


        Returns
        - - - -

        '''
        
        self.eye_bins = True
        sac_epochs = []

        # CODE FOR HEOG DATA
        idx_ch = self.ch_names.index(channel)

        idx_s, idx_e = tuple([np.argmin(abs(self.times - t))
                              for t in time_window])
        windowstep /= 1000 / self.info['sfreq']
        windowsize /= 1000 / self.info['sfreq']

        for i in range(len(self)):
            up_down = 0
            for j in np.arange(idx_s, idx_e - windowstep, windowstep):

                w1 = np.mean(self._data[i, idx_ch, int(
                    j):int(j + windowsize / 2) - 1])
                w2 = np.mean(self._data[i, idx_ch, int(
                    j + windowsize / 2):int(j + windowsize) - 1])

                if abs(w1 - w2) > threshold:
                    up_down += 1
                if up_down == 2:
                    sac_epochs.append(i)
                    break

        logging.info('Detected {0} epochs ({1:.2f}%) with a saccade based on HEOG'.format(
                len(sac_epochs), len(sac_epochs) / float(len(self)) * 100))

        # CODE FOR EYETRACKER DATA 
        EO = EYE(sfreq = eye_freq, viewing_dist = viewing_dist, 
                 screen_res = screen_res, screen_h = screen_h)
        # do binning based on eye-tracking data (if eyetracker data exists)
        eye_bins, window_bins, trial_nrs = EO.eyeBinEEG(self.sj, int(self.session), 
                                int((self.tmin + self.flt_pad + tracker_shift)*1000), int((self.tmax - self.flt_pad + tracker_shift)*1000),
                                drift_correct = (-200,0), start_event = start_event, extension = extension)

        if eye_bins.size > 0:
            logging.info('Window method detected {} epochs exceeding 0.5 threshold'.format(window_bins.size))

            # remove trials that could not be linked to eeg trigger
            if missing.size > 0:
                eye_bins = np.delete(eye_bins, missing)
                dropped = trial_nrs[missing]
                logging.info('{} trials removed from eye data to align to eeg. Correspond to trial_nr {} in beh'.format(missing.size, dropped))

            # remove trials that have been deleted from eeg 
            eye_bins = np.delete(eye_bins, self.drop_beh)

        # log eyetracker info
        unique_bins = np.array(np.unique(eye_bins), dtype = np.float64)
        for eye_bin in np.unique(unique_bins[~np.isnan(unique_bins)]):
            logging.info('{0:.1f}% of trials exceed {1} degree of visual angle'.format(sum(eye_bins> eye_bin) / eye_bins.size*100, eye_bin))

        # save array of deviation bins    
        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
             self.sj), self.session], filename='eye_bins.txt'), eye_bins)
            

        # # correct for missing data (if eye recording is stopped during experiment)
        # if eye_bins.size > 0 and eye_bins.size < self.nr_events:
        #     # create array of nan values for all epochs (UGLY CODING!!!)
        #     temp = np.empty(self.nr_events) * np.nan
        #     temp[trial_nrs - 1] = eye_bins
        #     eye_bins = temp

        #     temp = (np.empty(self.nr_events) * np.nan)
        #     temp[trial_nrs - 1] = trial_nrs
        #     trial_nrs = temp
        # elif eye_bins.size == 0:
        #     eye_bins = np.empty(self.nr_events + missing.size) * np.nan
        #     trial_nrs = np.arange(self.nr_events + missing.size) + 1



    def applyICA(self, raw, ica_fit, method='extended-infomax', decim=None, fit_params = None, inspect = True):
        '''

        Arguments
        - - - - -
        self(object): Epochs object
        raw (object):
        n_components ():
        method (str):
        decim ():


        Returns
        - - - -

        self

        '''

        # make sure that bad electrodes and 'good' epochs match between both data sets
        ica_fit.info['bads'] = self.info['bads']
        if str(type(ica_fit))[-3] == 's':
            print('fitting data on epochs object')
            to_drop = [i for i, v in enumerate(ica_fit.selection) if v not in self.selection]
            ica_fit.drop(to_drop)

        # initiate ica
        logging.info('Started ICA')
        picks = mne.pick_types(self.info, eeg=True, exclude='bads')
        ica = ICA(n_components=picks.size, method=method, fit_params = fit_params)
        
        # ica is fitted on epoched data
        ica.fit(ica_fit, picks=picks, decim=decim)

        # plot the components
        ica.plot_components(colorbar=True, picks=range(picks.size), show=False)
        plt.savefig(self.FolderTracker(extension=[
                    'preprocessing', 'subject-{}'.format(self.sj), self.session], filename='components.pdf'))
        plt.close()

        # advanced artifact detection
        eog_epochs = create_eog_epochs(raw, baseline=(None, None))
        eog_inds_a, scores = ica.find_bads_eog(eog_epochs)

        ica.plot_scores(scores, exclude=eog_inds_a, show=False)
        plt.savefig(self.FolderTracker(extension=[
                    'preprocessing', 'subject-{}'.format(self.sj), self.session], filename='ica_scores.pdf'))
        plt.close()

        # diagnostic plotting
        ica.plot_sources(self, show_scrollbars=False, show=False)
        if inspect:
            plt.show()
        else:
            plt.savefig(self.FolderTracker(extension=[
                    'preprocessing', 'subject-{}'.format(self.sj), self.session], filename='sources.pdf'))
        plt.close()

        # double check selected component with user input
        time.sleep(5)
        tcflush(sys.stdin, TCIFLUSH)
        print('You are preprocessing subject {}, session {}'.format(self.sj, self.session))
        conf = input(
            'Advanced detection selected component(s) {}. Do you agree (y/n)'.format(eog_inds_a))
        if conf == 'y':
            eog_inds = eog_inds_a
        else:
            eog_inds = []
            nr_comp = input(
                'How many components do you want to select (<10)?')
            for i in range(int(nr_comp)):
                eog_inds.append(
                    int(input('What is component nr {}?'.format(i + 1))))

        for i, cmpt in enumerate(eog_inds):
            ica.plot_properties(self, picks=cmpt, psd_args={
                                'fmax': 35.}, image_args={'sigma': 1.}, show=False)
            plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                self.sj), self.session], filename='property{}.pdf'.format(cmpt)))
            plt.close()


        ica.plot_overlay(raw, exclude=eog_inds, picks=[self.ch_names.index(e) for e in [
                'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8']], show = False)
        plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                        self.sj), self.session], filename='ica-frontal.pdf'))
        plt.close()

        ica.plot_overlay(raw, exclude=eog_inds, picks=[self.ch_names.index(e) for e in [
                'PO7', 'PO8', 'PO3', 'PO4', 'O1', 'O2', 'POz', 'Oz','Iz']], show = False)
        plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                        self.sj), self.session], filename='ica-posterior.pdf'))
        plt.close()
        # remove selected component
        ica.apply(self, exclude=eog_inds)
        logging.info(
            'The following components were removed from raw eeg with ica: {}'.format(eog_inds))

    def save_preprocessed(self, preproc_name, combine_sessions: bool = True):
        
        # save eeg
        self.save(self.FolderTracker(extension=[
                    'processed'], 
                    filename=f'subject-{self.sj}_ses-{self.session}_{preproc_name}-epo.fif'), 
                    split_size='2GB', overwrite = True)

        # check whether individual sessions need to be combined
        if combine_sessions and int(self.session) != 1:
            all_eeg = []
            for i in range(int(self.session)):
                session = i + 1
                all_eeg.append(mne.read_epochs(self.FolderTracker(extension=[
                               'processed'], 
                               filename=f'subject-{self.sj}_ses-{session}_{preproc_name}-epo.fif')))
            
            all_eeg = mne.concatenate_epochs(all_eeg)
            all_eeg.save(self.FolderTracker(extension=[
                         'processed'], filename=f'subject-{self.sj}_all_{preproc_name}-epo.fif'), 
                        split_size='2GB', overwrite = True)
        
    def link_behavior(self, beh: pd.DataFrame, combine_sessions: bool = True):
        """
        Saves linked eeg and behavior data. Preprocessed eeg is saved in the folder processed and behavior is saved
        in a processed subfolder of beh.

        Args:
            beh (pd.DataFrame): behavioral dataframe containing parameters of interest
            combine_sessions (bool, optional):If experiment contains seperate sessions, these are combined into a single datafile that
            is saved  alongside the individual sessions. Defaults to True.
        """

        # update beh dict after removing noise trials
        beh.drop(self.drop_beh, axis = 'index', inplace = True)

        # also include eye binned data
        if hasattr(self, 'eye_bins'):
            eye_bins = np.loadtxt(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session], 
                                filename='eye_bins.txt'))
        else:
            eye_bins = np.nan
        beh['eye_bins'] = pd.Series(eye_bins)

        # save behavior as pickle
        beh_dict = beh.to_dict(orient = 'list')
        with open(self.FolderTracker(extension=['beh', 'processed'],
            filename='subject-{}_ses-{}.pickle'.format(self.sj, self.session)), 'wb') as handle:
            pickle.dump(beh_dict, handle)

        # save eeg
        self.save(self.FolderTracker(extension=[
                    'processed'], filename='subject-{}_ses-{}-epo.fif'.format(self.sj, self.session)), 
                    split_size='2GB', overwrite = True)

        # update preprocessing information
        logging.info('Nr clean trials is {0}'.format(beh.shape[0]))

        if 'condition' in beh.index:
            cnd = beh['condition'].values
            min_cnd, cnd = min([sum(cnd == c) for c in np.unique(cnd)]), np.unique(cnd)[
                np.argmin([sum(cnd == c) for c in np.unique(cnd)])]
            logging.info(
                'Minimum condition ({}) number after cleaning is {}'.format(cnd, min_cnd))
        else:
            logging.info('no condition found in beh file')

        logging.info('EEG data linked to behavior file')

        # check whether individual sessions need to be combined
        if combine_sessions and int(self.session) != 1:

            # combine eeg and beh files of seperate sessions
            all_beh = []
            all_eeg = []
            nr_events = []
            for i in range(int(self.session)):
                with open(self.FolderTracker(extension=['beh', 'processed'],
                	        filename='subject-{}_ses-{}.pickle'.format(self.sj, i + 1)), 'rb') as handle:
                    all_beh.append(pickle.load(handle))
                    
                all_eeg.append(mne.read_epochs(self.FolderTracker(extension=[
                               'processed'], filename='subject-{}_ses-{}-epo.fif'.format(self.sj, i + 1))))

            # do actual combining 
            for key in beh_dict.keys():
                beh_dict.update(
                    {key: np.hstack([beh[key] for beh in all_beh])})

            with open(self.FolderTracker(extension=['beh', 'processed'],
            	filename='subject-{}_all.pickle'.format(self.sj)), 'wb') as handle:
                pickle.dump(beh_dict, handle)

            all_eeg = mne.concatenate_epochs(all_eeg)
            all_eeg.save(self.FolderTracker(extension=[
                         'processed'], filename='subject-{}_all-epo.fif'.format(self.sj)), 
                        split_size='2GB', overwrite = True)

            logging.info('EEG sessions combined')

class ArtefactReject(object):
    """ Multiple (automatic artefact rejection procedures)
    Work in progress
    """

    def __init__(self, z_thresh: float = 4.0, max_bad: int = 5, flt_pad: float = 0, filter_z: bool = True):
        self.flt_pad = flt_pad
        self.filter_z = filter_z
        self.z_thresh = z_thresh
        self.max_bad = max_bad

    def run_blink_ICA(self, fit_inst, raw, ica_inst, sj ,session, method = 'picard', threshold = 0.9, report  = None, report_path = None):

        # step 1: fit the data
        ica = self.fit_ICA(fit_inst, method = 'picard')

        # step 2: select the blink component (assumed to be component 1)
        eog_epochs, eog_inds, eog_scores = self.automated_ica_blink_selection(ica, raw, threshold = threshold)
        ica.exclude = [eog_inds[0]]

        if report is not None:
            report.add_ica(
                ica=ica,
                title='ICA blink cleaning',
                picks=range(15), 
                inst=eog_epochs,
                eog_evoked=eog_epochs.average(),
                eog_scores=eog_scores[0],
                )
            report.save(report_path, overwrite = True)

        #step 2a: manually check selected component
        ica = self.manual_check_ica(ica, sj, session)

        # step 3: apply ica
        ica_inst = self.apply_ICA(ica, ica_inst)

        return ica_inst

    def fit_ICA(self, fit_inst, method = 'picard'):

        if method == 'picard':
            fit_params = dict(fastica_it=5)
        elif method == 'extended_infomax':
            fit_params = dict(extended=True)
        elif method == 'fastica':
            fit_params = None

        #logging.info('started fitting ICA')
        picks = mne.pick_types(fit_inst.info, eeg=True, exclude='bads')
        ica = ICA(n_components=picks.size-1, method=method, fit_params = fit_params, random_state=97)
  
        # do actual fitting
        ica.fit(fit_inst, picks=picks)

        return ica

    def automated_ica_blink_selection(self, ica, raw, threshold = 0.9):

        pick_eog = mne.pick_types(raw.info, meg=False, eeg=False, ecg=False,
                                eog=True)
        ch_names = [raw.info['ch_names'][pick] for pick in pick_eog]

        if pick_eog.any():
            # create blink epochs
            eog_epochs = create_eog_epochs(raw, ch_name=ch_names,
                                        baseline=(None, -0.2),
                                        tmin=-0.5, tmax=0.5)

            eog_inds, eog_scores = ica.find_bads_eog(
                eog_epochs, threshold=threshold)

        else:
            eog_epochs = None
            eog_inds = list()
            print('No EOG channel is present. Cannot automate IC detection '
                'for EOG')

        return eog_epochs, eog_inds, eog_scores

    def manual_check_ica(self, ica, sj, session):
        
        time.sleep(5)
        tcflush(sys.stdin, TCIFLUSH)
        print('You are preprocessing subject {}, session {}'.format(sj, session))
        conf = input(
            'Advanced detection selected component(s) {} (see report). Do you agree (y/n)?'.format(ica.exclude))
        if conf == 'n':
            eog_inds = []
            nr_comp = input(
                'How many components do you want to select (<10)?')
            for i in range(int(nr_comp)):
                eog_inds.append(
                    int(input('What is component nr {}?'.format(i + 1))))

            ica.exclude = eog_inds

        return ica


    def apply_ICA(self, ica, ica_inst):

        # remove selected component
        ica_inst = ica.apply(ica_inst)

        return ica_inst

    def visualize_blinks(self, raw):
    
        eog_epochs = create_eog_epochs(raw, ch_name = ['V_up', 'V_do', 'Fp1','Fpz','Fp2'],  baseline=(-0.5, -0.2))
        eog_epochs.plot_image(combine='mean')
        plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                epochs.sj), epochs.session, 'ica'], filename=f'raw_blinks_combined.pdf'))

        eog_epochs.average().plot_joint()
        plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                epochs.sj), epochs.session, 'ica'], filename=f'raw_blinks_topo.pdf'))


    def update_heat_map(self, channels, ch_idx, tr_idx, upd_value):

        self.heat_map[tr_idx,  ch_idx] = upd_value
        if upd_value == -1: # bad epoch
            for ch in channels[ch_idx]:
                self.not_cleaned_info[ch] += 1

        else: # cleaned_epoch
            for ch in channels[ch_idx]:
                self.cleaned_info[ch] += 1

    def plot_auto_repair(self, channels):


        figs = []
        # plot heat_map
        fig, ax = plt.subplots(1)
        #sns.despine(offset = 10)
        plt.title('Interpolated electrodes per marked epoch \n (blue is bad epochs, red is cleaned epoch)')
        ax.imshow(self.heat_map, aspect  = 'auto', cmap = 'bwr', 
                    interpolation = 'nearest', vmin = -1, vmax = 1)
        ax.set(xlabel='channel', ylabel='bad epochs')
        ax.set_xticks(np.arange(channels.size))
        ax.set_xticklabels(channels, fontsize=6, rotation = 90)
        
        figs.append(fig)

        # plot histogram of bad channels
        clean_df = pd.DataFrame.from_dict(self.cleaned_info, orient = 'index', columns=['count'])
        clean_df = clean_df.loc[(clean_df != 0).any(axis=1)]
        if clean_df.size > 0: # marked at least one bad epoch
            fig, ax = plt.subplots(1)
            sns.despine(offset = 10)
            plt.title('Electrode count per cleaned epoch')          
            ax.barh(np.arange(clean_df.values.size), np.hstack(clean_df.values))
            ax.set_yticks(np.arange(clean_df.values.size)) 
            ax.set_yticklabels(clean_df.index, fontsize=5)
            figs.append(fig)

        noise_df = pd.DataFrame.from_dict(self.not_cleaned_info, orient = 'index', columns=['count'])
        noise_df = noise_df.loc[(noise_df != 0).any(axis=1)]
        if noise_df.size > 0: # marked at least one bad epoch
            fig, ax = plt.subplots(1)
            sns.despine(offset = 10)
            plt.title('Electrode count per noise epoch (i.e., not cleaned')          
            ax.barh(np.arange(noise_df.values.size), np.hstack(noise_df.values))
            ax.set_yticks(np.arange(noise_df.values.size)) 
            ax.set_yticklabels(noise_df.index, fontsize=5)
            figs.append(fig)

        return figs

        # # show bad epochs
        # all_epochs = np.arange(len(epochs))
        # all_epochs = np.delete(all_epochs, bad_epochs)
        # for plot, title in zip([bad_epochs, cleaned_epochs, all_epochs], ['bad_epochs', 'cleaned_epochs', 'all_epochs']):
        #     epochs[plot].average().plot(spatial_colors = True, gfp = True)
        #     plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
        #                 epochs.sj), epochs.session], filename=f'{title}.pdf'))
        #     plt.close()

    @blockPrinting   
    def iterative_interpolation(self, epochs, elecs_z, noise_inf, z_thresh, band_pass):
        
        # keep track of channel info for plotting purposes
        picks = mne.pick_types(epochs.info, eeg=True, exclude= 'bads')
        channels = np.array(epochs.info['ch_names'])[picks]
        self.heat_map = np.zeros((len(noise_inf), channels.size))
        self.cleaned_info = dict.fromkeys(channels, 0)
        self.not_cleaned_info = dict.fromkeys(channels, 0)

        # track bad and cleaned epochs
        bad_epochs, cleaned_epochs = [], []
        for i, event in enumerate(noise_inf):
            bad_epoch = epochs[event[0]]
            # search for bad channels in detected artefact periods
            z = np.concatenate([elecs_z[event[0]][:,slice_[0]] for slice_ in event[1:]], axis = 1)
            # limit interpolation to 'max_bad' noisiest channels 
            ch_idx = np.argsort(z.mean(axis = 1))[-self.max_bad:][::-1]
            interp_chs = channels[ch_idx]

            for c, ch in enumerate(interp_chs):
                # update heat map
                bad_epoch.info['bads'] += [ch]
                bad_epoch.interpolate_bads(exclude = epochs.info['bads'])
                epochs._data[event[0]] = bad_epoch._data
                # repeat preprocesing after interpolation to check whether epoch is now 'clean'
                Z_, _, _, _ = self.preprocess_epochs(epochs[event[0]], band_pass = band_pass)

                if not np.any(abs(Z_) > z_thresh):
                    # epoch no longer marked as bad
                    break

            if ch == interp_chs[-1]:
                self.update_heat_map(channels, ch_idx[:c+1], i, -1)
                bad_epochs.append(event[0])
            else:
                self.update_heat_map(channels, ch_idx[:c+1], i, 1)
                cleaned_epochs.append(event[0])

        return epochs, bad_epochs, cleaned_epochs

    def auto_repair_noise(self, epochs: mne.Epochs, band_pass: list =[110, 140], z_thresh: float = 4.0, report: mne.Report = None):

        # z score data (after hilbert transform)
        Z, elecs_z, z_thresh, times = self.preprocess_epochs(epochs, band_pass = band_pass)

        # mark noise epochs
        noise_inf = self.mark_bads(Z,z_thresh,times)
        
        # clean epochs
        epochs, bad_epochs, cleaned_epochs = self.iterative_interpolation(epochs, elecs_z, noise_inf, z_thresh, band_pass)
        picks = mne.pick_types(epochs.info, eeg=True, exclude= 'bads')
        
        # drop bad epochs     
        epochs.drop(np.array(bad_epochs), reason='Artefact reject')

        if report is not None:
            channels = np.array(epochs.info['ch_names'])[picks]
            report.add_figure(self.plot_auto_repair(channels), title = 'Iterative z cleaning procedure')


        return epochs, z_thresh, report

        print('This interactive window selectively shows epochs marked as bad. You can overwrite automatic artifact detection by clicking on selected epochs')
        bad_eegs = self[bad_epochs]
        idx_bads = bad_eegs.selection
        # display bad eegs with 50mV range
        bad_eegs.plot(
            n_epochs=5, n_channels=data.shape[1], scalings=dict(eeg = 50))
        plt.show()
        plt.close()
        missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eegs.selection],dtype = int)
        logging.info('Manually ignored {} epochs out of {} automatically selected({}%)'.format(
                        missing.size, len(bad_epochs),100 * round(missing.size / float(len(bad_epochs)), 2)))
        bad_epochs = np.delete(bad_epochs, missing)   

    def preprocess_epochs(self, epochs: mne.Epochs, band_pass: list =[110, 140]):       

        # set params
        flt_pad = self.flt_pad
        times = epochs.times
        tmin, tmax = epochs.tmin, epochs.tmax
        sfreq = epochs.info['sfreq']

        # filter data, apply hilbert (limited to 'good' EEG channels) and smooth the data (using defaults)
        X = self.apply_hilbert(epochs, band_pass[0], band_pass[1])
        X = self.box_smoothing(X, sfreq)

        # z score data (while ignoring flt_pad samples) using default settings
        mask = np.logical_and(tmin + self.flt_pad <= times, times <= tmax - self.flt_pad)
        Z, elecs_z, z_thresh = self.z_score_data(X, self.z_thresh, mask, (self.filter_z, sfreq))

        # control for filter padding
        time_idx = epochs.time_as_index([tmin + flt_pad, tmax - flt_pad])
        Z = Z[:, slice(*time_idx)]
        elecs_z = elecs_z[:,:,slice(*time_idx)]
        times = times[slice(*time_idx)]

        return Z, elecs_z, z_thresh, times

    def apply_hilbert(self, epochs: mne.Epochs, lower_band: int = 110, upper_band: int = 140) -> np.array:
        """
        Takes an mne epochs object as input and returns the eeg data after Hilbert transform

        Args:
            epochs (mne.Epochs): mne epochs object before muscle artefact detection
            lower_band (int, optional): Lower limit of the bandpass filter. Defaults to 110.
            upper_band (int, optional): Upper limit of the bandpass filter. Defaults to 140.

        Returns:
            X (np.array): eeg data after applying hilbert transform within the given frequency band
        """

        # exclude channels that are marked as overall bad
        epochs_ = epochs.copy()
        epochs_.pick_types(eeg=True, exclude='bads')

        # filter data and apply Hilbert
        epochs_.filter(lower_band, upper_band, fir_design='firwin', pad='reflect_limited')    
        epochs_.apply_hilbert(envelope=True)

        # get data
        X = epochs_.get_data()
        del epochs_

        return X

    def filt_pad(self, X: np.array, pad_length: int) -> np.array:
        """
        performs padding (using local mean method) on the data, i.e., adds samples before and after the data
        TODO: MOVE to signal processing folder (see https://github.com/fieldtrip/fieldtrip/blob/master/preproc/ft_preproc_padding.m))

        Args:
            X (np.array):  2-dimensional array [nr_elec X nr_time]
            pad_length (int): number of samples that will be padded

        Returns:
            X (np.array): 2-dimensional array with padded data [nr_elec X nr_time]
        """

        # set number of pad samples        
        pre_pad = int(min([pad_length, floor(X.shape[1]) / 2.0]))

        # get local mean on both sides
        edge_left = X[:, :pre_pad].mean(axis=1)
        edge_right = X[:, -pre_pad:].mean(axis=1)

        # pad data
        X = np.concatenate((np.tile(edge_left.reshape(X.shape[0], 1), pre_pad), X, np.tile(
            edge_right.reshape(X.shape[0], 1), pre_pad)), axis=1)

        return X

    def box_smoothing(self, X: np.array, sfreq: float, boxcar: float = 0.2) -> np.array:
        """
        performs boxcar smoothing with specified length. Modified version of ft_preproc_smooth as
        implemented in the FieldTrip toolbox (https://www.fieldtriptoolbox.org)

        Args:
            X (np.array): 3-dimensional array [nr_epoch X nr_elec X nr_time]
            sfreq (float): sampling frequency
            boxcar (float, optional): parameter that determines the length of the filter kernel. Defaults to 0.2 (optimal accrding to fieldtrip documentation).

        Returns:
            np.array: [description]
        """

        # create smoothing kernel
        pad = int(round(boxcar * sfreq))
        # make sure that kernel has an odd number of samples
        if pad % 2 == 0:
            pad += 1
        kernel = np.ones(pad) / pad

        # padding and smoothing functions expect 2-d input (nr_elec X nr_time)
        pad_length = int(ceil(pad / 2))
        for i, x in enumerate(X):
            # pad the data 
            x = self.filt_pad(x, pad_length = pad_length)

            # smooth the data
            x_smooth = sp.signal.convolve2d(
                                    x, kernel.reshape(1, kernel.shape[0]), 'same')
            X[i] = x_smooth[:, pad_length:(x_smooth.shape[1] - pad_length)]
 
        return X

    def z_score_data(self, X: np.array, z_thresh: int = 4, mask: np.array = None, filter_z: tuple = (False, 512)) -> Tuple[np.array, np.array, float]:
        """
        Z scores input data over the second dimension (i.e., electrodes). The z threshold, which is used to mark artefact segments,
        is then calculated using a data driven approach, where the upper limit of the 'bandwith' of obtained z scores is added 
        to the provided default threshold. Also returns z scored data per electrode which can be used during iterative automatic cleaning
        procedure.

        Args:
            X (np.array): 3-dimensional array of [trial repeats by electrodes by time points].
            z_thresh (int, optional): starting z value cut off. Defaults to 4.
            mask (np.array, optional): a boolean array that masks out time points such that they are excluded during z scoring. Note these datapoints are 
            still returned in Z_n and elecs_z (see below).
            filter_z (tuple, optional): prevents false-positive transient peaks by low-pass filtering
            the resulting z-score time series at 4 Hz. Note: Should never be used without filter padding the data. 
            Second argument in the tuple specifies the sampling frequency. Defaults to False, i.e., no filtering.

        Returns:
            Z_n (np.array):  2-dimensional array of normalized z scores across electrodes[trial repeats by time points].
            elecs_z (np.array): 3-dimensional array of z scores data per sample [trial repeats by electrodes by time points].
            z_thresh (float): z_score theshold used to mark segments of muscle artefacts
        """

        # set params
        nr_epoch, nr_elec, nr_time = X.shape

        # get the data and z_score over electrodes
        X = X.swapaxes(0,1).reshape(nr_elec,-1) 
        X_z = zscore(X, axis = 1) 
        if mask is not None:
            mask = np.tile(mask, nr_epoch)
            X_z[:, mask] = zscore(X[:,mask], axis = 1) 

        # reshape to get get epoched data in terms of z scores
        elecs_z = X_z.reshape(nr_elec, nr_epoch,-1).swapaxes(0,1)

        # normalize z_score
        Z_n = X_z.sum(axis = 0)/sqrt(nr_elec) 
        if mask is not None:
            Z_n[mask] = X_z[:,mask].sum(axis = 0)/sqrt(nr_elec) 
        if filter_z[0]:
            Z_n = filter_data(Z_n, filter_z[1], None, 4, pad='reflect_limited') 

        # adjust threshold (data driven) 
        if mask is None:
            mask = np.ones(Z_n.size, dtype = bool)  
        z_thresh += np.median(Z_n[mask]) + abs(Z_n[mask].min() - np.median(Z_n[mask])) 
        
        # transform back into epochs
        Z_n = Z_n.reshape(nr_epoch, -1) 

        return Z_n, elecs_z, z_thresh

    def mark_bads(self, Z: np.array, z_thresh: float, times: np.array) -> list:
        """
        Marks which epochs contain samples that exceed the data driven z threshold (as set by z_score_data).
        Outputs a list with marked epochs. Per marked epoch alongside the index, a slice of the artefact, the start and end point and the duration
        of the artefact are saved

        Args:
            Z (np.array): 2-dimensional array of normalized z scores across electrodes [trial repeats by time points]
            z_thresh (float): z_score theshold used to mark segments of muscle artefacts
            times (np.array): sample time points

        Returns:
            noise_events (list): indices of  marked epochs. Per epoch time information of each artefact is logged
            [idx, (artefact slice, start_time, end_time, duration)]
        """

        # start with empty  list
        noise_events = []
        # loop over each epoch
        for ep, x in enumerate(Z):
            # mask noise samples
            noise_mask = abs(x) > z_thresh
            # get start and end point of continuous noise segments
            starts = np.argwhere((~noise_mask[:-1] & noise_mask[1:]))
            if noise_mask[0]:
                starts = np.insert(starts, 0, 0)
            ends = np.argwhere((noise_mask[:-1] & ~noise_mask[1:])) + 1
            # update noise_events
            if starts.size > 0:
                starts = np.hstack(starts)
                if ends.size == 0:
                    ends = np.array([times.size-1])
                else:
                    ends = np.hstack(ends)
                ep_noise = [ep] + [(slice(s, e), times[s],times[e], abs(times[e] - times[s])) for s,e in zip(starts, ends)]
                noise_events.append(ep_noise)

        return noise_events

    def automatic_artifact_detection(self, z_thresh=4, band_pass=[110, 140], plot=True, inspect=True):
        """ Detect artifacts> modification of FieldTrip's automatic artifact detection procedure 
        (https://www.fieldtriptoolbox.org/tutorial/automatic_artifact_rejection/). 
        Artifacts are detected in three steps:
        1. Filtering the data within specified frequency range
        2. Z-transforming the filtered data across channels and normalize it over channels
        3. Threshold the accumulated z-score

        Counter to fieldtrip the z_threshold is ajusted based on the noise level within the data
        Note: all data included for filter padding is now taken into consideration to calculate z values

        Afer running this function, Epochs contains information about epeochs marked as bad (self.marked_epochs)
        
        Arguments:
            
        Keyword Arguments:
            z_thresh {float|int} -- Value that is added to difference between median 
                    and min value of accumulated z-score to obtain z-threshold
            band_pass {list} --  Low and High frequency cutoff for band_pass filter
            plot {bool} -- If True save detection plots (overview of z scores across epochs, 
                    raw signal of channel with highest z score, z distributions, 
                    raw signal of all electrodes)
            inspect {bool} -- If True gives the opportunity to overwrite selected components
        """
        
        # select data for artifact rejection
        sfreq = self.info['sfreq']
        self_copy = self.copy() 
        self_copy.pick_types(eeg=True, exclude='bads')

        #filter data and apply Hilbert
        self_copy.filter(band_pass[0], band_pass[1], fir_design='firwin', pad='reflect_limited') 
        #self_copy.filter(band_pass[0], band_pass[1], method='iir', iir_params=dict(order=6, ftype='butter'))    
        self_copy.apply_hilbert(envelope=True)

        # get the data and apply box smoothing
        data = self_copy.get_data()
        nr_epochs = data.shape[0]
        for i in range(data.shape[0]):
            data[i] = self.boxSmoothing(data[i])

        # get the data and z_score over electrodes
        data = data.swapaxes(0,1).reshape(data.shape[1],-1) 
        z_score = zscore(data, axis = 1) # check whether axis is correct!!!!!!!!!!!

        # z score per electrode and time point (adjust number of electrodes)
        elecs_z = z_score.reshape(nr_epochs, 64, -1)

        # normalize z_score
        z_score = z_score.sum(axis = 0)/sqrt(data.shape[0]) 
        #z_score = filter_data(z_score, self.info['sfreq'], None, 4, pad='reflect_limited') 

        # adjust threshold (data driven)    
        z_thresh += np.median(z_score) + abs(z_score.min() - np.median(z_score)) 
        
        # transform back into epochs
        z_score = z_score.reshape(nr_epochs, -1) 

        # control for filter padding
        if self.flt_pad > 0:
            idx_ep = self.time_as_index([self.tmin + self.flt_pad, self.tmax - self.flt_pad])
            z_score = z_score[:, slice(*idx_ep)]
            elecs_z

        # mark bad epochs
        bad_epochs = []
        noise_events = []
        cnt = 0
        for ep, X in enumerate(z_score):
            # get start, endpoint and duration in ms per continuous artefact
            noise_mask = X > z_thresh
            starts = np.argwhere((~noise_mask[:-1] & noise_mask[1:]))
            ends = np.argwhere((noise_mask[:-1] & ~noise_mask[1:])) + 1
            ep_noise = [(slice(s[0], e[0]), self.times[s][0],self.times[e][0], abs(self.times[e][0] - self.times[s][0])) for s,e in zip(starts, ends)]
            noise_events.append(ep_noise)
            noise_smp = np.where(X > z_thresh)[0]
            if noise_smp.size > 0:
                bad_epochs.append(ep)

        if inspect:
            print('This interactive window selectively shows epochs marked as bad. You can overwrite automatic artifact detection by clicking on selected epochs')
            bad_eegs = self[bad_epochs]
            idx_bads = bad_eegs.selection
            # display bad eegs with 50mV range
            bad_eegs.plot(
                n_epochs=5, n_channels=data.shape[1], scalings=dict(eeg = 50))
            plt.show()
            plt.close()
            missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eegs.selection],dtype = int)
            logging.info('Manually ignored {} epochs out of {} automatically selected({}%)'.format(
                            missing.size, len(bad_epochs),100 * round(missing.size / float(len(bad_epochs)), 2)))
            bad_epochs = np.delete(bad_epochs, missing)
        
        if plot:
            plt.figure(figsize=(10, 10))
            with sns.axes_style('dark'):

                plt.subplot(111, xlabel='samples', ylabel='z_value',
                            xlim=(0, z_score.size), ylim=(-20, 40))
                plt.plot(np.arange(0, z_score.size), z_score.flatten(), color='b')
                plt.plot(np.arange(0, z_score.size),
                         np.ma.masked_less(z_score.flatten(), z_thresh), color='r')
                plt.axhline(z_thresh, color='r', ls='--')

                plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                    self.sj), self.session], filename='automatic_artdetect.pdf'))
                plt.close()

        # drop bad epochs and save list of dropped epochs
        self.drop_beh = bad_epochs
        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
            self.sj), self.session], filename='noise_epochs.txt'), bad_epochs)
        print('{} epochs dropped ({}%)'.format(len(bad_epochs),
                                               100 * round(len(bad_epochs) / float(len(self)), 2)))
        logging.info('{} epochs dropped ({}%)'.format(
            len(bad_epochs), 100 * round(len(bad_epochs) / float(len(self)), 2)))
        self.drop(np.array(bad_epochs), reason='art detection ecg')
        logging.info('{} epochs left after artifact detection'.format(len(self)))

        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(self.sj), self.session], filename='automatic_artdetect.txt'),
                   ['Artifact detection z threshold set to {}. \n{} epochs dropped ({}%)'.
                    format(round(z_thresh, 1), len(bad_epochs), 100 * round(len(bad_epochs) / float(len(self)), 2))], fmt='%.100s')

        return z_thresh


if __name__ == '__main__':
    print('Please run preprocessing via a project script')
