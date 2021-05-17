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
from autoreject import Ransac


class RawBDF(mne.io.edf.edf.RawEDF, FolderStructure):
    '''
    Child originating from MNE built-in RawEDF, such that new methods can be added to this built in class
    '''

    def __init__(self, input_fname, montage=None, eog=None, stim_channel=-1,
                exclude=(), preload=True, verbose=None):

        super(RawBDF, self).__init__(input_fname=input_fname, eog=eog,
                                     stim_channel=stim_channel, preload=preload, verbose=verbose)

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
        
        # rerefence EOG data (vertical and horizontal)
        idx_v = [eog.ch_names.index(vert) for vert in vEOG]
        idx_h = [eog.ch_names.index(hor) for hor in hEOG]

        if len(idx_v) == 2:
            eog._data[idx_v[0]] -= self._data[idx_v[1]]
        if len(idx_h) == 2:   
            eog._data[idx_h[0]] -= self._data[idx_h[1]]

        print(
            'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')
        logging.info(
            'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')
        
        # add rereferenced vEOG and hEOG data to self
        ch_mapping = {vEOG[0]: 'VEOG', hEOG[0]: 'HEOG'}
        eog.rename_channels(ch_mapping)
        eog.drop_channels([vEOG[1], hEOG[1]])
        self.add_channels([eog])

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

    def matchBeh(self, sj, session, events, event_id, headers):
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
        beh_triggers = beh['trigger'].values  

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
            format(sum(beh['trigger'].values == bdf_triggers), bdf_triggers.size))           

        return beh, missing

class Epochs(mne.Epochs, FolderStructure):
    '''
    Child originating from MNE built-in Epochs, such that new methods can be added to this built in class
    '''

    def __init__(self, sj, session, raw, events, event_id, tmin, tmax, flt_pad=True, baseline=(None, None), picks=None, preload=True,
                 reject=None, flat=None, proj=False, decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                 on_missing='error', reject_by_annotation=False, verbose=None):

        # check whether a preprocessed folder for the current subject exists,
        # if not make onen
        self.sj = sj
        self.session = str(session)
        self.flt_pad = flt_pad
        if not os.path.isdir(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(sj), self.session])):
            os.makedirs(self.FolderTracker(
                extension=['preprocessing', 'subject-{}'.format(sj), self.session, 'channel_erps']))

        tmin, tmax = tmin - flt_pad, tmax + flt_pad
        
        super(Epochs, self).__init__(raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                                     baseline=baseline, picks=picks, preload=preload, reject=reject,
                                     flat=flat, proj=proj, decim=decim, reject_tmin=reject_tmin,
                                     reject_tmax=reject_tmax, detrend=detrend, on_missing=on_missing,
                                     reject_by_annotation=reject_by_annotation, verbose=verbose)

        # save number of detected events
        self.nr_events = len(self)
        logging.info('{} epochs created'.format(len(self)))

    def applyRansac(self):
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
        self.info['bads'] = ransac.bad_chs_


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

    def artifactDetection(self, z_thresh=4, band_pass=[110, 140], plot=True, inspect=True):
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
            missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eegs.selection])
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
            missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eegs.selection])
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
        
        # read in epochs removed via artifact detection
        trigger = np.unique(self.events[:,2]) 
        events[[i for i, idx in enumerate(events[:,2]) if idx in trigger],2] = list(range(nr_events))
        sel_tr = events[self.selection, 2]
        noise_epochs = np.array(list(set(list(range(nr_events))).difference(sel_tr)))  

        # do binning based on eye-tracking data
        # check whether eye tracker exists
        eye_bins, window_bins, trial_nrs = EO.eyeBinEEG(self.sj, int(self.session), 
                                int((self.tmin + self.flt_pad + tracker_shift)*1000), int((self.tmax - self.flt_pad + tracker_shift)*1000),
                                drift_correct = (-200,0), start_event = start_event, extension = extension)

        logging.info('Window method detected {} epochs exceeding 0.5 threshold'.format(window_bins.size))

        # correct for missing data (if eye recording is stopped during experiment)
        if eye_bins.size > 0 and eye_bins.size < self.nr_events:
            # create array of nan values for all epochs (UGLY CODING!!!)
            temp = np.empty(self.nr_events) * np.nan
            temp[trial_nrs - 1] = eye_bins
            eye_bins = temp

            temp = (np.empty(self.nr_events) * np.nan)
            temp[trial_nrs - 1] = trial_nrs
            trial_nrs = temp
        elif eye_bins.size == 0:
            eye_bins = np.empty(self.nr_events + missing.size) * np.nan
            trial_nrs = np.arange(self.nr_events + missing.size) + 1

        # remove trials that are not present in bdf file, if any
        miss_mask = np.in1d(trial_nrs, missing, invert = True)
        eye_bins = eye_bins[miss_mask]      

        # remove trials that have been deleted from eeg 
        eye_bins = np.delete(eye_bins, noise_epochs)
        # start logging eye_tracker info
        unique_bins = np.array(np.unique(eye_bins), dtype = np.float64)
        for eye_bin in np.unique(unique_bins[~np.isnan(unique_bins)]):
            logging.info('{0:.1f}% of trials exceed {1} degree of visual angle'.format(sum(eye_bins> eye_bin) / eye_bins.size*100, eye_bin))

        # save array of deviation bins    
        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
            self.sj), self.session], filename='eye_bins.txt'), eye_bins)


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

        
    def linkBeh(self, beh, events, trigger, combine_sessions=True):
        '''

        '''

        # check which trials were excluded
        #events[[i for i, idx in enumerate(events[:,2]) if idx in trigger],2] = range(beh.shape[0])
        sel_tr = events[self.selection, 2]

        # create behavior dictionary (only include clean trials after preprocessing)
        # also include eye binned data
        eye_bins = np.loadtxt(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session], 
                                filename='eye_bins.txt'))

        beh_dict = {'clean_idx': sel_tr, 'eye_bins': eye_bins}
        for header in beh.columns:
            beh_dict.update({header: beh[header].values[sel_tr]})

        # save behavior
        with open(self.FolderTracker(extension=['beh', 'processed'],
            filename='subject-{}_ses-{}.pickle'.format(self.sj, self.session)), 'wb') as handle:
            pickle.dump(beh_dict, handle)

        # save eeg
        self.save(self.FolderTracker(extension=[
                    'processed'], filename='subject-{}_ses-{}-epo.fif'.format(self.sj, self.session)), 
                 split_size='2GB', overwrite = True)

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

        if combine_sessions and int(self.session) != 1:

            # combine eeg and beh files of seperate sessions
            all_beh = []
            all_eeg = []
            nr_events = []
            for i in range(int(self.session)):
                with open(self.FolderTracker(extension=['beh', 'processed'],
                	filename='subject-{}_ses-{}.pickle'.format(self.sj, i + 1)), 'rb') as handle:
                    all_beh.append(pickle.load(handle))
                    
                    # if i > 0:
                    #     all_beh[i]['clean_idx'] += sum(nr_events)
                    # nr_events.append(all_beh[i]['condition'].size)

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

if __name__ == '__main__':
    print('Please run preprocessing via a project script')
