"""
analyze EEG data

Created by Dirk van Moorselaar on 10-03-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""

import mne
import os
import logging
import itertools
import h5py
import pickle
import copy
import glob
import sys
import time
#import matplotlib          # run these lines only when running sript via ssh connection
#matplotlib.use('agg')

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg') # run next two lines to interactively scroll through plots
import matplotlib
import seaborn as sns
from matplotlib import cm

from termios import tcflush, TCIFLUSH
from EYE import *
from math import sqrt
from IPython import embed
from FolderStructure import FolderStructure
from scipy.stats.stats import pearsonr
from mne.viz.epochs import plot_epochs_image
from mne.filter import filter_data
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from math import ceil, floor


class RawBDF(mne.io.edf.edf.RawEDF, FolderStructure):
    '''
    Child originating from MNE built-in RawEDF, such that new methods can be added to this built in class
    '''

    def __init__(self, input_fname, montage=None, eog=None, stim_channel=-1,
                 annot=None, annotmap=None, exclude=(), preload=True, verbose=None):

        super(RawBDF, self).__init__(input_fname=input_fname, montage=montage, eog=eog,
                                     stim_channel=stim_channel, annot=annot, annotmap=annotmap,
                                     preload=preload, verbose=verbose)

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

    def reReference(self, ref_channels=['EXG5', 'EXG6'], vEOG=['EXG1', 'EXG2'], hEOG=['EXG3', 'EXG4'], changevoltage=True):
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
        print('EEG data was rereferenced to channels {}'.format(ref_channels))
        logging.info(
            'EEG data was rereferenced to channels {}'.format(ref_channels))

        # rerefence EOG data (vertical and horizontal)
        idx_v = [self.ch_names.index(vert) for vert in vEOG]
        idx_h = [self.ch_names.index(hor) for hor in hEOG]

        self._data[idx_v[0]] -= self._data[idx_v[1]]
        self._data[idx_h[0]] -= self._data[idx_h[1]]
        ch_mapping = {vEOG[0]: 'VEOG', hEOG[0]: 'HEOG'}
        self.rename_channels(ch_mapping)
        print(
            'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')
        logging.info(
            'EOG data (VEOG, HEOG) rereferenced with subtraction and renamed EOG channels')

        # drop ref chans
        to_remove = ref_channels + [vEOG[1], hEOG[1]] + ['EXG7', 'EXG8']
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

        montage = mne.channels.read_montage(kind=montage)

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
        self.set_montage(montage)

        print('Channels renamed to 10-20 system, and montage added')
        logging.info('Channels renamed to 10-20 system, and montage added')

    def eventSelection(self, trigger, binary=False, consecutive=False, min_duration=0.003):
        '''
        Returns array of events necessary for epoching.

        Arguments
        - - - - -
        raw (object): raw mne eeg object
        binary (boolean): specifies whether or not triggers need to be adjusted for binary codes (subtracts 3840)

        Returns
        - - - -
        events(array): numpy array with trigger events (first column contains the event time in samples and the third column contains the event id)
        '''

        if binary:
            self._data[-1, :] -= 3840

        events = mne.find_events(self, stim_channel='STI 014', consecutive=consecutive, min_duration=min_duration)    

        # Check for consecutive 
        if not consecutive:
            spoke_idx = []
            for i in range(events[:-1,2].size):
                if events[i,2] == events[i + 1,2] and events[i,2] == trigger:
                    spoke_idx.append(i)

            events = np.delete(events,spoke_idx,0)    
            logging.info('{} spoke events removed from event file'.format(len(spoke_idx)))    

        return events


    def matchBeh(self, sj, session, events, trigger, headers, max_trigger = 66):
        '''

        '''

        # read in data file
        beh_file = self.FolderTracker(extension=[
                    'beh', 'raw'], filename='subject-{}_ses_{}.csv'.format(sj, session))

        # get triggers logged in beh file
        beh = pd.read_csv(beh_file)
        beh = beh[headers]
        beh = beh[beh['practice'] == 'no']
        beh = beh.drop(['practice'], axis=1)
        
        # get triggers bdf file
        idx_trigger = np.where(events[:,2] == trigger)[0] + 1
        trigger_bdf = events[idx_trigger,2] 

        # log number of unique triggers
        unique = np.unique(trigger_bdf)
        logging.info('{} detected unique triggers (min = {}, max = {})'.
                        format(unique.size, unique.min(), unique.max()))

        # make sure trigger info between beh and bdf data matches
        missing_trials = []
        while beh.shape[0] != trigger_bdf.size:

            # remove spoke triggers, update events
            if missing_trials == []:
                logging.info('removed {} spoke triggers from bdf'.format(sum(trigger_bdf > max_trigger)))
                # update events
                to_remove = idx_trigger[np.where(trigger_bdf > max_trigger)[0]] -1
                events = np.delete(events, to_remove, axis = 0)
                trigger_bdf = trigger_bdf[trigger_bdf < max_trigger]

            trigger_beh = beh['trigger'].values
            if trigger_beh.size > trigger_bdf.size:
                for i, trig in enumerate(trigger_bdf):
                    if trig != trigger_beh[i]:
                        beh.drop(beh.index[i], inplace=True)  
                        miss = beh['nr_trials'].iloc[i]
                        missing_trials.append(miss)
                        logging.info('Removed trial {} from beh file,because no matching trigger exists in bdf file'.format(miss))
                        break
           
        missing = np.array(missing_trials)        
        # log number of matches between beh and bdf       
        logging.info('{} matches between beh and epoched data'.
            format(sum(beh['trigger'].values == trigger_bdf)))           

        return beh, missing, events

class Epochs(mne.Epochs, FolderStructure):
    '''
    Child originating from MNE built-in Epochs, such that new methods can be added to this built in class
    '''

    def __init__(self, sj, session, raw, events, event_id, tmin, tmax, flt_pad=True, baseline=(None, None), picks=None, preload=True,
                 reject=None, flat=None, proj=False, decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                 on_missing='error', reject_by_annotation=False, verbose=None):

        # check whether a preprocessed folder for the current subject exists,
        # if not make one
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

    def selectBadChannels(self, channel_plots=True, inspect=True, n_epochs=10, n_channels=32):
        '''

        '''

        logging.info('Start selection of bad channels')

        # select channels to display
        picks = mne.pick_types(self.info, eeg=True, exclude='bads')

        # plot epoched data
        if channel_plots:

            for ch in picks:
                # plot evoked responses across channels
                try:  # handle bug in mne for plotting undefined x, y coordinates
                    plot_epochs_image(self, ch, show=False)
                    plt.savefig(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session, 'channel_erps'], filename='ch_{}'.format(ch)))
                    plt.close()
                except:
                    plt.savefig(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session,'channel_erps'], filename='ch_{}'.format(ch)))
                    plt.close()

            # plot power spectra topoplot to detect any clear bad electrodes
            self.plot_psd_topomap(bands=[(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (
                12, 30, 'Beta'), (30, 45, 'Gamma'), (45, 100, 'High')], show=False)
            plt.savefig(self.FolderTracker(extension=[
                        'preprocessing', 'subject-{}'.format(self.sj), self.session], filename='psd_topomap.pdf'))
            plt.close()


        if inspect:

            self.plot(block=True, n_epochs=n_epochs,
                      n_channels=n_channels, picks=picks, scalings='auto')

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

    def artifactDetection(self, z_cutoff=4, band_pass=[110, 140], plot=True, inspect=True):
        '''
        Detect artifacts based on FieldTrip's automatic artifact detection. 
        Artifacts are detected in three steps:
                1. Filtering the data
                2. Z-transforming the filtered data and normalize it over channels
                3. Threshold the accumulated z-score

        Arguments
        - - - - -
        self(object): Epochs object
        z_cutoff (int): Value that is added to difference between median 
                        and min value of accumulated z-score to obtain z-threshold
        band_pass (list): Low and High frequency cutoff for band_pass filter
        plot (bool): If True save detection plots (overview of z scores across epochs, 
                    raw signal of channel with highest z score, z distributions, 
                    raw signal of all electrodes)


        Returns
        - - - -

        self.marked_epochs (data): Adds a list of marked epochs to Epoch object
        '''

        # select channels for artifact detection
        picks = mne.pick_types(self.info, eeg=True, exclude='bads')
        nr_channels = picks.size
        sfreq = self.info['sfreq']

        # control for filter padding
        if self.flt_pad > 0:
            idx_ep = self.time_as_index([self.tmin + self.flt_pad, self.tmax - self.flt_pad])

        print('Started artifact detection')
        logging.info('Started artifact detection')
        ep_data = []

        # STEP 1: filter each epoch data, apply hilbert transform and boxsmooth
        # the resulting data before removing filter padds
        for epoch, X in enumerate(self):

            # CHECK IF THIS IS CORRECT ORDER IN FIELDTRIP CODE / ALSO CHANGE
            # FILTER TO MNE 0.14 STANDARD
            X = filter_data(X[picks, :], sfreq, band_pass[0], band_pass[
                            1], method='iir', iir_params=dict(order=9, ftype='butter'))
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

        bad_epochs = []
        for ep, X in enumerate(z_accumel_ep):
            # ADD LINES SPECIFYING THE MIN LENGTH OF A NOISE SEGMENT
            if (X > z_thresh).sum() > 0:
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

        if inspect:

            # ADD LINES ALLOWING TO COUNTERACT AUTOMATIC DETECTION
            self[bad_epochs].plot(
                n_epochs=10, n_channels=picks.size, picks=picks, scalings='auto')
            plt.show()
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

    def detectEye(self, missing, time_window, threshold=30, windowsize=50, windowstep=25, channel='HEOG', tracker = True):
        '''
        Marking epochs containing step-like activity that is greater than a given threshold

        Arguments
        - - - - -
        self(object): Epochs object
        missing
        time_window (tuple): start and end time in seconds
        threshold (int): range of amplitude in microVolt
        windowsize (int): total moving window width in ms. So each window's width is half this value
        windowsstep (int): moving window step in ms
        channel (str): name of HEOG channel
        tracker (boolean): is tracker data reliable or not


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
        EO = EYE()
        # read in epochs removed via artifact detection
        noise_epochs = np.loadtxt(self.FolderTracker(extension=[
                                'preprocessing', 'subject-{}'.format(self.sj), self.session], 
                                filename='noise_epochs.txt'))

        # do binning based on eye-tracking data
        eye_bins, trial_nrs = EO.eyeBinEEG(self.sj, int(self.session), 
                            int((self.tmin + self.flt_pad)*1000), int((self.tmax - self.flt_pad)*1000),
                            drift_correct = (-300,0))

        # correct for missing data (if eye recording is stopped during experiment)
        if eye_bins.size > 0 and eye_bins.size < self.nr_events:
            # create array of nan values for all epochs (UGLY CODING!!!)
            temp = np.empty(self.nr_events) * np.nan
            temp[trial_nrs - 1] = eye_bins
            eye_bins = temp

            temp = (np.empty(self.nr_events) * np.nan)
            temp[trial_nrs - 1] = trial_nrs
            trial_nrs = temp
        elif eye_bins.size == 0 or tracker == False:
            eye_bins = np.empty(self.nr_events + missing.size) * np.nan
            trial_nrs = np.arange(self.nr_events + missing.size) + 1

        # remove trials that are not present in bdf file, if any
        miss_mask = np.in1d(trial_nrs, missing, invert = True)
        eye_bins = eye_bins[miss_mask]      

        # remove trials that have been deleted from eeg 
        eye_bins = np.delete(eye_bins, noise_epochs)
        logging.info('Detected {0} bins ({1} with data) based on tracker ({2:.2f} > 1 degree)'.
                    format(eye_bins.size, (~np.isnan(eye_bins)).sum(), sum(eye_bins > 1) / float(len(self)) * 100))

        # save array of deviation bins    
        np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
            self.sj), self.session], filename='eye_bins.txt'), eye_bins)


    def applyICA(self, raw, method='extended-infomax', decim=3, inspect=False):
        '''

        ICA is run on continuous raw data (filtered with 1 Hz)

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

        # initiate ica
        logging.info('Started ICA')
        picks = mne.pick_types(self.info, eeg=True, exclude='bads')
        ica = ICA(n_components=picks.size, method=method)
        ica.fit(raw, picks=picks, decim=decim)

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

        ica.plot_sources(eog_epochs.average(), exclude=eog_inds_a, show=False)
        plt.savefig(self.FolderTracker(extension=[
                    'preprocessing', 'subject-{}'.format(self.sj), self.session], filename='sources.pdf'))
        plt.close()

        for i, cmpt in enumerate(eog_inds_a):
            ica.plot_properties(eog_epochs, picks=eog_inds_a[i], psd_args={
                                'fmax': 35.}, image_args={'sigma': 1.}, show=False)
            plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                self.sj), self.session], filename='property{}.pdf'.format(cmpt)))
            plt.close()

        # double check selected component with user input
        time.sleep(5)
        tcflush(sys.stdin, TCIFLUSH)
        print('You are preprocessing subject {}, session {}'.format(self.sj, self.session))
        conf = raw_input(
            'Advanced detection selected component(s) {}. Do you agree (y/n)'.format(eog_inds_a))
        if conf == 'y':
            eog_inds = eog_inds_a
        else:
            eog_inds = []
            nr_comp = raw_input(
                'How many components do you want to select (<10)?')
            for i in range(int(nr_comp)):
                eog_inds.append(
                    int(raw_input('What is component nr {}?'.format(i + 1))))
                if eog_inds[-1] not in eog_inds_a:
                    ica.plot_properties(eog_epochs, picks=eog_inds[-1], psd_args={
                                        'fmax': 35.}, image_args={'sigma': 1.}, show=False)
                    plt.savefig(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
                        self.sj), self.session], filename='property{}.pdf'.format(eog_inds[-1])))
                    plt.close()

        # remove selected component
        ica.apply(self, exclude=eog_inds)
        logging.info(
            'The following components were removed from raw eeg with ica: {}'.format(eog_inds))

        if inspect:
            front_electr = [self.ch_names.index(e) for e in [
                'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'VEOG', 'HEOG']]
            self.plot(block=True, n_epochs=12, n_channels=len(
                front_electr), picks=front_electr, scalings='auto')

    def linkBeh(self, beh, events, trigger, combine_sessions=True):
        '''

        '''

        # check which trials were excluded
        events[events[:, 2] == trigger, 2] = range(beh.shape[0])
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
                  'processed'], filename='subject-{}_ses-{}-epo.fif'.format(self.sj, self.session)), split_size='2GB')

        # update preprocessing information
        logging.info('Nr clean trials is {0} ({1:.0f}%)'.format(
            sel_tr.size, float(sel_tr.size) / beh.shape[0] * 100))
        cnd = beh['condition'].values
        min_cnd, cnd = min([sum(cnd == c) for c in np.unique(cnd)]), np.unique(cnd)[
            np.argmin([sum(cnd == c) for c in np.unique(cnd)])]

        logging.info(
            'Minimum condition ({}) number after cleaning is {}'.format(cnd, min_cnd))

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
                         'processed'], filename='subject-{}_all-epo.fif'.format(self.sj)), split_size='2GB')

            logging.info('EEG sessions combined')


if __name__ == '__main__':

    # Specify project parameters
    project_folder = '/home/dvmoors1/big_brother/Dist_suppression'
    os.chdir(project_folder)
    montage = mne.channels.read_montage(kind='biosemi64')
    subject = 16
    session = 2
    tracker = True
    data_file = 'subject_{}_session_{}_'.format(subject, session)
    eeg_runs = [1, 2]
    #eog = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
    #ref = ['EXG5', 'EXG6']
    eog =  ['V_up','V_do','H_r','H_l']
    ref =  ['Ref_r','Ref_l']
    trigger = 3
    t_min = -0.3
    t_max = 0.8
    flt_pad = 0.5

    replace = {'15': {'session_1': {'B1': 'EXG7'}
                      }}

    # start logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='processed/info/preprocess_sj{}_ses{}.log'.format(
                            subject, session),
                        filemode='w')

    # READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
    EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', data_file + '{}.bdf'.format(run)),
                                       montage=None, preload=True, eog=eog) for run in eeg_runs])
    EEG.replaceChannel(subject, session, replace)
    EEG.reReference(ref_channels=ref, vEOG=eog[
                    :2], hEOG=eog[2:], changevoltage=True)
    EEG.renameChannelAB(montage='biosemi64')

    # FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
    EEGica = EEG.filter(h_freq=None, l_freq=1,
                       fir_design='firwin', skip_by_annotation='edge')
    EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
               skip_by_annotation='edge')

    # MATCH BEHAVIOR FILE
    events = EEG.eventSelection(trigger, binary=True, min_duration=0)
    beh, missing, events = EEG.matchBeh(subject, session,events, trigger, headers=[
            'target_loc', 'dist_loc', 'condition', 'trigger', 
            'practice','nr_trials'])

    # EPOCH DATA
    epochs = Epochs(subject, session, EEG, events, event_id=trigger,
            tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

    # ARTIFACT DETECTION
    epochs.selectBadChannels(channel_plots = True, inspect=True)
    epochs.artifactDetection(inspect=False)

    # ICA
    epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = True)

    # EYE MOVEMENTS
    epochs.detectEye(missing, time_window=(t_min, t_max), tracker = tracker)

    # INTERPOLATE BADS
    epochs.interpolate_bads(reset_bads=True, mode='accurate')

    # LINK BEHAVIOR
    epochs.linkBeh(beh, events, trigger)
