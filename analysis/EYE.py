"""
Eye Tracking Data Analysis for the DvM Toolbox.

This module provides comprehensive tools for eye tracking data 
processing and analysis. It handles data from multiple eye tracker 
formats, performs drift correction, detects saccades and fixation 
breaks, and calculates visual angle deviations. Designed for 
integration with EEG/behavioral experiments.

The module implements a complete eye tracking analysis workflow 
including data loading, trial synchronization with behavioral files, 
blink interpolation, noise detection, drift correction, and saccade 
detection using adaptive thresholding algorithms. All classes integrate 
with the FolderStructure system for organized file management.

Main Classes
------------
EYE : Eye tracking data analysis class
    Handles eye tracker data loading, preprocessing, drift correction,
    and visual angle calculations. Supports EyeLink (.asc) and EyeTribe
    (.tsv) formats.

SaccadeDetector : Adaptive saccade and glissade detection
    Implements the Nyström & Holmqvist (2010) adaptive algorithm for
    detecting saccades and post-saccadic glissades using velocity-based
    thresholding with noise-dependent adaptation.

Key Features
------------
- Multi-format support: EyeLink ASCII (.asc) and EyeTribe (.tsv) files
- Automatic trial synchronization with behavioral data
- Blink detection and linear interpolation with configurable padding
- Drift correction with saccade-free fixation period validation
- Visual angle calculations from pixel coordinates
- Adaptive saccade detection with noise-dependent thresholds
- Glissade detection (post-saccadic drift movements)
- Trial binning by maximum sustained gaze deviation
- Integration with MNE Epochs for combined EEG-eye analysis
- Flexible trial exclusion based on fixation quality
- EOG fallback for missing eye tracker data

Typical Workflow
----------------
1. Initialize EYE object with screen geometry parameters
2. Load eye tracking and behavioral data via get_eye_data()
3. Extract gaze coordinates aligned to events via get_xy()
4. Apply drift correction and noise detection via set_xy()
5. Calculate visual angle deviations via angles_from_xy()
6. Detect saccades using SaccadeDetector class
7. Exclude trials with fixation breaks via exclude_eye()

See Also
--------
eeg_analyses.EEG.Epochs : EEG epoching with eye tracker integration
support.FolderStructure : File organization system

References
----------
Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
saccade, and glissade detection in eyetracking data. Behavior Research
Methods, 42(1), 188-204.

Created by Dirk van Moorselaar on 24-08-2017.
Copyright (c) 2017 DvM. All rights reserved.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

from math import degrees, atan2, floor, ceil
from typing import Optional, Tuple, Union
from numpy.lib.npyio import NpzFile
from scipy.signal import savgol_filter
from support.preprocessing_utils import get_time_slice, format_subject_id
from support.FolderStructure import *
from pygazeanalyser.edfreader import *
from pygazeanalyser.eyetribereader import *

class EYE(FolderStructure):
    """
    Eye tracking data analysis and preprocessing class.
    
    This class provides comprehensive tools for loading, processing, and
    analyzing eye tracking data. It handles data synchronization with
    behavioral files, drift correction, blink interpolation, saccade
    detection, and visual angle calculations. Supports EyeLink (.asc) 
    and EyeTribe (.tsv) file formats.
    
    Inherits from FolderStructure for standardized file organization and
    path management capabilities.

    Parameters
    ----------
    See __init__ method for initialization parameters.

    Attributes
    ----------
    view_dist : float
        Viewing distance from observer to screen in cm.
    scr_res : tuple of (int, int)
        Screen resolution (width, height) in pixels.
    scr_h : float
        Physical screen height in cm.
    sfreq : float
        Eye tracker sampling frequency in Hz.
        
    Methods
    -------
    get_eye_data(sj, eye_files, beh_files, start, trial_info, stop)
        Load and synchronize eye tracker and behavioral data files.
    interp_trial(trial)
        Interpolate blinks and missing data in a single trial.
    get_xy(eye, start, end, start_event, interpolate_blinks)
        Extract aligned gaze coordinates from trial data.
    set_xy(x, y, times, drift_correct)
        Apply noise detection and drift correction to gaze coordinates.
    angles_from_xy(x, y, times, drift_correct)
        Calculate gaze deviation angles from fixation.
    link_eye_to_eeg(eye_file, beh_file, start_trial, stop_trial, 
                    window_oi, trigger_msg, drift_correct)
        Load and process eye data aligned to EEG triggers.
    create_angle_bins(x, y, start, stop, step, min_segment)
        Calculate angles and bin trials by maximum deviation.
    calculate_angle(x, y, xc, yc)
        Calculate visual angle deviation from center point.
    degrees_to_pixels(h, d, r)
        Static method to calculate degrees per pixel.

    See Also
    --------
    SaccadeDetector : Adaptive saccade and glissade detection algorithm
    exclude_eye : Trial exclusion based on eye movements
    bin_tracker_angles : Classify trials by fixation breaks

    Examples
    --------
    >>> # Initialize with default 1080p setup
    >>> eye = EYE()
    
    >>> # Custom setup for 4K monitor
    >>> eye = EYE(viewing_dist=90, screen_res=(3840, 2160),
    ...           screen_h=35, sfreq=1000)
    
    >>> # Load and process eye tracking data
    >>> eye_data, beh, info = eye.get_eye_data(sj=1)
    >>> x, y, times = eye.get_xy(eye_data, -200, 1000, 'stimulus_on')
    
    >>> # Apply drift correction and calculate angles
    >>> x_clean, y_clean = eye.set_xy(x, y, times, 
    ...                               drift_correct=(-200, 0))
    >>> angles = eye.angles_from_xy(x_clean, y_clean, times)
    
    >>> # Complete pipeline with EEG alignment
    >>> x, y, times, bins, angles, info = eye.link_eye_to_eeg(
    ...     eye_file='sub_01.asc',
    ...     beh_file='sub_01.csv',
    ...     start_trial='start_trial',
    ...     stop_trial=None,
    ...     window_oi=(-200, 1000),
    ...     trigger_msg='stimulus_on',
    ...     drift_correct=(-200, 0)
    ... )
    """

    def __init__(
        self,
        viewing_dist: float = 60,
        screen_res: Tuple[int, int] = (1920, 1080),
        screen_h: float = 30,
        sfreq: float = 500
    ) -> None:
        """
        Initialize EYE analysis object with screen and tracker 
        parameters.
        
        Sets up screen geometry and sampling frequency for subsequent
        eye tracking data processing and visual angle calculations.
        All parameters are stored as instance attributes for use in
        pixel-to-degree conversions.
        
        Parameters
        ----------
        viewing_dist : float, default=60
            Distance from participant's eyes to screen in cm.
        screen_res : tuple of (int, int), default=(1920, 1080)
            Screen resolution as (width, height) in pixels.
        screen_h : float, default=30
            Physical screen height in cm.
        sfreq : float, default=500
            Eye tracker sampling frequency in Hz.
            
        Returns
        -------
        None
            
        Examples
        --------
        >>> # Default 1080p monitor at 60cm
        >>> eye = EYE()
        >>> 
        >>> # Custom setup: 4K monitor at 90cm
        >>> eye = EYE(viewing_dist=90, screen_res=(3840, 2160),
        ...           screen_h=35, sfreq=1000)
        
        Notes
        -----
        The screen geometry parameters (viewing_dist, screen_res, 
        screen_h) are used by calculate_angle() to convert pixel 
        coordinates to visual degrees. Ensure these match your actual
        experimental setup for accurate angle calculations.
        """

        self.view_dist = viewing_dist
        self.scr_res = screen_res
        self.scr_h = screen_h
        self.sfreq = sfreq

    def get_eye_data(
        self,
        sj: Union[int, str],
        eye_files: Union[str, list] = 'all',
        beh_files: Union[str, list] = 'all',
        start: str = 'start_trial',
        trial_info: Optional[list] = None,
        stop: Optional[str] = None
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Load and synchronize eye tracker and behavioral data files.
        
        Reads eye tracking data (.asc or .tsv) and behavioral logs 
        (.csv), aligns them by trial, handles missing trials, and 
        removes practice trials. Supports multiple session files per 
        subject.
        
        Parameters
        ----------
        sj : int or str
            Subject number/identifier. Used to auto-locate files if
            eye_files='all'.
        eye_files : str or list, default='all'
            Eye tracker file paths or 'all' to auto-detect files 
            matching pattern 'sub_{sj}_ses_*.asc' in eye/raw folder.
        beh_files : str or list, default='all'
            Behavioral file paths or 'all' to auto-detect files matching
            pattern 'sub_{sj}_ses_*.csv' in beh/raw folder.
        start : str, default='start_trial'
            Event marker string in eye tracker file marking trial onset.
        trial_info : list, optional
            Additional trial information to extract from eye files.
            Default is None.
        stop : str, optional
            Event marker string for trial offset. If None, uses trial
            start markers to segment data. Default is None.
            
        Returns
        -------
        eye : np.ndarray
            Array of trial dictionaries from pygazeanalyser, containing
            keys: 'x', 'y', 'trackertime', 'events', etc.
        df : pd.DataFrame
            Behavioral data with practice trials removed, aligned to
            eye tracker trials.
        trial_info : np.ndarray
            Trial-specific timing or event information extracted from
            eye tracker data, shape (n_trials,).
            
        Notes
        -----
        Trial synchronization handles several edge cases:
        - More behavioral than eye trials: Removes extra from df
        - More eye than behavioral trials: Removes extra from eye
        - Mismatched trial numbers: Attempts alignment via trial_nr 
          field
        - Missing event timestamps: Interpolates from neighboring trials
        - Practice trials: Automatically removed if 'practice' column 
          exists
        
        File format support:
        - .asc: EyeLink ASCII files (via read_edf/read_edf_time_overlap)
        - .tsv: EyeTribe tab-separated files (via read_eyetribe)
        
        Examples
        --------
        >>> eye_obj = EYE()
        >>> # Auto-load all files for subject 1
        >>> eye, df, times = eye_obj.get_eye_data(sj=1)
        >>> 
        >>> # Load specific files with custom event markers
        >>> eye, df, times = eye_obj.get_eye_data(
        ...     sj=5,
        ...     eye_files=['session1.asc', 'session2.asc'],
        ...     beh_files=['session1.csv', 'session2.csv'],
        ...     start='trial_start',
        ...     stop='trial_end'
        ... )
        """

        # Format subject ID with zero-padding (only if needed for file globbing)
        if eye_files == 'all' or beh_files == 'all':
            sj = format_subject_id(sj)

        # read in behavior and eyetracking data
        if eye_files == 'all':
            eye_files = glob.glob(
                self.FolderTracker(
                    extension=['eye', 'raw'],
                    filename='sub_{}_ses_*.asc'.format(sj)
                )
            )	
            beh_files = glob.glob(
                self.FolderTracker(
                    extension=['behavioral', 'raw'],
                    filename='sub_{}_ses_*.csv'.format(sj)
                )
            )
            # if eye file does not exit remove beh file 
            if len(beh_files) > len(eye_files):
                # Extract session numbers using regex to handle runs properly
                eye_sessions = [
                    int(re.search(r'ses_0?(\d+)', 
                                  os.path.basename(f)).group(1))
                    for f in eye_files
                ]
                beh_sessions = [
                    int(re.search(r'ses_0?(\d+)', 
                                  os.path.basename(f)).group(1))
                    for f in beh_files
                ]
                for i, ses in enumerate(beh_sessions):
                    if ses not in eye_sessions:
                        beh_files.pop(i)

        if eye_files[0][-3:] == 'tsv':			
            eye = [read_eyetribe(file, start = start, missing = 0) 
                                                        for file in eye_files]
        elif eye_files[0][-3:] == 'asc':	
            if stop is None:
                eye = [
                    read_edf(
                        file,
                        start=start,
                        trial_info=trial_info,
                        missing=0
                    ) for file in eye_files
                ]
            else:	
                eye = [
                    read_edf_time_overlap(
                        file,
                        start=start,
                        stop=stop,
                        missing=0
                    ) for file in eye_files
                ]
            trial_info = [e[1] for e in eye]
            eye = [e[0] for e in eye]
        eye = np.array(eye[0]) if len(eye_files) == 1 else np.hstack(eye)
        if len(eye_files) == 1:
            trial_info = np.array(trial_info[0])  
        else: 
            np.hstack(trial_info)
        # filthy hack to deal with missing events
        if np.where(np.isnan(trial_info))[0].size > 0:
            for idx in np.where(np.isnan(trial_info))[0]:
                trial_info[idx] = trial_info[np.array((idx-1,idx+1))].mean()

        df = self.read_raw_beh(files = beh_files)

        # Handle case where no behavioral files exist
        if isinstance(df, list) and len(df) == 0:
            print('Warning: No behavioral files found for this session.')
            df = pd.DataFrame(index=range(eye.shape[0]))
        
        # check whether each beh trial is logged within eye
        nr_miss =  eye.shape[0] - df.shape[0]
        if nr_miss < 0:
            print('Trials in beh and eye do not match. Trials removed from') 
            print(' beh. Please inspect data carefully')

            eye_trials = []

            for i, trial in enumerate(eye):
                for event in trial['events']['msg']:
                    if start in event[1]:
                        trial_nr = int(''.join(filter(str.isdigit, event[1])))
                        # control for OpenSesame trial counter
                        eye_trials.append(trial_nr + 1)
                        if 'nr_trials' in df.columns and trial_nr + 1 not in df['nr_trials'].values:
                            print(trial_nr)

            #  TODO: make linking more generic
            if len(eye_trials) == eye.shape[0]:
                df.drop(df.index[nr_miss:], inplace=True) 
            else:
                if 'nr_trials' in df.columns:
                    eye_mask = np.in1d(df['nr_trials'].values, eye_trials)
                    df = df[np.array(eye_mask)]		
        elif nr_miss > 0:
            print(f'Trials in beh and eye do not match. Final {nr_miss}')
            print(f' removed from eye. Please inspect data carefully')
            eye = eye[:-nr_miss]
            trial_info = trial_info[:-nr_miss]

        # remove practice trials from eye and beh data
        if 'practice' in df.keys():
            eye = eye[np.array(df['practice'] == 'no')]
            trial_info = trial_info[np.array(df['practice'] == 'no')]
            df = df[df['practice'] == 'no']

        return eye, df, np.squeeze(np.array(trial_info))

    def interp_trial(self, trial: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate blinks and missing data in a single eye tracker 
        trial.
        
        Detects blink events, pads them with 75ms buffers, and performs
        linear interpolation across missing data periods. Returns 
        original data if trial contains only NaN values.
        
        Parameters
        ----------
        trial : dict
            Single trial dictionary from pygazeanalyser containing keys:
            - 'x': X gaze coordinates in pixels
            - 'y': Y gaze coordinates in pixels
            - 'trackertime': Sample timestamps in ms
            - 'events': Dict with 'Eblk' key listing blink events as
              [(start_time, end_time), ...]
              
        Returns
        -------
        x : np.ndarray
            X coordinates with blinks interpolated, shape (n_samples,).
            Returns zeros if trial is entirely missing.
        y : np.ndarray
            Y coordinates with blinks interpolated, shape (n_samples,).
            Returns zeros if trial is entirely missing.
            
        Notes
        -----
        Processing steps:
        1. Extract blink events from trial['events']['Eblk']
        2. Pad each blink by 75ms before and after (based on sfreq)
        3. Set padded blink periods to NaN
        4. Linearly interpolate across all NaN periods
        5. If entire trial is NaN, return zero arrays
        
        The 75ms padding accounts for partial blinks and eye closure/
        opening transitions that may not be fully detected.
        
        Examples
        --------
        >>> eye_obj = EYE(sfreq=1000)
        >>> trial = {
        ...     'x': np.array([100, 100, np.nan, np.nan, 120]),
        ...     'y': np.array([200, 200, np.nan, np.nan, 220]),
        ...     'trackertime': np.array([0, 1, 2, 3, 4]),
        ...     'events': {'Eblk': [(2, 3)]}
        ... }
        >>> x_interp, y_interp = eye_obj.interp_trial(trial)
        """

        # if no blinks detected return x, y
        blinks = trial['events']['Eblk']
        x = np.array(trial['x'])
        y = np.array(trial['y'])

        if np.isnan(x).any():
            return x, y

        #pad 75ms before and after blink
        pad = int(75/(1000/self.sfreq))
        for blink in blinks:
            idx = get_time_slice(trial['trackertime'], blink[0], blink[1])
            idx = slice(idx.start - pad, idx.stop + pad)
            x[idx] = None
            y[idx] = None

        # interpolate all blinks/missing data in x,y
        if not np.isnan(x).all():
            no_blink_idx = (~np.isnan(x)).nonzero()[0]
            blink_idx  = np.isnan(x).nonzero()[0]
            x_no_blink = x[~np.isnan(x)]
            y_no_blink = y[~np.isnan(y)] 
            x[np.isnan(x)] = np.interp(blink_idx , no_blink_idx, x_no_blink)
            y[np.isnan(y)] = np.interp(blink_idx , no_blink_idx, y_no_blink)
        else:
            x = np.zeros(x.size)
            y = np.zeros(y.size)

        return x, y

    def get_xy(
        self,
        eye: np.ndarray,
        start: float,
        end: float,
        start_event: str,
        interpolate_blinks: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract aligned gaze coordinates from eye tracker trial data.
        
        Retrieves X and Y gaze coordinates epoched around a specific 
        event marker. Aligns all trials to the start_event, optionally 
        interpolates blinks, and returns data on a uniform time grid. 
        Missing data periods are filled with zeros.
        
        Parameters
        ----------
        eye : np.ndarray
            Array of trial dictionaries from get_eye_data(), each 
            containing:
            - 'x', 'y': Gaze coordinates
            - 'trackertime': Sample timestamps
            - 'events': Dict with 'msg' key containing event markers
        start : float
            Epoch start time relative to start_event in ms 
            (can be negative).
        end : float
            Epoch end time relative to start_event in ms.
        start_event : str
            Event marker string to align trials to 
            (e.g., 'stimulus_onset').
        interpolate_blinks : bool, default=True
            Whether to interpolate blinks using interp_trial(). 
            If False, returns raw data with blink periods as NaN or 
            zeros.
            
        Returns
        -------
        x : np.ndarray
            X gaze coordinates in pixels, shape (n_trials, n_times).
            Zeros indicate missing data or trials shorter than epoch.
        y : np.ndarray
            Y gaze coordinates in pixels, shape (n_trials, n_times).
            Zeros indicate missing data or trials shorter than epoch.
        times : np.ndarray
            Time points in ms relative to start_event, shape (n_times,).
            Uniformly spaced based on self.sfreq.
            
        Notes
        -----
        Processing steps per trial:
        1. Find start_event in trial['events']['msg']
        2. Align trial['trackertime'] to event (time 0)
        3. Optionally interpolate blinks via interp_trial()
        4. Extract samples within [start, end] window
        5. Map to uniform time grid based on sfreq
        6. Fill non-overlapping periods with zeros
        
        Trials without the start_event or with no valid data remain as
        zero arrays.
        
        Examples
        --------
        >>> eye_obj = EYE(sfreq=1000)
        >>> eye, beh, _ = eye_obj.get_eye_data(sj=1)
        >>> # Extract 500ms before to 1000ms after 'stimulus_on'
        >>> x, y, times = eye_obj.get_xy(
        ...     eye, start=-500, end=1000,
        ...     start_event='stimulus_on',
        ...     interpolate_blinks=True
        ... )
        >>> print(f'Shape: {x.shape}, 
        ...		Time range: {times[0]}-{times[-1]}ms')
        """	

        times = np.arange(start, end,1000/self.sfreq)
        # initiate x and y array  
        x = np.zeros((len(eye),times.size))
        y = np.copy(x)
        # look for start_event in all logged events

        for i, trial in enumerate(eye):
            for event in trial['events']['msg']:
                if start_event in event[1]:
                    # Only process if trial has data
                    if trial['trackertime'].size > 0:
                        # Align to start_event
                        tr_times = trial['trackertime'] - event[0]

                        if interpolate_blinks and tr_times[0] >= start:
                            x_, y_  = self.interp_trial(trial)
                        else:
                            x_ = np.array(trial['x'])
                            y_ = np.array(trial['y'])

                        # find overlapping time points
                        idx = get_time_slice(tr_times, start, end)
                        trial_times = tr_times[idx]

                        # Find corresponding indices in target time array
                        target_idx = np.where((times >= trial_times[0]) & 
                                                (times <= trial_times[-1]))[0]
            
                        # Only copy data where times overlap
                        if len(target_idx) > 0:
                            x[i,target_idx] = x_[idx][:len(target_idx)]
                            y[i,target_idx] = y_[idx][:len(target_idx)]

        return x, y, times

    def set_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        drift_correct: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove noise and apply drift correction to gaze coordinates.
        
        Modifies x and y coordinates using SaccadeDetector algorithm to
        identify and mark noise/blink segments as NaN. Optionally applies
        drift correction by shifting coordinates toward screen center
        based on fixation period analysis.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in pixels, shape (n_trials, n_times).
        y : np.ndarray
            Y coordinates in pixels, shape (n_trials, n_times).
        times : np.ndarray
            Time points in ms, shape (n_times,).
        drift_correct : tuple of (float, float), optional
            Time window (start, end) in ms for drift correction 
            analysis.If specified, shifts coordinates based on fixation 
            deviation from screen center. Only applied to trials with no 
            saccades during this window. 
            Default is None (no correction).
            
        Returns
        -------
        x : np.ndarray
            X coordinates with noise marked as NaN and drift corrected,
            shape (n_trials, n_times).
        y : np.ndarray
            Y coordinates with noise marked as NaN and drift corrected,
            shape (n_trials, n_times).
            
        Notes
        -----
        Processing pipeline per trial:
        1. Calculate velocity and acceleration (SaccadeDetector)
        2. Detect and mark noise/blink segments as NaN
        3. If drift_correct specified:
           - Extract gaze during fixation window
           - Check for saccades using detect_events()
           - If no saccades and no missing data: shift x, y toward 
             center
           - Shift amount = (screen_center - mean_fixation_position)
        
        Trials with missing data at epoch start/end are skipped in the
        processing loop but remain in the output arrays.
        
        Examples
        --------
        >>> eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080))
        >>> # Apply noise detection only
        >>> x_clean, y_clean = eye_obj.set_xy(x, y, times)
        >>> 
        >>> # Apply noise detection + drift correction
        >>> x_clean, y_clean = eye_obj.set_xy(
        ...     x, y, times,
        ...     drift_correct=(-200, 0)  # Use 200ms pre-stimulus
        ... )
        """	

        SD = SaccadeDetector(
            self.sfreq, 
            screen_h=self.scr_h,
            viewing_dist=self.view_dist,
            screen_res=self.scr_res
        )

        # check whether x,y contain missing data at start and/or end trial
        mask = np.ones(x.shape[1],dtype = bool)
        for i in range(mask.size):
            if not np.any(x[:,i]):
                mask[i] = False

        # set blink and noise trials to nan
        for i, (x_, y_) in enumerate(zip(x[:,mask],y[:,mask])):
            V, A = SD.calc_velocity(x_.copy(),y_.copy())
            x_,y_,V,A = SD.noise_detect(x_.copy(),y_.copy(),V.copy(),A.copy())
            if drift_correct:
                idx = get_time_slice(
                    times[mask], drift_correct[0], drift_correct[1]
                )
                x_d = np.array(x_[idx])
                y_d = np.array(y_[idx])
                #TODO: add fixation quality metric

                # only corrrect if fixation period contains no missing data,
                #  and has no saccades
                if not np.isnan(x_d).any():
                    nr_sac = SD.detect_events(x_d, y_d)


                    if nr_sac == 0: 
                        x_ += (self.scr_res[0]/2) - x_d.mean()
                        y_ += (self.scr_res[1]/2) - y_d.mean()
        
            x[i,mask] = x_
            y[i,mask] = y_

        return x, y	

    def angles_from_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        drift_correct: Optional[Tuple[float, float]] = None
    ) -> list:
        """
        Calculate gaze deviation angles from fixation for each trial.
        
        Computes visual angle deviation from screen center for all gaze
        samples. Optionally applies drift correction before angle 
        calculation. Returns binned angle summaries and per-sample 
        angles.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in pixels, shape (n_trials, n_times).
        y : np.ndarray
            Y coordinates in pixels, shape (n_trials, n_times).
        times : np.ndarray
            Time points in ms, shape (n_times,).
        drift_correct : tuple of (float, float), optional
            Time window (start, end) in ms for drift correction.
            If provided, calls set_xy() to apply correction before
            calculating angles. Default is None (no correction).
            
        Returns
        -------
        angles : list
            List of length n_trials, each element is np.ndarray of
            visual angle deviations in degrees, shape (n_times,).
            
        Notes
        -----
        Processing pipeline:
        1. Optionally apply drift correction via set_xy()
        2. Call createAngleBins() with fixed parameters:
           - start=0, stop=3, step=0.25 (bin range 0-3°)
           - min_segment=40ms
        3. Return only the angles (discard bins)
        
        The method hardcodes binning parameters suitable for fixation
        analysis. Modify createAngleBins() call if different parameters
        are needed.
        
        Examples
        --------
        >>> eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080),
        ...               viewing_dist=60, screen_h=30)
        >>> # Get angles without drift correction
        >>> angles = eye_obj.angles_from_xy(x, y, times)
        >>> 
        >>> # Get angles with drift correction
        >>> angles = eye_obj.angles_from_xy(
        ...     x, y, times,
        ...     drift_correct=(-200, 0)
        ... )
        >>> print(f'Trial 1 mean deviation: {np.nanmean(angles[0]):.2f}°')
        
        See Also
        --------
        set_xy : Apply noise detection and drift correction
        create_angle_bins : Calculate angles and bin by deviation thresholds
        calculate_angle : Core angle calculation from pixel coordinates
        """

        # apply drift correction if specified
        if drift_correct:
            x, y = self.set_xy(x, y, times, drift_correct)
        bins, angles = self.create_angle_bins(x, y, 0, 3, 0.25, 40)

        return angles
    
    def create_angle_bins(
        self,
        x: np.ndarray,
        y: np.ndarray,
        start: float,
        stop: float,
        step: float,
        min_segment: float
    ) -> Tuple[list, list]:
        """
        Calculate gaze deviation angles and bin trials by maximum 
        deviation.
        
        Computes visual angle deviation from screen center for each 
        sample in each trial. Classifies trials into bins based on the
        maximum sustained deviation observed. A deviation counts only if
        it persists for at least min_segment duration.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in pixels, shape (n_trials, n_times).
        y : np.ndarray
            Y coordinates in pixels, shape (n_trials, n_times).
        start : float
            Minimum bin value in degrees (typically 0).
        stop : float
            Maximum bin value in degrees.
        step : float
            Bin width in degrees (e.g., 0.25 for quarter-degree bins).
        min_segment : float
            Minimum sustained duration in ms for a deviation to count.
            Converted to samples using self.sfreq.
            
        Returns
        -------
        bins : list
            List of length n_trials. Each element is the maximum bin
            value (in degrees) where deviation persisted for
            min_segment, or np.nan if trial has no valid data.
        angles : list
            List of length n_trials. Each element is np.ndarray of
            per-sample visual angles in degrees, shape (n_times,).
            
        Notes
        -----
        Binning algorithm per trial:
        1. Calculate angle deviation for all samples via 
           calculate_angle()
        2. For each bin threshold b in [start, stop) with step:
           a. Find samples where angle > b
           b. Identify contiguous segments
           c. If any segment ≥ min_segment samples, mark bin b
        3. Return maximum marked bin (or nan if none)
        
        Trials without valid data (all nan angles) are assigned nan bin.
        
        Examples
        --------
        >>> eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080))
        >>> # Bin trials by deviations 0-3° in 0.25° steps
        >>> # Require 40ms sustained deviation
        >>> bins, angles = eye_obj.create_angle_bins(
        ... 	x, y, start=0, stop=3, step=0.25, min_segment=40
        ... )
        >>> print(f'Trial 1 max bin: {bins[0]:.2f}°')
        >>> print(f'Trial 1 mean angle: {np.nanmean(angles[0]):.2f}°')
        
        See Also
        --------
        calculate_angle : Compute visual angle from pixel coordinates
        angles_from_xy : Convenience wrapper with drift correction
        """

        min_segment *= self.sfreq/1000.0

        bins, angles = [], []

        for i, (x_, y_) in enumerate(zip(x, y)):
            angle = self.calculate_angle(x_, y_)
            angles.append(angle)

            trial_bin = []
            for b in np.arange(start,stop,step):
                # get segments of data where deviation from fixation (angle) 
                # is larger than the current bin value (b)
                binned = np.where(angle > b)[0]
                segments = np.split(binned, 
                                    np.where(np.diff(binned) != 1)[0]+1)

                # check whether segment exceeds min duration
                segment_sizes = np.array([s.size for s in segments])
                if np.where(segment_sizes > min_segment)[0].size > 0:
                    trial_bin.append(b)

            # insert max binning segment or nan 
            # (in case of a trial without data)		
            if trial_bin:	
                bins.append(max(trial_bin))		
            else:
                bins.append(np.nan)

        return bins, angles

    def link_eye_to_eeg(
        self,
        eye_file: Union[str, list],
        beh_file: Union[str, list],
        start_trial: str,
        stop_trial: Optional[str],
        window_oi: Tuple[float, float],
        trigger_msg: str,
        drift_correct: Optional[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, np.ndarray]:
        """
        Load and process eye tracker data aligned to EEG triggers.
        
        Convenience wrapper that loads eye tracking and behavioral data,
        extracts gaze coordinates around trigger events, optionally 
        applies drift correction, and calculates visual angle 
        deviations.
        
        Parameters
        ----------
        eye_file : str or list
            Path(s) to eye tracker file(s) (.asc or .tsv).
        beh_file : str or list
            Path(s) to behavioral data file(s) (.csv).
        start_trial : str
            Event marker string indicating trial start in eye tracker 
            file.
        stop_trial : str, optional
            Event marker string indicating trial end. If None, uses 
            start markers to segment trials.
        window_oi : tuple of (float, float)
            Time window (start, end) in ms relative to trigger_msg for
            data extraction.
        trigger_msg : str
            Event marker string to align trials to (e.g., 
            'stimulus_onset').
        drift_correct : tuple of (float, float), optional
            Time window (start, end) in ms for drift correction. If 
            None, no correction applied.
            
        Returns
        -------
        x : np.ndarray
            X gaze coordinates in pixels, shape (n_trials, n_times).
        y : np.ndarray
            Y gaze coordinates in pixels, shape (n_trials, n_times).
        times : np.ndarray
            Time points in ms relative to trigger_msg, shape (n_times,).
        bins : list
            Maximum sustained deviation bin per trial in degrees.
        angles : list
            Per-sample visual angle deviations per trial.
        trial_info : np.ndarray
            Trial timing information from eye tracker, 
            shape (n_trials,).
            
        Notes
        -----
        Processing pipeline:
        1. Load eye tracker and behavioral data via get_eye_data()
        2. Extract x, y coordinates in window_oi via get_xy()
        3. Optionally apply drift correction via set_xy()
        4. Calculate angle bins via create_angle_bins()
        
        This is a convenience method combining multiple processing steps
        for typical eye-EEG analysis workflows.
        
        Examples
        --------
        >>> eye_obj = EYE(sfreq=1000, screen_res=(1920, 1080))
        >>> x, y, times, bins, angles, info = eye_obj.link_eye_to_eeg(
        ...     eye_file='sub_01.asc',
        ...     beh_file='sub_01.csv',
        ...     start_trial='start_trial',
        ...     stop_trial=None,
        ...     window_oi=(-200, 1000),
        ...     trigger_msg='stimulus_on',
        ...     drift_correct=(-200, 0)
        ... )
        
        See Also
        --------
        get_eye_data : Load and synchronize eye and behavioral files
        get_xy : Extract aligned gaze coordinates
        set_xy : Apply noise detection and drift correction
        create_angle_bins : Calculate angle deviations and bins
        """
        
        # read in eye data (linked to behavior)
        print('reading in eye tracker data')
        eye, beh, trial_info = self.get_eye_data(
            '', eye_file, beh_file, start_trial, trigger_msg, stop_trial
        )
        # collect x, y data 
        x, y, times = self.get_xy(eye, window_oi[0], 
                                window_oi[1], trigger_msg)	
        # apply drift correction if specified
        if drift_correct:
            x, y = self.set_xy(x, y, times, drift_correct)
        bins, angles = self.create_angle_bins(x, y, 0, 3, 0.25, 40)

        return x, y, times, bins, angles, trial_info

    def calculate_angle(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xc: Optional[float] = None,
        yc: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate visual angle deviation from center point.
        
        Computes visual angle in degrees based on Euclidean distance in
        pixels between gaze coordinates and a reference center point.
        Uses screen geometry from instance attributes to convert pixel
        distance to visual degrees.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in pixels, any shape.
        y : np.ndarray
            Y coordinates in pixels, same shape as x.
        xc : float, optional
            X coordinate of reference center in pixels. If None, uses
            screen center (self.scr_res[0] / 2). Default is None.
        yc : float, optional
            Y coordinate of reference center in pixels. If None, uses
            screen center (self.scr_res[1] / 2). Default is None.
            
        Returns
        -------
        visual_angle : np.ndarray
            Visual angle deviation in degrees from (xc, yc), same shape
            as input x and y.
            
        Notes
        -----
        Calculation steps:
        1. Compute degrees per pixel: arctan(screen_h/2 / viewing_dist)
           divided by vertical resolution / 2
        2. Calculate Euclidean pixel distance: sqrt((x-xc)² + (y-yc)²)
        3. Convert to degrees: pixel_distance * degrees_per_pixel
        
        Typical values for degrees per pixel are ~0.02-0.04° depending
        on screen size and viewing distance.
        
        Examples
        --------
        >>> eye_obj = EYE(viewing_dist=60, screen_res=(1920, 1080),
        ...              screen_h=30)
        >>> x = np.array([960, 1000, 920])  # Around center
        >>> y = np.array([540, 540, 540])
        >>> angles = eye_obj.calculate_angle(x, y)
        >>> print(angles)  # [0.0, ~1.2°, ~1.2°]
        
        See Also
        --------
        create_angle_bins : Uses this method to compute trial-wise 
                            angles
        degrees_to_pixels : Static method for inverse conversion
        """
        
        # Use screen center if not specified
        if xc is None:
            xc = self.scr_res[0] / 2
        if yc is None:
            yc = self.scr_res[1] / 2
        
        # calculate visual angle of a single pixel
        deg_per_px = degrees(
            atan2(0.5 * self.scr_h, self.view_dist)
        ) / (0.5 * self.scr_res[1])

        # calculate euclidean distance from specified center
        pix_eye = np.sqrt((x - xc)**2 + (y - yc)**2)

        # transform pixels to visual degrees
        visual_angle = pix_eye * deg_per_px

        return visual_angle	

    @staticmethod
    def degrees_to_pixels(
        h: float = 30,
        d: float = 60,
        r: int = 1080
    ) -> float:
        """
        Calculate degrees of visual angle per pixel.
        
        Static utility method to compute the visual angle subtended by
        a single pixel based on screen geometry and viewing distance.
        This is the inverse operation of pixel-to-degree conversion used
        in calculate_angle().
        
        Parameters
        ----------
        h : float, default=30
            Screen height in cm.
        d : float, default=60
            Viewing distance from observer to screen in cm.
        r : int, default=1080
            Vertical screen resolution in pixels.
            
        Returns
        -------
        deg_per_pix : float
            Visual angle in degrees subtended by one pixel.
            Typical values are ~0.02-0.04 degrees per pixel.
            
        Notes
        -----
        The calculation uses the formula:
        deg_per_pixel = arctan(screen_h/2 / viewing_dist) / 
                        (resolution/2)
        
        This is a static method for convenience when screen parameters
        are not available as instance attributes. For instance-based
        calculations, use calculate_angle() which uses self.scr_h,
        self.view_dist, and self.scr_res.
        
        Examples
        --------
        >>> # Standard 1080p monitor at 60cm
        >>> deg_per_px = EYE.degrees_to_pixels(h=30, d=60, r=1080)
        >>> print(f'{deg_per_px:.4f} degrees per pixel')
        0.0286 degrees per pixel
        >>> 
        >>> # 4K monitor at 90cm
        >>> deg_per_px = EYE.degrees_to_pixels(h=35, d=90, r=2160)
        >>> print(f'{deg_per_px:.4f} degrees per pixel')
        
        See Also
        --------
        calculate_angle : Instance method for computing visual angles
        """

        deg_per_pix = degrees(atan2(0.5 * h, d)) / (0.5 * r)

        return deg_per_pix	

class SaccadeDetector(object):
    """
    Adaptive algorithm for saccade and glissade detection in 
    eye-tracking data.
    
    Implements the velocity-based algorithm from Nyström & Holmqvist 
    (2010):"An adaptive algorithm for fixation, saccade, and glissade 
    detection in eyetracking data". The algorithm uses adaptive, 
    noise-dependent velocity thresholds and is specifically designed to 
    detect glissades (small involuntary eye movements at the end of 
    saccades).
    
    Parameters
    ----------
    sfreq : float
        Sampling frequency of eye tracker in Hz.
        
    Attributes
    ----------
    sfreq : float
        Sampling frequency in Hz.
    min_sac : float
        Minimum saccade duration in seconds (default: 0.01 = 10ms).
    peak_thresh : float or None
        Adaptive peak velocity threshold for saccade detection. Set by
        estimate_thresh() method.
    sacc_thresh : float or None
        Adaptive onset/offset velocity threshold. Set by 
        estimate_thresh() method.
        
    Notes
    -----
    The algorithm operates in several stages:
    1. Velocity/acceleration calculation using Savitzky-Golay filtering
    2. Noise detection (blinks, physiologically impossible movements)
    3. Adaptive threshold estimation based on fixation periods
    4. Saccade detection using peak velocity thresholds
    5. Glissade detection following saccades
    
    References
    ----------
    Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for 
    fixation,saccade, and glissade detection in eyetracking data. 
    Behavior ResearchMethods, 42(1), 188-204.
    
    Examples
    --------
    >>> # Detect saccades in eye tracker data
    >>> detector = SaccadeDetector(sfreq=1000)
    >>> saccade_mask = detector.detect_events(x_coords, y_coords, 
    ... 										output='mask')
    >>> print(f'Found {saccade_mask.sum()} saccades')
    """

    
    def __init__(
        self, 
        sfreq: float,
        screen_h: float = 30.0,
        viewing_dist: float = 60.0,
        screen_res: tuple = (1920, 1080)
    ) -> None:
        """
        Initialize the saccade detector.
        
        Parameters
        ----------
        sfreq : float
            Sampling frequency of eye tracker in Hz.
        screen_h : float, default=30.0
            Screen height in cm.
        viewing_dist : float, default=60.0
            Viewing distance in cm.
        screen_res : tuple, default=(1920, 1080)
            Screen resolution (width, height) in pixels.
        """
        self.sfreq = float(sfreq)
        self.min_sac = 0.01
        self.screen_h = screen_h
        self.viewing_dist = viewing_dist
        self.screen_res = screen_res
        # Calculate degrees per pixel for this setup
        self.deg_per_pixel = degrees(atan2(0.5 * screen_h, viewing_dist)) / (0.5 * screen_res[1]) 

    def detect_events(
        self,
        x: np.ndarray,
        y: np.ndarray,
        output: str = 'mask'
    ) -> Union[np.ndarray, dict]:
        """
        Detect saccades and glissades in eye-tracking data.
        
        Classifies gaze points into fixations, saccades, glissades, 
        blinks,and noise using the adaptive threshold algorithm. 
        Processes each trial independently.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in pixels. Shape (n_trials, n_times) or 
            (n_times,).
        y : np.ndarray
            Y coordinates in pixels. Shape (n_trials, n_times) or 
            (n_times,).
        output : str, default='mask'
            Output format:
            - 'mask': Returns boolean array (True where saccade
              detected)
            - 'dict': Returns dictionary with saccade onset/offset info
            
        Returns
        -------
        np.ndarray or dict
            If output='mask': Boolean array of shape (n_trials,) 
            indicating trials with saccades.
            If output='dict': Dictionary mapping saccade numbers to 
            (onset_idx, offset_idx) tuples.
            
        Notes
        -----
        Trials containing NaN values are marked as np.nan in the output.
        The method automatically reshapes 1D input to 2D.
        
        Processing pipeline:
        1. Calculate velocity and acceleration (calc_velocity)
        2. Detect and remove noise (noise_detect)
        3. Estimate adaptive thresholds (estimate_thresh)
        4. Detect saccades and glissades (saccade_detection)
        
        Examples
        --------
        >>> detector = SaccadeDetector(sfreq=1000)
        >>> # Detect saccades across multiple trials
        >>> has_saccade = detector.detect_events(x_data, y_data)
        >>> print(f'{has_saccade.sum()}/{len(has_saccade)} trials '
        ... 'with saccades')
        """	

        if x.ndim == 1:
            x = x.reshape(1,-1)
            y = y.reshape(1,-1)

        sacc = np.empty(x.shape[0], dtype = dict)

        for i, (x_, y_) in enumerate(zip(x, y)):
            # check whether trial contains missing data
            if np.isnan(np.hstack((x_,y_))).any():
                sacc[i] = np.nan
                continue

            V, A = self.calc_velocity(x_,y_)
            x_, y_, V, A = self.noise_detect(x_, y_, V, A)
            self.estimate_thresh(V)
            if self.peak_thresh != None:
                sacc[i] = self.saccade_detection(V, output = output)	
            else:
                sacc[i] = np.nan	

        if output == 'mask':
            sacc = np.array(sacc, dtype = bool)
                
        return sacc

    def calc_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate angular velocity and acceleration from gaze 
        coordinates.
        
        Computes velocity (deg/sec) and acceleration (deg/sec²) using
        Savitzky-Golay filtering. Takes first and second derivatives of
        smoothed x and y coordinates, then calculates Euclidean 
        magnitude.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in pixels, shape (n_times,).
        y : np.ndarray  
            Y coordinates in pixels, shape (n_times,).
            
        Returns
        -------
        V : np.ndarray
            Angular velocity in degrees per second, shape (n_times,).
        A : np.ndarray
            Angular acceleration in degrees per second², 
            shape (n_times,).
            
        Notes
        -----
        Uses 2nd order polynomial Savitzky-Golay filter with window 
        length determined by min_sac (minimum saccade duration). Filter 
        window:F = 2 * ceil(min_sac * sfreq) - 1
        
        Velocity/acceleration are computed as:
        V = sqrt(Vx² + Vy²) * deg_per_pixel * sfreq
        A = sqrt(Ax² + Ay²) * deg_per_pixel * sfreq
        
        Returns zero arrays if filtering fails (e.g., insufficient 
        samples).
        
        See Also
        --------
        scipy.signal.savgol_filter : Savitzky-Golay filtering
        """		
    
        N = 2 											# order of poynomial
        span = np.ceil(self.min_sac * self.sfreq)		# span of filter
        F = int(2 * span - 1)							# window length
        
        # calculate the velocity and acceleration
        try:
            x_ = savgol_filter(x, F, N, deriv = 0)
            y_ = savgol_filter(y, F, N, deriv = 0)
        except:
            return np.zeros(x.size), np.zeros(x.size)


        V_x = savgol_filter(x, F, N, deriv = 1)
        V_y = savgol_filter(y, F, N, deriv = 1)
        V = np.sqrt(V_x**2 + V_y**2) * self.deg_per_pixel * self.sfreq  

        A_x = savgol_filter(x, F, N, deriv = 2)
        A_y = savgol_filter(y, F, N, deriv = 2)
        A = np.sqrt(A_x**2 + A_y**2) * self.deg_per_pixel * self.sfreq**2

        return V, A 

    def noise_detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        V: np.ndarray,
        A: np.ndarray,
        V_thresh: float = 1000,
        A_thresh: float = 100000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:	
        """
        Detect and remove noise from eye-tracking data.
        
        Identifies noise segments including blinks (indicated by [0,0]
        coordinates) and physiologically impossible movements (velocity/
        acceleration exceeding thresholds). Extends noise boundaries to
        include onset/offset transitions.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates in pixels, shape (n_times,).
        y : np.ndarray
            Y coordinates in pixels, shape (n_times,).
        V : np.ndarray
            Velocity data in deg/sec, shape (n_times,).
        A : np.ndarray
            Acceleration data in deg/sec², shape (n_times,).
        V_thresh : float, default=1000
            Maximum physiologically plausible velocity in deg/sec.
            Based on Bahill et al. (1981).
        A_thresh : float, default=100000
            Maximum physiologically plausible acceleration in deg/sec².
            Based on Bahill et al. (1981).
            
        Returns
        -------
        x : np.ndarray
            X coordinates with NaN inserted in noise segments.
        y : np.ndarray
            Y coordinates with NaN inserted in noise segments.
        V : np.ndarray
            Velocity with NaN inserted in noise segments.
        A : np.ndarray
            Acceleration with NaN inserted in noise segments.
            
        Notes
        -----
        Noise detection criteria:
        - Blinks: x ≤ 0 or y ≤ 0
        - Impossible velocity: V > V_thresh
        - Impossible acceleration: |A| > A_thresh
        
        The function extends noise boundaries by searching backward and
        forward from each noise segment until velocity drops below 
        2 * median(V), ensuring complete removal of noise artifacts.
        
        References
        ----------
        Bahill, A. T., Adler, D., & Stark, L. (1981). Most naturally
        occurring human saccades have magnitudes of 15 degrees or less.
        Investigative Ophthalmology & Visual Science, 14(6), 468-469.
        """	

        on_off = np.median(V)*2
        
        # detect noise segments 
        noise = np.where(
            (x <= 0) | (y <= 0) | (V > V_thresh) | (abs(A) > A_thresh)
        )[0]
        noise_seg = np.split(noise, np.where(np.diff(noise) != 1)[0]+1)

        # loop over segments 
        on_off_seg = []
        for seg in noise_seg:
            if seg.size > 0:
                # go back in time 
                for i, v in enumerate(np.flipud(V[:min(seg)])):
                    if v >= on_off:
                        on_off_seg.append(min(seg) - i - 1)
                    else:
                        break	

                # go forward in time
                for i, v in enumerate(V[max(seg) + 1:]):
                    if v >= on_off:
                        on_off_seg.append(max(seg) + i + 1)
                    else:
                        break	

        # add segments to noise			
        noise = np.array(np.hstack((noise,on_off_seg)), dtype = int)	

        if noise.size > 0:
            
            x[noise] = np.nan
            y[noise] = np.nan
            V[noise] = np.nan
            A[noise] = np.nan

        return x, y, V, A

    def estimate_thresh(
        self,
        V: np.ndarray,
        peak_thresh: float = 100,
        min_fix: float = 0.04
    ) -> None:	
        """
        Estimate adaptive velocity thresholds for saccade detection.
        
        Uses iterative procedure to set data-driven peak velocity and
        saccade onset/offset thresholds based on fixation period noise
        levels. Thresholds are stored as instance attributes.
        
        Parameters
        ----------
        V : np.ndarray
            Velocity data in deg/sec, shape (n_times,).
        peak_thresh : float, default=100
            Initial peak velocity threshold in deg/sec to start 
            iteration.
        min_fix : float, default=0.04
            Minimum fixation duration in seconds (40ms default).
            
        Returns
        -------
        None
            Sets self.peak_thresh and self.sacc_thresh attributes.
            
        Attributes Set
        --------------
        peak_thresh : float or None
            Peak velocity threshold (µ + 6σ of fixation noise).
            None if insufficient fixation periods found.
        sacc_thresh : float or None
            Saccade onset/offset threshold (µ + 3σ of fixation noise).
            None if insufficient fixation periods found.
            
        Notes
        -----
        Iterative algorithm:
        1. Identify samples below current peak_thresh as fixations
        2. Extract center portions of fixations (exclude edges)
        3. Calculate µ and σ of fixation velocity
        4. Update: peak_thresh = µ + 6σ, sacc_thresh = µ + 3σ
        5. Repeat until |new_thresh - old_thresh| < 1
        
        Iteration stops after 50 cycles to prevent infinite loops.
        Prints warning if no valid fixation periods found.
        
        Examples
        --------
        >>> detector = SaccadeDetector(sfreq=1000)
        >>> V, A = detector.calc_velocity(x, y)
        >>> detector.estimate_thresh(V)
        >>> print(f'Peak threshold: {detector.peak_thresh:.1f} deg/s')
        """	
        
        # Step 3: velocity threshold estimation
        old_thresh = float('inf')
        # used to extract the center of the fixation
        cent_fix = min_fix * self.sfreq/6    			

        flip = 0
        while abs(peak_thresh - old_thresh) > 1:

            # control for infinite flipping
            if peak_thresh > old_thresh:
                flip += 1 
                if flip > 50:
                    # ADD CODE TO COMBINE ALL OLD PEAK THESH????
                    print('Peak_thresh kept flipping, ' \
                    'broke from infinite loop')
                    break

            old_thresh = peak_thresh
            fix_samp = np.where(V <= peak_thresh)[0]	
            fix_seg = np.split(fix_samp, np.where(np.diff(fix_samp) != 1)[0]+1)
            fix_noise = []
            
            # epoch should contain at least one fixation period
            fix_durations = np.array([s.size / self.sfreq for s in fix_seg])
            if sum(fix_durations >= min_fix) == 0:
                peak_thresh = None
                sacc_thresh = None
                print('Segment does not contain sufficient samples '
                'for event detection')
                break
            
            # loop over all possible fixations
            for seg in fix_seg:

                # check whether fix duration exceeds minimum duration
                if (seg.size / self.sfreq) < min_fix:
                    continue

                # extract the samples from the center of fixation  
                # (exclude outer portions)
                f_noise = V[int(floor(seg[0] + cent_fix)): 
                                        int(ceil(seg[-1] - cent_fix) + 1)]	
                fix_noise.append(f_noise)

            # handle empty fix_noise case
            if len(fix_noise) == 0:
                peak_thresh = None
                sacc_thresh = None
                print('No valid fixation periods found for ' \
                'threshold estimation')
                break
            elif len(fix_noise) == 1:
                fix_noise = fix_noise[0]  
            else:
                fix_noise = np.hstack(fix_noise)	

            mean_noise = np.nanmean(fix_noise)	
            std_noise = np.nanstd(fix_noise)
        
            #adjust the peak velocity threshold based on the noise level
            peak_thresh = mean_noise + 6 * std_noise
            sacc_thresh = mean_noise + 3 * std_noise
             
        self.peak_thresh = peak_thresh	
        self.sacc_thresh = sacc_thresh

    def saccade_detection(
        self,
        V: np.ndarray,
        min_sac: float = 0.01,
        min_fix: float = 0.04,
        output: str = 'mask'
    ) -> Union[int, dict]:	
        """
        Detect saccades and glissades in velocity data.
        
        Searches for saccade peaks using adaptive thresholds, identifies
        onset/offset boundaries, and detects post-saccadic glissades.
        Processes velocity data that has been cleaned by noiseDetect and
        thresholded by estimateThresh.
        
        Parameters
        ----------
        V : np.ndarray
            Velocity data in deg/sec, shape (n_times,).
        min_sac : float, default=0.01
            Minimum saccade duration in seconds (10ms default).
        min_fix : float, default=0.04
            Minimum fixation duration in seconds (40ms default).
        output : str, default='mask'
            Output format:
            - 'mask': Returns number of saccades detected (int)
            - 'dict': Returns dictionary with saccade and glissade timing
            
        Returns
        -------
        int or dict
            If output='mask': Number of detected saccades (int).
            If output='dict': Dictionary with keys 'saccades' and 
            'glissades'. Each maps event number (str) to (onset_idx, 
            offset_idx) tuple.
            
        Notes
        -----
        Saccade detection algorithm:
        1. Find peaks exceeding self.peak_thresh
        2. Search backward for onset (V < sacc_thresh, dV/dt ≥ 0)
        3. Calculate local noise level for adaptive offset detection
        4. Search forward for offset (V < sacc_thresh, dV/dt ≥ 0)
        5. Validate duration ≥ min_sac and no NaN values
        
        Glissade detection (two mutually exclusive criteria):
        - Low velocity: V between sacc_thresh and peak_thresh
        - High velocity: V > peak_thresh
        
        Glissade onset = saccade offset
        Glissade offset = first local minimum after peak
        
        Glissades are rejected if:
        - Amplitude exceeds preceding saccade amplitude
        - Duration > 2 * min_fix (too long)
        - Contains NaN values
        
        References
        ----------
        Nyström, M., & Holmqvist, K. (2010). Behavior Research Methods,
        42(1), 188-204.
        
        Examples
        --------
        >>> detector = SaccadeDetector(sfreq=1000)
        >>> V, A = detector.calc_velocity(x, y)
        >>> x, y, V, A = detector.noise_detect(x, y, V, A)
        >>> detector.estimate_thresh(V)
        >>> sac_dict = detector.saccade_detection(V, output='dict')
        >>> print(f'Saccade 1: onset={sac_dict[\"1\"][0]}, 
        ...			offset={sac_dict[\"1\"][1]}')
        """	

        # create array to store indices of detected saccades and glissades
        saccade_idx = []
        glissade_idx = []

        # initiate saccade and glissade counters
        nr_sac = 0
        nr_gliss = 0
        sac_dict = {}
        gliss_dict = {}
        gliss_off = []  # we start with missing glissades

        # start by getting segments of data with velocities 
        # above the selected peak velocity
        poss_sac = np.where(V > self.peak_thresh)[0]	
        sac_seg = np.split(poss_sac, np.where(np.diff(poss_sac) != 1)[0]+1)

        for seg in sac_seg:

            # PART 1: DETECT SACCADES

            # if the peak consists of less than 1/6 of the min_sac duration, 
            # it is probably noise
            if seg.size <= ceil(min_sac/6.0 * self.sfreq): continue
            
            # check whether the peak is already included in the previous 
            # saccade (is possible for glissades) 
            if nr_sac > 0 and gliss_off:
                if len(set(seg).intersection(
                    np.hstack((saccade_idx,glissade_idx)))) > 0:
                    continue

            # get idx of saccade onset
            onset = np.where((V[:seg[0]] <= self.sacc_thresh) * 
                             (np.hstack((np.diff(V[:seg[0]]),0)) >= 0))[0]
            if onset.size == 0: continue
            sac_on = onset[-1]

            # calculate local fix noise (adaptive part), used for 
            # saccade offset back in time
            V_local = V[
                sac_on :int(ceil(max(0,sac_on - min_fix * self.sfreq))) :-1
            ] 
            V_noise = V_local.mean() + 3 * V_local.std()
            sacc_thresh = V_noise * 0.3 + self.sacc_thresh * 0.7	
            # check whether the local V noise exceeds the peak V threshold 		
            # (i.e. whether saccade is preceded by period of stillness)
            if V_noise > self.peak_thresh: continue
                
            # detect end of saccade (without glissade)
            offset = np.where((V[seg[-1]:] <= sacc_thresh) * 
                              (np.hstack((np.diff(V[seg[-1]:]), 0))	>= 0))[0]
            if offset.size == 0: continue
            sac_off = seg[-1] + offset[0]

            # make sure that saccade duration exceeds minimum duration and 
            # does not contain any nan value
            if (np.isnan(V[sac_on:sac_off]).any()) or \
                (sac_off - sac_on)/ self.sfreq < min_sac: continue

            # if all criteria are fulfilled the segment can be labelled 
            # as a saccade
            nr_sac += 1	
            saccade_idx = np.array(
                np.hstack((saccade_idx, np.arange(sac_on,sac_off))), 
                dtype = int)
            sac_dict.update({str(nr_sac):(sac_on, sac_off)})			
            # PART 2: DETECT GLISSADES (high and low velocity glissades 
            # are mutually exclusive)

            # only search for glissade peaks in a window smaller than 
            # fix duration after saccade end
            poss_gliss = V[
                sac_off: int(ceil(min(sac_off + min_fix * self.sfreq, V.size)))
            ]
            if poss_gliss.size == 0: continue			
            # option 1: low velocity criteria
            # detect only 'complete' peaks (i.e. with a beginning and end)
            peak_idx_w = np.array(poss_gliss >= sacc_thresh, dtype =int)
            end_idx = np.where(np.diff(peak_idx_w) != 0)[0]
            if end_idx.size > 1:
                gliss_w_off = end_idx[1:end_idx.size:2][-1]
            else:
                gliss_w_off = []	

            # option 2: high velocity glissade	
            peak_idx_s = np.array(poss_gliss >= self.peak_thresh, dtype =int)
            gliss_s_off = np.where(peak_idx_s > 0)[0]
            if gliss_s_off.size > 0:
                gliss_s_off = gliss_s_off[-1] 

            # make sure that saccade amplitude is larger than the 
            # glissade amplitude	
            if max(poss_gliss) > max(V[sac_on:sac_off]):
                gliss_w_off = gliss_s_off = []

            # if a glissade is detected, get the offset of the glissade
            if gliss_w_off:
                gliss_off = sac_off + gliss_w_off
                diff = np.diff(V[gliss_off:])
                #TODO: check whether this fix is ok
                if sum(diff > 0) > 0:
                    gliss_off += np.where(diff >= 0)[0][0] - 1
    
                if np.isnan(V[sac_off:gliss_off]).any() or \
                    (gliss_off - sac_off)/ self.sfreq > 2 * min_fix:
                        gliss_off = []
                else:
                    glissade_idx = np.array(
                        np.hstack((glissade_idx, 
                     np.arange(sac_off, gliss_off))), dtype=int)
                    nr_gliss += 1
                    gliss_dict.update({str(nr_gliss): (sac_off, gliss_off)})
                    
        if output == 'mask':	
            return nr_sac
        else:
            return {'saccades': sac_dict, 'glissades': gliss_dict}









