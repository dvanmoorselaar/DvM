"""
Eye tracking utility functions for the DvM Toolbox.

This module provides standalone utility functions for eye tracking
data processing that are used across multiple analysis modules.

Functions
--------------
exclude_eye : Trial exclusion based on eye movements
    Filters trials with fixation breaks using either eye tracker data
    (gaze deviation) or EOG channels (step algorithm). Updates
    preprocessing statistics.

bin_tracker_angles : Classify trials by fixation quality
    Summarizes eye tracker data per trial, marking trials with sustained
    deviations from fixation exceeding specified thresholds.

eog_filt : EOG-based eye movement detection
    Detects saccades using sliding window algorithm on EOG channels as
    fallback when eye tracker data is unavailable.

Created by Dirk van Moorselaar on 20-01-2026.
Copyright (c) 2026 DvM. All rights reserved.
"""

import os
import warnings
import numpy as np
import pandas as pd
import mne
from contextlib import redirect_stdout
from numpy.lib.npyio import NpzFile
from typing import Tuple, Optional
from IPython import embed
from support.preprocessing_utils import get_time_slice


def exclude_eye(
    sj: int,
    session: int,
    df: pd.DataFrame,
    epochs: mne.Epochs,
    eye_dict: dict,
    eye: Optional[NpzFile] = None,
    preproc_file: Optional[str] = None
) -> Tuple[pd.DataFrame, mne.Epochs]:
    """
    Exclude trials with eye movements using tracker or EOG data.

    Filters trials based on fixation breaks detected either via eye
    tracker data (gaze deviation) or EOG channels (step algorithm).
    Updates preprocessing statistics file if provided.

    Parameters
    ----------
    sj : int
        Subject identifier for logging purposes.
    session : int
        Session identifier for logging purposes.
    df : pd.DataFrame
        Behavioral data aligned to epochs. Modified in-place by
        removing excluded trials.
    epochs : mne.Epochs
        Epoched EEG data. Modified in-place by dropping trials.
    eye_dict : dict
        Eye movement detection parameters. Supported keys:
            - 'window_oi' : tuple, time window to analyze (default: 
               full epoch)
            - 'eye_ch' : str, EOG channel name (default: 'HEOG')
            - 'angle_thresh' : float, max gaze deviation in visual 
               degrees
            - 'step_param' : tuple, (windowsize_ms, step_ms, thresh_uV)
              for EOG step detection (default: (200, 10, 15e-6))
            - 'use_tracker' : bool, use eye tracker data if available
            - 'use_eog' : bool, use EOG for trials without tracker data
            - 'drift_correct' : tuple, time window for drift correction
            - 'viewing_dist' : float, viewing distance in cm 
               (for tracker)
            - 'screen_res' : tuple, screen resolution (width, height) in 
               pixels
            - 'screen_h' : float, screen height in cm
    eye : NpzFile, optional
        Eye tracker data file containing 'x', 'y', 'times', 'sfreq' 
        arrays. If None, attempts to use eye channels in epochs. 
        Default is None.
    preproc_file : str, optional
        Path to preprocessing CSV file to update with eye exclusion
        statistics. Default is None (no logging).

    Returns
    -------
    df : pd.DataFrame
        Behavioral data with excluded trials removed, index reset.
    epochs : mne.Epochs
        EEG epochs with excluded trials removed.

    Warnings
    --------
    Prints warnings if eye tracker data is not found and EOG fallback
    is used.

    Notes
    -----
    The function applies a two-stage exclusion process:
    1. Eye tracker-based: Detects gaze deviations exceeding angle_thresh
    2. EOG-based fallback: For trials without tracker data, applies
       step algorithm to EOG channel

    Exclusion statistics are printed and optionally saved to the
    preprocessing file.

    Examples
    --------
    >>> # Exclude trials with >100 visual degrees deviation
    >>> eye_params = {
    ...     'angle_thresh': 100,
    ...     'use_tracker': True,
    ...     'viewing_dist': 90,
    ...     'screen_res': (1920, 1080),
    ...     'screen_h': 30
    ... }
    >>> df, epochs = exclude_eye(
    ...     sj=1,
    ...     df=behavioral_data,
    ...     epochs=eeg_epochs,
    ...     eye_dict=eye_params,
    ...     eye=eye_file
    ... )

    See Also
    --------
    bin_tracker_angles : Summarize tracker data per trial
    eog_filt : Step algorithm for EOG-based detection
    """

    # initialize some parameters
    if 'drift_correct' in eye_dict:
        drift_correct = eye_dict['drift_correct']
    else:
        drift_correct = False
    if 'window_oi' not in eye_dict:
        eye_dict['window_oi'] = (epochs.tmin, epochs.tmax)
    if 'eye_ch' not in eye_dict:
        print('Eye channel is not specified in eyedict, using HEOG as default')
        eye_dict['eye_ch'] = 'HEOG'
    if 'use_eog' not in eye_dict:
        eye_dict['use_eog'] = True

    # specify window of interest
    s, e = eye_dict['window_oi']	
    if drift_correct:
        if drift_correct[0] < s:
            s = drift_correct[0]	

    # check whether selection should be based on eyetracker data
    if 'use_tracker' not in eye_dict or not eye_dict['use_tracker']:
        tracker_bins = np.full(df.shape[0], np.nan)
        perc_tracker = 'no tracker'
        window_idx = get_time_slice(epochs.times,s,e)
    else:
        if isinstance(eye,NpzFile) or ('x' in epochs.ch_names):
            if eye is not None:
                x, y, times = eye['x'], eye['y'], eye['times']
                sfreq = int(eye['sfreq'])
                window_idx = get_time_slice(times,s,e)
                x = x[:,window_idx]
                y = y[:,window_idx]
                times = times[window_idx]
            else:
                window_idx = get_time_slice(epochs.times,s,e)
                x = epochs._data[:, epochs.ch_names.index('x'), window_idx]
                y = epochs._data[:, epochs.ch_names.index('y'), window_idx]
                times = epochs.times[window_idx]
                sfreq = epochs.info['sfreq']

            from analysis.EYE import EYE
            EO = EYE(sfreq = sfreq,
                    viewing_dist = eye_dict['viewing_dist'],
                    screen_res = eye_dict['screen_res'],
                    screen_h = eye_dict['screen_h'])
            angles = EO.angles_from_xy(x.copy(), y.copy(), times, 
                                           drift_correct)
            if eye_dict['window_oi'][0] > times[0]:
                window_idx = get_time_slice(times, eye_dict['window_oi'][0], e)
                angles_oi = np.array(angles)[:,window_idx]
            else:
                angles_oi = np.array(angles)
            min_samples = 40 * epochs.info['sfreq'] / 1000  # 40 ms
            tracker_bins = bin_tracker_angles(angles_oi, 
                                               eye_dict['angle_thresh'], 
                                            min_samples)
            nr_bad_trials = sum(tracker_bins == 1)
            perc_tracker = np.round(nr_bad_trials / tracker_bins.size * 100, 1)
        else:
            warnings.warn('No eye tracker data found, \n'
                        'skipping eye tracker based exclusion')
            perc_tracker = 'no tracker data found'
            tracker_bins = np.full(beh.shape[0], np.nan)
 

    # apply step algorhytm to trials with missing data
    nan_idx = np.where(np.isnan(tracker_bins) > 0)[0]
    if nan_idx.size > 0 and eye_dict['use_eog']:
        eye_ch = eye_dict['eye_ch']
        eog = epochs._data[nan_idx,epochs.ch_names.index(eye_ch),window_idx]
        if 'step_param' not in eye_dict:
            size, step, thresh = (200, 10, 15e-6)
        else:
            size, step, thresh = eye_dict['step_param']
        idx_art = eog_filt(eog,sfreq = epochs.info['sfreq'], windowsize = size, 
                                windowstep = step, thresh = thresh)
        tracker_bins[nan_idx[idx_art]] = 2
        perc_eog = np.round(sum(tracker_bins == 2)/ tracker_bins.size*100,1)
        print('{} trials missing eyetracking'.format(len(nan_idx)))
        print('data (used eog instead)')
    else:
        perc_eog = 'eog not used for exclusion'

    perc_eye = np.round(sum(tracker_bins >= 1)/ tracker_bins.size*100,1)
    # if it exists update preprocessing information
    if os.path.isfile(preproc_file):
        print('Eye exclusion info saved in preprocessing '
              f'file (session {session})')
        idx = (sj, session)
        preproc_df = pd.read_csv(preproc_file, index_col=[0,1],
                                 on_bad_lines='skip')
        preproc_df = preproc_df.sort_index()  
        preproc_df.loc[idx,'% tracker'] = f'{perc_tracker}%' 
        preproc_df.loc[idx,'% eog'] = f'{perc_eog}% (N = {nan_idx.size})'
        preproc_df.loc[idx,'eye_excl'] = f'{perc_eye}%' 

        # save datafile
        preproc_df.to_csv(preproc_file)

    # remove trials from behavior and eeg
    if 'level_0' not in df:
        df.reset_index(inplace = True)
    else:
        df.drop('level_0', axis=1, inplace=True)
        df.reset_index(inplace = True, drop = True)
    to_drop = np.where(tracker_bins >= 1 )[0]	
    # Suppress verbose epoch drop output
    with open(os.devnull, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            epochs.drop(to_drop, reason='eye detection')
    df.drop(to_drop, inplace = True, axis = 0)
    df.reset_index(inplace = True, drop = True)

    return df, epochs


def bin_tracker_angles(
    angles: np.ndarray,
    thresh: float,
    min_samp: float
) -> np.ndarray:
    """Classify trials by fixation breaks from eye tracker data.

    Summarizes eye tracker gaze data per trial, marking trials that
    contain sustained deviations from fixation exceeding a threshold.

    Parameters
    ----------
    angles : np.ndarray
        Deviation from fixation in visual degrees, 
        shape (n_trials, n_times).
    thresh : float
        Maximum deviation threshold in visual degrees. Trials exceeding
        this value are marked for exclusion.
    min_samp : float
        Minimum number of consecutive samples that must exceed threshold
        to count as a fixation break. Filters out brief tracker noise.

    Returns
    -------
    tracker_bins : np.ndarray
        Binary array indicating trial status:
            - 0: Good fixation (below threshold)
            - 1: Fixation break detected
            - nan: Missing eye tracker data

    Notes
    -----
    The function identifies contiguous segments where gaze deviation
    exceeds threshold. Only segments longer than min_samp trigger
    exclusion.

    Examples
    --------
    >>> # Mark trials with >2° deviation for >40ms at 1000Hz
    >>> bins = bin_tracker_angles(
    ...     angles=gaze_deviation,
    ...     thresh=2.0,
    ...     min_samp=40  # samples
    ... )
    >>> print(f'{bins.sum()} trials marked for exclusion')

    See Also
    --------
    exclude_eye : Main exclusion function using this classifier
    """

    #TODO: how to deal with trials without data
    tracker_bins = []
    for i, angle in enumerate(angles):
        # get data where deviation from fix is larger than thresh
        binned = np.where(angle > thresh)[0]
        segments = np.split(binned, np.where(np.diff(binned) != 1)[0]+1)

        # check whether a segment exceeds min duration
        if np.where(np.array([s.size for s in segments])>min_samp)[0].size > 0:
            tracker_bins.append(1)
        elif np.any(np.isnan(angle)):
            tracker_bins.append(np.nan)
        else:
            tracker_bins.append(0)

    return np.array(tracker_bins)


def eog_filt(
    eog: np.ndarray,
    sfreq: float,
    windowsize: int = 200,
    windowstep: int = 10,
    thresh: float = 30e-6
) -> np.ndarray:
    """Detect eye movements using sliding window on EOG channel.

    Applies split-half sliding window algorithm to detect rapid voltage
    changes in EOG data indicative of saccades. Marks trials containing
    changes exceeding threshold for exclusion.

    Parameters
    ----------
    eog : np.ndarray
        EOG data of shape (n_epochs, n_times). Voltage values should be
        in Volts (MNE default).
    sfreq : float
        Sampling frequency in Hz.
    windowsize : int, default=200
        Sliding window size in milliseconds.
    windowstep : int, default=10
        Window step size in milliseconds. Smaller values increase
        sensitivity but computation time.
    thresh : float, default=30e-6
        Voltage threshold in Volts (30 µV default). If voltage change
        between window halves exceeds this, trial is marked.

    Returns
    -------
    eye_trials : np.ndarray
        Indices of epochs marked for rejection due to eye movements.

    Notes
    -----
    The algorithm:
    1. Slides window across each epoch in specified steps
    2. Compares mean voltage of first vs second half of window
    3. Marks entire trial if any window exceeds threshold

    Final samples may not be analyzed if epoch length is not evenly
    divisible by window parameters.

    Examples
    --------
    >>> # Detect saccades in HEOG with 200ms window
    >>> heog_data = epochs.get_data(picks=['HEOG'])
    >>> bad_trials = eog_filt(
    ...     eog=heog_data[:, 0, :],  # Remove channel dimension
    ...     sfreq=512,
    ...     windowsize=200,
    ...     windowstep=10,
    ...     thresh=30e-6
    ... )
    >>> print(f'Found {len(bad_trials)} trials with saccades')

    See Also
    --------
    exclude_eye : Main exclusion function using this detector
    bin_tracker_angles : Alternative using eye tracker data
    """

    # shift miliseconds to samples
    windowstep /= 1000.0 / sfreq
    windowsize /= 1000.0 / sfreq
    s, e = 0, eog.shape[-1]

    # create multiple windows based on window parameters 
    # (accept that final samples of epoch may not be included) 
    window_idx = [(i, i + int(windowsize)) 
                    for i in range(s, e, int(windowstep)) 
                    if i + int(windowsize) < e]

    # loop over all epochs and store all eye events into a list
    eye_trials = []
    for i, x in enumerate(eog):
        
        for idx in window_idx:
            window = x[idx[0]:idx[1]]

            w1 = np.mean(window[:int(window.size/2)])
            w2 = np.mean(window[int(window.size/2):])

            if abs(w1 - w2) > thresh:
                eye_trials.append(i)
                break

    eye_trials = np.array(eye_trials, dtype = int)			

    return eye_trials
