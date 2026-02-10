"""
EEG Preprocessing Pipeline.

This module provides a standardized preprocessing pipeline for EEG data
analysis in the DvM toolbox. The pipeline implements a comprehensive
workflow from raw BDF files to cleaned, epoched data ready for analysis,
following best practices in EEG preprocessing.

The preprocessing pipeline integrates multiple processing steps 
including filtering, epoching, behavioral data alignment, eye-tracking 
integration,ICA-based artefact removal, and automatic bad epoch 
detection. All stepsare documented in an HTML quality control report.

Key Features
------------
- Automated reading and concatenation of multi-run EEG sessions
- Flexible rereferencing with bad channel handling
- High-pass filtering and notch filtering for line noise removal
- Event-based epoching with filter padding to avoid edge artefacts
- Behavioral data alignment with trigger matching
- Eye-tracking data integration and alignment
- ICA-based blink component removal
- Automatic artefact rejection using autoreject algorithm
- Bad channel interpolation
- Comprehensive HTML quality control reports
- Group-level preprocessing statistics logging

Typical Usage
-------------
The main function `eeg_preprocessing_pipeline()` is designed to be 
called in a loop across subjects and sessions. Users define 
preprocessing parameters once and apply them consistently across the 
entire dataset.

Created by Dirk van Moorselaar on 8-12-2021.
Copyright (c) 2021 DvM. All rights reserved.
"""

from typing import Optional

from analysis.EEG import *
from support.FolderStructure import FolderStructure as FS
from support.preprocessing_utils import log_preproc, format_subject_id, find_raw_files

def eeg_preprocessing_pipeline(
    sj: int,
    session: int,
    eeg_runs: list[int],
    nr_sessions: int,
    eog: list[str],
    ref: list[str],
    t_min: float,
    t_max: float,
    event_id: list[int],
    montage: str,
    preproc_param: dict,
    sj_info: dict,
    eye_info: dict,
    beh_oi: list[str],
    trigger_header: str = 'trigger',
    flt_pad: float = 0.5,
    binary: int = 0,
    preproc_name: str = 'main',
    nr_sjs: int = 24,
    excl_factor: Optional[dict] = None
) -> None:
    """
    Preprocess EEG data with optional ICA and automatic artefact 
    rejection.

    This function performs a complete preprocessing pipeline including:
    raw data reading, rereferencing, filtering, epoching, behavioral
    data alignment, ICA-based blink removal, automatic artefact
    rejection, and bad channel interpolation. Progress is documented
    in an HTML report.

    Parameters
    ----------
    sj : int
        Subject number to preprocess.
    session : int
        Session number to preprocess.
    eeg_runs : list of int
        Run numbers within the session (e.g., [1] or [1, 2, 3]).
    nr_sessions : int
        Total number of sessions per subject in the experiment.
    eog : list of str
        EOG channel names (e.g., ['V_up', 'V_do', 'H_r', 'H_l']).
    ref : list of str
        Reference channel names (e.g., ['Ref_r', 'Ref_l']).
    t_min : float
        Start time of epochs in seconds relative to event (e.g., -0.2).
    t_max : float
        End time of epochs in seconds relative to event (e.g., 0.5).
    event_id : list of int
        Trigger values to include in epoching
        (e.g., list(range(1, 250))).
    montage : str
        EEG montage name (e.g., 'biosemi32', 'biosemi64').
    preproc_param : dict
        Preprocessing parameters with keys:
        - 'high_pass' : float or None - High-pass filter cutoff in
          Hz (e.g., 0.01)
        - 'notch' : bool - Whether to apply 50/100/150 Hz notch
          filter to remove line noise (frequencies adjusted based on
          sampling rate)
        - 'run_ica' : bool - Whether to run ICA for blink removal
        - 'run_autoreject' : bool - Whether to run automatic artefact
          rejection
        - 'drop_bads' : bool, optional - Whether to drop bad epochs
          (default: True)
    sj_info : dict
        Subject-specific information. Keys are subject numbers (as
        strings). Each value is a dict with:
        - 'bad_chs' : list of str or dict - Bad channel names. Can
          be a list (same for all sessions) or dict with 'session_X'
          keys
        - 'ch_remove' : list of str, optional - Channels to remove
          (default: ['EXG7', 'EXG8'])
        - 'bdf_remove' : list of int, optional - Indices of triggers
          to remove when aligning with behavior
    eye_info : dict
        Eye-tracking parameters with keys:
        - 'tracker_ext' : str - Eye tracker file extension
          (e.g., 'asc', 'edf')
        - 'sfreq' : int - Sampling frequency of eye tracker in Hz
        - 'trigger_msg' : str - Message in eye tracker file marking
          trial onset
        - 'window_oi' : tuple of float - Time window for eye data
          extraction (ms)
        - 'start' : str - Message marking trial start
        - 'drift_correct' : tuple of float or None - Drift
          correction window
        - 'viewing_dist' : float - Viewing distance in cm
        - 'screen_res' : tuple of int - Screen resolution
          (width, height) in pixels
        - 'screen_h' : float - Screen height in cm
        - 'eog' : list of str - EOG channel names (same as eog
          parameter)
    beh_oi : list of str
        Behavioral column names to include in epoch metadata
        (e.g., ['nr_trials', 'display_trigger', 'RT', 'correct',
        'block_type']).
    trigger_header : str, default='trigger'
        Column name in behavioral file containing trigger values.
    flt_pad : float, default=0.5
        Temporal padding in seconds added to epochs for filtering,
        then removed to avoid edge artefacts.
    binary : int, default=0
        Whether triggers are binary coded (0 = no, 1+ = yes).
    preproc_name : str, default='main'
        Preprocessing pipeline identifier used in file naming.
    nr_sjs : int, default=24
        Total number of subjects in the experiment.
    excl_factor : dict or None, default=None
        Behavioral conditions to exclude from preprocessing. Keys are
        column names, values are lists of values to exclude
        (e.g., {'practice': ['yes'], 'correct': [0]}).

    Returns
    -------
    None
        Saves preprocessed epochs to disk, generates HTML report, and 
        logs preprocessing statistics to a group-level CSV file.

    Notes
    -----
    File Structure Requirements:
    - Raw EEG files should be named:
      'subject_{sj}_session_{session}_{run}.bdf'
    - Behavioral files should follow project-specific naming
    - Eye tracker files should match behavioral trial structure
    - All file paths are managed automatically by the FolderStructure
      class based on the current working directory

    Preprocessing Steps:
    1. Read and concatenate raw EEG data from all runs
    2. Mark bad channels and rereference to specified electrodes
    3. Apply high-pass filter (if specified)
    4. Apply notch filter at 50/100/150 Hz to remove line noise
       (if enabled)
    5. Epoch data around event triggers with temporal padding
    6. Align epochs with behavioral data and eye-tracking
    7. Run ICA to detect and remove blink components (if enabled)
    8. Apply automatic artefact rejection (if enabled)
    9. Interpolate bad channels
    10. Save preprocessed epochs, generate report, and log
        statistics

    Output Files:
    - Preprocessed epochs:
      'sj_{sj}_ses_{session}_{preproc_name}-epo.fif'
      Contains the cleaned epoched data with metadata
    - HTML report: 'sj_{sj}_ses_{session}.html'
      Visual quality control report with raw data plots, PSD, and
      artefact detection results
    - Group log CSV: 'preproc_param_{preproc_name}.csv'
      Aggregates preprocessing statistics across all subjects
      including:
      * Number of clean epochs remaining after rejection
        (nr_clean)
      * Z-threshold used for artefact detection (z_thresh)
      * Number of bad channels interpolated (nr_bads)
      * List of bad channel names (bad_el)
      This file is updated incrementally as each subject is
      processed.

    Examples
    --------
    Basic preprocessing setup for a visual search experiment:

    >>> # Define subject-specific bad channels
    >>> sj_info = {
    ...     '1': {'bad_chs': []},
    ...     '2': {'bad_chs': ['C4', 'CP2']},
    ...     '3': {'bad_chs': ['P3', 'P7']}
    ... }
    >>> 
    >>> # Set preprocessing parameters
    >>> preproc_param = {
    ...     'high_pass': 0.01,
    ...     'run_ica': True,
    ...     'run_autoreject': True,
    ...     'notch': True
    ... }
    >>> 
    >>> # Define electrode configuration
    >>> eog = ['V_up', 'V_do', 'H_r', 'H_l']
    >>> ref = ['Ref_r', 'Ref_l']
    >>> 
    >>> # Set epoching parameters
    >>> event_id = list(range(1, 250))
    >>> t_min, t_max = -0.2, 0.5
    >>> flt_pad = 0.5
    >>> 
    >>> # Define behavioral columns to include
    >>> beh_oi = ['nr_trials', 'display_trigger', 'RT', 'correct',
    ...           'block_type', 'target_loc', 'dist_loc']
    >>> 
    >>> # Eye-tracking configuration
    >>> eye_info = {
    ...     'tracker_ext': 'asc',
    ...     'sfreq': 1000,
    ...     'trigger_msg': 'Onset search',
    ...     'window_oi': (-700, 1000),
    ...     'start': 'start trial',
    ...     'drift_correct': None,
    ...     'viewing_dist': 70,
    ...     'screen_res': (1920, 1080),
    ...     'screen_h': 29,
    ...     'eog': eog
    ... }
    >>> 
    >>> # Preprocess all subjects
    >>> for sj in [1, 2, 3]:
    ...     eeg_preprocessing_pipeline(
    ...         sj=sj, session=1, eog=eog, ref=ref,
    ...         eeg_runs=[1], nr_sessions=1,
    ...         t_min=t_min, t_max=t_max, flt_pad=flt_pad,
    ...         sj_info=sj_info, eye_info=eye_info,
    ...         event_id=event_id, montage='biosemi32',
    ...         preproc_param=preproc_param,
    ...         trigger_header='display_trigger',
    ...         beh_oi=beh_oi,
    ...         binary=0, preproc_name='main',
    ...         nr_sjs=24, excl_factor=None
    ...     )

    See Also
    --------
    RAW : Raw EEG data class with rereferencing and channel operations
    Epochs : Epoched EEG data class with metadata alignment
    ArtefactReject : Automatic artefact detection and repair
    """

    # check subject specific parameters
    if str(sj) in sj_info.keys():
        sj_info = sj_info[str(sj)]
    else:
        sj_info = {'bad_chs': []}

    # initiate report with zero-padded subject and session IDs
    sj_fmt = format_subject_id(sj)
    ses_fmt = format_subject_id(session)
    report_name = f'sj_{sj_fmt}_ses_{ses_fmt}.html'
    report_file = FS.folder_tracker(ext=['preprocessing', 'report', 
                                    preproc_name], 
                                    fname=report_name)
    report = mne.Report(title='preprocessing overview', 
                        subject = f'{sj}_{session}')

    # READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME 
    base_folder = FS.folder_tracker(ext=['eeg', 'raw'], fname='')
    
    raw_files = []
    for run in eeg_runs:
        files = find_raw_files(base_folder, sj, session, run, ext='bdf')
        if not files:
            run_str = f' (run {run})' if len(eeg_runs) > 1 else ''
            raise FileNotFoundError(
                f"No BDF file found for subject {sj}, session {session}{run_str}\n"
                f"Searched in: {base_folder}\n"
                f"Expected file pattern: sub_*_ses_*.bdf or sub_*_ses_*_run_*.bdf"
            )
        raw_files.append(files[0])
    
    EEG = mne.concatenate_raws([
        RAW(fpath, eog=eog) for fpath in raw_files
    ])
            
    EEG.info['bads'] = sj_info['bad_chs'] if type(sj_info['bad_chs']) \
                    == list else sj_info['bad_chs'][f'session_{session}']

    #EEG.replace_channel(replace)
    to_remove = sj_info['ch_remove'] if 'ch_remove' \
                in sj_info.keys() else ['EXG7','EXG8']
    EEG.rereference(ref_channels=ref, change_voltage=False, 
                    to_remove = to_remove)

    EEG.configure_montage(montage=montage)

    # get epoch triggers
    events = EEG.select_events(event_id=event_id, binary=binary, 
                               min_duration=0)

    #FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
    if preproc_param['run_ica']:
        EEG_ica = EEG.copy().filter(l_freq=1., h_freq=None)
        if preproc_param['notch']:
            freqs = [i for i in [50,100,150] if i < EEG.info['sfreq'] / 2]
            EEG_ica.notch_filter(freqs=freqs, 
                 method='fir', phase='zero')

    if preproc_param['high_pass']:                         
        EEG.filter(h_freq=None, l_freq=preproc_param['high_pass'], 
                   fir_design='firwin',
                    skip_by_annotation='edge')
        
    # report raw
    report = EEG.report_raw(report, events, event_id)
    report.save(report_file, overwrite=True, open_browser=False)

    if preproc_param['notch']:
        freqs = [i for i in [50,100,150] if i < EEG.info['sfreq'] / 2]
        EEG.notch_filter(freqs=freqs, 
                 method='fir', phase='zero')
        # Add PSD after notch filtering without duplicating events
        report.add_raw(EEG, title='After Notch Filter', psd=True)
        report.save(report_file, overwrite = True, open_browser=False)

    # EPOCH DATA 
    epochs = Epochs(sj, session, EEG, events, event_id=event_id,
            tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, 
            reject_by_annotation = False) 
    
    # MATCH BEHAVIOR FILE
    idx_remove = sj_info['bdf_remove'] if 'bdf_remove' \
                                            in sj_info.keys() else None
    missing, report_str = epochs.align_meta_data(events,trigger_header, 
                                                beh_oi=beh_oi,
                                                idx_remove=idx_remove,
                                                eye_inf = eye_info,
                                                del_practice=True,
                                                excl_factor=excl_factor)

    report.add_html(report_str, title='Linking events to behavior')
    report.add_epochs(epochs, title='initial epoch', psd=True)
    report.save(report_file, overwrite=True, open_browser=False)

    # ICA
    AR = ArtefactReject(z_thresh = 4, max_bad = 5, flt_pad = epochs.flt_pad, 
                        filter_z = True)
    if preproc_param['run_ica']: 
        epochs_ica = Epochs(sj, session, EEG_ica, events, event_id=event_id,
                    tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, 
                    reject_by_annotation = False) 
        _, _ = epochs_ica.align_meta_data(events,trigger_header, 
                                                beh_oi=beh_oi,
                                                idx_remove=idx_remove,
                                                eye_inf = eye_info,
                                                del_practice=True,
                                                excl_factor=excl_factor)   
        epochs = AR.run_blink_ICA(epochs_ica, EEG, epochs, sj, session, 
                                method='picard', threshold=0.9, 
                                report=report, report_path=report_file)
        del EEG_ica, epochs_ica

    # START AUTOMATIC ARTEFACT REJECTION 
    if preproc_param['run_autoreject']:
        drop_bads = preproc_param['drop_bads'] if 'drop_bads' in \
                                                preproc_param.keys() else True
        epochs, z_thresh, report = AR.auto_repair_noise(epochs, sj, session,
                                                      drop_bads, report=report)
        report.save(report_file, overwrite=True, open_browser=False)
    else:
        z_thresh = 0

    # INTERPOLATE BADS
    bads = epochs.info['bads']   
    epochs.interpolate_bads(reset_bads=True, mode='accurate')

    report.add_epochs(epochs, title='Epochs after artefact reject', psd=True)    
    report.save(report_file, overwrite=True, open_browser=False)

    # save
    epochs.save_preprocessed(preproc_name)
    log_file = FS.folder_tracker(ext=['preprocessing', 'group_info'], 
                     fname=f'preproc_param_{preproc_name}.json')
    to_update = dict(nr_clean=len(epochs), z_thresh=z_thresh, 
                    nr_bads=len(bads), bad_el=bads)
    log_preproc((sj, session), log_file, nr_sj=nr_sjs, 
                nr_sessions=nr_sessions, to_update=to_update)