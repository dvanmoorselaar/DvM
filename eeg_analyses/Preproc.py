"""
analyze EEG data

Created by Dirk van Moorselaar on 8-12-2021.
Copyright (c) 2021 DvM. All rights reserved.
"""

import logging

from eeg_analyses.EEG import *
from support.FolderStructure import *
from IPython import embed

def preproc_eeg(sj: int, session: int, eeg_runs: list, nr_sessions: int, eog: list, ref: list, t_min: float, 
                t_max: float, event_id: list, preproc_param: dict, project_folder: str, sj_info: dict, eye_info: dict, 
                project_param: list, trigger_header: str = 'trigger', flt_pad: float = 0.5, binary: int = 0, 
                channel_plots: bool = True, inspect: bool = True):

    # set subject specific parameters
    file = 'subject_{}_session_{}_'.format(sj, session)
    replace = sj_info[str(sj)]['replace']
    log_file = FolderStructure.FolderTracker(extension=['preprocessing', 'group_info'], 
                    filename='preprocessing_param.csv')

    # start logging
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M',
                filename= FolderStructure.FolderTracker(extension=['preprocessing', 'group_info', 'sj_logs'], 
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
    if preproc_param['run_ica']:
        EEGica = EEG.copy()
        EEGica.filter(h_freq=None, l_freq=1.5,
                                fir_design='firwin', skip_by_annotation='edge')

    if preproc_param['high_pass']:                         
        EEG.filter(h_freq=None, l_freq=preproc_param['high_pass'], fir_design='firwin',
                skip_by_annotation='edge')

    # EPOCH DATA
    events = EEG.eventSelection(event_id, binary=binary, min_duration=0)
    epochs = Epochs(sj, session, EEG, events, event_id=event_id,
            tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, reject_by_annotation = False) 

    # MATCH BEHAVIOR FILE
    bdm_remove = sj_info[str(sj)]['bdf_remove'] if 'bdf_remove' in sj_info[str(sj)].keys() else None
    beh, missing = epochs.align_behavior(events, trigger_header = trigger_header, headers = project_param, bdf_remove = bdm_remove)

    # ICA
    if preproc_param['run_ica']: 
        epochs.applyICA(EEG, EEGica, method='picard', fit_params = dict(ortho=False, extended=True), inspect = True)
        del EEGica

    # # AUTOMATED ARTIFACT DETECTION
    # epochs.selectBadChannels(run_ransac = True, channel_plots = False, inspect = True, RT = None)  
    z = epochs.automatic_artifact_detection(z_thresh=4, band_pass=[110, 140], plot=True, inspect=True)

    # EYE MOVEMENTS
    epochs.detectEye(missing, events, beh.shape[0], time_window=(t_min*1000, t_max*1000), 
                    tracker_shift = eye_info['tracker_shift'], start_event = eye_info['start_event'], 
                    extension = eye_info['tracker_ext'], eye_freq = eye_info['eye_freq'], 
                    screen_res = eye_info['screen_res'], viewing_dist = eye_info['viewing_dist'], 
                    screen_h = eye_info['screen_h'])

    # INTERPOLATE BADS
    bads = epochs.info['bads']   
    epochs.interpolate_bads(reset_bads=True, mode='accurate')

    # LINK BEHAVIOR
    epochs.link_behavior(beh)

    logPreproc((sj, session), log_file, nr_sj = len(sj_info.keys()), nr_sessions = nr_sessions, 
                to_update = dict(nr_clean = len(epochs), z_value = z, nr_bads = len(bads), bad_el = bads))