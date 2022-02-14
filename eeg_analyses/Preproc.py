"""
analyze EEG data

Created by Dirk van Moorselaar on 8-12-2021.
Copyright (c) 2021 DvM. All rights reserved.
"""

import logging

from eeg_analyses.EEG import *
from support.FolderStructure import FolderStructure as FS
from support.support import log_preproc
from IPython import embed

def preproc_eeg(sj: int, session: int, eeg_runs: list, nr_sessions: int, eog: list, ref: list, t_min: float, 
                t_max: float, event_id: list, preproc_param: dict, project_folder: str, sj_info: dict, eye_info: dict, 
                beh_oi: list, trigger_header: str = 'trigger', flt_pad: float = 0.5, binary: int = 0,
                preproc_name: str = 'main', nr_sjs: int = 24):

    # check subject specific parameters
    sj_info = sj_info[str(sj)] if str(sj) in sj_info.keys() else {'bad_chs': []}
    
    # initiate report
    report_file = FS.FolderTracker(extension=['preprocessing', 'report', preproc_name], 
                     filename=f'sj_{sj}_ses_{session}.html')
    report = mne.Report(title='preprocessing overview', subject = f'{sj}_{session}')

    # READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME 
    EEG = mne.concatenate_raws([RawBDF(FS.FolderTracker(extension=['raw_eeg'], 
                     filename=f'subject_{sj}_session_{session}_{run}.bdf'),
                     preload=True, eog=eog) for run in eeg_runs])
    EEG.info['bads'] = sj_info['bad_chs']
         
    #EEG.replaceChannel(sj, session, replace)
    EEG.reReference(ref_channels=ref, vEOG=eog[
                    :2], hEOG=eog[2:], changevoltage=False, 
                    to_remove = ['EXG7','EXG8'])
    EEG.setMontage(montage='biosemi64')

    # get epoch triggers
    events = EEG.eventSelection(event_id, binary=binary, min_duration=0)

    #FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
    if preproc_param['run_ica']:
        EEG_ica = EEG.copy().filter(l_freq=1., h_freq=None)

    if preproc_param['high_pass']:                         
        EEG.filter(h_freq=None, l_freq=preproc_param['high_pass'], fir_design='firwin',
                skip_by_annotation='edge')

    # report raw
    
    report = EEG.report_raw(report, events, event_id)
    report.save(report_file, overwrite = True)

    # EPOCH DATA 
    epochs = Epochs(sj, session, EEG, events, event_id=event_id,
            tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, reject_by_annotation = False) 

    # ICA
    AR = ArtefactReject(z_thresh = 4, max_bad = 5, flt_pad = epochs.flt_pad, filter_z = True)
    if preproc_param['run_ica']: 
        epochs_ica = Epochs(sj, session, EEG_ica, events, event_id=event_id,
            tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, reject_by_annotation = False) 
        epochs = AR.run_blink_ICA(epochs_ica, EEG, epochs, sj, session, 
                                method = 'picard', threshold = 0.9, 
                                report  = report, report_path = report_file)
        del EEG_ica, epochs_ica

    # MATCH BEHAVIOR FILE
    bdf_remove = sj_info[str(sj)]['bdf_remove'] if 'bdf_remove' in sj_info.keys() else None
    missing, report_str = epochs.align_behavior(events, trigger_header = trigger_header, headers = beh_oi, bdf_remove = bdf_remove)
    report.add_html(report_str, title = 'Linking events to behavior')
    report.add_epochs(epochs, title='initial epoch')
    report.save(report_file, overwrite = True)

    # LINK EYE MOVEMENTS
    #epochs.link_eye(eye_info, missing, vEOG=eog[:2], hEOG=eog[2:])

    # START AUTOMATIC ARTEFACT REJECTION 
    epochs, z_thresh, report = AR.auto_repair_noise(epochs, report = report)
    report.save(report_file, overwrite = True)

    # INTERPOLATE BADS
    bads = epochs.info['bads']   
    epochs.interpolate_bads(reset_bads=True, mode='accurate')

    report.add_epochs(epochs, title='Epochs after artefact reject')    
    report.save(report_file, overwrite = True)

    # epochs.detectEye(missing, events, epochs.metadata.shape[0], time_window=(t_min*1000, t_max*1000), 
    #                 tracker_shift = eye_info['tracker_shift'], start_event = eye_info['start_event'], 
    #                 extension = eye_info['tracker_ext'], eye_freq = eye_info['eye_freq'], 
    #                 screen_res = eye_info['screen_res'], viewing_dist = eye_info['viewing_dist'], 
    #                 screen_h = eye_info['screen_h'])

    # save
    epochs.save_preprocessed(preproc_name)
    log_file = FS.FolderTracker(extension=['preprocessing', 'group_info'], 
                     filename=f'preproc_param_{preproc_name}.csv')
    log_preproc((sj, session), log_file, nr_sj = nr_sjs, nr_sessions = nr_sessions, 
                to_update = dict(nr_clean = len(epochs), z_thresh = z_thresh, nr_bads = len(bads), bad_el = bads))