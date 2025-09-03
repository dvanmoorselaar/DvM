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

def preproc_eeg(sj:int,session:int,eeg_runs:list,nr_sessions:int,eog:list,
                ref:list,t_min:float,t_max:float,event_id:list,montage:str,
                preproc_param:dict,project_folder:str,sj_info:dict,
                eye_info:dict,beh_oi:list,trigger_header:str='trigger', 
                flt_pad:float=0.5,binary:int = 0,
                preproc_name:str='main',nr_sjs:int=24,excl_factor:dict = None):

    # check subject specific parameters
    if str(sj) in sj_info.keys():
        sj_info = sj_info[str(sj)]
    else:
        sj_info = {'bad_chs': []}

    # initiate report
    report_name = f'sj_{sj}_ses_{session}.html'
    report_file = FS.folder_tracker(ext=['preprocessing', 'report', 
                                    preproc_name], 
                                    fname=report_name)
    report = mne.Report(title='preprocessing overview', 
                        subject = f'{sj}_{session}')

    # READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME 
    EEG = mne.concatenate_raws([RawEEG(FS.folder_tracker(ext=['raw_eeg'], 
                     fname=f'subject_{sj}_session_{session}_{run}.bdf'),
                     preload=True, eog=eog) for run in eeg_runs])
            
    EEG.info['bads'] = sj_info['bad_chs'] if type(sj_info['bad_chs']) \
                    == list else sj_info['bad_chs'][f'session_{session}']

    #EEG.replaceChannel(sj, session, replace)
    to_remove = sj_info['ch_remove'] if 'ch_remove' \
                in sj_info.keys() else ['EXG7','EXG8']
    EEG.rereference(ref_channels=ref, change_voltage=False, 
                    to_remove = to_remove)

    EEG.configure_montage(montage=montage)

    # get epoch triggers
    events = EEG.select_events(event_id,binary=binary, min_duration=0)

    #FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
    if preproc_param['run_ica']:
        EEG_ica = EEG.copy().filter(l_freq=1., h_freq=None)
        if preproc_param['notch']:
            EEG_ica.notch_filter(freqs=[50,100,150], 
                 method='fir', phase='zero')

    if preproc_param['high_pass']:                         
        EEG.filter(h_freq=None, l_freq=preproc_param['high_pass'], 
                   fir_design='firwin',
                    skip_by_annotation='edge')
        
    # report raw
    report = EEG.report_raw(report, events, event_id)
    report.save(report_file, overwrite = True)

    if preproc_param['notch']:
        EEG.notch_filter(freqs=[50], 
                 method='fir', phase='zero')
        report = EEG.report_raw(report, events, event_id)
        report.save(report_file, overwrite = True)

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

    report.add_html(report_str, title = 'Linking events to behavior')
    report.add_epochs(epochs, title='initial epoch', psd = True)
    report.save(report_file, overwrite = True)

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
                                method = 'picard', threshold = 0.9, 
                                report  = report, report_path = report_file)
        del EEG_ica, epochs_ica

    # START AUTOMATIC ARTEFACT REJECTION 
    if preproc_param['run_autoreject']:
        drop_bads = preproc_param['drop_bads'] if 'drop_bads' in \
                                                preproc_param.keys() else True
        epochs,z_thresh,report = AR.auto_repair_noise(epochs, sj, session,
                                                      drop_bads,report=report)
        report.save(report_file, overwrite = True)
    else:
        z_thresh = 0

    # INTERPOLATE BADS
    bads = epochs.info['bads']   
    epochs.interpolate_bads(reset_bads=True, mode='accurate')

    report.add_epochs(epochs, title='Epochs after artefact reject',psd = True)    
    report.save(report_file, overwrite = True)

    # save
    epochs.save_preprocessed(preproc_name)
    log_file = FS.folder_tracker(ext=['preprocessing', 'group_info'], 
                     fname=f'preproc_param_{preproc_name}.csv')
    to_update = dict(nr_clean = len(epochs), z_thresh = z_thresh, 
                    nr_bads = len(bads), bad_el = bads)
    log_preproc((sj, session),log_file,nr_sj=nr_sjs, 
                nr_sessions=nr_sessions,to_update = to_update)