"""
File Management and Data Loading for the DvM Toolbox.

This module provides the FolderStructure class which handles all file
operations within the DvM toolbox. It manages automatic folder creation,
path generation, and provides standardized methods for loading
preprocessed EEG data, behavioral data, ERPs, time-frequency data, and
multivariate decoding results.

The FolderStructure class uses the current working directory as a base
and automatically creates subdirectory structures for different data
types (raw, processed, behavioral, ERP, TFR, BDM, etc.). This ensures
consistent organization across all projects using the toolbox.

Key Features
------------
- Automatic folder creation and path management
- Standardized file naming conventions
- Loading preprocessed EEG epochs with metadata
- Reading ERP, TFR, and BDM analysis results
- Subject and condition-based data organization
- Cross-analysis file matching and validation

Typical Usage
-------------
The FolderStructure class is typically inherited by other analysis
classes (EEG, Epochs, ERP, TFR, BDM, CTF) to provide file management
functionality. It can also be used standalone for loading previously
analyzed data.

Created by Dirk van Moorselaar on 8-12-2021.
Copyright (c) 2021 DvM. All rights reserved.
"""

import os
import sys
import mne
import pickle
import glob
import re
import copy

import numpy as np
import pandas as pd
from IPython import embed
from contextlib import redirect_stdout

from typing import List, Optional, Union, Tuple
from support.preprocessing_utils import (
    match_epochs_times,
    trial_exclusion,
    format_subject_id
)
from support.eye_utils import exclude_eye

def blockPrinting(func):
    """Decorator to suppress console output during function execution.

    Parameters
    ----------
    func : callable
        Function to wrap with output suppression.

    Returns
    -------
    func_wrapper : callable
        Wrapped function that suppresses stdout during execution.

    Notes
    -----
    This decorator redirects stdout to devnull during function 
    execution,then restores normal output. Useful for silencing verbose 
    library output when not needed.
    """
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

class FolderStructure(object):
    """Manage file operations and data loading for EEG analyses.

    This class provides standardized file management for the DvM 
    toolbox. It handles automatic folder creation, path generation based 
    on the current working directory, and provides methods for loading 
    various types of processed data (EEG epochs, ERPs, TFR, 
    BDM, CTF results).

    The class uses a consistent folder structure convention:
    - eeg/raw/ : Raw EEG BDF/EDF files
    - eeg/processed/ : Preprocessed EEG epoch files (-epo.fif)
    - behavioral/raw/ : Raw behavioral CSV files
    - behavioral/processed/ : Processed behavioral data
    - eye/raw/ : Eye tracker data files
    - eye/processed/ : Processed eye tracking data
    - erp/evoked/ : Evoked response files
    - tfr/[method]/ : Time-frequency analysis results
    - bdm/[analysis]/ : Multivariate decoding results
    - ctf/[analysis]/ : Channel tuning function results
    - preprocessing/report/ : HTML quality control reports
    - preprocessing/group_info/ : Group-level statistics

    Methods
    -------
    folder_tracker(ext, fname, overwrite)
        Generate folder paths and create directories if needed.
    load_processed_epochs(sj, fname, preproc_name, eye_dict, ...)
        Load preprocessed EEG epochs with behavioral metadata.
    read_raw_beh(sj, session, files)
        Read raw behavioral CSV files.
    read_erps(erp_name, cnds, sjs, match)
        Load evoked response data.
    read_tfr(tfr_folder_path, tfr_name, cnds, sjs)
        Load time-frequency analysis results.
    read_bdm(bdm_folder_path, bdm_name, sjs, analysis_labels)
        Load multivariate decoding results.
    read_ctfs(ctf_folder_path, output_type, ctf_name, sjs)
        Load channel tuning function results.

    Notes
    -----
    This class is designed to be inherited by analysis classes
    (EEG, Epochs, ERP, TFR, BDM, CTF) to provide them with file
    management capabilities.

    Examples
    --------
    >>> # Standalone usage
    >>> fs = FolderStructure()
    >>> 
    >>> # Load preprocessed epochs
    >>> df, epochs = fs.load_processed_epochs(
    ...     sj=1, fname='ses_1_main', preproc_name='main'
    ... )
    >>> 
    >>> # Load ERP data for multiple subjects
    >>> erps, times = fs.read_erps(
    ...     erp_name='target_locked',
    ...     cnds=['left', 'right'],
    ...     sjs=[1, 2, 3]
    ... )

    See Also
    --------
    eeg_analyses.EEG : EEG data classes that inherit FolderStructure
    eeg_analyses.ERP : ERP analysis class
    eeg_analyses.TFR : Time-frequency analysis class
    eeg_analyses.BDM : Multivariate decoding class
    eeg_analyses.CTF : Channel tuning function class
    """

    def __init__(self):
        """Initialize FolderStructure instance.

        Creates a new FolderStructure object that uses the current
        working directory as the base for all file operations.
        """

    @staticmethod
    def _extract_subject_number(fname: str) -> int:
        """Extract subject number from filename.

        Parameters
        ----------
        fname : str
            Filename containing subject number in format 'sub_X_'.

        Returns
        -------
        sj : int
            Extracted subject number.

        Raises
        ------
        ValueError
            If filename does not contain subject number in 
            expected format.
        """
        match = re.search(r'sub_0?(\d+)_', fname)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract subject number from {fname}")

    @staticmethod
    def folder_tracker(
        ext: Optional[list] = None,
        fname: Optional[str] = None,
        overwrite: bool = True
    ) -> str:
        """Generate file path with automatic folder creation.

        Creates a folder path using the current working directory as
        base. If the specified path does not exist, it is created
        automatically. Optionally prevents overwriting existing files.

        Parameters
        ----------
        ext : list, default=[]
            List of subfolders to append to current working directory.
            For example, ['behavioral', 'processed'] creates path
            'cwd/behavioral/processed/'.
        fname : str, optional
            Filename to append to path. If None, only the folder path
            is returned. Default is None.
        overwrite : bool, default=True
            If False and file exists, appends '+' characters to
            filename until a unique name is found. If True, returns
            path that may overwrite existing file.

        Returns
        -------
        path : str
            Complete file path including filename (if specified).

        Examples
        --------
        >>> # Get path to processed folder
        >>> path = FolderStructure.folder_tracker(ext=['processed'])
        >>> 
        >>> # Get path to specific file
        >>> file_path = FolderStructure.folder_tracker(
        ...     ext=['behavioral', 'processed'],
        ...     fname='sub_01_ses_01.csv'
        ... )
        >>> 
        >>> # Prevent overwriting
        >>> safe_path = FolderStructure.folder_tracker(
        ...     ext=['processed'],
        ...     fname='data.fif',
        ...     overwrite=False
        ... )
        """

        # create folder adress
        path = os.getcwd()
        if ext is None:
            ext = []
        if ext != []:
            path = os.path.join(path,*ext)

        # check whether folder exists
        if not os.path.isdir(path):
            os.makedirs(path)

        if fname != '':
            if not overwrite:
                while os.path.isfile(os.path.join(path,fname)):
                    end_idx = len(fname) - fname.index('.')
                    fname = fname[:-end_idx] + '+' + fname[-end_idx:]
            path = os.path.join(path,fname)

        return path

    def load_processed_epochs(self,sj:int,fname:str,preproc_name:str,
                        eye_dict:Optional[dict]=None,beh_file:bool=True,
                        excl_factor:Optional[dict]=None,
                        modality:str='eeg')->Tuple[pd.DataFrame,mne.Epochs]:
        """Load preprocessed epochs with behavioral metadata.

        Reads preprocessed neuroimaging data (MNE Epochs object) and 
        behavioral data for subsequent analyses. Supports both EEG and 
        MEG data. If eye movement criteria are specified, trials are 
        excluded based on fixation breaks. Updates preprocessing 
        overview file with eye movement statistics if it exists.

        Parameters
        ----------
        sj : int
            Subject identifier.
        fname : str
            Name of processed file (without -epo.fif extension).
        preproc_name : str
            Name specified for preprocessing pipeline (used to locate
            preprocessing parameter files).
        eye_dict : dict, optional
            Eye movement exclusion criteria. Default is None (no
            exclusion). Supported keys:
                - 'eye_window' : tuple, time window to search for eye
                  movements
                - 'eye_ch' : str, channel name for eye movement 
                  detection
                - 'angle_thresh' : float, threshold in degrees of visual
                  angle
                - 'step_param' : dict, parameters for step detection
                  algorithm
                - 'use_tracker' : bool, whether to use eye tracker data.
                  If False, uses EOG channel specified in 'eye_ch'
        beh_file : bool, default=True
            If True, reads behavioral data from separate CSV file. If
            False, uses condition information from epochs.events array.
        excl_factor : dict, optional
            Trial exclusion criteria based on experimental conditions.
            Default is None (no exclusion). Format:
                {column_name: [values_to_exclude]}
            For example, to exclude trials where cue pointed right:
                excl_factor = {'cue_direc': ['right']}
            Multiple columns and values can be specified.
        modality : str, default='eeg'
            Neuroimaging modality. Supported values: 'eeg' or 'meg'.
            Determines which folder to load epochs from:
                - 'eeg': loads from eeg/processed/
                - 'meg': loads from meg/processed/

        Returns
        -------
        df : pd.DataFrame
            Behavioral data aligned to epochs. Contains trial
            metadata and experimental variables.
        epochs : mne.Epochs
            Preprocessed epochs object (EEG or MEG) with bad trials 
            rejected.

        Notes
        -----
        If eye_dict is specified and eye tracker data file exists,
        eye movements are detected using tracker data. Otherwise,
        falls back to EOG-based detection using the step algorithm.

        The preprocessing parameter file is updated with eye movement
        exclusion statistics per subject if it exists.

        Examples
        --------
        >>> # Basic loading
        >>> df, epochs = self.load_processed_epochs(
        ...     sj=1,
        ...     fname='main',
        ...     preproc_name='main'
        ... )
        >>> 
        >>> # Load with eye movement exclusion
        >>> eye_params = {
        ...     'eye_window': (-0.5, 1.0),
        ...     'eye_ch': 'HEOG',
        ...     'angle_thresh': 100,
        ...     'use_tracker': True
        ... }
        >>> df, epochs = self.load_processed_epochs(
        ...     sj=1,
        ...     fname='main',
        ...     preproc_name='main',
        ...     eye_dict=eye_params
        ... )
        >>> 
        >>> # Load with condition exclusion
        >>> df, epochs = self.load_processed_epochs(
        ...     sj=1,
        ...     fname='main',
        ...     preproc_name='main',
        ...     excl_factor={'cue_direc': ['right']}
        ... )
        >>> 
        >>> # Load MEG data instead of EEG
        >>> df, epochs = self.load_processed_epochs(
        ...     sj=1,
        ...     fname='main',
        ...     preproc_name='main',
        ...     modality='meg'
        ... )

        See Also
        --------
        support.support.exclude_eye : Eye movement detection and 
        exclusion
        support.support.trial_exclusion : Condition-based trial 
        selection
        """
        
        # Format subject ID with zero-padding
        sj = format_subject_id(sj)
        
        # Load preprocessed epochs from appropriate modality folder
        modality = modality.lower()
        if modality not in ['eeg', 'meg']:
            raise ValueError(
                f"modality must be 'eeg' or 'meg', "
                f"got '{modality}'"
            )
        
        epochs = mne.read_epochs(
            self.folder_tracker(
                ext=[modality, 'processed'],
                fname=f'sub_{sj}_{fname}-epo.fif'
            )
        )

        # check whether metadata is saved alongside epoched eeg
        if epochs.metadata is not None:
            df = copy.copy(epochs.metadata)
        else:
            if beh_file:
                # read in seperate behavior file
                df = pd.read_csv(
                    self.folder_tracker(
                        ext=['behavioral', 'processed'],
                        fname=f'sub_{sj}_{fname}.csv'
                    )
                )
            else:
                df = pd.DataFrame({'condition': epochs.events[:,2]})


        # reset index(to properly align beh and epochs)
        df.reset_index(inplace = True, drop = True)

        # exclude eye movements based on threshold criteria in eye_dict
        if eye_dict is not None:
            file = FolderStructure().folder_tracker(
                            ext = ['preprocessing','group_info'],
                            fname = f'preproc_param_{preproc_name}.csv')
            # Check if the file exists before proceeding
            # Extract session number from fname (format: 'ses_XX_...')
            match = re.search(r'ses_(\d+)', fname)
            session = match.group(1) if match else '1'
            # Eye files are renamed 
            # to sub_{sj}_ses_{session}_{preproc_name}.npz during preprocessing
            eye_file = self.folder_tracker(ext=['eye','processed'],
                    fname=f'sub_{sj}_ses_{session}_{preproc_name}.npz')
            if os.path.isfile(eye_file):
                eye = np.load(eye_file)
                df, epochs = exclude_eye(sj, int(session), df, epochs, 
                                         eye_dict, eye, file)
            else:
                print(f"Warning: Preprocessing parameter file not found: \
                      {file}. Eye exclusion based on EOG data only")
                temp = eye_dict['use_tracker']
                eye_dict['use_tracker'] = False
                df, epochs = exclude_eye(sj, int(session), df, epochs, 
                                        eye_dict, None, file)
                eye_dict['use_tracker'] = temp

        # remove a subset of trials 
        if type(excl_factor) == dict: 
            df, epochs,_ = trial_exclusion(df, epochs, excl_factor)

        return df, epochs

    def read_raw_beh(self,sj:Optional[int]=None,session:Optional[int]=None,
                    files:Union[bool,list]=False)->Union[pd.DataFrame,list]:
        """Read raw behavioral data from CSV files.

        Loads and concatenates raw behavioral CSV files for a specific
        subject and session. Files are automatically located in the
        behavioral/raw/ folder based on subject and session numbers.

        Parameters
        ----------
        sj : int, optional
            Subject identifier. Required if files is False.
        session : int, optional
            Session identifier. Required if files is False.
        files : bool or list, default=False
            If False, automatically finds CSV files matching subject
            and session. If list, uses the provided file paths directly.

        Returns
        -------
        df : pd.DataFrame
            Concatenated behavioral data from all matching files. Empty
            list if no files found.

        Notes
        -----
        Files are expected to be named:
        'sub_{sj}_ses_{session}*.csv'
        
        All matching files are concatenated and the index is reset.

        Examples
        --------
        >>> # Load all behavioral files for subject 1, session 1
        >>> df = self.read_raw_beh(sj=1, session=1)
        >>> 
        >>> # Load specific files
        >>> files = ['path/to/file1.csv', 'path/to/file2.csv']
        >>> df = self.read_raw_beh(files=files)
        """

        # Handle case where files are explicitly provided (even if empty)
        if isinstance(files, list):
            # Files were explicitly provided, use them as-is
            if len(files) == 0:
                return []
        elif files is False:
            # Default case: need to glob for files based on sj/session
            if sj is None or session is None:
                raise ValueError(
                    "Must provide either 'files' parameter with file "
                    "paths or both 'sj' and 'session' parameters"
                )
            # Use flexible glob pattern that matches any digit padding
            beh_folder = self.folder_tracker(ext=['behavioral', 'raw'], fname='')
            pattern = os.path.join(beh_folder, 'sub_*_ses_*.csv')
            all_files = glob.glob(pattern)
            
            # Extract numeric values from sj and session (they might be strings like '02')
            sj_num = int(sj) if isinstance(sj, str) else sj
            session_num = int(session) if isinstance(session, str) else session
            
            # Filter for exact subject and session match (independent of padding)
            # Extract numeric values from filename and compare
            files = []
            for f in all_files:
                match = re.search(r'sub_(\d+)_ses_(\d+)', os.path.basename(f))
                if match:
                    file_sj = int(match.group(1))
                    file_ses = int(match.group(2))
                    if file_sj == sj_num and file_ses == session_num:
                        files.append(f)
            files = sorted(files)
            if not files:
                return []
        
        # read in as dataframe
        df = [pd.read_csv(file) for file in files]
        df = pd.concat(df)
        df.reset_index(inplace = True, drop = True)

        return df
    
    def read_erps(self,erp_name:str,
                cnds:Optional[list]=None,sjs:Union[list,str]='all',
                match:Union[str,bool]=False)->Tuple[dict,np.ndarray]:
        """Read evoked response files for specific analysis.

        Loads evoked ERP data from previously computed ERP analysis.
        Returns data organized by experimental condition.

        Parameters
        ----------
        erp_name : str
            Name assigned to the ERP analysis (used in filename).
        cnds : list, optional
            List of condition labels to load. Default is None, which
            loads 'all_data'.
        sjs : list or 'all', default='all'
            List of subject numbers to load. If 'all', loads all
            subjects found in the erp/evoked/ folder.
        match : str or bool, default=False
            If not False, explicitly checks whether timing events match
            between files. If mismatched, removes samples until aligned.

        Returns
        -------
        erps : dict
            Dictionary with condition labels as keys and lists of
            mne.Evoked objects as values. Each list contains one
            Evoked object per subject.
        times : np.ndarray
            Time points in seconds for the evoked responses.

        Warnings
        --------
        If match is True and sample counts differ across subjects,
        prints warning that data has been artificially aligned.

        Notes
        -----
        Expected file naming convention:
        'sub_{sj}_{cnd}_{erp_name}-ave.fif'

        Examples
        --------
        >>> # Load ERPs for all subjects, two conditions
        >>> erps, times = self.read_erps(
        ...     erp_name='target_locked',
        ...     cnds=['left', 'right']
        ... )
        >>> 
        >>> # Load specific subjects with timing alignment
        >>> erps, times = self.read_erps(
        ...     erp_name='target_locked',
        ...     cnds=['left', 'right'],
        ...     sjs=[1, 2, 3],
        ...     match=True
        ... )

        See Also
        --------
        eeg_analyses.ERP : ERP analysis class that generates these files
        support.support.match_epochs_times : Timing alignment function
        """

        if cnds is None:
            cnds = ['all_data']

        # initiate condtion dict
        erps = dict.fromkeys(cnds, 0)
        
        # loop over conditions
        for cnd in cnds:
            if sjs == 'all':
                files = sorted(
                    glob.glob(
                        self.folder_tracker(
                            ext=['erp', 'evoked'],
                            fname=f'sub_*_{cnd}_{erp_name}-ave.fif'
                        )
                    ),
                    key=lambda s: int(
                        re.search(r'sub_0?(\d+)_', s).group(1)
                    )
                )
            else:
                files = [
                    self.folder_tracker(
                        ext=['erp', 'evoked'],
                        fname=(
                            f'sub_{format_subject_id(sj)}_{cnd}_'
                            f'{erp_name}-ave.fif'
                        )
                    )
                    for sj in sjs
                ]

            # read in actual data
            with open(os.devnull, 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    erps[cnd] = [mne.read_evokeds(file)[0] for file in files]
            if match:
                nr_samples = [erp.times.size for erp in erps[cnd]]
                if len(set(nr_samples)) > 1:
                    print('samples are artificilaly aligned. Please inspect ') 
                    print('data carefully' )
                    erps[cnd] = match_epochs_times(erps[cnd])
        
        # Get times from last condition (cnds is guaranteed non-empty)
        times = erps[cnds[-1]][0].times

        return erps, times
    
    def read_tfr(
        self,
        tfr_folder_path: list,
        tfr_name: str,
        cnds: Optional[list] = None,
        sjs: Union[list, str] = 'all'
    ) -> dict:
        """Read time-frequency analysis results.

        Loads time-frequency data (TFR) computed by the TFR class.
        Returns data organized by experimental condition.

        Parameters
        ----------
        tfr_folder_path : list
            List of subfolders within tfr/ folder specifying analysis
            type. For example, ['wavelet'] or ['multitaper'].
        tfr_name : str
            Name assigned to the TFR analysis (used in filename).
        cnds : list, optional
            List of condition labels to load. Default is None, which
            loads 'all_data'.
        sjs : list or 'all', default='all'
            List of subject numbers to load. If 'all', loads all
            subjects found in the tfr/ folder.

        Returns
        -------
        tfr : dict
            Dictionary with condition labels as keys and lists of
            mne.time_frequency.AverageTFR objects as values. Each list
            contains one TFR object per subject.

        Notes
        -----
        Expected file naming convention:
        'sub_{sj}_{tfr_name}_{cnd}-tfr.h5'

        Files are stored in: tfr/{tfr_folder_path}/

        Examples
        --------
        >>> # Load wavelet TFR for all subjects
        >>> tfr = self.read_tfr(
        ...     tfr_folder_path=['wavelet'],
        ...     tfr_name='target_locked',
        ...     cnds=['left', 'right']
        ... )
        >>> 
        >>> # Load multitaper TFR for specific subjects
        >>> tfr = self.read_tfr(
        ...     tfr_folder_path=['multitaper'],
        ...     tfr_name='target_locked',
        ...     cnds=['left', 'right'],
        ...     sjs=[1, 2, 3]
        ... )

        See Also
        --------
        eeg_analyses.TFR : Time-frequency analysis class
        """

        if cnds is None:
            cnds = ['all_data']

        # initiate condtion dict
        tfr = dict.fromkeys(cnds, 0)

        # set extension
        ext = ['tfr'] + tfr_folder_path

        # loop over conditions
        for cnd in cnds:
            if sjs == 'all':
                files = sorted(
                    glob.glob(
                        self.folder_tracker(
                            ext=ext,
                            fname=f'sub_*_{tfr_name}_{cnd}-tfr.h5'
                        )
                    ),
                    key=lambda s: int(
                        re.search(r'sub_0?(\d+)_', s).group(1)
                    )
                )
            else:
                files = [
                    self.folder_tracker(
                        ext=ext,
                        fname=(
                            f'sub_{format_subject_id(sj)}_'
                            f'{tfr_name}_{cnd}-tfr.h5'
                        )
                    )
                    for sj in sjs
                ]

            # read in actual data
            tfr[cnd] = [mne.time_frequency.read_tfrs(file)[0] 
                                                        for file in files]

        return tfr

    def read_bdm(
        self,
        bdm_folder_path: list,
        bdm_name: Union[str, List[str]],
        sjs: Union[list, str] = 'all',
        analysis_labels: Optional[List[str]] = None
    ) -> list:
        """Read multivariate decoding analysis results.

        Loads classification/decoding data computed by the BDM
        (Backward Decoding Model) class. Supports loading single or
        multiple analyses.

        Parameters
        ----------
        bdm_folder_path : list
            List of subfolders within bdm/ folder pointing to files of
            interest. For example, ['target_loc', 'all_elecs', 'cross'].
        bdm_name : str or list of str
            Name(s) assigned to BDM analysis. If list, loads multiple
            analyses.
        sjs : list or 'all', default='all'
            List of subject numbers to load. If 'all', loads all
            subjects found in the bdm/ folder.
        analysis_labels : list of str, optional
            Labels for multiple analyses when bdm_name is a list. If
            None and bdm_name is a list, uses bdm_name values as labels.

        Returns
        -------
        bdm : list or dict
            If single analysis: list of decoding results (one per
            subject).
            If multiple analyses: dictionary with analysis_labels as
            keys and lists of results as values.

        Notes
        -----
        Expected file naming convention:
        'sj_{sj}_{bdm_name}.pickle'

        Files are stored in: bdm/{bdm_folder_path}/

        Results are loaded as pickle files containing decoding
        accuracies, patterns, or other multivariate metrics.

        Examples
        --------
        >>> # Load single BDM analysis
        >>> bdm = self.read_bdm(
        ...     bdm_folder_path=['target_loc', 'all_elecs'],
        ...     bdm_name='standard'
        ... )
        >>> 
        >>> # Load multiple BDM analyses
        >>> bdm = self.read_bdm(
        ...     bdm_folder_path=['target_loc', 'all_elecs'],
        ...     bdm_name=['standard', 'cross_temporal'],
        ...     analysis_labels=['within_time', 'across_time']
        ... )

        See Also
        --------
        eeg_analyses.BDM : Multivariate decoding class
        """

        # set extension
        ext = ['bdm'] + bdm_folder_path

        if isinstance(bdm_name, str):
            bdm_names = [bdm_name]
        else:
            bdm_names = bdm_name

        # Get files for all bdm_names
        all_files = []
        for name in bdm_names:
            if sjs == 'all':
                # First get all potential files
                pattern = self.folder_tracker(
                    ext=ext,
                    fname=f'sub_*{name}.pickle'
                )
                potential_files = glob.glob(pattern)
                
                # Then filter to only exact matches using regex
                regex_pattern = (
                    rf'sub_0?(\d+)_{re.escape(name)}\.pickle$'
                )
                files = sorted(
                    [
                        f for f in potential_files
                        if re.search(
                            regex_pattern,
                            os.path.basename(f)
                        )
                    ],
                    key=lambda s: int(
                        re.search(r'sub_0?(\d+)_', s).group(1)
                    )
                )
            else:
                files = [
                    self.folder_tracker(
                        ext=ext,
                        fname=(
                            f'sub_{format_subject_id(sj)}_'
                            f'{name}.pickle'
                        )
                    )
                    for sj in sjs
                ]
                
            if not files:
                raise ValueError(f"No files found for analysis {name}")
            all_files.append(files)

        # Check if we have matching files for each subject
        ref_subjects = [self._extract_subject_number(f) for f in all_files[0]]
        for files in all_files[1:]:
            curr_subjects = [self._extract_subject_number(f) for f in files]
            if curr_subjects != ref_subjects:
                raise ValueError(
                    f"Subject mismatch. Expected subjects {ref_subjects}, "
                    f"but got {curr_subjects}")

        bdm = []
        for sj_files in zip(*all_files):

            # load initial file to get reference data
            ref_data = pickle.load(open(sj_files[0], "rb"))
            ref_times = ref_data['info']['times']

            # initialize combined data
            combined_data = {
                'info': ref_data['info'],
                'bdm_info': {}
            }

            # process individual files
            for i, file in enumerate(sj_files):
                data = pickle.load(open(file, "rb"))
                
                # check time alignment
                if not np.array_equal(data['info']['times'], ref_times):
                    raise ValueError(f"Time mismatch in {file}")
                
                # Get analysis name for prefixing
                analysis = bdm_names[i]

                # Add bdm_info with analysis prefix
                for key, value in data['bdm_info'].items():
                    if analysis_labels and i < len(analysis_labels):
                        new_key = f"{analysis_labels[i]}_{key}"
                    else:
                        new_key = f"{analysis}_{key}"
                    combined_data['bdm_info'][new_key] = value

                # Add condition results with custom or default naming
                for key in data.keys():
                    if key not in ['info', 'bdm_info']:
                        update = False
                        if len(all_files) == 1:
                            new_key = key
                        elif analysis_labels and i < len(analysis_labels):
                            new_key = f"{analysis_labels[i]}_{key}"
                            update = True
                        else:
                            new_key = f"{analysis}_{key}"
                            update = True

                        if update and len(data.keys()) == 3:
                            last_underscore = new_key.rfind('_')
                            if last_underscore != -1:
                                new_key = new_key[:last_underscore]
                        
                        combined_data[new_key] = data[key]

            bdm.append(combined_data)

        return bdm
    
    def read_ctfs(self,ctf_folder_path:list,output_type:str,
                  ctf_name:str,sjs:Union[list,str]='all')->list:
        """Read channel tuning function analysis results.

        Loads CTF (Channel Tuning Function) data computed by the CTF
        class. Supports loading tuning functions, metadata, or
        parameters.

        Parameters
        ----------
        ctf_folder_path : list
            List of subfolders within ctf/ folder pointing to files of
            interest. For example, ['orientation', 'standard'].
        output_type : str
            Type of output to load. Options:
                - 'ctf' : Tuning function data
                - 'info' : Metadata and analysis information
                - 'param' : Analysis parameters
        ctf_name : str
            Name assigned to the CTF analysis (used in filename).
        sjs : list or 'all', default='all'
            List of subject numbers to load. If 'all', loads all
            subjects found in the ctf/ folder.

        Returns
        -------
        ctfs : list
            List of CTF results, one per subject. Content depends on
            output_type parameter.

        Notes
        -----
        Expected file naming conventions:
            - 'sub_{sj}_{ctf_name}_ctf.pickle' (for output_type='ctf')
            - 'sub_{sj}_{ctf_name}_info.pickle' (for output_type='info')
            - 'sub_{sj}_{ctf_name}_param.pickle' 
               (for output_type='param')

        Files are stored in: ctf/{ctf_folder_path}/

        Examples
        --------
        >>> # Load CTF data for all subjects
        >>> ctfs = self.read_ctfs(
        ...     ctf_folder_path=['orientation', 'standard'],
        ...     output_type='ctf',
        ...     ctf_name='main'
        ... )
        >>> 
        >>> # Load CTF parameters for specific subjects
        >>> params = self.read_ctfs(
        ...     ctf_folder_path=['orientation', 'standard'],
        ...     output_type='param',
        ...     ctf_name='main',
        ...     sjs=[1, 2, 3]
        ... )

        See Also
        --------
        eeg_analyses.CTF : Channel tuning function analysis class
        """

        # set extension
        ext = ['ctf'] + ctf_folder_path

        # determine output file suffix
        output = 'ctf'  # default
        if output_type=='ctf':
            output = 'ctf'
        elif output_type=='info':
            output = 'info'
        elif output_type=='param':
            output = 'param'

        if sjs == 'all':
            files = sorted(
                glob.glob(
                    self.folder_tracker(
                        ext=ext,
                        fname=f'sub_*_{ctf_name}_{output}.pickle'
                    )
                ),
                key=lambda s: int(
                    re.search(r'sub_0?(\d+)_', s).group(1)
                )
            )
        else:
            files = [
                self.folder_tracker(
                    ext=ext,
                    fname=(
                        f'sub_{format_subject_id(sj)}_'
                        f'{ctf_name}_{output}.pickle'
                    )
                )
                for sj in sjs
            ]

        ctfs = [pickle.load(open(file, 'rb')) for file in files]

        return ctfs