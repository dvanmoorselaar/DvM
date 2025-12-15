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
from mne import BaseEpochs
from mne.io import BaseRaw


from typing import Dict, List, Optional, Generic, Union, Tuple, Any
from termios import tcflush, TCIFLUSH
from autoreject import get_rejection_threshold
from eeg_analyses.EYE import *
from math import sqrt
from IPython import embed
from support.support import get_time_slice, trial_exclusion
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

class RAW(BaseRaw, FolderStructure):
    """
    Extended MNE Raw class with preprocessing functionality.
    
    This class wraps MNE's Raw objects and provides custom methods for 
    electrophysiological data preprocessing, including channel 
    replacement, re-referencing, montage configuration, and 
    event selection. While designed primarily for EEG, the class can 
    also handle MEG and other data types supported by MNE.
    
    Inherits from MNE's BaseRaw class and FolderStructure for file 
    organization capabilities.

    Parameters
    ----------
    See __init__ method for initialization parameters.

    Attributes
    ----------
    All attributes from mne.io.BaseRaw, including:
        info : mne.Info
            Measurement information
        ch_names : list
            Channel names
        times : ndarray
            Time points
        
    Methods
    -------
    replace_channel(sj, session, replace)
        Replace bad electrodes with designated replacement electrodes.
    rereference(ref_channels, change_voltage, to_remove)
        Re-reference data to specified reference channel(s).
    configure_montage(montage, ch_remove)
        Set electrode montage and rename channels.
    select_events(event_id, stim_channel, binary, consecutive, 
                  min_duration)
        Detect and extract trigger events from stimulus channel.

    See Also
    --------
    mne.io.Raw : MNE's base Raw class
    mne.io.read_raw_bdf : Read BioSemi BDF files
    mne.io.read_raw_fif : Read FIF files

    Examples
    --------
    >>> # Read BDF file
    >>> raw = RAW('data.bdf', preload=True)
    
    >>> # Read FIF file with auto-detection
    >>> raw = RAW('data_raw.fif')
    
    >>> # Read EDF with specific EOG channels
    >>> raw = RAW('data.edf', eog=['EOG1', 'EOG2'])
    
    >>> # Using convenience methods
    >>> raw = RAW.from_bdf('subject01.bdf', preload=True)
    
    >>> # Chain preprocessing methods
    >>> raw.replace_channel(sj=1, session=1, replace=replace_dict)
    >>> raw.rereference('average')
    >>> raw.configure_montage('biosemi64')
    >>> events = raw.select_events(event_id=[1, 2, 3])
    """

    def __init__(self, 
                 input_fname: Union[str, os.PathLike],
                 file_type: Optional[str] = None,
                 eog: Optional[Union[str, List[str]]] = None,
                 stim_channel: Union[int, str, List[str], None] = -1,
                 exclude: Union[List[str], Tuple[str, ...]] = (),
                 preload: bool = True,
                 verbose: Optional[Union[bool, str, int]] = None,
                 **kwargs):
        """
        Initialize RAW by reading data from various file formats.

        Parameters
        ----------
        input_fname : str or PathLike
            Path to the EEG/MEG data file. Can be a string or Path 
            object.
        file_type : str, optional
            Type of file to read. Supported options:
            
            - 'bdf': BioSemi BDF format (default if .bdf extension)
            - 'edf': European Data Format (default if .edf extension)
            - 'fif': Neuromag/Elekta format (default if .fif extension)
            - 'brainvision': BrainVision format (default 
               if .vhdr extension)
            - 'cnt': Neuroscan CNT format
            - 'set': EEGLAB format
            
            If None, automatically detects from file extension. 
            Default is None.
        eog : str or list of str, optional
            Channel name(s) for EOG channels. Can be a single channel 
            name or list of channel names. Default is None.
        stim_channel : int, str, list of str, or None, optional
            Channel name or index for stimulus/trigger channel.
            Use -1 for last channel (common default). Use None to 
            exclude. Default is -1.
        exclude : list of str or tuple of str, optional
            List of channel names to exclude from reading. 
            Default is () (no channels excluded).
        preload : bool, optional
            If True, load data into memory immediately. If False, 
            data is read on-demand. Default is True.
        verbose : bool, str, or int, optional
            Control verbosity of MNE output. Can be bool, str ('INFO', 
            'WARNING', 'ERROR'), or int. Default is None.
        **kwargs : dict
            Additional keyword arguments passed to the specific 
            MNE reader function (e.g., montage, misc for BDF files).

        Raises
        ------
        ValueError
            If file type cannot be determined or is unsupported.
        FileNotFoundError
            If input file does not exist.

        Examples
        --------
        >>> # Auto-detect file type from extension
        >>> raw = RAW('subject01.bdf', preload=True)
        
        >>> # Specify file type explicitly
        >>> raw = RAW('data.fif', file_type='fif')
        
        >>> # Specify EOG channels
        >>> raw = RAW('data.bdf', eog=['EXG1', 'EXG2'])
        """
        
        # Convert to Path object for easier handling
        input_fname = os.path.expanduser(input_fname)
        
        # Auto-detect file type from extension if not specified
        if file_type is None:
            ext = os.path.splitext(input_fname)[1].lower()
            file_type_map = {
                '.bdf': 'bdf',
                '.edf': 'edf',
                '.fif': 'fif',
                '.vhdr': 'brainvision',
                '.cnt': 'cnt',
                '.set': 'set',
            }
            file_type = file_type_map.get(ext)
            if file_type is None:
                raise ValueError(
                    f"Cannot determine file type from extension '{ext}'. "
                    f"Please specify file_type parameter. "
                    f"Supported types: {list(file_type_map.values())}"
                )
        
        # Read data using appropriate MNE function
        if file_type == 'bdf':
            raw = mne.io.read_raw_bdf(
                input_fname,
                eog=eog,
                stim_channel=stim_channel,
                exclude=exclude,
                preload=preload,
                verbose=verbose,
                **kwargs
            )
        elif file_type == 'edf':
            raw = mne.io.read_raw_edf(
                input_fname,
                eog=eog,
                stim_channel=stim_channel,
                exclude=exclude,
                preload=preload,
                verbose=verbose,
                **kwargs
            )
        elif file_type == 'fif':
            raw = mne.io.read_raw_fif(
                input_fname,
                preload=preload,
                verbose=verbose,
                **kwargs
            )
        elif file_type == 'brainvision':
            raw = mne.io.read_raw_brainvision(
                input_fname,
                eog=eog,
                preload=preload,
                verbose=verbose,
                **kwargs
            )
        elif file_type == 'cnt':
            raw = mne.io.read_raw_cnt(
                input_fname,
                eog=eog,
                preload=preload,
                verbose=verbose,
                **kwargs
            )
        elif file_type == 'set':
            raw = mne.io.read_raw_eeglab(
                input_fname,
                preload=preload,
                verbose=verbose,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported file type: {file_type}. "
                f"Supported types: bdf, edf, fif, brainvision, cnt, set"
            )
        
        # Copy all attributes from the loaded raw object to self
        self.__dict__.update(raw.__dict__)
        
        print(f'Loaded {file_type.upper()} file: {input_fname}')
        print(f'Channels: {len(self.ch_names)}, '
              f'Sampling rate: {self.info["sfreq"]} Hz')
    
    @classmethod
    def from_bdf(cls, 
                 input_fname: Union[str, os.PathLike],
                 **kwargs) -> 'RAW':
        """Convenience method to load BioSemi BDF files.
        
        Args:
            input_fname: Path to the BDF file.
            **kwargs: Additional arguments passed to __init__.
            
        Returns:
            RAW instance with loaded BDF data.
            
        Examples:
            >>> raw = RAW.from_bdf('subject01.bdf', preload=True)
        """
        return cls(input_fname, file_type='bdf', **kwargs)
    
    @classmethod
    def from_edf(cls, 
                 input_fname: Union[str, os.PathLike],
                 **kwargs) -> 'RAW':
        """Convenience method to load EDF files.
        
        Args:
            input_fname: Path to the EDF file.
            **kwargs: Additional arguments passed to __init__.
            
        Returns:
            RAW instance with loaded EDF data.
            
        Examples:
            >>> raw = RAW.from_edf('subject01.edf', preload=True)
        """
        return cls(input_fname, file_type='edf', **kwargs)
    
    @classmethod
    def from_fif(cls, 
                 input_fname: Union[str, os.PathLike],
                 **kwargs) -> 'RAW':
        """Convenience method to load FIF files.
        
        Args:
            input_fname: Path to the FIF file.
            **kwargs: Additional arguments passed to __init__.
            
        Returns:
            RAW instance with loaded FIF data.
            
        Examples:
            >>> raw = RAW.from_fif('subject01_raw.fif', preload=True)
        """
        return cls(input_fname, file_type='fif', **kwargs)
    
    @classmethod
    def from_brainvision(cls, 
                         input_fname: Union[str, os.PathLike],
                         **kwargs) -> 'RAW':
        """Convenience method to load BrainVision files.
        
        Args:
            input_fname: Path to the .vhdr file.
            **kwargs: Additional arguments passed to __init__.
            
        Returns:
            RAW instance with loaded BrainVision data.
            
        Examples:
            >>> raw = RAW.from_brainvision('subject01.vhdr', preload=True)
        """
        return cls(input_fname, file_type='brainvision', **kwargs)
  
    def report_raw(self, report, events, event_id):
        '''

        '''

        # report raw
        report.add_raw(self, title='raw EEG',psd=True)
        # and events
        events = events[np.in1d(events[:,2], event_id)]
        report.add_events(events, title = 'detected events', sfreq = self.info['sfreq'])

        return report

    def replace_channel(
        self,
        replace: Dict[str, str]
    ) -> 'RAW':
        """
        Replace bad electrodes with designated replacement electrodes.

        Replaces bad electrodes with alternative electrodes that were 
        used during recording. This is useful when electrodes were 
        replaced during setup due to high impedance or technical issues.

        Parameters
        ----------
        replace : dict of str
            Dictionary mapping original electrode names to replacement 
            electrode names. 
            Keys: Original electrode names to replace
            Values: Replacement electrode names containing the data

        Returns
        -------
        self : RAW
            Returns the instance itself with modified channel data.

        Notes
        -----
        - Modifications are performed in-place.
        - Only data from the replacement electrode is copied; the 
          channel name remains the original electrode name.
        - Replacement electrodes are removed from the data after 
          copying.
        - Prints a warning if either the original or replacement 
          electrode is not found in the data.
        - If an empty dictionary is passed, returns immediately without 
          modification.

        Examples
        --------
        >>> # Replace two electrodes
        >>> replace_dict = {'F1': 'EXG7', 'P3': 'EXG8'}
        >>> raw.replace_channel(replace=replace_dict)
        Electrode F1 replaced by EXG7
        Electrode P3 replaced by EXG8
        Removed replacement channels: ['EXG7', 'EXG8']
        
        >>> # Single electrode replacement
        >>> raw.replace_channel(replace={'Fp1': 'EXG7'})
        Electrode Fp1 replaced by EXG7
        Removed replacement channels: ['EXG7']
        """
        # Return early if no replacements specified
        if not replace:
            return self
        
        # Collect channels to remove after replacement
        channels_to_remove = []
        
        # Replace each electrode's data
        for orig_elec, repl_elec in replace.items():
            try:
                orig_idx = self.ch_names.index(orig_elec)
                repl_idx = self.ch_names.index(repl_elec)
                self._data[orig_idx] = self._data[repl_idx].copy()
                channels_to_remove.append(repl_elec)
                print(f'Electrode {orig_elec} replaced by {repl_elec}')
            except ValueError:
                print(f'Warning: Could not replace {orig_elec} with '
                      f'{repl_elec}. Channel not found in data.')
        
        # Remove replacement channels
        if channels_to_remove:
            self.drop_channels(channels_to_remove)
            print(f'Removed replacement channels: {channels_to_remove}')

        return self

    def rereference(
        self,
        ref_channels: Union[str, List[str]],
        change_voltage: bool = True,
        to_remove: Optional[List[str]] = None
    ) -> 'RAW':
        """
        Re-reference EEG data to specified reference channel(s).

        Re-references the raw EEG data to specified reference channels
        or average reference. Optionally converts voltage units from 
        Volts to microVolts and removes specified channels after 
        re-referencing.

        Parameters
        ----------
        ref_channels : str or list of str
            Channel(s) to use as reference. Use 'average' for average
            reference across all EEG channels, or provide a list of 
            channel names for specific reference electrode(s).
        change_voltage : bool, optional
            Whether to convert voltage from Volts to microVolts. 
            Default is True.
        to_remove : list of str, optional
            Additional channels to remove after re-referencing. 
            Default is None (no additional channels removed).

        Returns
        -------
        self : RAW
            Returns the instance itself with modified channel data.

        Notes
        -----
        - Modifications are performed in-place.
        - If change_voltage=True, converts EEG and EOG channel data from 
          V to μV (×1e6).
        - Reference channels are automatically removed from the data 
          unless 'average' reference is used.
        - Only channels present in the data will be removed; missing 
          channels are silently skipped.

        Examples
        --------
        >>> # Average reference
        >>> raw.rereference('average')
        
        >>> # Single channel reference
        >>> raw.rereference(['Cz'])
        
        >>> # Linked mastoids reference with additional channel removal
        >>> raw.rereference(['M1', 'M2'], to_remove=['EXG7', 'EXG8'])
        """

        if to_remove is None:
            to_remove = []
        else:
            to_remove = to_remove.copy()  

        # Convert voltage units if requested
        if change_voltage:
            # Get EEG and EOG channel indices
            picks = mne.pick_types(self.info, eeg=True, eog=True, exclude=[])
            self._data[picks, :] *= 1e6
            print('Converted voltage units from V to μV for'
            ' EEG and EOG channels')

        # Re-reference EEG channels
        self.set_eeg_reference(ref_channels=ref_channels)
        print(f'Re-referenced EEG data to: {ref_channels}')

        # Add reference channels to removal list if not using average reference
        if ref_channels != 'average':
            if isinstance(ref_channels, str):
                to_remove.append(ref_channels)
            else:
                to_remove.extend(ref_channels)

        # Remove specified channels if they exist
        to_remove = [ch for ch in to_remove if ch in self.ch_names]
        if to_remove:
            self.drop_channels(to_remove)
            print(f'Removed channels: {to_remove}')

        return self

    def configure_montage(
        self,
        montage: Union[str, mne.channels.montage.DigMontage] = 'biosemi64',
        ch_remove: Optional[List[str]] = None
    ) -> 'RAW':
        """
        Set electrode montage and rename channels to standard 
        nomenclature.

        Applies a standard electrode montage and optionally renames 
        channels from BioSemi A/B naming scheme (A1-A32, B1-B32) to 
        standard 10-20 system nomenclature. Channels can be removed 
        prior to montage application.

        Parameters
        ----------
        montage : str or mne.channels.montage.DigMontage, optional
            Montage to apply. Can be either:
            - String name of a standard montage (e.g., 'biosemi64', 
              'biosemi128', 'standard_1020')
            - An MNE DigMontage object with custom electrode positions
            Default is 'biosemi64'.
        ch_remove : list of str, optional
            Channel names to remove before applying the montage. 
            Default is None (no channels removed).

        Returns
        -------
        self : RAW
            Returns the instance itself with updated montage and 
            channel names.

        Notes
        -----
        - Modifications are performed in-place.
        - If channels use BioSemi naming (A1-A32, B1-B32), they are 
          automatically renamed to match the montage's standard names.
        - Channel removal is performed before montage application.
        - Missing channels in the montage trigger a warning but do not 
          raise an error.

        Examples
        --------
        >>> # Apply standard biosemi64 montage
        >>> raw.configure_montage('biosemi64')
        
        >>> # Apply montage and remove external channels
        >>> raw.configure_montage('biosemi64', 
        ...     ch_remove=['EXG7', 'EXG8'])
        
        >>> # Use custom montage
        >>> custom_montage = mne.channels.make_dig_montage(...)
        >>> raw.configure_montage(custom_montage)
        """

        if ch_remove is None:
            ch_remove = []
        else:
            ch_remove = ch_remove.copy()

        # Drop specified channels
        if ch_remove:
            ch_to_remove = [ch for ch in ch_remove if ch in self.ch_names]
            if ch_to_remove:
                self.drop_channels(ch_to_remove)
                print(f'Removed {len(ch_to_remove)} channels: {ch_to_remove}')

        # Get montage object if string was passed
        if isinstance(montage, str):
            montage_obj = mne.channels.make_standard_montage(montage)
        else:
            montage_obj = montage

        # Create mapping dictionary for renaming BioSemi channels
        ch_mapping = {}
        if len(self.ch_names) > 0 and self.ch_names[0].startswith('A'):
            # Check if using BioSemi naming convention
            biosemi_pattern = all(
                ch.startswith(('A', 'B')) and ch[1:].isdigit() 
                for ch in self.ch_names[:min(10, len(self.ch_names))]
                if not ch.startswith('EXG')
            )
            
            if biosemi_pattern and hasattr(montage_obj, 'ch_names'):
                idx = 0
                for hemi in ['A', 'B', 'C', 'D']:
                    for elec in range(1, 33):
                        ch_name = f'{hemi}{elec}'
                        if ch_name in self.ch_names and \
                            idx < len(montage_obj.ch_names):
                            ch_mapping[ch_name] = montage_obj.ch_names[idx]
                            idx += 1
                
                if ch_mapping:
                    self.rename_channels(ch_mapping)
                    print(f'Renamed {len(ch_mapping)} channels from BioSemi '
                          f'to 10-20 system')

        # Apply montage
        self.set_montage(montage=montage_obj, on_missing='warn')
        montage_name = montage if isinstance(montage, str) else "custom"
        print(f'Montage {montage_name} applied')

        return self

    def select_events(
        self,
        event_id: Optional[List[int]] = None,
        stim_channel: Optional[str] = None,
        binary: int = 0,
        consecutive: bool = False,
        min_duration: float = 0.003
    ) -> np.ndarray:
        """
        Detect and extract trigger events from stimulus channel.

        Finds trigger events in the raw EEG data using MNE's event 
        detection. Optionally corrects for binary offsets in the trigger 
        channel and removes consecutive duplicate events.

        Parameters
        ----------
        event_id : list of int, optional
            List of trigger values to retain in the output. If None, 
            all detected events are returned. Default is None.
        stim_channel : str, optional
            Name of the stimulus/trigger channel. If None, uses the 
            default stim channel specified during data loading. 
            Default is None.
        binary : int, optional
            Binary offset to subtract from the stimulus channel before 
            event detection. Used to correct for spoke triggers or other 
            systematic offsets. Default is 0 (no correction).
        consecutive : bool, optional
            If False, only report events where the trigger channel value 
            changes from/to zero. If True, report all trigger value 
            changes. Default is False.
        min_duration : float, optional
            Minimum duration (in seconds) required for a trigger value 
            change to be considered a valid event. Default is 0.003.

        Returns
        -------
        events : ndarray, shape (n_events, 3)
            Array of events with columns:
            - Column 0: Event sample number (including first_samp 
              offset)
            - Column 1: Previous trigger value (for MNE compatibility)
            - Column 2: Current trigger/event ID

        Notes
        -----
        - Binary offset correction is applied temporarily and does not 
          modify the underlying data.
        - Consecutive duplicate events (same trigger appearing twice in 
          a row) are removed unless consecutive=True.
        - If event_id is specified, only matching events are included 
          in duplicate removal logic.

        Examples
        --------
        >>> # Detect all events
        >>> events = raw.select_events()
        
        >>> # Detect specific trigger values
        >>> events = raw.select_events(event_id=[1, 2, 3])
        
        >>> # Correct for binary offset and detect events
        >>> events = raw.select_events(event_id=[10, 20], binary=3840)
        
        >>> # Include consecutive events
        >>> events = raw.select_events(event_id=[1, 2], 
        ...             consecutive=True)
        """

        # Get stim channel data (make a copy to avoid modifying original)
        if stim_channel is None:
            # Find the stim channel
            stim_picks = mne.pick_types(self.info, stim=True, exclude=[])
            if len(stim_picks) == 0:
                raise ValueError('No stim channel found in data')
            stim_idx = stim_picks[0]
        else:
            try:
                stim_idx = self.ch_names.index(stim_channel)
            except ValueError as exc:
                raise ValueError(f'Stimulus channel {stim_channel} not '
                                 f'found in data') from exc
        
        # Apply binary offset correction temporarily
        original_data = None
        if binary != 0:
            original_data = self._data[stim_idx, :].copy()
            self._data[stim_idx, :] -= binary

        try:
            # Find events using MNE
            events = mne.find_events(
                self,
                stim_channel=stim_channel,
                consecutive=consecutive,
                min_duration=min_duration
            )

            # Remove consecutive identical events if needed
            if not consecutive and event_id is not None:
                # Find consecutive duplicates for events in event_id
                is_duplicate = np.zeros(len(events), dtype=bool)
                for i in range(len(events) - 1):
                    if (events[i, 2] == events[i + 1, 2] and 
                        events[i, 2] in event_id):
                        is_duplicate[i] = True
                
                if np.any(is_duplicate):
                    events = events[~is_duplicate]
                    nr_removed = np.sum(is_duplicate)
                    print(f'{nr_removed} consecutive duplicate events removed')

            # Filter events by event_id if specified
            if event_id is not None:
                mask = np.isin(events[:, 2], event_id)
                events = events[mask]
                print(f'{len(events)} events detected matching event_id')
            else:
                print(f'{len(events)} events detected')

        finally:
            # Restore original data if binary offset was applied
            if original_data is not None:
                self._data[stim_idx, :] = original_data

        return events

class Epochs(mne.Epochs, BaseEpochs,FolderStructure):
    '''
    Epochs extracted from a Raw instance. Child class based on mne built-in
    Epochs, such that extract functionality can be added to this base class.
    For default documentation see:
    https://mne.tools/stable/generated/mne.Epochs
    '''

    def __init__(self, sj: int, session: int, raw: mne.io.Raw,
                events: np.array, event_id: Union[int, list, dict],
                tmin: float=-0.2, tmax: float=0.5,
                flt_pad: Union[float, tuple]=None,
                baseline: tuple=(None, None),
                picks: Union[str, list, slice]=None, preload: bool=True,
                reject: dict=None, flat: dict=None, proj: bool=False,
                decim: int=1, reject_tmin: float=None, reject_tmax: float=None,
                detrend: int=None,on_missing: str='raise',
                reject_by_annotation: bool=False,metadata: pd.DataFrame=None,
                event_repeated: str='error',
                verbose: Union[bool, str, int]=None):

        # set child class specific info
        self.sj = sj
        self.session = str(session)
        self.flt_pad = flt_pad
        if isinstance(flt_pad, (tuple,list)):
            tmin -= flt_pad[0]
            tmax += flt_pad[1]
        else:
            tmin -= flt_pad
            tmax += flt_pad
  
        super(Epochs, self).__init__(raw=raw, events=events, event_id=event_id,
                                    tmin=tmin, tmax=tmax,baseline=baseline,
                                    picks=picks, preload=preload,
                                    reject=reject, flat=flat, proj=proj,
                                    decim=decim, reject_tmin=reject_tmin,
                                    reject_tmax=reject_tmax, detrend=detrend,
                                    on_missing=on_missing,
                                    reject_by_annotation=reject_by_annotation,
                                    metadata=metadata,
                                    event_repeated=event_repeated,
                                    verbose=verbose)

    def align_meta_data(
        self,
        events: np.ndarray,
        trigger_header: str = 'trigger',
        beh_oi: Optional[List[str]] = None,
        idx_remove: Optional[np.ndarray] = None,
        eye_inf: Optional[dict] = None,
        del_practice: bool = True, 
        excl_factor: Optional[dict] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Align epoched EEG data with behavioral data from CSV file.

        Matches EEG epochs with behavioral trial data by comparing 
        trigger values. Handles mismatches by removing trials from 
        behavioral data or epochs from EEG data. Optionally integrates 
        eye-tracking data and removes practice trials.

        Parameters
        ----------
        events : np.ndarray
            Event array as returned by RAW.select_events, 
            shape (n_events, 3). Column 2 contains trigger values.
        trigger_header : str, optional
            Column name in behavioral CSV containing trigger values used 
            for epoching. Default is 'trigger'.
        beh_oi : list of str, optional
            List of column names from behavioral data to link to epochs.
            If None, all columns are used. Default is None.
        idx_remove : np.ndarray, optional
            Indices of trigger events to remove before alignment. Useful 
            for removing spurious triggers. Default is None.
        eye_inf : dict, optional
            Dictionary with eye tracking parameters. If None, no eye 
            tracking data is processed. Expected keys: 'eog' (list of 4 
            channel names), 'tracker_ext', 'sfreq', 'viewing_dist', 
            'screen_res', 'screen_h', 'start', 'window_oi', 
            'trigger_msg', 'drift_correct'. Default is None.
        del_practice : bool, optional
            If True, removes practice trials from behavioral data before 
            alignment. Requires 'practice' column with 'yes'/'no' 
            values. Default is True.
        excl_factor : dict, optional
            Dictionary specifying factors for trial exclusion. Passed to 
            trial_exclusion function. Default is None.

        Returns
        -------
        missing : np.ndarray
            Array of trial numbers removed from behavioral data due to 
            missing EEG triggers.
        report_str : str
            Detailed report of the alignment process including number of 
            matches, removed trials, and eye tracking status.

        Raises
        ------
        ValueError
            If behavioral file has more trials than EEG epochs and 
            'nr_trials' column is missing, preventing informed 
            alignment. If too many EEG triggers exist and automatic 
            alignment fails.

        Notes
        -----
        - Requires behavioral CSV file with columns: trigger values 
          column (specified by trigger_header) and 'nr_trials' 
          (trial counter).
        - self.selection attribute (inherited from mne.Epochs) contains 
          indices of non-dropped epochs.
        - Modifies self.metadata in-place to store aligned behavioral 
          data.
        - Eye tracking data alignment is performed if eye_inf is 
          provided.
        - **Recommended trigger scheme**: Use ascending numerical
          triggers (1, 2, 3, etc.) rather than condition-specific 
          values (e.g., 10 for left target, 20 for right target). After 
          alignment, all behavioral variables are accessible via 
          self.metadata, making trigger-condition coupling redundant and 
          error-prone.

        Examples
        --------
        >>> # Basic alignment
        >>> missing, report = epochs.align_meta_data(
        ...     events=events,
        ...     beh_oi=['trigger', 'condition', 'RT', 'accuracy']
        ... )
        
        >>> # With eye tracking and practice trial removal
        >>> missing, report = epochs.align_meta_data(
        ...     events=events,
        ...     beh_oi=['trigger', 'condition', 'RT'],
        ...     eye_inf=eye_params,
        ...     del_practice=True
        ... )
        """
        
        if beh_oi is None:
            beh_oi = []

        print('Linking behavior to epochs')
        report_str = ''

        # read in data file and select param of interest
        beh = self.read_raw_beh(self.sj, self.session)
        if len(beh) == 0:
            return np.array([]), 'No behavior file found'
        
        # Validate required columns
        if trigger_header not in beh.columns:
            raise ValueError(
                f"Trigger column '{trigger_header}' not found in behavioral "
                f"data. Available columns: {list(beh.columns)}"
            )
        
        # Select columns of interest
        if beh_oi:
            missing_cols = [col for col in beh_oi if col not in beh.columns]
            if missing_cols:
                raise ValueError(
                    f"Columns {missing_cols} not found in behavioral data. "
                    f"Available columns: {list(beh.columns)}"
                )
            beh = beh[beh_oi]

        # Get EEG triggers in epoched order (self.selection from mne.Epochs)
        eeg_triggers = events[self.selection, 2]

        # remove practice trials
        if del_practice and 'practice' in beh_oi:
            nr_remove = beh[beh.practice == 'yes'].shape[0]
            nr_exp = beh[beh.practice == 'no'].shape[0]
            # check whether EEG and practice triggers overlap
            practice_triggers = beh[trigger_header].values[:nr_remove]
            if all(practice_triggers == eeg_triggers[:nr_remove]) and \
                nr_exp -  eeg_triggers.size < 0:
                self.drop(np.arange(nr_remove))    
                report_str += (f'{nr_remove} practice events removed '
                           'from eeg based on automatic detection. '
                           'Please inspect data carefully')
                eeg_triggers = np.delete(eeg_triggers, np.arange(nr_remove))
            print(f'{nr_remove} practice trials removed from behavior')
            beh = beh[beh.practice == 'no']
            beh = beh.drop(['practice'], axis=1)
            beh.reset_index(inplace=True, drop=True)
        beh_triggers = beh[trigger_header].values

        # Remove specified trigger events if requested
        if idx_remove is not None:
            self.drop(idx_remove)
            nr_remove = idx_remove.size
            report_str += (f'{nr_remove} trigger events removed '
                           'as specified by the user\n')
            eeg_triggers = np.delete(eeg_triggers, idx_remove)

        # check alignment
        session_switch = np.diff(beh.nr_trials) < 1
        if sum(session_switch) > 0:
            idx = np.where(session_switch)[0]+1
            trial_split = np.array_split(beh.nr_trials.values,idx)
            for i in range(1,len(trial_split)):
                trial_split[i] += trial_split[i-1][-1]
            beh.nr_trials = np.hstack(trial_split)
            
        missing_trials = []
        nr_miss = beh_triggers.size - eeg_triggers.size
 
        if nr_miss > 0:
            report_str += (f'Behavior has {nr_miss} more trials than detected '
                          'events. Trial numbers will be '
                          'removed in an attempt to fix this (or see '
                          'terminal output): \n')

            if 'nr_trials' not in beh.columns:
                raise ValueError('Behavior file does not contain a column '
                                'with trial info named nr_trials. Please '
                                'adjust for automatic alignment')
        elif nr_miss < 0:
            report_str += ('EEG has more events than behavioral trials. '
                          'Removing excess EEG epochs to align data. '
                          'Please inspect your data carefully!\n')
            
            if all(beh_triggers == eeg_triggers[:nr_miss]):
                idx_remove = np.arange(eeg_triggers.size)[nr_miss:]
                nr_miss = 0
            else:
                idx_remove = []

            while nr_miss < 0:
                # continue to remove EEG triggers until data files are lined up
                for i, tr in enumerate(beh_triggers):
                    if tr != eeg_triggers[i]:
                        nr_miss += 1

            self.drop(idx_remove)
            eeg_triggers = np.delete(eeg_triggers, idx_remove)
            # check file sizes
            if sum(beh_triggers == eeg_triggers) < eeg_triggers.size:
                raise ValueError(
                    'Behavior and EEG cannot be linked: too many EEG triggers.'
                    ' Please pass indices of trials to be removed via ' 
                    'idx_remove parameter.'
                )
            
        add_info = False if nr_miss > 10 else True
        while nr_miss > 0:
            stop = True
            # continue to remove beh trials until data files are lined up
            for i, tr in enumerate(eeg_triggers):
                if tr != beh_triggers[i]: # remove trigger from beh_file
                    miss = beh['nr_trials'].iloc[i]
                    missing_trials.append(miss)
                    if add_info:
                        report_str += f'{miss}, '
                    else: 
                        print(f'removed trial {miss}')
                    beh.drop(beh.index[i], inplace=True)
                    beh_triggers = np.delete(beh_triggers, i, axis=0)
                    nr_miss -= 1
                    stop = False
                    break

            # check whether there are missing trials at end of beh file
            if beh_triggers.size > eeg_triggers.size and stop:
                # drop the last items from the beh file
                new_miss = beh.loc[beh.index[-nr_miss:].values, 'nr_trials']
                missing_trials = np.hstack((missing_trials,
                                           new_miss.values))
                beh.drop(beh.index[-nr_miss:], inplace=True)
                report_str += (f'\nRemoved final {nr_miss} trials from '
                              'behavior to align data. Please inspect your '
                              'data carefully!')
                nr_miss = 0

        # keep track of missing trials to align eye tracking data (if any)
        missing = np.array(missing_trials)
        beh.reset_index(inplace=True, drop=True)

        # log number of matches between beh and EEG
        nr_matches = sum(beh[trigger_header].values == eeg_triggers)
        nr_epochs = eeg_triggers.size
        report_str += (f'\n{nr_matches} matches between beh and epoched '
                      f'data out of {nr_epochs}. ')

        # link eye(tracker) data if parameters provided
        if eye_inf is not None:
            tracker, eye_report = self.align_eye_data(
                eye_inf, missing, nr_epochs,
                vEOG=eye_inf.get('eog', [None]*4)[:2],
                hEOG=eye_inf.get('eog', [None]*4)[2:]
            )
        else:
            tracker, eye_report = False, 'No eye tracker data linked'

        # add behavior to epochs object
        if excl_factor is not None:
            beh, self, idx = trial_exclusion(beh, self, excl_factor)
            if tracker:
                eye = np.load(self.folder_tracker(ext=['eye','processed'],
                        fname=f'sj_{self.sj}_ses_{self.session}_xy_eye.npz'))
                eye_x, eye_y = eye['x'], eye['y']
                eye_x = np.delete(eye_x, idx, axis=0)
                eye_y = np.delete(eye_y, idx, axis=0)
                np.savez(self.folder_tracker(ext=['eye','processed'],
                        fname=f'sj_{self.sj}_ses_{self.session}_xy_eye.npz'),
                        times = eye['times'], x = eye_x, y = eye_y, 
                        sfreq = eye['sfreq'])       

            report_str += 'Excluded trials based on factors. '
            report_str += f'Final set contains {beh.shape[0]} trials \n'
        report_str += eye_report
        self.metadata = beh

        return missing, report_str

    def align_eye_data(
        self,
        eye_info: Optional[dict],
        missing: np.ndarray,
        nr_epochs: int,
        vEOG: Optional[List[str]] = None,
        hEOG: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Process EOG rereferencing and align eye tracker data with EEG 
        epochs.

        This method performs two independent operations:
        1. Optionally rereferences EOG electrodes via bipolar montage 
           (subtraction)
        2. Links eye tracker data files to EEG epochs and saves aligned 
           gaze data

        Parameters
        ----------
        eye_info : dict, optional
            Dictionary with eye tracking parameters. If None, only EOG 
            processing is performed. Expected keys:
            - 'tracker_ext': str, file extension (e.g., 'asc', 'tsv')
            - 'sfreq': float, eye tracker sampling rate in Hz
            - 'viewing_dist': float, viewing distance in cm
            - 'screen_res': tuple, screen resolution (width, height) in 
               pixels
            - 'screen_h': float, screen height in cm
            - 'start': str, event marker for trial start
            - 'stop': str, optional event marker for trial end
            - 'window_oi': tuple, time window of interest in ms
            - 'trigger_msg': str, trigger message pattern in eye tracker 
               file
            - 'drift_correct': tuple, time window for drift correction 
               in ms
        missing : np.ndarray
            Array of trial numbers that were removed from behavioral 
            data during alignment in align_meta_data.
        nr_epochs : int
            Expected number of EEG epochs after behavioral alignment.
        vEOG : list of str, optional
            List of 2 channel names for vertical EOG 
            (e.g., ['FP1', 'VEOG_lower']). If provided, VEOG channel is 
            created via subtraction. Default is None.
        hEOG : list of str, optional
            List of 2 channel names for horizontal EOG 
            (e.g., ['HEOG_left', 'HEOG_right']). If provided, HEOG 
            channel is created via subtraction. Default is None.

        Returns
        -------
        tracker_data : bool
            True if eye tracker data was found and linked, False 
            otherwise.
        report_str : str
            Report describing the processing outcome, including number 
            of linked eye tracking files and epochs.

        Notes
        -----
        - EOG rereferencing creates bipolar montages: VEOG = ch1 - ch2, 
          HEOG = ch1 - ch2.
        - Only channels that exist in self.ch_names are used for EOG.
        - Eye tracker data is saved to npz file in 'eye/processed' 
          folder.
        - Trial alignment accounts for missing behavioral trials and 
          session boundaries.
        - Gaze data (x, y coordinates) is aligned to EEG epoch timing.

        Examples
        --------
        >>> # Process EOG only
        >>> tracker, report = epochs.align_eye_data(
        ...     eye_info=None,
        ...     missing=np.array([]),
        ...     nr_epochs=100,
        ...     vEOG=['Fp1', 'VEOG'],
        ...     hEOG=['HEOG_L', 'HEOG_R']
        ... )
        
        >>> # Process eye tracker data only
        >>> tracker, report = epochs.align_eye_data(
        ...     eye_info=eye_params,
        ...     missing=missing_trials,
        ...     nr_epochs=98,
        ...     vEOG=None,
        ...     hEOG=None
        ... )
        """

        report_str = 'No eye tracker data linked'

        # Optional: rereference external EOG electrodes via subtraction
        if vEOG is not None or hEOG is not None:
            # Filter to only channels that exist in the data
            vEOG = [ch for ch in vEOG if ch in self.ch_names] if vEOG else []
            hEOG = [ch for ch in hEOG if ch in self.ch_names] if hEOG else []
            
            if len(vEOG) > 0 or len(hEOG) > 0:
                eog = self.copy().pick_types(eeg=False, eog=True)
                ch_names = eog.ch_names
                
                # Get indices for vertical and horizontal EOG channels
                idx_v = [ch_names.index(v) for v in vEOG] if vEOG else []
                idx_h = [ch_names.index(h) for h in hEOG] if hEOG else []

                eog_data, eog_ch = [], []
                if len(idx_v) == 2:
                    VEOG = eog._data[:,idx_v[0]] - eog._data[:,idx_v[1]]
                    eog_data.append(VEOG)
                    eog_ch.append('VEOG')
                if len(idx_h) == 2:
                    HEOG = eog._data[:,idx_h[0]] - eog._data[:,idx_h[1]]
                    eog_data.append(HEOG)
                    eog_ch.append('HEOG')
                
                if len(eog_data) > 0:
                    eog_data =  np.stack(eog_data).swapaxes(0,1)   
                    self.add_channel_data(eog_data, eog_ch, self.info['sfreq'],
                                        'eog', self.tmin)
                    print('EOG data (VEOG, HEOG) rereferenced with subtraction '
                          'and renamed EOG channels')
            
        # Process eye tracker data if available
        if eye_info is not None:
            ext = eye_info['tracker_ext']
        else:
            ext = '.asc'
        eye_files = glob.glob(self.folder_tracker(ext = ['eye','raw'], \
                fname = f'sub_{self.sj}_session_{self.session}*'
                f'.{ext}') )
        beh_files = glob.glob(self.folder_tracker(ext=[
                'beh', 'raw'],
                fname=f'subject-{self.sj}_session_{self.session}*.csv'))
        eye_files = sorted(eye_files)
        beh_files = sorted(beh_files)

        tracker_data = False
        if len(eye_files) > 0:
            report_str = f'Found {len(eye_files)} matching eye tracking files'
            tracker_data = True
            if 'stop' not in eye_info:
                eye_info['stop'] = None
            EO = EYE(sfreq = eye_info['sfreq'],
                    viewing_dist = eye_info['viewing_dist'],
                    screen_res = eye_info['screen_res'],
                    screen_h = eye_info['screen_h'])
            
            (x,
            y,
            times,
            bins,
            angles,
            trial_inf) =EO.link_eye_to_eeg(eye_files,beh_files,
                            eye_info['start'],
                            eye_info['stop'],eye_info['window_oi'], 
                            eye_info['trigger_msg'],
                            eye_info['drift_correct'])

            # check alignment
            session_switch = np.diff(trial_inf) < 1
            if sum(session_switch) > 0:
                idx = np.where(session_switch)[0]+1
                trial_split = np.array_split(trial_inf,idx)
                for i in range(1,len(trial_split)):
                    trial_split[i] += trial_split[i-1][-1]
                trial_inf = np.hstack(trial_split)

            # check whether missing trials in trial_info
            remove = np.intersect1d(missing, trial_inf)

            idx = []
            # check whether additional trials need to be removed
            if remove.size > 0:
                _,_,idx = np.intersect1d(remove,trial_inf,return_indices=True) 
            if nr_epochs < trial_inf.size:
                if len(idx) > 0:
                    trial_inf = np.delete(trial_inf,idx)
                to_remove = trial_inf.size - nr_epochs
                if to_remove > 0:
                    idx = np.hstack((idx, 
                                    np.arange(trial_inf.size)[-to_remove:]))
                    idx = np.array(idx,dtype = int)

            bins = np.delete(bins,idx,axis = 0)
            angles = np.delete(angles, idx,axis = 0)
            x = np.delete(x, idx, axis =0)
            y = np.delete(y, idx, axis = 0)  
            # TODO: add timing check
            times /= 1000 # change to seconds

            # save eye data 
            report_str += f'\n {bins.size} eye epochs linked to EEG'
            np.savez(self.folder_tracker(ext=['eye','processed'],
                        fname=f'sj_{self.sj}_ses_{self.session}_xy_eye.npz'),
                        times = times,x = x,y = y, sfreq = eye_info['sfreq'])
 
        return tracker_data, report_str
            
    def add_channel_data(
        self,
        data: np.ndarray,
        ch_names: List[str],
        sfreq: float,
        ch_type: str,
        t_min: float
    ) -> None:
        """
        Add new channels to epochs with automatic resampling and 
        alignment.

        Creates new channels in the epochs object and fills them with 
        provided data. Handles resampling if the data sampling rate 
        differs from the EEG sampling rate. Useful for adding EOG 
        channels, eye tracker gaze data, or other auxiliary signals to 
        existing epochs.

        Parameters
        ----------
        data : np.ndarray
            Channel data to add, 
            shape (n_epochs, n_channels, n_samples). Must have same 
            number of epochs as self.
        ch_names : list of str
            Names for the new channels to add.
        sfreq : float
            Sampling rate of the input data in Hz. Data will be 
            resampled to match self.info['sfreq'] if different.
        ch_type : str
            Channel type designation for all new channels. 
            Common values: 'eeg', 'eog', 'ecg', 'emg', 'misc'.
            Determines how MNE processes and displays the channels.
        t_min : float
            Start time of the data relative to epoch time-locking event, 
            in seconds. Used to align data temporally with existing 
            epochs.

        Returns
        -------
        None
            Modifies the epochs object in-place by adding channels.

        Notes
        -----
        - New channels are appended to the end of the channel list.
        - Data is resampled using MNE's resample method if sfreq differs 
          from self.info['sfreq'].
        - The method assumes data has the same number of epochs as self.
        - Time alignment is performed using the t_min parameter to 
          determine where to place the data within the epoch time 
          window.
        - Channel names must be unique and not already exist in 
          self.ch_names.

        Examples
        --------
        >>> # Add VEOG and HEOG channels sampled at 512 Hz
        >>> eog_data = np.random.randn(100, 2, 256)  # 100 epochs, 
        ...         2 channels, 256 samples
        >>> epochs.add_channel_data(
        ...     data=eog_data,
        ...     ch_names=['VEOG', 'HEOG'],
        ...     sfreq=512,
        ...     ch_type='eog',
        ...     t_min=-0.2
        ... )
        
        >>> # Add eye tracker x/y coordinates sampled at 1000 Hz
        >>> gaze_data = np.random.randn(100, 2, 500)
        >>> epochs.add_channel_data(
        ...     data=gaze_data,
        ...     ch_names=['gaze_x', 'gaze_y'],
        ...     sfreq=1000,
        ...     ch_type='misc',
        ...     t_min=-0.1
        ... )
        """

        # create temp epochs object that allows for resampling
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
        temp_epochs = mne.EpochsArray(data, info)
        # downsample to align to eeg
        temp_epochs.resample(self.info['sfreq'])

        # now add the channels
        self.add_reference_channels(ch_names)
        mapping = {ch: ch_type for ch in ch_names}
        self.set_channel_types(mapping)

        # finally fill channels with data
        nr_samples = temp_epochs._data.shape[-1]
        time_idx = get_time_slice(self.times, t_min, 0)
        s, e = time_idx.start, time_idx.start + nr_samples
        self._data[:,-len(ch_names):,s:e] = temp_epochs._data

    def report_epochs(self, report, title, missing = None):

        if missing is not None:
            report.add_html(missing, title = 'missing trials in beh')

        report.add_epochs(self, title=title, psd = True)

        return report
    
    def save_preprocessed(
        self,
        preproc_name: str,
        combine_sessions: bool = True
    ) -> None:
        """
        Save preprocessed epochs and optionally combines multiple 
        sessions.

        Saves the current epochs object to disk in FIF format with a 
        preprocessing identifier. If matching eye tracking data exists, 
        it is renamed to match the preprocessing name. Optionally 
        combines all sessions up to the current session into a single 
        concatenated epochs file.

        Parameters
        ----------
        preproc_name : str
            Preprocessing pipeline identifier to append to filename 
            (e.g., 'clean', 'ica', 'filtered'). Used to distinguish 
            different preprocessing stages.
        combine_sessions : bool, optional
            If True and current session is not 1, combines all sessions 
            from 1 to current session into a single epochs file with 
            '_all' suffix. Default is True.

        Returns
        -------
        None
            Files are saved to disk.

        Notes
        -----
        - Individual session files are saved as: 
          'sj_{sj}_ses_{session}_{preproc_name}-epo.fif'
        - Combined session file is saved as: 
          'sj_{sj}_all_{preproc_name}-epo.fif'
        - Eye tracking data (if exists) is renamed from 
          'sj_{sj}_ses_{session}_xy_eye.npz' to 
          'sj_{sj}_ses_{session}_{preproc_name}.npz'
        - Files are split into 2GB chunks automatically to handle large 
          datasets.
        - All files are saved to the 'processed' folder managed by 
          folder_tracker.
        - Session combining uses self.session as the upper bound, so if 
          self.session=3, it will combine sessions 1, 2, and 3.

        Examples
        --------
        >>> # Save single session
        >>> epochs.save_preprocessed('clean')
        
        >>> # Save without combining sessions
        >>> epochs.save_preprocessed('ica', combine_sessions=False)
        
        >>> # Save session 3 and combine all sessions 1-3
        >>> epochs.session = '3'
        >>> epochs.save_preprocessed('filtered')  # Creates individual 
        ...         and combined files
        """

        # save eeg
        self.save(
            self.folder_tracker(
                ext=['processed'],
                fname=f'sj_{self.sj}_ses_{self.session}_{preproc_name}-epo.fif'
            ),
            split_size='2GB',
            overwrite=True
        )

        # check whether matching eye file exists and adjust name
        eye_file = self.folder_tracker(ext=['eye','processed'],
                        fname=f'sj_{self.sj}_ses_{self.session}_xy_eye.npz')
        if os.path.exists(eye_file):
            old_name = eye_file
            new_name = self.folder_tracker(ext=['eye','processed'],
                fname=f'sj_{self.sj}_ses_{self.session}_{preproc_name}.npz')
            os.rename(old_name, new_name)

        # check whether individual sessions need to be combined
        if combine_sessions and int(self.session) != 1:
            all_eeg = []
            for i in range(int(self.session)):
                session = i + 1
                all_eeg.append(
                    mne.read_epochs(
                        self.folder_tracker(
                            ext=['processed'],
                            fname=(
                                f'sj_{self.sj}_ses_{session}_'
                                f'{preproc_name}-epo.fif'
                            )
                        )
                    )
                )

            all_eeg = mne.concatenate_epochs(all_eeg)
            all_eeg.save(
                self.folder_tracker(
                    ext=['processed'],
                    fname=f'sj_{self.sj}_all_{preproc_name}-epo.fif'
                ),
                split_size='2GB',
                overwrite=True
            )


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


class ArtefactReject(object):
    """ Multiple (automatic artefact rejection procedures)
    Work in progress
    """

    def __init__(self,
                z_thresh: float = 4.0, max_bad: int = 5,
                flt_pad:Union[float,tuple,list]=0.0,filter_z: bool = True):

        self.flt_pad = flt_pad
        self.filter_z = filter_z
        self.z_thresh = z_thresh
        self.max_bad = max_bad

    def run_blink_ICA(self, fit_inst: Union[mne.Epochs, mne.io.Raw],
                    raw: mne.io.Raw, ica_inst: Union[mne.Epochs, mne.io.Raw] ,
                    sj: int, session: int, method: str = 'picard',
                    threshold: float = 0.9,
                    report: Optional[mne.Report] = None,
                    report_path: Optional[str] = None) -> Any:
        """
        Semi-automated ICA correction procedure to remove blinks. Fitting and
        applying of ICA can be done on independent data. After automatic
        component selectin, there is the possibility to manual overwrite the
        selected component after visual inspection of the report.

        Args:
            fit_inst ([mne.Epochs, mne.Raw]): Raw or Epochs mne object used to
            fit ICA
            raw (mne.Raw): Raw object used for blink detection
            ica_inst ([mne.Epochs, mne.Raw]): Raw or Epochs mne object to be
            cleaned
            sj (int): subject number
            session (int): session number
            method (str, optional): ICA method. Defaults to 'picard'.
            threshold (float, optional): threshold used for blink detection in
            eog. Defaults to 0.9.
            report ([mne.Report], optional): mne report containing ica
            overview. Defaults to None.
            report_path ([type], optional): file location of the report.
            Defaults to Optional[str]=None.

        Returns:
            [type]: [description]
        """
        # step 1: fit the data (after dropping noise trials)
        if str(type(fit_inst))[-3] == 's':
            reject = get_rejection_threshold(fit_inst, ch_types = 'eeg')
            fit_inst.drop_bad(reject)
        ica = self.fit_ICA(fit_inst, method = method)

        # step 2: select the blink component (assumed to be component 1)
        (eog_epochs,
        eog_inds,
        eog_scores) = self.automated_ica_blink_selection(ica, raw, threshold)
        ica.exclude = [eog_inds[0]] if len(eog_inds) > 0 else []
        exclude_prev = [eog_inds[0]] if len(eog_inds) > 0 else []
        manual_correct = True
        cnt = 0

        while True:
            if report is not None:
                if eog_epochs is not None:
                    eog_evoked = eog_epochs.average()
                    scores = eog_scores[0]
                else:
                    eog_evoked = None
                    scores =  None
                report.add_ica(
                    ica=ica,
                    title='ICA blink cleaning',
                    picks=range(15),
                    inst=fit_inst,
                    eog_evoked=eog_evoked,
                    eog_scores=scores,
                    )
                report.save(report_path, overwrite = True)

            if cnt > 0:
                time.sleep(5)
                tcflush(sys.stdin, TCIFLUSH)
                print('Please inspect the updated ICA report')
                conf = input(
                'Are you satisfied with the selected components? (y/n)')
                if conf == 'y':
                    manual_correct = False

            #step 2a: manually check selected component
            if manual_correct:
                ica = self.manual_check_ica(ica, sj, session)
                if ica.exclude != exclude_prev:
                    report.remove(title='ICA blink cleaning')
                else:
                    break
            else:
                break

            # track report updates
            exclude_prev = ica.exclude
            cnt += 1

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
        ica = ICA(n_components=picks.size-1, method=method,
                fit_params = fit_params, random_state=97)

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
            eog_scores = None
            print('No EOG channel is present. Cannot automate IC detection '
                'for EOG')

        return eog_epochs, eog_inds, eog_scores

    def manual_check_ica(self, ica, sj, session):

        time.sleep(5)
        tcflush(sys.stdin, TCIFLUSH)
        print('You are preprocessing subject {}, \
              session {}'.format(sj, session))
        conf = input(
            'Advanced detection selected component(s) \
                {} (see report). Do you agree (y/n)?'.format(ica.exclude))
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


    def plot_heat_map(self, channels):


        # plot heat_map
        bad = sum(np.any(self.heat_map == -1, axis = 1))
        cleaned = sum(np.any(self.heat_map == 1, axis = 1))
        fig, ax = plt.subplots(1)
        sns.despine(offset = 10)
        plt.title(f'Interpolated electrodes per marked epoch \n \
            (blue is bad epochs (N= {bad}), red is cleaned epoch \
            (N = {cleaned}))')
        ax.imshow(self.heat_map, aspect  = 'auto', cmap = 'bwr',
                    interpolation = 'nearest', vmin = -1, vmax = 1)
        ax.set(xlabel='channel', ylabel='bad epochs')
        ax.set_xticks(np.arange(channels.size))
        ax.set_xticklabels(channels, fontsize=6, rotation = 90)

        return fig

    def plot_hist_auto_repair(self, plot_type):

        if plot_type == 'cleaned':
            info = self.cleaned_info
        else:
            info = self.not_cleaned_info

        df = pd.DataFrame.from_dict(self.cleaned_info, orient = 'index',
                                        columns=['count'])
        df = df.loc[(df != 0).any(axis=1)]

        if df.size > 0: # marked at least one bad/cleaned epoch
            fig, ax = plt.subplots(1)
            sns.despine(offset = 10)
            plt.title(f'Electrode count per {plot_type} epoch')
            ax.barh(np.arange(df.values.size), np.hstack(df.values))
            ax.set_yticks(np.arange(df.values.size))
            ax.set_yticklabels(df.index, fontsize=5)
        else:
            fig = False

        return fig

    def plot_z_score_epochs(self, z_score, z_thresh):

        fig, ax = plt.subplots(1)
        plt.ylabel('accumulated Z')
        plt.xlabel('sample')
        plt.plot(np.arange(0, z_score.size), z_score.flatten(), color='b')
        plt.plot(np.arange(0, z_score.size),
                np.ma.masked_less(z_score.flatten(), z_thresh), color='r')
        plt.axhline(z_thresh, color='r', ls='--')

        return fig

    def plot_auto_repair(self, channels, z_score, z_thresh):

        figs = []

        # plot
        figs.append(self.plot_z_score_epochs(z_score, z_thresh))

        # plot heat map
        figs.append(self.plot_heat_map(channels))

        # plot histograms
        for plot in ['bad', 'cleaned']:
            fig = self.plot_hist_auto_repair(plot)
            if fig:
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

    def auto_repair_noise(self,epochs:mne.Epochs,sj:int,session:int,
                        drop_bads:bool,z_thresh:float=4.0,
                        band_pass: list =[110, 140],
                        report:mne.Report=None):

        # z score data (after hilbert transform)
        (Z, elecs_z,
        z_thresh, times) = self.preprocess_epochs(epochs, band_pass)

        # mark noise epochs
        noise_inf = self.mark_bads(Z,z_thresh,times)

        # clean epochs
        nr_bad = len(noise_inf)
        prc_bad = nr_bad/len(epochs)*100
        print(f'start iterative cleaning procedure. {nr_bad} epochs marked\
            ({prc_bad:.1f} %)')
        (epochs,
        bad_epochs,
        cleaned_epochs) = self.iterative_interpolation(epochs, elecs_z,
                          noise_inf, z_thresh, band_pass)
        picks = mne.pick_types(epochs.info, eeg=True, exclude= 'bads')

        # drop bad epochs
        if drop_bads:
            epochs.drop(np.array(bad_epochs), reason='Artefact reject')
            file = FolderStructure().folder_tracker(ext=['eye','processed'],
                        fname=f'sj_{sj}_ses_{session}_xy_eye.npz')

            if os.path.isfile(file) and len(bad_epochs) > 0:
                eye = np.load(file)
                eye_x, eye_y = eye['x'], eye['y']
                eye_x = np.delete(eye_x, np.array(bad_epochs), axis=0)
                eye_y = np.delete(eye_y, np.array(bad_epochs), axis=0)
                np.savez(file,times = eye['times'], x = eye_x, y = eye_y, 
                        sfreq = eye['sfreq'])   
        else:
            epochs.metadata.reset_index(inplace=True)
            epochs.metadata['bad_epochs'] = 0
            epochs.metadata.loc[epochs.metadata.index.isin(bad_epochs),'bad_epochs'] = 1


        if report is not None:
            channels = np.array(epochs.info['ch_names'])[picks]
            report.add_figure(self.plot_auto_repair(channels, Z, z_thresh),
                                title = 'Iterative z cleaning procedure')

        return epochs,z_thresh,report

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

    def preprocess_epochs(self,epochs:mne.Epochs,band_pass:list=[110, 140]):

        # set params
        flt_pad = self.flt_pad
        times = epochs.times
        tmin, tmax = epochs.tmin, epochs.tmax
        sfreq = epochs.info['sfreq']

        # filter data, apply hilbert (limited to 'good' EEG channels) and smooth the data (using defaults)
        if band_pass[1] > sfreq / 2:
            band_pass[1] = sfreq / 2 - 1
            UserWarning('High cutoff frequency is greater than Nyquist ' +
            'frequency. Setting to Nyquist frequency.')
        X = self.apply_hilbert(epochs, band_pass[0], band_pass[1])
        X = self.box_smoothing(X, sfreq)

        # z score data (while ignoring flt_pad samples) using default settings
        if isinstance(self.flt_pad,(float,int)):
            mask = np.logical_and(tmin + self.flt_pad <= times, times <= tmax - self.flt_pad)
            time_idx = epochs.time_as_index([tmin + flt_pad, tmax - flt_pad])
        else:
            mask = np.logical_and(tmin + self.flt_pad[0] <= times, times <= tmax - self.flt_pad[1])
            time_idx = epochs.time_as_index([tmin + flt_pad[0], tmax - flt_pad[1]])
        Z, elecs_z, z_thresh = self.z_score_data(X, self.z_thresh, mask, (self.filter_z, sfreq))

        # control for filter padding
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
            print(Z_n.shape)
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



