"""
Core EEG Data Classes for the DvM Toolbox.

This module provides the foundational classes for EEG data handling and
analysis in the DvM toolbox. Built on top of MNE-Python, it extends the
standard Raw and Epochs classes with domain-specific functionality 
tailoredfor cognitive neuroscience experiments.

The module implements a complete EEG analysis workflow including data 
loading, preprocessing, epoching, artefact rejection, and quality 
control reporting. All classes integrate with the FolderStructure system 
for organized file management and follow modern MNE-Python API 
conventions.

Main Classes
------------
RAW : Extended MNE Raw class
    Handles raw continuous EEG data with custom channel configuration,
    rereferencing, montage setup, and event extraction.

Epochs : Extended MNE Epochs class
    Manages epoched EEG data with behavioral metadata alignment,
    eye-tracking integration, and flexible trial selection.

ArtefactReject : Automatic artefact detection and repair
    Implements ICA-based blink removal and autoreject-based epoch 
    cleaning with comprehensive quality control reporting.

Key Features
------------
- BDF file reading with automatic channel type detection
- Flexible rereferencing with external reference electrode support
- Behavioral data alignment with trigger matching and validation
- Eye-tracking data integration with drift correction
- ICA-based ocular artefact removal with automated component selection
- Automatic bad epoch detection and repair using autoreject
- Comprehensive HTML quality control reports
- Filter padding to avoid edge artefacts in epoched data
- Cross-platform compatibility (Windows, macOS, Linux)

See Also
--------
eeg_analyses.preprocessing_pipeline.eeg_preprocessing_pipeline : 
    Complete preprocessing workflow. Standard preprocessing pipeline 
    that orchestrates RAW, Epochs, and ArtefactReject classes to process 
    data from raw BDF files to cleaned epochs. Includes comprehensive 
    example usage.

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
import warnings
import platform

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import convolve2d
from mne import BaseEpochs
from mne.io import BaseRaw

from typing import Dict, List, Optional, Union, Tuple, Any
from autoreject import get_rejection_threshold
from eeg_analyses.EYE import *
from math import sqrt, ceil, floor
from support.preprocessing_utils import get_time_slice, trial_exclusion
from support.FolderStructure import *
from mne.viz.epochs import plot_epochs_image
from mne.filter import filter_data
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from autoreject import Ransac, AutoReject

def flush_input():
    """
    Flush terminal input buffer in a cross-platform way.
    
    Clears any pending keyboard input to prevent stale input from 
    interfering with interactive prompts.
    """
    system = platform.system()
    if system in ['Linux', 'Darwin']:  # Unix-like systems
        try:
            from termios import tcflush, TCIFLUSH
            tcflush(sys.stdin, TCIFLUSH)
        except ImportError:
            pass  # Fallback: do nothing if termios unavailable
    elif system == 'Windows':
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except ImportError:
            pass  # Fallback: do nothing if msvcrt unavailable

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
        """
        Add raw EEG data and events to MNE report.

        Parameters
        ----------
        report : mne.Report
            MNE Report object to add content to.
        events : np.ndarray
            Event array with shape (n_events, 3).
        event_id : list or range
            Event IDs to include in the report.

        Returns
        -------
        report : mne.Report
            Updated report object.
        """
        
        # Add raw data visualization
        report.add_raw(self, title='Raw EEG', psd=True)
        
        # Add events
        events_filtered = events[np.isin(events[:, 2], event_id)]
        report.add_events(
            events_filtered, 
            title='Detected Events', 
            sfreq=self.info['sfreq']
        )

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
            picks = self.copy().pick(['eeg', 'eog'], exclude=[]).ch_names
            picks_idx = [self.ch_names.index(ch) for ch in picks]
            self._data[picks_idx, :] *= 1e6
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
            stim_picks = self.copy().pick('stim', exclude=[]).ch_names
            if len(stim_picks) == 0:
                raise ValueError('No stim channel found in data')
            stim_idx = self.ch_names.index(stim_picks[0])
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

class Epochs(mne.Epochs, BaseEpochs, FolderStructure):
    """
    Extended MNE Epochs class with behavioral data integration and 
    filtering.

    Extends `mne.Epochs` to add experimental workflow functionality 
    includingfilter padding, behavioral data alignment, eye-tracking 
    integration, and trial exclusion based on experimental factors.

    Parameters
    ----------
    sj : int
        Subject number for file tracking and organization.
    session : int
        Session number for file tracking and organization.
    raw : mne.io.Raw
        Continuous data from which to extract epochs.
    events : np.ndarray
        Event array with shape (n_events, 3) containing sample indices 
        and event IDs.
    event_id : int, list, or dict
        Event identifier(s) to extract epochs for.
    tmin : float, optional
        Start time before event (in seconds). Default is -0.2.
    tmax : float, optional
        End time after event (in seconds). Default is 0.5.
    flt_pad : float or tuple, optional
        Filter padding to add before tmin and after tmax. If tuple, 
        specifies(pad_before, pad_after). If float, same padding is 
        used for both sides. Padding is removed after filtering to avoid 
        edge artifacts.
    baseline : tuple, optional
        Baseline correction interval (start, end) in seconds.
        Default is (None, None).
    picks : str, list, or slice, optional
        Channels to include. Default is None (all channels).
    preload : bool, optional
        Whether to load data into memory. Default is True.
    reject : dict, optional
        Rejection parameters based on peak-to-peak amplitude 
        (e.g., {'eeg': 100e-6}).
    flat : dict, optional
        Rejection parameters based on flatness (e.g., {'eeg': 1e-6}).
    proj : bool, optional
        Whether to apply SSP projection vectors. Default is False.
    decim : int, optional
        Downsampling factor. Default is 1 (no downsampling).
    reject_tmin : float, optional
        Start time for rejection window (relative to epoch start).
    reject_tmax : float, optional
        End time for rejection window (relative to epoch start).
    detrend : int, optional
        Detrending order (0 for constant, 1 for linear). 
        Default is None.
    on_missing : str, optional
        How to handle missing events. Default is 'raise'.
    reject_by_annotation : bool, optional
        Whether to reject epochs based on annotations. Default is False.
    metadata : pd.DataFrame, optional
        Metadata for each epoch.
    event_repeated : str, optional
        How to handle repeated events. Default is 'error'.
    verbose : bool, str, or int, optional
        Verbosity level.

    Attributes
    ----------
    sj : int
        Subject number.
    session : str
        Session number (converted to string).
    flt_pad : float or tuple
        Filter padding specification.

    Notes
    -----
    This class extends MNE's Epochs with experimental workflow features:
    - Filter padding to avoid edge artifacts during preprocessing
    - Behavioral data alignment via `align_meta_data` method
    - Eye-tracking data integration
    - Trial exclusion based on experimental factors
    - Integration with FolderStructure for file management

    For base MNE.Epochs documentation, see:
    https://mne.tools/stable/generated/mne.Epochs.html

    Examples
    --------
    Create epochs with 0.5s filter padding:

        >>> epochs = Epochs(
        ...     sj=1, session=1, raw=raw, events=events,
        ...     event_id={'target': 1}, tmin=-0.2, tmax=0.8,
        ...     flt_pad=0.5, baseline=(-0.2, 0)
        ... )

    Create epochs with asymmetric filter padding:

        >>> epochs = Epochs(
        ...     sj=1, session=1, raw=raw, events=events,
        ...     event_id={'target': 1}, tmin=-0.2, tmax=0.8,
        ...     flt_pad=(0.3, 0.5), baseline=(-0.2, 0)
        ... )
    """

    def __init__(
        self,
        sj: int,
        session: int,
        raw: mne.io.Raw,
        events: np.ndarray,
        event_id: Union[int, list, dict],
        tmin: float = -0.2,
        tmax: float = 0.5,
        flt_pad: Union[float, tuple] = None,
        baseline: tuple = (None, None),
        picks: Union[str, list, slice] = None,
        preload: bool = True,
        reject: dict = None,
        flat: dict = None,
        proj: bool = False,
        decim: int = 1,
        reject_tmin: float = None,
        reject_tmax: float = None,
        detrend: int = None,
        on_missing: str = 'raise',
        reject_by_annotation: bool = False,
        metadata: pd.DataFrame = None,
        event_repeated: str = 'error',
        verbose: Union[bool, str, int] = None,
    ):
        """
        Initialize Epochs with filter padding and behavioral data
        integration.

        Parameters
        ----------
        sj : int
            Subject number for file tracking and organization.
        session : int
            Session number for file tracking and organization.
        raw : mne.io.Raw
            Continuous data from which to extract epochs.
        events : np.ndarray
            Event array with shape (n_events, 3) containing sample 
            indices and event IDs.
        event_id : int, list, or dict
            Event identifier(s) to extract epochs for. Can be single 
            event ID, list of IDs, or dictionary mapping names to IDs.
        tmin : float, default=-0.2
            Start time before event (in seconds).
        tmax : float, default=0.5
            End time after event (in seconds).
        flt_pad : float or tuple, optional
            Filter padding to add before tmin and after tmax to avoid 
            edge artifacts during filtering. If tuple, specifies 
            (pad_before, pad_after). If float, same padding is used 
            for both sides. Padding is removed after filtering. 
            Default is None.
        baseline : tuple, default=(None, None)
            Baseline correction interval (start, end) in seconds.
        picks : str, list, or slice, optional
            Channels to include. Default is None (all channels).
        preload : bool, default=True
            Whether to load data into memory.
        reject : dict, optional
            Rejection parameters based on peak-to-peak amplitude 
            (e.g., {'eeg': 100e-6}).
        flat : dict, optional
            Rejection parameters based on flatness (e.g., 
            {'eeg': 1e-6}).
        proj : bool, default=False
            Whether to apply SSP projection vectors.
        decim : int, default=1
            Downsampling factor (1 = no downsampling).
        reject_tmin : float, optional
            Start time for rejection window (relative to epoch start).
        reject_tmax : float, optional
            End time for rejection window (relative to epoch start).
        detrend : int, optional
            Detrending order (0 for constant, 1 for linear). 
            Default is None.
        on_missing : str, default='raise'
            How to handle missing events ('raise', 'warn', or 'ignore').
        reject_by_annotation : bool, default=False
            Whether to reject epochs based on annotations.
        metadata : pd.DataFrame, optional
            Metadata for each epoch.
        event_repeated : str, default='error'
            How to handle repeated events ('error', 'drop', or 'merge').
        verbose : bool, str, or int, optional
            Verbosity level for MNE output.

        Examples
        --------
        Create epochs with symmetric filter padding:

        >>> epochs = Epochs(
        ...     sj=1, session=1, raw=raw, events=events,
        ...     event_id={'target': 1}, tmin=-0.2, tmax=0.8,
        ...     flt_pad=0.5, baseline=(-0.2, 0)
        ... )

        Create epochs with asymmetric filter padding:

        >>> epochs = Epochs(
        ...     sj=1, session=1, raw=raw, events=events,
        ...     event_id={'target': 1, 'distractor': 2},
        ...     flt_pad=(0.3, 0.5)
        ... )
        """
        
        # Set child class specific info
        self.sj = sj
        self.session = str(session)
        self.flt_pad = flt_pad
        
        # Apply filter padding to time window if specified
        if flt_pad is not None:
            if isinstance(flt_pad, (tuple, list)):
                tmin -= flt_pad[0]
                tmax += flt_pad[1]
            else:
                tmin -= flt_pad
                tmax += flt_pad
  
        # Initialize parent MNE Epochs class
        super(Epochs, self).__init__(
            raw=raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            preload=preload,
            reject=reject,
            flat=flat,
            proj=proj,
            decim=decim,
            reject_tmin=reject_tmin,
            reject_tmax=reject_tmax,
            detrend=detrend,
            on_missing=on_missing,
            reject_by_annotation=reject_by_annotation,
            metadata=metadata,
            event_repeated=event_repeated,
            verbose=verbose,
        )

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
                eog = self.copy().pick('eog')
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
                    print('EOG data (VEOG, HEOG) rereferenced with subtraction'
                          ' and renamed EOG channels')
            
        # Process eye tracker data if available
        if eye_info is not None:
            ext = eye_info['tracker_ext']
        else:
            ext = '.asc'
        eye_files = glob.glob(self.folder_tracker(ext = ['eye','raw'], \
                fname = f'sub_{self.sj}_ses_{self.session}*'
                f'.{ext}') )
        beh_files = glob.glob(self.folder_tracker(ext=[
                'beh', 'raw'],
                fname=f'sub_{self.sj}_ses_{self.session}*.csv'))
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
          'sub_{sj}_ses_{session}_{preproc_name}-epo.fif'
        - Combined session file is saved as: 
          'sub_{sj}_all_{preproc_name}-epo.fif'
        - Eye tracking data (if exists) is renamed from 
          'sub_{sj}_ses_{session}_xy_eye.npz' to 
          'sub_{sj}_ses_{session}_{preproc_name}.npz'
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
                fname=f'sub_{self.sj}_ses_{self.session}_{preproc_name}-epo.fif'
            ),
            split_size='2GB',
            overwrite=True
        )

        # check whether matching eye file exists and adjust name
        eye_file = self.folder_tracker(ext=['eye','processed'],
                        fname=f'sub_{self.sj}_ses_{self.session}_xy_eye.npz')
        if os.path.exists(eye_file):
            old_name = eye_file
            new_name = self.folder_tracker(ext=['eye','processed'],
                fname=f'sub_{self.sj}_ses_{self.session}_{preproc_name}.npz')
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
                                f'sub_{self.sj}_ses_{session}_'
                                f'{preproc_name}-epo.fif'
                            )
                        )
                    )
                )

            all_eeg = mne.concatenate_epochs(all_eeg)
            all_eeg.save(
                self.folder_tracker(
                    ext=['processed'],
                    fname=f'sub_{self.sj}_all_{preproc_name}-epo.fif'
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
        picks = self.copy().pick('eeg', exclude='bads').ch_names
        picks = [self.ch_names.index(ch) for ch in picks]

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

    def baseline_by_condition(
        self,
        df: pd.DataFrame,
        cnds: list,
        cnd_header: str,
        base_period: tuple = (-0.1, 0),
        nr_elec: int = 64
    ) -> 'Epochs':
        """
        Apply baseline correction separately for each condition.

        Performs condition-specific baseline correction when different
        conditions require different baseline periods or when baseline
        should be computed within-condition rather than globally.

        Parameters
        ----------
        beh : pd.DataFrame
            Behavioral data with condition labels for each epoch.
        cnds : list
            List of condition values to process. Must exist in 
            beh[cnd_header].
        cnd_header : str
            Column name in beh DataFrame containing condition labels.
        base_period : tuple, default=(-0.1, 0)
            Baseline time window in seconds (start, end).
        nr_elec : int, default=64
            Number of EEG electrodes to baseline correct. Non-EEG 
            channels (EOG, etc.) after this index are not corrected.

        Returns
        -------
        self : Epochs
            Returns self for method chaining.

        Warnings
        --------
        Modifies EEG data in-place! Cannot be undone without reloading.

        This is an advanced function. Standard baseline correction via
        mne.Epochs(..., baseline=(-0.1, 0)) is recommended for most 
        cases.

        Notes
        -----
        For each condition:
        1. Selects trials matching that condition
        2. Computes mean baseline per electrode across those trials
        3. Subtracts baseline from all electrodes in those trials

        Use cases:
        - Different experimental phases require different baselines
        - Condition-specific artifacts affect baseline
        - Within-condition normalization needed

        Examples
        --------
        >>> # Apply -100 to 0ms baseline per condition
        >>> epochs.baseline_by_condition(
        ...     beh=beh,
        ...     cnds=['cue_left', 'cue_right'],
        ...     cnd_header='condition',
        ...     base_period=(-0.1, 0)
        ... )

        See Also
        --------
        shift_by_condition : Shift timing by condition
        """
        # Select indices baseline period
        start, end = [np.argmin(abs(self.times - b)) for b in base_period]

        # Loop over conditions
        for cnd in cnds:
            # Get indices of interest
            idx = np.where(df[cnd_header] == cnd)[0]

            # Get data
            X = self._data[idx, :nr_elec]
            # Get base_mean (per electrode)
            X_base = X[:, :, start:end].mean(axis=(0, 2))

            # Do baselining per electrode
            for i in range(nr_elec):
                X[:, i, :] -= X_base[i]

            self._data[idx, :nr_elec] = X

        return self

    def shift_by_condition(
        self,
        beh: pd.DataFrame,
        cnd_info: dict,
        cnd_header: str
    ) -> 'Epochs':
        """
        Shift epoch timings by condition to align events of interest.

        Artificially shifts the time series data for specific conditions
        when events of interest occur at different latencies across
        conditions. Useful for aligning responses that vary in timing.

        Parameters
        ----------
        df : pd.DataFrame
            Behavioral data with condition labels.
        cnd_info : dict
            Dictionary mapping condition values to shift amounts in 
            seconds. Positive values shift forward (delay), negative 
            shift backward. Example: {'fast': -0.1, 'slow': 0.1}
        cnd_header : str
            Column name in beh containing condition labels (keys from 
            cnd_info).

        Returns
        -------
        self : Epochs
            Returns self for method chaining.

        Warnings
        --------
        **DATA IS ARTIFICIALLY MANIPULATED!**
        
        - Modifies EEG data in-place
        - Shifts are circular (uses np.roll) - edge effects occur!
        - Cannot be undone without reloading
        - Be extremely careful selecting analysis windows
        - Original timing information is lost

        Notes
        -----
        The function prints warnings about data manipulation and shows
        original timing range.

        Shifting is done via np.roll which wraps around 
        (circular shift):
        
        - Forward shift: early samples wrap to end
        - Backward shift: late samples wrap to beginning

        Consider alternatives:
        
        - Re-epoch data with condition-specific time windows
        - Use different event codes for analysis
        - Align on different events during preprocessing

        Examples
        --------
        >>> # Shift slow trials backward by 100ms, fast forward by 100ms
        >>> epochs.shift_by_condition(
        ...     beh=beh,
        ...     cnd_info={'slow': -0.1, 'fast': 0.1},
        ...     cnd_header='rt_bin'
        ... )
        Data will be artificially shifted...
        Original timings range from -0.2 to 1.0
        EEG data is shifted backward in time for all slow trials
        EEG data is shifted forward in time for all fast trials

        See Also
        --------
        baseline_by_condition : Condition-specific baseline correction
        """

        print('Data will be artificially shifted. Be careful in selecting '
              'the window of interest for further analysis')
        print(f'Original timings range from {self.tmin} to {self.tmax}')
        
        # Loop over all conditions
        for cnd in cnd_info.keys():
            # Set how much data needs to be shifted
            to_shift = cnd_info[cnd]
            to_shift = int(np.diff([np.argmin(abs(self.times - t)) 
                                   for t in (0, to_shift)]))
            
            if to_shift < 0:
                print(f'EEG data is shifted backward in time for all '
                      f'{cnd} trials')
            elif to_shift > 0:
                print(f'EEG data is shifted forward in time for all '
                      f'{cnd} trials')

            # Find indices of epochs to shift
            mask = (beh[cnd_header] == cnd).values

            # Do actual shifting
            self._data[mask] = np.roll(self._data[mask], to_shift, axis=2)

        return self

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

    def run_blink_ICA(
        self,
        fit_inst: Union[mne.Epochs, mne.io.Raw],
        raw: mne.io.Raw,
        ica_inst: Union[mne.Epochs, mne.io.Raw],
        sj: int,
        session: int,
        method: str = 'picard',
        threshold: float = 0.9,
        report: Optional[mne.Report] = None,
        report_path: Optional[str] = None
    ) -> Union[mne.Epochs, mne.io.Raw]:
        """
        Semi-automated ICA correction to remove blink and other artifacts.

        Fits ICA on clean epochs, automatically detects blink components 
        using EOG correlation, allows manual verification/override, and 
        applies ICA to remove artifacts. While designed primarily for 
        blink detection, manual verification allows exclusion of other 
        artifact components (e.g., heartbeat, muscle activity, electrode 
        noise) identified through visual inspection of the ICA report. 
        Fitting and application can use independent data instances 
        (e.g., fit on subset, apply to all).

        Parameters
        ----------
        fit_inst : mne.Epochs or mne.io.Raw
            Data instance to fit ICA model on. If Epochs, noisy trials 
            are automatically dropped using autoreject before fitting.
        raw : mne.io.Raw
            Raw data instance for EOG-based blink detection. Must 
            contain EOG channels.
        ica_inst : mne.Epochs or mne.io.Raw
            Data instance to apply ICA cleaning to. Can be same as or 
            different from fit_inst.
        sj : int
            Subject number for user prompts and identification.
        session : int
            Session number for user prompts and identification.
        method : str, optional
            ICA decomposition algorithm. Options: 'picard' (default, 
            fastest), 'fastica', 'extended_infomax'. Default is 
            'picard'.
        threshold : float, optional
            Correlation threshold for automatic blink component 
            detection (0-1). Higher values require stronger EOG 
            correlation. Default is 0.9.
        report : mne.Report, optional
            MNE report object to add ICA visualizations to. If None, no 
            report is generated. Default is None.
        report_path : str, optional
            File path to save report. Required if report is provided. 
            Default is None.

        Returns
        Notes
        -----
        - Interactive workflow: automatically suggests components, user 
          can verify/override via terminal prompts
        - If Epochs provided for fitting, uses autoreject to drop noisy 
          trials before ICA fit
        - Reports show component topographies, time courses, and EOG 
          correlations for all components (not just blink-related)
        - Manual override loop continues until user confirms selection
        - Typical workflow: fit on clean subset of trials, apply to all 
          trials
        - Users can manually select additional components beyond 
          automatically detected blink artifacts (e.g., heartbeat, 
          muscle artifacts) by inspecting component topographies and 
          time courses in the report

        Examples override loop continues until user confirms selection
        - Typical workflow: fit on clean subset of trials, apply to all 
          trials

        Examples
        --------
        >>> # Basic usage with automatic component selection
        >>> ar = ArtefactReject()
        >>> epochs_clean = ar.run_blink_ICA(
        ...     fit_inst=epochs,
        ...     raw=raw,
        ...     ica_inst=epochs,
        ...     sj=1,
        ...     session=1
        ... )
        
        >>> # With report generation
        >>> report = mne.Report()
        >>> epochs_clean = ar.run_blink_ICA(
        ...     fit_inst=epochs,
        ...     raw=raw,
        ...     ica_inst=epochs,
        ...     sj=1,
        ...     session=1,
        ...     report=report,
        ...     report_path='reports/ica_sj01.html'
        ... )
        """
        # step 1: fit the data (after dropping noise trials if Epochs)
        if isinstance(fit_inst, mne.Epochs):
            reject = get_rejection_threshold(fit_inst, ch_types='eeg')
            fit_inst.drop_bad(reject)
        ica = self.fit_ICA(fit_inst, method=method)

        # step 2: select the blink component
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
                else:
                    eog_evoked = None
                report.add_ica(
                    ica=ica,
                    title='ICA blink cleaning',
                    picks=range(15),
                    inst=fit_inst,
                    eog_evoked=eog_evoked,
                    eog_scores=eog_scores,
                )
                report.save(report_path, overwrite=True)

            if cnt > 0:
                time.sleep(5)
                flush_input()
                print('Please inspect the updated ICA report')
                conf = input(
                    'Are you satisfied with the selected components? (y/n)'
                )
                if conf == 'y':
                    manual_correct = False

            # step 2a: manually check selected component
            if manual_correct:
                ica = self.manual_check_ica(ica, sj, session)
                if ica.exclude != exclude_prev:
                    if report is not None:
                        report.remove(title='ICA blink cleaning')
                    exclude_prev = ica.exclude
                else:
                    break
            else:
                break

            # track report updates
            cnt += 1

        # step 3: apply ica
        ica_inst = self.apply_ICA(ica, ica_inst)

        return ica_inst

    def fit_ICA(
        self,
        fit_inst: Union[mne.Epochs, mne.io.Raw],
        method: str = 'picard'
    ) -> ICA:
        """
        Fit ICA decomposition model on EEG data.

        Decomposes EEG data into independent components using specified 
        ICA algorithm. Excludes bad channels from decomposition.

        Parameters
        ----------
        fit_inst : mne.Epochs or mne.io.Raw
            Data to fit ICA on. Should be adequately preprocessed 
            (filtered, bad channels marked).
        method : str, optional
            ICA algorithm to use:
            - 'picard': Fast ICA with preconditioning (recommended)
            - 'fastica': FastICA algorithm
            - 'extended_infomax': Extended Infomax algorithm
            Default is 'picard'.

        Returns
        -------
        ica : mne.preprocessing.ICA
            Fitted ICA object with n_components = n_channels - 1.

        Notes
        -----
        - Number of components is n_good_channels - 1
        - Bad channels (in fit_inst.info['bads']) are excluded
        - Random state is fixed at 97 for reproducibility
        - Picard method uses fastica_it=5 for preconditioning

        Examples
        --------
        >>> ar = ArtefactReject()
        >>> ica = ar.fit_ICA(epochs, method='picard')
        >>> print(f'Fitted {ica.n_components_} components')
        """
        if method == 'picard':
            fit_params = dict(fastica_it=5)
        elif method == 'extended_infomax':
            fit_params = dict(extended=True)
        elif method == 'fastica':
            fit_params = None
        else:
            raise ValueError(
                f"Unknown ICA method '{method}'. "
                "Choose from 'picard', 'fastica', or 'extended_infomax'."
            )

        picks = fit_inst.copy().pick('eeg', exclude='bads').ch_names
        ica = ICA(
            n_components=len(picks) - 1,
            method=method,
            fit_params=fit_params,
            random_state=97
        )
        ica.fit(fit_inst, picks=picks)

        return ica

    def automated_ica_blink_selection(
        self,
        ica: ICA,
        raw: mne.io.Raw,
        threshold: float = 0.9
    ) -> Tuple[Optional[mne.Epochs], List[int], Optional[np.ndarray]]:
        """
        Automatically detect ICA components correlated with blinks.

        Identifies ICA components that correlate with EOG channels above 
        a threshold, indicating blink artifacts. Creates blink-locked 
        epochs from EOG and correlates with ICA components.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Fitted ICA object to search for blink components.
        raw : mne.io.Raw
            Raw data containing EOG channels for blink detection.
        threshold : float, optional
            Absolute correlation threshold (0-1) for component 
            selection. Components with |correlation| > threshold are 
            flagged. Default is 0.9.

        Returns
        -------
        eog_epochs : mne.Epochs or None
            Blink-locked epochs created from EOG channel, or None if no 
            EOG channels found.
        eog_inds : list of int
            Indices of ICA components exceeding correlation threshold.
        eog_scores : ndarray or None
            Correlation scores for each component, or None if no EOG 
            channels found.

        Notes
        -----
        - Requires at least one EOG channel in raw.info['chs']
        - Blink epochs: -0.5 to 0.5 s, baseline -0.5 to -0.2 s
        - Returns empty list for eog_inds if no components exceed 
          threshold
        - Prints warning if no EOG channels are available

        Examples
        --------
        >>> ar = ArtefactReject()
        >>> ica = ar.fit_ICA(epochs)
        >>> eog_epochs, inds, scores = ar.automated_ica_blink_selection(
        ...     ica, raw, threshold=0.9
        ... )
        >>> print(f'Found {len(inds)} blink components')
        """
        ch_names = raw.copy().pick('eog').ch_names
        pick_eog = [raw.ch_names.index(ch) for ch in ch_names]

        if pick_eog.any():
            # create blink epochs
            eog_epochs = create_eog_epochs(
                raw,
                ch_name=ch_names,
                baseline=(None, -0.2),
                tmin=-0.5,
                tmax=0.5
            )
            eog_inds, eog_scores = ica.find_bads_eog(
                eog_epochs,
                threshold=threshold
            )
        else:
            eog_epochs = None
            eog_inds = []
            eog_scores = None
            print(
                'No EOG channel is present. Cannot automate IC detection '
                'for EOG'
            )

        return eog_epochs, eog_inds, eog_scores

    def manual_check_ica(
        self,
        ica: ICA,
        sj: int,
        session: int
    ) -> ICA:
        """
        Allow manual verification and override of ICA component 
        selection.

        Prompts user via terminal to confirm or override automatically 
        detected ICA components. User can specify custom component 
        indices to exclude.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA object with automatically selected components in 
            ica.exclude.
        sj : int
            Subject number for display in prompts.
        session : int
            Session number for display in prompts.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA object with potentially updated exclude list based on 
            user input.

        Notes
        -----
        - Clears input buffer before prompting to avoid stale input
        - User should inspect ICA report before responding
        - If user disagrees, prompts for number of components and their 
          indices
        - Component indices are 0-based

        Examples
        --------
        >>> ar = ArtefactReject()
        >>> ica = ar.fit_ICA(epochs)
        >>> ica.exclude = [0, 5]  # Automatically detected
        >>> ica = ar.manual_check_ica(ica, sj=1, session=1)
        # User prompted: "Advanced detection selected component(s) 
        # [0, 5]. Do you agree (y/n)?"
        """
        time.sleep(5)
        flush_input()
        print(f'You are preprocessing subject {sj}, session {session}')
        conf = input(
            f'Advanced detection selected component(s) {ica.exclude} '
            '(see report). Do you agree (y/n)? '
        )
        if conf == 'n':
            eog_inds = []
            nr_comp = input(
                'How many components do you want to select (<10)? '
            )
            for i in range(int(nr_comp)):
                eog_inds.append(
                    int(input(f'What is component nr {i + 1}? '))
                )
            ica.exclude = eog_inds

        return ica


    def apply_ICA(
        self,
        ica: ICA,
        ica_inst: Union[mne.Epochs, mne.io.Raw]
    ) -> Union[mne.Epochs, mne.io.Raw]:
        """
        Apply ICA to remove selected artifact components from data.

        Removes ICA components specified in ica.exclude from the data 
        by reconstructing the signal without those components.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Fitted ICA object with components to exclude specified in 
            ica.exclude.
        ica_inst : mne.Epochs or mne.io.Raw
            Data to apply ICA cleaning to. Must have same channel 
            configuration as data used to fit ICA.

        Returns
        -------
        ica_inst : mne.Epochs or mne.io.Raw
            Cleaned data with specified ICA components removed.

        Notes
        -----
        - Modifies ica_inst in-place
        - Components in ica.exclude are removed from the reconstruction
        - If ica.exclude is empty, returns data unchanged

        Examples
        --------
        >>> ar = ArtefactReject()
        >>> ica = ar.fit_ICA(epochs)
        >>> ica.exclude = [0, 5]  # Blink and heartbeat components
        >>> epochs_clean = ar.apply_ICA(ica, epochs)
        """
        ica_inst = ica.apply(ica_inst)
        return ica_inst

    def auto_repair_noise(
        self,
        epochs: mne.Epochs,
        sj: int,
        session: int,
        drop_bads: bool,
        z_thresh: float = 4.0,
        band_pass: List[int] = [110, 140],
        report: Optional[mne.Report] = None
    ) -> Tuple[mne.Epochs, float, Optional[mne.Report]]:
        """
        Detect and repair high-frequency noise artifacts using iterative 
        channel interpolation.

        Implements FieldTrip's artifact detection procedure with an 
        intelligent repair strategy. Detects muscle/noise artifacts by 
        filtering in high-frequency band (110-140 Hz), applying Hilbert 
        transform, and z-scoring across channels. Epochs with artifacts 
        are iteratively cleaned by interpolating the noisiest channels 
        (up to max_bad). Epochs that cannot be cleaned are either 
        dropped or marked in metadata.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data to clean. Modified in-place.
        sj : int
            Subject number for eye tracking file synchronization.
        session : int
            Session number for eye tracking file synchronization.
        drop_bads : bool
            If True, drop epochs that cannot be cleaned. If False, mark 
            them in epochs.metadata['bad_epochs'] column (0=good, 
            1=bad).
        z_thresh : float, optional
            Starting z-value threshold for artifact detection. Final 
            threshold is data-driven: 
            z_thresh + median(Z) + |min(Z) - median(Z)|. 
            Default is 4.0.
        band_pass : list of int, optional
            [lower, upper] frequency bounds in Hz for bandpass filter. 
            Default is [110, 140] for muscle artifact detection.
        report : mne.Report, optional
            MNE report object to add visualization figures to. If None, 
            no figures are generated. Default is None.

        Returns
        -------
        epochs : mne.Epochs
            Cleaned epochs with artifacts removed via interpolation or 
            epoch dropping.
        z_thresh : float
            Final data-driven z-threshold used for detection.
        report : mne.Report or None
            Updated report with artifact detection figures, or None if 
            no report provided.

        Notes
        -----
        - Detection algorithm (3 steps):
          1. Bandpass filter → Hilbert envelope → box smoothing (0.2s)
          2. Z-score across channels, normalize by √(n_channels)
          3. Data-driven threshold adjustment
        - Cleaning strategy: For each flagged epoch, iteratively 
          interpolate noisiest channels (up to self.max_bad), 
          re-checking after each interpolation
        - Eye tracking synchronization: If drop_bads=True and eye 
          tracking data exists, corresponding eye epochs are deleted
        - Visualizations in report: z-score time series, channel 
          interpolation heatmap, interpolation histograms
        - Based on FieldTrip tutorial:
          https://www.fieldtriptoolbox.org/tutorial/
          automatic_artifact_rejection/

        Examples
        --------
        >>> # Drop bad epochs and generate report
        >>> ar = ArtefactReject(z_thresh=4.0, max_bad=5)
        >>> report = mne.Report()
        >>> epochs_clean, thresh, report = ar.auto_repair_noise(
        ...     epochs, sj=1, session=1, drop_bads=True, report=report
        ... )
        
        >>> # Mark bad epochs in metadata without dropping
        >>> epochs_clean, thresh, _ = ar.auto_repair_noise(
        ...     epochs, sj=1, session=1, drop_bads=False
        ... )
        >>> bad_count = epochs_clean.metadata['bad_epochs'].sum()
        """

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
        picks = epochs.copy().pick('eeg', exclude='bads').ch_names
        picks = [epochs.ch_names.index(ch) for ch in picks]

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
            epochs.metadata.loc[
                epochs.metadata.index.isin(bad_epochs), 'bad_epochs'
            ] = 1


        if report is not None:
            channels = np.array(epochs.info['ch_names'])[picks]
            report.add_figure(self.plot_auto_repair(channels, Z, z_thresh),
                                title = 'Iterative z cleaning procedure')

        return epochs, z_thresh, report
    
    def preprocess_epochs(
        self,
        epochs: mne.Epochs,
        band_pass: List[int] = [110, 140]
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Preprocess epochs for artifact detection using FieldTrip method.

        Applies bandpass filtering, Hilbert transform with envelope 
        extraction, box smoothing, and z-scoring across channels. 
        Handles filter padding to exclude edge artifacts from analysis.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data to preprocess for artifact detection.
        band_pass : list of int, optional
            [lower, upper] frequency bounds in Hz for bandpass filter. 
            Upper bound is automatically adjusted if it exceeds Nyquist 
            frequency. Default is [110, 140].

        Returns
        -------
        Z : np.ndarray
            2D array of normalized z-scores across channels, 
            shape (n_epochs, n_times). Excludes filter padding samples.
        elecs_z : np.ndarray
            3D array of z-scores per electrode, 
            shape (n_epochs, n_channels, n_times). Excludes filter 
            padding samples.
        z_thresh : float
            Data-driven z-score threshold calculated as: 
            self.z_thresh + median(Z) + |min(Z) - median(Z)|.
        times : np.ndarray
            Time points in seconds corresponding to returned data, 
            excluding filter padding.

        Notes
        -----
        - Processing pipeline:
          1. Bandpass filter at specified frequencies
          2. Hilbert transform with envelope extraction
          3. Box smoothing with 0.2s window
          4. Z-score across channels, normalize by √(n_channels)
          5. Data-driven threshold adjustment
        - Filter padding (self.flt_pad) is excluded from z-scoring and 
          returned data to avoid edge artifacts
        - If band_pass upper bound exceeds Nyquist frequency 
          (sfreq/2), it's automatically reduced to Nyquist - 1 Hz
        - Bad channels (in epochs.info['bads']) are excluded from 
          processing

        Examples
        --------
        >>> ar = ArtefactReject(flt_pad=0.1)
        >>> Z, elecs_z, thresh, times = ar.preprocess_epochs(
        ...     epochs, band_pass=[110, 140]
        ... )
        >>> print(f'Threshold: {thresh:.2f}, Shape: {Z.shape}')
        """

        # set params
        flt_pad = self.flt_pad
        times = epochs.times
        tmin, tmax = epochs.tmin, epochs.tmax
        sfreq = epochs.info['sfreq']

        # filter, apply hilbert, and smooth the data
        if band_pass[1] > sfreq / 2:
            band_pass[1] = sfreq / 2 - 1
            warnings.warn(
                'High cutoff frequency is greater than Nyquist '
                'frequency. Setting to Nyquist frequency.'
            )
        X = self.apply_hilbert(epochs, band_pass[0], band_pass[1])
        X = self.box_smoothing(X, sfreq)

        # z score data (while ignoring flt_pad samples)
        if isinstance(self.flt_pad, (float, int)):
            mask = np.logical_and(
                tmin + self.flt_pad <= times,
                times <= tmax - self.flt_pad
            )
            time_idx = epochs.time_as_index(
                [tmin + flt_pad, tmax - flt_pad]
            )
        else:
            mask = np.logical_and(
                tmin + self.flt_pad[0] <= times,
                times <= tmax - self.flt_pad[1]
            )
            time_idx = epochs.time_as_index(
                [tmin + flt_pad[0], tmax - flt_pad[1]]
            )
        Z, elecs_z, z_thresh = self.z_score_data(
            X, self.z_thresh, mask, (self.filter_z, sfreq)
        )

        # control for filter padding
        Z = Z[:, slice(*time_idx)]
        elecs_z = elecs_z[:, :, slice(*time_idx)]
        times = times[slice(*time_idx)]

        return Z, elecs_z, z_thresh, times

    def apply_hilbert(
        self, 
        epochs: mne.Epochs, 
        lower_band: int = 110, 
        upper_band: int = 140
    ) -> np.ndarray:
        """
        Apply Hilbert transform to extract signal envelope in specified 
        frequency band.

        This method filters epochs to the specified frequency band, 
        applies the Hilbert transform, and extracts the envelope 
        (amplitude) of the analytic signal. Used primarily for muscle 
        artifact detection in high-frequency bands (e.g., 110-140 Hz).

        Parameters
        ----------
        epochs : mne.Epochs
            MNE epochs object containing EEG data. Channels marked as 
            'bads' will be excluded from processing.
        lower_band : int, optional
            Lower cutoff frequency of bandpass filter in Hz. 
            Default is 110.
        upper_band : int, optional
            Upper cutoff frequency of bandpass filter in Hz. 
            Default is 140.

        Returns
        -------
        X : np.ndarray
            Envelope data after Hilbert transform, 
            shape (n_epochs, n_channels, n_times). Contains amplitude 
            values representing signal envelope in the specified 
            frequency band.

        Notes
        -----
        - Processing steps:
          1. Copy epochs and exclude bad channels
          2. Bandpass filter to specified frequency range (FIR filter 
             with 'firwin' design)
          3. Apply Hilbert transform with envelope extraction
          4. Extract and return data array
        - The filter uses 'reflect_limited' padding to minimize edge 
           artifacts.
        - Original epochs object is not modified.

        Examples
        --------
        >>> # Extract muscle artifact envelope (110-140 Hz)
        >>> envelope_data = self.apply_hilbert(epochs, 110, 140)
        >>> print(f'Envelope shape: {envelope_data.shape}')
        
        >>> # Use custom frequency band
        >>> envelope_data = self.apply_hilbert(epochs, 50, 70)
        """

        # exclude channels that are marked as overall bad
        epochs_ = epochs.copy()
        epochs_.pick('eeg', exclude='bads')

        # filter data and apply Hilbert
        epochs_.filter(
            lower_band, 
            upper_band, 
            fir_design='firwin', 
            pad='reflect_limited'
        )
        epochs_.apply_hilbert(envelope=True)

        # get data
        X = epochs_.get_data()
        del epochs_

        return X
    
    def filt_pad(self, X: np.ndarray, pad_length: int) -> np.ndarray:
        """
        Pad data using local mean method for filtering edge artifact 
        reduction.

        Adds samples before and after the data by computing the mean of 
        edge regions and replicating those means. This prevents edge 
        artifacts when applying filters. Based on FieldTrip's 
        ft_preproc_padding.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_channels, n_times) containing signal 
            data.
        pad_length : int
            Number of samples to pad on each edge. Actual padding will 
            be limited to half the signal length if pad_length is too 
            large.

        Returns
        -------
        X_padded : np.ndarray
            2D array of shape (n_channels, n_times + 2*pre_pad) with 
            padded data.

        Notes
        -----
        - Padding length is limited to floor(n_times / 2) to prevent 
          excessive padding relative to signal length.
        - Left edge: padded with mean of first pre_pad samples
        - Right edge: padded with mean of last pre_pad samples
        - Based on FieldTrip implementation:
          https://github.com/fieldtrip/fieldtrip/blob/master/
          preproc/ft_preproc_padding.m

        Examples
        --------
        >>> # Pad 64-channel EEG data with 100 samples on each side
        >>> X = np.random.randn(64, 1000)  # 64 channels, 1000 samples
        >>> X_padded = self.filt_pad(X, pad_length=100)
        >>> print(X_padded.shape)  # (64, 1200)
        """

        # set number of pad samples
        pre_pad = int(min([pad_length, floor(X.shape[1]) / 2.0]))

        # get local mean on both sides
        edge_left = X[:, :pre_pad].mean(axis=1)
        edge_right = X[:, -pre_pad:].mean(axis=1)

        # pad data
        X = np.concatenate(
            (np.tile(edge_left.reshape(X.shape[0], 1), pre_pad), 
             X, 
             np.tile(edge_right.reshape(X.shape[0], 1), pre_pad)), 
            axis=1
        )

        return X

    def box_smoothing(
        self, 
        X: np.ndarray, 
        sfreq: float, 
        boxcar: float = 0.2
    ) -> np.ndarray:
        """
        Apply boxcar (moving average) smoothing to epoched data.

        Smooths data using a uniform kernel of specified length. Each 
        epoch is padded before smoothing to reduce edge artifacts, then 
        trimmed back to original length. Based on FieldTrip's 
        ft_preproc_smooth.

        Parameters
        ----------
        X : np.ndarray
            3D array of shape (n_epochs, n_channels, n_times) containing 
            data to be smoothed.
        sfreq : float
            Sampling frequency in Hz, used to convert boxcar duration to 
            samples.
        boxcar : float, optional
            Duration of smoothing kernel in seconds. Default is 0.2 
            seconds (optimal according to FieldTrip documentation for 
            muscle artifact detection).

        Returns
        -------
        X_smoothed : np.ndarray
            3D array of shape (n_epochs, n_channels, n_times) with 
            smoothed data.

        Notes
        -----
        - Kernel length is computed as round(boxcar * sfreq) and forced 
          to be odd.
        - Each epoch is independently:
          1. Padded using local mean method (via filt_pad)
          2. Convolved with uniform kernel
          3. Trimmed to remove padding
        - Based on FieldTrip implementation: 
          https://www.fieldtriptoolbox.org

        Examples
        --------
        >>> # Smooth 30 epochs of 64-channel data with 200ms boxcar
        >>> X = np.random.randn(30, 64, 1000)  # epochs by ch by samples
        >>> X_smooth = self.box_smoothing(X, sfreq=500.0, boxcar=0.2)
        >>> print(X_smooth.shape)  # (30, 64, 1000)
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
            x = self.filt_pad(x, pad_length=pad_length)

            # smooth the data
            x_smooth = convolve2d(
                x, 
                kernel.reshape(1, kernel.shape[0]), 
                'same'
            )
            X[i] = x_smooth[:, pad_length:(x_smooth.shape[1] - pad_length)]

        return X
    
    def mark_bads(
        self, 
        Z: np.ndarray, 
        z_thresh: float, 
        times: np.ndarray
    ) -> List[list]:
        """
        Identify epochs containing samples exceeding z-score threshold.

        Detects continuous segments of artifacts in z-scored data and 
        returns detailed information about each artifact event including 
        timing and duration. Used for marking epochs with muscle 
        artifacts or other high-amplitude noise.

        Parameters
        ----------
        Z : np.ndarray
            2D array of shape (n_epochs, n_times) containing normalized 
            z-scores accumulated across channels. Output from 
            z_score_data method.
        z_thresh : float
            Z-score threshold for artifact detection. 
            Samples with |z| > z_thresh are marked as artifacts. 
            Typical values: 4.0-6.0.
        times : np.ndarray
            1D array of time points in seconds corresponding to samples 
            in Z.

        Returns
        -------
        noise_events : List[list]
            List of epochs containing artifacts. Each element is a list 
            with:
            - First element: epoch index (int)
            - Subsequent elements: tuples for each artifact segment 
              containing:
              (slice_obj, start_time, end_time, duration) where 
              slice_obj is slice(start_idx, end_idx)

        Notes
        -----
        - Only epochs with at least one artifact segment are included in 
          output.
        - Continuous artifact segments are identified using threshold 
          crossings.
        - Edge cases handled:
          - Artifact at epoch start: included from index 0
          - Artifact at epoch end: included through last sample
        - Multiple artifact segments per epoch are all logged.

        Examples
        --------
        >>> # Mark epochs with z-scores exceeding threshold of 4.0
        >>> Z = np.random.randn(30, 1000)  # 30 epochs, 1000 time points
        >>> times = np.linspace(-0.5, 1.5, 1000)
        >>> noise_events = self.mark_bads(Z, z_thresh=4.0, times=times)
        >>> 
        >>> # Inspect first bad epoch
        >>> if noise_events:
        ...     ep_idx, *artifacts = noise_events[0]
        ...     for art_slice, t_start, t_end, duration in artifacts:
        ...         print(
        ...             f'Epoch {ep_idx}: artifact '
        ...             f'{t_start:.3f}-{t_end:.3f}s '
        ...             f'(duration: {duration:.3f}s)'
        ...         )
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
                ep_noise = [ep] + [
                    (slice(s, e), times[s], times[e], abs(times[e] - times[s])) 
                    for s, e in zip(starts, ends)
                ]
                noise_events.append(ep_noise)

        return noise_events
    
    def z_score_data(
        self, 
        X: np.ndarray, 
        z_thresh: float = 4.0, 
        mask: Optional[np.ndarray] = None, 
        filter_z: Tuple[bool, float] = (False, 512)
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute z-scores across channels with data-driven threshold 
        adjustment.

        Z-scores data across the channel dimension (axis=1), accumulates 
        z-scores across channels, and calculates a data-driven threshold 
        based on the distribution of z-scores. Optionally applies 
        temporal masking and low-pass filtering to reduce false 
        positives.

        Parameters
        ----------
        X : np.ndarray
            3D array of shape (n_epochs, n_channels, n_times) containing
            envelope or filtered data.
        z_thresh : float, optional
            Starting z-value threshold for artifact detection. Will be 
            adjusted upward based on data distribution. Default is 4.0.
        mask : np.ndarray, optional
            Boolean array of shape (n_times,) indicating time points to 
            include in z-score calculation. Masked-out points are still 
            returned in output but excluded from statistics. Used to 
            ignore filter padding. Default is None (all time points 
            included).
        filter_z : tuple of (bool, float), optional
            If first element is True, applies 4 Hz low-pass filter to 
            z-score time series to reduce transient false positives. 
            Second element is sampling frequency in Hz. Should only be 
            used with filter-padded data. Default is (False, 512).

        Returns
        -------
        Z_n : np.ndarray
            2D array of shape (n_epochs, n_times) containing normalized 
            z-scores accumulated across channels.
        elecs_z : np.ndarray
            3D array of shape (n_epochs, n_channels, n_times) 
            containing z-scores per channel. Used for identifying 
            noisiest channels during iterative interpolation.
        z_thresh : float
            Data-driven z-score threshold calculated as: 
            z_thresh + median(Z) + |min(Z) - median(Z)|. 
            Adjusts for baseline noise level in the data.

        Notes
        -----
        - Processing steps:
          1. Reshape data to (n_channels, n_epochs * n_times)
          2. Z-score along time dimension (axis=1) for each channel
          3. Accumulate z-scores across channels, normalize 
             by √(n_channels)
          4. Optionally low-pass filter at 4 Hz
          5. Calculate data-driven threshold adjustment
        - The normalization by √(n_channels) accounts for accumulation 
          across independent observations
        - Data-driven threshold prevents over-sensitivity in low-noise
          data and under-sensitivity in high-noise data
        - If mask is provided, z-scoring is recomputed only on masked 
          time points to ensure accurate statistics

        Examples
        --------
        >>> # Basic usage without masking
        >>> ar = ArtefactReject()
        >>> X = np.random.randn(30, 64, 1000)  # 30 epochs, 64 channels
        >>> Z, elecs_z, thresh = ar.z_score_data(X, z_thresh=4.0)
        >>> print(f'Adjusted threshold: {thresh:.2f}')
        
        >>> # With temporal masking for filter padding
        >>> mask = np.ones(1000, dtype=bool)
        >>> mask[:50] = False  # Exclude first 50 samples
        >>> mask[-50:] = False  # Exclude last 50 samples
        >>> Z, elecs_z, thresh = ar.z_score_data(X, z_thresh=4.0,
        ...                                      mask=mask)
        
        >>> # With low-pass filtering
        >>> Z, elecs_z, thresh = ar.z_score_data(
        ...     X, z_thresh=4.0, filter_z=(True, 500.0)
        ... )
        """

        # set params
        nr_epoch, nr_elec, nr_time = X.shape

        # get the data and z_score over electrodes
        X = X.swapaxes(0, 1).reshape(nr_elec, -1)
        X_z = zscore(X, axis=1)
        if mask is not None:
            mask = np.tile(mask, nr_epoch)
            X_z[:, mask] = zscore(X[:, mask], axis=1)

        # reshape to get get epoched data in terms of z scores
        elecs_z = X_z.reshape(nr_elec, nr_epoch, -1).swapaxes(0, 1)

        # normalize z_score
        Z_n = X_z.sum(axis=0) / sqrt(nr_elec)
        if mask is not None:
            Z_n[mask] = X_z[:, mask].sum(axis=0) / sqrt(nr_elec)
        if filter_z[0]:
            print(Z_n.shape)
            Z_n = filter_data(
                Z_n, filter_z[1], None, 4, pad='reflect_limited'
            )

        # adjust threshold (data driven)
        if mask is None:
            mask = np.ones(Z_n.size, dtype=bool)
        z_thresh += (
            np.median(Z_n[mask]) + 
            abs(Z_n[mask].min() - np.median(Z_n[mask]))
        )

        # transform back into epochs
        Z_n = Z_n.reshape(nr_epoch, -1)

        return Z_n, elecs_z, z_thresh
    
    @blockPrinting
    def iterative_interpolation(
        self, 
        epochs: mne.Epochs, 
        elecs_z: np.ndarray, 
        noise_inf: List[list], 
        z_thresh: float, 
        band_pass: List[int]
    ) -> Tuple[mne.Epochs, List[int], List[int]]:
        """
        Iteratively interpolate bad channels to salvage 
        artifact-contaminated epochs.

        For each epoch marked as containing artifacts, attempts to clean 
        it by iteratively interpolating the noisiest channels 
        (up to self.max_bad). After each interpolation, re-checks if the 
        epoch still exceeds the z-score threshold. Epochs that cannot be 
        cleaned are marked as bad.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data to clean. Modified in-place.
        elecs_z : np.ndarray
            3D array of z-scores per electrode, 
            shape (n_epochs, n_channels, n_times). Output from 
            preprocess_epochs method.
        noise_inf : List[list]
            List of noise events from mark_bads. Each element is a list 
            with:
            [epoch_idx, (slice_obj, start_time, end_time, duration),...]
        z_thresh : float
            Z-score threshold for artifact detection. Used to determine 
            if interpolation successfully cleaned the epoch.
        band_pass : List[int]
            [lower, upper] frequency bounds in Hz for bandpass filter. 
            Passed to preprocess_epochs for re-checking after 
            interpolation.

        Returns
        -------
        epochs : mne.Epochs
            Cleaned epochs with bad channels interpolated 
            where possible.
        bad_epochs : List[int]
            Indices of epochs that could not be cleaned even after 
            interpolating self.max_bad channels.
        cleaned_epochs : List[int]
            Indices of epochs successfully cleaned via channel 
            interpolation.

        Notes
        -----
        - Cleaning strategy:
          1. Extract artifact segments for the epoch from noise_inf
          2. Compute mean z-score for each channel during artifact 
            periods
          3. Sort channels by z-score, interpolate noisiest first
          4. After each interpolation, re-run preprocess_epochs
          5. Stop if epoch is clean or self.max_bad channels 
             interpolated
        - Updates internal tracking for visualization:
          - self.heat_map: 2D array tracking interpolated channels per 
            epoch
          - self.cleaned_info: dict counting interpolations per channel 
            (cleaned)
          - self.not_cleaned_info: dict counting interpolations per 
            channel (bad)
        - Decorated with @blockPrinting to suppress MNE interpolation
          messages
        - Modifies epochs._data in-place

        Examples
        --------
        >>> ar = ArtefactReject(max_bad=5)
        >>> Z, elecs_z, thresh, times = ar.preprocess_epochs(epochs)
        >>> noise_inf = ar.mark_bads(Z, thresh, times)
        >>> epochs, bad, cleaned = ar.iterative_interpolation(
        ...     epochs, elecs_z, noise_inf, thresh, [110, 140]
        ... )
        >>> print(f'Cleaned: {len(cleaned)}, Still bad: {len(bad)}')
        """

        # keep track of channel info for plotting purposes
        channels = np.array(epochs.copy().pick('eeg', exclude='bads').ch_names)
        self.heat_map = np.zeros((len(noise_inf), channels.size))
        self.cleaned_info = dict.fromkeys(channels, 0)
        self.not_cleaned_info = dict.fromkeys(channels, 0)

        # track bad and cleaned epochs
        bad_epochs, cleaned_epochs = [], []
        for i, event in enumerate(noise_inf):
            bad_epoch = epochs[event[0]]
            # search for bad channels in detected artefact periods
            z = np.concatenate(
                [elecs_z[event[0]][:,slice_[0]] for slice_ in event[1:]], 
                axis=1
            )
            # limit interpolation to 'max_bad' noisiest channels
            ch_idx = np.argsort(z.mean(axis=1))[-self.max_bad:][::-1]
            interp_chs = channels[ch_idx]

            for c, ch in enumerate(interp_chs):
                # update heat map
                bad_epoch.info['bads'] += [ch]
                bad_epoch.interpolate_bads(exclude=epochs.info['bads'])
                epochs._data[event[0]] = bad_epoch._data
                # repeat preprocesing after interpolation to check whether 
                # epoch is now 'clean'
                Z_, _, _, _ = self.preprocess_epochs(
                    epochs[event[0]], 
                    band_pass=band_pass
                )

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
    
    def plot_auto_repair(
        self, 
        channels: np.ndarray, 
        z_score: np.ndarray, 
        z_thresh: float
    ) -> List[plt.Figure]:
        """
        Generate all visualization figures for auto_repair_noise 
        results.

        Creates comprehensive set of plots showing artifact detection 
        and repair results: z-score time series, channel interpolation 
        heatmap, and histograms of interpolation counts.

        Parameters
        ----------
        channels : np.ndarray
            1D array of channel names (strings).
        z_score : np.ndarray
            2D array of z-scores, shape (n_epochs, n_times).
        z_thresh : float
            Z-score threshold used for artifact detection.

        Returns
        -------
        figs : List[matplotlib.figure.Figure]
            List of figures: [z-score plot, heatmap, bad histogram 
            (if any), cleaned histogram (if any)].
        """

        figs = []

        # plot z-score time series
        figs.append(self.plot_z_score_epochs(z_score, z_thresh))

        # plot heat map
        figs.append(self.plot_heat_map(channels))

        # plot histograms
        for plot in ['bad', 'cleaned']:
            fig = self.plot_hist_auto_repair(plot)
            if fig:
                figs.append(fig)

        return figs
    
    def plot_heat_map(self, channels: np.ndarray) -> plt.Figure:
        """
        Create heatmap showing which channels were interpolated per 
        epoch.

        Visualizes the pattern of channel interpolations across flagged 
        epochs. Blue indicates epochs that remained bad after 
        interpolation, red indicates successfully cleaned epochs.

        Parameters
        ----------
        channels : np.ndarray
            1D array of channel names (strings) for x-axis labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing heatmap with channels on x-axis, epochs on 
            y-axis, and interpolation status color-coded (blue=-1: bad, 
            white=0: not interpolated, red=1: cleaned).
        """

        # plot heat_map
        bad = sum(np.any(self.heat_map == -1, axis=1))
        cleaned = sum(np.any(self.heat_map == 1, axis=1))
        fig, ax = plt.subplots(1)
        sns.despine(offset=10)
        plt.title(
            f'Interpolated electrodes per marked epoch \n'
            f'(blue is bad epochs (N={bad}), red is cleaned epoch '
            f'(N={cleaned}))'
        )
        ax.imshow(
            self.heat_map, 
            aspect='auto', 
            cmap='bwr',
            interpolation='nearest', 
            vmin=-1, 
            vmax=1
        )
        ax.set(xlabel='channel', ylabel='bad epochs')
        ax.set_xticks(np.arange(channels.size))
        ax.set_xticklabels(channels, fontsize=6, rotation=90)

        return fig

    def update_heat_map(
        self, 
        channels: np.ndarray, 
        ch_idx: np.ndarray, 
        tr_idx: int, 
        upd_value: int
    ) -> None:
        """
        Update heatmap and tracking dictionaries for interpolated 
        channels.

        Records which channels were interpolated for a specific epoch 
        and updates counters tracking successful vs unsuccessful 
        interpolations per channel.

        Parameters
        ----------
        channels : np.ndarray
            1D array of all channel names (strings).
        ch_idx : np.ndarray
            1D array of indices indicating which channels were 
            interpolated for this epoch.
        tr_idx : int
            Index of the epoch in the heatmap (row index).
        upd_value : int
            Status value: -1 for bad epoch (could not be cleaned), 
            1 for cleaned epoch (successfully repaired).

        Returns
        -------
        None
            Modifies self.heat_map, self.not_cleaned_info, and 
            self.cleaned_info in-place.
        """

        self.heat_map[tr_idx, ch_idx] = upd_value
        if upd_value == -1:  # bad epoch
            for ch in channels[ch_idx]:
                self.not_cleaned_info[ch] += 1

        else:  # cleaned_epoch
            for ch in channels[ch_idx]:
                self.cleaned_info[ch] += 1

    def plot_hist_auto_repair(
        self, 
        plot_type: str
    ) -> Union[plt.Figure, bool]:
        """
        Create histogram of channel interpolation counts.

        Shows how many times each channel was interpolated across either 
        bad or cleaned epochs.

        Parameters
        ----------
        plot_type : str
            Type of epochs to plot: 'cleaned' (successfully repaired) or 
            'bad' (could not be cleaned).

        Returns
        -------
        fig : matplotlib.figure.Figure or bool
            Horizontal bar plot showing interpolation counts per 
            channel, or False if no epochs of the specified type exist.
        """

        if plot_type == 'cleaned':
            info = self.cleaned_info
        else:
            info = self.not_cleaned_info

        df = pd.DataFrame.from_dict(info, orient='index', columns=['count'])
        df = df.loc[(df != 0).any(axis=1)]

        if df.size > 0:  # marked at least one bad/cleaned epoch
            fig, ax = plt.subplots(1)
            sns.despine(offset=10)
            plt.title(f'Electrode count per {plot_type} epoch')
            ax.barh(np.arange(df.values.size), np.hstack(df.values))
            ax.set_yticks(np.arange(df.values.size))
            ax.set_yticklabels(df.index, fontsize=5)
        else:
            fig = False

        return fig

    def plot_z_score_epochs(
        self, 
        z_score: np.ndarray, 
        z_thresh: float
    ) -> plt.Figure:
        """
        Plot z-score time series with threshold overlay.

        Visualizes accumulated z-scores across all epochs, highlighting 
        samples exceeding the threshold in red.

        Parameters
        ----------
        z_score : np.ndarray
            2D array of z-scores, shape (n_epochs, n_times).
        z_thresh : float
            Z-score threshold for artifact detection.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Line plot with z-scores in blue, threshold-exceeding samples 
            in red, and threshold line as red dashed.
        """

        fig, ax = plt.subplots(1)
        plt.ylabel('accumulated Z')
        plt.xlabel('sample')
        plt.plot(np.arange(0, z_score.size), z_score.flatten(), color='b')
        plt.plot(
            np.arange(0, z_score.size),
            np.ma.masked_less(z_score.flatten(), z_thresh), 
            color='r'
        )
        plt.axhline(z_thresh, color='r', ls='--')

        return fig














