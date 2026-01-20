"""
Event-Related Potential (ERP) Analysis for EEG Data.

This module provides comprehensive functionality for analyzing 
event-related potentials (ERPs) in EEG data, leveraging the MNE-Python 
ecosystem for professional neuroscience analysis. The module supports 
standard ERP analysis workflows including trial averaging, baseline 
correction, condition-based comparisons, and lateralized component 
analysis.

The primary ERP class inherits from FolderStructure to provide organized 
file management and output saving capabilities. The module implements 
both basic ERP analysis (mean amplitude, peak latency) and advanced 
statistical methods (jackknife latency comparisons, AUC measurements).

Key functionality includes:
    - Condition-based ERP averaging with flexible trial selection
    - Lateralized ERP analysis with topographic flipping
    - Peak detection and time window optimization
    - Statistical comparison methods for ERP components
    - Export capabilities for further statistical analysis
    - Integration with MNE-Python for professional visualization

The module is designed for cognitive neuroscience research and supports 
common ERP components analysis including N2pc, P3, lateralized readiness 
potentials, and other event-related brain responses.


Examples
--------
Basic ERP analysis workflow:
    >>> import mne
    >>> from ERP import ERP
    >>> # Load preprocessed epochs and behavioral data
    >>> erp_analyzer = ERP(sj=1, epochs=epochs, df=behavior_df, 
    ...                   baseline=(-0.2, 0))
    >>> # Create condition-based ERPs
    >>> erp_analyzer.condition_erps(pos_labels={'target_loc': [2, 6]},
    ...                            cnds={'target_type': ['standard', 
    ...'deviant']})

See Also
--------
mne.Evoked : MNE-Python evoked data structure
mne.Epochs : MNE-Python epochs data structure

Created by Dirk van Moorselaar on 01-07-2015.
Copyright (c) 2015 DvM. All rights reserved.
"""
import os
import mne
import pickle
import math
import warnings
import copy

import numpy as np
import pandas as pd

from itertools import combinations
from typing import Optional, Generic, Union, Tuple, Any


from IPython import embed
from scipy.fftpack import fft, ifft
from scipy.signal import butter, lfilter, freqz
from sklearn.metrics import auc
from support.FolderStructure import *
from support.preprocessing_utils import (select_electrodes,trial_exclusion,
                                        get_time_slice, format_subject_id)

class ERP(FolderStructure):
    """
    Event-Related Potential (ERP) analysis for EEG data using
    MNE-Python.

    The ERP class provides comprehensive functionality for analyzing 
    event-related potentials in EEG data. It supports standard ERP 
    analysis workflows including trial averaging, condition-based 
    comparisons, baseline correction, and lateralized component 
    analysis.

    The class inherits from FolderStructure to provide organized file 
    management and automatic saving of analysis outputs. It integrates 
    seamlessly with MNE-Python for professional neuroscience analysis 
    and visualization.

    Parameters
    ----------
    sj : int
        Subject identifier number for file naming and organization.
    epochs : mne.Epochs
        Preprocessed EEG data segmented into epochs. Should contain 
        all trials for analysis with proper event markers.
    df : pd.DataFrame
        Behavioral data corresponding to the EEG epochs. Must have 
        the same number of trials as epochs and optionally contain 
        condition labels for analysis. Can be used for trial selection, 
        exclusion criteria, or collapsed across all trials.
    baseline : tuple of float
        Time range (start, end) in seconds for baseline correction 
        (e.g., (-0.2, 0) for 200ms pre-stimulus baseline).
    l_filter : float, optional
        Low-pass filter cutoff frequency in Hz. Applied to epochs 
        during data selection if specified. 
        Default is None (no filtering).
    h_filter : float, optional
        High-pass filter cutoff frequency in Hz. Applied to epochs 
        during data selection if specified. 
        Default is None (no filtering).
    downsample : int, optional
        Target sampling frequency in Hz for downsampling. Only 
        applied if lower than current sampling rate. Default is None.
    laplacian : bool, optional
        Whether to apply Laplacian (current source density) filtering 
        to enhance spatial resolution. Default is False.
    report : bool, optional
        Whether to generate HTML reports with ERP visualizations 
        using MNE-Python's reporting functionality. Default is False.

    Methods
    -------
    select_erp_data(excl_factor=None, topo_flip=None)
        Select and preprocess ERP data with optional trial exclusion 
        and topographic flipping.
    create_erps(epochs, df, idx=None, time_oi=None, erp_name='all', 
                RT_split=False, save=True)
        Create averaged ERP objects from epochs with optional 
        RT splitting.
    condition_erps(pos_labels=None, cnds=None, midline=None, 
                   topo_flip=None, time_oi=None, excl_factor=None, 
                   RT_split=False, name='main')
        Generate condition-specific ERPs with flexible trial selection.
    generate_erp_report(evokeds, report_name)
        Generate HTML reports for ERP visualization.

    Notes
    -----
    This class is specifically designed for cognitive neuroscience 
    research and supports analysis of common ERP components such as 
    N2pc, P300, lateralized readiness potentials, and other 
    event-related brain responses.

    The class handles lateralized designs by supporting topographic 
    flipping, allowing all trials to be analyzed as if stimuli were 
    presented in a consistent hemifield (typically right/contralateral).

    Baseline correction follows standard ERP analysis practices and 
    is applied to all evoked objects using the MNE-Python baseline 
    correction implementation.

    Examples
    --------
    Basic ERP analysis:
        >>> erp = ERP(sj=1, epochs=epochs_data, df=behavioral_data,
        ...          baseline=(-0.2, 0))
        >>> erp.condition_erps(cnds={'condition': ['standard', 
        ...'deviant']})

    Lateralized ERP analysis:
        >>> erp = ERP(sj=1, epochs=epochs_data, df=behavioral_data,
        ...          baseline=(-0.2, 0))
        >>> erp.condition_erps(pos_labels={'target_loc': [2, 4, 6, 8]},
        ...                    topo_flip={'target_loc': [2, 8]},
        ...                    cnds={'condition': ['target', 
        ... 'distractor']})

    See Also
    --------
    mne.Evoked : MNE-Python evoked data structure
    mne.Epochs : MNE-Python epochs data structure
    mne.baseline.apply_baseline : Baseline correction implementation
    """

    def __init__(
        self, 
        sj: int, 
        epochs: mne.Epochs, 
        df: pd.DataFrame, 
        baseline: Tuple[float, float], 
        l_filter: Optional[float] = None, 
        h_filter: Optional[float] = None, 
        downsample: Optional[int] = None, 
        laplacian: bool = False,
        report: bool = False
    ):
        """
        Initialize ERP analysis object.
        
        Sets up event-related potential analysis parameters and 
        validates input data. Configures baseline correction, filtering 
        options, and report generation settings for comprehensive ERP 
        analysis.
        
        Parameters are documented in the class docstring above.
        """


        self.sj = format_subject_id(sj)
        self.l_filter = l_filter
        self.h_filter = h_filter
        self.epochs = epochs
        self.df = df
        self.baseline = baseline
        self.downsample = downsample
        self.laplacian = laplacian
        self.report = report

    def select_erp_data(self, excl_factor: dict = None,
                        topo_flip: dict = None) -> Tuple[pd.DataFrame, 
                                                        mne.Epochs]:
        """
        Select and preprocess ERP data with optional trial exclusion and 
        topographic flipping.

        This function filters behavioral and EEG data based on specified 
        exclusion criteria and applies topographic adjustments for 
        lateralized designs. It also applies any specified filtering, 
        downsampling, and Laplacian filtering before trial averaging.

        Parameters
        ----------
        excl_factor : dict, optional
            Dictionary specifying criteria for excluding trials from 
            analysis. Keys should be column names in the behavioral 
            DataFrame, values should be lists of labels to exclude. 
            For example, `{'target_color': ['red']}` excludes trials 
            where target color is red. Default is None.
        topo_flip : dict, optional
            Dictionary specifying criteria for flipping topography of 
            certain trials. Key should be column name in behavioral 
            data, value should be list of labels indicating trials to 
            flip. For example, `{'cue_loc': [2, 3]}` flips trials where
            cue location is 2 or 3. Default is None.

        Returns
        -------
        df : pd.DataFrame
            Filtered behavioral data with reset index for proper 
            alignment with epochs.
        epochs : mne.Epochs
            Processed EEG epochs with applied filtering, downsampling, 
            Laplacian filtering, and topographic flipping as specified.

        Notes
        -----
        For lateralized designs, topographic flipping ensures all
        stimuli of interest are treated as if presented in the right 
        hemifield (contralateral relative to the stimulus of interest).

        The function applies preprocessing steps in the following order:
        1. Trial exclusion based on behavioral criteria
        2. Frequency filtering (if l_filter or h_filter specified)
        3. Laplacian filtering (if laplacian=True)
        4. Downsampling (if downsample specified and lower than current 
           rate)
        5. Topographic flipping (if topo_flip specified)

        Examples
        --------
        Exclude trials and flip topography:
        >>> df, epochs = erp.select_erp_data(
        ...     excl_factor={'correct': [0]}, # exclude incorrect trials
        ...     topo_flip={'target_loc': [2, 8]}  # flip left positions
        ... )

        Apply only trial exclusion:
        >>> df, epochs = erp.select_erp_data(
        ...     excl_factor={'reaction_time': [999]}  # exclude timeouts
        ... )
        """

        df = self.df.copy()
        epochs = self.epochs.copy()

        # if not already done reset index (to properly align beh and epochs)
        df.reset_index(inplace = True, drop = True)

        # if specified remove trials matching specified criteria
        if excl_factor is not None:
            df, epochs, _ = trial_exclusion(df, epochs, excl_factor)

        # if filters are specified, filter data before trial averaging  
        if self.l_filter is not None or self.h_filter is not None:
            epochs.filter(l_freq=self.l_filter, h_freq=self.h_filter)

        # apply laplacian filter using mne defaults
        if self.laplacian:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        if self.downsample is not None:
            if self.downsample < int(epochs.info['sfreq']):
                print('downsampling data')
                epochs.resample(self.downsample)

        # check whether left stimuli should be 
        # artificially transferred to left hemifield
        if topo_flip is not None:
            epochs = self.flip_topography(epochs, df, topo_flip)
        else:
            print('No topography info specified. In case of a lateralized '
                'design. It is assumed as if all stimuli of interest are '
                'presented right (i.e., left  hemifield')

        return df, epochs
    
    def create_erps(
        self, 
        epochs: mne.Epochs, 
        df: pd.DataFrame, 
        idx: Optional[np.array] = None, 
        time_oi: Optional[tuple] = None, 
        erp_name: str = 'all', 
        RT_split: bool = False, 
        save: bool = True
    ) -> mne.Evoked:
        """
        Average epochs to create evoked responses with baseline 
        correction.

        Computes trial-averaged evoked responses (ERPs) from EEG epochs 
        using MNE-Python's averaging functionality. The method applies 
        baseline correction based on the instance's baseline parameter
        and can optionally crop data to specific time windows of 
        interest. When RT_split is enabled, trials are divided at the 
        median reaction time to create separate fast and slow evoked 
        responses for RT analyses.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed EEG epochs for trial averaging. Should contain 
            the trials specified by idx parameter or all available 
            trials if idx is None.
        df : pd.DataFrame
            Behavioral dataframe with trial-level information. Used for 
            RT splitting when RT_split=True. Must contain 'RT' column if 
            RT splitting is requested.
        idx : array-like or None, default=None
            Boolean or integer array specifying which trials to include 
            in averaging. If None, all epochs are averaged. The indices 
            apply to both epochs and df to maintain trial 
            correspondence.
        time_oi : tuple or None, default=None
            Time window of interest as (tmin, tmax) in seconds. If 
            provided, evoked responses are cropped to this window after 
            averaging. Useful for focusing on specific ERP components or 
            analysis periods.
        erp_name : str, default='all'
            Base filename for saving evoked objects. Files are saved 
            with '-ave.fif' suffix. When RT_split=True, '_fast' and 
            '_slow' are appended to create separate files.
        RT_split : bool, default=False
            Whether to create separate evoked responses for fast and 
            slow trials based on median reaction time. When True, 
            requires 'RT' column in df and creates additional evoked 
            objects.
        save : bool, default=True
            Whether to save evoked objects to disk. Files are saved in 
            the ERP evoked subfolder with .fif format. When False, only 
            returns the main evoked object without saving.

        Returns
        -------
        mne.Evoked
            Trial-averaged evoked response with baseline correction 
            applied. If time_oi is specified, the evoked object is 
            cropped to that time window. When RT_split=True, this 
            returns the evoked response from all trials, while fast/slow 
            splits are saved separately.

        Notes
        -----
        The averaging process follows standard ERP computation:
        1. Select trials using idx parameter
        2. Average epochs across trials using MNE's average() method
        3. Apply baseline correction using instance's baseline parameter
        4. Optionally crop to time window of interest
        5. Save to disk if requested

        When RT_split=True, the method performs median split analysis:
        - Computes median RT across selected trials
        - Creates 'fast' group for trials below median
        - Creates 'slow' group for trials above median
        - Generates separate evoked objects for each group
        - Both split evoked objects are automatically saved

        File naming convention:
        - Main evoked: '{erp_name}-ave.fif'
        - Fast trials: '{erp_name}_fast-ave.fif'
        - Slow trials: '{erp_name}_slow-ave.fif'

        Examples
        --------
        Create evoked response from all trials:

        >>> erp_analyzer = ERP(...)
        >>> evoked = erp_analyzer.create_erps(epochs, df, 
        ...   erp_name='target')

        Average specific trial subset with time cropping:

        >>> target_idx = df['condition'] == 'target'
        >>> evoked = erp_analyzer.create_erps(
        ...     epochs, df, idx=target_idx,
        ...     time_oi=(-0.1, 0.6), erp_name='target_crop'
        ... )

        Generate RT split analysis:

        >>> evoked = erp_analyzer.create_erps(
        ...     epochs, df, idx=target_idx,
        ...     RT_split=True, erp_name='target_rt'
        ... )
        """

        df = df.iloc[idx].copy()
        epochs = epochs[idx]

        # create evoked objects using mne functionality and save file
        evoked = epochs.average().apply_baseline(baseline = self.baseline)

        # if specified select time window of interest
        if time_oi is not None:
            evoked = evoked.crop(tmin = time_oi[0],tmax = time_oi[1])
        if save: 
            evoked.save(self.folder_tracker(['erp','evoked'],
                                        f'{erp_name}-ave.fif'),
                        overwrite=True)
            
        # split trials in fast and slow trials based on median RT
        if RT_split:
            median_rt = np.median(df.RT)
            df.loc[df.RT < median_rt, 'RT_split'] = 'fast'
            df.loc[df.RT > median_rt, 'RT_split'] = 'slow'
            for rt in ['fast', 'slow']:
                mask = df['RT_split'] == rt
                # create evoked objects using mne functionality and save file
                evoked_split = epochs[mask].average().apply_baseline(baseline = 
                                                            self.baseline)
                # if specified select time window of interest
                if time_oi is not None:
                    evoked_split = evoked_split.crop(tmin = time_oi[0],
                                                     tmax = time_oi[1]) 															
                evoked_split.save(self.folder_tracker(['erp', 'evoked'],
                                                f'{erp_name}_{rt}-ave.fif'))
        return evoked
                
    def generate_erp_report(self, evokeds: dict, report_name: str):
        """
        Generate HTML report for ERP visualization using MNE-Python.

        Creates an HTML report containing ERP visualizations for all 
        provided conditions. The report includes topographic plots, 
        butterfly plots, and other standard ERP visualizations 
        generated by MNE-Python's reporting functionality.

        Parameters
        ----------
        evokeds : dict
            Dictionary where keys are condition names and values are 
            mne.Evoked objects containing the averaged ERP data.
        report_name : str
            Base name for the output report file (without extension).

        Notes
        -----
        The report is saved as an HTML file in the 'erp/report' 
        subfolder of the project directory. Laplacian-filtered data 
        may have visualization limitations in current MNE versions.

        The generated report provides interactive visualizations that 
        are useful for quality control and preliminary analysis of 
        ERP data across conditions.

        Examples
        --------
        >>> erp_analyzer.generate_erp_report(
        ...     evokeds={'standard': evoked_std, 'deviant': evoked_dev},
        ...     report_name='P300_analysis'
        ... )
        """


        report_name = self.folder_tracker(['erp', 'report'],
                                        f'{report_name}.h5')
        
        report = mne.Report(title='Single subject evoked overview')
        for cnd in evokeds.keys():
            if self.laplacian:
                #TODO: remove after updating mne
                pass
            else:
                report.add_evokeds(evokeds=evokeds[cnd],titles=cnd)

        report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html', 
                    overwrite=True, open_browser=False)
                        
    def condition_erps(
        self, 
        pos_labels: Optional[dict] = None, 
        cnds: Optional[dict] = None, 
        midline: Optional[dict] = None, 
        topo_flip: Optional[dict] = None, 
        time_oi: Optional[tuple] = None, 
        excl_factor: Optional[dict] = None, 
        RT_split: bool = False, 
        name: str = 'main'
    ):
        """
        Generate condition-specific ERPs with flexible trial selection 
        and lateralization support.

        This is the primary method for ERP analysis in the class, 
        providing comprehensive functionality for creating 
        condition-specific event-related potentials from EEG epochs. 
        The method supports lateralized experimental designs through 
        position-based trial selection, topographic flipping for 
        contralateral analysis, and flexible condition comparisons. 
        It handles the complete ERP analysis pipeline from trial 
        selection to evoked object creation and optional reporting.

        Parameters
        ----------
        pos_labels : dict or None, default=None
            Position-based trial selection criteria for lateralized
            designs. Key specifies the behavioral dataframe column 
            containing position information, value provides list of 
            position labels to include. Essential for lateralized ERP 
            components like N2pc, SPCN, or CDA analysis.
            Example: {'target_loc': [2, 4, 6, 8]} selects trials where 
            targets appeared at positions 2, 4, 6, or 8.
        cnds : dict or None, default=None
            Condition-specific ERP generation criteria. Key specifies 
            the behavioral dataframe column for condition labels, value 
            provides list of conditions to analyze separately. 
            When None, creates single ERP collapsed across all selected 
            trials.Example: {'stimulus_type': ['standard', 'deviant']} 
            creates separate ERPs for standard and deviant stimuli.
        midline : dict or None, default=None
            Additional selection criteria for trials where other stimuli 
            appear on the vertical midline. Useful for controlling 
            distractor or non-target stimulus positions in lateralized 
            designs. Format matches pos_labels and cnds.
            Example: {'distractor_loc': [0]} limits analysis to trials 
            where distractors appeared at central position.
        topo_flip : dict or None, default=None
            Topographic flipping criteria for lateralized analysis. 
            Key specifies behavioral column, value lists labels for 
            trials requiring hemisphere flipping. Enables analysis as if 
            all stimuli were contralateral by flipping electrode data 
            for ipsilateral presentations.
            Example: {'target_loc': [2, 8]} flips topography for left 
            visual field trials.
        time_oi : tuple or None, default=None
            Time window of interest as (tmin, tmax) in seconds for 
            temporal cropping. Applied after trial averaging to focus 
            analysis on specific ERP components or time periods. 
            When None, uses full epoch duration.
            Example: (0.1, 0.4) crops to 100-400ms post-stimulus for 
            P300 analysis.
        excl_factor : dict or None, default=None
            Trial exclusion criteria based on behavioral or data quality 
            measures. Key specifies column for exclusion logic, value 
            lists labels to exclude. Applied before any other trial 
            selection to remove unwanted trials.
            Example: {'response_accuracy': ['incorrect']} excludes error 
            trials.
        RT_split : bool, default=False
            Whether to perform median reaction time split analysis. When 
            True, creates additional fast/slow ERPs based on median RT 
            division. Requires 'RT' column in behavioral dataframe. 
            Generates separate evoked objects for speed-based 
            comparisons.
        name : str, default='main'
            Base identifier for output files and analysis labeling. 
            Used in evoked object filenames as 
            'sub_{subject}_{condition}_{name}-ave.fif'. Helps 
            organize multiple analyses of the same dataset.

        Returns
        -------
        None
            Method saves evoked objects to disk and optionally generates 
            HTML reports. Individual evoked objects are stored in ERP 
            evoked subfolder with systematic naming convention. 
            When self.report=True, creates comprehensive visualization 
            report.

        Notes
        -----
        Analysis Pipeline:
        1. Select and preprocess ERP data using select_erp_data()
        2. Apply position-based trial selection if pos_labels specified
        3. Loop through conditions creating separate ERPs
        4. Apply time window cropping if time_oi provided
        5. Save evoked objects with systematic naming
        6. Generate optional visualization reports

        Lateralized Design Support:
        The method is specifically designed for lateralized ERP 
        components where stimuli can appear in left or right visual 
        fields. Position labels typically represent spatial locations 
        (e.g., 2,4,6,8 for right field; 1,3,5,7 for left field) and 
        topographic flipping ensures all analyses are performed as if 
        stimuli were contralateral.

        File Organization:
        - ERP: 'erp/evoked/sub_{sj}_{condition}_{name}-ave.fif'
        - RT split: 'erp/evoked/sub_{sj}_{condition}_{name}_fast-ave.fif'
        - HTML reports: 'erp/report/sub_{sj}_{name}.html'

        The method handles missing conditions by printing warnings and
        continuing analysis for available conditions.

        Examples
        --------
        Basic condition comparison without lateralization:

        >>> erp_analyzer = ERP(sj=1, epochs=epochs, df=behavioral_data, 
        ...                   baseline=(-0.2, 0))
        >>> erp_analyzer.condition_erps(
        ...     cnds={'stimulus_type': ['standard', 'deviant']},
        ...     time_oi=(0.25, 0.45), name='P300_analysis'
        ... )

        Lateralized N2pc analysis with topographic flipping:

        >>> erp_analyzer.condition_erps(
        ...     pos_labels={'target_loc': [2, 4, 6, 8]},
        ...     cnds={'set_size': ['2', '4', '6']},
        ...     topo_flip={'target_loc': [2, 8]},
        ...     time_oi=(0.18, 0.28), name='N2pc_setsize'
        ... )

        Complex lateralized design with distractor control:

        >>> erp_analyzer.condition_erps(
        ...     pos_labels={'target_loc': [1, 3, 5, 7, 2, 4, 6, 8]},
        ...     cnds={'target_color': ['red', 'green']},
        ...     midline={'distractor_loc': [0]},
        ...     topo_flip={'target_loc': [1, 3, 5, 7]},
        ...     excl_factor={'accuracy': ['error']},
        ...     RT_split=True, name='lateralized_attention'
        ... )

        RT split analysis for speed-accuracy effects:

        >>> erp_analyzer.condition_erps(
        ...     cnds={'difficulty': ['easy', 'hard']},
        ...     RT_split=True, time_oi=(0.3, 0.6), 
        ...     name='difficulty_RT_effects'
        ... )
        """

        # get data
        df, epochs = self.select_erp_data(excl_factor,topo_flip)
        
        # select trials of interest (e.g., lateralized stimuli)
        if isinstance(pos_labels, dict):
            idx = ERP.select_lateralization_idx(df, pos_labels, midline)
        elif pos_labels is None:
            idx = np.arange(len(df))
        else:
            raise TypeError(f"pos_labels must be a dict or None, \
                            got {type(pos_labels).__name__}")

        # loop over all conditions
        if cnds is None:
            cnds = ['all_data']
        else:
            (cnd_header, cnds), = cnds.items()

        # create evoked dictionary based on conditions
        evokeds = {key: [] for key in cnds} 

        for cnd in cnds:
            # set erp name
            erp_name = f'sub_{self.sj}_{cnd}_{name}'	

            # slice condition trials
            if cnd == 'all_data':
                idx_c = idx
            else:
                idx_c = np.where(df[cnd_header] == cnd)[0]
                idx_c = np.intersect1d(idx, idx_c)

            if idx_c.size == 0:
                print('no data found for {}'.format(cnd))
                continue

            evokeds[cnd] = self.create_erps(epochs, df, idx_c, time_oi, 
                                            erp_name, RT_split)

        if self.report:
            self.generate_erp_report(evokeds,f'sub_{self.sj}_{name}'	)

    @staticmethod
    def flip_topography(
        epochs: mne.Epochs, 
        df: pd.DataFrame, 
        topo_flip: dict, 
        flip_dict: Optional[dict] = None, 
        heog: str = 'HEOG'
    ) -> mne.Epochs:
        """
        Transform lateralized EEG data to unified hemispheric reference 
        frame.

        This method performs topographic flipping to enable lateralized 
        ERP analysis by transforming trials with ipsilateral stimulus 
        presentations to appear as if they were contralateral. The 
        transformation swaps left-right electrode pairs for specified 
        trials, creating a consistent hemispheric reference frame where
        all stimuli can be analyzed as contralateral presentations.

        This is essential for lateralized ERP components (N2pc, SPCN, 
        CDA, LRP) where the neural response depends on the spatial 
        relationship between stimulus location and recording hemisphere. 
        After flipping, difference waves (contralateral - ipsilateral) 
        can be computed across all trials regardless of original 
        stimulus location.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed EEG epochs containing trial data for 
            topographic transformation. The epochs object is modified 
            in-place with flipped electrode data for specified trials.
        df : pd.DataFrame
            Behavioral dataframe with trial-level stimulus information. 
            Must contain position or location columns that correspond to 
            the epochs trial order for identifying flipping targets.
        topo_flip : dict
            Dictionary specifying topographic flipping criteria. Key 
            indicates the behavioral column name containing position 
            labels, value provides list of position labels identifying 
            trials requiring hemisphere transformation. 
            Example: {'target_loc': [1, 3, 5, 7]} flips trials where 
            targets appeared at left visual field positions.
        flip_dict : dict or None, default=None
            Custom electrode pairing dictionary for topographic 
            transformation. Keys and values specify electrode pairs to 
            swap (e.g., {'O1': 'O2', 'P7': 'P8'}). If None, 
            automatically generates pairs based on standard naming 
            convention (odd=left, even=right).
        heog : str, default='HEOG'
            Name of horizontal electrooculogram channel for polarity 
            reversal. When present, the HEOG signal is inverted for 
            flipped trials to maintain consistent eye movement polarity 
            across the transformed reference frame.

        Returns
        -------
        mne.Epochs
            The input epochs object with topographically flipped 
            electrode data for specified trials. The transformation is 
            applied in-place, modifying the original epochs._data array 
            for identified trials.

        Notes
        -----
        Topographic Flipping Logic:
        1. Extract position criteria from topo_flip dictionary
        2. Identify trials matching flipping criteria using behavioral 
           data
        3. Create or use provided flip_dict for electrode pair mapping
        4. Swap data between paired electrodes for identified trials
        5. Invert HEOG polarity to maintain eye movement consistency
        6. Return modified epochs with unified hemispheric reference

        Automatic Flip Dictionary Generation:
        When flip_dict=None, the method automatically creates electrode 
        pairs based on standard EEG naming conventions where 
        odd-numbered electrodes represent left hemisphere positions and 
        even-numbered electrodes represent right hemisphere positions 
        (e.g., O1↔O2, P7↔P8).

        Electrode Selection:
        The method operates on EEG and CSD (current source density) 
        channels only, preserving other channel types unchanged. This 
        ensures that only spatially-relevant neural signals undergo 
        transformation.

        Common Applications:
        - N2pc analysis: Flip ipsilateral target trials to create 
          contralateral reference frame
        - SPCN studies: Transform memory array presentations for 
          consistent lateralization
        - CDA experiments: Align memory load conditions regardless of 
          original stimulus side
        - LRP analysis: Create consistent motor preparation reference 
          frame

        Examples
        --------
        Basic N2pc topographic flipping for left visual field trials:

        >>> flipped_epochs = ERP.flip_topography(
        ...     epochs, behavioral_df, 
        ...     topo_flip={'target_position': [1, 3, 5, 7]}
        ... )

        Custom electrode pairing for specialized montage:

        >>> custom_pairs = {'O1': 'O2', 'P7': 'P8', 'PO7': 'PO8'}
        >>> flipped_epochs = ERP.flip_topography(
        ...     epochs, behavioral_df,
        ...     topo_flip={'stimulus_location': ['left_field']},
        ...     flip_dict=custom_pairs
        ... )

        Memory task with HEOG correction:

        >>> flipped_epochs = ERP.flip_topography(
        ...     epochs, behavioral_df,
        ...     topo_flip={'memory_array_side': [2, 8]},
        ...     heog='Horizontal_EOG'
        ... )

        See Also
        --------
        select_lateralization_idx : Select trials for lateralized 
        analysis
        condition_erps : Main ERP analysis method using topographic 
        flipping
        group_lateralized_erp : Group-level lateralized difference waves
        """

        # Extract position criteria from topo_flip dictionary
        (header, left), = topo_flip.items()

        picks = mne.pick_types(epochs.info, eeg=True, csd = True)   
        # dictionary to flip topographic layout
        if flip_dict is None:
            # create flip dictionary based electrodes in layout
            print('No flip dictionary specified. Creating flip ' \
            'based on epochs layout. Assumes that odd electrodes are' \
            ' left and even electrodes are right')
            flip_dict = {}
            for elec in epochs.ch_names:
                if elec[-1].isdigit():  
                    base_name = elec[:-1] 
                    number = int(elec[-1])  
                    if number % 2 == 1: 
                        mirror_elec = f"{base_name}{number + 1}"  
                        if mirror_elec in epochs.ch_names:  
                            flip_dict[elec] = mirror_elec

        idx_l = np.hstack([np.where(df[header] == l)[0] for l in left])

        # left stimuli are flipped as if presented right
        pre_flip = np.copy(epochs._data[idx_l][:,picks])

        # do actual flipping
        print('flipping topography')
        for l_elec, r_elec in flip_dict.items():
            l_elec_data = pre_flip[:,epochs.ch_names.index(l_elec)]
            r_elec_data = pre_flip[:,epochs.ch_names.index(r_elec)]
            epochs._data[idx_l,epochs.ch_names.index(l_elec)] = r_elec_data
            epochs._data[idx_l,epochs.ch_names.index(r_elec)] = l_elec_data

        if heog in epochs.ch_names:
            epochs._data[idx_l,epochs.ch_names.index(heog)] *= -1

        return epochs

    @staticmethod
    def select_lateralization_idx(
        df: pd.DataFrame, 
        pos_labels: dict, 
        midline: Optional[dict] = None
    ) -> np.array:
        """
        Select trial indices for lateralized ERP analysis with optional 
        midline constraints.

        This method identifies trials based on spatial position criteria 
        for lateralized experimental designs. It selects trials where 
        stimuli of interest appear at specified lateral positions and 
        optionally restricts selection to trials where additional 
        stimuli are presented on the vertical midline. This function is 
        essential for lateralized ERP components analysis such as N2pc, 
        SPCN, CDA, and other spatial attention paradigms.

        The method supports both simple position-based selection and 
        complex experimental designs where multiple stimuli must meet 
        specific spatial criteria simultaneously.

        Parameters
        ----------
        df : pd.DataFrame
            Behavioral dataframe containing trial-level stimulus and 
            response information. Must include position columns 
            corresponding to the spatial layout used in the experiment. 
            Trial order must match the corresponding EEG epochs for 
            proper alignment.
        pos_labels : dict
            Position-based selection criteria specifying which trials to 
            include. Key indicates the behavioral column containing 
            spatial position information, value provides list of 
            position labels to select. Supports both lateralized and 
            non-lateralized position selection.
            Example: {'target_loc': [2, 4, 6, 8]} selects trials where 
            targets appeared at right visual field positions.
        midline : dict or None, default=None
            Additional spatial constraint for trials where other stimuli 
            appear at midline positions. Key specifies behavioral column 
            for secondary stimulus positions, value lists acceptable 
            midline position labels. Used to control for distractor 
            placement or enforce specific stimulus configurations.
            Example: {'distractor_loc': [0]} limits selection to trials 
            where distractors appeared at central fixation.

        Returns
        -------
        np.ndarray
            Array of trial indices (integers) identifying selected 
            trials in the behavioral dataframe. These indices can be 
            used directly for EEG epoch selection and maintain 
            correspondence between behavioral and neural data.

        Notes
        -----
        Selection Logic:
        1. Extract position criteria from pos_labels dictionary
        2. Identify all trials matching position requirements
        3. If midline specified, apply additional spatial constraints
        4. Return intersection of position and midline criteria

        Spatial Position Systems:
        The method works with any spatial position encoding system 
        commonly used in lateralized paradigms:
        - Clock positions: 1-8 representing locations around fixation
        - Cartesian coordinates: explicit x,y position values  
        - Categorical labels: 'left', 'right', 'center' descriptors
        - Custom encoding: any consistent position labeling scheme

        Multiple Constraint Handling:
        When midline parameter is provided, the method applies 
        intersection logic to ensure trials meet both primary position 
        criteria and secondary midline constraints. This enables 
        experimental control over stimulus configurations.

        Common Applications:
        - N2pc paradigms: Select lateral target trials with central 
          distractors
        - SPCN studies: Choose memory array positions with specific 
          layouts
        - CDA experiments: Control bilateral vs unilateral memory loads
        - Attention cueing: Select valid/invalid cue-target combinations

        Examples
        --------
        Basic lateral position selection for N2pc analysis:

        >>> target_trials = ERP.select_lateralization_idx(
        ...     behavioral_df, 
        ...     pos_labels={'target_position': [2, 4, 6, 8]}
        ... )

        Complex selection with midline distractor control:

        >>> controlled_trials = ERP.select_lateralization_idx(
        ...     behavioral_df,
        ...     pos_labels={'target_loc': [1, 3, 5, 7]},
        ...     midline={'distractor_loc': [0]}
        ... )

        Memory paradigm with bilateral item selection:

        >>> memory_trials = ERP.select_lateralization_idx(
        ...     behavioral_df,
        ...     pos_labels={'memory_items': ['left', 'right']},
        ...     midline={'central_cue': ['fixation']}
        ... )

        Multiple distractor positions with lateral targets:

        >>> multi_constraint = ERP.select_lateralization_idx(
        ...     behavioral_df,
        ...     pos_labels={'target_side': [2, 8]},
        ...     midline={'dist1_loc': [0], 'dist2_loc': [4, 6]}
        ... )

        See Also
        --------
        condition_erps : Primary ERP analysis method using this 
        selection function
        """

        # select all lateralized trials	
        (header, labels), = pos_labels.items()
        idx = np.hstack([np.where(df[header] == l)[0] for l in labels])

        # limit to midline trials
        if  midline is not  None:
            idx_m = []
            for key in midline.keys():
                idx_m.append(np.hstack([np.where(df[key] == m)[0] 
                                        for m in midline[key]]))
            idx_m = np.hstack(idx_m)
            idx = np.intersect1d(idx, idx_m)

        return idx
    
    @staticmethod
    def select_erp_window(
        erps: Union[dict, list], 
        elec_oi: list, 
        method: str = 'cnd_avg', 
        window_oi: Optional[tuple] = None, 
        polarity: str = 'pos', 
        window_size: float = 0.05
    ) -> Union[tuple, dict]:
        """
        Determine optimal ERP analysis time windows using peak detection 
        algorithms.

        This method automatically identifies time windows for ERP 
        component analysis by detecting peak activity within specified
        electrodes and time ranges. The peak detection can operate on 
        grand averaged data across all conditions or on 
        condition-specific waveforms, supporting both standard electrode 
        averaging and lateralized difference wave analysis. This is 
        essential for objective, data-driven selection of analysis 
        windows for ERP components.

        The method provides robust window selection for ERP components 
        where peak latency may vary across conditions, subjects, or 
        experimental manipulations, eliminating subjective bias in 
        temporal window definition.

        Parameters
        ----------
        erps : dict or list
            ERP data for peak detection analysis. If dictionary, keys 
            represent condition names and values contain lists of 
            mne.Evoked objects for each subject/trial. If list, 
            contains mne.Evoked objects directly for grand average 
            analysis. The method is designed for group-level analysis 
            where multiple evoked objects are averaged together.
        elec_oi : list
            Electrode specification for peak detection. Can be either:
            - List of electrode names (str) for standard averaging 
              across specified channels (e.g., ['Cz', 'CPz'] for 
              midline analysis)
            - Nested list for lateralized difference waves with format 
              [contralateral_electrodes, ipsilateral_electrodes] 
              enabling automatic difference wave computation 
              (e.g., [['O1', 'PO7'], ['O2', 'PO8']])
        method : str, default='cnd_avg'
            Peak detection strategy determining window selection
            approach:
            - 'cnd_avg': Uses grand average across all conditions for 
              unified window selection, ensuring consistent temporal 
              analysis across conditions
            - 'cnd_spc': Generates condition-specific windows based on 
              individual condition peaks, allowing adaptive windows for 
              each experimental condition
        window_oi : tuple or None, default=None
            Temporal search range for peak detection as (tmin, tmax) in
            seconds. When specified, constrains peak detection to this 
            interval, useful for targeting specific ERP components or 
            avoiding artifacts. If None, searches across the full epoch 
            duration.Example: (0.1, 0.4) searches for P300 peaks 
            between 100-400ms.
        polarity : str, default='pos'
            Expected polarity of the target ERP component for peak 
            detection:
            - 'pos': Searches for positive-going peaks (e.g., P300, P1)
            - 'neg': Searches for negative-going peaks (e.g., N2pc)
            Determines whether to use maximum or minimum detection 
            algorithms.
        window_size : float, default=0.05
            Size of the analysis window in seconds, centered on the 
            detected peak. The final window extends ±window_size/2 
            around the peak latency. Should be chosen based on component 
            characteristics and temporal resolution requirements 
            (typically 20-100ms for most ERP components).

        Returns
        -------
        list or dict
            Analysis time window(s) with format depending on method 
            parameter:
            - If method='cnd_avg': Returns single list 
              [tmin, tmax, polarity] representing the unified time 
              window across all conditions
            - If method='cnd_spc': Returns dictionary with condition 
              names as keys and [tmin, tmax, polarity] lists as values 
              for condition-specific windows.

        Notes
        -----
        Peak Detection Algorithm:
        1. Extract electrode data based on elec_oi specification
        2. For lateralized analysis, compute difference waves 
           (contra - ipsi)
        3. Average across specified electrodes to create summary 
           waveform
        4. Apply temporal constraints from window_oi parameter
        5. Detect peak using polarity-appropriate algorithm (max/min)
        6. Center analysis window around detected peak latency

        Lateralized Difference Waves:
        When elec_oi contains nested lists, the method automatically 
        computes contralateral minus ipsilateral difference waves. This 
        is essential for lateralized ERP components (N2pc, SPCN, CDA) 
        where the signal of interest is the hemispheric difference 
        rather than absolute amplitude.

        Method Selection Guidelines:
        - Use 'cnd_avg' for components with consistent timing across 
          conditions
        - Use 'cnd_spc' when experimental manipulations affect 
          component latency
        - Consider 'cnd_avg' for group analyses requiring uniform time 
          windows
        - Consider 'cnd_spc' for exploratory analyses or variable 
          components

        Examples
        --------
        Grand average P300 window selection:

        >>> p300_window = ERP.select_erp_window(
        ...     erps={'standard': std_erps, 'deviant': dev_erps},
        ...     elec_oi=['Cz', 'CPz', 'Pz'],
        ...     method='cnd_avg', window_oi=(0.25, 0.45),
        ...     polarity='pos', window_size=0.05
        ... )

        Condition-specific N170 windows:

        >>> n170_windows = ERP.select_erp_window(
        ...     erps={'faces': face_erps, 'objects': obj_erps},
        ...     elec_oi=['P7', 'P8', 'PO7', 'PO8'],
        ...     method='cnd_spc', window_oi=(0.12, 0.22),
        ...     polarity='neg', window_size=0.03
        ... )

        Lateralized N2pc difference wave window:

        >>> n2pc_window = ERP.select_erp_window(
        ...     erps={'target_left': left_erps, 
        ...    'target_right': right_erps},
        ...     elec_oi=[['O1', 'PO7'], ['O2', 'PO8']],
        ...     method='cnd_avg', window_oi=(0.18, 0.28),
        ...     polarity='neg', window_size=0.04
        ... )

        Automatic LPP window with full epoch search:

        >>> lpp_window = ERP.select_erp_window(
        ...     erps=emotion_erps,
        ...     elec_oi=['CPz', 'Pz'],
        ...     method='cnd_avg', window_oi=(0.4, 0.8),
        ...     polarity='pos', window_size=0.1
        ... )

        See Also
        --------
        get_erp_params : Extract channel and timing information from 
        ERP data
        export_erp_metrics_to_csv : Export metrics using selected 
        windows
        """

        # set params
        channels, times = ERP.get_erp_params(erps)

        if isinstance(elec_oi[0], str):
            elec_oi_idx = np.array([channels.index(elec) 
                                            for elec in elec_oi])
        else:
            contra_idx = np.array([channels.index(elec) 
                                            for elec in elec_oi[0]])
            ipsi_idx = np.array([channels.index(elec) 
                                            for elec in elec_oi[1]])

        # get window of interest
        if window_oi is None:
            window_oi = (times[0], times[-1])
        window_idx = get_time_slice(times, window_oi[0], window_oi[1])

        if method == 'cnd_avg':
            # find peak in grand average waveform
            # step 1: create condition averaged waveform
            grand_mean = mne.combine_evoked(
                        [mne.combine_evoked(v,weights='equal') 
                                        for (k,v) in erps.items()]
                                ,weights = 'equal')

            # step 2: limit data to electrodes of interest
            if isinstance(elec_oi[0], str):
                X = grand_mean._data[elec_oi_idx]
            else:
                X = grand_mean._data[contra_idx] - grand_mean._data[ipsi_idx]

            # average over electrodes
            X = X.mean(axis = 0)
            # step 3: get time window based on peak detection
            if polarity == 'pos':
                idx_peak = np.argmax(X[window_idx])
            elif polarity == 'neg':
                idx_peak = np.argmin(X[window_idx])
            
            erp_window = [times[window_idx][idx_peak] - window_size/2, 
                         times[window_idx][idx_peak] + window_size/2,
                         polarity]
        
        elif method == 'cnd_spc':
            erp_window = {}
            # loop over conditins
            for cnd in list(erps.keys()):
                cnd_mean = mne.combine_evoked(erps[cnd], weights = 'equal')

                if isinstance(elec_oi[0], str):
                    X = cnd_mean._data[elec_oi_idx]
                else:
                    X = cnd_mean._data[contra_idx] - cnd_mean._data[ipsi_idx]
                
                # average over electrodes
                X = X.mean(axis = 0)
                if polarity == 'pos':
                    idx_peak = np.argmax(X[window_idx])
                elif polarity == 'neg':
                    idx_peak = np.argmin(X[window_idx])
                
                erp_window[cnd] = [times[window_idx][idx_peak] - window_size/2, 
                                   times[window_idx][idx_peak] + window_size/2,
                                   polarity]

        return erp_window
    
    @staticmethod
    def get_erp_params(erps:Union[dict,list])->Tuple[list,np.array]:
        """
        Extract channel names and time points from ERP data.

        Parameters
        ----------
        erps : dict or list
            ERP data as dictionary of condition lists or direct list 
            of mne.Evoked objects.

        Returns
        -------
        channels : list
            EEG channel names.
        times : np.ndarray
            Time points in seconds.
        """

        # set params
        if type(erps) == dict:
            channels = list(erps.items())[0][1][0].ch_names
            times = list(erps.items())[0][1][0].times
        else:
            channels= erps[0].ch_names
            times = erps[0].times

        return channels, times

    @staticmethod
    def export_erp_metrics_to_csv(
        erps: Union[dict, list], 
        window_oi: Union[list, dict], 
        elec_oi: list, 
        cnds: list = None, 
        method: str = 'mean_amp', 
        name: str = 'main'
    ):
        """
        Export ERP metrics to CSV file for statistical analysis.

        Calculates and exports ERP amplitude metrics across conditions 
        and electrodes to CSV format. Supports both standard electrode 
        averaging and lateralized difference wave analysis. The output 
        is organized for direct import into statistical software with 
        subjects as rows and condition/electrode combinations as 
        columns.

        Parameters
        ----------
        erps : dict or list
            ERP data for metric extraction. If dictionary, keys are 
            condition names and values are lists of mne.Evoked objects 
            (one per subject). If list, contains evoked objects directly 
            and is treated as single condition.
        window_oi : list or dict
            Time window for metric calculation in seconds as 
            [tmin, tmax]. If list, same window applied to all 
            conditions. If dictionary, keys match condition names with 
            condition-specific windows as values.
        elec_oi : list
            Electrode specification for analysis. Either list of 
            electrode names (str) for standard averaging, or nested list 
            [[contralateral_electrodes], [ipsilateral_electrodes]] for 
            lateralized difference wave analysis.
        cnds : list, optional
            Condition names to include in export. If None, processes all 
            available conditions from erps dictionary. Default is None.
        method : str, default='mean_amp'
            Metric calculation method. Supported options:
            - 'mean_amp': Mean amplitude across time window
            - 'auc': Area under curve for all values
            - 'auc_pos': Area under curve for positive values only
            - 'auc_neg': Area under curve for negative values only  
            - 'onset_latency': Component onset latency detection
        name : str, default='main'
            Output CSV filename (without extension). File is saved in 
            'erp/stats/' subfolder.

        Notes
        -----
        Output Organization:
        - Rows represent individual subjects/observations
        - Columns represent condition/electrode combinations
        - For lateralized analysis: generates contralateral, 
          ipsilateral, and difference wave columns per condition
        - Header row contains condition labels for data identification

        File Structure:
        The CSV file is saved with comma delimiters and includes a 
        header row. For lateralized designs, each condition generates 
        three columns: '{condition}_contra', '{condition}_ipsi', and 
        '{condition}_diff' representing contralateral, ipsilateral, and 
        difference wave metrics respectively.

        Statistical Compatibility:
        Output format is designed for direct import into R, SPSS, Python 
        pandas, or other statistical software. The transpose 
        organization (subjects x conditions) follows standard 
        conventions for repeated measures analysis.

        Examples
        --------
        Export standard P300 amplitude metrics:

        >>> ERP.export_erp_metrics_to_csv(
        ...     erps={'standard': std_erps, 'deviant': dev_erps},
        ...     window_oi=[0.3, 0.5],
        ...     elec_oi=['Cz', 'CPz', 'Pz'],
        ...     method='mean_amp', name='P300_analysis'
        ... )

        Export lateralized N2pc difference waves with 
        condition-specific windows:

        >>> windows = {'target': [0.18, 0.28], 
        ...           'distractor': [0.20, 0.30]}
        >>> ERP.export_erp_metrics_to_csv(
        ...     erps={'target': targ_erps, 'distractor': dist_erps},
        ...     window_oi=windows,
        ...     elec_oi=[['O1', 'PO7'], ['O2', 'PO8']],
        ...     method='mean_amp', name='N2pc_lateralized'
        ... )

        Export area under curve for emotion ERP analysis:

        >>> ERP.export_erp_metrics_to_csv(
        ...     erps={'positive': pos_erps, 'negative': neg_erps},
        ...     window_oi=[0.4, 0.8],
        ...     elec_oi=['CPz', 'Pz'],
        ...     method='auc', name='emotion_LPP'
        ... )

        See Also
        --------
        extract_erp_features : Individual metric calculation methods
        group_erp : Group-level ERP data preparation
        select_erp_window : Automatic time window selection
        """

        if not isinstance(erps, (dict, list)):
            raise ValueError("erps must be a dictionary or a list of " \
                            "evoked objects.")
        if not isinstance(window_oi, (list, dict)):
            raise ValueError("window_oi must be a list or a dictionary.")

        # initialize output list and set parameters
        X, headers = [], []
        if isinstance(erps, list):
            erps = {'data': erps}
            cnds = ['data']
        elif cnds is None:
            cnds = list(erps.keys())
        
        # get channels and times
        channels, times = ERP.get_erp_params(erps)

        if isinstance(window_oi, list):
            idx = get_time_slice(times, window_oi[0], window_oi[1])

        # extract condition specific data
        for cnd in cnds:
            if isinstance(window_oi, dict):
                idx = get_time_slice(times,window_oi[cnd][0],window_oi[cnd][1])

            # check whether output needs to be lateralized
            if isinstance(elec_oi[0], str):
                evoked_X, _ = ERP.group_erp(erps[cnd],elec_oi)
                y = ERP.extract_erp_features(evoked_X[:,idx],times[idx],method)
                X.append(y)
                headers.append(cnd)
            else:
                d_wave = []
                for h, hemi in enumerate(['contra','ipsi']):
                    evoked_X, _ = ERP.group_erp(erps[cnd],elec_oi[h])
                    d_wave.append(evoked_X)
                    y = ERP.extract_erp_features(evoked_X[:,idx],
                                                 times[idx],method)
                    X.append(y)
                    headers.append(f'{cnd}_{hemi}')

                # add contra vs hemi difference
                d_wave = d_wave[0] - d_wave[1]
                y = ERP.extract_erp_features(d_wave[:,idx],times[idx],method)
                X.append(y)
                headers.append(f'{cnd}_diff')

        # save data
        np.savetxt(ERP.folder_tracker(['erp','stats'], 
               fname = f'{name}.csv'),np.stack(X).T, 
               delimiter = "," ,header = ",".join(headers),comments='')
        
    @staticmethod
    def group_erp(
        erp: list, 
        elec_oi: list = 'all'
    ) -> Tuple[np.array, mne.Evoked]:
        """
        Combine individual ERP data at the group level.

        This function aggregates individual ERP data across subjects 
        or conditions to create group-level averages and extract 
        electrode-specific data for further analysis.

        Parameters
        ----------
        erp : list
            List of mne.Evoked objects containing individual ERP data.
        elec_oi : list or str, optional
            Electrodes of interest for data extraction. If 'all', 
            includes all available electrodes. Default is 'all'.

        Returns
        -------
        evoked_X : np.ndarray
            Stacked individual ERP data for specified electrodes, 
            averaged across electrodes (n_subjects, n_timepoints).
        evoked : mne.Evoked
            Group-level evoked object created by averaging all 
            individual evoked objects with equal weights.

        Notes
        -----
        The function uses MNE-Python's combine_evoked with equal 
        weights to create unbiased group averages. Individual data 
        is preserved in evoked_X for subsequent statistical analysis.

        Examples
        --------
        >>> evoked_data, group_evoked = ERP.group_erp(
        ...     [subj1_erp, subj2_erp, subj3_erp], 
        ...     elec_oi=['Cz', 'Pz']
        ... )
        """

        # Compute group-level evoked object
        evoked = mne.combine_evoked(erp, weights='equal')
        channels = evoked.ch_names

        # Handle electrodes of interest
        if elec_oi == 'all':
            elec_oi = channels
        elec_oi_idx = np.array([channels.index(elec) for elec in elec_oi])
        
        # Stack individual ERP data for the specified electrodes
        evoked_X = np.stack([e._data[elec_oi_idx] for e in erp])
        evoked_X = evoked_X.mean(axis = 1)
        
        return evoked_X, evoked
    
    @staticmethod
    def extract_erp_features(
        X: np.ndarray, 
        times: np.ndarray, 
        method: str,
        threshold: float = 0.5,
        polarity: str = 'pos'
    ) -> np.ndarray:
        """
        Calculate amplitude and timing metrics from ERP data.

        Computes various metrics from ERP waveform data including mean 
        amplitude, area under curve measurements, and onset latency 
        detection. This function supports standard amplitude analysis, 
        polarity-specific area calculations, and timing-based measures 
        needed for ERP component characterization.

        Parameters
        ----------
        X : np.ndarray
            ERP amplitude data with shape (n_subjects, n_timepoints) 
            containing averaged waveforms for metric calculation. Each 
            row represents one subject or trial, and columns represent 
            time samples within the analysis window.
        times : np.ndarray
            Time points in seconds corresponding to the columns in X. 
            Must have same length as X.shape[1]. Used for area 
            calculations and onset latency timing.
        method : str
            Metric calculation method specifying the type of analysis:
            - 'mean_amp': Mean amplitude across time window
            - 'auc': Total area under curve (positive + negative)
            - 'auc_pos': Area under curve for positive values only
            - 'auc_neg': Area under curve for negative values only
            - 'onset_latency': Component onset latency detection
        threshold : float, default=0.5
            Threshold percentage (0-1) of peak amplitude for onset 
            latency detection. For example, 0.5 finds the first time 
            point reaching 50% of peak amplitude. Only used when 
            method='onset_latency'.
        polarity : str, default='pos'
            Expected component polarity for onset latency calculation:
            - 'pos': Positive-going components (P300, P1, LPP)
            - 'neg': Negative-going components (N170, N2pc, N400)
            Only used when method='onset_latency'.

        Returns
        -------
        np.ndarray
            Calculated metrics with shape (n_subjects,) containing the 
            requested measure for each subject/trial. Units depend on 
            method: microvolts for amplitudes, microvolt*seconds for 
            areas, seconds for onset latencies.

        Notes
        -----
        Metric Calculations:
        - **Mean Amplitude**: Simple average across time window, most 
          common ERP measure
        - **Total AUC**: Integrates all signal deflections using 
          trapezoidal rule
        - **Positive AUC**: Integrates only positive deflections, 
          negative values treated as zero
        - **Negative AUC**: Integrates only negative deflections, 
          positive values treated as zero  
        - **Onset Latency**: Time to reach threshold percentage of 
          peak amplitude

        Onset Latency Algorithm:
        1. Invert signal for negative polarity components
        2. Find peak amplitude in time window
        3. Calculate threshold as percentage of peak
        4. Identify first time point exceeding threshold
        5. Return corresponding time value or NaN if no onset found

        Examples
        --------
        Calculate mean P300 amplitude:

        >>> p300_amps = ERP.extract_erp_features(
        ...     erp_data, times, method='mean_amp'
        ... )

        Calculate area under curve for positive LPP deflections:

        >>> lpp_auc = ERP.extract_erp_features(
        ...     erp_data, times, method='auc_pos'
        ... )

        Detect N170 onset latency at 75% of peak:

        >>> n170_onset = ERP.extract_erp_features(
        ...     erp_data, times, method='onset_latency',
        ...     threshold=0.75, polarity='neg'
        ... )

        Calculate total signed area for difference waves:

        >>> diff_auc = ERP.extract_erp_features(
        ...     difference_waves, times, method='auc'
        ... )

        See Also
        --------
        export_erp_metrics_to_csv : Export calculated metrics to CSV
        sklearn.metrics.auc : Underlying area calculation function
        """

        if method == 'mean_amp':
            output = X.mean(axis=-1)
            
        elif 'auc' in method:
            output = []
            for x in X:
                # Create copy to avoid modifying input data
                x_copy = x.copy()
                
                if 'pos' in method:
                    x_copy[x_copy < 0] = 0
                elif 'neg' in method:
                    x_copy[x_copy > 0] = 0
                    
                output.append(auc(times, x_copy))
            output = np.array(output)
            
        elif method == 'onset_latency':
            output = []
            for x in X:
                # Create copy to avoid modifying input data
                x_copy = x.copy()
                
                if polarity == 'neg':
                    x_copy *= -1  # Invert signal for negative polarity
                    
                peak_value = x_copy[np.argmax(x_copy)]
                thresh_value = peak_value * threshold
                
                # Find first time point exceeding the threshold
                idx = np.where(x_copy >= thresh_value)[0]
                if idx.size > 0:
                    output.append(times[idx[0]])
                else:
                    output.append(np.nan)  # No onset found
                    
            output = np.array(output)
        
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods: "
                           "'mean_amp', 'auc', 'auc_pos', 'auc_neg', " 
                           "'onset_latency'")

        return output

    @staticmethod
    def lateralized_erp_idx(erp: list, elec_oi_c: list, 
                           elec_oi_i: list) -> Tuple[np.array, np.array]:
        """
        Get electrode indices for lateralized ERP analysis.

        This helper function identifies the indices of contralateral 
        and ipsilateral electrodes in the ERP data for lateralized 
        component analysis.

        Parameters
        ----------
        erp : list
            List of mne.Evoked objects containing ERP data.
        elec_oi_c : list
            Names of contralateral electrodes of interest.
        elec_oi_i : list
            Names of ipsilateral electrodes of interest.

        Returns
        -------
        contra_idx : np.ndarray
            Array of indices corresponding to contralateral electrodes.
        ipsi_idx : np.ndarray
            Array of indices corresponding to ipsilateral electrodes.

        Notes
        -----
        This is a utility function primarily used by other methods 
        for lateralized ERP analysis. It ensures proper electrode 
        mapping for difference wave calculations.

        Examples
        --------
        >>> contra_idx, ipsi_idx = ERP.lateralized_erp_idx(
        ...     erp_list, ['O1', 'PO7'], ['O2', 'PO8']
        ... )
        """
        
        # extract channels from erps
        channels = erp[0].ch_names

        # get indices
        contra_idx = np.array([channels.index(ch) for ch in elec_oi_c])
        ipsi_idx = np.array([channels.index(ch) for ch in elec_oi_i])

        return contra_idx, ipsi_idx
    
    @staticmethod
    def select_waveform(erps: list, elec_oi: list):
        """
        Extract and average ERP waveforms for specified electrodes.

        This function extracts ERP data from specified electrodes across 
        multiple evoked objects and supports both standard electrode 
        selection and lateralized difference wave calculations.

        Parameters
        ----------
        erps : list
            List of mne.Evoked objects containing ERP data.
        elec_oi : list
            Electrodes of interest. Can be:
            - List of electrode names (str) for standard averaging
            - List of two lists for lateralized analysis 
              [contralateral_electrodes, ipsilateral_electrodes]

        Returns
        -------
        np.ndarray
            Array of averaged ERP waveforms for each evoked object.
            Shape: (n_evoked, n_times).

        Notes
        -----
        For lateralized analysis, the function computes the difference 
        between contralateral and ipsilateral electrodes for each 
        evoked object, then averages across the specified electrode 
        pairs.

        Examples
        --------
        Standard electrode averaging:
        >>> waveforms = ERP.select_waveform(erp_list, ['Cz', 'Pz'])

        Lateralized difference waves:
        >>> lat_waveforms = ERP.select_waveform(
        ...     erp_list, [['O1', 'PO7'], ['O2', 'PO8']]
        ... )
        """

        channels, times = ERP.get_erp_params(erps)
        if type(elec_oi[0]) == str:
            ch_idx = [channels.index(elec) for elec in elec_oi]
            x = np.stack([evoked._data[ch_idx] for evoked in erps])
        elif type(elec_oi[0]) == list:
            contra_idx = [channels.index(elec) for elec in elec_oi[0]]
            contra = np.stack([evoked._data[contra_idx] for evoked in erps])
            ipsi_idx = [channels.index(elec) for elec in elec_oi[1]]
            ipsi = np.stack([evoked._data[ipsi_idx] for evoked in erps])
            x = (contra - ipsi)

        x = x.mean(axis = 1)

        return x

    @staticmethod
    def compare_latencies(
        erps: Union[dict, list], 
        elec_oi: list = None, 
        window_oi: tuple = None, 
        times: np.ndarray = None, 
        percent_amp: int = 75, 
        polarity: str = 'pos', 
        phase: str = 'onset'
    ):
        """
        Compare ERP component latencies using jackknife method.

        This function compares latencies between ERP conditions using 
        the jackknife method for robust statistical estimation. It 
        calculates latency differences based on a specified percentage 
        of peak amplitude and provides statistical significance testing.

        Parameters
        ----------
        erps : dict or list
            ERP data for comparison. If dict, keys are condition names 
            and values are lists of evoked objects. If list, should 
            contain two ERP waveforms to compare directly.
        elec_oi : list, optional
            Electrode specification for ERP waveform extraction. 
            Required if erps is a dictionary. Can be either:
            - List of electrode names (str) for standard averaging 
              across specified channels (e.g., ['Cz', 'CPz'])
            - Nested list for lateralized difference waves with format 
              [contralateral_electrodes, ipsilateral_electrodes] 
              enabling automatic difference wave computation 
              (e.g., [['O1', 'PO7'], ['O2', 'PO8']])
            Default is None.
        window_oi : tuple, optional
            Time window of interest (start, end) in seconds. If None, 
            uses the full time range. Default is None.
        times : np.ndarray, optional
            Array of time points corresponding to ERP data. 
            Default is None.
        percent_amp : int, optional
            Percentage of peak amplitude for latency calculation. 
            Default is 75.
        polarity : str, optional
            Polarity of the ERP component. Use 'pos' for positive 
            components or 'neg' for negative components.
            Default is 'pos'.
        phase : str, optional
            Phase for latency calculation. Use 'onset' for component 
            onset or 'offset' for component offset. Default is 'onset'.

        Returns
        -------
        tuple or dict
            If comparing single pair: tuple of 
            (latency_difference, t_value). If comparing multiple pairs: 
            dict with condition pairs as keys and 
            (latency_difference, t_value) tuples as values.

        Notes
        -----
        The jackknife method provides robust estimates of latency 
        differences and their statistical significance. This is the 
        recommended approach for ERP latency comparisons as described 
        in Miller et al. (1998) and Ulrich & Miller (2001).

        For offset latency analysis, the waveforms are time-reversed 
        before analysis to detect the offset timing.

        References
        ----------
        Miller, J., Patterson, T., & Ulrich, R. (1998). Jackknife-based 
        method for measuring LRP onset latency differences. 
        Psychophysiology, 35(1), 99-115.

        Examples
        --------
        Compare onset latencies between conditions using 
        standard electrodes:
        >>> lat_diff, t_val = ERP.compare_latencies(
        ...     erps={'standard': std_erps, 'deviant': dev_erps},
        ...     elec_oi=['Cz'], window_oi=(0.1, 0.4),
        ...     percent_amp=50, polarity='pos'
        ... )

        Compare lateralized difference wave latencies:
        >>> lat_diff, t_val = ERP.compare_latencies(
        ...     erps={'target_left': left_erps, 
        ...'target_right': right_erps},
        ...     elec_oi=[['O1', 'PO7'], ['O2', 'PO8']], 
        ...     window_oi=(0.18, 0.28), percent_amp=75, polarity='neg'
        ... )
        """

        #set params
        if type(erps) == dict:
            pairs = list(combinations(erps.keys(),2))
            times = erps[pairs[0][0]][0].times
        elif type(erps) == list:
            pairs = ['']

        # set window of interest
        if window_oi is None:
            window_oi = (times[0],times[-1])
        window_idx = get_time_slice(times, window_oi[0], window_oi[1])
        times_oi = times[window_idx] 

        output = {}
        for p, pair in enumerate(pairs):
            print(f'Contrasting {pair} using jackknife method')

            if type(erps) == dict:
                x1 = ERP.select_waveform(erps[pair[0]], elec_oi)[:,window_idx]
                x2 = ERP.select_waveform(erps[pair[1]], elec_oi)[:,window_idx]
            elif type(erps) == list:
                x1 = erps[0][:,window_idx]
                x2 = erps[1][:,window_idx]   

            if phase == 'offset':
                x1 = np.fliplr(x1)
                x2 = np.fliplr(x2)   
                # For offset analysis, flip times to analyze from end
                #TODO: double check this is correct
                if p == 0:
                    times_oi = times_oi[::-1] 

            if polarity == 'neg':
                x1 *= -1
                x2 *= -1
                
            (d_latency, 
            t) = ERP.jackknife_contrast(x1,x2,times_oi,percent_amp)

            if len(pairs) == 1:
                return (d_latency, t)
            else:
                output['_'.join(pair)] = (d_latency, t)
        
        return output

    @staticmethod
    def jackknife_contrast(
        x1: np.ndarray, 
        x2: np.ndarray, 
        times: np.ndarray, 
        percent_amp: int
    ) -> Tuple[float, float]:
        """
        Perform jackknife-based latency contrast between ERP waveforms.

        This function implements the jackknife method for robust 
        estimation of latency differences between two ERP conditions. 
        The method provides both the latency difference estimate and 
        its statistical significance.

        Parameters
        ----------
        x1 : np.ndarray
            ERP waveform for condition 1 (trials x timepoints).
        x2 : np.ndarray
            ERP waveform for condition 2 (trials x timepoints).
        times : np.ndarray
            Array of time points corresponding to the ERP data.
        percent_amp : int
            Percentage of peak amplitude for latency threshold 
            calculation.

        Returns
        -------
        d_latency : float
            Estimated latency difference between the two conditions.
        t_value : float
            T-statistic for the latency difference based on jackknife 
            variance estimation.

        Notes
        -----
        The jackknife method estimates the variance of the latency 
        difference by systematically omitting each observation and 
        recalculating the statistic. This provides robust statistical 
        inference for latency comparisons.

        The method follows the approach described in Miller et al. 
        (1998) for ERP latency analysis.

        References
        ----------
        Miller, J., Patterson, T., & Ulrich, R. (1998). Jackknife-based 
        method for measuring LRP onset latency differences. 
        Psychophysiology, 35(1), 99-115.

        Examples
        --------
        >>> lat_diff, t_stat = ERP.jackknife_contrast(
        ...     condition1_data, condition2_data, times, 75
        ... )
        """ 

        # set params
        nr_obs = x1.shape[0]
        
        if nr_obs != x2.shape[0]:
            raise ValueError("x1 and x2 must have the same number of " \
                            "observations")
        if nr_obs < 2:
            raise ValueError("Need at least 2 observations for " \
                            "jackknife analysis")

        # step 1: get grand mean latency difference
        x1_ = x1.mean(axis = 0) 
        x2_ = x2.mean(axis = 0) 
        c1 = max(x1_) * percent_amp/100.0
        c2 = max(x2_) * percent_amp/100.0 

        try:
            d_latency = ERP.jack_latency_contrast(x1_,x2_,c1,c2,times,True)
        except ValueError as e:
            raise ValueError(f"Error in grand mean latency calculation: {e}")

        # step 2: jackknifing procedure ()
        D = []
        idx = np.arange(nr_obs)
        for i in range(nr_obs):
            x1_ = x1[i != idx,:].mean(axis = 0)
            x2_ = x2[i != idx,:].mean(axis = 0)
            c1 = max(x1_) * percent_amp/100.0
            c2 = max(x2_) * percent_amp/100.0 
            try:
                D.append(ERP.jack_latency_contrast(x1_,x2_,c1,c2,times))
            except ValueError:
                # Skip this iteration if threshold not reached
                continue
        
        if len(D) == 0:
            raise ValueError("No valid jackknife iterations - threshold " \
                            "may be too high")

        # compute the jackknife estimate 
        Sd = np.sqrt((nr_obs - 1.0)/ nr_obs \
            * np.sum([(d - np.mean(D))**2 for d in np.array(D)]))	

        if Sd == 0:
            raise ValueError("Standard deviation is zero - cannot" \
                            " compute t-statistic")
            
        t_value = d_latency/ Sd 

        return d_latency, t_value

    @staticmethod
    def jack_latency_contrast(
        x1: np.ndarray, 
        x2: np.ndarray, 
        c1: float, 
        c2: float, 
        times: np.ndarray, 
        print_output: bool = False
    ) -> float:
        """
        Calculate latency difference at specified amplitude thresholds.

        This helper function determines the precise latency at which 
        two ERP waveforms reach specified amplitude thresholds and 
        calculates the difference between these latencies using 
        linear interpolation.

        Parameters
        ----------
        x1 : np.ndarray
            ERP waveform for condition 1 (timepoints).
        x2 : np.ndarray
            ERP waveform for condition 2 (timepoints).
        c1 : float
            Amplitude threshold for condition 1 (e.g., 75% of peak).
        c2 : float
            Amplitude threshold for condition 2 (e.g., 75% of peak).
        times : np.ndarray
            Array of time points corresponding to the ERP data.
        print_output : bool, optional
            Whether to print the estimated latencies for both 
            conditions. Default is False.

        Returns
        -------
        float
            Latency difference between the two conditions (lat2 - lat1).

        Notes
        -----
        This function uses linear interpolation between adjacent time 
        points to provide precise latency estimates that are not 
        limited by the temporal resolution of the data sampling.

        The function is typically called by jackknife_contrast for 
        each jackknife iteration and the grand mean calculation.

        Examples
        --------
        >>> lat_diff = ERP.jack_latency_contrast(
        ...     wave1, wave2, 0.75*peak1, 0.75*peak2, times
        ... )
        """

        # get latency exceeding thresh with error handling
        idx_1_candidates = np.where(x1 >= c1)[0]
        if len(idx_1_candidates) == 0:
            raise ValueError(f"No time points exceed threshold {c1:.4f} "
                             f"for condition 1")
        idx_1 = idx_1_candidates[0]
        
        idx_2_candidates = np.where(x2 >= c2)[0]
        if len(idx_2_candidates) == 0:
            raise ValueError(f"No time points exceed threshold {c2:.4f} "
                             f"for condition 2")
        idx_2 = idx_2_candidates[0]
        
        # Handle edge case where threshold is exceeded at first time point
        if idx_1 == 0:
            lat_1 = times[0]
        else:
            # Linear interpolation with division by zero protection
            denominator_1 = x1[idx_1] - x1[idx_1-1]
            if abs(denominator_1) < 1e-10:  # Very small difference
                lat_1 = times[idx_1]
            else:
                lat_1 = (times[idx_1 - 1] + 
                         (times[idx_1] - times[idx_1 - 1]) * 
                         (c1 - x1[idx_1 - 1]) / denominator_1)
        
        if idx_2 == 0:
            lat_2 = times[0]
        else:
            # Linear interpolation with division by zero protection
            denominator_2 = x2[idx_2] - x2[idx_2-1]
            if abs(denominator_2) < 1e-10:  # Very small difference
                lat_2 = times[idx_2]
            else:
                lat_2 = (times[idx_2 - 1] + 
                         (times[idx_2] - times[idx_2 - 1]) * 
                         (c2 - x2[idx_2 - 1]) / denominator_2)

        d_latency = lat_2 - lat_1
        if print_output:
            print(f'Estimated onset latency waveform1 = {lat_1:.2f}'
                f' and waveform2 = {lat_2:.2f}')	

        return d_latency   

    @staticmethod
    def group_lateralized_erp(erp: list, elec_oi_c: list,
                             elec_oi_i: list, set_mean: bool = False,
                             montage: str = 'biosemi64') -> Tuple[np.array, 
                                                                  mne.Evoked]:
        """
        Create group-level lateralized difference waveforms and 
        topographies.

        This function combines individual ERP data to create lateralized 
        difference waveforms (contralateral - ipsilateral) and generates 
        topographic lateralized evoked objects where each electrode 
        represents the difference from its contralateral counterpart.

        Parameters
        ----------
        erp : list
            List of mne.Evoked objects containing individual ERP data.
        elec_oi_c : list
            Names of contralateral electrodes of interest.
        elec_oi_i : list
            Names of ipsilateral electrodes of interest.
        set_mean : bool, optional
            If True, returns averaged data across subjects. If False, 
            returns individual subject data stacked in first dimension. 
            Default is False.
        montage : str, optional
            EEG montage name for electrode layout. Currently supports 
            'biosemi64'. Default is 'biosemi64'.

        Returns
        -------
        diff : np.ndarray
            Lateralized difference waveform data. Shape depends on 
            set_mean parameter: (n_subjects, n_times) or (n_times,).
        evoked : mne.Evoked
            Evoked object with lateralized topography where each 
            electrode contains the difference from its mirror electrode 
            (e.g., O1 contains O1-O2 data).

        Notes
        -----
        The lateralized topography is created by subtracting each 
        electrode's mirror counterpart based on the standard electrode 
        naming convention (odd numbers = left, even numbers = right).

        Midline electrodes (Fz, Cz, Pz, etc.) are preserved as-is 
        since they don't have lateral counterparts.

        This method is particularly useful for analyzing lateralized 
        ERP components such as N2pc, SPCN, or lateralized readiness 
        potentials.

        Examples
        --------
        >>> diff_wave, lat_evoked = ERP.group_lateralized_erp(
        ...     erp_list, ['O1', 'PO7'], ['O2', 'PO8']
        ... )
        """

        #TODO: check whether function is still necessary
        #TODO2: add more montages

        # get mean and individual data
        evoked_X = np.stack([evoked._data for evoked in erp])
        evoked = mne.combine_evoked(erp, weights = 'equal')
        
        # calculate difference waveform
        (contra_idx, 
        ipsi_idx) = ERP.lateralized_erp_idx(erp, elec_oi_c, elec_oi_i)
        diff = evoked_X[:,contra_idx] - evoked_X[:,ipsi_idx]
        # average over electrodes
        diff = diff.mean(axis = 1)

        # set lateralized topography
        channels = evoked.ch_names

        if montage == 'biosemi64':
            lat_dict = {'Fp1':'Fp2','AF7':'AF8','AF3':'AF4','F7':'F8',
                            'F5':'F6','F3':'F4','F1':'F2','FT7':'FT8','FC5':'FC6',
                            'FC3':'FC4','FC1':'FC2','T7':'T8','C5':'C6','C3':'C4',
                            'C1':'C2','TP7':'TP8','CP5':'CP6','CP3':'CP4',
                            'CP1':'CP2','P9':'P10','P7':'P8','P5':'P6','P3':'P4',
                            'P1':'P2','PO7':'PO8','PO3':'PO4','O1':'O2',
                            'Fpz':'Fpz','AFz':'AFz','Fz':'Fz','FCz':'FCz',
                            'Cz':'Cz','CPz':'CPz','Pz':'Pz','POz':'POz','Oz':'Oz',
                            'Iz':'Iz'
                            }
        else:
            print(f'The {montage} montage is not yet supported')
            return diff

        pre_flip = np.copy(evoked._data)
        for elec_1, elec_2 in lat_dict.items():
            elec_1_data = pre_flip[channels.index(elec_1)]
            elec_2_data = pre_flip[channels.index(elec_2)]
            evoked._data[channels.index(elec_1)] = elec_1_data - elec_2_data
            evoked._data[channels.index(elec_2)] = elec_2_data - elec_1_data

        return diff, evoked                        
          
    def residual_eye(self, left_info: dict = None, right_info: dict = None,
                    ch_oi: list = ['HEOG'], cnds: dict = None,
                    midline: dict = None, window_oi: tuple = None,
                    excl_factor: dict = None, name: str = 'resid_eye'):
        """
        Calculate residual eye movement activity for lateralized designs.

        This method computes residual eye movement waveforms by averaging 
        HEOG activity for left and right stimulus presentations, accounting 
        for the expected directional differences in eye movements. This is 
        useful for validating that observed lateralized ERP effects are 
        not contaminated by systematic eye movements.

        Parameters
        ----------
        left_info : dict, optional
            Dictionary specifying criteria for left stimulus trials. 
            Format: {column_name: [labels]}. Default is None.
        right_info : dict, optional
            Dictionary specifying criteria for right stimulus trials. 
            Format: {column_name: [labels]}. Default is None.
        ch_oi : list, optional
            List of eye movement channels to analyze (typically HEOG). 
            Default is ['HEOG'].
        cnds : dict, optional
            Dictionary specifying conditions for separate analysis. 
            Default is None.
        midline : dict, optional
            Dictionary specifying midline stimulus criteria to limit 
            trial selection. Default is None.
        window_oi : tuple, optional
            Time window of interest (start, end) in seconds. If None, 
            uses full epoch time range. Default is None.
        excl_factor : dict, optional
            Dictionary specifying trial exclusion criteria. 
            Default is None.
        name : str, optional
            Base name for output file. Default is 'resid_eye'.

        Notes
        -----
        The function calculates residual eye movements as:
        residual = mean(left_trials, right_trials * -1)

        This approach accounts for the expected opposite polarity of 
        eye movements to left vs right stimuli, allowing detection of 
        systematic biases or artifacts.

        Output is saved as a pickled dictionary in the 'erp/eog' subfolder.

        Examples
        --------
        >>> erp.residual_eye(
        ...     left_info={'target_loc': [2, 8]},
        ...     right_info={'target_loc': [4, 6]},
        ...     cnds={'condition': ['standard', 'deviant']},
        ...     window_oi=(0.1, 0.5)
        ... )
        """

        #TODO: check whether function is correct

        # set file name
        erp_name= f'sub_{self.sj}_{name}.p'	
        f_name = self.folder_tracker(['erp', 'eog'],erp_name)
        # get data
        beh, epochs = self.select_erp_data(excl_factor)

        # get index of channels of interest
        ch_oi_idx = [epochs.ch_names.index(ch) for ch in ch_oi]

        # get window of interest
        if window_oi is None:
            window_oi = (epochs.tmin, epochs.tmax)
        time_idx = get_time_slice(epochs.times, window_oi[0], window_oi[1])

        # split left and right trials
        if left_info is not None:
            idx_l = self.select_lateralization_idx(beh, left_info, midline)
        if right_info is not None:
            idx_r = self.select_lateralization_idx(beh, right_info, midline)            

       # loop over all conditions
        if cnds is None:
            cnds = ['all_data']
        else:
            (cnd_header, cnds), = cnds.items()
        eye_dict = {cnd:[] for cnd in cnds}

        for cnd in cnds:
            # set erp name
            erp_name = f'sub_{self.sj}_{cnd}_{name}'	

            # slice condition trials
            if cnd == 'all_data':
                idx_c_l = idx_l
                idx_c_r = idx_r
            else:
                idx_c = np.where(beh[cnd_header] == cnd)[0]
                idx_c_l = np.intersect1d(idx_l, idx_c)
                idx_c_r = np.intersect1d(idx_r, idx_c)

            # extract data
            left_wave = epochs._data[idx_c_l][:,ch_oi_idx].mean(axis=(0,1))
            right_wave = epochs._data[idx_c_r][:,ch_oi_idx].mean(axis=(0,1))
            eye_wave = np.mean((left_wave, right_wave*-1), axis = 0)
            eye_dict[cnd] = eye_wave[time_idx]

        # save data
        pickle.dump(eye_dict, open(f_name, 'wb'))



