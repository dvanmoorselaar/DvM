"""
analyze EEG data

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
from support.support import select_electrodes,trial_exclusion,create_cnd_loop,\
                            get_time_slice

class ERP(FolderStructure):
    """
    The ERP class supports functionality for event-related potential 
    (ERP) analysis on EEG data. This class relies on the MNE-Python 
    functionality to handle evoked data. It provides methods for ....
    TODO: UPDATE DOCSTRING!!!!!

    This class inherits from FolderStructure, which provides 
    functionality for managing file paths and saving outputs. 

    Args:
        sj (int): Subject number.
        epochs (mne.Epochs): Preprocessed EEG data segmented into 
        epochs.
        df (pd.DataFrame): Behavioral data associated with 
        the EEG epochs.
        baseline (Tuple[float, float]): Time range (start, end) in 
        seconds for baseline correction.
        l_filter (Optional[float]): Low cutoff frequency for filtering. 
        Defaults to None.
        h_filter (Optional[float]): High cutoff frequency for filtering. 
        Defaults to None.
        downsample (Optional[int]): Target sampling frequency for 
        downsampling. Defaults to None.
        report (bool): Whether to generate ERP reports. 
        Defaults to False.

    Attributes:
        sj (int): Subject number.
        epochs (mne.Epochs): Preprocessed EEG data segmented into 
        epochs.
        df (pd.DataFrame): Behavioral data associated with the EEG 
        epochs.
        baseline (Tuple[float, float]): Time range (start, end) in 
        seconds for baseline correction.
        report (bool): Whether to generate ERP reports.

    Methods:

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
        """class constructor"""


        self.sj = sj
        self.l_filter = l_filter
        self.h_filter = h_filter
        self.epochs = epochs
        self.df = df
        self.baseline = baseline
        self.downsample = downsample
        self.laplacian = laplacian
        self.report = report

    def select_erp_data(self,excl_factor:dict=None,
                        topo_flip:dict=None)->Tuple[pd.DataFrame, 
                                                            mne.Epochs]:
        """
        Selects the data of interest by excluding a subset of trials and 
        optionally flipping the topography of certain trials.

        This function filters the behavioral and EEG data based on 
        specified exclusion criteria and adjusts the topography for 
        lateralized designs, ensuring that all stimuli of interest are 
        treated as if presented on the right hemifield.

        Args:
            excl_factor (Optional[dict]): A dictionary specifying 
                criteria for excluding trials from the analysis. 
                For example, `dict(target_color=['red'])` excludes all 
                trials where the target color is red. Defaults to None.
            topo_flip (Optional[dict]): A dictionary specifying criteria 
                for flipping the topography of certain trials. The key 
                should be the column name in the behavioral data, and 
                the value should be a list of labels indicating trials 
                to flip. For example, dict(cue_loc=[2, 3]) flips trials 
                where the cue location is 2 or 3. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, mne.Epochs]: 
                - `df`: A pandas DataFrame containing the filtered 
                    behavioral data.
                - `epochs`: An mne.Epochs object containing the filtered 
                    EEG data.
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
            (header, left), = topo_flip.items()
            epochs = self.flip_topography(epochs, df, left, header)
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
        Creates evoked objects using MNE functionality and optionally 
        saves them to disk.

        This function averages EEG epochs to create evoked objects, 
        applies baseline correction, and optionally crops the evoked 
        objects to a specified time window. It can also split 
        trials into fast and slow groups based on median reaction time 
        (RT) and generate separate evoked objects for each group.

        Args:
            epochs (mne.Epochs): Preprocessed EEG data segmented into 
                epochs.
            df (pd.DataFrame): Behavioral data associated with the 
                EEG epochs.
            idx (Optional[np.array]): Indices used for trial averaging. 
                If None, all trials are used. Defaults to None.
            time_oi (Optional[tuple]): Time window of interest 
                (start, end) in seconds. If specified, evoked objects 
                are cropped to this window. Defaults to None.
            erp_name (str): Filename to save the evoked object. 
                Defaults to 'all'.
            RT_split (bool): If True, data is analyzed separately for 
                fast and slow trials based on median RT. Requires that 
                the DataFrame contains a column RT. Defaults to False.
            save (bool): If True, the evoked object is saved to disk. 
                If False, the evoked object is returned instead. 
                Defaults to True.

        Returns:
            mne.Evoked: The evoked object created from the averaged EEG 
            epochs.
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
                                        f'{erp_name}-ave.fif',
                                        overwrite=True))
            
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
                    evoked_split = evoked_splot.crop(tmin = time_oi[0],
                                                     tmax = time_oi[1]) 															
                evoked_split.save(self.folder_tracker(['erp', 'evoked'],
                                                f'{erp_name}_{rt}-ave.fif'))
        return evoked
                
    def generate_erp_report(self,evokeds:dict, report_name: str):


        report_name = self.folder_tracker(['erp', 'report'],
                                        f'{report_name}.h5')
        
        report = mne.Report(title='Single subject evoked overview')
        for cnd in evokeds.keys():
            if self.laplacian:
                #TODO: remove after updating mne
                pass
            else:
                report.add_evokeds(evokeds=evokeds[cnd],titles=cnd)

        report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html', overwrite=True)
                        
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
        Creates event-related potentials (ERPs) for specified 
        conditions.

        This function selects trials of interest based on provided 
        position labels and generates condition-specific ERPs. It 
        optionally applies exclusion criteria, flips the topography for 
        lateralized designs (if specified via topo_flip),and crops the 
        time window of interest. ERPs can also be split into fast and 
        slow trials based on median reaction time (RT).

        Args:
            pos_labels (dict): A dictionary specifying the position 
                labels for stimuli of interest.The key should be the 
                column in the corresponding behavioral DataFrame that 
                contains position labels, and the value should be a list 
                of position labels to be used in the analysis.
                For example, `dict(target_loc=[2,6])` selects trials 
                where the target is presented on positions 2 and 6.
            cnds (Optional[dict]): Dictionary specifying conditions for 
                ERP creation. The key should be the column name in the 
                behavioral data, and the value should be a list of 
                condition labels. 
                For example, `dict(target_cnd=['red','green'])` 
                creates seperate ERPs for trials where the target was
                green and red.  Defaults to None, which creates ERPs 
                collapsed across all data.
            midline (Optional[dict]): Dictionary specifying trials where 
                another stimulus of interest is presented on the 
                vertical midline. The key should be the column name, 
                and the value should be a list of labels (see pos_labels 
                and cnds for example logic). Defaults to None.
            topo_flip (Optional[dict]): Dictionary specifying criteria 
                for flipping the topography of certain trials. 
                The key should be the column name in the behavioral 
                data, and the value should be a list of labels 
                indicating trials to flip. (see pos_labels 
                and cnds for example logic). Defaults to None.
            time_oi (Optional[tuple]): Time window of interest 
                (start, end) in seconds. If specified, 
                evoked objects are cropped to this window. 
                Defaults to None.
            excl_factor (Optional[dict]): Dictionary specifying criteria 
                for excluding trials from the analysis. 
                For example, `dict(dist_color=['red'])` excludes all 
                trials where the distractor color is red. 
                Defaults to None.
            RT_split (bool): If True, data is analyzed separately 
                for fast and slow trials based on median RT. Requires
                that the DataFrame contains a column 'RT'.
                Defaults to False.
            name (str): Name used for saving the ERP files. 
            Defaults to 'main'.

        Returns:
            None: The function saves the generated ERPs to disk.
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
            erp_name = f'sj_{self.sj}_{cnd}_{name}'	

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
            self.generate_erp_report(evokeds,f'sj_{self.sj}_{name}'	)

    @staticmethod
    def flip_topography(
        epochs: mne.Epochs, 
        df: pd.DataFrame, 
        left: list, 
        header: str, 
        flip_dict: Optional[dict] = None, 
        heog: str = 'HEOG'
    ) -> mne.Epochs:
        """
        Flips the topography of trials where the stimuli of interest 
        were presented on the left.

        This function adjusts the EEG data so that trials where the 
        stimuli of interest were presented on the left 
        (i.e., right hemifield) are flipped as if they were presented on 
        the right (i.e., left hemifield). After running this function, 
        all stimuli are treated as if presented on the right hemifield 
        (contralateral relative to the stimulus of interest).

        By default, flipping is performed based on the Biosemi 
        64 spatial layout. The function also supports flipping 
        the horizontal electrooculogram (HEOG) channel if specified.

        Args:
            epochs (mne.Epochs): Preprocessed EEG epochs object.
            df (pd.DataFrame): Behavioral DataFrame containing 
                trial-specific parameters.
            left (list): Position labels of trials where the topography 
                will be flipped to the other hemifield.
            header (str): Column name in the behavioral DataFrame that 
            contains position labels to be flipped.
            flip_dict (Optional[dict]): Dictionary specifying electrode 
                pairs for flipping. The key-value pairs represent 
                electrodes to swap (e.g., `{'Fp1': 'Fp2'}`). 
                Defaults to None, in which case the flip dict is 
                generated based on electrode layout in epochs.
            heog (str, optional): Name of the HEOG channel. If this 
                channel is present in the epochs object, the sign of all 
                left trials will be flipped. Defaults to 'HEOG'.

        Returns:
            mne.Epochs: The epochs object with flipped topography for 
                specified trials.
        """

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
        Selects trial indices based on lateralized position labels 
        and optionally limits selection to trials where another stimulus 
        is presented on the vertical midline.

        This function identifies trials where the stimuli of interest 
        are presented left or right of the vertical midline based on 
        position labels. If specified, trial selection can be further 
        restricted to those where another stimulus is concurrently 
        presented on the midline. It can also be used to select 
        non-lateralized trials based on the provided position labels.

        Args:
            df (pd.DataFrame): Behavioral DataFrame containing 
                trial-specific parameters linked to EEG epochs.
            pos_labels (dict): Dictionary specifying the column with 
                position labels in the `df` DataFrame and the values to 
                include in the analysis. 
                For example, `dict(target_loc=[2, 6])` selects trials
                where the target is presented at positions 2 or 6.
            midline (Optional[dict]): Dictionary specifying the column 
                and values for trials where another stimulus is 
                presented on the vertical midline. 
                For example, `dict(dist_loc=[0, 2])` limits selection to 
                trials where the distractor is presented at positions 
                0 or 2. Defaults to None.

        Returns:
            np.array: Array of selected trial indices.
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
        Determines an ERP time window using peak detection, based either 
        on grand averaged data or condition-specific data.

        This function identifies the peak within the specified 
        electrodes and time window and returns the ERP window centered 
        on the detected peak. The window can be based on either the 
        grand average waveform across all conditions, condition-specific 
        waveforms. Window selection can be based on a set of electrodes
        of interest (elec_oi), which can be a single list of electrodes
        or a lateralized difference waveform 
        (e.g., contralateral - ipsilateral; see Args).

        Args:
            erps (Union[dict, list]): Either a dictionary, as generated 
                by e.g., lateralized_erp, where keys are condition names 
                and values are lists of evoked objects (mne.Evoked), 
                or a list of evoked objects for grand averaging.
            elec_oi (list): Electrodes of interest. If the data of 
                interestis a difference waveform (e.g., contralateral 
                vs. ipsilateral), specify a list of lists, where the 
                first list contains contralateral electrodes and the 
                second list contains ipsilateral electrodes 
                For example elec_oi = `[['O1'],['O2']]` selects the 
                window of interest based on the difference between 
                O1 and O2.
            method (str, optional): Specifies whether the window is 
                based on the grand averaged data (`'cnd_avg'`) or 
                condition-specific data (`'cnd_spc'`).
                Defaults to `'cnd_avg'`.
            window_oi (Optional[tuple]): Time window of interest 
                (start, end) in seconds. If specified, peak detection is 
                restricted to this window. Defaults to None, 
                which uses the full time range.
            polarity (str, optional): Specifies whether the peak is 
                positive (`'pos'`) or negative (`'neg'`). 
                Defaults to `'pos'`.
            window_size (float, optional): Size of the ERP window in 
                seconds, centered on the detected peak. 
                Defaults to 0.05.

        Returns:
            Union[list, dict]: 
                - If `method='cnd_avg'`, returns a list representing 
                the selected time window [start, end].
                - If `method='cnd_spc'`, returns a dictionary where keys 
                are condition names and values are lists representing 
                condition-specific time windows.
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
        Extracts EEG channel names and sample times from ERP data.

        This function retrieves the channel names and time points 
        from the provided ERP data, which can be either a dictionary 
        (with condition names as keys and lists of evoked objects as 
        values) or a list of evoked objects.

        Args:
            erps (Union[dict, list]): ERP data to extract 
                parameters from. 
                If a dictionary, the keys represent condition names and 
                the values are lists of evoked objects (mne.Evoked). 
                If a list, it contains evoked objects directly.

        Returns:
            Tuple[list, np.array]: 
                - `channels` (list): List of EEG channel names.
                - `times` (np.array): Array of sample times in the 
                    evoked data.
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
        Exports ERP metrics (e.g., mean amplitude, area under the curve) 
        to a CSV file.

        This function calculates ERP metrics for specified conditions 
        and electrodes and saves the results to a CSV file. The file is 
        stored in the subfolder `erp/stats` in the main project folder. 
        It supports lateralized difference waveforms and 
        condition-specific time windows.

        Args:
            erps (Union[dict, list]): ERP data to process. Can be a list 
                of evoked objects (mne.Evoked) or a dictionary where 
                keys are condition names and values are lists of evoked 
                objects.
            window_oi (Union[list, dict]): Time window of interest for 
                calculating metrics. If a list, the same window is 
                applied to all conditions. If a dictionary, 
                condition-specific windows are used 
                (keys correspond to condition names).
            elec_oi (list): Electrodes of interest. For lateralized 
                difference waveforms, specify a list of lists, where the 
                first list contains contralateral electrodes 
                and the second list contains ipsilateral electrodes.
            cnds (list, optional): List of conditions to include in the 
                export. If None, all conditions are processed. 
                Defaults to None.
            method (str, optional): Metric calculation method 
                (e.g., `'mean_amp'`, `'auc'`). Defaults to `'mean_amp'`.
            name (str, optional): Name of the output CSV file. 
                Defaults to `'main'`.

        Returns:
            None: The function saves the calculated metrics to a 
                CSV file.
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
        Combines all individual data at the group level.

        Args:
            erp (list): List of evoked items (mne.Evoked).
            elec_oi (list): Electrodes of interest. If 'all', all 
                electrodes are included.

        Returns:
            Tuple[np.array, mne.Evoked]: 
                - `evoked_X`: Stacked individual ERP data for the 
                    specified electrodes.
                - `evoked`: Group-level evoked object created by 
                    averaging individual evoked items.
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
        Calculates metrics for ERP data, such as mean amplitude 
        or area under the curve (AUC).

        This function computes specific metrics for ERP data based on 
        the provided method. Currently supported methods include:
            - `'mean_amp'`: Calculates the mean amplitude of 
                the ERP data.
            - `'auc_pos'`: Calculates the area under the curve (AUC) 
                for positive values only.
            - `'auc_neg'`: Calculates the AUC for negative values only.
            - `'auc'`: Calculates the AUC for all values.
            - `'onset_latency'`: Calculates the onset latency of the 
                ERP component based on a threshold (e.g., 50% of the 
                peak amplitude).

        Args:
            X (np.ndarray): ERP data (trials x timepoints).
            times (np.ndarray): Array of time points corresponding to 
                the ERP data.
            method (str): Specifies the metric to calculate. 
                Supported values are `'mean_amp'`, `'auc'`, `'auc_pos'`, 
                and `'auc_neg'`.
            threshold (float, optional): Threshold for onset latency 
                calculation. If `method` is `'onset_latency'`, this 
                specifies the percentage of the peak amplitude 
                (e.g., 0.5 for 50%). Defaults to 0.5.
            polarity (str, optional): Polarity of the ERP component for 
                onset latency calculation. Use `'pos'` for positive 
                components (e.g., Pd) and `'neg'` for negative 
                components (e.g., N2Pc). Defaults to `'pos'`.

        Returns:
            np.ndarray: A NumPy array containing the calculated metrics 
                for each trial.
        """

        if method == 'mean_amp':
            output = X.mean(axis = -1)
        elif 'auc' in method:
            if 'pos' in method:
                X[X<0] = 0
            elif 'neg' in method:
                X[X>0] = 0
            output = np.array([auc(times, x) for x in X])
        elif method == 'onset_latency':
            output = []
            for x in X:
                if polarity == 'neg':
                    x *= -1  # invert signal for negative polarity
                thresh_value = x[np.argmax(x)] * threshold
                # find first time point exceeding the threshold
                idx = np.where(x >= thresh_value)[0]
                if idx.size > 0:
                    output.append(times[idx[0]])
                else:
                    output.append(np.nan)  # no onset found
            output = np.array(output)     

        return output

    @staticmethod
    def lateralized_erp_idx(erp:list,elec_oi_c:list,
                            elec_oi_i:list)->Tuple[np.array, np.array]:
        """
        get indices of contralateral and ipsilateral electrodes

        Args:
            erps (list): list with evoked items (mne)
            elec_oi_c (list): contralateral electrodes
            elec_oi_i (list): ipsilateral electrodes

        Returns:
            contra_idx (array): indices corresponding to contralateral 
            electrodes
            ipsi_idx (array): indices corresponding to ipsilateral 
            electrodes
        """
        
        # extract channels from erps
        channels = erp[0].ch_names

        # get indices
        contra_idx = np.array([channels.index(ch) for ch in elec_oi_c])
        ipsi_idx = np.array([channels.index(ch) for ch in elec_oi_i])

        return contra_idx, ipsi_idx
    
    @staticmethod
    def select_waveform(erps:list, elec_oi:list):
        """
        Extracts and averages ERP waveforms for specified electrodes 
        from a list of evoked objects.

        This function selects the data from the provided electrodes of 
        interest (`elec_oi`) across all evoked objects in `erps`. 
        If `elec_oi` is a list of electrode names (strings),
        it extracts and averages the data for those electrodes. 
        If `elec_oi` is a list of two lists 
        (e.g., for lateralized analysis), it computes the difference
        between the contralateral and
        ipsilateral electrodes for each evoked object, 
        then averages across electrodes.

        Args:
            erps (list): List of mne.Evoked objects containing ERP data.
            elec_oi (list): Electrodes of interest. Can be a list of 
                electrode names (strings),or a list of two lists for 
                lateralized analysis (e.g., [['O1'], ['O2']]).

        Returns:
            np.ndarray: Array of averaged ERP waveforms for each evoked 
            object,shape (n_evoked, n_times).
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
        times: np.array = None, 
        percent_amp: int = 75, 
        polarity: str = 'pos', 
        phase: str = 'onset'
    ):
        """
        Compares the latencies of ERP components between two conditions
        using the jackknife method.

        This function calculates the latency difference between two ERP 
        conditions or waveforms based on a specified percentage of the 
        peak amplitude. It supports both onset and offset latency 
        comparisons and handles positive and negative polarities.

        Args:
            erps (Union[dict, list]): ERP data to compare. Can be a 
                dictionary where keys are condition names and values are 
                lists of evoked objects, or a list of two ERP waveforms 
                to compare.
            elec_oi (list, optional): Electrodes of interest for 
                extracting the ERP waveforms. Defaults to None.
            window_oi (tuple, optional): Time window of interest 
                (start, end) in seconds. If None, the full time range 
                is used. Defaults to None.
            times (np.array, optional): Array of time points 
                corresponding to the ERP data. Defaults to None.
            percent_amp (int, optional): Percentage of the peak 
                amplitude to use for latency calculation. 
                Defaults to 75.
            polarity (str, optional): Polarity of the ERP component 
                ('pos' for positive, 'neg' for negative). 
                Defaults to 'pos'.
            phase (str, optional): Specifies whether to calculate onset 
                or offset latency. Supported values are `'onset'` and 
                `'offset'`. Defaults to `'onset'`.

        Returns:
            Union[tuple, dict]: 
                - If comparing a single pair of conditions, returns 
                    a tuple containing the latency difference and 
                    the t-value.
                - If comparing multiple pairs, returns a dictionary 
                    where keys are condition pairs and values are tuples 
                    of latency differences and t-values.
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
                #TODO: check flipping of time 
                if p == 0:
                    times_oi = np.fliplr(times_oi) 

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
        x1: np.array, 
        x2: np.array, 
        times: np.array, 
        percent_amp: int
    ) -> Tuple[float, float]:
        """
        Performs a jackknife-based contrast to estimate the latency 
        difference between two ERP waveforms.

        This function calculates the latency difference between two ERP 
        waveforms using the jackknife method. It computes the grand mean 
        latency difference and estimates the variability of the 
        difference using jackknife resampling.

        Args:
            x1 (np.array): ERP waveform for condition 1 
                (trials x timepoints).
            x2 (np.array): ERP waveform for condition 2 
                (trials x timepoints).
            times (np.array): Array of time points corresponding to the
                ERP data.
            percent_amp (int): Percentage of the peak amplitude to use 
                for latency calculation.

        Returns:
            Tuple[float, float]: 
                - `d_latency`: The latency difference between the 
                    two conditions.
                - `t_value`: The t-value for the latency difference 
                    based on jackknife resampling.
        """ 

        # set params
        nr_obs = x1.shape[0]

        # step 1: get grand mean latency difference
        x1_ = x1.mean(axis = 0) 
        x2_ = x2.mean(axis = 0) 
        c1 = max(x1_) * percent_amp/100.0
        c2 = max(x2_) * percent_amp/100.0 

        d_latency = ERP.jack_latency_contrast(x1_,x2_,c1,c2,times,True)

        # step 2: jackknifing procedure ()
        D = []
        idx = np.arange(nr_obs)
        for i in range(nr_obs):
            x1_ = x1[i != idx,:].mean(axis = 0)
            x2_ = x2[i != idx,:].mean(axis = 0)
            c1 = max(x1_) * percent_amp/100.0
            c2 = max(x2_) * percent_amp/100.0 
            D.append(ERP.jack_latency_contrast(x1_,x2_,c1,c2,times))

        # compute the jackknife estimate 
        Sd = np.sqrt((nr_obs - 1.0)/ nr_obs \
            * np.sum([(d - np.mean(D))**2 for d in np.array(D)]))	

        t_value = d_latency/ Sd 

        return d_latency, t_value

    @staticmethod
    def jack_latency_contrast(
        x1: np.array, 
        x2: np.array, 
        c1: float, 
        c2: float, 
        times: np.array, 
        print_output: bool = False
    ) -> float:
        """
        Calculates the latency difference between two ERP waveforms at a 
        specified threshold.

        This function determines the latency at which two ERP waveforms 
        reach a specified threshold (e.g., a percentage of the peak 
        amplitude) and calculates the difference between these 
        latencies.

        Args:
            x1 (np.array): ERP waveform for condition 1 (timepoints).
            x2 (np.array): ERP waveform for condition 2 (timepoints).
            c1 (float): Threshold value for condition 1 
                (e.g., 75% of the peak amplitude).
            c2 (float): Threshold value for condition 2 
                (e.g., 75% of the peak amplitude).
            times (np.array): Array of time points corresponding to the 
                ERP data.
            print_output (bool, optional): If True, prints the estimated 
                latencies for both conditions. Defaults to False.

        Returns:
            float: The latency difference between the two conditions.
        """

        # get latency exceeding thresh 
        idx_1 = np.where(x1 >= c1)[0][0]
        lat_1 = times[idx_1 - 1] + (times[idx_1] - times[idx_1 - 1]) * \
                    (c1 - x1[idx_1 - 1])/(x1[idx_1] - x1[idx_1-1])
        idx_2 = np.where(x2 >= c2)[0][0]
        lat_2 = times[idx_2 - 1] + (times[idx_2] - times[idx_2 - 1]) * \
                (c2 - x2[idx_2 - 1])/(x2[idx_2] - x2[idx_2-1])

        d_latency = lat_2 - lat_1
        if print_output:
            print(f'Estimated onset latency waveform1 = {lat_1:.2f}'
                f' and waveform2 = {lat_2:.2f}')	

        return d_latency   

    def condition_erp(self, cnds:dict=None,time_oi:tuple=None,
                    excl_factor:dict=None,RT_split:bool=False, 
                    name:str='main'):

        # get data
        beh, epochs = self.select_erp_data(excl_factor)
        beh.reset_index(inplace = True, drop = True)

        # loop over all conditions
        if cnds is None:
            cnds = ['all_data']
        else:
            cnds = create_cnd_loop(cnds)

        for cnd in cnds:
            # set erp name
            if type(cnd) == str:
                erp_name = f'sj_{self.sj}_{cnd}_{name}'	
            else:
                erp_name = f'sj_{self.sj}_{cnd[1]}_{name}'

            # slice condition trials
            if cnd == 'all_data':
                idx_c = np.arange(beh.shape[0])
            else:
                idx_c = beh.query(cnd[0]).index.values

            if idx_c.size == 0:
                print('no data found for {}'.format(cnd))
                continue

            self.create_erps(epochs, beh, idx_c, time_oi, erp_name, RT_split)

    def residual_eye(self,left_info:dict=None,right_info:dict=None,
                    ch_oi:list=['HEOG'],cnds:dict=None,
                    midline:dict=None,window_oi:tuple=None,
                    excl_factor:dict=None,name:str='resid_eye'):

        # set file name
        erp_name= f'sj_{self.sj}_{name}.p'	
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
            erp_name = f'sj_{self.sj}_{cnd}_{name}'	

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

    @staticmethod
    def group_lateralized_erp(erp:list,elec_oi_c:list,
                            elec_oi_i:list,set_mean:bool=False,
                            montage:str='biosemi64')->Tuple[np.array,
                                                    mne.Evoked]:
        """
        Combines all individual data at the group level by creating a 
        difference waveform (contralateral - ipsilateral). Also returns a 
        topographic lateralized evoked object by subtracting the 
        lateralized counterpart from each electrode 
        (e.g., data at PO7 reflects PO7 - PO8).

        Args:
            erp (list): list with evoked items (mne)
            elec_oi_c (list): Contralateral electrodes of interest
            elec_oi_i (list): Ipsilateral electrodes of interest
            set_mean (bool, optional): If True, returns array with averaged 
            data. Otherwise data from individual datasets is stacked in the
            first dimension. Defaults to False.

        Returns:
            diff(np.array): Difference waveform (n X nr_timepoints)
            evoked(mne.Evoked):
        """

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
          
if __name__ == '__main__':
    pass





