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

    def __init__(self, sj, epochs, beh, header, baseline, 
                l_filter = None, h_filter = None, downsample:int=None,
                report=False):

        # if filters are specified, filter data before trial averaging  
        if l_filter is not None or h_filter is not None:
            epochs.filter(l_freq = l_filter, h_freq = h_filter)
        self.sj = sj
        self.epochs = epochs
        self.beh = beh
        self.header = header
        self.baseline = baseline
        
        if downsample is not None:
            if downsample < int(epochs.info['sfreq']):
                print('downsampling data')
                self.epochs.resample(downsample)
        self.report = report

    def report_erps(self, evoked: mne.Evoked, erp_name: str):

        # set report and condition name
        name_info = erp_name.split('_')
        report_name = name_info[-1]
        report_name = self.folder_tracker(['erp', self.header],
                                        f'report_{report_name}.h5')
        cnd_name = '_'.join(map(str, name_info[:-1]))

        # check whether report exists
        if os.path.isfile(report_name):
            with mne.open_report(report_name) as report:
                # if section exists delete it first
                report.remove(title=cnd_name)
                report.add_evokeds(evokeds=evoked,titles=cnd_name,	
                                   n_time_points=21)
            report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html', 
                        overwrite = True)
        else:
            report = mne.Report(title='Single subject evoked overview')
            report.add_evokeds(evokeds=evoked,titles=cnd_name,	
                                 n_time_points=21)
            report.save(report_name)
            report.save(report_name.rsplit( ".", 1 )[ 0 ]+ '.html')

    def select_erp_data(self,excl_factor:dict=None,
                        topo_flip:dict=None)->Tuple[pd.DataFrame, 
                                                            mne.Epochs]:
        """
        Selects the data of interest by excluding a subset of trials

        Args:
            excl_factor (dict, optional): If specified, a subset of trials that 
            matches the specified criteria is excluded from further analysis
            (e.g., dict(target_color = ['red']) will exclude all trials 
            where the target color is red). Defaults to None.
            topo_flip (dict, optional): If specified a subset of trials is 
            flipped such that it is as if all stimuli of interest are 
            presented right (see lateralized_erp). 

        Returns:
            beh: pandas Dataframe
            epochs: epochs data of interest
        """

        beh = self.beh.copy()
        epochs = self.epochs.copy()

        # if not already done reset index (to properly align beh and epochs)
        beh.reset_index(inplace = True, drop = True)

        # if specified remove trials matching specified criteria
        if excl_factor is not None:
            beh, epochs = trial_exclusion(beh, epochs, excl_factor)

        # check whether left stimuli should be 
        # artificially transferred to left hemifield
        if topo_flip is not None:
            (header, left), = topo_flip.items()
            epochs = self.flip_topography(epochs, beh,  left,  header)
        else:
            print('No topography info specified. In case of a lateralized '
                'design. It is assumed as if all stimuli of interest are '
                'presented right (i.e., left  hemifield')

        return beh, epochs

    def create_erps(self,epochs:mne.Epochs,beh:pd.DataFrame,idx:np.array=None, 
                    time_oi:tuple=None,erp_name:str = 'all',
                    RT_split:bool=False,save:bool=True):
        """
        Creates evoked objects using mne functionality

        Args:
            epochs (mne.Epochs): mne epochs object
            beh (pd.DataFrame): behavioral parameters linked to behavior
            idx (np.array, optional): indices used for trial averaging. 
            time_oi (tuple, optional): If specified, evoked objects are cropped
            to this time window. Defaults to None.
            erp_name (str, optional): filename to save evoked object. 
            Defaults to 'all'.
            RT_split (bool, optional): If True data will also be analyzed 
            seperately for fast and slow trials. Defaults to False.
            save (bool, optional): If False, rather than saving the 
            evoked instance is returned 
        """

        beh = beh.iloc[idx].copy()
        epochs = epochs[idx]

        # create evoked objects using mne functionality and save file
        evoked = epochs.average().apply_baseline(baseline = self.baseline)
        # if specified select time window of interest
        if time_oi is not None:
            evoked = evoked.crop(tmin = time_oi[0],tmax = time_oi[1])
        if save: 
            evoked.save(self.folder_tracker(['erp', self.header],
                                        f'{erp_name}-ave.fif',
                                        overwrite=True))
        else:
            return evoked

        # update report
        if self.report:
            self.report_erps(evoked, erp_name)

        # split trials in fast and slow trials based on median RT
        if RT_split:
            median_rt = np.median(beh.RT)
            beh.loc[beh.RT < median_rt, 'RT_split'] = 'fast'
            beh.loc[beh.RT > median_rt, 'RT_split'] = 'slow'
            for rt in ['fast', 'slow']:
                mask = beh['RT_split'] == rt
                # create evoked objects using mne functionality and save file
                evoked = epochs[mask].average().apply_baseline(baseline = 
                                                            self.baseline)
                # if specified select time window of interest
                if time_oi is not None:
                    evoked = evoked.crop(tmin = time_oi[0],tmax = time_oi[1]) 															
                evoked.save(self.folder_tracker(['erp', self.header],
                                                f'{erp_name}_{rt}-ave.fif'))

    @staticmethod
    def flip_topography(epochs:mne.Epochs,beh: pd.DataFrame,left:list, 
                        header:str,flip_dict:dict=None,
                        heog:str='HEOG') -> mne.Epochs:
        """
        Flips the topography of trials where the stimuli of interest was 
        presented on the left (i.e. right hemifield). After running this 
        function it is as if all stimuli are presented right 
        (i.e. the left hemifield is contralateral relative to the 
        stimulus of interest).

        By default flipping is done on the basis of a Biosemi 
        64 spatial layout 

        Args:
            epochs (mne.Epochs): preprocessed epochs object
            beh (pd.DataFrame): linked behavioral parameters
            left (list): position labels of trials where the 
            topography will be flipped to the other hemifield
            header (str): column in behavior that contains position 
            labels to be flipped
            flip_dict(dict, optional): Dictionary used to flip 
            topography. Data corresponding to all key value pairs will 
            be flipped (e.g., flip_dict = dict(FP1 = 'Fp2') will copy 
            the data from Fp1 into Fp2 and vice versa)
            heog (str, optional): Channel should represent the 
            diff score between right and left heog. if this channel name 
            is present in epochs object, sign of all left trials will be 
            flipped
            
        Returns:
            epochs: epochs with flipped topography for specified trials
        """

        picks = mne.pick_types(epochs.info, eeg=True, csd = True)   
        # dictionary to flip topographic layout
        if flip_dict is None:
            flip_dict = {'Fp1':'Fp2','AF7':'AF8','AF3':'AF4','F7':'F8',
                        'F5':'F6','F3':'F4','F1':'F2','FT7':'FT8','FC5':'FC6',
                        'FC3':'FC4','FC1':'FC2','T7':'T8','C5':'C6','C3':'C4',
                        'C1':'C2','TP7':'TP8','CP5':'CP6','CP3':'CP4',
                        'CP1':'CP2','P9':'P10','P7':'P8','P5':'P6','P3':'P4',
                        'P1':'P2','PO7':'PO8','PO3':'PO4','O1':'O2'}

        idx_l = np.hstack([np.where(beh[header] == l)[0] for l in left])

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
    def select_lateralization_idx(beh:pd.DataFrame,pos_labels:dict, 
                                  midline:dict)->np.array:
        """
        Based on position labels selects only those trial indices where 
        the stimuli of interest are presented left or right from the 
        vertical midline. If specified trial selection can be limited
        to those trials where another stimulus of interest is presented 
        on the vertical midline. 

        Function can also be used to select non-lateralized trials as 
        the key,value pair of pos_labels determins which trials 
        are ultimately selected 

        Args:
            beh (pd.DataFrame): DataFrame with behavioral parameters 
            per linked epoch
            pos_labels (dict): Dictionary key specifies the column with 
            position labels in the beh DataFrame. Values should be a 
            list of all labels that are included in the analysis 
            (e.g., dict(target_loc = [2,6])) 
            midline (dict): If specified, selected trials are limited to 
            trials where another stimuli of interest is concurrently 
            presented on the vertical midline 
            (e.g., dict(dist_loc = [0,2])). Key again specifies the 
            column of interest. Multiple keys can be specified

        Returns:
            idx (np.array): selected trial indices
        """

        # select all lateralized trials	
        (header, labels), = pos_labels.items()
        idx = np.hstack([np.where(beh[header] == l)[0] for l in labels])

        # limit to midline trials
        if  midline is not  None:
            idx_m = []
            for key in midline.keys():
                idx_m.append(np.hstack([np.where(beh[key] == m)[0] 
                                        for m in midline[key]]))
            idx_m = np.hstack(idx_m)
            idx = np.intersect1d(idx, idx_m)

        return idx

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


    def create_diff_wave(self,epochs:mne.Epochs,idx:np.array,
                        contra_elec:list,ipsi_elec:list)->np.array:

        # get contra and ipsi indices
        idx_ipsi = [epochs.ch_names.index(ipsi) for ipsi in ipsi_elec]
        idx_contra = [epochs.ch_names.index(contra) for contra in contra_elec]

        # get waveforms
        ipsi = epochs._data[idx, idx_ipsi].mean(axis = (0,1))
        contra = epochs._data[idx, idx_contra].mean(axis = (0,1))

    def lateralized_erp(self,pos_labels:np.array,cnds:dict=None,
                        midline:dict=None,topo_flip:dict=None,
                        time_oi:tuple=None,excl_factor:dict=None,
                        RT_split:bool=False,name:str='main'):

        # get data
        beh, epochs = self.select_erp_data(excl_factor,topo_flip)
    
        # select trials of interest (i.e., lateralized stimuli)
        idx = self.select_lateralization_idx(beh,pos_labels,midline)

        # loop over all conditions
        if cnds is None:
            cnds = ['all_data']
        else:
            (cnd_header, cnds), = cnds.items()

        for cnd in cnds:
            # set erp name
            erp_name = f'sj_{self.sj}_{cnd}_{name}'	

            # slice condition trials
            if cnd == 'all_data':
                idx_c = idx
            else:
                idx_c = np.where(beh[cnd_header] == cnd)[0]
                idx_c = np.intersect1d(idx, idx_c)

            if idx_c.size == 0:
                print('no data found for {}'.format(cnd))
                continue

            self.create_erps(epochs, beh, idx_c, time_oi, erp_name, RT_split)

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
    def group_erp(erp:list,elec_oi:list='all',
                 set_mean:bool=False)->Tuple[np.array,mne.Evoked]:
        """
        Combines all individual data at the group level

        Args:
            erp (list): list with evoked items (mne)
            elec_oi (list): electrodes of interest
            set_mean (bool, optional): If True, returns array with averaged 
            data. Otherwise data from individual datasets is stacked in the
            first dimension. Defaults to False.

        Returns:
            Tuple[np.array,mne.Evoked]: _description_
        """

        # get mean and individual data
        evoked = mne.combine_evoked(erp, weights = 'equal')
        channels = evoked.ch_names
        if elec_oi == 'all':
            elec_oi = channels
        elec_oi_idx = np.array([channels.index(elec) for elec in elec_oi])
        evoked_X = np.stack([e._data[elec_oi_idx] for e in erp])

        evoked_X = evoked_X.mean(axis = 1)

        if set_mean:
            evoked_X = np.mean(evoked_X, axis = 0)
        
        return evoked_X, evoked

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

    @staticmethod
    def measure_erp(X, times, method):

        if method == 'mean_amp':
            output = X.mean(axis = -1)
        if 'auc' in method:
            if 'pos' in method:
                X[X<0] = 0
            elif 'neg' in method:
                X[X>0] = 0
            output = [auc(times, x) for x in X]

        return output

    @staticmethod
    def erp_to_csv(erps:Union[dict,list],window_oi:Union[tuple,dict],
                  elec_oi:list,cnds:list=None,method:str='mean_amp',
                  name:str='main'):
        ## TODO: add different methods (peak, onset latency)
        """
        Outputs ERP metrics (e.g., mean activity) to a csv file. The csv file
        is stored in the subfolder erp/stats in the main project folder.

        Note that this function also allows one to output lateralized 
        difference waveforms (see elec_oi)

        Args:
            erps ([dict,list]): Either a list with evoked items (mne) or a 
            dictionary where key, value pairs are condition names and a list
            with conditin specific evoked data, respectively
            window_oi ([tuple,dict]): time window used to calculate the 
            dependent measure of interest (see methods)
            elec_oi (list): electrodes of interest. In case, the data of 
            interest is a difference waveform (i.e., contra - ipsi), specify a 
            list of lists, where the first list contains contralateral 
            electrodes and the second list ipsilateral electrodes. 
            cnds (list, optional): If specified allows to limit export
            to a subset of conditions as specified in the erps dictionary. 
            Defaults to None.
            method (str, optional): 
            name (str, optional): Name ofoutput file. Defaults to 'main'.
        """

        # initialize output list and set parameters
        X, headers = [], []
        if cnds is None and type(erps) == dict:
            cnds = list(erps.keys())
        if type(erps) == list:
            erps = {'data':erps}
            cnds = ['data']
        
        channels, times = ERP.get_erp_params(erps)

        if type(window_oi) == tuple:
            idx = get_time_slice(times, window_oi[0], window_oi[1])

        # extract condition specific data
        for cnd in cnds:
            if type(window_oi) == dict:
                idx = get_time_slice(times,window_oi[cnd][0],window_oi[cnd][1])

            # check whether output needs to be lateralized
            if isinstance(elec_oi[0], str):
                evoked_X, _ = ERP.group_erp(erps[cnd],elec_oi)
                y = ERP.measure_erp(evoked_X[:,idx],times[idx],method)
                X.append(y)
                #X.append(evoked_X[:,idx].mean(axis = 1))
                headers.append(cnd)
            else:
                d_wave = []
                for h, hemi in enumerate(['contra','ipsi']):
                    evoked_X, _ = ERP.group_erp(erps[cnd],elec_oi[h])
                    d_wave.append(evoked_X)
                    y = ERP.measure_erp(evoked_X[:,idx],times[idx],method)
                    X.append(y)
                    #X.append(evoked_X[:,idx].mean(axis = 1))
                    headers.append(f'{cnd}_{hemi}')

                # add contra vs hemi difference
                d_wave = d_wave[0] - d_wave[1]
                y = ERP.measure_erp(d_wave[:,idx],times[idx],method)
                X.append(y)
                #X.append(X[-2] - X[-1])
                headers.append(f'{cnd}_diff')

        # save data
        np.savetxt(ERP.folder_tracker(['erp','stats'], 
               fname = f'{name}.csv'),np.stack(X).T, 
               delimiter = ",",header = ",".join(headers),comments='')

    @staticmethod
    def get_erp_params(erps:Union[dict,list])->Tuple[list,np.array]:
        """
        Extracts relevant parameters (i.e., times and channels)
        out of condition dict with evoked data or list with evoked data

        Args:
            erps (Union[dict,list]): _description_

        Returns:
            channels (list): eeg channel names
            times (np.array): sample times in evoked data
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
    def find_erp_window(erps:Union[dict,list],elec_oi:list,
                        method:str='cnd_avg',window_oi:tuple=None,
                        polarity:str='pos',window_size:int=0.05) \
                        -> Union[tuple, dict]:
        """
        Uses peak detection to determine an ERP window, based either on grand 
        averaged data or conditionspecific data.

        Args:
            erps ([dict,list]): Either a list with evoked items (mne) or a 
            dictionary where key, value pairs are condition names and a list
            with conditin specific evoked data, respectively
            elec_oi (list): electrodes of interest. In case, the data of 
            interest is a difference waveform (i.e., contra - ipsi), specify a 
            list of lists, where the first list contains contralateral 
            electrodes and the second list ipsilateral electrodes. 
            method (str, optional): Is the window based on the grand averaged 
            ('cnd_avg') or condition specific data ('cnd_spc'). 
            Defaults to 'cnd_avg'.
            window_oi (tuple, optional): If specified peak detection is 
            restricted to this time window. Defaults to None.
            polarity (str, optional): Is the peak positive ('pos') 
            or negative ('neg'). Defaults to 'pos'.
            window_size (int, optional): Size in seconds of the erp window, 
            which is centered on the detected peak. Defaults to 0.05.

        Returns:
            erp_window([tuple,dict]): Tuple with selected window or, in case of 
            condition specific windows, a dict where key value pairs are 
            condition names and condition specific windows, respectively
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
            
            erp_window = (times[window_idx][idx_peak] - window_size/2, 
                         times[window_idx][idx_peak] + window_size/2)
        
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
                
                erp_window[cnd] = (times[window_idx][idx_peak] - window_size/2, 
                                   times[window_idx][idx_peak] + window_size/2)

        return erp_window

    @staticmethod
    def select_waveform(erps:list, elec_oi:list):

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
    def compare_latencies(erps:Union[dict,list],elec_oi:list=None,
                        window_oi:tuple=None,times:np.array=None,
                        percent_amp:int=75,polarity:str='pos',
                        phase:str='onset'):

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
    def jackknife_contrast(x1:np.array,x2:np.array,times:np.array,
                        percent_amp:int): 

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
    def jack_latency_contrast(x1:np.array,x2:np.array,c1:float,c2:float,
                            times:np.array,print_output:bool=False):

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
	


            

        
        
        
        # get mean and individual data
        evoked_X = np.stack([evoked._data for evoked in erp])
        evoked = mne.combine_evoked(erp, weights = 'equal')







    @staticmethod	
    def baselineCorrect(X, times, base_period):
        ''' 

        Applies baseline correction to an array of data by subtracting the average 
        from the base_period from data array.

        Arguments
        - - - - - 
        X (array): numpy array (trials x electrodes x timepoints)
        times (array): eeg timepoints 
        base_period (list): baseline window (start and end time)


        Returns
        - - - 
        X (array): baseline corrected EEG data
        '''

        # select indices baseline period
        start, end = [np.argmin(abs(times - b)) for b in base_period]

        nr_time = X.shape[2]
        nr_elec = X.shape[1]

        X = X.reshape(-1,X.shape[2])

        X = np.array(np.matrix(X) - np.matrix(X[:,start:end]).mean(axis = 1)).reshape(-1,nr_elec,nr_time)

        return X

    @staticmethod
    def selectMaxTrial(idx, cnds, all_cnds):
        ''' 

        Loops over all conditions to determine which conditions contains the 
        least number of trials. Can be used to balance the number of trials
        per condition.

        Arguments
        - - - - - 
        idx (array): array of trial indices
        cnds (list| str): list of conditions checked for trial balancing. If all, all conditions are used
        all_cnds (array): array of trial specific condition information

        Returns
        - - - 
        max_trial (int): number of trials that can maximally be taken from a condition 
        '''

        if cnds == 'all':
            cnds = np.unique(all_cnds)

        max_trial = []
        for cnd in cnds:
            count = sum(all_cnds[idx] == cnd)
            max_trial.append(count)

        max_trial = min(max_trial)	

        return max_trial


    

    def ipsiContraElectrodeSelection(self):
        '''

        '''	

        # left and right electrodes in standard set-up
        left_elecs = ['Fp1','AF7','AF3','F7','F5','F3','F1','FT7','FC5','FC3',
                    'FC1','T7','CP1','P9','P7','P5','P3','P1','PO7','PO3','O1']
        right_elecs = ['Fp2','AF8','AF4','F8','F6','F4','F2','FT8','FC6','FC4',
                        'FC2','T8','CP2','P10','P8','P6','P4','P2','PO8','PO4','O2']

        # check which electrodes are present in the current set-up				
        left_elecs = [l for l in left_elecs if l in self.ch_names]
        right_elecs = [r for r in right_elecs if r in self.ch_names]

        # select indices of left and right electrodes
        idx_l_elec = np.sort([self.ch_names.index(e) for e in l_elec])
        idx_r_elec = np.sort([self.ch_names.index(e) for e in r_elec])

        return idx_l_elec, idx_r_elec

    def cndSplit(self, beh, conditions, cnd_header):
        '''
        splits condition data in fast and slow data based on median split	

        Arguments
        - - - - - 
        beh (dataframe): pandas dataframe with trial specific info
        conditions (list): list of conditions. Each condition will be split individually
        cnd_header (str): string of column in beh that contains conditions
        '''	

        for cnd in conditions:
            median_rt = np.median(beh.RT[beh[cnd_header] == cnd])
            beh.loc[(beh.RT < median_rt) & (beh[cnd_header] == cnd), cnd_header] = '{}_{}'.format(cnd, 'fast')
            beh.loc[(beh.RT > median_rt) & (beh[cnd_header] == cnd), cnd_header] = '{}_{}'.format(cnd, 'slow')

        return beh	

    def createDWave(self, data, idx_l, idx_r, idx_l_elec, idx_r_elec):
        """Creates a baseline corrected difference wave (contralateral - ipsilateral).
        For this function stimuli need not have been artificially shifted to the same hemifield
        
        Arguments:
            data {array}  -- eeg data (epochs X electrodes X time)
            idx_l {array} -- Indices of trials where stimuli of interest is presented left 
            idx_r {array} -- Indices of trials where stimuli of interest is presented right 
            l_elec {array} -- list of indices of left electrodes
            r_elec {array} -- list of indices from right electrodes
        
        Returns:
            d_wave {array} -- contralateral vs. ipsilateral difference waveform
        """

        # create ipsi and contra waveforms
        ipsi = np.vstack((data[idx_l,:,:][:,idx_l_elec], data[idx_r,:,:][:,idx_r_elec]))
        contra = np.vstack((data[idx_l,:,:][:,idx_r_elec], data[idx_r,:,:][:,idx_l_elec]))
    
        # baseline correct data	
        ipsi = self.baselineCorrect(ipsi, self.eeg.times, self.baseline)
        contra = self.baselineCorrect(contra, self.eeg.times, self.baseline)

        # create ipsi and contra ERP
        ipsi = np.mean(ipsi, axis = (0,1)) 
        contra = np.mean(contra, axis = (0,1))

        d_wave = contra - ipsi

        return d_wave

    def permuteIpsiContra(self, eeg, contra_idx, ipsi_idx, nr_perm = 1000):
        """Calls upon createDWave to create permuted difference waveforms. Can for example be used to calculate 
        permuted area under the curve to establish reliability of a component. Function assumes that it is if all 
        stimuli are presented within one hemifield
        
        Arguments:
            eeg {mne object}  -- epochs object mne
            contra_idx {array} -- Indices of contralateral electrodes 
            ipsi_idx {array} -- Indices of ipsilateral electrodes
        
        Keyword Arguments:
            nr_perm {int} -- number of permutations (default: {1000})
        
        Returns:
            d_wave {array} -- contralateral vs. ipsilateral difference waveform (can be used as sanity check)
            d_waves_p {array} -- permuted contralateral vs. ipsilateral difference waveforms (nr_perms X timepoints)
        """		

        data = eeg._data
        nr_epochs = data.shape[0]

        # create evoked objects using mne functionality
        evoked = eeg.average().apply_baseline(baseline = self.baseline)
        d_wave = np.mean(evoked._data[contra_idx] - evoked._data[ipsi_idx], axis = 0)

        # initiate empty array for shuffled waveforms
        d_waves_p = np.zeros((nr_perm, eeg.times.size))

        for p in range(nr_perm):
            idx_p = np.random.permutation(nr_epochs)
            idx_left = idx_p[::2]
            idx_right = idx_p[1::2]
            d_waves_p[p] = self.createDWave(data, idx_left, idx_right, contra_idx, ipsi_idx)

        return d_wave, d_waves_p




    def conditionERP(self, sj, conditions, cnd_header, erp_name = '', collapsed = True, RT_split = False):
        '''

        '''

        if collapsed and conditions != ['all']:
            cnd += ['all']

        # loop over unique levels of interest
        for factor in np.unique(self.beh[self.header]):
            
            idx_f = np.where(self.beh[self.header] == factor)[0]	
            
            # loop over conditions
            for cnd in conditions:
                
                # select condition indices
                if cnd == 'all':
                    idx_c = np.arange(self.beh[cnd_header].size)
                else:	
                    idx_c = np.where(self.beh[cnd_header] == cnd)[0]

                idx = np.array([idx for idx in idx_c if idx in idx_f])
            
                if idx.size == 0:
                    print('no data found for {}'.format(cnd))
                    continue
                
                fname = 'sj_{}-{}-{}-{}'.format(sj, erp_name, factor, cnd)
                self.createERP(self.beh, self.eeg, idx, fname, RT_split = RT_split)














    def ipsiContra(self, sj, left, right, l_elec = ['PO7'], r_elec = ['PO8'], conditions = 'all', cnd_header = 'condition', midline = None, erp_name = '', RT_split = False, permute = False):
        ''' 

        Creates laterilized ERP's by cross pairing left and right electrodes with left and right position labels.
        ERPs are made for all conditios collapsed and for individual conditions

        Arguments
        - - - - - 
        sj (int): subject id (used for saving)
        left (list): stimulus labels indicating left spatial position(s) as logged in beh 
        right (list): stimulus labels indicating right spatial position(s) as logged in beh
        l_elec (list): left electrodes (right hemisphere)
        r_elec (list): right hemisphere (left hemisphere)
        conditions (str | list): list of conditions. If all, all unique conditions in beh are used
        cnd_header (str): name of condition column
        midline (None | dict): Can be used to limit analysis to trials where a specific 
                                stimulus (key of dict) was presented on the midline (value of dict)
        erp_name (str): name of the pickle file to store erp data
        RT_split (bool): If true each condition is also analyzed seperately for slow and fast RT's (based on median split)
        permute (bool | int): If true (in case of a number), randomly flip the hemifield of the stimulus of interest and calculate ERPs

        Returns
        - - - -
         

        '''

        # make sure it is as if all stimuli of interest are presented right from fixation	
        if not self.flipped:
            print('Flipping is done based on {} column in beh and relative to values {}. \
                If not correct please flip trials beforehand'.format(self.header, left))
            self.topoFlip(left , self.header)
        else:
            print('It is assumed as if all stimuli are presented right')

        # report that left right specification contains invalid values
        # ADD WARNING!!!!!!
        idx_l, idx_r = [],[]
        if len(left)>0:
            idx_l = np.sort(np.hstack([np.where(self.beh[self.header] == l)[0] for l in left]))
        if len(right)>0:	
            idx_r = np.sort(np.hstack([np.where(self.beh[self.header] == r)[0] for r in right]))

        # if midline, only select midline trials
        if midline != None:
            idx_m = []
            for key in midline.keys():
                idx_m.append(np.sort(np.hstack([np.where(self.beh[key] == m)[0] for m in midline[key]])))
            idx_m = np.hstack(idx_m)
            idx_l = np.array([idx for idx in idx_l if idx in idx_m])
            idx_r = np.array([idx for idx in idx_r if idx in idx_m])

        #if balance:
        #	max_trial = self.selectMaxTrial(np.hstack((idx_l, idx_r)), conditions, self.beh[cnd_header])

        # select indices of left and right electrodes
        idx_l_elec = np.sort([self.eeg.ch_names.index(e) for e in l_elec])
        idx_r_elec = np.sort([self.eeg.ch_names.index(e) for e in r_elec])

        if conditions == 'all':
            conditions = ['all'] + list(np.unique(self.beh[cnd_header]))

        for cnd in conditions:

            # select left and right trials for current condition
            # first select condition indices
            if cnd == 'all':
                idx_c = np.arange(self.beh[cnd_header].size)
            else:	
                idx_c = np.where(self.beh[cnd_header] == cnd)[0]
        
            # split condition indices in left and right trials	
            idx_c_l = np.array([l for l in idx_c if l in idx_l], dtype = int)
            idx_c_r = np.array([r for r in idx_c if r in idx_r], dtype = int)

            if idx_c_l.size == 0 and idx_c_r.size == 0:
                print('no data found for {}'.format(cnd))
                continue
            
            fname = 'sj_{}-{}-{}'.format(sj, erp_name, cnd)
            idx = np.hstack((idx_c_l, idx_c_r))
            self.createERP(self.beh, self.eeg, idx, fname, RT_split = RT_split)

            if permute:
                # STILL NEEDS DOUBLE CHECKING AGAINST MNE OUTPUT (looks ok!)
                # make it as if stumuli were presented left and right
                eeg = self.eeg[idx]
                d_wave, d_waves_p = self.permuteIpsiContra(eeg, idx_l_elec, idx_r_elec, nr_perm = permute)

                # save results
                perm_erps = {'d_wave': d_wave, 'd_waves_p': d_waves_p}
                pickle.dump(perm_erps, open(self.FolderTracker(['erp',self.header],'sj_{}_{}_{}_perm.pickle'.format(sj, erp_name, cnd)) ,'wb'))

            # create erp (nr_elec X nr_timepoints)
            #if cnd != 'all' and balance:
            #	idx_balance = np.random.permutation(ipsi.shape[0])[:max_trial]
            #	ipsi = ipsi[idx_balance,:,:]
            #	contra = contra[idx_balance,:,:]
            

    def topoSelection(self, sj, conditions = 'all', loc = 'all', midline = None, balance = False, topo_name = ''):
        ''' 

        Arguments
        - - - - - 


        Returns
        - - - -

        '''

        try:
            with open(self.FolderTracker(['erp',self.header],'topo_{}.pickle'.format(topo_name)) ,'rb') as handle:
                topos = pickle.load(handle)
        except:
            topos = {}

        # update dictionary
        if str(sj) not in topos.keys():
            topos.update({str(sj):{}})

        if conditions == 'all':
            conditions = ['all'] + list(np.unique(self.beh['condition']))
        else:
            conditions = ['all'] + conditions

        # filthy hack to get rid of 'None' in index array
        if self.beh[self.header].dtype != 'int64':
            self.beh[self.header][self.beh[self.header] == 'None'] = np.nan

        if loc != 'all':
            idx_l = np.sort(np.hstack([np.where(np.array(self.beh[self.header], dtype = float) == l)[0] for l in loc]))

        if balance:
            max_trial = self.selectMaxTrial(idx_l, conditions, self.beh['condition'])
            
            # if midline, only select midline trials
            if midline != None:
                idx_m = []
                for key in midline.keys():
                    idx_m.append(np.sort(np.hstack([np.where(self.beh[key] == m)[0] for m in midline[key]])))
                idx_m = np.hstack(idx_m)
                idx_l = np.array([idx for idx in idx_l if idx in idx_m])

        for cnd in conditions:

            # get condition data
            if cnd == 'all':
                idx_c = np.arange(self.beh['condition'].size)	
            else:	
                idx_c = np.where(self.beh['condition'] == cnd)[0]
            
            if loc != 'all':
                idx_c_l = np.array([l for l in idx_c if l in idx_l])
            elif loc == 'all' and midline != None:
                idx_c_l = np.array([l for l in idx_c if l in idx_m])
            else:
                idx_c_l = idx_c	

            if idx_c_l.size == 0:
                print('no topo data found for {}'.format(cnd))
                continue				

            topo = self.eeg[idx_c_l,:,:]

            # baseline correct topo data
            topo = self.baselineCorrect(topo, self.times, self.baseline)
            
            if cnd != 'all' and balance:
                idx_balance = np.random.permutation(topo.shape[0])[:max_trial]
                topo = topo[idx_balance,:,:]

            topo = np.mean(topo, axis = 0)

            topos[str(sj)].update({cnd:topo})
        
        with open(self.FolderTracker(['erp',self.header],'topo_{}.pickle'.format(topo_name)) ,'wb') as handle:
            pickle.dump(topos, handle)	


    def ipsiContraCheck(self,sj, left, right, l_elec = ['PO7'], r_elec = ['PO8'], conditions = 'all', midline = None, erp_name = ''):
        '''

        '''

        file = self.FolderTracker(['erp',self.header],'{}.pickle'.format(erp_name))

        if os.path.isfile(file):
            with open(file ,'rb') as handle:
                erps = pickle.load(handle)
        else:
            erps = {}

        # update dictionary
        if str(sj) not in erps.keys():
            erps.update({str(sj):{}})

        # select left and right trials
        idx_l = np.sort(np.hstack([np.where(self.beh[self.header] == l)[0] for l in left]))
        idx_r = np.sort(np.hstack([np.where(self.beh[self.header] == r)[0] for r in right]))

        # select indices of left and right electrodes
        idx_l_elec = np.sort([self.ch_names.index(e) for e in l_elec])
        idx_r_elec = np.sort([self.ch_names.index(e) for e in r_elec])

        if conditions == 'all':
            conditions = ['all'] + list(np.unique(self.beh['condition']))

        for cnd in conditions:

            erps[str(sj)].update({cnd:{}})

            # select left and right trials for current condition
            if cnd == 'all':
                idx_c = np.arange(self.beh['condition'].size)
            else:	
                idx_c = np.where(self.beh['condition'] == cnd)[0]
            
            idx_c_l = np.array([l for l in idx_c if l in idx_l])
            idx_c_r = np.array([r for r in idx_c if r in idx_r])

            l_ipsi = self.eeg[idx_c_l,:,:][:,idx_l_elec,:]
            l_contra = self.eeg[idx_c_l,:,:][:,idx_r_elec,:] 
            r_ipsi = self.eeg[idx_c_r,:,:][:,idx_r_elec,:]
            r_contra = self.eeg[idx_c_r,:,:][:,idx_l_elec,:] 

            # baseline correct data	
            l_ipsi = self.baselineCorrect(l_ipsi, self.times, self.baseline)
            l_contra = self.baselineCorrect(l_contra, self.times, self.baseline)

            r_ipsi = self.baselineCorrect(r_ipsi, self.times, self.baseline)
            r_contra = self.baselineCorrect(r_contra, self.times, self.baseline)

            # create erp
            l_ipsi = np.mean(l_ipsi, axis = (0,1))
            l_contra = np.mean(l_contra, axis = (0,1))

            r_ipsi = np.mean(r_ipsi, axis = (0,1))
            r_contra = np.mean(r_contra, axis = (0,1))

            erps[str(sj)][cnd].update({'l_ipsi':l_ipsi,'l_contra':l_contra,'r_ipsi':r_ipsi,'r_contra':r_contra})	

        # save erps	
        with open(self.FolderTracker(['erp',self.header],'{}.pickle'.format(erp_name)) ,'wb') as handle:
            pickle.dump(erps, handle)	

            
if __name__ == '__main__':

    pass

    # project_folder = '/home/dvmoors1/big_brother/Dist_suppression'
    # os.chdir(project_folder) 
    # subject_id = [1,2,5,6,7,8,10,12,13,14,15,18,19,21,22,23,24]
    # subject_id = [3,4,9,11,17,20]	
    # header = 'dist_loc'

    # session = ERP(header = header, baseline = [-0.3,0])
    # if header == 'target_loc':
    # 	conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
    # 	midline = {'dist_loc': [0,3]}
    # else:
    # 	conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
    # 	midline = {'target_loc': [0,3]}
    
    # for sj in subject_id:

    # 	session.selectERPData(sj, time = [-0.3, 0.8], l_filter = 30) 
    # 	session.ipsiContra(sj, left = [2], right = [4], l_elec = ['P7','P5','P3','PO7','PO3','O1'], 
    # 								r_elec = ['P8','P6','P4','PO8','PO4','O2'], midline = None, balance = True, erp_name = 'lat-down1')
    # 	session.ipsiContra(sj, left = [2], right = [4], l_elec = ['P7','P5','P3','PO7','PO3','O1'], 
    # 								r_elec = ['P8','P6','P4','PO8','PO4','O2'], midline = midline, balance = True, erp_name = 'lat-down1-mid')

    # 	session.topoFlip(left = [1,2])
        
    # 	session.topoSelection(sj, loc = [2,4], midline = None, topo_name = 'lat-down1')
        # session.topoSelection(sj, loc = [2,4], midline = midline, topo_name = 'lat-down1-mid')




