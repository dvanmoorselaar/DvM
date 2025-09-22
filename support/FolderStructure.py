import os
import sys
import mne
import pickle
import glob
import re
import copy

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Generic, Union, Tuple, Any
from support.support import exclude_eye, match_epochs_times, trial_exclusion
from IPython import embed

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

class FolderStructure(object):

    '''
    Handles reading and saving of files within dvm toolbox
    '''

    def __init__(self):
        pass

    @staticmethod
    def _extract_subject_number(fname: str) -> int:
        """Extract subject number from filename."""
        match = re.search(r'sj_(\d+)_', fname)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract subject number from {fname}")

    @staticmethod
    def folder_tracker(ext:list=[], fname:str=None,overwrite:bool=True)->str:
        """
        Creates a folder address with the current working directory as 
        a base. In case the specified path does not exits, it is created

        Args:
            ext (list, optional): list of subfolders that are attached 
            to current working directory. Defaults to [] (i.e., 
            no subfolders).
            fname (str, optional): filename. Defaults to None.
            overwrite (bool, optional): if overwrite is False, the 
            original file remains untouched and an * is appended to the 
            specified file name Defaults to True.

        Returns:
            path: specified file path
        """

        # create folder adress
        path = os.getcwd()
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

    def load_processed_eeg(self,sj:int,fname:str,preproc_name:str,
                        eye_dict:dict=None,beh_file:bool=True,
                        excl_factor:dict=None)->\
                        Tuple[pd.DataFrame,mne.Epochs]:
        """
        Reads in preprocessed eeg data (mne.Epochs object) and 
        behavioral data to be used for subsequent analyses. If data is 
        excluded based on eye movement criteria and a preproccesing 
        overview file exists, this file is updated with eyemovement 
        info per subject.
 
        Args:
            sj (int): subject identifier
            fname (str): name of processed eeg data
            preproc_name (str): name specified for specific 
            preprocessing pipeline
            eye_dict (dict, optional): Each key specifies criteria
            used to exclude data based on eye_movement data. 
            Defaults to None, i.e., no eye-movement exclusion.
            Currently, supports the following keys:
            eye_window (tuple): window used to search for eye movements
            eye_ch (str): channel to search for eye movements
            angle_thresh (float): threshold in degrees of visual angles
            step_param (dict)
            use_tracker (bool): should eye tracker data be used to 
            search eye movements. If not exclusion will be based on 
            step algorhytm applied to eog channel as specified in eye_ch
            beh_file (bool, optional): Is epoch info stored in a 
            seperate file or within epochs data. Defaults to True.
            excl_factor (dict, optional): This gives the option to 
			exclude specific conditions from the data that is read in. 
			
			For example, to only include trials where the cue was 
			pointed to the left and not to the right 
			specify the following: 

			excl_factor = dict(cue_direc = ['right']). 
			
			Mutiple column headers and multiple variables per header can 
			be specified. Defaults to None (i.e., no trial exclusion).

        Returns:
            beh (pd.DataFrame): behavioral data alligned to eeg data
            epochs (mne.Epochs): preprocessed eeg data
        """
        
        # start by reading in processed eeg data
        epochs = mne.read_epochs(self.folder_tracker(ext = ['processed'],
                            fname = f'sj_{sj}_{fname}-epo.fif'))

        # check whether metadata is saved alongside epoched eeg
        if epochs.metadata is not None:
            df = copy.copy(epochs.metadata)
        else:
            if beh_file:
                # read in seperate behavior file
                df = pd.read_csv(self.folder_tracker(ext=['beh','processed'],
                    fname = f'subject-{sj}_{fname}.csv'))
            else:
                df = pd.DataFrame({'condition': epochs.events[:,2]})

        # remove a subset of trials 
        if type(excl_factor) == dict: 
            df, epochs,_ = trial_exclusion(df, epochs, excl_factor)

        # reset index(to properly align beh and epochs)
        df.reset_index(inplace = True, drop = True)

        # exclude eye movements based on threshold criteria in eye_dict
        if eye_dict is not None:
            file = FolderStructure().folder_tracker(
                            ext = ['preprocessing','group_info'],
                            fname = f'preproc_param_{preproc_name}.csv')
            eye = np.load(self.folder_tracker(ext=['eye','processed'],
                fname=f'sj_{sj}_{fname}.npz'))
            df, epochs = exclude_eye(sj,df,epochs,eye_dict,eye,file)

        return df, epochs


    def load_data(self, sj, name = 'all', eyefilter = False, eye_window = None, eye_ch = 'HEOG', eye_thresh = 1, eye_dict = None, beh_file = True, use_tracker = True):
        '''
        loads EEG and behavior data

        Arguments
        - - - - -
        sj (int): subject number
        eye_window (tuple|list): timings to scan for eye movements
        eyefilter (bool): in or exclude eye movements based on step like algorythm
        eye_ch (str): name of channel to scan for eye movements
        eye_thresh (int): exclude trials with an saccades exceeding threshold (in visual degrees)
        eye_dict (dict): if not None, needs to be dict with three keys specifying parameters for sliding window detection
        beh_file (bool): Is epoch info stored in a seperate file or within behavior file
        use_tracker (bool): specifies whether eye tracker data should be used (i.e., is reliable)

        Returns
        - - - -
        beh (Dataframe): behavior file
        eeg (mne object): preprocessed eeg data

        '''

        # read in processed EEG data
        eeg = mne.read_epochs(self.folder_tracker(ext = ['processed'],
                            fname = 'subject-{}_{}-epo.fif'.format(sj, name)))
        if eeg.metadata is not None:
            beh = eeg.metadata
        else:
            # read in processed behavior from pickle file
            if beh_file:
                beh = pickle.load(open(self.folder_tracker(ext = ['beh','processed'],
                                    fname = 'subject-{}_{}.pickle'.format(sj, name)),'rb'), encoding='latin1')
                beh = pd.DataFrame.from_dict(beh)
            else:
                beh = pd.DataFrame({'condition': eeg.events[:,2]})

        if eyefilter:
            beh, eeg = filter_eye(beh, eeg, eye_window, eye_ch, eye_thresh, eye_dict, use_tracker)

        return beh, eeg

    def read_raw_beh(self,sj:int=None,session:int=None,
                    files:bool=False)->pd.DataFrame:
        """Reads in raw behavior data from csv file to link to
        epochs data.

        Args:
            sj (int): subject identifier
            session (int): session identifier
            files ()

        Returns:
            beh (pd.DataFrame): dataframe with raw behavior
        """

        # get all files for this subject's session
        if not files:
            files = sorted(glob.glob(self.folder_tracker(ext=[
                    'beh', 'raw'],
                    fname=f'subject-{sj}_session_{session}*.csv')))
        if files == []:
            return []
        # read in as dataframe
        beh = [pd.read_csv(file) for file in files]
        beh = pd.concat(beh)
        beh.reset_index(inplace = True, drop = True)

        # control for duplicate trial numbers
        if len(files) > 1 and 'nr_trials' in beh:
            nr_trials = beh.nr_trials.values

        return beh

    #@blockPrinting
    def read_erps(self,erp_name:str,
                cnds:list=None,sjs:list='all',
                match:str=False)->Tuple[dict,np.array]:
        """
        Read in evoked files of a specific analysis. Evoked data is 
        returned in dictionary

        Args:
            erp_name (str): name assigned to erp analysis
            cnds (list, optional): conditions of interest. 
            Defaults to None
            (i.e., no conditions).
            sjs (list, optional): List of subjects. Defaults to 'all'.
            match (str, optional): If match is not False, it will be 
            explicitly checked whether timing events match between
            read in files. If not, samples will be removed until 
            matching. Defaults to False.

        Returns:
            erps (dict): Dictionary with evoked data (with conditions as keys)
        """

        if cnds is None:
            cnds = ['all_data']

        # initiate condtion dict
        erps = dict.fromkeys(cnds, 0)
        
        # loop over conditions
        for cnd in cnds:
            if sjs == 'all':
                files = sorted(glob.glob(self.folder_tracker(
                            ext = ['erp','evoked'],
                            fname = f'sj_*_{cnd}_{erp_name}-ave.fif')),
                            key = lambda s: int(re.search(r'\d+', s).group()))
            else:
                files = [self.folder_tracker(
                                ext = ['erp','evoked'],
                                fname = f'sj_{sj}_{cnd}_{erp_name}-ave.fif')
                                            for sj in sjs]

            # read in actual data
            erps[cnd] = [mne.read_evokeds(file)[0] for file in files]
            if match:
                nr_samples = [erp.times.size for erp in erps[cnd]]
                if len(set(nr_samples)) > 1:
                    print('samples are artificilaly aligned. Please inspect ') 
                    print('data carefully' )
                    erps[cnd] = match_epochs_times(erps[cnd])
        
        times = erps[cnd][0].times

        return erps, times
    
    def read_tfr(self,tfr_folder_path:list,tfr_name:str,cnds:list=None,
                sjs:list='all')->list:
        """
        Read in time-frequency data as created by TFR class.
        Time-frequency data is returned within a dictionary.

        Args:
            tfr_folder_path (list): List of folders (as created by TFR)
            within tfr folder pointing towards files of interest (e.g.,
            ['wavelet'])
            tfr_name (str): name assigned to tfr analysis
            cnds (list, optional): conditions of interest. Defaults to None
            sjs (list, optional): List of subjects. Defaults to 'all'.

        Returns:
            tfr (list): list with time-frequency data
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
                files = sorted(glob.glob(self.folder_tracker(
                            ext = ext,
                            fname = f'sj_*_{tfr_name}_{cnd}-tfr.h5')),
                            key = lambda s: int(re.search(r'\d+', s).group()))
            else:
                files = [self.folder_tracker(ext = ext,
                    fname = f'sj_{sj}_{tfr_name}_{cnd}-tfr.h5')for sj in sjs]

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
        """
        Read in classification data as created by BDM class.

        Args:
            bdm_folder_path (list): List of folders (as created by BDM) within
            bdm folder pointing towards files of interest (e.g.,
            ['target_loc', 'all_elecs', 'cross'])
            bdm_name (str): name assigned to bdm analysis
            sjs (list, optional): List of subjects. Defaults to 'all'.

        Returns:
            bdm (list): list with decoding data
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
                files = sorted(glob.glob(self.folder_tracker(
                    ext = ext,
                    fname = f'sj_*_{name}.pickle')),
                    key = lambda s: int(re.search(r'sj_(\d+)_', s).group(1)))
            else:
                files = [self.folder_tracker(ext = ext,
                    fname = f'sj_{sj}_{name}.pickle')for sj in sjs]
                
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
                  ctf_name:str,sjs:list='all')->list:

        # set extension
        ext = ['ctf'] + ctf_folder_path

        if output_type=='ctf':
            output = 'ctfs'
        elif output_type=='info':
            output = 'ctf_info'
        elif output_type=='param':
            output = 'ctf_param'

        if sjs == 'all':
            files = sorted(glob.glob(self.folder_tracker(
                            ext = ext,
                            fname = f'{output}_*_{ctf_name}.pickle')),
                            key = lambda s: int(re.search(r'\d+', s).group()))
        else:
            files = [self.folder_tracker(ext = ext,
                    fname = f'{output}_{sj}_{ctf_name}.pickle')for sj in sjs]

        ctfs = [pickle.load(open(file, 'rb')) for file in files]

        return ctfs