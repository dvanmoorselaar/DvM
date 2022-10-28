import os
import sys
import mne
import pickle
import glob
import re

import numpy as np
import pandas as pd

from typing import Optional, Generic, Union, Tuple, Any
from support.support import *
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
                        eye_dict:dict=None,beh_file:bool=True)->\
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

        Returns:
            beh (pd.DataFrame): behavioral data alligned to eeg data
            epochs (mne.Epochs): preprocessed eeg data
        """
        
        # start by reading in processed eeg data
        epochs = mne.read_epochs(self.folder_tracker(ext = ['processed'],
                            fname = f'subject-{sj}_{fname}-epo.fif'))

        # check whether metadata is saved alongside epoched eeg
        if epochs.metadata is not None:
            beh = epochs.metadata
        else:
            if beh_file:
                # read in seperate behavior file
                beh = pd.read_csv(self.folder_tracker(ext=['beh','processed'],
                    fname = f'subject-{sj}_{fname}.csv'))
            else:
                beh = pd.DataFrame({'condition': epochs.events[:,2]})

        # exclude eye movements based on threshold criteria in eye_dict
        if eye_dict is not None:
            file = FolderStructure().folder_tracker(
                            ext = ['preprocessing','group_info'],
                            fname = f'preproc_param_{preproc_name}.csv')
            beh, epochs = exclude_eye(sj,beh,epochs,eye_dict,file)

        return beh, epochs


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

    def read_raw_beh(self,sj:int=None,session:int=None,files:bool=False)->pd.DataFrame:
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
                    
        # read in as dataframe
        beh = [pd.read_csv(file) for file in files]
        beh = pd.concat(beh)

        return beh

    #@blockPrinting
    def read_erps(self, erp_folder:str,erp_name:str,
                cnds:list=None,sjs:list='all')->Tuple[dict,np.array]:
        """
        Read in evoked files of a specific analysis. Evoked data is returned
        in dictionary

        Args:
            erp_folder (str): name of folder within erp folder that contains
            data of interest
            erp_name (str): name assigned to erp analysis
            cnds (list, optional): conditions of interest. Defaults to None
            (i.e., no conditions).
            sjs (list, optional): List of subjects. Defaults to 'all'.

        Returns:
            erps (dict): Dictionary with evoked data (with conditions as keys)
        """

        # initiate condtion dict
        erps = dict.fromkeys(cnds, 0)

        # loop over conditions
        for cnd in cnds:
            if sjs == 'all':
                files = sorted(glob.glob(self.folder_tracker(
                            ext = ['erp',erp_folder],
                            fname = f'sj_*_{cnd}_{erp_name}-ave.fif')),
                            key = lambda s: int(re.search(r'\d+', s).group()))
            else:
                files = [self.folder_tracker(
                                ext = ['erp',erp_folder],
                                fname = f'sj_{sj}_{cnd}_{erp_name}-ave.fif')
                                            for sj in sjs]

            # read in actual data
            erps[cnd] = [mne.read_evokeds(file)[0] for file in files]
        times = erps[cnd][0].times

        return erps, times

    def read_bdm(self,bdm_folder_path:list,bdm_name:str,sjs:list='all')->dict:
        """
        Read in classification data as created by BDM class.
        Decoding scores are returned within a dictionary.

        Args:
            bdm_folder_path (list): List of folders (as created by BDM) within
            bdm folder pointing towards files of interest (e.g.,
            ['target_loc', 'all_elecs', 'cross'])
            bdm_name (str): name assigned to bdm analysis
            sjs (list, optional): List of subjects. Defaults to 'all'.

        Returns:
            erps (dict): Dictionary with evoked data (with conditions as keeys)
        """

        # set extension
        ext = ['bdm'] + bdm_folder_path

        if sjs == 'all':
            files = sorted(glob.glob(self.folder_tracker(
                            ext = ext,
                            fname = f'sj_*_{bdm_name}.pickle')),
                            key = lambda s: int(re.search(r'\d+', s).group()))
        else:
            files = [self.folder_tracker(ext = ext,
                    fname = f'sj_{sj}_{bdm_name}.pickle')for sj in sjs]

        bdm = [pickle.load(open(file, "rb")) for file in files]

        return bdm

