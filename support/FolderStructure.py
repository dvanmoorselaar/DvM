import os
import mne
import pickle
import glob

import pandas as pd
from support.support import *
from IPython import embed

class FolderStructure(object):

    '''
    Creates the folder structure
    '''

    def __init__(self):
        pass

    @staticmethod
    def folder_tracker(extension = [], filename = '', overwrite = True):
        '''
        Creates a folder address. At the same time it
        checks whether the specific folder already exists (if not it is created)

        Arguments
        - - - - -
        extension (list): list of subfolders that are attached to current working directory
        filename (str): name of file
        overwrite (bool): if overwrite is False, an * is added to the filename

        Returns
        - - - -
        folder (str): file adress

        '''

        # create folder adress
        folder = os.getcwd()
        if extension != []:
            folder = os.path.join(folder,*extension)

        # check whether folder exists
        if not os.path.isdir(folder):
            os.makedirs(folder)

        if filename != '':
            if not overwrite:
                while os.path.isfile(os.path.join(folder,filename)):
                    end_idx = len(filename) - filename.index('.')
                    filename = filename[:-end_idx] + '+' + filename[-end_idx:]
            folder = os.path.join(folder,filename)

        return folder


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
        eeg = mne.read_epochs(self.folder_tracker(extension = ['processed'],
                            filename = 'subject-{}_{}-epo.fif'.format(sj, name)))
        if eeg.metadata is not None:
            beh = eeg.metadata
        else:
            # read in processed behavior from pickle file
            if beh_file:
                beh = pickle.load(open(self.folder_tracker(extension = ['beh','processed'],
                                    filename = 'subject-{}_{}.pickle'.format(sj, name)),'rb'), encoding='latin1')
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
            files = glob.glob(self.folder_tracker(extension=[
                    'beh', 'raw'],
                    filename=f'subject-{sj}_session_{session}*.csv'))

        # read in as dataframe
        beh = [pd.read_csv(file) for file in files]
        beh = pd.concat(beh)

        return beh

    def read_erps(self, erp_folder:str,erp_name:str,
                cnds:list=None,sjs:list='all')->dict:
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
                                extension = ['erp',erp_folder],
                                filename = f'sj_*_{cnd}_{erp_name}-ave.fif')))
            else:
                files = [self.folder_tracker(
                                extension = ['erp',erp_folder],
                                filename = f'sj_{sj}_{cnd}_{erp_name}-ave.fif')
                                            for sj in sjs]

            # read in actual data
            erps[cnd] = [mne.read_evokeds(file)[0] for file in files]

        return erps

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
        extension = ['bdm'] + bdm_folder_path

        if sjs == 'all':
            files = sorted(glob.glob(self.folder_tracker(
                            extension = extension,
                            filename = f'sj_*_{bdm_name}.pickle')))
        else:
            files = [self.folder_tracker(extension = extension,
                    filename = f'sj_{sj}_{bdm_name}.pickle')for sj in sjs]

        bdm = [pickle.load(open(file, "rb")) for file in files]

        return bdm

