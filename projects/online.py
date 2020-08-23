import os
import mne
import sys
import glob
import pickle
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM_3')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#plt.switch_backend('Qt4Agg') # run next two lines to interactively scroll through plots
#import matplotlib

#import matplotlib          # run these lines only when running sript via ssh connection
#matplotlib.use('agg')

from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.TF import * 
from eeg_analyses.BDM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
#from stats.nonparametric import *


# project specific info

project_param_exp_set_size = ['practice','nr_trials','corr_angle', 'subject_nr','deviation','mem_bin','mem_search_match',
				'response_resp_question1', 'response_resp_question2_1','response_time_mouse_resp','response_time_search_resp',
				'search_correct','search_prob','set_size','target_loc']

project_param_exp_1 = ['practice','nr_trials','corr_angle', 'subject_nr','deviation','mem_bin','mem_search_loc',
				'response_question1', 'response_question2','response_time_mouse_resp','response_time_search_resp',
				'search_correct','target_high_prob','target_loc']	

project_param_dist = ['practice','nr_trials','corr_angle', 'subject_nr','deviation','mem_bin','mem_dist_match',
				'response_question1', 'response_resp_question2_1','response_time_mouse_resp','response_time_search_resp',
				'search_correct','dist_prob','target_loc', 'dist_loc']				


class Online(FolderStructure):

	def __init__(self): pass


	def prepareBEH(self, project, part, factors, labels, project_param, to_filter):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False, ext = '.xlsx')
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and search_correct == 1', cnd_sel = True, save = True)
		PP.exclude_outliers(criteria = dict(response_time_search_resp = 'response_time_search_resp_filter == True', search_correct = '', deviation = ''))
		PP.save_data_file()


if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/Online'
	os.chdir(project_folder)

	# initiate current project
	PO = Online()

	# analyze behavior
	# set size manipulation
	#PO.prepareBEH('Online', 'set_size', ['set_size','mem_search_match','search_prob'], 
	#			  [[4,6,8],['match','mismatch'],['high','low']], project_param_exp_set_size, ['response_time_search_resp'] )

	#PO.prepareBEH('Online', 'exp_1', ['mem_search_loc','target_high_prob'], 
	#			  [['match','mismatch'],['no','yes']], project_param_exp_1, ['response_time_search_resp'] )

	PO.prepareBEH('Online', 'distractor', ['mem_dist_match','dist_prob'], 
				  [['match','mismatch'],['high','low']], project_param_dist, ['response_time_search_resp'] )