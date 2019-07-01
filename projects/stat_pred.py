import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import os
import mne
import sys
import glob
import pickle
import logging
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed
from beh_analyses.PreProcessing import *
#from eeg_analyses.TF import * 
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *

# subject specific info

# project specific info

project = 'stat_pred'
part = 'exp1'
factors = ['trial_type','block_type','dist_high']
labels = [['distractor','single_target'],['unpred','shape_pred','color_pred', 'pred'],['yes','no']]
to_filter = ['RT'] 
project_param = ['practice','nr_trials','trigger','condition','RT', 'subject_nr','selected_high_prob',
				'block_type', 'correct','dist_high','dist_loc','dist_shape',
                'high_loc', 'target_high','target_loc','target_shape', 'target_color','trial_type','block_cnt']

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class StatPred(FolderStructure):

	def __init__(self): pass


	def prepareBEH(self, project, part, factors, labels, project_param, to_filter):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and correct == 1', cnd_sel = False, save = True)
		PP.exclude_outliers(criteria = dict(RT = 'RT_filter == True', correct = ''))
		#PP.prep_JASP(agg_func = 'mean', voi = 'RT', data_filter = 'RT_filter == True', save = True)
		PP.save_data_file()

if __name__ == '__main__':

	os.environ['MKL_NUM_THREADS'] = '5' 
	os.environ['NUMEXP_NUM_THREADS'] = '5'
	os.environ['OMP_NUM_THREADS'] = '5'
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/stat_pred'
	os.chdir(project_folder)

	# initiate current project
	PO = StatPred()


	# analyze behavior
	# behavioral experiment 1 
	PO.prepareBEH('stat_pred', 'exp1', factors, labels, project_param, to_filter)