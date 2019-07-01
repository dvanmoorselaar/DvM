"""
Created by Dirk van Moorselaar on 2014-08-14.
Copyright (c) 2014 DvM. All rights reserved.
"""

import os
import ast

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed
from scipy.stats import t
from math import sqrt
from operator import mul
from itertools import product
from scipy.optimize import curve_fit

class PreProcessing(object):
	"""
	
	Prepares OpenSesame output for further analysis steps.
		Includes functionilty for cobining seperate subject files, filtering, outlier selection.
	"""
	
	def __init__(self, project = '', part = '', factor_headers = [], factor_labels = []):
		"""

		Arguments
		- - - - - 
		project (str): Name of project folder
		part (str): subfolder where behavior of specific experiment is stored
		factor_headers (list): list of experimental manipulations (column names in .csv file)
		factor_labels (list): list of factors per experimental manipulation in factor_headers

		Returns
		- - - -

		"""


		self.project_folder = os.path.join(os.getcwd(), part)
		self.factor_headers = factor_headers
		self.factor_labels = factor_labels
		self.outlier = np.array(())

	def create_folder_structure(self):
		"""
		Creates folder structure for behavioral analysis to ensure that raw data is separated from subsequent analysis
		- raw folder (single subject files and combined subject file)
		- analysis folder (combined subject file, figures)
		"""

		# assuming that project folder is defined we must make a raw folder and an analysis folder
		if not os.path.isdir(os.path.join(self.project_folder,'raw')):
			os.makedirs (os.path.join(self.project_folder,'raw'))
		if not os.path.isdir(os.path.join(self.project_folder,'analysis')):
			os.makedirs (os.path.join(self.project_folder,'analysis'))
			os.makedirs (os.path.join(self.project_folder,'analysis','figs'))

	def combine_single_subject_files(self, save = False):
		"""
		Combines all csv files into a single xlsx file. The resulting xlsx file has a single header row and contains experiment data from all participants
		
		Arguments
		- - - - - 
		save (bool): If True, save a datafile with all single subject files combined 

		"""	

		# I want a statement here that raises an error when the raw and analysis folder do not exist in the project folder

		# get all csv files from raw data folder (make sure only csv files are selected)
		filenames = os.listdir(os.path.join(self.project_folder,'raw'))
		subject_files = [filename for filename in filenames if filename.endswith('.csv')]
	
		# read csv file into dataframe
		raw_data_comb = []
		for subject in subject_files:
			print subject
			try:
				csv_file = os.path.join(self.project_folder,'raw',subject)
				raw_data = pd.read_csv(csv_file)
			except:
				print(csv_file)	
			raw_data_comb.append(raw_data)
		
		raw_data_comb = pd.concat(raw_data_comb,ignore_index = True)
		
		# store or save data
		if save:
			print('saving combined data file')
			raw_data_comb.to_excel(os.path.join(self.project_folder,'raw','raw_data_combined.xlsx'), sheet_name ='raw_data')		
		
		self.raw_data_comb = raw_data_comb

	def select_data(self, project_parameters = [], save = False):
		"""
		From data frame only include those columns that are specified in project_parameters. At the same time practice trials are omitted

		Arguments
		- - - - - 
		project_paraeters (list): column names of interest
		save (bool): If True, save a datafile with updated data
		
		"""

		# select relevant variables
		params = [p for p in project_parameters if p in self.raw_data_comb.keys()]
		params.sort()
		data_comb = self.raw_data_comb[params]

		# filter out logged practice trials
		try:
			data_comb = data_comb[data_comb.practice != 'yes']
		except:
			print('Data file does not contain practice trials. Check if correct')

		try: # temporary line to analyze CFS data for Cortex revision
			for idx in data_comb.index:
				data_comb.ix[idx,'color_cat_target'] = self.select_color_cat(ast.literal_eval(data_comb.ix[idx,'shapes'])['target'][0])
		except:
			print('??')

		# store or save data	
		if save:
			print('saving selected data')
			data_comb.to_excel(os.path.join(self.project_folder,'analysis','data_combined.xlsx'), sheet_name ='data_combined')		
		
		self.work_data = data_comb

	def filter_data(self, to_filter = ['RT'], filter_crit = ' and search_resp == 1', cnd_sel = False, min_cut_off = 200, max_cut_off = 5000, save = False):
		"""
	
		Creates a new column in the data frame with an RT_filter

		RT data is filtered per ANOVA cell (i.e. per subject per condition). Filter has a two step procedure:
		1. All RTs shorter than 250 ms and longer than 5000 ms are removed
		2. RTs shorter or longer than 2.5 SD from the mean are excluded (mean and SD are calculated per subject per condition)

		Arguments
		- - - - - 
		to_filter (list): list of column names for which an RT filter column will be added
		filter_crit (str): Adds any additional filter criteria (e.g. only use correct trials)
		cnd_sel (bool): specifies whether filter is done per (True) or across all (False) conditions
		min_cut_off (int): min cut-off for first filter step
		max_cut_off (int): max cut-off for first filter step
		save (bool): specifies whether excell file with new filter column is saved

		Returns
		- - - -

		"""
		
		work_data = self.work_data 

		for f, filt in enumerate(to_filter):
		
			# filter RTs step 1
			self.work_data['raw_filter'] = (work_data[filt] > min_cut_off) & (work_data[filt] < max_cut_off)

			# filter RTs step 2
			self.work_data['{}_filter'.format(filt)] = False # trial is not included in analysis unless it is set to True by the RT filter

			filter_list = []
			for sj in work_data['subject_nr'].unique():
				print 'filtering sj {}'.format(sj)
				# set basis filter
				base_filter = 'subject_nr == {} and raw_filter == True'.format(sj)
				base_filter += filter_crit

				# filtering done for each condition seperately	
				if cnd_sel:
					for labels in product(*self.factor_labels):
						for i in range(len(labels)):
							if isinstance(labels[i],str):
								current_filter = base_filter + ' and {} == \'{}\''.format(self.factor_headers[i],labels[i])	
							else:
								current_filter = base_filter + ' and {} == {}'.format(self.factor_headers[i],labels[i])
					
	 					# filter data based on current filter for this specific cell of the ANOVA
						current_data = self.work_data.query(current_filter)

						# use filter to set RT filter to True if it is within SD range for that specific condition
						self.SDtrimmer(current_data, filt)	
						#for index in current_data.index:
						#	if (work_data.ix[index,filt] >= current_data[filt].mean() - 2.5 * current_data[filt].std()) and (work_data.ix[index,filt] <= current_data[filt].mean() + 2.5 * current_data[filt].std()):
						#		work_data.ix[index,'{}_filter'.format(filt)] = True
				# filtering collapsed across conditions	
				else:
					current_data = self.work_data.query(base_filter)
					self.SDtrimmer(current_data, filt)
					#for index in current_data.index:
					#	if (work_data.ix[index,filt] >= current_data[filt].mean() - 2.5 * current_data[filt].std()) and (work_data.ix[index,filt] <= current_data[filt].mean() + 2.5 * current_data[filt].std()):
					#		work_data.ix[index,'{}_filter'.format(filt)] = True

		# store or save data					
		if save:
			print('saving filtered data')
			work_data.to_excel(os.path.join(self.project_folder,'analysis','data_comb_filter.xlsx'), sheet_name ='data_comb_filter')		
		
		self.work_data = work_data

	def SDtrimmer(self, df, filt, sd = 2.5):
		'''

		'''

		lower_bound = df[filt].mean() - 2.5 * df[filt].std()
		upper_bound = df[filt].mean() + 2.5 * df[filt].std()

		for index in df.index:
			if (self.work_data.ix[index,filt] >= lower_bound) and (self.work_data.ix[index,filt] <= upper_bound):
				self.work_data.ix[index,'{}_filter'.format(filt)] = True




		

	def exclude_outliers(self, criteria = dict(RT = "RT_filter == True"), agg_func = 'mean', sd = 2.5):
		'''

		Select outliers based on a SD critaria. PP with data that are more than the specified number of SD's (defualt = 2.5) from the 
		group mean are considered as outliers and removed from the data. 

		Arguments
		- - - - - 
		criteria (dict): Columns corresponding to the keys will be used for outlier selection. 
							If value of dict is not '', data will first be filtered to select subset of data 
							(e.g. only do outlier selection after applying RT filter)
		agg_func (str): summmary statistic for outlier selection (e.g mean/median/sum, etc) 
		sd (float): SD criteria 

		Returns
		- - - -
		
		'''

		for c in criteria.keys():

			if criteria[c] != "":
				outl_data = self.work_data.query(criteria[c])
			else:
				outl_data = self.work_data

			pivot = outl_data.pivot_table(values = c, index = 'subject_nr', columns = self.factor_headers, aggfunc = agg_func)
			self.outlier = np.hstack((self.outlier,self.select_outliers(pivot.values, pivot.index, sd = sd)))				

		self.outlier = np.unique(self.outlier)	
		for i in self.outlier:
			self.work_data = self.work_data[self.work_data['subject_nr'] != i]

		with open(os.path.join(self.project_folder,'analysis','Outlier.txt'), 'w') as text_file:
			text_file.write('These subjects ({}) have been selected as outliers based on a {} sd criteria'.format(self.outlier, sd))

	def select_outliers(self, X, subjects, sd = 2.5):
		"""

		helper function of select_outliers that does the actual work

		Arguments
		- - - - - 
		X (array): array of data (subjects x conditions)
		subjects (array): array of subject numbers
		sd (float): SD criteria 

		Returns
		- - - -

		outliers (array): selected outliers based on SD criteria
		
		"""

		cut_off = [X.mean() + i * sd * X.mean(axis = 1).std() for i in [-1,1]]
		excl = np.logical_or(X.mean(axis = 1)<cut_off[0],X.mean(axis = 1)>cut_off[1])
		outliers = np.array(subjects)[excl]

		return outliers 	
		

	def prep_JASP(self, agg_func = 'mean', voi = 'RT',rows = 'subject_nr', data_filter = "", save = True):	
		"""
		Returns a pivot table with voi as dependent variable
		"""	


		# To create a filtered pivot table in python, unlike in excell, we need to filter data before creating the pivot
		if data_filter != "":
			pivot_data = self.work_data.query(data_filter)
		else:
			pivot_data = self.work_data

		# Create pivot table and extract individual headers for .csv file (input to JASP)
		pivot = pivot_data.pivot_table(values = voi, index = rows, columns = self.factor_headers, aggfunc = agg_func)
		headers = ['sj'] + ['_'.join(np.array(labels,str)) for labels in product(*self.factor_labels)]
		p_values = np.hstack((pivot.index.values.reshape(-1,1), np.zeros(pivot.shape)))
		for i, labels in enumerate(product(*self.factor_labels)):
			p_values[:,i + 1] = pivot[labels]

		if save:
			np.savetxt(os.path.join(self.project_folder,'analysis', '{}_JASP.csv'.format(voi)), p_values, delimiter = "," ,header = ",".join(headers), comments='')


	def save_data_file(self):
		'''

		'''

		self.work_data.to_csv(os.path.join(self.project_folder,'analysis','preprocessed.csv'))

	def select_color_cat(self, color):
		'''
		Function that takes the correct color from a shape dictionary (for CFS study Cortex)
		'''


		color_dict = {
			'red' : ['#EF1E52','#E43756','#D84659','#EF213F','#E43845','#D9474B','#ED2B2A','#E33C33','#D8493C'],
			'green' : ['#5B8600','#618427','#68823F','#47891D','#528636','#5D8349','#2E8B32','#448843','#548551'],
			'blue'  : ['#0079EA','#2A79DA','#4179CB','#5A6FE6','#5F71D8','#6473CA','#6B6CE3','#6D6FD6','#6F71C8'],
			'yellow': ['#FEBE25','#F8C04A','#F2C165','#F5C208','#F0C342','#EAC560','#ECC200','#E5C739','#E1C85A'],
			'purple': ['#C241D6','#AE55C1','#9E66A1','#CF3CC8','#BF4EB6','#AC609A','#D834BB','#C946AE','#BC569F'],
			}

		if color in color_dict['red']:
			color_cat = 'red'
		elif color in color_dict['green']:
			color_cat = 'green'		
		elif color in color_dict['blue']:
			color_cat = 'blue'	
		elif color in color_dict['yellow']:
			color_cat = 'yellow'
		elif color in color_dict['purple']:
			color_cat = 'purple'
				
		return color_cat	
			


	def congruency_filter(self, save = False):	
		"""
		Filter adds an extra column to work_data. Value is True if distractor and target have the same orientation.
		"""

		# to be able to loop over index we have to reset the indices (account for filtered data)
		work_data = self.work_data.reset_index(drop = True)
		work_data['congruency_filter'] = False
		for index in work_data['congruency_filter'].index[:]:
			target = work_data.ix[index,'target_loc_int']
			dist = work_data.ix[index,'dist_loc_int']
			if eval(work_data.ix[index,'target_list'])[target] == eval(work_data.ix[index,'target_list'])[dist]: 
				work_data.ix[index,'congruency_filter'] = True

		if save:
			work_data.to_excel(os.path.join(self.project_folder,'analysis','data_comb_filter.xlsx'), sheet_name ='data_comb_filter')	

		self.work_data = work_data

	def bin_locations_set_size(self, save = True):
		'''
		'''	

		# to be able to loop over index we have to reset the indices (account for filtered data)
		work_data = self.work_data.reset_index(drop = True)
		work_data['dist'] = 'NA'

		for idx in work_data['bin'].index[:]:
			d = abs(work_data.ix[idx,'target_loc'] - work_data.ix[idx,'dist_loc'])
			if work_data.ix[idx,'set_size'] == 4:
				if d > 2:
					d = 1
			elif work_data.ix[idx,'set_size'] == 8:	
				if d > 4:
					d -= (d-4)*2
			work_data.ix[idx,'dist'] = d

		if save:
			work_data.to_excel(os.path.join(self.project_folder,'analysis','data_comb_filter_dist.xlsx'), sheet_name ='data_comb_filter_dist')	

		self.work_data = work_data	

	def post_error_filter(self, save = False):
		"""
		Filter adds an extra column to work_data. Value is True if memory response on n-1 trial is incorrect and False if memory response on n-1 trial
		is correct
		"""

		# to be able to loop over index we have to reset the indices (account for filtered data)
		work_data = self.work_data.reset_index(drop = True)
		work_data['PE_filter'] = False

		for index in work_data['PE_filter'].index[1:]:
			# check memory reponse on n-1 trial and check whether trial n is not the start of a new block
			if work_data.ix[index - 1,'memory_resp'] == 0 and work_data.ix[index - 1,'block_count'] == work_data.ix[index,'block_count']:
				work_data.ix[index,'PE_filter'] = True

		if save:
			work_data.to_excel(os.path.join(self.project_folder,'analysis','data_comb_filter.xlsx'), sheet_name ='data_comb_filter')	

		self.work_data = work_data







