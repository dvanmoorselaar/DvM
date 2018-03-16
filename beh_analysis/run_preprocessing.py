import os

from IPython import embed 
from PreProcessing import PreProcessing


project = 'Dist_suppr'
part = 'DT_sim'
factors = ['block_type','dist_high']
labels = [['DTsim','DTdisU','DTdisP'],['yes','no']]
to_filter = ['RT'] 

project_parameters = ['subject_nr','subject_parity','condition','cue','invalid_cue_move','degrees_response',
						'deviation', 'level_staircase','location_bin_mem','location_bin_search','memory_orient',
						'missed_resp','number','trigger_info','correct_response','test_location','test_rotation',
						'direction','load','reward','block_type','trial_type','block_count','block_cnt','RT_search','practice',
						'nr_trials','target_loc_int','target_loc','target_list','dist_loc_int','dist_loc', 'search_resp',
						'search_response','suppr_loc','suppression','memory_response','memory_resp','points','resp_test',
						'correct_resp','value','shapes','repetition','set_size','RT','acc','cue_validity','stim', 'correct',
						'RT_0','RT_1','RT_2','RT4','RT5','RT6','cue_correct','cue_resp_pos','nr_correct','onset','onset_correct',
						'onset_resp_pos','onset_test_correct','resp1','resp2','resp3','resp4','resp5','resp6','color_pool_trial',
						'dist_orient','target_orient','cue_colour','target_stimulus', 'suppressed','dev_0','dev_1','dev_2', 'dist_high', 'target_high',
						'correct_0','deg_0']

def standardPreProcessing(project, part, factors, labels, to_filter):
	'''
	Function executes standard PreProcessing steps
	'''

	# execute standard PreProcessing Pipeline
	
	PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
	PP.create_folder_structure()
	PP.combine_single_subject_files(save = False)
	PP.select_data(project_parameters = project_parameters, save = False)
	PP.filter_data(to_filter = to_filter, filter_crit = ' and correct == 1', cnd_sel = False, save = True)
	PP.exclude_outliers(criteria = dict(RT = 'RT_filter == True', correct = ''))
	PP.prep_JASP(agg_func = 'mean', voi = 'RT', data_filter = "RT_filter == True", save = True)
	PP.save_data_file()

if __name__ == '__main__':

	os.chdir('/home/dvmoors1/big_brother') 
	standardPreProcessing(project, part, factors, labels, to_filter)

					


