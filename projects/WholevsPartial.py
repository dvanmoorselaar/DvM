import sys
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
sys.path.append('/Users/dvm/DvM')

import seaborn as sns

from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
#from eeg_analyses.BDM import * 
from eeg_analyses.TF import * 
from support.FolderStructure import *
from support.support import *
from stats.nonparametric import *
from visuals.visuals import MidpointNormalize

# subject specific info
sj_info = {'1': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
		  	'2': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
		  	'3': {'tracker': (False, '', None, '',0), 'replace':{}},
		  	'4': {'tracker': (False, '', None, '',0), 'replace':{}}, # 2 sessions (headache during recording in session 1)
		  	'5': {'tracker': (False, '', None, '',0), 'replace':{}},
		  	'6': {'tracker': (False, '', None, '',0), 'replace':{}}, #  first trial is spoke trigger, because wrong experiment was started
		  	'7': {'tracker': (False, '', None, '',0), 'replace':{}},
		  	'8': {'tracker': (False, '', None, '',0), 'replace':{}},
		  	'9': {'tracker': (False, '', None, '',0), 'replace':{}}, # recording started late (missing trials are removed from behavior file)
		  	'10': {'tracker': (False, '', None, '',0), 'replace':{}},
		  	'11': {'tracker': (False, '', None, '',0), 'replace':{}},
		  	'12': {'tracker': (False, '', None, '',0), 'replace':{}},
		  	'13': {'tracker': (True, 'asc', 500, 'Onset memory',0.8), 'replace':{}},
		  	'14': {'tracker': (True, 'asc', 500, 'Onset memory',0.8), 'replace':{}},
			'15': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			'16': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}}, 
			'17': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			'18': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			'19': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			'20': {'tracker': (False, '', None, '',0), 'replace':{}},# final beh trials are missing??????
			'21': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			'22': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			'23': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			'24': {'tracker': (True, 'asc', 500, 'Onset cue',0), 'replace':{}},
			} 

# project specific info
# Behavior
project = 'wholevspartial'
part = 'beh'
factors = ['block_type','cue']
labels = [['whole','partial'],['cue','no']]
to_filter = [] 
project_param = ['practice','nr_trials','trigger','condition',
				'block_type', 'cue','cue_loc','dev_0','dev_1','dev_2',
		        'correct_0','correct_1','correct_2','deg_0','deg_1','deg_2',
		        'test_order','points','subject_nr']

# EEG
nr_sessions = 1
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
event_id = [10,11,12,19,20,21,22,29]
t_min = -0.5 # offset memory
t_max = 0.85 # onset test display
flt_pad = 0.5
eeg_runs = [1]
binary = 3840

# eye tracker info
tracker_ext = 'asc'
eye_freq = 500
start_event = 'Onset memory'
tracker_shift = 0#-0.8
viewing_dist = 80 
screen_res = (1680, 1050) 
screen_h = 29

class WholevsPartial(FolderStructure):

	def __init__(self): pass

	def prepareBEH(self, project, part, factors, labels, project_param):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		#PP.filter_data(to_filter = to_filter, filter_crit = ' and correct == 1', cnd_sel = False, save = True)
		#PP.exclude_outliers(criteria = dict(dev_0 = ''))
		#PP.prep_JASP(agg_func = 'mean', voi = 'dev_0', data_filter = "", save = True)
		PP.save_data_file()

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, event_id, project_param, project_folder, binary, channel_plots, inspect):
		'''
		EEG preprocessing as preregistred @ https://osf.io/b2ndy/register/5771ca429ad5a1020de2872e
		'''

		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']
		log_file = self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocessing_param.csv')

		# start logging
		logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename= self.FolderTracker(extension=['processed', 'info'], 
                        filename='preprocess_sj{}_ses{}.log'.format(
                        sj, session), overwrite = False),
                    filemode='w')
		

		# READ IN RAW DATA, APPLY REREFERENCING AND CHANGE NAMING SCHEME
		EEG = mne.concatenate_raws([RawBDF(os.path.join(project_folder, 'raw', file + '{}.bdf'.format(run)),
		                                  preload=True, eog=eog) for run in eeg_runs])

		#EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['V_do','H_l','Ref_r','Ref_l','EXG7','EXG8'])
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEG_ica = EEG.copy()
		EEG.filter(h_freq=None, l_freq=0.01, fir_design='firwin',
		            skip_by_annotation='edge')
		EEG_ica.filter(h_freq=None, l_freq=1, fir_design='firwin',
		            skip_by_annotation='edge')		

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(event_id, binary=binary, min_duration=0)
		if sj == 6:
			events = np.delete(events,3,0) # delete spoke trigger
		beh, missing = EEG.matchBeh(sj, session, events, event_id, 
		                             headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=event_id,
		        tmin=t_min, tmax=t_max, baseline=None, flt_pad = flt_pad, reject_by_annotation = True) 
		epochs_ica = Epochs(sj, session, EEG_ica, events, event_id=event_id,
		        tmin=t_min, tmax=t_max, baseline=(None,None), flt_pad = flt_pad, reject_by_annotation = True)

		# AUTMATED ARTIFACT DETECTION
		epochs.selectBadChannels(run_ransac = True, channel_plots = False, inspect = True, RT = None)  
		z = epochs.artifactDetection(z_thresh=4, band_pass=[110, 140], plot=True, inspect=True)

		# ICA
		epochs.applyICA(EEG, EEG_ica, method='picard', fit_params = dict(ortho=False, extended=True), inspect = True)

		# EYE MOVEMENTS
		epochs.detectEye(missing, events, beh.shape[0], time_window=(t_min*1000, t_max*1000), 
						tracker_shift = tracker_shift, start_event = start_event, 
						extension = tracker_ext, eye_freq = eye_freq, 
						screen_res = screen_res, viewing_dist = viewing_dist, 
						screen_h = screen_h)

		# INTERPOLATE BADS
		bads = epochs.info['bads']   
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, event_id)

		logPreproc((sj, session), log_file, nr_sj = len(sj_info.keys()), nr_sessions = nr_sessions, 
					to_update = dict(nr_clean = len(epochs), z_value = z, nr_bads = len(bads), bad_el = bads))

	def visualMarkSaccades(self, sj, overwrite_tracker = False):
		'''
		Visual inspection of saccades detected by automatic step algorythm
		'''

		# read in data
		beh, eeg = PO.loadData(sj, 'ses-1', False, (-0.5,0.85),'HEOG', 1,
		 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = False)
		
		# crop data to time window of interest to inspect eyemovements
		eeg.crop(tmin = -0.5, tmax = 0.8)
		
		# apply baseline correction for visualization purposes
		eeg.apply_baseline((None,None))

		# get eog data of interest
		eog = eeg._data[:,eeg.ch_names.index('HEOG')]
		eye_idx = eog_filt(eog, eeg.info['sfreq'], windowsize = 200, windowstep = 10, threshold = 20)	

		# inspect eog channels
		bad_eogs = eeg[eye_idx]
		idx_bads = bad_eogs.selection
		bad_eogs.plot(block=True, n_epochs=5, n_channels=2, picks='eog', scalings='auto')

		missing = np.array([list(idx_bads).index(idx) for idx in idx_bads if idx not in bad_eogs.selection])
		eye_idx = np.delete(eye_idx, missing)

		if beh['eye_bins'].isnull().values.any() or overwrite_tracker:
			beh['eye_bins'] = 0  

		beh['eye_bins'][eye_idx] = 99
		pickle.dump(beh,open(self.FolderTracker(extension=['beh', 'processed'],
            	filename='subject-{}_ses-1.pickle'.format(sj)),'wb'))

	def eyePositionControl(self, sj, eeg, beh, baseline = (-0.2,0)):
		'''
		calculate systematic eye bias (as indexed via HEOG) on lateralized cue trials
		'''

		base_idx = [np.argmin(abs(eeg.times - t)) for t in baseline]
		base_idx = slice(base_idx[0], base_idx[1]+1)

		l_idx = np.where(beh.cue_loc == '1')
		r_idx = np.where(beh.cue_loc == '2')

		cnd_heog = {'partial':[], 'whole':[]}
		cnd_heog['times'] = eeg.times

		# analyze eye bias seperate for partial and whole conditions
		for cnd in ['partial', 'whole']:
			l_idx = np.where((beh.cue_loc == '1') & (beh.block_type == cnd))[0]
			r_idx = np.where((beh.cue_loc == '2') & (beh.block_type == cnd))[0]
			
			# get HEOG data of interest
			left = np.mean(eeg._data[l_idx,eeg.ch_names.index('HEOG') ], axis = 0)  
			right = np.mean(eeg._data[r_idx,eeg.ch_names.index('HEOG') ] * -1, axis = 0) 

			# average left and right trials and baseline correct
			avg = np.stack((left, right)).mean(axis = 0)
			avg_base = avg - avg[base_idx].mean()   
			cnd_heog[cnd] = avg_base

		pickle.dump(cnd_heog,open(self.FolderTracker(extension=['eye', 'cue_control'],
            	filename='subject-{}_heog.pickle'.format(sj)),'wb'))



	# analyze BEHAVIOR
	def behExp1(self):
		'''

		'''

		# read in data
		file = self.FolderTracker(['beh-exp1','analysis'], filename = 'preprocessed.csv')
		data = pd.read_csv(file)
		data['dev_1'][data['dev_1'] == 'None'] = np.nan
		data['dev_1'] = data.dev_1.astype(float) 
		data['dev_2'][data['dev_2'] == 'None'] = np.nan
		data['dev_2'] = data.dev_2.astype(float)        

		# create pivot (main analysis)
		pivot = data.pivot_table(values = 'dev_0', index = 'subject_nr', columns = ['condition','set_size','cue'], aggfunc = 'mean')
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())

		# plot whole and partial in seperate plots
		plt.figure(figsize = (30,10))
		ax = plt.subplot(1,2, 1, title = 'Response 1', ylabel = 'raw error (deg.)', ylim = (25,70))
		for i, load in enumerate([3,5]):
			for idx, cnd in enumerate(['partial', 'whole']):
				pivot[cnd][load].mean().plot(color = ['red','green'][i], 
					ls = ['-','--'][idx],label = '{}-{}'.format(cnd,load), yerr = pivot_error[cnd][load])
			sns.despine(offset=50, trim = False)	
			plt.legend(loc = 'best')
		
		ax = plt.subplot(1,2, 2, title = 'Response 2-3', ylabel = 'raw error (deg.)', ylim = (45,90))
		for r, resp in enumerate(['dev_1', 'dev_2']):
			pivot = data.pivot_table(values = resp, index = 'subject_nr', columns = ['condition','set_size','cue'], aggfunc = 'mean')
			pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())
			for i, load in enumerate([3,5]):
				pivot['whole'][load].mean().plot(color = ['red','green'][r], 
						ls = ['-','--'][i],label = '{}-{}'.format(['R1','R2'][r],load), yerr = pivot_error['whole'][load])
		sns.despine(offset=50, trim = False)	
		plt.legend(loc = 'best')

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['beh-exp1','analysis'], filename = 'beh-main.pdf'))
		plt.close()

if __name__ == '__main__':
	
	# Specify project parameters
	project_folder = '/Users/dvm/Desktop/Whole_partial'
	os.chdir(project_folder)
	PO =  WholevsPartial()

	# behavior analysis
	#PO.prepareBEH(project, 'beh', ['condition','cue'], [['whole','partial'],['cue','no']], project_param)
	#PO.prepareBEH(project, 'beh-exp1', ['condition','cue','set_size'], [['whole','partial'],['cue','no'],[3,5]], project_param + ['set_size'])

	#PO.visualMarkSaccades(12, overwrite_tracker = False)

	# PO.prepareEEG(sj = 3, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
	#    		t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
	#    		event_id = event_id, project_param = project_param, 
	#    		project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)

	for sj in range(1,25):
		
		# read in data
		beh, eeg = PO.loadData(sj, 'ses-1',True, (-0.4,0.85),'HEOG', 1,
		 		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = True)

		# eyemovent control
		PO.eyePositionControl(sj, eeg, beh)

		# # CDA analysis
		# erp = ERP(eeg, beh, 'cue_loc', (-0.2,0))
		# erp.selectERPData(time = [-0.2, 0.85], h_filter = 6, excl_factor = None) 
		# erp.topoFlip(left = ['1'], header = 'cue_loc')
		# erp.ipsiContra(sj = sj, left = ['1'], right = ['2'], l_elec = ['P7'], r_elec = ['P8'], 
		#  				conditions = ['partial','whole'], cnd_header = 'block_type', 
		#  				midline = None, erp_name = 'cda', RT_split = False)

		# CDA analysis (topoplot left minus right cue)
		# read in data
		# beh, eeg = PO.loadData(sj, 'ses-1',True, (-0.4,0.85),'HEOG', 1,
		#  		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = True)

		# erp = ERP(eeg, beh, 'cue_loc', (-0.2,0), flipped = True)
		# erp.selectERPData(time = [-0.2, 0.85], h_filter = 6, excl_factor = None) 
		# left trials
		# erp.ipsiContra(sj = sj, left = ['1'], right = [], l_elec = ['P7'], r_elec = ['P8'], 
		#  				conditions = ['partial','whole'], cnd_header = 'block_type', 
		#  				midline = None, erp_name = 'cda_left', RT_split = False)

		# beh, eeg = PO.loadData(sj, 'ses-1',True, (-0.4,0.85),'HEOG', 1,
		#  		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = True)	
		
		# right trials	
		# erp.ipsiContra(sj = sj, left = [], right = ['2'], l_elec = ['P7'], r_elec = ['P8'], 
		#  				conditions = ['partial','whole'], cnd_header = 'block_type', 
		#  				midline = None, erp_name = 'cda_right', RT_split = False)				

		# # BDM analysis
		#bdm = BDM('cue_loc', nr_folds = 10, eye = False)
		#bdm.Classify(sj, cnds = ['partial','whole'], cnd_header = 'block_type', bdm_labels = ['0','1','2'], time = (-0.5, 0.85), nr_perm = 0, gat_matrix = True)

		# cue trials (whole vs partial)
		#bdm = BDM('block_type', nr_folds = 10, eye = False)
		#bdm.Classify(sj, cnds = 'all', cnd_header = 'block_type', bdm_labels = ['partial','whole'], factor = dict(cue = ['cue']), time = (-0.5, 0.85), nr_perm = 0, bdm_matrix = True)

		# no-cue trials (whole vs partial)
		#bdm = BDM('block_type', nr_folds = 10, eye = False)
		#bdm.Classify(sj, cnds = 'all', cnd_header = 'block_type', bdm_labels = ['partial','whole'], factor = dict(cue = ['no']), time = (-0.5, 0.85), nr_perm = 0, bdm_matrix = True)

		# read in data
		# beh, eeg = PO.loadData(sj, 'ses-1',True, (-0.4,0.85),'HEOG', 1,
		#  		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = True)

		# # do TF analysis
		# tf = TF(beh, eeg, laplacian = False)
		# tf.TFanalysis(sj, cnds = ['whole','partial'], cnd_header = 'block_type', base_type = 'conspec', min_freq = 4, max_freq = 40,
		# 			num_frex = 18, time_period = (-0.4,0.85), base_period = (-0.4,-0.1), elec_oi = 'all', method = 'wavelet', 
		# 			flip = dict(cue_loc = ['1']), downsample = 4, tf_name = 'no_lapl_4-40')


	
		# read in preprocessed data
		#beh, eeg = PO.loadData(sj, True, (-0.2,0.85),'HEOG', 1,
		#		 eye_dict = dict(windowsize = 200, windowstep = 10, threshold = 20), use_tracker = True)
		# do TF analysis
		#tf = TF(beh, eeg, laplacian=False)
		#tf.TFanalysis(sj, cnds = ['partial','whole'], cnd_header = 'block_type', base_period = (-0.2,0), base_type = 'conspec',
		# 			time_period = (-0.2,0.85), tf_name = 'cue_power', elec_oi = 'all', method = 'wavelet', flip = dict(cue_loc = ['1']), factor = dict(cue_loc = ['None','0']))

	#PO.behExp1() 
	#PO.plotCDA()
	#PO.bdmBlock()
	#PO.bdmCue()
	#PO.plotTF()





