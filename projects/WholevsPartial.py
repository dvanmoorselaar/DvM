import matplotlib          # run these lines only when running sript via ssh connection
matplotlib.use('agg')

import sys
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import seaborn as sns

from IPython import embed
from beh_analyses.PreProcessing import *
from eeg_analyses.EEG import * 
from eeg_analyses.ERP import * 
from eeg_analyses.BDM import * 
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
		  	'9': {'tracker': (False, '', None, '',0), 'replace':{}}, # recodring started late (missing trials are removed from behavior file)
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
			'20': {'tracker': (False, '', None, '',0), 'replace':{}},
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
eog =  ['V_up','V_do','H_r','H_l']
ref =  ['Ref_r','Ref_l']
trigger = [10,11,12,19,20,21,22,29]
t_min = -0.5
t_max = 0.85
flt_pad = 0.5
eeg_runs = [1]
binary = 3840

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

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

	def prepareEEG(self, sj, session, eog, ref, eeg_runs, t_min, t_max, flt_pad, sj_info, trigger, project_param, project_folder, binary, channel_plots, inspect):
		'''
		EEG preprocessing as preregistred @ https://osf.io/b2ndy/register/5771ca429ad5a1020de2872e
		'''

		# set subject specific parameters
		file = 'subject_{}_session_{}_'.format(sj, session)
		replace = sj_info[str(sj)]['replace']
		tracker, ext, t_freq, start_event, shift = sj_info[str(sj)]['tracker']

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
		                                   montage=None, preload=True, eog=eog) for run in eeg_runs])

		#EEG.replaceChannel(sj, session, replace)
		EEG.reReference(ref_channels=ref, vEOG=eog[
		                :2], hEOG=eog[2:], changevoltage=True, to_remove = ['V_do','H_l','Ref_r','Ref_l','EXG7','EXG8'])
		EEG.setMontage(montage='biosemi64')

		#FILTER DATA TWICE: ONCE FOR ICA AND ONCE FOR EPOCHING
		EEGica = EEG.filter(h_freq=None, l_freq=1,
		                   fir_design='firwin', skip_by_annotation='edge')
		EEG.filter(h_freq=None, l_freq=0.1, fir_design='firwin',
		            skip_by_annotation='edge')

		# MATCH BEHAVIOR FILE
		events = EEG.eventSelection(trigger, binary=binary, min_duration=0)
		if sj == 6:
			events = np.delete(events,3,0) # delete spoke trigger
		beh, missing = EEG.matchBeh(sj, session, events, trigger, 
		                             headers = project_param)

		# EPOCH DATA
		epochs = Epochs(sj, session, EEG, events, event_id=trigger,
		        tmin=t_min, tmax=t_max, baseline=(None, None), flt_pad = flt_pad) 

		# ARTIFACT DETECTION
		epochs.selectBadChannels(channel_plots = False, inspect=False, RT = None)    
		epochs.artifactDetection(inspect=False, run = False)

		# ICA
		epochs.applyICA(EEGica, method='extended-infomax', decim=3, inspect = True)

		# EYE MOVEMENTS
		self.binEye(epochs, missing, time_window=(t_min*1000, t_max*1000), tracker = tracker, tracker_shift = shift, start_event = start_event, extension = ext, eye_freq = t_freq)

		# INTERPOLATE BADS
		epochs.interpolate_bads(reset_bads=True, mode='accurate')

		# LINK BEHAVIOR
		epochs.linkBeh(beh, events, trigger)

	def anticipatoryEye(self, sj_info, nr_bins, start, end):
		'''

		'''
		
		# initiate SGD class and EYE class
		SD = SaccadeGlissadeDetection(500.0)
		EO = EYE(sfreq = 500.0)
		# create timing bins
		bins = np.linspace(start,end, nr_bins)

		sac_info = {'cue':[],'no':[]}
		beh_info = {'cue':[],'no':[]}

		for sj in sj_info.keys():
			
			if not sj_info[sj]['tracker'][0] or not sj_info[sj]['tracker'][3] == 'Onset cue': continue

			eye_file = self.FolderTracker(extension = ['eye','raw'], \
					filename = 'sub_{}_session_1.asc'.format(sj))
			beh_file = self.FolderTracker(extension = ['beh','raw'], \
										filename = 'subject-{}_ses_1.csv'.format(sj))

			eye, beh = EO.readEyeData(sj, [eye_file], [beh_file])
			x, y, times = EO.getXY(eye, start = start, end = end, start_event = sj_info[sj]['tracker'][3])	
			x, y = EO.setXY(x,y, times, (-200,0))
			# loop over all trials
			for i, (x_, y_) in enumerate(zip(x,y)):
				info = SD.detectEvents(x_, y_, output = 'dict')
				
				if info[0] not in  [{},np.nan]:
					# loop over saccades
					for s in info[0].keys():
						sac_info[beh['cue'].values[i]].append(times[info[0][s][0]: info[0][s][1]].mean())	
						beh_info[beh['cue'].values[i]].append(beh['block_type'].values[i])
		
		# bin data for whole and partial blocks seperately
		plt.figure(figsize = (30,10))
		x = np.arange(bins.size)
		width = 0.2
		# set informative x labels
		x_label = x[[np.argmin(abs(bins - t)) for t in (0,100)]]

		for idx, cue in enumerate(['cue','no']):
			bin_nr = np.digitize(sac_info[cue], bins)
			binned = np.array([(sum((np.array(beh_info[cue])[bin_nr == b] == 'partial')),
				  sum((np.array(beh_info[cue])[bin_nr == b] == 'whole'))) for b in range(bins.size)])
			ax = plt.subplot(1,2 , idx + 1, title = cue, ylim = (0,180))
			plt.bar(x, binned[:,0], width, color = 'green', label = 'partial')	
			plt.bar(x +width, binned[:,1], width, color = 'red', label = 'whole')
			ax.set_xticks(x_label)
			ax.set_xticklabels(['cue \n on','cue \n off'])
			sns.despine(offset=50, trim = False)	
			plt.legend(loc = 'best')

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['eye','analysis'], filename = 'anticipatory1.pdf'))
		plt.close()

		





	def binEye(self, EEG, missing, time_window, threshold=30, windowsize=50, windowstep=25, channel='HEOG', tracker = True, tracker_shift = 0, start_event = '', extension = 'asc', eye_freq = 500):
		'''
		Marking epochs containing step-like activity that is greater than a given threshold

		Arguments
		- - - - -
		self():
		EEG(object): Epochs object
		missing
		time_window (tuple): start and end time in seconds
		threshold (int): range of amplitude in microVolt
		windowsize (int): total moving window width in ms. So each window's width is half this value
		windowsstep (int): moving window step in ms
		channel (str): name of HEOG channel
		tracker (boolean): is tracker data reliable or not
		tracker_shift (float): specifies difference in ms between onset trigger and event in eyetracker data
		start_event (str): marking onset of trial in eyetracker data
		extension (str): type of eyetracker file (now supports .asc/ .tsv)
		eye_freq (int): sampling rate of the eyetracker


		Returns
		- - - -

		'''
        
		sac_epochs = []

		# CODE FOR HEOG DATA
		idx_ch = EEG.ch_names.index(channel)
		idx_s, idx_e = tuple([np.argmin(abs(EEG.times - t))
		                      for t in time_window])
		windowstep /= 1000 / EEG.info['sfreq']
		windowsize /= 1000 / EEG.info['sfreq']

		for i in range(len(EEG)):
			up_down = 0
			for j in np.arange(idx_s, idx_e - windowstep, windowstep):

				w1 = np.mean(EEG._data[i, idx_ch, int(
					j):int(j + windowsize / 2) - 1])
				w2 = np.mean(EEG._data[i, idx_ch, int(
					j + windowsize / 2):int(j + windowsize) - 1])

			if abs(w1 - w2) > threshold:
				up_down += 1
			if up_down == 2:
				sac_epochs.append(i)
				break

		logging.info('Detected {0} epochs ({1:.2f}%) with a saccade based on HEOG'.format(
				len(sac_epochs), len(sac_epochs) / float(len(EEG)) * 100))

		# read in epochs removed via artifact detection
		if os.path.exists(self.FolderTracker(extension=[
		                        'preprocessing', 'subject-{}'.format(EEG.sj), EEG.session], 
		                        filename='noise_epochs.txt')):
			noise_epochs = np.loadtxt(self.FolderTracker(extension=[
		                        'preprocessing', 'subject-{}'.format(EEG.sj), EEG.session], 
		                        filename='noise_epochs.txt'))
		else:
			noise_epochs = np.array([])    

		# do binning based on eye-tracking data
		if tracker:
			# CODE FOR EYETRACKER DATA 
			EO = EYE(sfreq = eye_freq)
			# correct timings for subjects without correct eyetracker log files
			s_time = (EEG.tmin + EEG.flt_pad) * 1000
			e_time = (EEG.tmax - EEG.flt_pad) * 1000
			drift_correct = (-200,0)
			if start_event == 'Onset memory':
				s_time += tracker_shift * 1000
				e_time += tracker_shift * 1000
				drift_correct = (600,800)
			eye_bins, trial_nrs = EO.eyeBinEEG(EEG.sj, int(EEG.session), 
		                        int(s_time), int(e_time), drift_correct = drift_correct, start_event = start_event, extension = extension)
		else:
			eye_bins = np.array([])    

		# correct for missing data (if eye recording is stopped during experiment)
		if eye_bins.size > 0 and eye_bins.size < EEG.nr_events:
			# create array of nan values for all epochs (UGLY CODING!!!)
			temp = np.empty(EEG.nr_events) * np.nan
			temp[trial_nrs - 1] = eye_bins
			eye_bins = temp

			temp = (np.empty(EEG.nr_events) * np.nan)
			temp[trial_nrs - 1] = trial_nrs
			trial_nrs = temp
		elif eye_bins.size == 0 or tracker == False:
			eye_bins = np.empty(EEG.nr_events + missing.size) * np.nan
			trial_nrs = np.arange(EEG.nr_events + missing.size) + 1

		# remove trials that are not present in bdf file, if any
		miss_mask = np.in1d(trial_nrs, missing, invert = True)
		eye_bins = eye_bins[miss_mask]      

		# remove trials that have been deleted from eeg 
		eye_bins = np.delete(eye_bins, noise_epochs)
		logging.info('Detected {0} bins ({1} with data) based on tracker ({2:.2f} > 1 degree)'.
		            format(eye_bins.size, (~np.isnan(eye_bins)).sum(), sum(eye_bins > 1) / float(len(EEG)) * 100))

		# save array of deviation bins    
		np.savetxt(self.FolderTracker(extension=['preprocessing', 'subject-{}'.format(
		    EEG.sj), EEG.session], filename='eye_bins.txt'), eye_bins)	

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


	def cdaTopo(self):
		'''

		'''	

		# read in cda data
		with open(self.FolderTracker(['erp','cue_loc'], filename = 'topo_cda.pickle') ,'rb') as handle:
			topo = pickle.load(handle)

		with open(self.FolderTracker(['erp','cue_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		
		for idx, cnd in enumerate(['partial-cue', 'whole-cue']):
			plt.figure(figsize = (30,10))
			T = np.mean(np.stack([topo[t][cnd] for t in topo]), axis = 0)

			embed()

	def plotCDA(self):
		
		# read in cda data
		with open(self.FolderTracker(['erp','cue_loc'], filename = 'cda.pickle') ,'rb') as handle:
			erp = pickle.load(handle)

		with open(self.FolderTracker(['erp','cue_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		plt.figure(figsize = (30,10))
		diff = []
		for idx, cnd in enumerate(['partial-cue', 'whole-cue']):
			ax =  plt.subplot(1,3, idx + 1, title = cnd, ylabel = 'mV', xlabel = 'time (ms)')
			ipsi = np.mean(np.stack([erp[e][cnd]['ipsi'] for e in erp]), axis = 1)
			contra = np.mean(np.stack([erp[e][cnd]['contra'] for e in erp]), axis = 1)
			diff.append(contra - ipsi)
			err_i, ipsi  = bootstrap(ipsi)	
			err_c, contra  = bootstrap(contra)	

			plt.ylim(-6,6)
			plt.axhline(y = 0, color = 'black')
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'red')
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], ipsi + err_i, ipsi - err_i, alpha = 0.2, color = 'red')	
			plt.fill_between(info['times'], contra + err_c, contra - err_c, alpha = 0.2, color = 'green')	

			plt.legend(loc = 'best')
			sns.despine(offset=50, trim = False)

		ax =  plt.subplot(1,3, 3, title = 'cda', ylabel = 'mV', xlabel = 'time (ms)')
		for i, cnd in enumerate(['partial','whole']):
			err, x  = bootstrap(diff[i])
			plt.plot(info['times'], x, label = cnd, color = ['red','green'][i])
			plt.fill_between(info['times'], x + err, x - err, alpha = 0.2, color = ['red','green'][i])
			self.clusterPlot(diff[i], 0, 0.05, info['times'], ax.get_ylim()[0] + 0.05*(i+1), color = ['red','green'][i])
		plt.axhline(y = 0, color = 'black')

		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)	

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','cue_loc'], filename = 'cda.pdf'))
		plt.close()

	def clusterPlot(self, X1, X2, p_val, times, y, color, ls = '-'):
		'''
		plots significant clusters in the current plot
		'''	

		# indicate significant clusters of individual timecourses
		sig_cl = clusterBasedPermutation(X1, X2, p_val = p_val)
		mask = np.where(sig_cl < 1)[0]
		sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
		for cl in sig_cl:
			plt.plot(times[cl], np.ones(cl.size) * y, color = color, ls = ls)

	def plotTF(self):
		'''

		'''

		with open(self.FolderTracker(['tf','wavelet'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	

		files = glob.glob(self.FolderTracker(['tf','wavelet'], filename = '*-tf.pickle'))
		tf = []
		for file in files:
			with open(file ,'rb') as handle:
				tf.append(pickle.load(handle))

		contra_idx = info['ch_names'].index('PO7')	
		ipsi_idx = info['ch_names'].index('PO8')

		# plot conditions
		plt.figure(figsize = (30,10))
		alpha = {'partial':[],'whole':[]}
		for idx, cnd in enumerate(['partial','whole']):
			X = np.stack([tf[i][cnd]['base_power'][:,contra_idx,:] for i in range(len(tf))]) - \
				 np.stack([tf[i][cnd]['base_power'][:,ipsi_idx,:] for i in range(len(tf))])	 
			alpha[cnd] = X[:,6:12,:].mean(axis = 1)
			#cl_p_vals = clusterBasedPermutation(X,0)
			X = X.mean(axis = 0)
			#X[cl_p_vals == 1] = 0
			ax = plt.subplot(2,2, idx + 1, title = 'TF-{}'.format(cnd), ylabel = 'freq', xlabel = 'Time (ms)')
			plt.imshow(X, aspect = 'auto', origin = 'lower', 
						cmap = cm.jet, interpolation = None, vmin = -1, vmax = 1, extent = [times[0], times[-1],0,40])
			plt.yticks(info['frex'][::4])
			plt.colorbar()
			sns.despine(offset=50, trim = False)

		ax = plt.subplot(2,1, 2, title = 'alpha suppression', ylim = (-1,0),ylabel = 'mV', xlabel = 'Time (ms)')
		for i, cnd in enumerate(['partial','whole']):
			err, x = bootstrap(alpha[cnd])
			plt.plot(times, x, label = cnd, color = ['red','green'][i])
			plt.fill_between(times, x + err, x - err, alpha = 0.2, color = ['red','green'][i])
			self.clusterPlot(alpha[cnd], 0, 0.05, times, ax.get_ylim()[0] + 0.05*(i+1), color = ['red','green'][i])
		plt.legend(loc = 'best')
		sns.despine(offset=50, trim = False)

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['tf','wavelet'], filename = 'tf.pdf'))
		plt.close()

	def bdmBlock(self):
		'''

		'''

		with open(self.FolderTracker(['bdm','block_type'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	


		# plot conditions
		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/2.0)

		diag = {'no-cue':[],'cue':[]}
		for idx, cue in enumerate(['no-cue','cue']):

			files = glob.glob(self.FolderTracker(['bdm','block_type',cue], filename = 'class_*_perm-False-broad.pickle'))
			bdm = []
			for file in files:
				with open(file ,'rb') as handle:
					bdm.append(pickle.load(handle))

			X = np.stack([bdm[i]['all']['standard'] for i in range(len(bdm))])
			diag[cue] = np.vstack([np.diag(x) for x in X]) 
			cl_p_vals = clusterBasedPermutation(X,1/2.0)
			X = X.mean(axis = 0)
			X[cl_p_vals == 1] = 1/2.0
	
			# plot GAT
			ax = plt.subplot(2,2, idx +1, title = 'GAT-{}'.format(cue), ylabel = 'train time (ms)', xlabel = 'test time (ms)')
			plt.imshow(X, norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
						cmap = cm.bwr, interpolation = None, vmin = 0.45, vmax = 0.6)
			plt.colorbar()
			sns.despine(offset=50, trim = False)

		ax = plt.subplot(2,1, 2, title = 'Diagonal', ylabel = 'dec acc (%)', xlabel = 'time (ms)', ylim = (0.45,0.6), xlim = (times[0],times[-1]))
		for i, key in enumerate(diag.keys()):
			err, x = bootstrap(diag[key])
			plt.plot(times, x, color = ['red','green'][i], label = key)
			plt.fill_between(times, x + err, x - err, alpha = 0.2, color = ['red','green'][i])
			self.clusterPlot(diag[key], 1/2.0, 0.05, times, ax.get_ylim()[0] + 0.01*(i+1), color = ['red','green'][i])
		
		self.clusterPlot(diag['cue'], diag['no-cue'], 0.05, times, ax.get_ylim()[0] + 0.01*(3), color = 'black')

		plt.axhline(y = 1/2.0, color = 'black', ls = '--')
		sns.despine(offset=50, trim = False)
		plt.legend(loc = 'best')

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['bdm'], filename = 'block_type-decoding.pdf'))
		plt.close()	

	def bdmCue(self):
		'''

		'''

		with open(self.FolderTracker(['bdm','cue_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times']	

		files = glob.glob(self.FolderTracker(['bdm','cue_loc'], filename = 'class_*_perm-False-broad.pickle'))
		bdm = []
		for file in files:
			print file
			with open(file ,'rb') as handle:
				bdm.append(pickle.load(handle))

		# plot conditions
		plt.figure(figsize = (30,20))
		norm = MidpointNormalize(midpoint=1/3.0)
		diag = {'partial':0,'whole':0}
		for idx, cnd in enumerate(['partial','whole']):
						
			X = np.stack([bdm[i][cnd]['standard'] for i in range(len(bdm))])
			diag[cnd] = np.vstack([np.diag(x) for x in X]) 
			cl_p_vals = clusterBasedPermutation(X,1/3.0)
			X = X.mean(axis = 0)
			
			
			# plot diagonal
			# ax = plt.subplot(2,2, idx + 3, title = 'Diagonal-{}'.format(cnd), ylabel = 'dec acc (%)', xlabel = 'time (ms)', ylim = (0.3,0.4))
			# plt.plot(times, np.diag(X))
			# plt.axhline(y = 1/3.0, color = 'black', ls = '--')
			# sns.despine(offset=50, trim = False)

			# plot GAT
			X[cl_p_vals == 1] = 1/3.0
			ax = plt.subplot(2,2, idx + 1, title = 'GAT-{}'.format(cnd), ylabel = 'train time (ms)', xlabel = 'test time (ms)')
			plt.imshow(X, norm = norm, aspect = 'auto', origin = 'lower',extent = [times[0],times[-1],times[0],times[-1]], 
						cmap = cm.bwr, interpolation = None, vmin = 1/3.0, vmax = 0.5)
			plt.colorbar()
			sns.despine(offset=50, trim = False)

		ax = plt.subplot(2,1, 2, title = 'Diagonal', ylabel = 'dec acc (%)', xlabel = 'time (ms)')
		for i, cnd in enumerate(['partial', 'whole']):
				err, x = bootstrap(diag[cnd])
				plt.plot(times, x, label = cnd, color = ['red','green'][i])
				plt.fill_between(times, x + err, x - err, alpha = 0.2, color = ['red','green'][i])
				self.clusterPlot(diag[cnd], 1/3.0, 0.05, times, ax.get_ylim()[0] + 0.01*(i+1), color = ['red','green'][i])
				
		self.clusterPlot(diag['partial'], diag['whole'], 0.05, times, 0.3,'black', '--')
		plt.axhline(y = 1/3.0, color = 'black', ls = '--')
		sns.despine(offset=50, trim = False)
		plt.legend(loc = 'best')

		plt.tight_layout()	
		plt.savefig(self.FolderTracker(['bdm'], filename = 'cue_loc-decoding.pdf'))
		plt.close()

if __name__ == '__main__':
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/Cue-whole/wholevspartial'
	os.chdir(project_folder)
	PO =  WholevsPartial()

	# behavior analysis
	#PO.prepareBEH(project, 'beh-exp1', ['condition','cue','set_size'], [['whole','partial'],['cue','no'],[3,5]], project_param + ['set_size'])

	# eye analysis
	#PO.anticipatoryEye(sj_info,  nr_bins = 40, start = -500, end = 850)

	for sj in [23]:
		pass
		
		# PO.prepareEEG(sj = sj, session = 1, eog = eog, ref = ref, eeg_runs = eeg_runs, 
		#   t_min = t_min, t_max = t_max, flt_pad = flt_pad, sj_info = sj_info, 
		#   trigger = trigger, project_param = project_param, 
		#   project_folder = project_folder, binary = binary, channel_plots = True, inspect = True)

		# # CDA analysis
		# erp = ERP(header = 'cue_loc', baseline = [-0.2,0], eye = False)
		# erp.selectERPData(sj = sj, time = [-0.2, 0.85], l_filter = 40) 
		# erp.ipsiContra(sj = sj, left = [1], right = [2], l_elec = ['PO7','PO3','O1'], 
		# 								r_elec = ['PO8','PO4','O2'], midline = None, balance = False, erp_name = 'cda')
		# erp.topoFlip(left = [1])
		# erp.topoSelection(sj = sj, loc = [1,2], midline = None, topo_name = 'cda')

		# # BDM analysis
		bdm = BDM('cue_loc', nr_folds = 10, eye = False)
		bdm.Classify(sj, cnds = ['partial','whole'], cnd_header = 'block_type', bdm_labels = ['0','1','2'], time = (-0.5, 0.85), nr_perm = 0, gat_matrix = True)

		# cue trials (whole vs partial)
		#bdm = BDM('block_type', nr_folds = 10, eye = False)
		#bdm.Classify(sj, cnds = 'all', cnd_header = 'block_type', bdm_labels = ['partial','whole'], factor = dict(cue = ['cue']), time = (-0.5, 0.85), nr_perm = 0, bdm_matrix = True)

		# no-cue trials (whole vs partial)
		#bdm = BDM('block_type', nr_folds = 10, eye = False)
		#bdm.Classify(sj, cnds = 'all', cnd_header = 'block_type', bdm_labels = ['partial','whole'], factor = dict(cue = ['no']), time = (-0.5, 0.85), nr_perm = 0, bdm_matrix = True)
		
		# TF analysis
		# tf = TF()
		# tf.TFanalysis(sj = sj, cnds = ['partial','whole'], 
	 #  			  cnd_header ='block_type', base_period = (-0.2,0), 
		# 		  time_period = (0,0.85), method = 'wavelet', flip = dict(cue_loc = '1'), downsample = 4)

	
	PO.behExp1() 
	#PO.plotCDA()
	#PO.bdmBlock()
	#PO.bdmCue()
	#PO.plotTF()





