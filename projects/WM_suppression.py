import sys
import matplotlib
matplotlib.use('agg') #
sys.path.append('/home/dvmoors1/BB/ANALYSIS/DvM')

import seaborn as sns
from IPython import embed
from beh_analyses.PreProcessing import *
from support.FolderStructure import *
from support.support import *

# project specific info
project = 'WM_suppression'
part = 'exp1'
factors = ['suppression','condition']
labels = [['suppression','no_suppression'],['match','neutral','no']]
to_filter = [] 
project_param = ['practice','nr_trials','RT_search','condition','suppression','target_loc',
				 'dist_loc','memory_resp','search_resp', 'suppr_loc','subject_nr', 'dist_color','color_cat', 
				 'load', 'suppressed']
to_filter = ['RT_search']

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})

class WM_suppression(FolderStructure):

	def __init__(self): pass

	def prepareBEH(self, project, part, factors, labels, project_param):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and search_resp == 1', cnd_sel = False, save = False)
		PP.exclude_outliers(criteria = dict(RT_search = "RT_search_filter == True", search_resp = "", memory_resp = ""))
		#PP.prep_JASP(agg_func = 'mean', voi = 'RT_search', data_filter = "search_resp == 1", save = True)
		PP.save_data_file()

	def prepareRep(self, project, part, factors, labels, project_param):
		'''
		standard Behavior processing
		'''
		PP = PreProcessing(project = project, part = part, factor_headers = factors, factor_labels = labels)
		PP.create_folder_structure()
		PP.combine_single_subject_files(save = False)
		PP.select_data(project_parameters = project_param, save = False)
		PP.filter_data(to_filter = to_filter, filter_crit = ' and search_resp == 1', cnd_sel = False, save = True)
		PP.exclude_outliers(criteria = dict(RT_search = "RT_search_filter == True", search_resp = "", memory_resp = ""))

		# create JASP output
		embed()
		pivot_data = PP.work_data.query("RT_search_filter == True")
		pivot = pivot_data.pivot_table(values = 'RT_search', index = 'subject_nr', columns = PP.factor_headers, aggfunc = 'mean')
		# limit analysis to load 1
		headers = ['no-suppr_match','no-suppr_neutral', 'suppr_match','suppr_neutral', 'no']
		X = np.zeros((pivot.shape[0],len(headers)))

		X[:,0] = np.stack((pivot['rel-match']['no'][1].values, pivot['rel-mis']['no'][1].values)).mean(axis = 0)
		X[:,1] = pivot['unrel']['no'][1].values
		X[:,2] = np.stack((pivot['rel-match']['yes'][1].values, pivot['rel-mis']['yes'][1].values)).mean(axis = 0)
		X[:,3] = pivot['unrel']['yes'][1].values
		X[:,4] = np.stack((pivot['no']['no'][1].values, pivot['no']['yes'][1].values)).mean(axis = 0)


		embed()
		
		
		headers = ['sj'] + ['_'.join(np.array(labels,str)) for labels in product(*self.factor_labels)]
		p_values = np.hstack((pivot.index.values.reshape(-1,1), np.zeros(pivot.shape)))
		for i, labels in enumerate(product(*self.factor_labels)):
			p_values[:,i + 1] = pivot[labels]

		if save:
			np.savetxt(os.path.join(self.project_folder,'analysis', '{}_JASP.csv'.format(voi)), p_values, delimiter = "," ,header = ",".join(headers), comments='')



		#PP.prep_JASP(agg_func = 'mean', voi = 'RT_search', data_filter = "search_resp == 1", save = True)
		PP.save_data_file()

	def plotRT(self):
		'''
		Creates bar plot with individual data points overlayed
		'''	

		# read in preprocessed data and create pivot table
		file = self.FolderTracker(['exp1','analysis'], filename = 'preprocessed.csv')
		DF = pd.read_csv(file)
		DF = DF.query("RT_search_filter == True")
		pivot = DF.pivot_table(values = 'RT_search', index = 'subject_nr', columns = ['suppression','condition'], aggfunc = 'mean')
		
		# plot no suppression and suppression seperate
		plt.figure(figsize = (15,10))
		# no suppression
		ax = plt.subplot(1,2, 1, ylim = (400,900))
		df = pd.melt(pivot['no_suppression'], value_name = 'RT (ms)')
		sns.stripplot(x = 'condition', y = 'RT (ms)', data = df, size = 10,jitter = True, color = 'grey')
		sns.violinplot(x = 'condition', y = 'RT (ms)', data = df, color= 'white', cut = 1)
		sns.despine(offset=50, trim = False)

		# suppression
		ax = plt.subplot(1,2, 2, ylabel = 'RT (ms)', ylim = (400,900))
		# replace match and neutral by suppression data (makes sure that layout of figures is identical)
		df['RT (ms)'][df['condition'] == 'match'] = pivot['suppression']['match'].values
		df['RT (ms)'][df['condition'] == 'neutral'] = pivot['suppression']['neutral'].values
		sns.stripplot(x = 'condition', y = 'RT (ms)', data = df, size = 10,jitter = True, color = 'grey')
		sns.violinplot(x = 'condition', y = 'RT (ms)', data = df, color= 'white', cut = 1)
		sns.despine(offset=50, trim = False)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['exp1','analysis','figs'], filename = 'RT-main.pdf'))
		plt.close()

		print '???????'

	def distractorReps(self):
		'''
		Excludes the trials where a distractor color was repeated 
		'''	
		
		# read in preprocessed data and set distractor color categories
		file = self.FolderTracker(['exp1','analysis'], filename = 'preprocessed.csv')
		DF = pd.read_csv(file)
		DF['dist_cat'] = self.setDistCategory(DF['dist_color'])

		# set distractor repetition filter
		reps = [0]
		for i in range(1, DF.shape[0]):
			if DF['dist_cat'][i] == DF['color_cat'][i - 1]:
				reps.append(1)
			else:
				reps.append(0)	
		reps = np.array(reps, dtype = bool)
		DF['nonmatch_dist_repeat'] = (DF['condition'] == 'neutral') * reps	
		print 'nonmatch repeat {0:.1f} %'.format(reps.sum()/float(sum(DF['condition'] == 'neutral') ) * 100)

		# set dist match repeat
		reps = [0]
		for i in range(1, DF.shape[0]):
			if DF['color_cat'][i] == DF['color_cat'][i - 1]:
				reps.append(1)
			else:
				reps.append(0)	
		reps = np.array(reps, dtype = bool)
		DF['match_dist_repeat'] = (DF['condition'] == 'match') * reps
		print 'match repeat {0:.1f}%'.format(reps.sum()/float(sum(DF['condition'] == 'match') ) * 100)	

		DF = DF.query("RT_search_filter == True")
		print 'nonmatch repeat {0:.1f} %'.format(DF['nonmatch_dist_repeat'].sum()/float(sum(DF['condition'] == 'neutral') ) * 100)
		print 'match repeat {0:.1f}%'.format(DF['match_dist_repeat'].sum()/float(sum(DF['condition'] == 'match') ) * 100)	

		DF = DF.query("nonmatch_dist_repeat == False")
		DF = DF.query("match_dist_repeat == False")
		pivot = DF.pivot_table(values = 'RT_search', index = 'subject_nr', columns = ['suppression','condition'], aggfunc = 'mean')

		# plot no suppression and suppression seperate
		plt.figure(figsize = (15,10))
		# no suppression
		ax = plt.subplot(1,2, 1, ylim = (400,900), title = 'Low probable location')
		df = pd.melt(pivot['no_suppression'], value_name = 'RT (ms)')
		sns.stripplot(x = 'condition', y = 'RT (ms)', data = df, size = 10,jitter = True, color = 'grey')
		sns.violinplot(x = 'condition', y = 'RT (ms)', data = df, color= 'white', cut = 1)
		sns.despine(offset=50, trim = False)

		# suppression
		ax = plt.subplot(1,2, 2, ylabel = 'RT (ms)', ylim = (400,900), title = 'High probable location')
		# replace match and neutral by suppression data (makes sure that layout of figures is identical)
		df['RT (ms)'][df['condition'] == 'match'] = pivot['suppression']['match'].values
		df['RT (ms)'][df['condition'] == 'neutral'] = pivot['suppression']['neutral'].values
		sns.stripplot(x = 'condition', y = 'RT (ms)', data = df, size = 10,jitter = True, color = 'grey')
		sns.violinplot(x = 'condition', y = 'RT (ms)', data = df, color= 'white', cut = 1)
		sns.despine(offset=50, trim = False)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['exp1','analysis','figs'], filename = 'RT-no-repetition.pdf'))
		plt.close()


	def setDistCategory(self, colors):
		'''

		'''

		categories = []
		for color in colors:
			if color in ['#EF1E52','#E43756','#D84659','#EF213F','#E43845','#D9474B','#ED2B2A','#E33C33','#D8493C']:
				categories.append('red')
			elif color in ['#5B8600','#618427','#68823F','#47891D','#528636','#5D8349','#2E8B32','#448843','#548551']:
				categories.append('green')
			elif color in ['#0079EA','#2A79DA','#4179CB','#5A6FE6','#5F71D8','#6473CA','#6B6CE3','#6D6FD6','#6F71C8']:
				categories.append('blue')
			elif color in ['#FEBE25','#F8C04A','#F2C165','#F5C208','#F0C342','#EAC560','#ECC200','#E5C739','#E1C85A']:
				categories.append('yellow')
			elif color in ['#C241D6','#AE55C1','#9E66A1','#CF3CC8','#BF4EB6','#AC609A','#D834BB','#C946AE','#BC569F']:
				categories.append('purple')
			else:
				categories.append(color)

		return categories







if __name__ == '__main__':
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/WM_suppression'
	os.chdir(project_folder)

	# behavior analysis
	PO =  WM_suppression()
	#PO.prepareBEH(project, part, factors, labels, project_param)
	PO.prepareRep(project, 'exp2', ['condition','suppressed','load'], [['rel-match','rel-mis','unrel','no'],['yes','no'],[1,2]], project_param)
	#PO.plotRT()
	PO.distractorReps()


