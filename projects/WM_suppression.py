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
				 'dist_loc','memory_resp','search_resp', 'suppr_loc','subject_nr']
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
		PP.filter_data(to_filter = to_filter, filter_crit = ' and search_resp == 1', cnd_sel = False, save = True)
		PP.exclude_outliers(criteria = dict(RT_search = "RT_search_filter == True", search_resp = ""))
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


		



if __name__ == '__main__':
	
	# Specify project parameters
	project_folder = '/home/dvmoors1/BB/WM_suppression'
	os.chdir(project_folder)

	# behavior analysis
	PO =  WM_suppression()
	#PO.prepareBEH(project, part, factors, labels, project_param)
	PO.plotRT()


