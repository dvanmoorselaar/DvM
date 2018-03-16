import os
import mne
import glob
import pickle
import matplotlib
matplotlib.use('agg') # now it works via ssh connection

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from visuals.taskdisplays import *
from support.support import *
from IPython import embed 
from scipy.stats import pearsonr
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from scipy.stats import ttest_rel
from eeg_analyses.FolderStructure import FolderStructure

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 10, 'ytick.major.size': 10})


class EEGDistractorSuppression(FolderStructure):

	def __init__(self): pass

	def erpReader(self, header, erp_name):
		'''

		'''

		# read in data and shift timing
		with open(self.FolderTracker(['erp',header], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			erp = pickle.load(handle)

		with open(self.FolderTracker(['erp',header], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)
		times = info['times'] - 0.25

		return erp, info, times

	def topoChannelSelection(self, header, topo_name, erp_window = dict(P1 = (0.09, 0.13), N1 = (0.15, 0.2), N2Pc = (0.18, 0.25))):
		'''

		'''

		
		# read in data and shift timing
		topo, info, times = self.erpReader(header, topo_name)
		print topo.keys()

		# loop over all ERP components of interest
		for erp in erp_window.keys():


			# select time window of interest
			s, e = [np.argmin(abs(times - t)) for t in erp_window[erp]]

			# extract mean TOPO for window of interest
			T = np.mean(np.stack(
				[topo[j]['all'][:,s:e] for j in topo.keys()], axis = 0), 
				axis = (0,2))

			# create figure
			plt.figure(figsize = (10,10))
			ax = plt.subplot(1,1 ,1, title = 'erp-{}'.format(header))
			im = mne.viz.plot_topomap(T, info['info'], names = info['ch_names'],
				show_names = True, show = False, axes = ax, cmap = cm.jet)

			plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'evoked-{}-{}.pdf'.format(erp, header)))
			plt.close()	

	def erpInspection(self, header, erp_name):
		'''

		'''

		# read in data and shift timing
		erps, info, times = self.erpReader(header, erp_name)

		# extract mean ERP
		ipsi = np.mean(np.stack(
				[erps[key]['all']['ipsi'] for key in erps.keys()], axis = 0),
				axis = 0)

		contra = np.mean(np.stack(
				[erps[key]['all']['contra'] for key in erps.keys()], axis = 0),
				axis = 0)

		# initiate figure
		plt.figure(figsize = (20,10))
		
		for plot, data in enumerate([ipsi, contra]):
			ax = plt.subplot(1,2 , plot + 1, title = ['ipsi','contra'][plot], ylabel = 'mV')
			ax.tick_params(axis = 'both', direction = 'outer')

			for i, erp in enumerate(data):
				plt.plot(times, erp, label = '{}-{}'.
				format(erps['2']['all']['elec'][0][i], erps['2']['all']['elec'][1][i]))

			plt.legend(loc = 'best')
			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=-0.25, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
		
		sns.despine(offset=50, trim = False)
		plt.tight_layout()

		plt.savefig(self.FolderTracker(['erp','MS-plots'], filename = 'elecs-{}.pdf'.format(header)))
		plt.close()	


	def repetitionRaw(self):

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# create pivot (only include trials valid trials from RT_filter)
		RT = beh.query("RT_filter == True")
		RT_piv = RT.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')
		pivot_error = pd.Series(confidence_int(RT_piv.values), index = RT_piv.keys())
		
		# plot conditions
		plt.figure(figsize = (20,10))

		ax = plt.subplot(1,2, 1, title = 'Repetition effect', ylabel = 'RT (ms)', xlabel = 'repetition', ylim = (250,500), xlim = (0,4))
		for cnd in ['DvTv','DrTv','DvTr']:
			RT_piv[cnd].mean().plot(yerr = pivot_error[cnd], label = cnd)
		
		plt.xlim(-0.5,3.5)
		plt.xticks([0,1,2,3])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		# and plot normalized data
		norm = RT_piv.values
		for i,j in [(0,4),(4,8),(8,12)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		pivot = pd.DataFrame(norm, index = np.unique(beh['subject_nr']), columns = RT_piv.keys())
		pivot_error = pd.Series(confidence_int(pivot.values), index = pivot.keys())	

		ax = plt.subplot(1,2, 2, title = 'Normalized RT', ylabel = 'au', xlabel = 'repetition', ylim = (0.5,1), xlim = (0,4))
		for cnd in ['DvTv','DrTv','DvTr']:

			popt, pcov = curvefitting(range(4),np.array(pivot[cnd].mean()),bounds=(0, [1,1])) 
			pivot[cnd].mean().plot(yerr = pivot_error[cnd], label = '{0}: alpha = {1:.2f}; delta = {2:.2f}'.format(cnd,popt[0],popt[1]))

		plt.xlim(-0.5,3.5)
		plt.xticks([0,1,2,3])	
		plt.legend(loc='best', shadow = True)
		sns.despine(offset=10, trim = False)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'main_beh.pdf'))		
		plt.close()


	def spatialGradient(self, yrange = (350,500)):
		'''

		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# add spatial dist filter
		beh['dist_bin'] = abs(beh['dist_loc'] - beh['target_loc'])
		beh['dist_bin'][beh['dist_bin'] > 3] = 6 - beh['dist_bin'][beh['dist_bin'] > 3]

		# create pivot
		beh = beh.query("RT_filter == True")
		gradient = beh.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition','dist_bin'], aggfunc = 'mean')
		gradient_err = pd.Series(confidence_int(gradient.values), index = gradient.keys())

		# Create pivot table and extract individual headers for .csv file (input to JASP)
		gradient_array = np.hstack((np.array(gradient.index).reshape(-1,1),gradient.values))
		headers = ['sj'] + ['_'.join(np.array(labels,str)) for labels in product(*gradient.keys().levels)]
		np.savetxt(self.FolderTracker(['beh','analysis'], filename = 'gradient_JASP.csv'), gradient_array, delimiter = "," ,header = ",".join(headers), comments='')

		for cnd in ['DvTr','DrTv','DvTv']:
			plt.figure(figsize = (15,15 ))
			for i in range(4):

				ax = plt.subplot(2,2, i + 1, title = 'Repetition {}'.format(i) , ylim = yrange)
				if i % 2 == 0:
					plt.ylabel('RT (ms)')
				gradient[cnd].mean()[i].plot(kind = 'bar', yerr = gradient_err[cnd][i], color = 'grey')
			
			plt.tight_layout()
			plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'gradient_{}.pdf'.format(cnd)))
			plt.close()


	def primingCheck(self):
		'''

		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# filter out RT outliers
		DR = beh.query("RT_filter == True")

		# get effect of first repetition in distractor repetition block
		DR = DR.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')
		DR = DR['DrTv'][1] - DR['DrTv'][0]

		# get priming effect (only look at chance repetitions within DvTv); first get repetitions and then filter out outliers
		beh['priming'] = np.nan
		beh['priming'] = beh['priming'].apply(pd.to_numeric)

		rep = False
		for i, idx in enumerate(beh.index[1:]):

			if (beh.loc[idx - 1,'dist_loc'] == beh.loc[idx,'dist_loc']) and \
			(beh.loc[idx -1 ,'subject_nr'] == beh.loc[idx,'subject_nr']) and \
			(beh.loc[idx - 1,'block_cnt'] == beh.loc[idx,'block_cnt']) and \
			(rep == False) and beh.loc[idx,'RT_filter'] == True and beh.loc[idx - 1,'RT_filter'] == True:
				rep = True
				beh.loc[idx,'priming'] = beh.loc[idx,'RT'] - beh.loc[idx - 1,'RT']
			else:
				rep = False	
							
		# get priming effect
		PR = beh.pivot_table(values = 'priming', index = 'subject_nr', columns = ['block_type'], aggfunc = 'mean')['DvTv']	
		t, p = ttest_rel(DR, PR)


		# plot comparison
		plt.figure(figsize = (15,10))
		df = pd.DataFrame(np.hstack((DR.values,PR.values)),columns = ['effect'])
		df['subject_nr'] = range(DR.index.size) * 2
		df['block_type'] = ['DR'] * DR.index.size + ['PR'] * DR.index.size

		ax = sns.stripplot(x = 'block_type', y = 'effect', data = df, hue = 'subject_nr', size = 10,jitter = True)
		ax.legend_.remove()
		sns.violinplot(x = 'block_type', y = 'effect', data = df, color= 'white', cut = 1)

		plt.title('p = {0:.3f}'.format(p))
		plt.tight_layout()
		sns.despine(offset=10, trim = False)
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'priming.pdf'))	
		plt.close()

	def splitHalf(self, header, sj_id, index):
		'''

		'''	

		if header == 'dist_loc':
			block_type = 'DrTv'
		elif header == 'target_loc':
			block_type = 'DvTr'	

		# read in beh
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		# create pivot (only include trials valid trials from RT_filter)
		RT = beh.query("RT_filter == True")
		RT_piv = RT.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')[block_type]
		
		# get repetition effect and sort
		effect = RT_piv[3] - RT_piv[0]
		if sj_id != 'all':
			effect = effect[sj_id]

		if index == 'index':	
			sj_order = np.argsort(effect.values)
		elif index == 'sj_nr':
			sj_order = effect.sort_values().index.values	

		groups = {'high':sj_order[:sj_order.size/2],
		'low':sj_order[sj_order.size/2:]}

		return groups, block_type	


	def indDiffBeh(self):
		'''

		'''

		# read in data
		file = self.FolderTracker(['beh','analysis'], filename = 'preprocessed.csv')
		beh = pd.read_csv(file)

		RT = beh.query("RT_filter == True")
		RT_piv = RT.pivot_table(values = 'RT', index = 'subject_nr', columns = ['block_type','repetition'], aggfunc = 'mean')

		target = RT_piv['DvTr'][0] - RT_piv['DvTr'][3]
		dist = RT_piv['DrTv'][0] - RT_piv['DrTv'][3]

		plt.figure(figsize = (30,10))

		# plot correlation between target and distractor (repetition effect) 
		r, p = pearsonr(target,dist)
		ax = plt.subplot(1,3, 1, title = 'r = {0:0.2f}, p = {1:0.2f}'.format(r,p))
		sns.regplot(target, dist)
		plt.ylabel('distractor suppression')
		plt.xlabel('target facilitation')

		# plot individual learning effects (normalized data relative to first repetition)
		norm = RT_piv.values
		for i,j in [(0,4),(4,8),(8,12)]:
			norm[:,i:j] /= np.matrix(norm[:,i]).T

		normed_RT = pd.DataFrame(norm, index = np.unique(beh['subject_nr']), columns = RT_piv.keys())

		ax = plt.subplot(1,3, 2, title = 'Distractor', 
			xlabel = 'repetition', ylabel = 'RT (ms)')
		plt.plot(normed_RT['DrTv'].T)

		ax = plt.subplot(1,3, 3, title = 'Target', 
			xlabel = 'repetition', ylabel = 'RT (ms)')
		plt.plot(normed_RT['DvTr'].T)

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['beh','analysis','figs'], filename = 'individual.pdf'))	
		plt.close()

	def readCTFdata(self, sj_id = 'all', channel = 'all', header = 'target_loc', fname = '*_slopes_all.pickle', fband = 'all'):
		'''

		'''

		if sj_id == 'all':
			files = glob.glob(self.FolderTracker(['ctf',channel, header], filename = fname))
		else:
			fname = '{}' + fname[1:]
			files = [self.FolderTracker(['ctf',channel, header], filename = fname.format(sj)) for sj in sj_id]	
		ctf = []
		for file in files:
			print(file)
			# resad in classification dict
			with open(file ,'rb') as handle:
				ctf.append(pickle.load(handle))	

		with open(self.FolderTracker(['ctf',channel, header], filename = '{}_info.pickle'.format(fband)),'rb') as handle:
			info = pickle.load(handle)	

		return ctf, info			


	def timeFreqCTF(self, channel, header, perm = True, p_map = False):
		'''

		'''

		# read in CTF data
		slopes, info = self.readCTFdata('all',channel, header, '*_slopes_all.pickle')

		if perm:
			slopes_p, info = self.readCTFdata('all', channel, header,'*_slopes_perm_all.pickle')

		times = info['times'] -250
		freqs = (info['freqs'].min(), info['freqs'].max())
		#freqs = (info['freqs'].min(), 20)

		if header == 'dist_loc':
			conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
		elif header == 'target_loc':
			conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']

		for power in ['evoked', 'total']:
			if power == 'evoked' and header == 'target_loc':
				crange = (0, 0.3)
			elif power == 'total' and header == 'target_loc':
				crange = (0, 0.15)
			elif power == 'evoked' and header == 'dist_loc':
				crange = (-0.15 , 0.3)	
			elif power == 'total' and header == 'dist_loc':
				crange = (-0.15, 0.15)

			crange = (-0.15,0.15)	
			repeat = []
			variable = []
			plt.figure(figsize = (20,15))

			for i, cnd in enumerate(conditions):
				ax =  plt.subplot(2,2, i + 1, title = cnd, ylabel = 'freqs', xlabel = 'time (ms)')
				xy = np.stack([slopes[j][cnd][power] for j in range(len(slopes))])#[:,:7,:]
				xy = np.swapaxes(xy, 1,2) # swap to time frequency matrix
				XY = np.mean(xy,axis = 0)

				if 'r' in cnd:
					repeat.append(xy)
				else:
					variable.append(xy)	
				
				if perm:
					xy_perm = np.stack([slopes_p[j][cnd][power] for j in range(len(slopes_p))])#[:,:7,:,:]
					xy_perm = np.swapaxes(xy_perm, 1,2)
					p_val, sig = permTTest(xy, xy_perm, p_thresh = 0.05)
					XY[sig == 0] = 0

				if p_map:
					plt.imshow(p_val.T, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = 0, vmax = 1)
				else:
					plt.imshow(XY.T, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = crange[0], vmax = crange[1])
				plt.axvline(x=-250, ls = '--', color = 'white')
				plt.axvline(x=0, ls = '--', color = 'white')
				plt.colorbar(ticks = (crange[0],crange[1]))

			plt.tight_layout()
			if perm:
				if p_map:
					plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'tf-p_map_{}_{}.pdf'.format(header, power)))
				else:	
					plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'tf_{}_{}.pdf'.format(header, power)))
			else:
				plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'tf_noperm_{}_{}.pdf'.format(header, power)))	
			plt.close()	


			embed()
			# temp export matlab code
			# import scipy.io
			# rep = np.swapaxes(np.swapaxes(repeat[1] - repeat[0], 1,2),0,1)
			# print rep.shape
			# scipy.io.savemat('{}_{}_rep.mat'.format(power, header), mdict={'X': rep})
			# var = np.swapaxes(np.swapaxes(variable[1] - variable[0], 1,2),0,1)
			# print var.shape
			# scipy.io.savemat('{}_{}_var.mat'.format(power, header), mdict={'X': var})

			#self.clusterTestTimeFreq(variable, repeat, times, freqs, channel, header, power)

	def timeFreqCTFInd(self, channel, header):
		'''

		'''

		# read in CTF data
		slopes, info = self.readCTFdata('all',channel, header, '*_slopes_all.pickle')

		times = info['times'] -250
		freqs = (info['freqs'].min(), info['freqs'].max())

		if header == 'dist_loc':
			conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']
		elif header == 'target_loc':
			conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']


		power = 'total'
		for sj in range(len(slopes)):

			crange = (-0.15,0.15)	

			plt.figure(figsize = (20,15))
			for i, cnd in enumerate(conditions):
				ax =  plt.subplot(2,2, i + 1, title = cnd, ylabel = 'freqs', xlabel = 'time (ms)')
				xy = slopes[sj][cnd][power]

				plt.imshow(xy, cmap = cm.jet, interpolation='none', aspect='auto', 
					origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = crange[0], vmax = crange[1])
				plt.axvline(x=-250, ls = '--', color = 'white')
				plt.axvline(x=0, ls = '--', color = 'white')
				plt.colorbar(ticks = (crange[0],crange[1]))

			plt.tight_layout()
			plt.savefig(self.FolderTracker(['ctf',channel,'figs','ind'], filename = 'tf_{}_{}.pdf'.format(sj,header)))	
			plt.close()	

	def splitTimeFreqCTF(self, channel, header, perm = False):
		'''

		'''

		sj_id = np.array([1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
		groups, block_type = self.splitHalf(header, sj_id, 'index')
  
		# read in ctf
		slopes, info = self.readCTFdata(sj_id,channel, header, '*_slopes_all.pickle')
		times = info['times']
		freqs = (info['freqs'].min(), info['freqs'].max())

		if perm:
			slopes_p, info = self.readCTFdata(sj_id, channel, header,'*_slopes_perm_all.pickle')

		crange = (-0.15,0.15)	
		repeat = []
		for power in ['total','evoked']:

			plt.figure(figsize = (20,15))
			idx = 1
			for rep in [0,3]:
				for group in groups.keys():

					ax = plt.subplot(2,2, idx, title = 'rep_{}_{}'.format(rep,group), ylabel = 'freqs', xlabel = 'time (ms)')
					xy = np.stack([slopes[j]['{}_{}'.format(block_type,rep)][power] for j in groups[group]])
					XY = np.mean(xy,axis = 0)

					if power == 'total' and rep == 3:
						repeat.append(np.swapaxes(xy,1,2))
					
					if perm:
						xy_perm = np.stack([slopes_p[j]['{}_{}'.format(block_type,rep)][power] for j in groups[group]])
						p_val, sig = permTTest(xy, xy_perm,  p_thresh = 0.05)
						XY[sig == 0] = 0

					plt.imshow(XY, cmap = cm.jet, interpolation='none', aspect='auto', 
						origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = crange[0], vmax = crange[1])
					plt.axvline(x=0, ls = '--', color = 'white')
					plt.axvline(x=250, ls = '--', color = 'white')
					plt.colorbar(ticks = (crange[0],crange[1]))

					idx += 1

			plt.tight_layout()
			if perm:
				plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'split_{}_{}.pdf'.format(header, power)))	
			else:
				plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'split_noperm_{}_{}.pdf'.format(header, power)))	
			plt.close()	


	def clusterTestTimeFreq(self, variable, repeat, times, freqs, channel,header, power):
		'''

		'''

		plt.figure(figsize = (30,10))

		ax = plt.subplot(1,3, 1, title = 'variable', ylabel = 'freqs', xlabel = 'time (ms)')
		print 'variable'
		T_obs_plot = permTestMask2D(variable, p_value = 0.05)
		# plot 3rd - 1st repetition
		plt.imshow(T_obs_plot.T, cmap = cm.jet, interpolation='none', aspect='auto', 
			origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = 0, vmax = 5)
		plt.colorbar(ticks = (0,5))

		print 'repeat'
		ax = plt.subplot(1,3, 2, title = 'repeat', ylabel = 'freqs', xlabel = 'time (ms)')
		# plot 3rd - 1st repetition
		T_obs_plot = permTestMask2D(repeat, p_value = 0.05)
		plt.imshow(T_obs_plot.T, cmap = cm.jet, interpolation='none', aspect='auto', 
			origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = 0, vmax = 5)
		plt.colorbar(ticks = (0,5))
		
		print 'interaction'
		ax = plt.subplot(1,3, 3, title = 'interaction', ylabel = 'freqs', xlabel = 'time (ms)')
		# plot repeat - variable
		T_obs_plot = permTestMask2D([variable[1] - variable[0], repeat[1] - repeat[0]], p_value = 0.05)
		plt.imshow(T_obs_plot.T, cmap = cm.jet, interpolation='none', aspect='auto', 
			origin = 'lower', extent=[times[0],times[-1],freqs[0],freqs[1]], vmin = 0, vmax = 5)
		plt.colorbar(ticks = (0,5))

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'TF_comparison_{}_{}.pdf'.format(header, power)))
		plt.close()	

	def inspectSlopes(self, channel, header):
		'''

		'''

		# read in data
		ctf, info = self.readCTFdata(channel, header, '*_ctf_all.pickle', fband = 'all')

		# get X (time),Y (channel), Z data (channel response)
		X = info['times'][::info['downsample']]
		Y = np.arange(7)
		X, Y = np.meshgrid(X, Y)
		
		for fr, band in enumerate(info['freqs']):
			if band[1] <= 20:
				for power in ['total','evoked']:
					f = plt.figure(figsize = (20,15))
					for i, cnd in enumerate(info['conditions']):

						ax = f.add_subplot(2, 2, i + 1, projection='3d', title = cnd)
						if header == 'target_loc':
							crange = (0,1)
						elif header == 'dist_loc':
							crange = (-0.5,0.5)

						Z = np.dstack([np.mean(ctf[j][cnd]['ctf'][power][fr,:], axis = (0,2)).T for j in range(len(ctf))])
						Z = np.vstack((Z.mean(axis =2), Z.mean(axis =2)[0,:]))
						surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
			                      linewidth=0, antialiased=False, rstride = 1, cstride = 1, vmin = crange[0], vmax = crange[1])
						
						ax.set_zlim(crange)
						f.colorbar(surf, shrink = 0.5, ticks = crange)

					plt.tight_layout()
					plt.savefig(self.FolderTracker(['ctf',channel,'figs'], filename = 'ctfs_{}_{}_{}-{}.pdf'.format(header,power,band[0],band[1])))
					plt.close()


	def inspectClassification(self, header):
		'''

		'''

		if header == 'target_loc':
			conditions = ['DvTv_0','DvTv_3','DvTr_0','DvTr_3']
		elif header == 'dist_loc':
			conditions = ['DvTv_0','DvTv_3','DrTv_0','DrTv_3']	

		# read in data
		with open(self.FolderTracker(['bdm',header], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		files = glob.glob(self.FolderTracker(['bdm', header], filename = 'classify_shuffle_*.pickle'))
		bdm = []
		for file in files:
			with open(file ,'rb') as handle:
				bdm.append(pickle.load(handle))
		
		plt.figure(figsize = (20,15))
		perm = []
		for i, cnd in enumerate(conditions):

			if i == 0:
				ax = plt.subplot(1,2 , 1, title = 'Variable', ylabel = 'Classification acc', xlabel = 'Time (ms)')
			elif i == 2:
				ax = plt.subplot(1,2 , 2, title = 'Repeat', xlabel = 'Time (ms)')	

			acc = np.vstack([bdm[j][cnd] for j in range(len(bdm))])
			perm.append(acc)
			err, acc = bootstrap(acc)
			
			plt.plot(info['times'], acc, label = cnd)
			plt.fill_between(info['times'], acc + err, acc - err, alpha = 0.2)

			plt.axhline(y=1/6.0, color = 'black')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			if i % 2 == 1:
				mask, sig_clusters = permTestMask1D(perm[-2:])
				plt.fill_between(info['times'], 1/6.0 - 0.001, 1/6.0 + 0.001, where = mask == True, color = 'grey', label = 'p < 0.05')
				plt.legend(loc = 'best')

		plt.savefig(self.FolderTracker(['bdm','figs'], filename = 'class_shuffle_{}.pdf'.format(header)))		
		plt.close()

	def ipsiContraCheck(self, header, erp_name):
		'''

		'''


		# read in data
		with open(self.FolderTracker(['erp','dist_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		with open(self.FolderTracker(['erp','target_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			t_erps = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			d_erps = pickle.load(handle)

		print t_erps.keys(), d_erps.keys()
		plt.figure(figsize = (20,20))
		titles = ['T0-left','T0-right', 'T3-left','T3-right','D0-left','D0-right','D3-left','D3-right']
		for i, cnd in enumerate(['DvTr_0','DvTr_0','DvTr_3','DvTr_3','DrTv_0','DrTv_0','DrTv_3','DrTv_3']):
			
			ax = plt.subplot(4,2 , i + 1, title = titles[i], ylabel = 'mV')
			
			if i < 4:
				if i % 2 == 0:
					ipsi = np.vstack([t_erps[str(key)][cnd]['l_ipsi'] for key in t_erps.keys()])
					contra = np.vstack([t_erps[str(key)][cnd]['l_contra'] for key in t_erps.keys()])
				else:
					ipsi = np.vstack([t_erps[str(key)][cnd]['r_ipsi'] for key in t_erps.keys()])
					contra = np.vstack([t_erps[str(key)][cnd]['r_contra'] for key in t_erps.keys()])
			else:
				if i % 2 == 0:
					ipsi = np.vstack([d_erps[str(key)][cnd]['l_ipsi'] for key in d_erps.keys()])
					contra = np.vstack([d_erps[str(key)][cnd]['l_contra'] for key in d_erps.keys()])
				else:
					ipsi = np.vstack([d_erps[str(key)][cnd]['r_ipsi'] for key in d_erps.keys()])
					contra = np.vstack([d_erps[str(key)][cnd]['r_contra'] for key in d_erps.keys()])

			err, ipsi = bootstrap(ipsi)
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'blue')
			plt.fill_between(info['times'], ipsi + err, ipsi - err, alpha = 0.2, color = 'blue')

			err, contra = bootstrap(contra)
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], contra + err, contra  - err, alpha = 0.2, color = 'green')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','figs'], filename = '{}-check-1.pdf'.format(erp_name)))
		plt.close()

		plt.figure(figsize = (20,20))

		# plot repetition effect
		ax = plt.subplot(2,2 , 1, title = 'Target repetition Left', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DvTr_0','DvTr_3']):
			L_ipsi = np.vstack([t_erps[str(key)][cnd]['l_ipsi'] for key in t_erps.keys()])
			L_contra = np.vstack([t_erps[str(key)][cnd]['l_contra'] for key in t_erps.keys()])
			err, diff = bootstrap(L_contra - L_ipsi)
			perm.append(L_contra - L_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		ax = plt.subplot(2,2 , 2, title = 'Target repetition Right', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DvTr_0','DvTr_3']):
			R_ipsi = np.vstack([t_erps[str(key)][cnd]['r_ipsi'] for key in t_erps.keys()])
			R_contra = np.vstack([t_erps[str(key)][cnd]['r_contra'] for key in t_erps.keys()])
			err, diff = bootstrap(R_contra - R_ipsi)
			perm.append(R_contra - R_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		ax = plt.subplot(2,2 , 3, title = 'Distractor repetition Left', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DrTv_0','DrTv_3']):
			L_ipsi = np.vstack([d_erps[str(key)][cnd]['l_ipsi'] for key in d_erps.keys()])
			L_contra = np.vstack([d_erps[str(key)][cnd]['l_contra'] for key in d_erps.keys()])
			err, diff = bootstrap(L_contra - L_ipsi)
			perm.append(L_contra - L_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		ax = plt.subplot(2,2 , 4, title = 'Distractor repetition Right', ylabel = 'mV')

		perm = []
		for i, cnd in enumerate(['DrTv_0','DrTv_3']):
			R_ipsi = np.vstack([d_erps[str(key)][cnd]['r_ipsi'] for key in d_erps.keys()])
			R_contra = np.vstack([d_erps[str(key)][cnd]['r_contra'] for key in d_erps.keys()])
			err, diff = bootstrap(R_contra - R_ipsi)
			perm.append(R_contra - R_ipsi)
			plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
		mask, sig_clusters = permTestMask1D(perm)
		plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
		plt.legend(loc = 'best')

		plt.axhline(y=0, ls = '--', color = 'grey')
		plt.axvline(x=0, ls = '--', color = 'grey')
		plt.axvline(x=0.25, ls = '--', color = 'grey')

		plt.tight_layout()
		plt.savefig(self.FolderTracker(['erp','figs'], filename = '{}-check-2.pdf'.format(erp_name)))
		plt.close()


	def N2pCvsPd(self, erp_name, split = False):
		'''		

		'''

		sj_id = np.array([1,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21])

		# read in data
		with open(self.FolderTracker(['erp','dist_loc'], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		with open(self.FolderTracker(['erp','target_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			t_erps = pickle.load(handle)

		with open(self.FolderTracker(['erp','dist_loc'], filename = '{}.pickle'.format(erp_name)) ,'rb') as handle:
			d_erps = pickle.load(handle)

		if split:
			groups, block_type = self.splitHalf(split, sj_id, 'sj_nr')

		else:
			groups = {'all':t_erps.keys()}	

		for group in groups.keys():	

			# get ipsilateral and contralateral erps tuned to the target and tuned to the distractor (collapsed across all conditions)
			#T_ipsi = np.vstack([t_erps[str(key)]['all']['ipsi'] for key in t_erps.keys()])
			#T_contra = np.vstack([t_erps[str(key)]['all']['contra'] for key in t_erps.keys()])
			T_ipsi = np.vstack([t_erps[str(key)]['all']['ipsi'] for key in groups[group]])
			T_contra = np.vstack([t_erps[str(key)]['all']['contra'] for key in groups[group]])

			#D_ipsi = np.vstack([d_erps[str(key)]['all']['ipsi'] for key in d_erps.keys()])
			#D_contra = np.vstack([d_erps[str(key)]['all']['contra'] for key in d_erps.keys()])
			D_ipsi = np.vstack([d_erps[str(key)]['all']['ipsi'] for key in groups[group]])
			D_contra = np.vstack([d_erps[str(key)]['all']['contra'] for key in groups[group]])

			plt.figure(figsize = (20,20))
			
			# plot ipsi and contralateral erps with bootsrapped error bar
			ax = plt.subplot(4,2 , 1, title = 'Target ERPs', ylabel = 'mV')

			err, ipsi = bootstrap(T_ipsi)
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'blue')
			plt.fill_between(info['times'], ipsi + err, ipsi - err, alpha = 0.2, color = 'blue')

			err, contra = bootstrap(T_contra)
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], contra + err, contra  - err, alpha = 0.2, color = 'green')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			ax = plt.subplot(4,2 , 2, title = 'Distractor ERPs', ylabel = 'mV')
			
			err, ipsi = bootstrap(D_ipsi)
			plt.plot(info['times'], ipsi, label = 'ipsi', color = 'blue')
			plt.fill_between(info['times'], ipsi + err, ipsi - err, alpha = 0.2, color = 'blue')
			plt.legend(loc = 'best')	

			err, contra = bootstrap(D_contra)
			plt.plot(info['times'], contra, label = 'contra', color = 'green')
			plt.fill_between(info['times'], contra + err, contra - err, alpha = 0.2, color = 'green')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			# plot diff wave collapsed across all conditions
			ax = plt.subplot(4,2 , 3, title = 'Target diff', ylabel = 'mV')
			
			err, diff = bootstrap(T_contra - T_ipsi)
			plt.plot(info['times'], diff, color = 'black')
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = 'black')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')
			
			ax = plt.subplot(4,2 , 4, title = 'Distractor diff', ylabel = 'mV')
			
			err, diff = bootstrap(D_contra - D_ipsi)
			plt.plot(info['times'], diff, color = 'black')
			plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = 'black')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')
		
			# plot repetition effect
			ax = plt.subplot(4,2 , 5, title = 'Target repetition', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DvTr_0','DvTr_3']):
				T_ipsi = np.vstack([t_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				T_contra = np.vstack([t_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(T_contra - T_ipsi)
				perm.append(T_contra - T_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			ax = plt.subplot(4,2 , 6, title = 'Distractor repetition', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DrTv_0','DrTv_3']):
				D_ipsi = np.vstack([d_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				D_contra = np.vstack([d_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(D_contra - D_ipsi)
				perm.append(D_contra - D_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			# plot repetition effect (control)
			ax = plt.subplot(4,2, 7, title = 'Target repetition (control)', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DvTv_0','DvTv_3']):
				T_ipsi = np.vstack([t_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				T_contra = np.vstack([t_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(T_contra - T_ipsi)
				perm.append(T_contra - T_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			ax = plt.subplot(4,2, 8, title = 'Distractor repetition (control)', ylabel = 'mV')

			perm = []
			for i, cnd in enumerate(['DvTv_0','DvTv_3']):
				D_ipsi = np.vstack([d_erps[str(key)][cnd]['ipsi'] for key in groups[group]])
				D_contra = np.vstack([d_erps[str(key)][cnd]['contra'] for key in groups[group]])
				err, diff = bootstrap(D_contra - D_ipsi)
				perm.append(D_contra - D_ipsi)
				plt.plot(info['times'], diff, label = cnd, color = ['r','y'][i])
				plt.fill_between(info['times'], diff + err, diff - err, alpha = 0.2, color = ['r','y'][i])
			mask, sig_clusters = permTestMask1D(perm)
			plt.fill_between(info['times'], -0.05, 0.05, where = mask == True, color = 'grey', label = 'p < 0.05')
			plt.legend(loc = 'best')

			plt.axhline(y=0, ls = '--', color = 'grey')
			plt.axvline(x=0, ls = '--', color = 'grey')
			plt.axvline(x=0.25, ls = '--', color = 'grey')

			sns.despine(offset=10, trim = False)
			plt.tight_layout()
			if split:
				plt.savefig(self.FolderTracker(['erp','figs'], filename = 'n2pc-Pd-{}-{}_{}.pdf'.format(group,split,erp_name)))	
			else:
				plt.savefig(self.FolderTracker(['erp','figs'], filename = 'n2pc-Pd_{}_{}.pdf'.format(group,erp_name)))		
			plt.close()


	def clusterTopo(self, header, fname = ''):
		'''

		'''

		# read in data
		files = glob.glob(self.FolderTracker(['erp', header], filename = fname))
		topo = []
		for file in files:
			with open(file ,'rb') as handle:
				topo.append(pickle.load(handle))



	def topoAnimation(self, header):
		'''

		'''


		# read in data
		files = glob.glob(self.FolderTracker(['erp', header], filename = 'topo_*.pickle'))
		topo = []
		for file in files:
			print file
			# read in erp dict
			with open(file ,'rb') as handle:
				topo.append(pickle.load(handle))

		# read in processed data object (contains info for plotting)
		EEG = mne.read_epochs(self.FolderTracker(extension = ['processed'], filename = 'subject-1_all-epo.fif'))

		# read in plot dict		
		with open(self.FolderTracker(['erp',header], filename = 'plot_dict.pickle') ,'rb') as handle:
			info = pickle.load(handle)

		plt_idx = [1,3,7,9]
		for image in range(564):
			f = plt.figure(figsize = (20,20))	
			for i, cnd in enumerate(np.sort(topo[0].keys())):

				ax = plt.subplot(3,3 , plt_idx[i], title = cnd)

				T = np.mean(np.dstack([np.mean(topo[j][cnd], axis = 0) for j in range(len(topo))]), axis = 2)
				mne.viz.plot_topomap(T[:,image], EEG.info, show_names = False, show = False, vmin = -4, vmax = 3)

			ax = plt.subplot(3,3 , 5, title = '{0:0.2f}'.format(info['times'][image]))
			if info['times'][image] <= 0:	
				searchDisplayEEG(ax, fix = True)
			elif info['times'][image] <= 0.25:
				searchDisplayEEG(ax, fix = False)	
			else:
				searchDisplayEEG(ax, fix = False, stimulus = 4, erp_type = header)		

			plt.tight_layout()	
			plt.savefig(self.FolderTracker(['erp', 'figs','video'], filename = 'topo_{0}_{1:03}.png'.format(header,image + 1)))
			plt.close()

		plt_idx = [1,3]	
		for image in range(564):
			f = plt.figure(figsize = (20,20))	
			for i in range(2):

				if i == 0:
					title = 'variable'
					T = np.mean(np.dstack([np.mean(topo[j]['DvTv_0'], axis = 0) for j in range(len(topo))]), axis = 2) - \
					np.mean(np.dstack([np.mean(topo[j]['DvTv_3'], axis = 0) for j in range(len(topo))]), axis = 2)
				else:
					T = np.mean(np.dstack([np.mean(topo[j]['DrTv_0'], axis = 0) for j in range(len(topo))]), axis = 2) - \
					np.mean(np.dstack([np.mean(topo[j]['DrTv_3'], axis = 0) for j in range(len(topo))]), axis = 2)
					title = 'repeat'	
				ax = plt.subplot(1,3 ,plt_idx[i] , title = title)
				mne.viz.plot_topomap(T[:,image], EEG.info, show_names = False, show = False, vmin = -1, vmax = 1)

			ax = plt.subplot(1,3 , 2, title = '{0:0.2f}'.format(info['times'][image]))
			if info['times'][image] <= 0:	
				searchDisplayEEG(ax, fix = True)
			elif info['times'][image] <= 0.25:
				searchDisplayEEG(ax, fix = False)	
			else:
				searchDisplayEEG(ax, fix = False, stimulus = 4, erp_type = header)		

			plt.tight_layout()	
			plt.savefig(self.FolderTracker(['erp', 'figs','video'], filename = 'topo_diff_{0}_{1:03}.png'.format(header,image + 1)))
			plt.close()


if __name__ == '__main__':
	
	os.chdir('/home/dvmoors1/big_brother/Dist_suppression') 

	PO = EEGDistractorSuppression()

	# Behavior plots 
	#PO.repetitionRaw()	
	#PO.spatialGradient()
	#PO.primingCheck()
	#PO.indDiffBeh()

	# CTF plots
	#PO.timeFreqCTFInd(channel = 'posterior_channels', header = 'target_loc')
	#PO.timeFreqCTFInd(channel = 'posterior_channels', header = 'dist_loc')
	#PO.timeFreqCTF(channel = 'posterior_channels',header = 'target_loc', perm = False, p_map = False)
	#PO.timeFreqCTF(channel = 'posterior_channels',header = 'dist_loc', perm = True, p_map = True)
	#PO.timeFreqCTF(channel = 'posterior_channels',header = 'target_loc', perm = False, p_map = False)
	#PO.timeFreqCTF(channel = 'posterior_channels',header = 'dist_loc', perm = False, p_map = False)	

	#PO.splitTimeFreqCTF(channel = 'posterior_channels', header = 'target_loc', perm = True)
	#PO.splitTimeFreqCTF(channel = 'posterior_channels', header = 'dist_loc', perm = True)
	#PO.inspectSlopes(channel = 'posterior_channels', header = 'target_loc')
	#PO.inspectSlopes(channel = 'posterior_channels', header = 'dist_loc')

	# BDM plots
	#PO.inspectClassification(header = 'target_loc')
	#PO.inspectClassification(header = 'dist_loc')

	# ERP plots
	PO.topoChannelSelection(header = 'dist_loc', topo_name = 'topo_lat-down1')
	PO.erpInspection(header = 'dist_loc', erp_name = 'lat-down1')
	PO.topoChannelSelection(header = 'target_loc', topo_name = 'topo_lat-down1')
	PO.erpInspection(header = 'target_loc', erp_name = 'lat-down1')

	#PO.ipsiContraCheck(header = 'target_loc', erp_name = 'left-right-1')
	#PO.ipsiContraCheck(header = 'target_loc', erp_name = 'left-right-2')
	#PO.N2pCvsPd(erp_name = 'ipsi_contra', split = 'dist_loc')
	#PO.N2pCvsPd(erp_name = 'ipsi_contra', split = 'target_loc')
	#PO.N2pCvsPd(erp_name = 'ipsi_contra', split = False)
	#PO.topoAnimation('target_loc')

