import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helperFunctions import *
from IPython import embed 
from mne.stats import permutation_cluster_test



header = 'target_loc'


def paired_t(*args):
	"""Call scipy.stats.ttest_rel, but return only f-value."""
	from scipy.stats import ttest_rel
	return ttest_rel(*args)[0]

def ipsiContra():
	'''

	'''

	#sns.set(font_scale=2.5)
	sns.set_style('white')
	sns.set_style('white', {'axes.linewidth': 2})	

	plt.figure(figsize = (20,10))

	with open('/home/dvmoors1/big_brother/Dist_suppression/erp/plot_dict.pickle','rb') as handle:
		plot = pickle.load(handle)

	times = plot['times']  	

	with open('/home/dvmoors1/big_brother/Dist_suppression/erp/ipsi_contra.pickle' ,'rb') as handle:
		erps = pickle.load(handle)

	ax = plt.subplot(1,2 , 1, title = 'Target repeat diff', ylabel = 'mV')

	for cnd in ['DvTr_0','DvTr_3']:#['DvTr0','DvTr1','DvTr2','DvTr3']:
		ipsi_t = np.mean(np.vstack([erps[key][cnd]['ipsi'] for key in erps.keys()]), axis = 0)
		contra_t = np.mean(np.vstack([erps[key][cnd]['contra'] for key in erps.keys()]), axis = 0)
		plt.plot(times, contra_t - ipsi_t, label = cnd)	
	plt.legend(loc = 'best')

	ax = plt.subplot(1,2 , 2, title = 'Distractor repeat diff', ylabel = 'mV')

	for cnd in ['DrTv_0','DrTv_3']: #['DrTv0','DrTv1','DrTv2','DrTv3']:
		ipsi_d = np.mean(np.vstack([erps[key][cnd]['ipsi'] for key in erps.keys()]), axis = 0)
		contra_d = np.mean(np.vstack([erps[key][cnd]['contra'] for key in erps.keys()]), axis = 0)
		plt.plot(times, contra_d - ipsi_d, label = cnd)	
	plt.legend(loc = 'best')

	sns.despine()
	plt.tight_layout()
	plt.savefig('/home/dvmoors1/big_brother/Dist_suppression/erp/n2pc_Pd_modulation.pdf')		
	plt.close()


def ipsiContra():
	'''

	'''

	#sns.set(font_scale=2.5)
	sns.set_style('white')
	sns.set_style('white', {'axes.linewidth': 2})	

	plt.figure(figsize = (20,20))

	with open('/Users/dirk/Desktop/suppression/erp/{}/plot_dict.pickle'.format(header) ,'rb') as handle:
		plot = pickle.load(handle)

	time = [-0.3, 0.8]  	
	start, end = [np.argmin(abs(plot['times'] - t)) for t in time]
	times = plot['times'][start:end]     	

	with open('/Users/dirk/Desktop/suppression/erp/{}/ipsi_contra.pickle'.format('target_loc') ,'rb') as handle:
		erps_t = pickle.load(handle)

	with open('/Users/dirk/Desktop/suppression/erp/{}/ipsi_contra.pickle'.format('dist_loc') ,'rb') as handle:
		erps_d = pickle.load(handle)

	# plot ipsi and contra (target left and distractor right)
	ipsi_t = np.mean(np.vstack([erps_t[key]['all']['ipsi'] for key in erps_t.keys()]), axis = 0)
	contra_t = np.mean(np.vstack([erps_t[key]['all']['contra'] for key in erps_t.keys()]), axis = 0)	
	diff_t = contra_t - ipsi_t

	ipsi_d = np.mean(np.vstack([erps_d[key]['all']['ipsi'] for key in erps_d.keys()]), axis = 0)
	contra_d = np.mean(np.vstack([erps_d[key]['all']['contra'] for key in erps_d.keys()]), axis = 0)	
	diff_d = contra_d - ipsi_d

	ax = plt.subplot(3,2 , 1, title = 'Target ERPs', ylabel = 'mV')

	plt.plot(times, ipsi_t, label = 'ipsi')
	plt.plot(times, contra_t, label = 'contra')
	plt.legend(loc = 'best')
	
	ax = plt.subplot(3,2 , 2, title = 'Distractor ERPS', ylabel = 'mV')

	plt.plot(times, ipsi_d, label = 'ipsi')
	plt.plot(times, contra_d, label = 'contra')
	plt.legend(loc = 'best')

	ax = plt.subplot(3,2 , 3, title = 'Target diff', ylabel = 'mV')

	plt.plot(times, contra_t - ipsi_t)

	ax = plt.subplot(3,2 , 4, title = 'Distractor diff', ylabel = 'mV')

	plt.plot(times, contra_d - ipsi_d)

	ax = plt.subplot(3,2 , 5, title = 'Target repeat diff', ylabel = 'mV')

	for cnd in ['DvTr0','DvTr3']:#['DvTr0','DvTr1','DvTr2','DvTr3']:
		ipsi_t = np.mean(np.vstack([erps_t[key][cnd]['ipsi'] for key in erps_t.keys()]), axis = 0)
		contra_t = np.mean(np.vstack([erps_t[key][cnd]['contra'] for key in erps_t.keys()]), axis = 0)
		plt.plot(times, contra_t - ipsi_t, label = cnd)	
	plt.legend(loc = 'best')

	ax = plt.subplot(3,2 , 6, title = 'Distractor repeat diff', ylabel = 'mV')

	for cnd in ['DrTv0','DrTv3']: #['DrTv0','DrTv1','DrTv2','DrTv3']:
		ipsi_d = np.mean(np.vstack([erps_d[key][cnd]['ipsi'] for key in erps_d.keys()]), axis = 0)
		contra_d = np.mean(np.vstack([erps_d[key][cnd]['contra'] for key in erps_d.keys()]), axis = 0)
		plt.plot(times, contra_d - ipsi_d, label = cnd)	
	plt.legend(loc = 'best')

	sns.despine()
	plt.tight_layout()
	plt.savefig('/Users/dirk/Desktop/suppression/erp/n2pc_Pd_modulation.pdf')		
	plt.close()

def plotCTFSlopeAcrossTime(header, power, channel = 'posterior', freqs = 'all'):
	'''

	'''

	# plotting parameters
	sns.set(font_scale=2.5)
	sns.set_style('white')
	sns.set_style('white', {'axes.linewidth': 2})	

	# read in CTF data
	ctf = []
	for sj in subject_id:
		# read in classification dict
		with open('/home/dvmoors1/big_brother/Dist_suppression/ctf/{}_channels/{}/{}_slopes_{}.pickle'.format(channel,header,sj, freqs) ,'rb') as handle:
			ctf.append(pickle.load(handle))	

	with open('/home/dvmoors1/big_brother/Dist_suppression/ctf/{}_channels/{}/{}_info.pickle'.format(channel,header, freqs),'rb') as handle:
		plot_dict = pickle.load(handle)			

	if header == 'target_loc':
		rep_cond = ['DvTr_0','DvTr_3']
	else:
		rep_cond = ['DrTv_0','DrTv_3']	

	plt.figure(figsize = (20,10))

	for idx, plot in enumerate(['variable', 'repeat']):

		ax = plt.subplot(1,2, idx + 1, title = plot, ylabel = 'CTF slope', ylim = (-0.2, 0.2))

		if plot == 'variable':
			diff = []
			for i, cnd in enumerate(['DvTv_0','DvTv_3']):
				X = np.vstack([ctf[j][cnd][power] for j in range(len(ctf))])
				diff.append(X)
				error = bootstrap(X)
				X =  X.mean(axis = 0)

				plt.plot(plot_dict['times'],  X, color = ['g','r'][i], label = cnd)
				plt.fill_between(plot_dict['times'], X + error, X - error, alpha = 0.2, color = ['g','r'][i])

			plt.axhline(y=0, ls = '--', color = 'black')
			plt.axvline(x=0.258, ls = '--', color = 'grey', label = 'onset gabor')
			T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)
			print('T',header, cluster_pv)
			mask = np.zeros(plot_dict['times'].size,dtype = bool)
			for cl in np.array(clusters)[np.where(cluster_pv < 0.05)[0]]:
				mask[cl[0]] = True
			plt.fill_between(plot_dict['times'], -0.002, 0.002, where = mask == True, color = 'grey', label = 'p < 0.05')	

			plt.legend(loc='best', shadow = True)	
			
		
		elif plot == 'repeat':
			diff = []
			for i, cnd in enumerate(rep_cond):		
				X = np.vstack([ctf[j][cnd][power] for j in range(len(ctf))])
				diff.append(X)
				error = bootstrap(X)
				X =  X.mean(axis = 0)

				plt.plot(plot_dict['times'],  X, color = ['g','r'][i], label = cnd)
				plt.fill_between(plot_dict['times'], X + error, X - error, alpha = 0.2, color = ['g','r'][i])
			
			plt.axhline(y=0, ls = '--', color = 'black')
			plt.axvline(x=0.258, ls = '--', color = 'grey', label = 'onset gabor')
			T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)	
			print('T',header, cluster_pv)
			mask = np.zeros(plot_dict['times'].size,dtype = bool)
			for cl in np.array(clusters)[np.where(cluster_pv < 0.05)[0]]:
				mask[cl[0]] = True
			plt.fill_between(plot_dict['times'], -0.002, 0.002, where = mask == True, color = 'grey', label = 'p < 0.05')

			plt.legend(loc='best', shadow = True)
			sns.despine(offset=10, trim = False)

	plt.savefig('/home/dvmoors1/big_brother/Dist_suppression/ctf/{}_channels/{}/figs/group_slopes_{}.pdf'.format(channel,header,power))		
	plt.close()	

subject_id = [3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
for header in ['dist_loc','target_loc']:
	for power in ['total','evoked']:
		plotCTFSlopeAcrossTime(header, power,'posterior','alpha')


def plotBdmAcrossTime():
	'''

	'''

	# plotting parameters
	sns.set(font_scale=2.5)
	sns.set_style('white')
	sns.set_style('white', {'axes.linewidth': 2})	

	# read in BDM data
	bdm = []
	for sj in subject_id:
		# read in classification dict
		with open('/Users/dirk/Desktop/suppression/bdm/{}/class_acc_{}.pickle'.format(header,sj) ,'rb') as handle:
			bdm.append(pickle.load(handle))	

	with open('/Users/dirk/Desktop/suppression/bdm/{}/plot_dict.pickle'.format(header),'rb') as handle:
		plot_dict = pickle.load(handle)			

	if header == 'target_loc':
		rep_cond = ['DvTr0','DvTr3']
	else:
		rep_cond = ['DrTv0','DrTv3']	

	plt.figure(figsize = (30,20))

	for idx, plot in enumerate(['variable', 'repeat']):

		ax = plt.subplot(1,2, idx + 1, title = plot, ylabel = 'classification acc', ylim = (0, 0.3))

		if plot == 'variable':
			diff = []
			for i, cnd in enumerate(['DvTv0','DvTv3']):
				X = np.vstack([bdm[j][cnd] for j in range(len(bdm))])
				diff.append(X)

				error = bootstrap(X)
				X =  X.mean(axis = 0)

				plt.plot(plot_dict['times'],  X, color = ['g','r'][i], label = cnd)
				plt.fill_between(plot_dict['times'], X + error, X - error, alpha = 0.2, color = ['g','r'][i])

			plt.axhline(y=1/6.0, ls = '--', color = 'black', label = 'chance')
			plt.axvline(x=0.258, ls = '--', color = 'grey', label = 'onset gabor')
			T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)
			print(header, cluster_pv)
			mask = np.zeros(plot_dict['times'].size,dtype = bool)
			for cl in np.array(clusters)[np.where(cluster_pv < 0.05)[0]]:
				mask[cl[0]] = True
			plt.fill_between(plot_dict['times'], 0.100, .104, where = mask == True, color = 'grey', label = 'p < 0.05')	

			plt.legend(loc='best', shadow = True)	
			
		
		elif plot == 'repeat':
			diff = []
			for i, cnd in enumerate(rep_cond):		
				X = np.vstack([bdm[j][cnd] for j in range(len(bdm))])
				diff.append(X)
				error = bootstrap(X)
				X =  X.mean(axis = 0)

				plt.plot(plot_dict['times'],  X, color = ['g','r'][i], label = cnd)
				plt.fill_between(plot_dict['times'], X + error, X - error, alpha = 0.2, color = ['g','r'][i])
			
			plt.axhline(y=1/6.0, ls = '--', color = 'black',label = 'chance')
			plt.axvline(x=0.258, ls = '--', color = 'grey', label = 'onset gabor')
			T_obs, clusters, cluster_pv, HO = permutation_cluster_test(diff, stat_fun = paired_t)	
			print(header, cluster_pv)
			mask = np.zeros(plot_dict['times'].size,dtype = bool)
			for cl in np.array(clusters)[np.where(cluster_pv < 0.05)[0]]:
				mask[cl[0]] = True
			plt.fill_between(plot_dict['times'], .100, .104, where = mask == True, color = 'grey', label = 'p < 0.05')

			plt.legend(loc='best', shadow = True)
			sns.despine(offset=10, trim = False)

	plt.savefig('/Users/dirk/Desktop/suppression/bdm/{}/figs/group_classification.pdf'.format(header))		
	plt.close()			
		
def plotIpsiContra(subject_id, header):
	'''

	'''

	# plotting parameters
	sns.set(font_scale=2.5)
	sns.set_style('white')
	sns.set_style('white', {'axes.linewidth': 2})

	if header == 'target_loc':
		rep_cond = ['DvTr0','DvTr3']
	else:
		rep_cond = ['DrTv0','DrTv3']

	# read in data
	# read in erp data
	erp = []

	for sj in subject_id:
		# read in classification dict
		with open('/Users/dirk/Desktop/suppression/erp/{}/ipsi_contra_{}.pickle'.format(header,sj) ,'rb') as handle:
			erp.append(pickle.load(handle))	

	with open('/Users/dirk/Desktop/suppression/erp/{}/plot_dict.pickle'.format(header),'rb') as handle:
		plot_dict = pickle.load(handle)			

	times = plot_dict['times']	
	start, end = [np.argmin(abs(times - t)) for t in (-0.3,0.8)]
	times = times[start:end]

	plt.figure(figsize = (30,20))

	for cnd in ['variable','repeat']:

		if cnd == 'variable':

			for i, rep in enumerate(['DvTv0','DvTv3']):

				ax = plt.subplot(2,2, i + 1, title = cnd + rep[-1], ylabel = 'micro Volt', xlabel = 'Time (ms)', ylim = (-6, 6))

				ipsi = np.vstack([erp[i][rep]['ipsi'] for i in range(len(erp))])
				contra = np.vstack([erp[i][rep]['contra'] for i in range(len(erp))])

				T_obs, clusters, cluster_pv, HO = permutation_cluster_test([ipsi, contra], stat_fun = paired_t)
 
				ipsi_err = bootstrap(ipsi)
				contra_err = bootstrap(contra)

				ipsi = ipsi.mean(axis = 0)
				contra = contra.mean(axis = 0)

				plt.plot(times,ipsi,'g', label = 'ipsi ' + rep)
				plt.plot(times,contra,'r', label = 'contra ' + rep)
				plt.fill_between(times, ipsi + ipsi_err, ipsi - ipsi_err, alpha = 0.2, color = 'g')
				plt.fill_between(times, contra + contra_err, contra - contra_err, alpha = 0.2, color = 'r')

				mask = np.zeros(times.size,dtype = bool)
				for cl in np.array(clusters)[np.where(cluster_pv < 0.05)[0]]:
					mask[cl[0]] = True
				plt.fill_between(times, -.1, 0.1, where = mask == True, color = 'grey', label = 'p < 0.05')

				plt.axhline(y=0, ls = '--')
				plt.axvline(x=0.258, ls = '--', color = 'grey', label = 'onset gabor')
				plt.legend(loc='best', shadow = True)
				sns.despine(offset=10, trim = False)

		elif cnd == 'repeat':

			for i,rep in enumerate(rep_cond):

				ax = plt.subplot(2,2, 3 + i, title = cnd + rep[-1], ylabel = 'micro Volt', xlabel = 'Time (ms)', ylim = (-6, 6))

				ipsi = np.vstack([erp[i][rep]['ipsi'] for i in range(len(erp))])
				contra = np.vstack([erp[i][rep]['contra'] for i in range(len(erp))])

				T_obs, clusters, cluster_pv, HO = permutation_cluster_test([ipsi, contra], stat_fun = paired_t)
 
				ipsi_err = bootstrap(ipsi)
				contra_err = bootstrap(contra)

				ipsi = ipsi.mean(axis = 0)
				contra = contra.mean(axis = 0)

				plt.plot(times,ipsi,'g', label = 'ipsi ' + rep)
				plt.plot(times,contra,'r', label = 'contra ' + rep)
				plt.fill_between(times, ipsi + ipsi_err, ipsi - ipsi_err, alpha = 0.2, color = 'g')
				plt.fill_between(times, contra + contra_err, contra - contra_err, alpha = 0.2, color = 'r')

				mask = np.zeros(times.size,dtype = bool)
				for cl in np.array(clusters)[np.where(cluster_pv < 0.05)[0]]:
					mask[cl[0]] = True
				plt.fill_between(times, -.1, 0.1, where = mask == True, color = 'grey', label = 'p < 0.05')

				plt.axhline(y=0, ls = '--')
				plt.axvline(x=0.258, ls = '--', color = 'grey', label = 'onset gabor')
				plt.legend(loc='best', shadow = True)
				sns.despine(offset=10, trim = False)

			
	plt.savefig('/Users/dirk/Desktop/suppression/erp/{}/figs/ipsi_contra_group.pdf'.format(header))		
	plt.close()	

def plotCTFSlopes(subject_id, header):		
	'''
	'''

		


	#plot_index = [1,3]
	for cnd_idx, power in enumerate(['evoked','total']):
		
		if plot == 'group':
			slopes = self.bootstrapSlopes(subject_id, all_slopes, power, info, 10000)
			
			plt.figure(figsize= (20,10))
			ax = plt.subplot(1,1,1)
			plt.xlim(info['times'][0], info['times'][-1])
			plt.ylim(-0.3, 0.3)
			plt.title(power + ' power across time')
			plt.axhline(y=0, xmin = 0, xmax = 1, color = 'black')

			for i, cond in enumerate(info['conditions']):
				dat = slopes[cond]['M']
				error = slopes[cond]['SE']
				plt.plot(info['times'], dat, color = ['g','r','y','b'][i], label = cond)
				plt.fill_between(info['times'], dat + error, dat - error, alpha = 0.2, color = ['g','r','y','b'][i])
			
			#if len(info['conditions']) > 1:
			#	dif = slopes[info['conditions'][0]]['slopes'] - slopes[info['conditions'][1]]['slopes']
			#	zmap_thresh = self.clusterPerm(dif)
			#	plt.fill_between(info['times'], -0.002, 0.002, where = zmap_thresh != 0, color = 'grey')
				
			plt.legend(loc='upper right', shadow=True)
			#plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder,self.decoding,'figs'], filename = '{}_{}_slopes_group.png'.format(band,power)))
			plt.savefig(self.FolderTrackerCTF('ctf', extension = [self.channel_folder,self.decoding,'figs'], filename = '{}_{}_slopes_group.pdf'.format(band,power)))
			plt.close()	

		elif plot == 'individual':										
			idx = [1,2,5,6,9,10,13,14]
			for i, sbjct in enumerate(subject_id):
				if i % 8 == 0: 
					plt.figure(figsize= (15,10))
					idx_cntr = 0	
				ax = plt.subplot(7,2,idx[idx_cntr],title = 'SUBJECT ' + str(sbjct), ylabel = 'CTF Slope', xlim = (info['times'][0],info['times'][-1]), ylim = (-0.3,0.3))
				for j, cond in enumerate(info['conditions']): 	
					plt.plot(info['times'], all_slopes[i][cond][power].squeeze(), color = ['g','r','y','b'][j], label = cond)	
				plt.legend(loc='upper right', frameon=False)	
				plt.axhline(y=0, xmin=info['times'][0], xmax=info['times'][-1], color = 'black')
				idx_cntr += 1

			nr_plots = int(np.ceil(len(subject_id)/8.0))
			for i in range(nr_plots,0,-1):
				plt.savefig(self.FolderTrackerCTF('ctf', [self.channel_folder,self.decoding,'figs'],filename = '{}_{}_slopes_ind_{}.pdf'.format(band,power,str(i))))
				plt.close()	