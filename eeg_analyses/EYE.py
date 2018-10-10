"""
analyze Eyetracker data

Created by Dirk van Moorselaar on 24-08-2017.
Copyright (c) 2017 DvM. All rights reserved.
"""

import glob
import os
import pickle
import warnings
import matplotlib
matplotlib.use('agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from IPython import embed
from math import degrees, radians, cos, sin, atan2, floor, ceil
from itertools import groupby
from operator import itemgetter
from copy import copy
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import PatchCollection
from scipy.signal import savgol_filter
from FolderStructure import FolderStructure
from pygazeanalyser.edfreader import *
from pygazeanalyser.eyetribereader import *
from pygazeanalyser.detectors import *
from pygazeanalyser.gazeplotter import *

warnings.filterwarnings('ignore')

class EYE(FolderStructure):

	def __init__(self, viewing_dist = 60, screen_res = (1920, 1080), screen_h = 30, sfreq = 500):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		self.view_dist = viewing_dist
		self.scr_res = screen_res
		self.scr_h = screen_h
		self.sfreq = sfreq

	def readEyeData(self, sj, eye_files = 'all', beh_files = 'all', start = 'start_trial'):
		''' 

		Reads in eyetracker and behavioral file for subsequent processing 

		Arguments
		- - - - - 

		sj (int): subject number
		eye_files (list | int): list of asc files. If int reads in all files for given subject in eye folder 
		beh_files (list | str): list of csv files. If all reads in all files for given subject in beh folder
		start (str): string indexing the start of the trial in asc file 

		Returns
		- - - -

		eye (array): array of dictionaries as created by pygazeanalyzer
		beh (Dataframe): pandas object with all logged variables

		'''

		# read in behavior and eyetracking data
		if eye_files == 'all':
			eye_files = glob.glob(self.FolderTracker(extension = ['eye','raw'],filename = 'sub_{}_session_*.asc'.format(sj)))	
			beh_files = glob.glob(self.FolderTracker(extension = ['beh','raw'],filename = 'subject-{}_ses_*.csv'.format(sj)))
			# if eye file does not exit remove beh file 
			if len(beh_files) > len(eye_files):
				eye_sessions = [int(file[-5]) for file in eye_files]
				beh_sessions = [int(file[-5]) for file in beh_files]
				for i, ses in enumerate(beh_sessions):
					if ses not in eye_sessions:
						beh_files.pop(i)
		
		if eye_files[0][-3:] == 'tsv':			
			eye = [read_eyetribe(file, start = start) for file in eye_files]
		elif eye_files[0][-3:] == 'asc':	
			eye = [read_edf(file, start = start) for file in eye_files]
		eye = np.array(eye[0]) if len(eye_files) == 1 else np.hstack(eye)
		beh = pd.concat([pd.read_csv(file) for file in beh_files])

		# check whether each beh trial is logged within eye 
		if eye.shape[0] < beh.shape[0]:
			print 'Trials in beh and eye do not match. Trials removed from beh'
			eye_trials = []

			for i, trial in enumerate(eye):
				for event in trial['events']['msg']:
					if 'end trial' in event[1]:
						trial_nr = int(''.join(filter(str.isdigit, event[1])))
						# control for OpenSesame trial counter
						eye_trials.append(trial_nr + 1)
						if trial_nr + 1 not in beh['nr_trials'].values:
							print trial_nr

			eye_mask = np.in1d(beh['nr_trials'].values, eye_trials)
			beh = beh[np.array(eye_mask)]		

		# remove practice trials from eye and beh data
		eye = eye[np.array(beh['practice'] == 'no')]
		beh = beh[beh['practice'] == 'no']

		return eye, beh

	def getXY(self, eye, start, end, start_event = 'Onset placeholder'):
		''' 
		getXY takes edfreader dictionary as input and returns x, y coordinates in pixels. 
		Returned arrays have zeros (i.e. blinks) inserted on missing data points (e.g. trials shorter than specified interval).
		These blinks can be set to nan later by using setXY()

		Arguments
		- - - - - 
		eye (dict): list of dictionaries as returned by edfreader
		start (int): start of trial (ms)
		end( int): end of trial (ms)
		start_event (str): event string that marks 0 ms in trial sequence

		Returns
		- - - -

		x (array): x coordinates in pixels with zeros on missing data points
		y (array): y coordinates in pixels with zeros on missing data points
		times (array): time points of returned array based on specified start and enpoint and sampling frequency
		'''	

		times = np.arange(start, end,1000/self.sfreq)
		# initiate x and y array  (INSERTS ARTIFICIAL BLINKS)
		x = np.zeros((len(eye),times.size))
		y = np.copy(x)

		# look for start_event in all logged events
		for i, trial in enumerate(eye):
			for event in trial['events']['msg']:
				if start_event in event[1]:

					# adjust trial times such that start_event is at 0 ms
					tr_times = trial['trackertime'] - event[0]

					# get x, y cordinates between start and end
					s, e = [np.argmin(abs(tr_times - t)) for t in (start,end)]
					x_ = np.array(trial['x'][s:e]) # array makes sure that any subsequent manipulations do not effect information in eye
					y_ = np.array(trial['y'][s:e])

					x[i,:x_.size] = x_
					y[i,:y_.size] = y_

		return x, y, times	

	def setXY(self, x, y, times, drift_correct = None, fix_range = 125):	
		''' 
		setXY modifies x and y coordinates based on SaccadeGlissadeDetection algorhytm.
		Noise and blink segments are set to nan.
		If drift_correct is specified x and y cordinates are shifted towards center based 
		on drift in fixation period. Drift correction is only performed on those trials where 
		no saccade is detected during fixation period and fixation is within a specific range
		(as specified by fix_range)
		

		Arguments
		- - - - -

		x (array): x coordinates in pixels 
		y (array): y coordinates in pixels 
		times (array): array of epoch time points
		drift_correct (None, tuple): if not None, tuple specifying start and end 
									point used for drift_correction
		fix_range (int): max distance from center for a fixation (corrects for measurement error)

		Returns
		- - - -

		x (array): x coordinates in pixels 
		y (array): y coordinates in pixels 
		'''	

		# initiate SGD class
		SD = SaccadeGlissadeDetection(self.sfreq)

		# set blink and noise trials to nan
		for i, (x_, y_) in enumerate(zip(x,y)):

			V, A = SD.calcVelocity(x_,y_)
			x_, y_, V, A = SD.noiseDetect(x_, y_, V, A)

			if drift_correct != None:
				s_idx, e_idx = [np.argmin(abs(times - t)) for t in drift_correct]

				x_d = np.array(x_[s_idx:e_idx])
				y_d = np.array(y_[s_idx:e_idx])
				fix_d = np.mean(np.sqrt((self.scr_res[0]/2-x_d)**2 + 
								(self.scr_res[1]/2-y_d)**2))

				# only corrrect if fixation period contains no missing data,
				# is within a specific range from fixation and has no saccades
				if not np.isnan(x_d).any():# and fix_d < fix_range: THIS NEEDS TO FIXED 
					nr_sac = SD.detectEvents(x_d, y_d)

					if nr_sac == 0: 
						x_ += (self.scr_res[0]/2) - x_d.mean()
						y_ += (self.scr_res[1]/2) - y_d.mean()
		
			x[i,:] = x_
			y[i,:] = y_

		return x, y	

	def saccadeVector(self, sj, start = -300, end = 800):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		fname = self.FolderTracker(['eye'],'saccade_info.pickle')

		# initiate saccade detection operator
		SD = SaccadeGlissadeDetection(self.sfreq)

		# collect x, y data 
		eye, beh = self.readEyeData(sj, 'all', 'all')
		x, y, times = self.getXY(eye, start = start, end = end)	

		# read in data (or create new dictionary)
		if os.path.isfile(fname): 
			with open(fname ,'rb') as handle:
				sac_d = pickle.load(handle)
		else:
			sac_d = {}

		# update dictionary
		sac_d.update({str(sj):{'target': 0, 'dist':0, 'other': 0, 'vector': []}})
		
		# loop over all trials
		for i, (x_, y_) in enumerate(zip(x,y)):

			# check whether at least one saccade was recorded
			sac = SD.detectEvents(x_, y_, output = 'dict')[0]
			if sac == {}: continue
			if type(sac) != dict:
				if np.isnan(sac): continue

			# for each saccade get the direction(s) in degrees; stored in a list
			v_info = [EYE.pointAngle((x_[sac[key][0]], y_[sac[key][0]]),
					(x_[sac[key][1]], y_[sac[key][1]])) for key in sac]

			# check direction of the saccade
			for d in v_info:
				# loop over all bins
				for j, binn in enumerate([(60,120),(120,180),(180,240),(240,300),(300,360),(0,60)]):
					if d[0] > binn[0] and d[0] < binn[1]:
						if j == beh['target_loc'].iloc[i]:
							sac_d[str(sj)]['target'] += 1
						elif j == beh['dist_loc'].iloc[i]:
							sac_d[str(sj)]['dist'] += 1
						else:
							sac_d[str(sj)]['other'] += 1

			# shift saccades as if stimuli of interest were presented on the same location
			v_info = [(EYE.shiftAngle(v[0], stim_pos = beh['target_loc'].iloc[i], 
				new_pos = 0, nr_stim = 6), v[1]) for v in v_info]

			# create trial vector
			trial_v = [np.array([v[1] * cos(radians(v[0])), v[1] * sin(radians(v[0]))]) for v in v_info]

			# average vectors together (in case of multiple saccades)
			v = trial_v[0] if len(trial_v) == 1 else np.mean(trial_v, axis = 0)
			sac_d[str(sj)]['vector'].append(v)

		sac_d[str(sj)]['vector'] = np.mean(sac_d[str(sj)]['vector'], axis = 0)	

		# plot pies
		plt.figure(figsize = (15,15))
		pies = [sac_d[str(sj)]['target'], sac_d[str(sj)]['dist'], sac_d[str(sj)]['other']]
		labels = ['target', 'dist', 'other']
		ax = plt.subplot(2,2, 1, title = 'Ind')
		plt.pie(pies, labels = labels, autopct='%.0f%%')

		ax = plt.subplot(2,2, 2, title = 'group')
		pies = np.mean(np.vstack([[sac_d[key]['target'], sac_d[key]['dist'], sac_d[key]['other']] for key in sac_d]), axis = 0)
		plt.pie(pies, labels = labels, autopct='%.0f%%')

		# plot 
		ax = plt.subplot(2,2, 3, title = 'Ind')
		
		plt.plot([0, sac_d[str(sj)]['vector'][0]/225.0], [0, sac_d[str(sj)]['vector'][1]/225.0], 'k-', lw=2)
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		plt.axvline(x=0, ls = '--', color = 'grey')	
		plt.axhline(y=0, ls = '--', color = 'grey')	

		ax = plt.subplot(2,2, 4, title = 'Group')
		vector = np.mean(np.vstack([sac_d[key]['vector'] for key in sac_d]), axis = 0)

		plt.plot([0, vector[0]/225.0], [0, vector[1]/225.0], 'k-', lw=2)
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		plt.axvline(x=0, ls = '--', color = 'grey')	
		plt.axhline(y=0, ls = '--', color = 'grey')	


		plt.savefig(self.FolderTracker(extension = ['eye','figs'],filename = 'pie_{}.pdf'.format(sj)))
		plt.close()	

		with open(fname ,'wb') as handle:
			pickle.dump(sac_d, handle)

	def eyeBinEEG(self, sj, session, start, end, drift_correct = (-300,0), start_event = '', extension = 'asc'):
		''' 

		Function is called during preprocessing of raw EEG data. If eye-tracking data is available,
		reads in behavior (.csv) and eye data (.asc). x, y data is then shifted to the center of the screen
		if no saccades are recorded during the specified period (to correct for recording drifts). Resulting 
		x, y data are then used to create deviation bins for each recorded trial.


		Arguments
		- - - - - 

		sj (int): subject number
		session (int): session number
		start (int): start point in seconds used for epoching
		end (int): end point in seconds used for epoching
		drift_correct (tuple): interval used for drift correction (shift fixation to center of screen) 

		Returns
		- - - -

		bins (array): 
		trial_nrs (array): trial numbers from beh file

		'''

		# check whether eye tracking data is availabe
		eye_file = glob.glob(self.FolderTracker(extension = ['eye','raw'], \
					filename = 'sub_{}_session_{}*.{}'.format(sj,session, extension)))
		
		# only run eye analysis if data is available for this session
		if eye_file == []: 
			bins = np.array([])
			trial_nrs = [] # CHECK THIS
		else:	
			# read eye and beh file (with removed practice trials from .asc file)
			beh_file = self.FolderTracker(extension = ['beh','raw'], \
										filename = 'subject-{}_ses_{}.csv'.format(sj,session))
			eye, beh = self.readEyeData(sj, eye_file, [beh_file])

			# collect x, y data 
			x, y, times = self.getXY(eye, start = start, end = end, start_event = start_event)	

			# create deviation bins for for each trial(after correction for drifts in fixation period)	
			x, y = self.setXY(x,y, times, drift_correct)
			bins = np.array(self.createAngleBins(x,y, 0,3,0.25, 40))
			trial_nrs = beh['nr_trials'].values

		return bins, trial_nrs

	def createAngleBins(self, x, y, start, stop, step, min_segment):
		''' 

		createAngleBins for each epoch calculates the deviation from fixation in degrees. 
		These deviation values are then summarized with a single value, specifying the maximum 
		deviation from from fixation in that trial

		Arguments
		- - - - - 

		x (array): x coordinates in pixels 
		y (array): y coordinates in pixels 
		start (int): min value in degrees used for binning
		stop (int): max value in degrees used for binning
		step (float): step value to go from min to max bin
		min_segment (float): min duration in msec to use a segment for binning

		Returns
		- - - -

		bins (list): list with for each epoch in x, y the max bin in degrees

		'''

		min_segment *= self.sfreq/1000.0

		bins = []

		for i, (x_, y_) in enumerate(zip(x, y)):

			angle = self.calculateAngle(x = x_, y = y_, xc = self.scr_res[0]/2, \
										yc = self.scr_res[1]/2, screen_h = self.scr_h,
										pix_v = self.scr_res[1],dist = self.view_dist)
			trial_bin = []
			for b in np.arange(start,stop,step):
				# get segments of data where deviation from fixation (angle) is larger than the current bin value (b)
				binned = np.where(angle > b)[0]
				segments = np.split(binned, np.where(np.diff(binned) != 1)[0]+1)

				# check whether segment exceeds min duration
				if np.where(np.array([s.size for s in segments])>min_segment)[0].size > 0:
					trial_bin.append(b)

			# insert max binning segment or nan (in case of a trial without data)		
			if trial_bin != []:	
				bins.append(max(trial_bin))		
			else:
				bins.append(np.nan)

		return bins	

	@staticmethod	
	def pointAngle(p1, p2):

		''' 

		Function calculates the CCW angle between point 1 and point 2,where point 1 
		is assumed to be the middle of the coordinate system (0,0) and a rigthwards 
		horizontal line has an angle of 0 degrees (i.e. a straight line upwards is 90 degrees). 
		Also returns the length between p1 and p2 via pythagorean theorem.

		Arguments
		- - - - - 

		p1 (tuple): x,y coordinates (assumed to be the center ) 
		p2 (tuple): x,y coordinates 

		Returns
		- - - -

		deg (float): CCW angle between p1 and p2
		dist (float): 
		
		'''
		
		# get length of adjacent and opposite side
		dx = p2[0] - p1[0]
		dy = p2[1] - p1[1]

		# calculate CCW angle
		rad = atan2(dy,dx)
		deg = (360 - round(degrees(rad),1)) % 360

		# calculate distance
		dist = round(np.sqrt(dx**2 + dy**2),1)

		return deg, dist

	@staticmethod	
	def shiftAngle(angle, stim_pos = 0, new_pos = 0, nr_stim = 6):

		''' 

		Arguments
		- - - - - 

		angle (float): angle in degrees
		stim_pos (int): position of stimulus in circular space (0 = top; corresponds to 90 degrees)
		new_pos (int): position of stimulus after shifting (0 = top)
		nr_stim (int): nr of stimuli presented

		Returns
		- - - -

		angle (float): shifted angle
		
		'''

		# use stimulus settting to determine how many degrees the angle needs to be shifted 
		step = 360 / nr_stim
		shift = (new_pos - stim_pos) * step

		# shift and correct for circular space
		angle = (angle + shift) % 360

		return angle


	def calculateAngle(self, x, y, xc = 960, yc = 540, screen_h = 30.0, pix_v = 1080, dist = 60):
		''' 

		calculateAngle calculates the number of degrees that correspond to a single pixel (typically in the order of 0.03).
		Visual angle is than calculated based on the eucledian distance in pixels between each (x,y) coordinate and (xc,yc).

		Arguments
		- - - - - 

		x (array): x coordinates in pixels 
		y (array): y coordinates in pixels 
		xc (int): center of stimulus display in x direction (0 at most left) 
		yc (int): center of stimulus display in y direction (0 at top)
		screen_h (int | float): height of the screen in cm
		pix_v (int): number of pixels in vertical direction
		dist (int| float): viewing distance in cm from the screen

		Returns
		- - - -

		visual_angle (array): array of visual angle between (x,y) and (xc, yc) per timepoint

		'''

		# calculate visual angle of a single pixel
		deg_per_px = degrees(atan2(.5 * screen_h, dist)) / (.5 * pix_v)

		# calculate eucledian distance from specified center
		pix_eye = np.sqrt((x - xc)**2 + (y - yc)**2)

		# transform pixels to visual degrees
		visual_angle = pix_eye * deg_per_px

		return visual_angle	

	def plotSaccades(self, sj, start = -300, end = 800):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		eye, beh = self.readEyeData(sj)
		x, y = self.getXY(eye, start = start, end = end)

		SD = SaccadeGlissadeDetection(500.0)

		for i, (x_, y_) in enumerate(zip(x, y)):
			nr_sac = SD.detectEvents(x_,y_)
			if nr_sac > 0:
				drawTaskDisplay(beh.iloc[i]['target_loc'], beh.iloc[i]['dist_loc'], beh.iloc[i]['condition'])
				draw_raw(x_,y_,(1920,1080), imagefile = 'trial_display.png')
				plt.savefig('saccade{}.png'.format(i))
				plt.close()
		

	def driftCheck(self, sj, start = -300, end = 0, condition = ['DvTr_0','DvTr_3','DvTv_0','DvTv_3'], loc = 'target_loc'):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		# check whether dictionary for this location type already exists
		try:
			with open(self.FolderTracker(['eye','analysis'], filename = 'drift_{}.pickle'.format(loc)),'rb') as handle:
				info = pickle.load(handle)
		except:
			info = {}	

		info.update({str(sj):{}})		

		# read in eye data and get x,y coordinates
		eye, beh = self.readEyeData(sj)

		x, y = self.getXY(eye, start = start, end = end)
		x, y = self.setXY(x,y, drift_correct = None)

		# get mean x,y coordinate per condition
		locs = np.unique(beh[loc])

		for cnd in condition:
			info[str(sj)].update({cnd:{}})
			for l in locs:

				x_ = np.nanmean(x[(beh['condition'] == cnd) * (beh[loc] == l)])
				y_ = np.nanmean(y[(beh['condition'] == cnd) * (beh[loc] == l)])

				info[str(sj)][cnd].update({'loc_{}'.format(l):(x_,y_)})

		with open(self.FolderTracker(['eye','analysis'], filename = 'drift_{}.pickle'.format(loc)),'wb') as handle:
			pickle.dump(info,handle)

	def conditionDrift(self, sj, condition = ('block_type', 'repetition')):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		all_eye = {}

		for sj in [3,4,6,7,9,10,11,12,17,18,19,20,21,22]:

			all_eye.update({str(sj):{}})

			# read in eye data and get x,y coordinates
			eye, beh = self.readEyeData(sj = sj, practice = 24)
			x, y = self.getXY(eye, start = -300, end = 0)
			x, y = self.setXY(x,y, drift_correct = None)

			# create condition array
			condition = beh['block_type'].values + np.array(beh['repetition'],dtype = str)

			# start by analyzing target loc
			for cnd in ['DvTv0','DvTv3','DvTr0','DvTr3']:
				for loc in range(6):

					idx = (condition == cnd) * beh['target_loc'] == loc

					x_ = np.nanmean(x[idx])
					y_ = np.nanmean(y[idx])

					all_eye[str(sj)].update({'{}_loc{}'.format(cnd,loc): {'x': x_,'y': y_}})

		all_eye.pop('17', None)		

		for key in all_eye[str(3)].keys():
			globals().update( locals() )

			x_n = np.mean([all_eye[sj][key]['x'] for sj in all_eye.keys()])
			y_n = np.mean([all_eye[sj][key]['y'] for sj in all_eye.keys()])		

			plt.scatter(x_n, y_n, marker = ['o','v','>','<','s','p'][int(key[-1])], label = key)

		#plt.xlim(645, 1275)
		#plt.ylim(225, 855)
		plt.scatter(1920/2,1080/2, marker = 'x', label ='fix')

		plt.legend(loc = 'best')
		plt.show()

	@staticmethod
	def degreesToPixels(h = 30 , d = 60, r = 1080):
		'''

		screen_h = 30, pix_v = 1080, dist = 60)
		h = monitor height in cm
		d = distance between monitor and observer in cm
		r = vertical resolution of the monitor
		'''

		deg_per_pix = degrees(atan2(.5*h, d)) / (.5*r)

		return deg_per_pix	

	def plotFixation(self, sj):
		''' 

		Arguments
		- - - - - 


		Returns
		- - - -

		'''

		colors = cm.get_cmap().colors
		colors = np.array(colors)[::19]

		# read in eye data and get x,y coordinates
		eye, beh = self.readEyeData(sj = sj, practice = 24)
		x, y = self.getXY(eye, start = -300, end = 800)
		x, y = self.setXY(x,y, drift_correct = 200)

		# initiate plot
		a = plt.subplot(111, aspect='equal')
		conditions = np.unique(beh['condition'])

		# plot  fixation variance
		ells = [Ellipse((np.nanmean(x_), np.nanmean(y_)), np.nanstd(x_), np.nanstd(y_), fill = False, alpha = 0.1, color = colors[np.where(conditions == beh['condition'][i])[0][0]]) for i, (x_,y_) in enumerate(zip(x,y))]
		for e in ells:
			e.set_clip_box(a.bbox)
			a.add_artist(e)

		# plot task display
		display = circles(np.array([960.,765.2,765.2,956.,1154.8,1154.8]),np.array([765.,652.5,427.5,315.,427.5,652.5]),np.array([75,75,75,75,75,75]), color = 'black', alpha = 0.1)
			
		plt.xlim(645, 1275)
		plt.ylim(225, 855)

		plt.savefig(self.FolderTracker(['eye', 'figs'],'{}_fix_corr.pdf'.format(sj)))		
		plt.close()	

class SaccadeGlissadeDetection(object):
	
	'''
	Class based on algorhythm and Matlab code presented in Nystrom and Holmquist (2010): 
	An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking data.
	The velocity-based algorithm is endowed with an adaptive noise-dependent velocity threshold 
	(and thus does not require user input) and is designed with the explicit goal of detecting glissades

	'''

	
	def __init__(self, sfreq):
		
		self.sfreq = sfreq
		self.min_sac = 0.01
		self.sfreq = 500.0 # needs to be a float!!!!

	def detectEvents(self, x, y, output = 'mask'):
		'''
		detectEvents calls all functions to classify recorded gaze points into periods of fixation, saccade, blink and noise 

		Arguments
		- - - - - 

		x (array): x coordinates in pixels
		y (array): y coordinates in pixels
		output (str): 'mask' returns boolean array (True = saccade present); 'dict' returns dict with saccade info

		Returns
		- - - -

		'''	

		if x.ndim == 1:
			x = x.reshape(1,-1)
			y = y.reshape(1,-1)

		sacc = np.empty(x.shape[0], dtype = dict)

		for i, (x_, y_) in enumerate(zip(x, y)):

			V, A = self.calcVelocity(x_,y_)
			x_, y_, V, A = self.noiseDetect(x_, y_, V, A)
			self.estimateThresh(V)
			if self.peak_thresh != None:
				sacc[i] = self.saccadeDetection(V, output = output)	
			else:
				sacc[i] = np.nan	

		if output == 'mask':
			sacc = np.array(sacc, dtype = bool)
				
		return sacc

	def calcVelocity(self, x, y):
		''' 
		calcvelocity calculates the angular velocity (deg/sec) and angular acceleration (deg/sec2) based on x and y coordinates.
		Velocity and acceleration are calculated by taking the first and second order derivative Savitzky-Golay filter of raw x and y coordinates.
		Total angular velocity and acceleration are then calculated as the Euclidian distancy of the x and y components multiplied 
		by the sampling frequency and degrees per pixel.

		Arguments
		- - - - - 

		x (array): x coordinates in pixels
		y (array): y coordinates in pixels

		Returns
		- - - -

		V (array): velocity in degrees per second
		A (array): acceleration in degrees per second2

		'''		
	
		N = 2 											# order of poynomial
		span = np.ceil(self.min_sac * self.sfreq)		# span of filter
		F = int(2 * span - 1)							# window length

		# calculate the velocity and acceleration

		x_ = savgol_filter(x, F, N, deriv = 0)
		y_ = savgol_filter(y, F, N, deriv = 0)

		V_x = savgol_filter(x, F, N, deriv = 1)
		V_y = savgol_filter(y, F, N, deriv = 1)
		V = np.sqrt(V_x**2 + V_y**2) *  EYE.degreesToPixels() * self.sfreq  

		A_x = savgol_filter(x, F, N, deriv = 2)
		A_y = savgol_filter(y, F, N, deriv = 2)
		A = np.sqrt(A_x**2 + A_y**2) * EYE.degreesToPixels() * self.sfreq # CHECK WHETHER CALCULATION OF A IS CORRECT!!!!!!!

		return V, A 

	def noiseDetect(self, x, y, V, A, V_thresh = 1000, A_thresh = 100000):	
		''' 
		noiseDetect detect noise within a trial. Blinks, as indicated by (0,0) coordinates, 
		and trial segments where velocity or acceleration are above a specified threshold
		and therefore considered as physiologically impossible are considered as noise. 
		To make sure that samples before the start and after the end of a noisy period are not 
		left to contaminate the data, the function searches for onset and offset of noise 
		(i.e. the median value of velocity over the whole trial). 

		Arguments
		- - - - - 

		x (array): x coordinates in pixels
		y (array): y coordinates in pixels
		V (array): velocity data
		A (array): acceleration data
		V_thresh (int): threshold for physiologically impossible eye movements 
						(velocity; default based on Bahill et al., 1981)
		A_thresh (int): threshold for physiologically impossible eye movements 
						(acceleration; default based on Bahill et al., 1981)

		Returns
		- - - -

		x (array): x coordinates in pixels with nan inserted on noise segments
		y (array): y coordinates in pixels with nan inserted on noise segments
		V (array): velocity data with nan inserted on noise segments
		A (array): acceleration data with nan inserted on noise segments
		'''

		on_off = np.median(V)*2
		
		# detect noise segments 
		noise = np.where((x <= 0) | (y <= 0) | (V > V_thresh) | (abs(A) > A_thresh))[0]
		noise_seg = np.split(noise, np.where(np.diff(noise) != 1)[0]+1)

		# loop over segments 
		on_off_seg = []
		for seg in noise_seg:
			if seg.size > 0:
				# go back in time 
				for i, v in enumerate(np.flipud(V[:min(seg)])):
					if v >= on_off:
						on_off_seg.append(min(seg) - i - 1)
					else:
						break	

				# go forward in time
				for i, v in enumerate(V[max(seg) + 1:]):
					if v >= on_off:
						on_off_seg.append(max(seg) + i + 1)
					else:
						break	

		# add segments to noise			
		noise = np.array(np.hstack((noise,on_off_seg)), dtype = int)	

		if noise.size > 0:
			
			x[noise] = np.nan
			y[noise] = np.nan
			V[noise] = np.nan
			A[noise] = np.nan

		return x, y, V, A

	def estimateThresh(self, V, peak_thresh = 100, min_fix = 0.04):	
		''' 
		estimateThresh sets a data driven peak velocity threshold for saccade labelling. 
		An iterative procedure is used in which all samples are selected with a velocity 
		smaller than peak_velocity. The average and standard deviation of these samples 
		are then taken and used to set a new peak threshold with the following formula: mu + 6*sd. 
		This iterative procedure is repeated until the difference between the current and the previous peak is < 1
		
		Arguments
		- - - - - 

		V (array): velocity data
		peak_thresh (int): initial peak velocity threshold to initiate iteration procedure
		min_fix (float): minimual duration in seconds for a segment of data to be considered a fixation

		Returns
		- - - -

		peak_thresh (return in place)

		'''	
		
		# Step 3: velocity threshold estimation
		old_thresh = float('inf')
		cent_fix = min_fix * self.sfreq/6    			# used to extract the center of the fixation

		flip = 0
		while abs(peak_thresh - old_thresh) > 1:

			# control for infinite flipping
			if peak_thresh > old_thresh:
				flip += 1 
				if flip > 50:
					# ADD CODE TO COMBINE ALL OLD PEAK THESH????
					print 'Peak_thresh kept flipping, broke from infinite loop'
					break

			old_thresh = peak_thresh
			fix_samp = np.where(V <= peak_thresh)[0]	
			fix_seg = np.split(fix_samp, np.where(np.diff(fix_samp) != 1)[0]+1)
			fix_noise = []
			
			# epoch should contain at least one fixation period
			if sum(np.array([s.size/self.sfreq for s in fix_seg]) >= min_fix) == 0:
				peak_thresh = None
				sacc_thresh = None
				print ('Segment does not contain sufficient samples for event detection')
				break
			
			# loop over all possible fixations
			for seg in fix_seg:

				# check whether fix duration exceeds minimum duration
				if (seg.size / self.sfreq) < min_fix:
					continue

				# extract the samples from the center of fixation  (exclude outer portions)
				f_noise = V[int(floor(seg[0] + cent_fix)): int(ceil(seg[-1] - cent_fix) + 1)]	
				fix_noise.append(f_noise)

				
			fix_noise = fix_noise[0] if len(fix_noise) == 1 else np.hstack(fix_noise)	# WHAT IF LEN FIX NOISE IS 0

			mean_noise = np.nanmean(fix_noise)	
			std_noise = np.nanstd(fix_noise)
		
			#adjust the peak velocity threshold based on the noise level
			peak_thresh = mean_noise + 6 * std_noise
			sacc_thresh = mean_noise + 3 * std_noise
			 
		self.peak_thresh = peak_thresh	
		self.sacc_thresh = sacc_thresh

	def saccadeDetection(self, V, min_sac = 0.01, min_fix = 0.04, output = 'mask'):	
		''' 
		saccadeDetection detects saccades and glissades within an epoched segment of data. 
		For each detected saccade peak (based on thresh peak set by estimateThresh searches 
		backward and forward in time for the saccade onset and offset. Saccade onset is defined 
		as the first sample that goes below the saccade onset threshold (via detection of local minimun)

		Glissade detection is based on two mutual exlusive velocity (low and high) criteria. The onset
		of the glissade is defined as the offset of the preceding saccade. Similar to saccade detection, 
		glissade offset is defined as the first sample that goes below the saccadic threshold and where
		velocity starts to increase again (if understood correct).  

		THIS FUNCTION NEEDS TO BE ADJUSTED!!!
		NOW DOES ONLY DETECT GLISSADES BUT INFO IS NOT RETURNED
	
		Arguments
		- - - - - 


		Returns
		- - - -

		'''	

		# create array to store indices of detected saccades and glissades
		saccade_idx = []
		glissade_idx = []

		# initiate saccade counter
		nr_sac = 0
		sac_dict = {}
		gliss_off = []  # we start with missing glissades

		# start by getting segments of data with velocities above the selected peak velocity
		poss_sac = np.where(V > self.peak_thresh)[0]	
		sac_seg = np.split(poss_sac, np.where(np.diff(poss_sac) != 1)[0]+1)

		for seg in sac_seg:

			# PART 1: DETECT SACCADES

			# if the peak consists of less than 1/6 of the min_sac duration, it is probably noise
			if seg.size <= ceil(min_sac/6.0 * self.sfreq): continue
			
			# check whether the peak is already included in the previous saccade (is possible for glissades) 
			if nr_sac > 0 and gliss_off != []:
				if len(set(seg).intersection(np.hstack((saccade_idx,glissade_idx)))) > 0:
					continue

			# get idx of saccade onset
			onset = np.where((V[:seg[0]] <= self.sacc_thresh) * 
							 (np.hstack((np.diff(V[:seg[0]]),0)) >= 0))[0]
			if onset.size == 0: continue
			sac_on = onset[-1]

			# calculate local fix noise (adaptive part), used for saccade offset
			V_local = V[sac_on :int(ceil(max(0,sac_on - min_fix * self.sfreq))) :-1] # back in time
			V_noise = V_local.mean() + 3 * V_local.std()
			sacc_thresh = V_noise * 0.3 + self.sacc_thresh * 0.7

			# check whether the local V noise exceeds the peak V threshold 
			# (i.e. whether saccade is preceded by period of stillness)
			if V_noise > self.peak_thresh: continue
				
			# detect end of saccade (without glissade)
			offset = np.where((V[seg[-1]:] <= sacc_thresh) * 
							  (np.hstack((np.diff(V[seg[-1]:]), 0))	>= 0))[0]
			if offset.size == 0: continue
			sac_off = seg[-1] + offset[0]

			# make sure that saccade duration exceeds minimum duration and does not contain any nan value
			if (np.isnan(V[sac_on:sac_off]).any()) or \
				(sac_off - sac_on)/ self.sfreq < min_sac: continue

			# if all criteria are fulfilled the segment can be labelled as a saccade
			nr_sac += 1	
			saccade_idx = np.array(np.hstack((saccade_idx, np.arange(sac_on,sac_off))),dtype = int)
			sac_dict.update({str(nr_sac):(sac_on, sac_off)})
			
			# PART 2: DETECT GLISSADES (high and low velocity glissades are mutually exclusive)

			# only search for glissade peaks in a window smaller than fix duration after saccade end
			poss_gliss = V[sac_off: int(ceil(min(sac_off + min_fix * self.sfreq, V.size)))]
			if poss_gliss.size == 0: continue
		
			# option 1: low velocity criteria
			# detect only 'complete' peaks (i.e. with a beginning and end)
			peak_idx_w = np.array(poss_gliss >= sacc_thresh, dtype =int)
			end_idx = np.where(np.diff(peak_idx_w) != 0)[0]
			if end_idx.size > 1:
				gliss_w_off = end_idx[1:end_idx.size:2][-1]
			else:
				gliss_w_off = []	

			# option 2: high velocity glissade	
			peak_idx_s = np.array(poss_gliss >= self.peak_thresh, dtype =int)
			gliss_s_off = np.where(peak_idx_s > 0)[0]
			if gliss_s_off.size > 0:
				gliss_s_off = gliss_s_off[-1] 

			# make sure that saccade amplitude is larger than the glissade amplitude	
			if max(poss_gliss) > max(V[sac_on:sac_off]):
				gliss_w_off = gliss_s_off = []

			# if a glissade is detected, get the offset of the glissade
			if gliss_w_off != []:
				gliss_off = sac_off + gliss_w_off
				gliss_off += np.where(np.diff(V[gliss_off:]) >= 0)[0][0] - 1
	
				if np.isnan(V[sac_off:gliss_off]).any() or \
					(gliss_off - sac_off)/ self.sfreq > 2 * min_fix:
						gliss_off = []
				else:
					glissade_idx = np.array(np.hstack((glissade_idx, np.arange(sac_off,gliss_off))), dtype = int)		

		if output == 'mask':	
			return nr_sac
		else:
			return sac_dict		

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
	"""
	Make a scatter of circles plot of x vs y, where x and y are sequence 
	like objects of the same lengths. The size of circles are in data scale.

	Parameters
	----------
	x,y : scalar or array_like, shape (n, )
	    Input data
	s : scalar or array_like, shape (n, ) 
	    Radius of circle in data unit.
	c : color or sequence of color, optional, default : 'b'
	    `c` can be a single color format string, or a sequence of color
	    specifications of length `N`, or a sequence of `N` numbers to be
	    mapped to colors using the `cmap` and `norm` specified via kwargs.
	    Note that `c` should not be a single numeric RGB or RGBA sequence 
	    because that is indistinguishable from an array of values
	    to be colormapped. (If you insist, use `color` instead.)  
	    `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
	vmin, vmax : scalar, optional, default: None
	    `vmin` and `vmax` are used in conjunction with `norm` to normalize
	    luminance data.  If either are `None`, the min and max of the
	    color array is used.
	kwargs : `~matplotlib.collections.Collection` properties
	    Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
	    norm, cmap, transform, etc.

	Returns
	-------
	paths : `~matplotlib.collections.PathCollection`

	Examples
	--------
	a = np.arange(11)
	circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
	plt.colorbar()

	License
	--------
	This code is under [The BSD 3-Clause License]
	(http://opensource.org/licenses/BSD-3-Clause)
	"""

	try:
		basestring
	except NameError:
		basestring = str

	if np.isscalar(c):
		kwargs.setdefault('color', c)
		c = None
	if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
	if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
	if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
	if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

	patches = [Circle((x_, y_), s_, fill = False) for x_, y_, s_ in np.broadcast(x, y, s)]
	collection = PatchCollection(patches, **kwargs)
	if c is not None:
		collection.set_array(np.asarray(c))
		collection.set_clim(vmin, vmax)

	ax = plt.gca()
	ax.add_collection(collection)
	ax.autoscale_view()
	if c is not None:
		plt.sci(collection)

	return collection

def drawTaskDisplay(target_loc = None, dist_loc = None, cnd = None):

	fig, ax = draw_display((1920,1080), imagefile=None)	
	x_list = np.array([960.,765.2,765.2,956.,1154.8,1154.8])
	y_list = np.array([765.,652.5,427.5,315.,427.5,652.5])

	display = circles(x_list, y_list, np.array([75,75,75,75,75,75]),color = 'white', alpha = 0.5)

	plt.text(960, 900, cnd, ha = 'center', size = 20, color = 'white')
	
	if target_loc != None:
		plt.text(x_list[target_loc],y_list[target_loc],'T',ha = 'center',va = 'center', size = 20)

	if dist_loc != None:
		plt.text(x_list[dist_loc],y_list[dist_loc],'D',ha = 'center',va = 'center', size = 20)
	
	plt.savefig('trial_display.png')
	plt.close()

if __name__ == '__main__':

	project_folder = '/home/dvmoors1/big_brother/Dist_suppression'
	os.chdir(project_folder) 

	eye = EYE()

	for sj in [2,3,4,5,6,7,8,9,10,11,12,13,15,17,18,19,20,21,22,23,24]:
		print sj
		eye.saccadeVector(sj)
		#eye.eyeBinEEG(17, '1', -300, 800, drift_correct = (-300,0))
		#eye.eyeDetectEEG(sj, sessions = [1, 2],start = -300, end = 800, drift_correct = (-300,0))
		#eye.plotSaccades(sj)
		#eye.driftCheck(sj = sj)

	
	#eye.conditionDrift(sj = 22)
	#eye.plotFixation(sj = 19)
	#eye.createAngleBins()








