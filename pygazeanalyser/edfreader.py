# EDF Reader
#
# Does not actually read EDFs directly, but the ASC files that are produced
# by edf2asc (SR Research). Information on saccades, fixations and blinks is
# read from the EDF, therefore based on SR Research algorithms. For optimal
# event detection, it might be better to use a different algorithm, e.g.
# Nystrom, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
# saccade, and glissade detection in eyetracking data. Behavior Research
# Methods, 42, 188-204. doi:10.3758/BRM.42.1.188
#
# (C) Edwin Dalmaijer, 2013-2014
# edwin.dalmaijer@gmail.com
#
# version 2 (24-Apr-2014)

__author__ = "Edwin Dalmaijer"


import copy
import os.path
import re

import numpy


def replace_missing(value, missing=0.0):
	
	"""Returns missing code if passed value is missing, or the passed value
	if it is not missing; a missing value in the EDF contains only a
	period, no numbers; NOTE: this function is for gaze position values
	only, NOT for pupil size, as missing pupil size data is coded '0.0'
	
	arguments
	value		-	either an X or a Y gaze position value (NOT pupil
					size! This is coded '0.0')
	
	keyword arguments
	missing		-	the missing code to replace missing data with
					(default = 0.0)
	
	returns
	value		-	either a missing code, or a float value of the
					gaze position
	"""
	
	if value.replace(' ','') == '.':
		return missing
	else:
		return float(value)

def read_edf(filename, start, stop=None, trial_info=None, missing=0.0, debug=False):
	
	"""Returns a list with dicts for every trial. A trial dict contains the
	following keys:
		x		-	numpy array of x positions
		y		-	numpy array of y positions
		size		-	numpy array of pupil size
		time		-	numpy array of timestamps, t=0 at trialstart
		trackertime	-	numpy array of timestamps, according to EDF
		events	-	dict with the following keys:
						Sfix	-	list of lists, each containing [starttime]
						Ssac	-	list of lists, each containing [starttime]
						Sblk	-	list of lists, each containing [starttime]
						Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
						Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
						Eblk	-	list of lists, each containing [starttime, endtime, duration]
						msg	-	list of lists, each containing [time, message]
						NOTE: timing is in EDF time!
	
	arguments
	filename		-	path to the file that has to be read
	start		-	trial start string
	
	keyword arguments
	stop			-	trial ending string (default = None)
	missing		-	value to be used for missing data (default = 0.0)
	debug		-	Boolean indicating if DEBUG mode should be on or off;
				if DEBUG mode is on, information on what the script
				currently is doing will be printed to the console
				(default = False)
	
	returns
	data			-	a list with a dict for every trial (see above)
	"""

	# # # # #
	# debug mode
	
	if debug:
		def message(msg):
			print(msg)
	else:
		def message(msg):
			pass
		
	
	# # # # #
	# file handling
	
	# check if the file exists
	if os.path.isfile(filename):
		# open file
		message("opening file '%s'" % filename)
		f = open(filename, 'r')
	# raise exception if the file does not exist
	else:
		raise Exception("Error in read_edf: file '%s' does not exist" % filename)
	
	# read file contents
	message("reading file '%s'" % filename)
	raw = f.readlines()
	
	# close file
	message("closing file '%s'" % filename)
	f.close()

	
	# # # # #
	# parse lines
	
	# variables
	data = []
	eye_trial_nr = []
	x = []
	y = []
	size = []
	time = []
	trackertime = []
	events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
	starttime = 0
	started = False
	trialend = False
	finalline = raw[-1]
	
	# loop through all lines
	for line in raw:
		
		# check if trial has already started
		if started:
			# only check for stop if there is one
			if stop != None:
				if stop in line:
					started = False
					trialend = True
			# check for new start otherwise
			else:
				if (start in line) or (line == finalline):
					started = True
					trialend = True
			
			# # # # #
			# trial ending
			
			if trialend:
				message("trialend %d; %d samples found" % (len(data),len(x)))
				# trial dict
				trial = {}
				trial['x'] = numpy.array(x)
				trial['y'] = numpy.array(y)
				trial['size'] = numpy.array(size)
				trial['time'] = numpy.array(time)
				trial['trackertime'] = numpy.array(trackertime)
				trial['events'] = copy.deepcopy(events)
				# add trial to data
				data.append(trial)
				# reset stuff
				x = []
				y = []
				size = []
				time = []
				trackertime = []
				found_trial_info = False
				for event in events['msg']:
					if trial_info in event[1]:
						trial = int(re.findall(r'\d+',event[1])[0])
						found_trial_info = True
						break
				if found_trial_info:
					eye_trial_nr.append(trial)
				else:
					eye_trial_nr.append(numpy.nan)
 
				events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
				trialend = False
				
		# check if the current line contains start message
		else:
			if start in line:
				message("trialstart %d" % len(data))
				# set started to True
				started = True
				# find starting time
				starttime = int(line[line.find('\t')+1:line.find(' ')])
		
		# # # # #
		# parse line
		
		if started:
			# message lines will start with MSG, followed by a tab, then a
			# timestamp, a space, and finally the message, e.g.:
			#	"MSG\t12345 something of importance here"
			if line[0:3] == "MSG":
				ms = line.find(" ") # message start
				t = int(line[4:ms]) # time
				m = line[ms+1:] # message
				events['msg'].append([t,m])
	
			# EDF event lines are constructed of 9 characters, followed by
			# tab separated values; these values MAY CONTAIN SPACES, but
			# these spaces are ignored by float() (thank you Python!)
					
			# fixation start
			elif line[0:4] == "SFIX":
				message("fixation start")
				l = line[9:]
				events['Sfix'].append(int(l))
			# fixation end
			elif line[0:4] == "EFIX":
				message("fixation end")
				l = line[9:]
				l = l.split('\t')
				st = int(l[0]) # starting time
				et = int(l[1]) # ending time
				dur = int(l[2]) # duration
				sx = replace_missing(l[3], missing=missing) # x position
				sy = replace_missing(l[4], missing=missing) # y position
				events['Efix'].append([st, et, dur, sx, sy])
			# saccade start
			elif line[0:5] == 'SSACC':
				message("saccade start")
				l = line[9:]
				events['Ssac'].append(int(l))
			# saccade end
			elif line[0:5] == "ESACC":
				message("saccade end")
				l = line[9:]
				l = l.split('\t')
				st = int(l[0]) # starting time
				et = int(l[1]) # endint time
				dur = int(l[2]) # duration
				sx = replace_missing(l[3], missing=missing) # start x position
				sy = replace_missing(l[4], missing=missing) # start y position
				ex = replace_missing(l[5], missing=missing) # end x position
				ey = replace_missing(l[6], missing=missing) # end y position
				events['Esac'].append([st, et, dur, sx, sy, ex, ey])
			# blink start
			elif line[0:6] == "SBLINK":
				message("blink start")
				l = line[9:]
				events['Sblk'].append(int(l))
			# blink end
			elif line[0:6] == "EBLINK":
				message("blink end")
				l = line[9:]
				l = l.split('\t')
				st = int(l[0])
				et = int(l[1])
				dur = int(l[2])
				events['Eblk'].append([st,et,dur])
			
			# regular lines will contain tab separated values, beginning with
			# a timestamp, follwed by the values that were asked to be stored
			# in the EDF and a mysterious '...'. Usually, this comes down to
			# timestamp, x, y, pupilsize, ...
			# e.g.: "985288\t  504.6\t  368.2\t 4933.0\t..."
			# NOTE: these values MAY CONTAIN SPACES, but these spaces are
			# ignored by float() (thank you Python!)
			else:
				# see if current line contains relevant data
				try:
					# split by tab
					l = line.split('\t')
					# if first entry is a timestamp, this should work
					int(l[0])
				except:
					message("line '%s' could not be parsed" % line)
					continue # skip this line

				# check missing
				if float(l[3]) == 0.0:
					l[1] = 0.0
					l[2] = 0.0
				
				# extract data
				x.append(float(l[1]))
				y.append(float(l[2]))
				size.append(float(l[3]))
				time.append(int(l[0])-starttime)
				trackertime.append(int(l[0]))
	
	
	# # # # #
	# return
	
	return data, eye_trial_nr

def read_edf_time_overlap(filename, start, stop, missing=0.0, debug=False):
	
	"""Due to overlapping between trials, trl_N+1 starts before trl_N stops 
	   First find indice of start and end, then extract data using scripts from read_edf 
	   TODO: add an argument to extract L or R or both
	"""

	"""Returns a list with dicts for every trial. A trial dict contains the
	following keys:
		x		-	numpy array of x positions
		y		-	numpy array of y positions
		size		-	numpy array of pupil size
		time		-	numpy array of timestamps, t=0 at trialstart
		trackertime	-	numpy array of timestamps, according to EDF
		events	-	dict with the following keys:
						Sfix	-	list of lists, each containing [starttime]
						Ssac	-	list of lists, each containing [starttime]
						Sblk	-	list of lists, each containing [starttime]
						Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
						Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
						Eblk	-	list of lists, each containing [starttime, endtime, duration]
						msg	-	list of lists, each containing [time, message]
						NOTE: timing is in EDF time!
	
	arguments
	filename		-	path to the file that has to be read
	start		-	trial start string
	
	keyword arguments
	stop			-	trial ending string (default = None)
	missing		-	value to be used for missing data (default = 0.0)
	debug		-	Boolean indicating if DEBUG mode should be on or off;
				if DEBUG mode is on, information on what the script
				currently is doing will be printed to the console
				(default = False)
	
	returns
	data			-	a list with a dict for every trial (see above)
	"""

	# # # # #
	# debug mode
	
	if debug:
		def message(msg):
			print(msg)
	else:
		def message(msg):
			pass

	# # # # #
	# file handling
	
	# check if the file exists
	if os.path.isfile(filename):
		# open file
		message("opening file '%s'" % filename)
		f = open(filename, 'r')
	# raise exception if the file does not exist
	else:
		raise Exception("Error in read_edf: file '%s' does not exist" % filename)
	
	# read file contents
	message("reading file '%s'" % filename)
	raw = f.readlines()
	
	# close file
	message("closing file '%s'" % filename)
	f.close()

	# find all trials info
	start_trial_info = []
	stop_trial_info  = []
	for idx, line in enumerate(raw): # eg, MSG	6581878 long_start_trial: 1
		if start in line:
			trial = {}
			trial['l_idx']  = idx
			trial['timing'] = int(line[4:line.find(' ')]) # might realize padding in the future
			trial['nr']     = int(line[line.find(':')+2:-1])
			start_trial_info.append(trial)
		if stop in line:
			trial = {}
			trial['l_idx']  = idx
			trial['timing'] = int(line[4:line.find(' ')])
			trial['nr']     = int(line[line.find(':')+2:-1])
			stop_trial_info.append(trial)
	
	# extract every trial in raw[s,e] 
	data = []
	eye_trial_nr = []
	for s, e in zip(start_trial_info, stop_trial_info):
		x = []
		y = []
		size = []
		time = []
		trackertime = []
		events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
		starttime = s['timing']

		if s['nr']!=e['nr']: 
			warnings.warn('--- NOTE: start_trial_{} and stop_trial not match ---'.format(s['nr']), UserWarning)
		else:
			eye_trial_nr.append(s['nr'])
			for line in raw[s['l_idx']:e['l_idx']]: # below is copied from read_edf
				# message lines will start with MSG, followed by a tab, then a
				# timestamp, a space, and finally the message, e.g.:
				#	"MSG\t12345 something of importance here"
				if line[0:3] == "MSG":
					ms = line.find(" ") # message start
					t = int(line[4:ms]) # time
					m = line[ms+1:] # message
					if start in line and t!=starttime: # only keep current trial start info
						m = 'next trl already starts'
					events['msg'].append([t,m])
					
				# EDF event lines are constructed of 9 characters, followed by
				# tab separated values; these values MAY CONTAIN SPACES, but
				# these spaces are ignored by float() (thank you Python!)
						
				# fixation start
				elif line[0:4] == "SFIX":
					message("fixation start")
					l = line[9:]
					events['Sfix'].append(int(l))
				# fixation end
				elif line[0:4] == "EFIX":
					message("fixation end")
					l = line[9:]
					l = l.split('\t')
					st = int(l[0]) # starting time
					et = int(l[1]) # ending time
					dur = int(l[2]) # duration
					sx = replace_missing(l[3], missing=missing) # x position
					sy = replace_missing(l[4], missing=missing) # y position
					events['Efix'].append([st, et, dur, sx, sy])
				# saccade start
				elif line[0:5] == 'SSACC':
					message("saccade start")
					l = line[9:]
					events['Ssac'].append(int(l))
				# saccade end
				elif line[0:5] == "ESACC":
					message("saccade end")
					l = line[9:]
					l = l.split('\t')
					st = int(l[0]) # starting time
					et = int(l[1]) # endint time
					dur = int(l[2]) # duration
					sx = replace_missing(l[3], missing=missing) # start x position
					sy = replace_missing(l[4], missing=missing) # start y position
					ex = replace_missing(l[5], missing=missing) # end x position
					ey = replace_missing(l[6], missing=missing) # end y position
					events['Esac'].append([st, et, dur, sx, sy, ex, ey])
				# blink start
				elif line[0:6] == "SBLINK":
					message("blink start")
					l = line[9:]
					events['Sblk'].append(int(l))
				# blink end
				elif line[0:6] == "EBLINK":
					message("blink end")
					l = line[9:]
					l = l.split('\t')
					st = int(l[0])
					et = int(l[1])
					dur = int(l[2])
					events['Eblk'].append([st,et,dur])
				
				# regular lines will contain tab separated values, beginning with
				# a timestamp, follwed by the values that were asked to be stored
				# in the EDF and a mysterious '...'. Usually, this comes down to
				# timestamp, x, y, pupilsize, ...
				# e.g.: "985288\t  504.6\t  368.2\t 4933.0\t..."
				# NOTE: these values MAY CONTAIN SPACES, but these spaces are
				# ignored by float() (thank you Python!)
				else:
					# see if current line contains relevant data
					try:
						# split by tab
						l = line.split('\t')
						# if first entry is a timestamp, this should work
						int(l[0])
					except:
						message("line '%s' could not be parsed" % line)
						continue # skip this line

					# check missing
					if float(l[3]) == 0.0:
						l[1] = 0.0
						l[2] = 0.0
					
					# extract data
					x.append(float(l[1]))
					y.append(float(l[2]))
					size.append(float(l[3]))
					time.append(int(l[0])-starttime)
					trackertime.append(int(l[0]))
			
			# trial dict
			trial = {}
			trial['x'] = numpy.array(x)
			trial['y'] = numpy.array(y)
			trial['size'] = numpy.array(size)
			trial['time'] = numpy.array(time)
			trial['trackertime'] = numpy.array(trackertime)
			trial['events'] = copy.deepcopy(events)
			# add trial to data
			data.append(trial)

	# # # # #
	# return
	return data, eye_trial_nr

# DEBUG #
if __name__ == "__main__":
	# start: MSG	3120773 TRIALNR 8 TARX 662 TARY 643 DISX 771 DISY 233 CSTPE 0 REINFORCE 0
	# stop: MSG	3118572 TRIALNR END 7
	data = read_edf('1.asc', "TRIALNR", stop="TRIALNR END", debug=False)
	
	x = numpy.zeros(len(data[0]['x'])*1.5)
	y = numpy.zeros(len(data[0]['y'])*1.5)
	size = y = numpy.zeros(len(data[0]['size'])*1.5)
	
	for i in range(len(data)):
		x[:len(data[i]['x'])] = x[:len(data[i]['x'])] + data[i]['x']
		y[:len(data[i]['y'])] = y[:len(data[i]['y'])] + data[i]['y']
		y[:len(data[i]['size'])] = y[:len(data[i]['size'])] + data[i]['size']
	x = x/len(data)
	y = y/len(data)
	size = size/len(data)
	
	from matplotlib import pyplot
	pyplot.figure()
	pyplot.plot(data[0]['time'],data[0]['x'],'r')
	pyplot.plot(data[0]['time'],data[0]['y'],'g')
	pyplot.plot(data[0]['time'],data[0]['size'],'b')
	
	pyplot.figure()
	pyplot.plot(size,'b')
	
	pyplot.figure()
	pyplot.plot(x,y,'ko')
# # # # #
