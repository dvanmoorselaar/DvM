import os
from IPython import embed as shell

class FolderStructure(object):
	'''
	Creates the folder structure
	'''
	 
	def __init__(self):
		pass 	

	def FolderTracker(self, extension = [], filename = '', overwrite = True):
		'''
		FolderTracker creates folder address. At the same time it 
		checks whether the specific folder already exists (if not it is created)

		Arguments
		- - - - - 
		extension (list): list of subfolders that are attached to current working directory
		filename (str): name of file
		overwrite (bool): if overwrite is False, an * is added to the filename 

		Returns
		- - - -
		folder (str): file adress

		'''				

		# create folder adress
		folder = os.getcwd()
		if extension != []:
			folder = os.path.join(folder,*extension)	

		# check whether folder exists
		if not os.path.isdir(folder):
			os.makedirs(folder)

		if filename != '':	
			if not overwrite:
				while os.path.isfile(os.path.join(folder,filename)):
					filename = filename[:-4] + '*' + filename[-4:]
			folder = os.path.join(folder,filename)
			
		return folder	
