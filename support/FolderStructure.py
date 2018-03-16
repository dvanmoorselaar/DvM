import os
from IPython import embed as shell

class FolderStructure(object):
	'''
	Creates the folder structure
	'''
	 
	def __init__(self):
		pass 	

	def FolderTracker(self, extension = [], filename = ''):
		'''
		DocString for folderTracker
		'''				

		#folder = self.project_folder
		folder = os.getcwd()

		if extension != []:
			folder = os.path.join(folder,*extension)	

		if filename != '':	
			folder = os.path.join(folder,filename)	
		
		return folder	
