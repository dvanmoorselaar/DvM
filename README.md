## DvM

Analysis scripts for EEG, eye tracking and behaviour.
EEG analysis scripts are based on MNE (www.martinos.org/mne/stable/index.html) with added functionality.

EEG analysis includes
	- Semi automatic preprocessing pipeline
	- ERP analysis
	- Multivariate decoding
	- Forward encoding models
	

Especially suited to integrate EEG data with behavioural and/or eye tracking experiments s using OpenSesame  

# Installation

To make use of this toolbox you have to set up a virtual python environment. 
The necessary steps are outlined below:

1. install anaconda (python 2 is recommended)
2. launch terminal
3. conda create -n mne python=2 pip
4. source activate mne
5. conda install scipy matplotlib scikit-learn mayavi jupyter spyder
6. pip install PySurfer mne
7. conda install seaborn
8. pip install opencv-python
9. Within the virtual environment install pygazeanalyser by copying it into the following folder (ananconda2/envs/mne/lib/python2.7/site-packages).
pygazeanalyser can be downloaded here: https://github.com/esdalmaijer/PyGazeAnalyser 

#Folderstructure

Analysis are run from specfic project scripts (e.g. Wholevspartial.py). To make sure this setup works
on your own computer within this script you have to make sure that the project folder is specified.
This folder needs to contain the following subfolders:

1. raw (with raw eeg file)
2. beh/raw (raw behavior files .csv)
3. eye/raw (raw eye files if available .asc or .tsv)

Also make sure that the system is pointed to the folder where you stored the ANALYSIS toolbox.
For any questions or suggestions, please email me: dirkvanmoorselaar@gmail.com

NOTE: The scipts, documentation and instructions are still being developed
