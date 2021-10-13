# DvM

Analysis scripts for EEG, eye tracking and behaviour.
EEG analysis scripts are based on MNE (www.martinos.org/mne/stable/index.html) with added functionality.

EEG analysis includes
	- Semi automatic preprocessing pipeline
	- ERP analysis
	- Multivariate decoding
	- Forward encoding models
	

Especially suited to integrate EEG data with behavioural and/or eye tracking experiments s using OpenSesame  

## Installation

To make use of this toolbox you have to set up a virtual python environment. 
The necessary steps are outlined below:

1. install anaconda (switching all code to python 3 now)
2. launch terminal
3. conda create -n mne python=3 pip
4. source activate mne
5. conda install scipy matplotlib scikit-learn mayavi jupyter spyder
6. pip install PySurfer mne
7. conda install seaborn
8. pip install opencv-python
9. pip install -U autoreject
In case you want to make use of picard during ica
10. pip install python-picard


## Folderstructure

Analysis are run from specfic project scripts (e.g. Wholevspartial.py). To make sure this setup works
on your own computer within this script you have to make sure that the project folder is specified.
This folder needs to contain the following subfolders:

1. raw (with raw eeg file)
2. beh/raw (raw behavior files .csv)
3. eye/raw (raw eye files if available .asc or .tsv)

Also make sure that the system is pointed to the folder where you stored the ANALYSIS toolbox.
For any questions or suggestions, please email me: dirkvanmoorselaar@gmail.com

NOTE: The scipts, documentation and instructions are still being developed

## Citation

Please cite:

Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. S. (2014). MNE software for processing MEG and EEG data. Neuroimage, 86, 446-460.

If you use the pygazeanalyser functionality to analyse eyetracker data in any way please cite:

Dalmaijer, E.S., Mathôt, S., & Van der Stigchel, S. (2013). PyGaze: an open-source, cross-platform toolbox for minimal-effort programming of eye tracking experiments. Behaviour Research Methods. doi:10.3758/s13428-013-0422-2
