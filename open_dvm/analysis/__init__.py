"""
Core EEG analysis classes and preprocessing pipeline.

Main Classes
------------
RAW : Raw EEG data handling
Epochs : Epoched EEG data
ArtefactReject : Artifact rejection and trial exclusion
ERP : Event-Related Potential analysis
TFR : Time-Frequency Representation (wavelet/Fourier analysis)
BDM : Backward Decoding Model (multivariate decoding)
CTF : Computational Temporal Filtering (spatial encoding models)
EYE : Eye-tracking integration and analysis
SaccadeDetector : Detect saccades from eye-tracking data

Functions
---------
eeg_preprocessing_pipeline : Complete preprocessing workflow
"""

from open_dvm.analysis.EEG import RAW, Epochs, ArtefactReject
from open_dvm.analysis.ERP import ERP
from open_dvm.analysis.TFR import TFR
from open_dvm.analysis.BDM import BDM
from open_dvm.analysis.CTF import CTF
from open_dvm.analysis.EYE import EYE, SaccadeDetector
from open_dvm.analysis.preprocessing_pipeline import eeg_preprocessing_pipeline

__all__ = [
    'RAW',
    'Epochs',
    'ArtefactReject',
    'ERP',
    'TFR',
    'BDM',
    'CTF',
    'EYE',
    'SaccadeDetector',
    'eeg_preprocessing_pipeline',
]
