"""
open_dvm: Open source EEG analysis toolbox for cognitive neuroscience.

A comprehensive Python package for EEG, eye-tracking, and behavioral 
data analysis built on top of MNE-Python. Includes preprocessing,
ERP analysis, time-frequency decomposition, multivariate decoding, 
spatial encoding models, and eye-trackingintegration.

Main Modules
------------
analysis : Core EEG analysis classes and preprocessing
    RAW, Epochs, ERP, TFR, BDM, CTF, EYE classes and preprocessing 
    pipeline

visualization : Publication-quality plotting utilities
    Timecourse plots, topographies, ERPs, TFR, decoding results

stats : Statistical analysis utilities
    Bootstrap statistics, significance testing

support : Helper utilities for data management
    FolderStructure for file organization, preprocessing utilities, 
    eye-tracking helpers

pygazeanalyser : Eye-tracking data analysis
    Third-party eye tracker format readers and analysis

See Also
--------
MNE-Python: https://mne.tools/
Documentation: See NAMING_CONVENTIONS.md for file organization standards

Examples
--------
Basic ERP Analysis:
    >>> from open_dvm.analysis import ERP, preprocessing_pipeline
    >>> from open_dvm.support import FolderStructure
    >>> # Preprocess EEG data
    >>> preprocessing_pipeline(sj=1, session=1, eeg_runs=[1], ...)
    >>> # Load and analyze
    >>> erp = ERP(sj=1, epochs=epochs, df=behavior_df)
    >>> erp.condition_erps(pos_labels={'target_loc': [2, 6]})

Installation
------------
    pip install open_dvm

For development:
    git clone https://github.com/dvm/open_dvm.git
    cd open_dvm
    pip install -e .

Created by Dirk van Moorselaar.
"""

__version__ = "0.1.0"
__author__ = "Dirk van Moorselaar"

# Expose main analysis classes for convenience imports
try:
    from open_dvm.analysis.EEG import RAW, Epochs, ArtefactReject
    from open_dvm.analysis.ERP import ERP
    from open_dvm.analysis.TFR import TFR
    from open_dvm.analysis.BDM import BDM
    from open_dvm.analysis.CTF import CTF
    from open_dvm.analysis.EYE import EYE, SaccadeDetector
    from open_dvm.analysis.preprocessing_pipeline import eeg_preprocessing_pipeline
except ImportError:
    # Allow importing module even if submodules have import errors during development
    pass

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
