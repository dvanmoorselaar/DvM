# DvM Toolbox File Naming Conventions

## Overview

The DvM toolbox uses standardized file naming conventions to ensure consistency, facilitate automated file discovery, and reflect the session-based preprocessing philosophy where each session is preprocessed independently before concatenation.

## Core Naming Pattern

Data files follow consistent naming schemes that vary by processing stage:

**Preprocessing & Raw Data (with session information):**
```
sub_{sj}_ses_{session}_{analysis_name}.{extension}
```

**Analysis Results (without session - operates on concatenated data):**
```
sub_{sj}_{condition}_{analysis_name}.{extension}
```

### Components:
- `sub_{sj}` - Subject identifier (supports both zero-padded and standard formats: `sub_1` or `sub_01`)
- `ses_{session}` - Session identifier for raw/preprocessing files only (e.g., `ses_1`, `ses_01`)
- `{condition}` - Experimental condition for analysis files (e.g., `left`, `right`, `easy`)
- `{analysis_name}` - Specific analysis type (e.g., `main`, `rest`, `task`, `p300`, `alpha`)
- `.{extension}` - File format appropriate to data type

## File Naming by Data Type

### Raw EEG Data

**Raw EEG files (single run):**
```
sub_{sj}_ses_{session}.{bdf/edf}
```

Example: `sub_01_ses_01.bdf`

**Raw EEG files (multiple runs per session):**
```
sub_{sj}_ses_{session}_run_{run}.{bdf/edf}
```

Example: `sub_01_ses_01_run_01.bdf`, `sub_01_ses_01_run_02.bdf`

Notes:
- If a session contains only one run, the `_run_{run}` suffix is optional
- When multiple runs exist within a session, the preprocessing pipeline automatically concatenates them before applying filters and preprocessing
- Run numbers also use two-digit zero-padding for consistency (run_01, run_02, ..., run_99)

### EEG Data (Preprocessed Epochs)

**Individual Sessions:**
```
sub_{sj}_ses_{session}_{preproc_name}-epo.fif
```

Example: `sub_01_ses_01_main-epo.fif`

**Combined Sessions (All):**
```
sub_{sj}_all_{preproc_name}-epo.fif
```

Example: `sub_01_all_main-epo.fif`

### Behavioral Data

**Raw Behavioral Files:**
```
sub_{sj}_ses_{session}.csv
```

Or with optional run numbers (for multi-run sessions):
```
sub_{sj}_ses_{session}_run_{run}.csv
```

Examples: 
- `sub_01_ses_01.csv` (single file per session)
- `sub_01_ses_01_run_01.csv`, `sub_01_ses_01_run_02.csv` (multiple runs concatenated)

Notes:
- One behavioral file per session (or one per run if session has multiple runs)
- Multiple behavioral files with the same subject/session/run are automatically concatenated
- Different experimental conditions should be stored as separate columns in the CSV, not separate files

**Processed Behavioral Files:**
```
sub_{sj}_{fname}.csv
```

Example: `sub_01_main.csv`

### MEG Data

**Raw MEG files (single run):**
```
sub_{sj}_ses_{session}.{fif/ds}
```

Example: `sub_01_ses_01.fif`

**Raw MEG files (multiple runs per session):**
```
sub_{sj}_ses_{session}_run_{run}.{fif/ds}
```

Example: `sub_01_ses_01_run_01.fif`, `sub_01_ses_01_run_02.fif`

**Preprocessed MEG Epochs:**
```
sub_{sj}_ses_{session}_{preproc_name}-epo.fif
```

Example: `sub_01_ses_01_main-epo.fif`

Notes:
- MEG data follows the same session-based preprocessing and naming conventions as EEG
- Run support is identical to EEG (multiple runs per session automatically concatenated)
- All analysis scripts (ERP, TFR, BDM, CTF) work identically for both EEG and MEG data since they operate on MNE Epochs objects

### Eye Tracking Data

**Raw Eye Tracker Files:**
```
sub_{sj}_ses_{session}.{asc/csv}
```

Example: `sub_01_ses_01.asc`

**Processed Eye Data:**
```
sub_{sj}_{preproc_name}.npz
```

Example: `sub_01_preproc_main.npz`

### ERP Data (Evoked Responses)
**Note: No session information in analysis results**

**Evoked Response Files:**
```
sub_{sj}_{condition}_{erp_name}-ave.fif
```

Example: `sub_01_left_target_locked-ave.fif`

### Time-Frequency Analysis (TFR)
**Note: No session information in analysis results**

**TFR Data Files:**
```
sub_{sj}_{condition}_{tfr_name}-tfr.h5
```

Example: `sub_01_left_wavelet_main-tfr.h5`

### Multivariate Decoding (BDM)
**Note: No session information in analysis results**

**BDM Result Files:**
```
sub_{sj}_{condition}_{bdm_name}.pickle
```

Example: `sub_01_left_decoding_main.pickle`

### Channel Tuning Functions (CTF)
**Note: No session information in analysis results**

**CTF Data Files:**
```
sub_{sj}_{ctf_name}_{output_type}.pickle
```

Where `{output_type}` is one of:
- `ctf` - Tuning function data
- `info` - Metadata and analysis information
- `param` - Analysis parameters

Example: `sub_01_orientation_ctf.pickle`

## Folder Structure

The DvM toolbox organizes data in standardized folders:

**MEG Support**: The toolbox supports both EEG and MEG data. Use the `modality` parameter in `load_processed_epochs()` to specify which modality to load:
- `modality='eeg'` (default) - loads from `eeg/processed/`
- `modality='meg'` - loads from `meg/processed/`

```
project_root/
├── eeg/
│   ├── raw/                    # Raw EEG BDF/EDF files
│   └── processed/              # Preprocessed EEG epochs (-epo.fif)
├── meg/
│   ├── raw/                    # Raw MEG FIF/DS files (if available)
│   └── processed/              # Preprocessed MEG epochs (-epo.fif)
├── behavioral/
│   ├── raw/                    # Raw behavioral CSV files
│   └── processed/              # Processed behavioral data
├── eye/
│   ├── raw/                    # Raw eye tracker files
│   └── processed/              # Processed eye tracking data (-xy_eye.npz)
├── erp/
│   ├── evoked/                 # Evoked response files (-ave.fif)
│   └── stats/                  # ERP statistics and group analysis
├── tfr/
│   ├── wavelet/               # Wavelet analysis results
│   ├── multitaper/            # Multitaper analysis results
│   └── [other_methods]/       # Additional TFR methods
├── bdm/
│   ├── decoding/              # Decoding analysis results
│   └── [other_analyses]/      # Additional BDM analyses
├── ctf/
│   ├── orientation/           # Orientation tuning
│   ├── standard/              # Standard tuning functions
│   └── [other_parameters]/    # Additional CTF parameters
└── preprocessing/
    ├── report/                # HTML quality control reports
    └── group_info/            # Group-level statistics and parameters
```

## Session-Based Preprocessing Philosophy

The DvM toolbox follows a session-based preprocessing approach with optional run support, but **sessions and runs apply only to preprocessing and raw data**. Analysis scripts operate on concatenated data without session/run information:

1. **Multi-Run Sessions**: Sessions can contain multiple runs (e.g., `sub_01_ses_01_run_01.bdf`, `sub_01_ses_01_run_02.bdf`)
   - Multiple runs within a session are automatically concatenated during preprocessing
   - After concatenation, all runs from a session are treated as a single epoch file
2. **Independent Preprocessing**: Each session is preprocessed independently with the same pipeline
3. **Subject-Session Naming**: Raw and preprocessed files are named with both subject and session identifiers (e.g., `sub_01_ses_01_main-epo.fif`)
4. **Session Concatenation**: After individual session preprocessing, sessions can be concatenated into `all` versions for analysis
5. **Analysis Without Sessions**: Analysis scripts (ERP, TFR, BDM, CTF) operate on concatenated data and use condition-based naming without session information (e.g., `sub_01_face_erp-ave.fif`)
6. **Consistent Naming**: The naming convention enables automated file discovery using glob patterns

### Example Workflow:

**Preprocessing stage (with sessions and optional runs):**
```python
# Step 0: Load raw data (multiple runs per session automatically concatenated)
sub_01_ses_01_run_01.bdf  # Session 1, Run 1
sub_01_ses_01_run_02.bdf  # Session 1, Run 2  (both concatenated internally)
sub_01_ses_02_run_01.bdf  # Session 2, Run 1
sub_01_ses_02_run_02.bdf  # Session 2, Run 2  (both concatenated internally)

# Step 1: After preprocessing, single epoch file per session
sub_01_ses_01_main-epo.fif  # Session 1 (runs 1-2 already concatenated)
sub_01_ses_02_main-epo.fif  # Session 2 (runs 1-2 already concatenated)

# Step 2: Concatenate sessions (optional)
sub_01_all_main-epo.fif     # All sessions combined
```

**Analysis stage (without sessions - operates on concatenated data):**
```python
# Step 3: Run analyses on concatenated data
sub_01_face_erp-ave.fif          # ERP analysis result (no session)
sub_01_left_wavelet_main-tfr.h5  # TFR analysis result (no session)
sub_01_face_decoding_main.pickle # BDM analysis result (no session)
```

## Subject ID Formats

The DvM toolbox **enforces consistent two-digit zero-padded subject and session IDs** across all preprocessing and analysis scripts for reliable file discovery and sorting.

### Standard Format: Two-Digit Zero-Padding

```
sub_01       # Two-digit zero-padding (standard for all studies)
ses_01       # Sessions also use two-digit zero-padding
```

Note: This format supports up to 99 subjects. For larger studies (100+ subjects), contact the development team for three-digit padding support.

### Format Enforcement

All analysis scripts automatically convert subject and session inputs to two-digit zero-padded format. Users can input IDs in any format, and they will be standardized internally:

```python
# Input format doesn't matter - all are converted to two-digit zero-padding
sj = 1           # Converted to '01'
sj = '1'         # Converted to '01'
sj = 'sub_1'     # Extracted and converted to '01'
sj = '01'        # Remains '01'
sj = 5           # Converted to '05'
```

### Implementation Details

The `format_subject_id()` utility function handles all formatting automatically (users do not need to call this directly):

```python
# Internal conversion (automatic)
# Input: 1, '1', 'sub_1', '01' → Output: '01'
# Input: 5, '5', 'sub_5', '05' → Output: '05'
# Input: 12, '12', 'sub_12' → Output: '12'
```

### File Discovery and Sorting

All file discovery methods use regex patterns that correctly extract numeric values and sort files in numeric order:

```python
r'sub_0?(\d+)_'  # Extracts subject number for proper numeric sorting
# Files are sorted: sub_01, sub_02, ..., sub_09, sub_10, ..., sub_99
# NOT alphabetically: sub_01, sub_10, sub_11, ..., sub_09
```
# Works with both sub_1 and sub_01, sorts numerically not alphabetically
```

Affected methods in FolderStructure:
- `load_processed_epochs(modality='eeg'|'meg')` - Loads epochs from either eeg/ or meg/ folder; defaults to EEG
- `read_raw_beh()` - Formats subject and session IDs
- `read_erps()` - Sorts by extracted numeric subject number
- `read_tfr()` - Sorts by extracted numeric subject number
- `read_bdm()` - Sorts by extracted numeric subject number
- `read_ctfs()` - Sorts by extracted numeric subject number

All analysis classes (ERP, TFR, BDM, CTF) automatically format subject IDs during initialization.

## Integration with Python Code

### Loading Files with Automatic Discovery

The `FolderStructure` class uses glob patterns that automatically discover files following the naming convention:

```python
import glob
from support.FolderStructure import FolderStructure

fs = FolderStructure()

# Load all subjects' ERP data
erps = fs.read_erps(
    erp_name='target_locked',
    cnds=['left', 'right'],
    sjs='all'  # Automatically finds: sub_*_*_target_locked-ave.fif
)

# Load EEG preprocessed epochs (default)
df, epochs = fs.load_processed_epochs(
    sj=1,
    fname='main',
    preproc_name='main'  # Finds: eeg/processed/sub_1_main-epo.fif
)

# Load MEG preprocessed epochs
df, epochs = fs.load_processed_epochs(
    sj=1,
    fname='main',
    preproc_name='main',
    modality='meg'  # Finds: meg/processed/sub_1_main-epo.fif
)
```

### Manual File Discovery

You can also discover files using standard Python glob patterns:

```python
import glob

# Find all preprocessed epochs for subject 1
files = glob.glob('eeg/processed/sub_01_ses_*_main-epo.fif')

# Find all ERP files for subject 1
files = glob.glob('erp/evoked/sub_01_*_target_locked-ave.fif')

# Find all TFR results
files = glob.glob('tfr/wavelet/sub_*_*_*.pickle')
```

## Migration Guide

### From Old Naming Conventions

The following old naming patterns have been standardized:

| Old Pattern | New Pattern | Notes |
|---|---|---|
| `sj_{sj}_ses_{session}_{name}` | `sub_{sj}_ses_{session}_{name}` | Subject prefix standardized |
| `sj_{sj}_{name}` | `sub_{sj}_{name}` | Subject prefix standardized |
| `subject-{sj}_session_{session}` | `sub_{sj}_ses_{session}` | Prefix and separator standardized |
| `subject-{sj}_{name}` | `sub_{sj}_{name}` | Prefix standardized |
| `ctfs_{sj}_` | `sub_{sj}_` | Prefix changed to subject format |
| `ctf_info_{sj}_` | `sub_{sj}_` | Prefix changed to subject format |
| `sj_*` glob patterns | `sub_*` | Updated glob patterns |


## Best Practices

1. **Consistency**: Always use lowercase with underscores separating components
2. **Precision**: Include session information even for single-session studies
3. **Simplicity**: Use descriptive but concise analysis names
4. **Flexibility**: Leverage the pattern to organize multiple analyses
5. **Documentation**: Record analysis names in preprocessing parameter files

## Examples

### Complete Project Structure

```
my_study/
├── eeg/
│   └── processed/
│       ├── sub_01_ses_01_main-epo.fif
│       ├── sub_01_ses_02_main-epo.fif
│       ├── sub_01_all_main-epo.fif
│       └── sub_02_ses_01_main-epo.fif
├── erp/
│   ├── evoked/
│   │   ├── sub_01_left_p300-ave.fif
│   │   ├── sub_01_right_p300-ave.fif
│   │   ├── sub_02_left_p300-ave.fif
│   │   └── sub_02_right_p300-ave.fif
│   └── stats/
│       └── p300_analysis.csv
├── tfr/wavelet/
│   ├── sub_01_left_alpha-tfr.h5
│   ├── sub_01_right_alpha-tfr.h5
│   ├── sub_02_left_alpha-tfr.h5
│   └── sub_02_right_alpha-tfr.h5
├── bdm/
│   ├── decoding/
│   │   ├── sub_01_left_decoding_main.pickle
│   │   ├── sub_01_right_decoding_main.pickle
│   │   ├── sub_02_left_decoding_main.pickle
│   │   └── sub_02_right_decoding_main.pickle
│   └── stats/
│       └── decoding_summary.csv
├── ctf/
│   ├── orientation/
│   │   ├── sub_01_orientation_ctf.pickle
│   │   ├── sub_01_orientation_info.pickle
│   │   └── sub_01_orientation_param.pickle
│   └── stats/
│       └── ctf_summary.csv
└── preprocessing/
    ├── report/
    └── group_info/
        └── preproc_param_main.json
```

### Python Usage Examples

```python
from support.FolderStructure import FolderStructure

fs = FolderStructure()

# Load preprocessed EEG data for subject 1
beh, epochs = fs.load_processed_epochs(
    sj=1,
    fname='main',
    preproc_name='main'
)

# Load preprocessed MEG data for subject 1
beh, epochs = fs.load_processed_epochs(
    sj=1,
    fname='main',
    preproc_name='main',
    modality='meg'
)

# Load all ERP data
erps, times = fs.read_erps(
    erp_name='p300',
    cnds=['left', 'right'],
    sjs='all'
)

# Load TFR data for specific subjects
tfr_data, freqs, times = fs.read_tfr(
    tfr_folder_path=['wavelet'],
    tfr_name='alpha',
    cnds=['left', 'right'],
    sjs=[1, 2, 3]
)

# Load decoding results
bdm_results = fs.read_bdm(
    bdm_folder_path=['decoding'],
    bdm_name='stimulus',
    sjs='all'
)
```

## Questions and Support

For questions about naming conventions or issues with file discovery, refer to:

- **README.md** - General toolbox documentation
- **RAWEEG_USAGE_GUIDE.md** - Raw EEG preprocessing guide
- **Code Documentation** - Docstrings in FolderStructure, EEG, and analysis classes

---


