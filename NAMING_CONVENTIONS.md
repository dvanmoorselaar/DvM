# DvM Toolbox File Naming Conventions

## Overview

The DvM toolbox uses standardized file naming conventions to ensure consistency, facilitate automated file discovery, and reflect the session-based preprocessing philosophy where each session is preprocessed independently before concatenation.

## Core Naming Pattern

All data files follow a consistent naming scheme:

```
sub_{sj}_ses_{session}_{analysis_name}.{extension}
```

### Components:
- `sub_{sj}` - Subject identifier (supports both zero-padded and standard formats: `sub_1` or `sub_01`)
- `ses_{session}` - Session identifier (e.g., `ses_1`, `ses_01`)
- `{analysis_name}` - Specific analysis type (e.g., `main`, `rest`, `task`)
- `.{extension}` - File format appropriate to data type

## File Naming by Data Type

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
sub_{sj}_ses_{session}*.csv
```

Example: `sub_01_ses_01_task.csv`

**Processed Behavioral Files:**
```
sub_{sj}_{fname}.csv
```

Example: `sub_01_main.csv`

### Eye Tracking Data

**Processed Eye Data:**
```
sub_{sj}_{analysis_name}.npz
```

Example: `sub_01_preproc_main.npz`

### ERP Data (Evoked Responses)

**Evoked Response Files:**
```
sub_{sj}_{condition}_{erp_name}-ave.fif
```

Example: `sub_01_left_target_locked-ave.fif`

### Time-Frequency Analysis (TFR)

**TFR Data Files:**
```
sub_{sj}_{condition}_{tfr_name}.pickle
```

Example: `sub_01_left_wavelet_main.pickle`

### Multivariate Decoding (BDM)

**BDM Result Files:**
```
sub_{sj}_{condition}_{bdm_name}.pickle
```

Example: `sub_01_left_decoding_main.pickle`

### Channel Tuning Functions (CTF)

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

```
project_root/
├── raw_eeg/                    # Raw BDF/EDF files
├── beh/
│   ├── raw/                    # Raw behavioral CSV files
│   └── processed/              # Processed behavioral data
├── eye/
│   ├── raw/                    # Raw eye tracker files
│   └── processed/              # Processed eye tracking data (-xy_eye.npz)
├── processed/                  # Preprocessed epochs (-epo.fif)
├── erp/
│   └── evoked/                 # Evoked response files (-ave.fif)
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

The DvM toolbox follows a session-based preprocessing approach:

1. **Independent Preprocessing**: Each session is preprocessed independently with the same pipeline
2. **Subject-Session Naming**: Files are named with both subject and session identifiers
3. **Session Concatenation**: After individual session preprocessing, sessions can be concatenated into `all` versions
4. **Consistent Naming**: The naming convention enables automated file discovery using glob patterns

### Example Workflow:

```python
# Step 1: Preprocess individual sessions
sub_01_ses_01_main-epo.fif  # Session 1
sub_01_ses_02_main-epo.fif  # Session 2
sub_01_ses_03_main-epo.fif  # Session 3

# Step 2: Concatenate sessions (optional)
sub_01_all_main-epo.fif     # All sessions combined
```

## Subject ID Formats

The naming convention supports both zero-padded and standard subject IDs:

```
sub_1        # Standard format
sub_01       # Zero-padded format (common for large studies)
sub_001      # Triple-padded format (as needed)
```

Regular expressions automatically handle both formats:
```python
r'sub_0?(\d+)_'  # Matches optional zero-padding
```

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

# Load specific subject's preprocessed epochs
df, epochs = fs.load_processed_epochs(
    sj=1,
    fname='main',
    preproc_name='main'  # Finds: sub_1_main-epo.fif or sub_01_main-epo.fif
)
```

### Manual File Discovery

You can also discover files using standard Python glob patterns:

```python
import glob

# Find all preprocessed epochs for subject 1
files = glob.glob('processed/sub_01_ses_*_main-epo.fif')

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
├── sub_01_ses_01_main-epo.fif
├── sub_01_ses_02_main-epo.fif
├── sub_01_all_main-epo.fif
├── sub_02_ses_01_main-epo.fif
├── erp/evoked/
│   ├── sub_01_left_p300-ave.fif
│   ├── sub_01_right_p300-ave.fif
│   ├── sub_02_left_p300-ave.fif
│   └── sub_02_right_p300-ave.fif
├── tfr/wavelet/
│   ├── sub_01_left_alpha.pickle
│   ├── sub_01_right_alpha.pickle
│   ├── sub_02_left_alpha.pickle
│   └── sub_02_right_alpha.pickle
└── preprocessing/group_info/
    └── preproc_param_main.csv
```

### Python Usage Examples

```python
from support.FolderStructure import FolderStructure

fs = FolderStructure()

# Load preprocessed data for subject 1
beh, epochs = fs.load_processed_epochs(
    sj=1,
    fname='main',
    preproc_name='main'
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


