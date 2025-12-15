# RawEEG Class - Usage Guide and Migration Instructions

## Overview

The `RawEEG` class has been updated to work with modern versions of MNE-Python (≥0.23). It now supports multiple recording systems and provides a cleaner interface for reading and preprocessing EEG data.

## What Changed?

### Old Implementation
```python
class RawEEG(mne.io.edf.edf.RawEDF, BaseRaw, FolderStructure):
    def __init__(self, input_fname, eog=None, stim_channel=-1,
                 exclude=(), preload=True, verbose=None):
        super(RawEEG, self).__init__(input_fname=input_fname, eog=eog,
                                     stim_channel=stim_channel, preload=preload, 
                                     verbose=verbose)
```

**Problems:**
- Hardcoded to only support EDF files via `mne.io.edf.edf.RawEDF`
- Not compatible with modern MNE API (>= 0.23)
- Limited to single file format

### New Implementation
```python
class RawEEG(BaseRaw, FolderStructure):
    def __init__(self, input_fname, file_type=None, eog=None, 
                 stim_channel=-1, exclude=(), preload=True, 
                 verbose=None, **kwargs):
```

**Improvements:**
- ✅ Works with modern MNE-Python
- ✅ Supports multiple file formats (BDF, EDF, FIF, BrainVision, CNT, SET)
- ✅ Auto-detects file type from extension
- ✅ Adds convenience class methods
- ✅ Maintains all existing functionality

## Supported File Formats

The updated `RawEEG` class now supports:

| Format | Extension | Common Systems |
|--------|-----------|----------------|
| BioSemi BDF | `.bdf` | BioSemi ActiveTwo |
| European Data Format | `.edf` | Various EEG systems |
| Neuromag/Elekta | `.fif` | MEG/EEG systems |
| BrainVision | `.vhdr` | BrainProducts |
| Neuroscan CNT | `.cnt` | Neuroscan |
| EEGLAB | `.set` | EEGLAB |

## Usage Examples

### Basic Usage (Auto-detection)

The simplest way to load data - file type is automatically detected from the extension:

```python
from eeg_analyses.EEG import RawEEG

# Load BDF file (auto-detected from .bdf extension)
raw = RawEEG('subject_01.bdf', preload=True)

# Load EDF file (auto-detected from .edf extension)
raw = RawEEG('subject_01.edf', preload=True)

# Load FIF file (auto-detected from .fif extension)
raw = RawEEG('subject_01_raw.fif', preload=True)
```

### Using Convenience Class Methods

For explicit file type specification:

```python
# Load BDF file
raw = RawEEG.from_bdf('subject_01.bdf', preload=True)

# Load EDF file with EOG channels
raw = RawEEG.from_edf('subject_01.edf', eog=['EOG1', 'EOG2'])

# Load FIF file
raw = RawEEG.from_fif('subject_01_raw.fif')

# Load BrainVision file
raw = RawEEG.from_brainvision('subject_01.vhdr', preload=True)
```

### Specifying File Type Explicitly

If auto-detection fails or you want to be explicit:

```python
# Explicitly specify file type
raw = RawEEG('data_file', file_type='bdf', preload=True)
```

### Common Preprocessing Workflow

```python
from eeg_analyses.EEG import RawEEG

# 1. Load data
raw = RawEEG('subject_01.bdf', preload=True, stim_channel=-1)

# 2. Replace bad channels (if needed)
replace_dict = {'1': {'session_1': {'F1': 'EXG7'}}}
raw.replace_channel(sj=1, session=1, replace=replace_dict)

# 3. Re-reference to average
raw.rereference(ref_channels='average', change_voltage=True, to_remove=['EXG7', 'EXG8'])

# 4. Configure montage
raw.configure_montage(montage='biosemi64', ch_remove=['M1', 'M2'])

# 5. Select events
events = raw.select_events(event_id=[1, 2, 3, 4], binary=0)

print(f"Found {len(events)} events")
```

### Advanced Options

```python
# Load with specific parameters
raw = RawEEG(
    'subject_01.bdf',
    file_type='bdf',
    eog=['EOG1', 'EOG2'],  # Specify EOG channels
    stim_channel='Status',  # Named stim channel
    exclude=['EXG7', 'EXG8'],  # Exclude channels from loading
    preload=True,
    verbose='WARNING'
)

# Additional MNE-specific parameters can be passed via **kwargs
raw = RawEEG(
    'subject_01.bdf',
    preload=True,
    misc=['GSR1', 'GSR2']  # BDF-specific: mark channels as misc
)
```

## Migration Guide

### If you were using the old RawEEG class:

**Old code:**
```python
raw = RawEEG('subject_01.bdf', preload=True)
```

**New code (no changes needed!):**
```python
raw = RawEEG('subject_01.bdf', preload=True)
```

The new implementation is **backward compatible** for basic usage. Your existing code should continue to work!

### If you need to support multiple file types:

**Old code (only worked with EDF/BDF):**
```python
raw = RawEEG('subject_01.edf', preload=True)
```

**New code (works with any supported format):**
```python
# Auto-detection
raw = RawEEG('subject_01.fif', preload=True)
raw = RawEEG('subject_01.vhdr', preload=True)

# Or use convenience methods
raw = RawEEG.from_fif('subject_01_raw.fif')
raw = RawEEG.from_brainvision('subject_01.vhdr')
```

## Key Methods

All existing methods remain available:

- `replace_channel()` - Replace bad electrodes with designated replacements
- `rereference()` - Re-reference EEG data and handle voltage conversion
- `configure_montage()` - Set electrode montage and rename channels
- `select_events()` - Find and process events from raw data
- `report_raw()` - Add raw data to MNE report

These methods work exactly as before, maintaining full backward compatibility.

## Architecture Notes

The new `RawEEG` class:

1. **Inherits from `BaseRaw`** - No longer tied to a specific file format
2. **Uses composition** - Loads the appropriate MNE Raw object internally
3. **Copies attributes** - All MNE Raw methods and attributes are available
4. **Adds custom methods** - Preprocessing functionality specific to your workflow

This approach provides maximum flexibility while maintaining all MNE functionality.

## Troubleshooting

### File type not detected
```python
# If you get: "Cannot determine file type from extension"
# Solution: Specify file_type explicitly
raw = RawEEG('myfile.dat', file_type='bdf', preload=True)
```

### Unsupported file format
```python
# If you get: "Unsupported file type: xyz"
# The format may not be supported yet. Supported formats:
# bdf, edf, fif, brainvision, cnt, set
```

### MNE version compatibility
Make sure you're using MNE-Python >= 0.23:
```python
import mne
print(mne.__version__)  # Should be >= 0.23
```

## Examples by Recording System

### BioSemi ActiveTwo
```python
raw = RawEEG.from_bdf(
    'subject_01.bdf',
    eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'],
    stim_channel='Status',
    preload=True
)
```

### BrainVision
```python
raw = RawEEG.from_brainvision(
    'subject_01.vhdr',
    eog=['VEOG', 'HEOG'],
    preload=True
)
```

### Already preprocessed FIF files
```python
raw = RawEEG.from_fif('subject_01_preprocessed_raw.fif')
```

### EDF files
```python
raw = RawEEG.from_edf(
    'subject_01.edf',
    eog=['EOG1', 'EOG2'],
    stim_channel='STI 014',
    preload=True
)
```

## Questions?

If you encounter any issues or need additional file format support, please check:
1. Your MNE-Python version (`import mne; print(mne.__version__)`)
2. The file format is in the supported list
3. The file can be read with standard MNE functions

For additional help, refer to the [MNE-Python documentation](https://mne.tools/stable/documentation/reading.html).
