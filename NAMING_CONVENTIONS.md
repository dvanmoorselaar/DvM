# open_dvm File Naming Conventions

## Overview

`open_dvm` organizes data on disk under a project folder using consistent `sub_{sj}_...` file naming, so that `FolderStructure`'s automatic file-discovery methods (`load_processed_epochs`, `read_erps`, `read_tfr`, `read_bdm`, `read_ctfs`, `read_raw_beh`) can find files without you having to track paths by hand.

This document describes what the code *actually* does, verified directly against `open_dvm/support/FolderStructure.py` and each analysis module's save code -- not an idealized convention. A few real quirks and rough edges are called out explicitly rather than glossed over.

## Folder Structure

```
project_root/
├── eeg/
│   ├── raw/                    # Raw EEG files (.bdf)
│   └── processed/              # Preprocessed epochs (-epo.fif)
├── meg/
│   └── processed/              # Preprocessed MEG epochs (-epo.fif) -- see MEG note below
├── behavioral/
│   ├── raw/                    # Raw behavioral CSV files
│   └── processed/              # Companion behavioral CSVs for processed epochs
├── eye/
│   ├── raw/                    # Raw eye-tracker files (.asc / .tsv)
│   └── processed/              # Derived eye-tracking data (.npz)
├── erp/
│   └── evoked/                 # Evoked response files (-ave.fif)
├── tfr/
│   └── {method}/               # TFR results, one folder per method (e.g. wavelet/)
├── bdm/
│   └── {to_decode}/{elec_oi}_elecs/[cross/][classifier/]  # BDM results
├── ctf/
│   └── {to_decode}/            # CTF results
└── preprocessing/
    ├── report/{preproc_name}/  # Per-subject/session HTML QC reports
    └── group_info/             # Group-level preprocessing parameter JSON
```

## Raw Data

**Raw EEG** (`eeg/raw/`), discovered via `find_raw_files()`:
```
sub_{sj}_ses_{session}.bdf
sub_{sj}_ses_{session}_run_{run}.bdf   # multi-run session
```
Subject/session/run digits aren't required to be zero-padded for discovery -- the regex accepts any digit width. Note: `eeg_preprocessing_pipeline()` hardcodes the `.bdf` extension; other formats MNE supports (`.edf`, `.fif`, `.vhdr`, `.cnt`, `.set`) can be read via the `RAW` class directly but aren't wired into the standard pipeline. When `eeg_runs` names more than one run, they're concatenated automatically before filtering/preprocessing.

**Raw behavioral** (`behavioral/raw/`), discovered via `read_raw_beh()`:
```
sub_{sj}_ses_{session}.csv
sub_{sj}_ses_{session}_run_{run}.csv
```
All files matching a given subject+session are concatenated automatically (the run number, if present, doesn't affect matching -- only subject and session are used to group files).

**Raw eye-tracker data** (`eye/raw/`), discovered via `EYE.get_eye_data()`:
```
sub_{sj}_ses_{session}.asc   # EyeLink ASCII
sub_{sj}_ses_{session}.tsv   # EyeTribe
```
`.csv` is **not** a supported raw eye-tracker extension (that's the behavioral-data extension) -- don't confuse the two.

## Preprocessed Epochs

`load_processed_epochs(sj, fname, preproc_name, modality='eeg')` reads:
```
{modality}/processed/sub_{sj}_{fname}-epo.fif
```

**Important**: `fname` is inserted verbatim -- it is *not* automatically prefixed with `ses_`. To load a file that was saved as `sub_01_ses_01_main-epo.fif`, call:
```python
FolderStructure().load_processed_epochs(sj=1, fname='ses_01_main', preproc_name='main')
```
For a session-combined file (`sub_01_all_main-epo.fif`, written by `Epochs.save_preprocessed(..., combine_sessions=True)`), pass `fname='all_main'`.

`preproc_name` is a separate parameter -- it does **not** appear in the epochs filename. It's used to locate two other things: the preprocessing parameter JSON (`preprocessing/group_info/preproc_param_{preproc_name}.json`) and the companion eye-tracking `.npz` file (see below). Because of this split, `preproc_name` should usually match the tail of whatever you pass as `fname`.

The eye-`.npz` lookup extracts a session number from `fname` via a `ses_(\d+)` regex, or, when `fname` starts with `all_` (a session-combined epochs file), looks up the combined-session eye file directly (`sub_{sj}_all_{preproc_name}.npz`) instead.

**MEG**: `modality='meg'` is a real, tested parameter -- it changes only which folder is read from (`meg/processed/` instead of `eeg/processed/`). This is currently the *only* MEG-specific functionality in the codebase; there is no MEG equivalent of `eeg_preprocessing_pipeline()` or MEG raw-file handling.

Companion behavioral data (when epochs have no attached metadata) is read from:
```
behavioral/processed/sub_{sj}_{fname}.csv
```
(same verbatim-`fname` rule as above).

## Preprocessing Reports and Logs

Written by `eeg_preprocessing_pipeline()`:
```
preprocessing/report/{preproc_name}/sj_{sj:02d}_ses_{ses:02d}.html   # MNE Report, note the "sj_" prefix
preprocessing/group_info/preproc_param_{preproc_name}.json           # incrementally-updated JSON, not CSV
```

Inside the group-info JSON, each subject/session gets its own entry keyed `sub_{sj:02d}_ses_{session}` (via `log_preproc()`), e.g.:
```json
{
  "sub_01_ses_01": {"high_pass": 0.1, "bad_chs": "['Fp1', 'Fp2']"},
  "sub_01_ses_all": {"high_pass": 0.1}
}
```
`ses_all` is used for session-combined entries (`session='all'`) rather than a zero-padded number.

## Analysis Results

Each analysis class's `to_decode`/method/condition parameters determine the folder and filename -- there is no user-chosen arbitrary folder name.

**ERP** (`ERP.condition_erps`, one file per condition):
```
erp/evoked/sub_{sj}_{condition}_{erp_name}-ave.fif
```
Read back via `FolderStructure().read_erps(erp_name=..., cnds=[...], sjs='all')`.

**TFR** (`TFR`, one file per condition; note the order -- name before condition):
```
tfr/{method}/sub_{sj}_{tfr_name}_{condition}-tfr.h5
```
Read back via `FolderStructure().read_tfr(tfr_folder_path=[method], tfr_name=..., cnds=[...], sjs='all')`.

**BDM** (`BDM.classify`): the folder is built from the actual decoding target and electrode selection, not a condition or an arbitrary label:
```
bdm/{to_decode}/{elec_oi}_elecs/[cross/][{classifier}/]sub_{sj}_{f_name}.pickle
```
`cross/` is only added for cross-condition (train/test) analyses; `{classifier}/` is only added when using a non-default classifier (anything other than `'LDA'`). **The condition is not part of the filename** -- results for every condition in a run are stored as separate keys inside the single pickled dict. Read back via `FolderStructure().read_bdm(bdm_folder_path=[to_decode, f'{elec_oi}_elecs', ...], bdm_name=f_name, sjs='all')` -- the folder path must include every component `set_folder_path()` would have added.

**CTF** (`CTF.spatial_ctf`), folder based on `to_decode` only (no electrode-selection subfolder, unlike BDM):
```
ctf/{to_decode}/sub_{sj}_{f_name}_{ctf|info|param}.pickle
```
Read back via `FolderStructure().read_ctfs(ctf_folder_path=[to_decode], output_type='ctf'|'info'|'param', ctf_name=f_name, sjs='all')`.

## Subject and Session ID Formatting

`format_subject_id(sj_id, zero_pad=2)` (in `open_dvm/support/preprocessing_utils.py`) accepts an `int`, a numeric `str`, or a prefixed string like `'sub_1'`, extracts the numeric part, and zero-pads it to `zero_pad` digits (default 2) via `str.zfill()`:

```python
format_subject_id(1)        # '01'
format_subject_id('sub_5')  # '05'
format_subject_id(123)      # '123'  -- zfill only pads, never truncates, so 3+ digit IDs already work
```

The same function is reused for session numbers (there is no separate `format_session_id`). **There is no equivalent helper for run numbers** -- run digit width in raw filenames isn't enforced or auto-formatted anywhere; use whatever width you like as long as it's consistent within a study.

## Python Usage Examples

```python
from open_dvm.support.FolderStructure import FolderStructure

fs = FolderStructure()

# Preprocessed EEG epochs
df, epochs = fs.load_processed_epochs(sj=1, fname='ses_01_main', preproc_name='main')

# Preprocessed MEG epochs
df, epochs = fs.load_processed_epochs(sj=1, fname='ses_01_main', preproc_name='main', modality='meg')

# ERPs across subjects
erps, times = fs.read_erps(erp_name='target_locked', cnds=['left', 'right'], sjs='all')

# TFR results
tfr_data = fs.read_tfr(tfr_folder_path=['wavelet'], tfr_name='alpha', cnds=['left', 'right'], sjs='all')

# BDM results (within-condition analysis, to_decode='img_loc', elec_oi='all')
bdm_results = fs.read_bdm(bdm_folder_path=['img_loc', 'all_elecs'], bdm_name='decoding', sjs='all')

# CTF results
ctfs = fs.read_ctfs(ctf_folder_path=['img_loc'], output_type='param', ctf_name='standard', sjs='all')
```

## Best Practices

1. Keep `fname`/`preproc_name` consistent with each other when calling `load_processed_epochs` -- since `fname` is used verbatim for the epochs file and `preproc_name` for the JSON/eye lookups, a mismatch will silently look in the wrong place rather than error.
2. When reading BDM results, always pass the *full* folder path `set_folder_path()` would have produced (`[to_decode, '{elec_oi}_elecs', ...]`) -- a partial path won't match anything.
3. Don't rely on zero-padding for sorting -- `FolderStructure`'s read methods sort by the extracted numeric subject id, not the string, so `sub_2` and `sub_02` both sort correctly relative to `sub_10`.
4. Record `preproc_name`/analysis names in a place you'll remember later (e.g. a lab notebook or the preprocessing JSON log) -- there's no built-in registry of what names have been used.
