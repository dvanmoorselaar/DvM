# OpenDvM

![tests](https://github.com/dvanmoorselaar/open_dvm/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

A comprehensive Python package for EEG, eye-tracking, and behavioral data analysis, built on top of [MNE-Python](https://mne.tools/). `open_dvm` covers the full pipeline from raw-data preprocessing through ERP analysis, time-frequency decomposition, multivariate decoding, and spatial encoding models -- especially suited to integrating EEG data with behavioral and/or eye-tracking experiments (e.g. via OpenSesame).

EEG analysis includes:
- Semi-automatic preprocessing pipeline (filtering, ICA, artifact rejection)
- ERP analysis
- Time-frequency decomposition
- Multivariate decoding (BDM)
- Forward encoding / channel tuning function models (CTF)
- Eye-tracking integration and quality control

## Installation

`open_dvm` isn't published on PyPI yet. Install directly from GitHub:

```bash
pip install git+https://github.com/dvanmoorselaar/open_dvm.git
```

For development (editable install with test/lint dependencies):

```bash
git clone https://github.com/dvanmoorselaar/open_dvm.git
cd open_dvm
pip install -e ".[dev]"
```

## Quick Start

```python
from open_dvm.analysis import ERP
from open_dvm.support import FolderStructure

# Load preprocessed data
df, epochs = FolderStructure().load_processed_epochs(sj=1, fname='ses_01_main', preproc_name='main')

# Compute condition-specific ERPs
erp = ERP(sj=1, epochs=epochs, df=df, baseline=(-0.2, 0))
erp.condition_erps(pos_labels={'target_loc': [2, 6]})
```

## Tutorials

The `tutorials/` folder contains 10 Jupyter notebooks that walk through the toolbox end to end, from preprocessing to advanced decoding:

| Notebook | Description |
|---|---|
| [00_visualization_and_statistics](tutorials/00_visualization_and_statistics.ipynb) | Tour of every plotting/statistics option in `open_dvm.visualization.plot`, using synthetic data with guaranteed-significant effects. Read this one first. |
| [01_preprocessing](tutorials/01_preprocessing.ipynb) | EEG preprocessing pipeline: filtering, ICA, artifact rejection |
| [02_erp_analysis](tutorials/02_erp_analysis.ipynb) | Event-related potential (ERP) computation and visualization |
| [03_tfr_analysis](tutorials/03_tfr_analysis.ipynb) | Time-frequency decomposition via Morlet wavelets |
| [04_tfr_advanced](tutorials/04_tfr_advanced.ipynb) | Advanced TFR: wavelet parameters, alternative decompositions, group-level stats |
| [05_bdm_decoding](tutorials/05_bdm_decoding.ipynb) | Multivariate decoding (BDM) basics: localizer/main-task decoding, cross-task generalization |
| [06_bdm_advanced](tutorials/06_bdm_advanced.ipynb) | Advanced BDM: temporal generalization, classifier comparisons, time-frequency decoding |
| [07_ctf_analysis](tutorials/07_ctf_analysis.ipynb) | Spatial channel tuning functions (CTF) via an inverted encoding model |
| [08_ctf_advanced](tutorials/08_ctf_advanced.ipynb) | Advanced CTF: homogeneous ("ping") displays, subject-specific reference-location alignment |
| [09_bdm_ctf_comparison](tutorials/09_bdm_ctf_comparison.ipynb) | BDM vs. CTF side by side on the same question -- decodability vs. interpretability |

## Folder Structure

Analyses are run from project-specific scripts, pointed at a project folder containing:

1. `eeg/raw` (raw EEG files, `.bdf` or `.edf`)
2. `behavioral/raw` (raw behavior files, `.csv`)
3. `eye/raw` (raw eye-tracking files, if available: `.asc` or `.csv`)

**For detailed file naming conventions and folder organization standards, see [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md).**

## Contributing

Contributions are welcome -- see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, running tests, and the PR process.

## Citation

If you use `open_dvm`, please cite it via [CITATION.cff](CITATION.cff) (or GitHub's "Cite this repository" button). The accompanying methods paper is still in preparation; the citation entry will be updated with full details once it's available.

This toolbox is built on MNE-Python -- please also cite:

Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. S. (2014). MNE software for processing MEG and EEG data. Neuroimage, 86, 446-460.

## Questions or Issues?

Please open a [GitHub issue](https://github.com/dvanmoorselaar/open_dvm/issues), or contact Dirk van Moorselaar at dirkvanmoorselaar@gmail.com.
