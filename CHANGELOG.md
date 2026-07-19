# Changelog

All notable changes to `open_dvm` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

Initial release baseline. Not yet tagged -- pending a final review pass.

### Added
- EEG preprocessing pipeline: filtering, ICA-based artifact removal, autoreject-based trial rejection, eye-tracking-based quality control.
- Event-Related Potential (ERP) analysis: condition-specific ERPs, lateralization, topography plots.
- Time-Frequency Representation (TFR) analysis: Morlet wavelet decomposition, evoked vs. total power.
- Brain Decoding Multivariate (BDM) analysis: within- and cross-condition decoding, generalization across time (GAT), permutation testing, trial-history analyses via `special_col`.
- Channel Tuning Function (CTF) analysis: inverted encoding models for spatial reconstruction, cross-task generalization, subject-specific reference-location alignment via `special_loc`.
- Eye-tracking integration: saccade detection, fixation-based trial exclusion.
- Statistical utilities: cluster-based permutation tests, FDR correction, bootstrap statistics.
- Publication-quality plotting for all four analysis modalities (`open_dvm.visualization.plot`), including condition-difference (`cnd_diff`) testing and visualization.
- Synthetic-data generators (`open_dvm.support.synthetic_data`) for demonstrating plotting/statistics independent of any real dataset.
- 10 tutorial notebooks (`tutorials/`) covering the full workflow from preprocessing through advanced BDM/CTF analyses.
- Full test suite (760 tests) across all analysis and visualization modules.
