"""
Publication-quality plotting utilities for EEG analysis results.

Main Plotting Functions
-----------------------
plot_timecourse : Plot neural timecourse with optional error bands
plot_2d : Plot 2D heatmaps
plot_significance : Overlay significance markers on timecourse plots
plot_erp_timecourse : Plot ERP waveforms with optional topographies
plot_tfr_timecourse : Plot time-frequency representation results
plot_bdm_timecourse : Plot decoding model results
plot_ctf_timecourse : Plot computational temporal filtering results
plot_erp_topography : Plot topographic maps at specific latencies
plot_topography : Generic topographic mapping function
"""

from open_dvm.visualization.plot import (
    plot_timecourse,
    plot_2d,
    plot_significance,
    plot_erp_timecourse,
    plot_tfr_timecourse,
    plot_bdm_timecourse,
    plot_ctf_timecourse,
    plot_erp_topography,
    plot_topography,
)

__all__ = [
    'plot_timecourse',
    'plot_2d',
    'plot_significance',
    'plot_erp_timecourse',
    'plot_tfr_timecourse',
    'plot_bdm_timecourse',
    'plot_ctf_timecourse',
    'plot_erp_topography',
    'plot_topography',
]
