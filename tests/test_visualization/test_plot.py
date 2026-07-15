"""
Test suite for open_dvm.visualization.plot.

Organization
------------
- TestPlotTimecourse: 1D line + optional bootstrap SE band, smoothing
- TestPlot2D: heatmap/contour, masking, diverging colormap, ticks
- TestGetContinuousSegments: boolean-mask -> contiguous index segments
- TestPlotSignificance: 1D/2D significance overlays
- TestPlotErpTimecourse: ERP waveform plotting, lateralization
- TestPlotTfrTimecourse: TFR timecourse/2D plotting, contour integration
- TestPlotBdmTimecourse: decoding timecourse/GAT/TFR plotting
- TestPlotCtfTimecourse: CTF timecourse/GAT/TFR/2D-CTF plotting
- TestPlotTopography / TestPlotErpTopography: scalp topographies
- TestRegressions: targeted checks for bugs fixed in this module
"""

import warnings
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pytest
import mne
from scipy.signal import savgol_filter

from open_dvm.visualization.plot import (
    plot_timecourse,
    plot_2d,
    _get_continuous_segments,
    plot_significance,
    plot_erp_timecourse,
    plot_tfr_timecourse,
    plot_bdm_timecourse,
    plot_ctf_timecourse,
    plot_erp_topography,
    plot_topography,
)
import open_dvm.visualization.plot as plotmod

from tests.fixtures.sample_data import create_biosemi64_evoked_pair
from tests.fixtures.plot_sample_data import (
    make_condition_evokeds,
    make_average_tfr,
    make_bdm_result,
    make_ctf_result,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _cluster(rows, cols):
    """Build an MNE-style cluster index tuple: paired index arrays of
    equal length (as returned by np.where on a boolean cluster mask),
    covering the full rows x cols block."""
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    return (rr.ravel(), cc.ravel())


# ============================================================================
# plot_timecourse
# ============================================================================

class TestPlotTimecourse:
    @pytest.mark.unit
    def test_1d_input_plotted_as_is(self):
        x = np.linspace(0, 1, 50)
        y = np.sin(x * 10)

        plot_timecourse(x, y)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), y)

    @pytest.mark.unit
    def test_2d_input_averaged_without_se(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 30)
        y = rng.normal(0, 1, size=(10, 30))

        plot_timecourse(x, y, show_SE=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), y.mean(axis=0))

    @pytest.mark.unit
    def test_show_se_draws_band_around_bootstrap_mean(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 30)
        y = rng.normal(5, 1, size=(200, 30))

        plot_timecourse(x, y, show_SE=True)

        ax = plt.gca()
        line = ax.get_lines()[0]
        # bootstrap mean should closely track the true sample mean
        np.testing.assert_allclose(line.get_ydata(), y.mean(axis=0), atol=0.2)
        # fill_between band exists and straddles the line
        poly = ax.collections[0]
        path_ys = poly.get_paths()[0].vertices[:, 1]
        assert path_ys.max() > line.get_ydata().max()
        assert path_ys.min() < line.get_ydata().min()

    @pytest.mark.unit
    def test_smooth_matches_savgol_filter(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 50)
        y = rng.normal(0, 1, size=50)

        plot_timecourse(x, y, smooth=True)

        line = plt.gca().get_lines()[0]
        expected = savgol_filter(y, 9, 1)
        np.testing.assert_allclose(line.get_ydata(), expected)

    @pytest.mark.unit
    def test_kwargs_forwarded_to_plot(self):
        x = np.linspace(0, 1, 10)
        y = np.zeros(10)

        plot_timecourse(x, y, color='red', label='mycond')

        line = plt.gca().get_lines()[0]
        assert line.get_color() == 'red'
        assert line.get_label() == 'mycond'


# ============================================================================
# plot_2d
# ============================================================================

class TestPlot2D:
    @pytest.mark.unit
    def test_none_xval_yval_defaults_to_indices_no_crash(self):
        # regression: used to crash via len(None) with set_y_ticks=True default
        Z = np.random.default_rng(0).normal(0, 1, (8, 20))

        plot_2d(Z)

        img = plt.gca().get_images()[0]
        np.testing.assert_array_equal(img.get_extent(), [0, 19, 0, 7])

    @pytest.mark.unit
    def test_3d_input_averaged_over_first_axis(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(0, 1, (5, 6, 10))

        plot_2d(Z)

        img = plt.gca().get_images()[0]
        np.testing.assert_allclose(img.get_array(), Z.mean(axis=0))

    @pytest.mark.unit
    def test_extent_uses_xval_yval(self):
        Z = np.zeros((5, 8))
        x_val = np.linspace(-100, 300, 8)
        y_val = np.linspace(2, 30, 5)

        plot_2d(Z, x_val=x_val, y_val=y_val)

        img = plt.gca().get_images()[0]
        assert img.get_extent() == [x_val[0], x_val[-1], y_val[0], y_val[-1]]

    @pytest.mark.unit
    def test_mask_nan_hides_pixels_transparent(self):
        Z = np.arange(20, dtype=float).reshape(4, 5)
        mask = np.zeros_like(Z, dtype=bool)
        mask[1:3, 2:4] = True  # True = keep

        plot_2d(Z, mask=mask, mask_value=np.nan)

        data = plt.gca().get_images()[0].get_array()
        assert np.ma.is_masked(data)
        np.testing.assert_array_equal(np.ma.getmaskarray(data), ~mask)

    @pytest.mark.unit
    def test_mask_value_zero_fills_literal_not_hidden(self):
        Z = np.arange(20, dtype=float).reshape(4, 5) + 1  # avoid zeros in data
        mask = np.zeros_like(Z, dtype=bool)
        mask[1:3, 2:4] = True

        plot_2d(Z, mask=mask, mask_value=0)

        data = np.asarray(plt.gca().get_images()[0].get_array())
        assert not np.ma.is_masked(data)
        np.testing.assert_allclose(data[mask], Z[mask])
        np.testing.assert_allclose(data[~mask], 0)

    @pytest.mark.unit
    def test_cluster_list_mask_filters_by_pvals_threshold(self):
        Z = np.ones((1, 10))
        # two "clusters": first is significant, second is not
        mask = [_cluster([0], [1, 2]), _cluster([0], [7, 8])]
        p_vals = np.array([0.01, 0.5])

        plot_2d(Z, mask=mask, p_vals=p_vals, p_thresh=0.05, mask_value=np.nan)

        data = plt.gca().get_images()[0].get_array()
        kept = ~np.ma.getmaskarray(data)
        assert kept[0, 1] and kept[0, 2]
        assert not kept[0, 7] and not kept[0, 8]

    @pytest.mark.unit
    def test_diverging_cmap_shifts_colorbar_ticks_by_center(self):
        Z = np.linspace(0, 1, 20).reshape(4, 5)

        plot_2d(Z, diverging_cmap=True, center=0.5, colorbar=True)

        cbar = plt.gcf().axes[-1]
        tick_labels = [t.get_text() for t in cbar.get_yticklabels()]
        # labels are shifted back by +0.5 and should be parseable floats
        floats = [float(t) for t in tick_labels if t]
        assert len(floats) > 0

    @pytest.mark.unit
    def test_contour_true_draws_contourf_not_image(self):
        Z = np.random.default_rng(0).normal(0, 1, (8, 20))

        plot_2d(Z, contour=True)

        ax = plt.gca()
        assert len(ax.get_images()) == 0
        assert len(ax.collections) > 0

    @pytest.mark.unit
    def test_yscale_linear_for_uniform_spacing(self):
        Z = np.zeros((6, 10))
        plot_2d(Z, y_val=np.linspace(1, 30, 6))
        assert plt.gca().get_yscale() == 'linear'

    @pytest.mark.unit
    def test_yscale_log_for_nonuniform_spacing(self):
        Z = np.zeros((6, 10))
        plot_2d(Z, y_val=np.geomspace(1, 30, 6))
        assert plt.gca().get_yscale() == 'log'

    @pytest.mark.unit
    def test_set_y_ticks_false_skips_tick_customization(self):
        Z = np.zeros((6, 10))
        default_ticks = None
        plot_2d(Z, y_val=np.geomspace(1, 30, 6), set_y_ticks=False)
        # should not force log scale when tick customization is skipped
        assert plt.gca().get_yscale() == 'linear'

    @pytest.mark.unit
    def test_colorbar_label_applied(self):
        Z = np.zeros((4, 5))
        plot_2d(Z, cbar_label='Power (au)')
        cbar_ax = plt.gcf().axes[-1]
        assert cbar_ax.get_ylabel() == 'Power (au)'

    @pytest.mark.unit
    def test_colorbar_false_no_extra_axis(self):
        Z = np.zeros((4, 5))
        plot_2d(Z, colorbar=False)
        assert len(plt.gcf().axes) == 1

    @pytest.mark.unit
    def test_single_row_yval_no_crash(self):
        # regression: np.diff on a length-1 y_val is empty, and indexing
        # [0] into it used to raise IndexError
        plot_2d(np.ones((1, 10)))
        assert plt.gca().get_yscale() == 'linear'

    @pytest.mark.unit
    def test_diverging_cmap_with_masked_array(self):
        # diverging_cmap's shift logic has a separate branch for masked
        # arrays (mask + diverging_cmap combined)
        Z = np.random.default_rng(0).normal(0, 1, (4, 10))
        mask = np.zeros_like(Z, dtype=bool)
        mask[1:3, 2:5] = True

        plot_2d(Z, mask=mask, mask_value=np.nan, diverging_cmap=True, center=0)

        data = plt.gca().get_images()[0].get_array()
        assert np.ma.is_masked(data)

    @pytest.mark.unit
    def test_nr_ticks_x_sets_explicit_xtick_count(self):
        plot_2d(np.zeros((4, 10)), nr_ticks_x=3)
        assert len(plt.gca().get_xticks()) == 3

    @pytest.mark.unit
    def test_yval_as_plain_list_converted_to_array(self):
        # default nr_ticks_y=5 with only 4 y-values means one tick repeats
        plot_2d(np.zeros((4, 10)), y_val=[1, 2, 3, 4])
        np.testing.assert_array_equal(plt.gca().get_yticks(), [1, 1, 2, 3, 4])

    @pytest.mark.unit
    def test_nr_ticks_y_none_defaults_to_five_for_long_axis(self):
        plot_2d(np.zeros((20, 10)), y_val=np.arange(20), nr_ticks_y=None)
        assert len(plt.gca().get_yticks()) == 5


# ============================================================================
# _get_continuous_segments
# ============================================================================

class TestGetContinuousSegments:
    @pytest.mark.unit
    def test_multiple_interior_segments(self):
        mask = np.array([False, True, True, False, True, False])
        segments = _get_continuous_segments(mask)
        assert [s.tolist() for s in segments] == [[1, 2], [4]]

    @pytest.mark.unit
    def test_segment_touching_start(self):
        mask = np.array([True, True, False, False])
        segments = _get_continuous_segments(mask)
        assert [s.tolist() for s in segments] == [[0, 1]]

    @pytest.mark.unit
    def test_segment_touching_end(self):
        mask = np.array([False, False, True, True])
        segments = _get_continuous_segments(mask)
        assert [s.tolist() for s in segments] == [[2, 3]]

    @pytest.mark.unit
    def test_all_true_single_segment(self):
        mask = np.array([True, True, True])
        segments = _get_continuous_segments(mask)
        assert [s.tolist() for s in segments] == [[0, 1, 2]]

    @pytest.mark.unit
    def test_all_false_empty_list(self):
        mask = np.array([False, False, False])
        assert _get_continuous_segments(mask) == []


# ============================================================================
# plot_significance
# ============================================================================

class TestPlotSignificance:
    @pytest.mark.unit
    def test_1d_perm_draws_segment_over_cluster_indices(self):
        x = np.linspace(0, 1, 20)
        y = np.random.default_rng(0).normal(0, 1, size=(10, 20))
        sig_mask = [np.arange(5, 10)]

        plt.plot(x, y.mean(axis=0), color='blue')
        plot_significance(x, y, stats='perm', sig_mask=sig_mask, color='blue')

        sig_line = plt.gca().get_lines()[-1]
        np.testing.assert_allclose(sig_line.get_xdata(), x[5:10])

    @pytest.mark.unit
    def test_1d_boolean_mask_draws_continuous_segments(self):
        x = np.linspace(0, 1, 20)
        y = np.random.default_rng(0).normal(0, 1, size=(10, 20))
        sig_mask = np.zeros(20, dtype=bool)
        sig_mask[3:7] = True

        plt.plot(x, y.mean(axis=0), color='green')
        plot_significance(x, y, stats='ttest', sig_mask=sig_mask, color='green')

        sig_line = plt.gca().get_lines()[-1]
        np.testing.assert_allclose(sig_line.get_xdata(), x[3:7])

    @pytest.mark.unit
    def test_1d_color_none_uses_current_line_color(self):
        x = np.linspace(0, 1, 20)
        y = np.random.default_rng(0).normal(0, 1, size=(10, 20))
        sig_mask = np.zeros(20, dtype=bool)
        sig_mask[3:7] = True

        plt.plot(x, y.mean(axis=0), color='purple')
        plot_significance(x, y, stats='ttest', sig_mask=sig_mask, color=None)

        sig_line = plt.gca().get_lines()[-1]
        assert mcolors.to_rgba(sig_line.get_color()) == mcolors.to_rgba('purple')

    @pytest.mark.unit
    def test_2d_without_yval_raises(self):
        y = np.random.default_rng(0).normal(0, 1, size=(10, 5, 20))
        with pytest.raises(ValueError, match='y_val'):
            plot_significance(np.arange(20), y, stats='ttest',
                               sig_mask=np.ones((5, 20), dtype=bool))

    @pytest.mark.unit
    def test_2d_perm_draws_contour_per_cluster(self):
        y = np.random.default_rng(0).normal(0, 1, size=(10, 5, 20))
        cluster = _cluster([1, 2], [8, 9, 10])

        plt.imshow(y.mean(axis=0), aspect='auto', origin='lower')
        plot_significance(np.arange(20), y, stats='perm', y_val=np.arange(5),
                           sig_mask=[cluster])

        assert len(plt.gca().collections) > 0

    @pytest.mark.unit
    def test_smooth_applies_gaussian_filter_1d(self):
        x = np.linspace(0, 1, 20)
        y = np.random.default_rng(0).normal(0, 1, size=(10, 20))
        sig_mask = np.zeros(20, dtype=bool)
        sig_mask[3:7] = True

        plt.plot(x, y.mean(axis=0), color='green')
        plot_significance(x, y, stats='ttest', sig_mask=sig_mask,
                           color='green', smooth=True)

        assert len(plt.gca().get_lines()) == 2

    @pytest.mark.unit
    def test_smooth_2d_perm_no_crash(self):
        y = np.random.default_rng(0).normal(0, 1, size=(10, 5, 20))
        cluster = _cluster([1, 2], [8, 9, 10])

        plt.imshow(y.mean(axis=0), aspect='auto', origin='lower')
        plot_significance(np.arange(20), y, stats='perm', y_val=np.arange(5),
                           sig_mask=[cluster], smooth=True)

        assert len(plt.gca().collections) > 0

    @pytest.mark.unit
    def test_smooth_2d_ttest_no_crash(self):
        y = np.random.default_rng(0).normal(0, 1, size=(10, 5, 20))
        sig_mask = np.zeros((5, 20), dtype=bool)
        sig_mask[1:3, 8:12] = True

        plt.imshow(y.mean(axis=0), aspect='auto', origin='lower')
        plot_significance(np.arange(20), y, stats='ttest', y_val=np.arange(5),
                           sig_mask=sig_mask, smooth=True)

        assert len(plt.gca().collections) > 0

    @pytest.mark.unit
    def test_2d_fully_significant_mask_falls_back_to_contourf(self):
        # when smoothing pushes the whole mask to one side of 0.5, a
        # 0.5-level contour line can't be drawn -- contourf fills instead
        y = np.random.default_rng(0).normal(0, 1, size=(10, 5, 20))
        sig_mask = np.ones((5, 20), dtype=bool)

        plt.imshow(y.mean(axis=0), aspect='auto', origin='lower')
        n_before = len(plt.gca().collections)
        plot_significance(np.arange(20), y, stats='ttest', y_val=np.arange(5),
                           sig_mask=sig_mask, smooth=True)

        assert len(plt.gca().collections) > n_before

    @pytest.mark.unit
    def test_no_sig_mask_computes_stats_internally(self):
        # inject a clear effect so ttest finds something without crashing
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=(20, 30))
        y[:, 10:20] += 3

        plt.plot(np.arange(30), y.mean(axis=0), color='black')
        plot_significance(np.arange(30), y, chance=0, stats='ttest', color='black')

        assert len(plt.gca().get_lines()) >= 2


# ============================================================================
# plot_erp_timecourse
# ============================================================================

class TestPlotErpTimecourse:
    @pytest.mark.unit
    def test_amplitude_converted_to_microvolts(self):
        # 5 uV stored as volts (5e-6) should be auto-rescaled to plot as 5
        erps = {'A': make_condition_evokeds({'C3': 5e-6}, ch_names=['C3'],
                                             n_subjects=3, noise_sd=0)}
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=['C3'], show_legend=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 5.0, atol=1e-6)

    @pytest.mark.unit
    def test_already_microvolt_scale_not_rescaled(self):
        # amplitude of 5 (already uV-scale) should NOT trigger the 1e6 rescale
        erps = {'A': make_condition_evokeds({'C3': 5.0}, ch_names=['C3'],
                                             n_subjects=3, noise_sd=0)}
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=['C3'], show_legend=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 5.0, atol=1e-6)

    @pytest.mark.unit
    def test_multiple_electrode_groups_get_contra_ipsi_labels(self):
        erps = {'A': make_condition_evokeds({'C3': 3.0, 'C4': 1.0},
                                             n_subjects=3, noise_sd=0)}
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=[['C3'], ['C4']])

        labels = [l.get_label() for l in plt.gca().get_lines()]
        assert 'A contra' in labels
        assert 'A ipsi' in labels

    @pytest.mark.unit
    def test_lateralized_computes_contra_minus_ipsi(self):
        erps = {'A': make_condition_evokeds({'C3': 3.0, 'C4': 1.0},
                                             n_subjects=3, noise_sd=0)}
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=[['C3'], ['C4']],
                             lateralized=True)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 2.0, atol=1e-6)
        assert line.get_label() == 'A (contra-ipsi)'

    @pytest.mark.unit
    def test_colors_list_not_mutated_across_calls(self):
        erps = {'A': make_condition_evokeds({'C3': 1.0}, ch_names=['C3'],
                                             n_subjects=3, noise_sd=0)}
        times = erps['A'][0].times
        my_colors = ['red', 'blue']

        plot_erp_timecourse(erps, times, elec_oi=['C3'], colors=my_colors)
        plt.close('all')
        plot_erp_timecourse(erps, times, elec_oi=['C3'], colors=my_colors)

        assert my_colors == ['red', 'blue']

    @pytest.mark.unit
    def test_color_assignment_sequential_across_conditions_and_groups(self):
        erps = {
            'A': make_condition_evokeds({'C3': 1.0, 'C4': 1.0}, n_subjects=3, noise_sd=0),
            'B': make_condition_evokeds({'C3': 1.0, 'C4': 1.0}, n_subjects=3, noise_sd=0, seed=1),
        }
        times = erps['A'][0].times
        my_colors = ['red', 'green', 'blue', 'orange']

        plot_erp_timecourse(erps, times, elec_oi=[['C3'], ['C4']], colors=my_colors)

        # exclude the default (unlabeled) zero-line/onset-marker reference lines
        colors_used = [l.get_color() for l in plt.gca().get_lines()
                       if not l.get_label().startswith('_')]
        assert colors_used == ['red', 'green', 'blue', 'orange']

    @pytest.mark.unit
    def test_window_oi_tuple_draws_rectangle(self):
        # regression: window_oi as a documented tuple used to crash on
        # list + tuple concatenation when times were in seconds
        erps = {'A': make_condition_evokeds({'C3': 1.0}, ch_names=['C3'],
                                             n_subjects=3, noise_sd=0,
                                             sfreq=200, n_samples=60)}
        times = erps['A'][0].times  # in seconds

        plot_erp_timecourse(erps, times, elec_oi=['C3'], window_oi=(0.0, 0.1))

        rects = [p for p in plt.gca().patches if isinstance(p, plt.Rectangle)]
        assert len(rects) == 1
        assert rects[0].get_x() == pytest.approx(0.0)
        assert rects[0].get_width() == pytest.approx(100.0)  # 0.1s -> 100ms

    @pytest.mark.unit
    def test_window_oi_three_element_tuple_no_crash(self):
        erps = {'A': make_condition_evokeds({'C3': 1.0}, ch_names=['C3'],
                                             n_subjects=3, noise_sd=0)}
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=['C3'], window_oi=(0.0, 0.1, 'pos'))

        rects = [p for p in plt.gca().patches if isinstance(p, plt.Rectangle)]
        assert len(rects) == 1

    @pytest.mark.unit
    def test_onset_times_draws_vertical_line_at_zero(self):
        erps = {'A': make_condition_evokeds({'C3': 1.0}, ch_names=['C3'],
                                             n_subjects=3, noise_sd=0)}
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=['C3'], onset_times=[0])

        vlines = [l for l in plt.gca().get_lines()
                  if len(set(l.get_xdata())) == 1 and l.get_xdata()[0] == 0]
        assert len(vlines) >= 1

    @pytest.mark.unit
    def test_stats_overlay_draws_extra_line(self):
        erps = {
            'A': make_condition_evokeds({'C3': 5.0}, ch_names=['C3'],
                                         n_subjects=15, noise_sd=0.2, seed=0),
        }
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=['C3'], stats='ttest')

        # base waveform line + at least one significance-marker line
        assert len(plt.gca().get_lines()) >= 2

    @pytest.mark.unit
    def test_list_input_treated_as_single_unnamed_condition(self):
        evokeds = make_condition_evokeds({'C3': 1.0}, ch_names=['C3'],
                                          n_subjects=3, noise_sd=0)
        times = evokeds[0].times

        plot_erp_timecourse(evokeds, times, elec_oi=['C3'])

        labels = [l.get_label() for l in plt.gca().get_lines()
                  if not l.get_label().startswith('_')]
        assert labels == ['temp']

    @pytest.mark.unit
    def test_cnds_filters_to_requested_conditions(self):
        erps = {
            'A': make_condition_evokeds({'C3': 1.0}, ch_names=['C3'], n_subjects=3, noise_sd=0),
            'B': make_condition_evokeds({'C3': 2.0}, ch_names=['C3'], n_subjects=3, noise_sd=0, seed=1),
        }
        times = erps['A'][0].times

        plot_erp_timecourse(erps, times, elec_oi=['C3'], cnds=['A'])

        labels = [l.get_label() for l in plt.gca().get_lines()
                  if not l.get_label().startswith('_')]
        assert labels == ['A']


# ============================================================================
# plot_tfr_timecourse
# ============================================================================

class TestPlotTfrTimecourse:
    @pytest.mark.unit
    def test_1d_freq_band_matches_manual_average(self):
        tfr = make_average_tfr(ch_names=['C3'], freqs=np.linspace(4, 20, 8),
                                amplitude_by_channel={'C3': 2.0})
        tfr_dict = {'A': tfr}

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='1d',
                             freq_oi=(8, 12), cnds=['A'])

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 2.0, atol=1e-10)

    @pytest.mark.unit
    def test_lateralized_computes_contra_minus_ipsi(self):
        tfr = make_average_tfr(ch_names=['C3', 'C4'], freqs=np.linspace(4, 20, 8),
                                amplitude_by_channel={'C3': 3.0, 'C4': 1.0})
        tfr_dict = {'A': tfr}

        plot_tfr_timecourse(tfr_dict, elec_oi=[['C3'], ['C4']], cnds=['A'],
                             timecourse='1d', freq_oi=(8, 12), lateralized=True)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 2.0, atol=1e-10)

    @pytest.mark.unit
    def test_baseline_comment_sets_db_colorbar_label(self):
        tfr = make_average_tfr(ch_names=['C3'])
        tfr.comment = 'baseline'
        tfr_dict = {'A': tfr}

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='2d', cnds=['A'])

        cbar_ax = plt.gcf().axes[-1]
        assert cbar_ax.get_ylabel() == 'Power (dB)'

    @pytest.mark.unit
    def test_onset_marker_drawn_in_1d_mode(self):
        # regression: onset_times used to be silently ignored outside
        # the 2D+contour combination
        tfr = make_average_tfr(ch_names=['C3'])
        tfr_dict = {'A': tfr}

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='1d',
                             freq_oi=(8, 12), cnds=['A'], onset_times=[0])

        vlines = [l for l in plt.gca().get_lines()
                  if len(set(l.get_xdata())) == 1 and l.get_xdata()[0] == 0]
        assert len(vlines) >= 1

    @pytest.mark.unit
    def test_colors_list_not_mutated_across_conditions(self):
        tfr_dict = {
            'A': make_average_tfr(ch_names=['C3', 'C4'],
                                   amplitude_by_channel={'C3': 1.0, 'C4': 1.0}),
            'B': make_average_tfr(ch_names=['C3', 'C4'],
                                   amplitude_by_channel={'C3': 1.0, 'C4': 1.0}),
        }
        my_colors = ['red', 'green', 'blue', 'orange']

        plot_tfr_timecourse(tfr_dict, elec_oi=[['C3'], ['C4']], cnds=['A', 'B'],
                             timecourse='1d', freq_oi=(8, 12), colors=my_colors)

        assert my_colors == ['red', 'green', 'blue', 'orange']

    @pytest.mark.unit
    def test_2d_contour_and_heatmap_share_data_range(self):
        tfr = make_average_tfr(ch_names=['C3'], amplitude_by_channel={'C3': 4.0})
        tfr_dict = {'A': tfr}

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='2d',
                             cnds=['A'], contour=False)
        heatmap_data = plt.gca().get_images()[0].get_array()
        plt.close('all')

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='2d',
                             cnds=['A'], contour=True)
        assert len(plt.gca().get_images()) == 0
        assert len(plt.gca().collections) > 0
        np.testing.assert_allclose(np.asarray(heatmap_data), 4.0)

    @pytest.mark.unit
    def test_contour_respects_mask_nonsig(self):
        # regression: the old bespoke contour branch ignored significance
        # masking entirely; now it's forwarded through plot_2d
        tfr = make_average_tfr(ch_names=['C3'], n_samples=30,
                                amplitude_by_channel={'C3': 1.0})
        tfr_dict = {'A': [make_average_tfr(ch_names=['C3'], n_samples=30,
                                            amplitude_by_channel={'C3': 1.0})
                           for _ in range(12)]}

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='2d',
                             cnds=['A'], contour=True, stats='ttest',
                             mask_nonsig=True)
        # two plot_2d calls (greyscale background + significant overlay)
        assert len(plt.gcf().axes) >= 2

    @pytest.mark.unit
    def test_2d_only_first_condition_plotted(self):
        tfr_dict = {
            'A': make_average_tfr(ch_names=['C3']),
            'B': make_average_tfr(ch_names=['C3']),
        }

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='2d')

        # one heatmap image only (first condition), not two
        assert len(plt.gca().get_images()) == 1

    @pytest.mark.unit
    def test_1d_single_frequency_selection(self):
        # regression: np.abs(...).argmin() returns np.integer, which
        # `isinstance(freq_idx, int)` doesn't match -- the freq axis was
        # dropped instead of preserved as a size-1 axis, corrupting the
        # later .mean(axis=1) reduction
        freqs = np.linspace(4, 20, 8)
        tfr_dict = {
            'A': [make_average_tfr(ch_names=['C3'], freqs=freqs,
                                    amplitude_by_channel={'C3': 3.0})
                  for _ in range(3)]
        }

        plot_tfr_timecourse(tfr_dict, elec_oi=['C3'], timecourse='1d',
                             freq_oi=8, cnds=['A'])

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 3.0, atol=1e-10)

    @pytest.mark.unit
    def test_1d_stats_overlay_draws_extra_line(self):
        rng = np.random.default_rng(0)
        freqs = np.linspace(4, 20, 8)
        tfrs = []
        for _ in range(15):
            tfr = make_average_tfr(ch_names=['C3'], freqs=freqs,
                                    amplitude_by_channel={'C3': 3.0})
            tfr.data += rng.normal(0, 0.2, tfr.data.shape)
            tfrs.append(tfr)

        plot_tfr_timecourse({'A': tfrs}, elec_oi=['C3'], timecourse='1d',
                             freq_oi=(8, 12), cnds=['A'], stats='ttest')

        assert len(plt.gca().get_lines()) >= 2


# ============================================================================
# plot_bdm_timecourse
# ============================================================================

class TestPlotBdmTimecourse:
    @pytest.mark.unit
    def test_1d_extracts_diagonal_from_gat(self):
        times = np.linspace(-0.1, 0.5, 10)
        gat = np.zeros((10, 10))
        np.fill_diagonal(gat, np.arange(10) * 0.1 + 0.5)
        bdms = make_bdm_result(gat, times=times, test_times=times, noise_sd=0)

        plot_bdm_timecourse(bdms, timecourse='1d')

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), np.diagonal(gat))

    @pytest.mark.unit
    def test_2d_gat_axis_labels(self):
        times = np.linspace(-0.1, 0.5, 10)
        bdms = make_bdm_result(np.full((10, 10), 0.6), times=times,
                                test_times=times, noise_sd=0)

        plot_bdm_timecourse(bdms, timecourse='2d_GAT')

        assert plt.gca().get_xlabel() == 'Test time (ms)'
        assert plt.gca().get_ylabel() == 'Train time (ms)'

    @pytest.mark.unit
    def test_2d_tfr_uses_frequency_axis(self):
        times = np.linspace(-0.1, 0.5, 10)
        freqs = np.linspace(4, 30, 6)
        bdms = make_bdm_result(np.full((6, 10), 0.6), times=times,
                                freqs=freqs, noise_sd=0)

        plot_bdm_timecourse(bdms, timecourse='2d_tfr')

        assert plt.gca().get_ylabel() == 'Frequency (Hz)'

    @pytest.mark.unit
    def test_chance_level_line_drawn_in_1d(self):
        times = np.linspace(-0.1, 0.5, 10)
        bdms = make_bdm_result(np.full(10, 0.6), times=times, noise_sd=0)

        plot_bdm_timecourse(bdms, timecourse='1d', chance_level=0.5, stats=False)

        hlines = [l for l in plt.gca().get_lines()
                  if len(set(l.get_ydata())) == 1 and l.get_ydata()[0] == 0.5]
        assert len(hlines) == 1

    @pytest.mark.unit
    def test_empty_colors_list_falls_back_in_2d_mode(self):
        # regression: operator-precedence bug meant undersized colors
        # lists were never backfilled outside timecourse=='1d'
        times = np.linspace(-0.1, 0.5, 10)
        bdms = make_bdm_result(np.full((10, 10), 0.6), times=times,
                                test_times=times, noise_sd=0)

        plot_bdm_timecourse(bdms, timecourse='2d_GAT', colors=[], stats=False)
        # no crash is the assertion; sanity check a heatmap was drawn
        assert len(plt.gca().get_images()) == 1

    @pytest.mark.unit
    def test_ttest_stats_2d_gat_no_crash(self):
        # regression: plot_significance used to crash on a dead p_vals kwarg
        rng = np.random.default_rng(0)
        times = np.linspace(-0.1, 0.5, 15)
        base = rng.normal(0.5, 0.05, (15, 15))
        bdms = [{'A': {'dec_scores': base + rng.normal(0, 0.3, (15, 15))},
                 'info': {'times': times, 'test_times': times}}
                for _ in range(12)]

        plot_bdm_timecourse(bdms, timecourse='2d_GAT', stats='ttest')

        assert len(plt.gca().get_images()) == 1

    @pytest.mark.unit
    def test_single_dict_input_treated_as_one_subject(self):
        times = np.linspace(-0.1, 0.5, 10)
        bdm = {'A': {'dec_scores': np.full(10, 0.6)}, 'info': {'times': times}}

        plot_bdm_timecourse(bdm, timecourse='1d', stats=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 0.6)

    @pytest.mark.unit
    def test_multi_condition_2d_mode_plots_first_only(self):
        times = np.linspace(-0.1, 0.5, 10)
        bdms = make_bdm_result(np.full((10, 10), 0.6), times=times,
                                test_times=times, cnd='A', noise_sd=0)
        for b in bdms:
            b['B'] = {'dec_scores': np.full((10, 10), 0.9)}

        plot_bdm_timecourse(bdms, timecourse='2d_GAT', cnds=['A', 'B'], stats=False)

        assert len(plt.gca().get_images()) == 1

    @pytest.mark.unit
    def test_mask_nonsig_draws_greyscale_then_color_overlay(self):
        rng = np.random.default_rng(0)
        times = np.linspace(-0.1, 0.5, 15)
        base = rng.normal(0.5, 0.05, (15, 15))
        bdms = [{'A': {'dec_scores': base + rng.normal(0, 0.3, (15, 15))},
                 'info': {'times': times, 'test_times': times}}
                for _ in range(12)]

        plot_bdm_timecourse(bdms, timecourse='2d_GAT', stats='ttest',
                             mask_nonsig=True)

        # greyscale background image + significant-overlay image
        assert len(plt.gca().get_images()) == 2


# ============================================================================
# plot_ctf_timecourse
# ============================================================================

class TestPlotCtfTimecourse:
    @pytest.mark.unit
    def test_1d_single_band_matches_input(self):
        times = np.linspace(-0.1, 0.5, 10)
        ctfs = make_ctf_result(np.full((1, 10), 0.3), times=times,
                                bands=['all'], noise_sd=0)

        plot_ctf_timecourse(ctfs, timecourse='1d', stats=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 0.3)

    @pytest.mark.unit
    def test_1d_multi_band_no_band_oi_warns_and_averages(self):
        # regression: this used to construct-and-discard a Warning object
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.stack([np.full(10, 0.2), np.full(10, 0.6)])  # 2 bands
        ctfs = make_ctf_result(raw, times=times, bands=['theta', 'alpha'],
                                noise_sd=0)

        with pytest.warns(UserWarning, match='band_oi'):
            plot_ctf_timecourse(ctfs, timecourse='1d', band_oi=None, stats=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 0.4)  # mean(0.2, 0.6)

    @pytest.mark.unit
    def test_band_oi_selects_specific_band(self):
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.stack([np.full(10, 0.2), np.full(10, 0.6)])
        ctfs = make_ctf_result(raw, times=times, bands=['theta', 'alpha'],
                                noise_sd=0)

        plot_ctf_timecourse(ctfs, timecourse='1d', band_oi='alpha', stats=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 0.6)

    @pytest.mark.unit
    def test_1d_multi_bin_uses_bin_sized_colors_not_condition_sized(self):
        # regression: colors[b] indexed by bin count while colors was
        # only sized for condition count (1 condition here, 6 bins)
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.random.default_rng(0).normal(1, 0.1, (1, 10, 6))
        ctfs = make_ctf_result(raw, times=times, bands=['all'], noise_sd=0)

        plot_ctf_timecourse(ctfs, timecourse='1d', stats=False)

        bin_lines = [l for l in plt.gca().get_lines() if 'bin_' in l.get_label()]
        assert len(bin_lines) == 6

    @pytest.mark.unit
    def test_2d_ctf_even_bin_count_wraps_around(self):
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.zeros((1, 10, 6))  # even bin count -> should wrap
        ctfs = make_ctf_result(raw, times=times, bands=['all'], noise_sd=0)

        plot_ctf_timecourse(ctfs, timecourse='2d_ctf')

        img = plt.gca().get_images()[0]
        # wrapped to 7 rows (6 + 1 duplicated first bin)
        assert img.get_array().shape[0] == 7

    @pytest.mark.unit
    def test_2d_ctf_odd_bin_count_no_wrap(self):
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.zeros((1, 10, 7))  # odd bin count -> no wrap needed
        ctfs = make_ctf_result(raw, times=times, bands=['all'], noise_sd=0)

        plot_ctf_timecourse(ctfs, timecourse='2d_ctf')

        img = plt.gca().get_images()[0]
        assert img.get_array().shape[0] == 7

    @pytest.mark.unit
    def test_2d_ctf_multi_output_warns(self):
        # regression: bare Warning() construction that never fired
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.zeros((2, 10, 6))  # 2 "bands" -> shape[1] > 1 in 2d_ctf branch
        ctfs = make_ctf_result(raw, times=times, bands=['theta', 'alpha'], noise_sd=0)

        with pytest.warns(UserWarning, match='single output'):
            plot_ctf_timecourse(ctfs, timecourse='2d_ctf')

    @pytest.mark.unit
    def test_empty_colors_list_falls_back_in_2d_mode(self):
        # regression: same operator-precedence bug as plot_bdm_timecourse
        times = np.linspace(-0.1, 0.5, 10)
        ctfs = make_ctf_result(np.full((1, 10), 0.3), times=times,
                                bands=['all'], noise_sd=0)

        plot_ctf_timecourse(ctfs, timecourse='2d_gat', colors=[], stats=False)

        assert len(plt.gca().get_images()) == 1

    @pytest.mark.unit
    def test_ttest_stats_2d_gat_no_crash(self):
        # regression: plot_significance used to crash on a dead p_vals kwarg
        rng = np.random.default_rng(0)
        times = np.linspace(-0.1, 0.5, 15)
        raw = np.full((1, 15), 0.3)
        ctfs = make_ctf_result(raw, times=times, bands=['all'],
                                n_subjects=12, noise_sd=0.2, seed=1)

        plot_ctf_timecourse(ctfs, timecourse='2d_gat', stats='ttest')

        assert len(plt.gca().get_images()) == 1

    @pytest.mark.unit
    def test_single_dict_input_treated_as_one_subject(self):
        times = np.linspace(-0.1, 0.5, 10)
        ctf = {'A': {'raw_slopes': np.full((1, 10), 0.3)},
               'info': {'times': times, 'bands': ['all']}}

        plot_ctf_timecourse(ctf, timecourse='1d', stats=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 0.3)

    @pytest.mark.unit
    def test_multi_condition_2d_mode_plots_first_only(self):
        times = np.linspace(-0.1, 0.5, 10)
        ctfs = make_ctf_result(np.full((1, 10), 0.3), times=times,
                                bands=['all'], cnd='A', noise_sd=0)
        for c in ctfs:
            c['B'] = {'raw_slopes': np.full((1, 10), 0.9)}

        plot_ctf_timecourse(ctfs, timecourse='2d_gat', cnds=['A', 'B'], stats=False)

        assert len(plt.gca().get_images()) == 1

    @pytest.mark.unit
    def test_multiple_outputs_get_combined_labels(self):
        times = np.linspace(-0.1, 0.5, 10)
        rng = np.random.default_rng(0)
        ctfs = []
        for _ in range(3):
            ctfs.append({
                'A': {'raw_slopes': np.full((1, 10), 0.2) + rng.normal(0, 0.01, (1, 10)),
                      'param_slopes': np.full((1, 10), 0.5) + rng.normal(0, 0.01, (1, 10))},
                'info': {'times': times, 'bands': ['all']},
            })

        plot_ctf_timecourse(ctfs, timecourse='1d', output=['raw_slopes', 'param_slopes'],
                             stats=False)

        labels = sorted(l.get_label() for l in plt.gca().get_lines()
                         if not l.get_label().startswith('_'))
        assert labels == ['A - param_slopes', 'A - raw_slopes']

    @pytest.mark.unit
    def test_avg_bins_averages_only_nonzero_bins(self):
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.zeros((1, 10, 6))
        raw[0, :, 0] = 1.0  # only bin 0 is non-zero across all subjects
        ctfs = make_ctf_result(raw, times=times, bands=['all'], noise_sd=0)

        plot_ctf_timecourse(ctfs, timecourse='1d', avg_bins=True, stats=False)

        line = plt.gca().get_lines()[0]
        np.testing.assert_allclose(line.get_ydata(), 1.0)

    @pytest.mark.unit
    def test_1d_single_band_stats_overlay_draws_extra_line(self):
        times = np.linspace(-0.1, 0.5, 15)
        raw = np.full((1, 15), 0.3)
        ctfs = make_ctf_result(raw, times=times, bands=['all'],
                                n_subjects=15, noise_sd=0.2, seed=2)

        plot_ctf_timecourse(ctfs, timecourse='1d', stats='ttest')

        assert len(plt.gca().get_lines()) >= 2

    @pytest.mark.unit
    def test_2d_tfr_uses_frequency_axis(self):
        times = np.linspace(-0.1, 0.5, 10)
        freq_bands = [[4, 8], [8, 12], [12, 20]]
        raw = np.random.default_rng(0).normal(0, 1, (3, 10))
        ctfs = make_ctf_result(raw, times=times, bands=freq_bands, noise_sd=0)
        for c in ctfs:
            c['info']['freqs'] = freq_bands

        plot_ctf_timecourse(ctfs, timecourse='2d_tfr', stats=False)

        assert plt.gca().get_ylabel() == 'Frequency (Hz)'

    @pytest.mark.unit
    def test_2d_ctf_averages_over_extra_channel_dimension(self):
        # regression-adjacent: y.ndim > 3 after band-squeeze should warn
        # and average over the channel axis rather than crash
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.zeros((1, 10, 3, 6))  # 1 band, 10 times, 3 channels, 6 bins
        ctfs = make_ctf_result(raw, times=times, bands=['all'], noise_sd=0)

        with pytest.warns(UserWarning, match='channel'):
            plot_ctf_timecourse(ctfs, timecourse='2d_ctf')

        assert len(plt.gca().get_images()) == 1


# ============================================================================
# plot_topography / plot_erp_topography
# ============================================================================

class TestPlotTopography:
    @pytest.mark.unit
    def test_channel_count_matches_montage(self):
        montage = 'biosemi64'
        n_ch = len(mne.channels.make_standard_montage(montage).ch_names)
        X = np.zeros(n_ch)

        plot_topography(X, montage=montage)  # no crash

        assert len(plt.gcf().axes) >= 1


class TestPlotErpTopography:
    @pytest.mark.unit
    def test_raw_mode_averages_time_window(self):
        evokeds = create_biosemi64_evoked_pair()
        times = evokeds[0].times
        erps = {'A': evokeds}

        captured = {}
        def fake_plot_topography(data, **kwargs):
            captured['data'] = data.copy()

        with patch.object(plotmod, 'plot_topography', side_effect=fake_plot_topography):
            plot_erp_topography(erps, times, montage='biosemi64', topo='raw')

        ch_names = mne.channels.make_standard_montage('biosemi64').ch_names
        data = captured['data']
        assert data[ch_names.index('O1')] == pytest.approx(5.0)
        assert data[ch_names.index('P8')] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_diff_mode_computes_contra_minus_ipsi(self):
        # regression: this path called ERP.group_erp(set_mean=True) and
        # get_diff_pairs(montage, ch_names) -- both TypeErrors, so this
        # mode had never been callable before
        evokeds = create_biosemi64_evoked_pair()
        times = evokeds[0].times
        erps = {'A': evokeds}

        captured = {}
        def fake_plot_topography(data, **kwargs):
            captured['data'] = data.copy()

        with patch.object(plotmod, 'plot_topography', side_effect=fake_plot_topography):
            plot_erp_topography(erps, times, montage='biosemi64', topo='diff')

        ch_names = mne.channels.make_standard_montage('biosemi64').ch_names
        data = captured['data']
        # O1=5, O2=2, P7=3, P8=1 (see create_biosemi64_evoked_pair)
        assert data[ch_names.index('O1')] == pytest.approx(3.0)   # O1 - O2
        assert data[ch_names.index('O2')] == pytest.approx(-3.0)  # O2 - O1
        assert data[ch_names.index('P7')] == pytest.approx(2.0)   # P7 - P8
        assert data[ch_names.index('P8')] == pytest.approx(-2.0)  # P8 - P7
        assert data[ch_names.index('Cz')] == pytest.approx(0.0)   # midline

    @pytest.mark.unit
    def test_window_oi_restricts_averaging_window(self):
        ch_names = ['Cz']
        info = mne.create_info(ch_names, 100., ch_types='eeg')
        times = np.linspace(-0.1, 0.5, 61)
        # value ramps over time so window selection is verifiable
        data = times.reshape(1, -1) * 10
        evoked = mne.EvokedArray(data, info, tmin=times[0])
        erps = {'A': [evoked]}

        captured = {}
        def fake_plot_topography(data, **kwargs):
            captured['data'] = data.copy()

        with patch.object(plotmod, 'plot_topography', side_effect=fake_plot_topography):
            plot_erp_topography(erps, times, montage=None if False else 'biosemi64',
                                 topo='raw', window_oi=(0.0, 0.1))

        # sanity: captured a single-value-per-channel array sized for
        # the montage, not for this 1-channel evoked (plot_topography
        # always builds its own montage-based info)
        assert captured['data'].shape == (1,)
        expected = data[0, (times >= 0.0) & (times <= 0.1)].mean()
        assert captured['data'][0] == pytest.approx(expected)

    @pytest.mark.unit
    def test_list_input_treated_as_single_unnamed_condition(self):
        evokeds = create_biosemi64_evoked_pair()
        times = evokeds[0].times

        # a bare list (not a dict) should still produce exactly one subplot
        plot_erp_topography(evokeds, times, montage='biosemi64')

        assert len(plt.gcf().axes) == 1

    @pytest.mark.unit
    def test_cnds_filters_to_requested_conditions(self):
        evokeds = create_biosemi64_evoked_pair()
        times = evokeds[0].times
        erps = {'A': evokeds, 'B': evokeds}

        plot_erp_topography(erps, times, montage='biosemi64', cnds=['A'])

        assert len(plt.gcf().axes) == 1


# ============================================================================
# Cross-cutting regressions
# ============================================================================

class TestRegressions:
    @pytest.mark.unit
    def test_plot_2d_mask_value_and_nan_produce_different_output(self):
        Z = np.full((4, 5), 2.0)
        mask = np.zeros_like(Z, dtype=bool)
        mask[0, 0] = True

        plot_2d(Z.copy(), mask=mask, mask_value=0)
        data_zero = np.asarray(plt.gca().get_images()[0].get_array())
        plt.close('all')

        plot_2d(Z.copy(), mask=mask, mask_value=np.nan)
        data_nan = plt.gca().get_images()[0].get_array()

        assert not np.ma.is_masked(data_zero)
        assert np.ma.is_masked(data_nan)

    @pytest.mark.unit
    def test_ctf_timecourse_warning_actually_fires(self):
        # bare `Warning(...)` used to construct-and-discard; confirm the
        # fixed warnings.warn(...) calls are real, catchable warnings
        times = np.linspace(-0.1, 0.5, 10)
        raw = np.stack([np.full(10, 0.2), np.full(10, 0.6)])
        ctfs = make_ctf_result(raw, times=times, bands=['theta', 'alpha'], noise_sd=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plot_ctf_timecourse(ctfs, timecourse='1d', band_oi=None, stats=False)

        assert any('band_oi' in str(w.message) for w in caught)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
