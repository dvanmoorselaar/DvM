"""
Test suite for open_dvm.visualization.plot_utils.

Organization
------------
- TestResolveCndDiffList: normalize a single contrast / list of contrasts
- TestResolveCndDiffColors: None/str/list -> per-contrast flat colors
- TestCndDiffPointColors: per-timepoint auto-coloring from condition colors
"""

import numpy as np
import pytest

from open_dvm.visualization.plot_utils import (
    resolve_cnd_diff_list,
    resolve_cnd_diff_colors,
    cnd_diff_point_colors,
)


class TestResolveCndDiffList:
    @pytest.mark.unit
    def test_none_returns_empty_list(self):
        assert resolve_cnd_diff_list(None) == []

    @pytest.mark.unit
    def test_single_pair_wrapped_in_list(self):
        assert resolve_cnd_diff_list(('easy', 'hard')) == [('easy', 'hard')]

    @pytest.mark.unit
    def test_list_of_pairs_passed_through(self):
        contrasts = [('easy', 'hard'), ('medium', 'hard')]
        assert resolve_cnd_diff_list(contrasts) == contrasts


class TestResolveCndDiffColors:
    @pytest.mark.unit
    def test_none_single_contrast_falls_back_to_default(self):
        assert resolve_cnd_diff_colors(None, n_contrasts=1) == ['grey']

    @pytest.mark.unit
    def test_none_multiple_contrasts_auto_cycles_palette(self):
        colors = resolve_cnd_diff_colors(None, n_contrasts=3)
        assert len(colors) == 3
        assert len(set(colors)) == 3  # all distinct

    @pytest.mark.unit
    def test_str_applied_to_every_contrast(self):
        colors = resolve_cnd_diff_colors('black', n_contrasts=3)
        assert colors == ['black', 'black', 'black']

    @pytest.mark.unit
    def test_list_matching_length_used_as_is(self):
        colors = resolve_cnd_diff_colors(['black', 'purple'], n_contrasts=2)
        assert colors == ['black', 'purple']

    @pytest.mark.unit
    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='must match'):
            resolve_cnd_diff_colors(['black'], n_contrasts=2)


class TestCndDiffPointColors:
    """Colors alternate strictly by position among the significant
    timepoints (independent of the underlying data direction) -- this is
    a readability device, not a claim about which condition is higher."""

    @pytest.mark.unit
    def test_perm_alternates_by_position_across_one_cluster(self):
        sig_mask = [(np.array([2, 3, 4, 5]),)]  # one cluster, mne format

        colors = cnd_diff_point_colors('a', 'b', sig_mask, 'perm',
                                        n_timepoints=8,
                                        cnds=['a', 'b'], colors=['red', 'green'])

        np.testing.assert_array_equal(
            colors[2:6], ['red', 'green', 'red', 'green'])

    @pytest.mark.unit
    def test_perm_alternation_continues_across_multiple_clusters(self):
        # two separate clusters -- alternation continues in temporal
        # order across both, not reset at each cluster boundary
        sig_mask = [(np.array([1, 2]),), (np.array([6, 7]),)]

        colors = cnd_diff_point_colors('a', 'b', sig_mask, 'perm',
                                        n_timepoints=9,
                                        cnds=['a', 'b'], colors=['red', 'green'])

        np.testing.assert_array_equal(colors[1:3], ['red', 'green'])
        np.testing.assert_array_equal(colors[6:8], ['red', 'green'])

    @pytest.mark.unit
    def test_ttest_boolean_mask_alternates_by_position(self):
        sig_mask = np.array([False, True, True, True, False])

        colors = cnd_diff_point_colors('a', 'b', sig_mask, 'ttest',
                                        n_timepoints=5,
                                        cnds=['a', 'b'], colors=['red', 'green'])

        np.testing.assert_array_equal(colors[1:4], ['red', 'green', 'red'])

    @pytest.mark.unit
    def test_no_significant_points_returns_all_first_color(self):
        sig_mask = []  # no significant clusters at all

        colors = cnd_diff_point_colors('a', 'b', sig_mask, 'perm',
                                        n_timepoints=4,
                                        cnds=['a', 'b'], colors=['red', 'green'])

        np.testing.assert_array_equal(colors, ['red', 'red', 'red', 'red'])

    @pytest.mark.unit
    def test_cnds_none_returns_none(self):
        assert cnd_diff_point_colors('a', 'b', [], 'perm', 4,
                                      cnds=None, colors=['red', 'green']) is None

    @pytest.mark.unit
    def test_colors_none_returns_none(self):
        assert cnd_diff_point_colors('a', 'b', [], 'perm', 4,
                                      cnds=['a', 'b'], colors=None) is None

    @pytest.mark.unit
    def test_condition_not_in_cnds_returns_none(self):
        assert cnd_diff_point_colors('a', 'c', [], 'perm', 4,
                                      cnds=['a', 'b'], colors=['red', 'green']) is None

    @pytest.mark.unit
    def test_color_index_out_of_range_returns_none(self):
        # cnds has 3 entries but colors only covers the first 2
        assert cnd_diff_point_colors('a', 'c', [], 'perm', 4,
                                      cnds=['a', 'b', 'c'],
                                      colors=['red', 'green']) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
