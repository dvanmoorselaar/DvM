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
    @pytest.mark.unit
    def test_colors_by_which_condition_is_higher_per_timepoint(self):
        y1 = np.array([[1.0, 1.0], [1.0, 1.0]])  # mean = [1, 1]
        y2 = np.array([[2.0, 0.0], [2.0, 0.0]])  # mean = [2, 0]

        colors = cnd_diff_point_colors('a', 'b', y1, y2,
                                        cnds=['a', 'b'], colors=['red', 'green'])

        # timepoint 0: y2 > y1 -> 'b' color; timepoint 1: y1 > y2 -> 'a' color
        np.testing.assert_array_equal(colors, ['green', 'red'])

    @pytest.mark.unit
    def test_cnds_none_returns_none(self):
        y1 = np.zeros((2, 3))
        y2 = np.zeros((2, 3))
        assert cnd_diff_point_colors('a', 'b', y1, y2,
                                      cnds=None, colors=['red', 'green']) is None

    @pytest.mark.unit
    def test_colors_none_returns_none(self):
        y1 = np.zeros((2, 3))
        y2 = np.zeros((2, 3))
        assert cnd_diff_point_colors('a', 'b', y1, y2,
                                      cnds=['a', 'b'], colors=None) is None

    @pytest.mark.unit
    def test_condition_not_in_cnds_returns_none(self):
        y1 = np.zeros((2, 3))
        y2 = np.zeros((2, 3))
        assert cnd_diff_point_colors('a', 'c', y1, y2,
                                      cnds=['a', 'b'], colors=['red', 'green']) is None

    @pytest.mark.unit
    def test_color_index_out_of_range_returns_none(self):
        # cnds has 3 entries but colors only covers the first 2
        y1 = np.zeros((2, 3))
        y2 = np.zeros((2, 3))
        assert cnd_diff_point_colors('a', 'c', y1, y2,
                                      cnds=['a', 'b', 'c'],
                                      colors=['red', 'green']) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
