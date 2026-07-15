"""
Test suite for open_dvm.stats.stats_utils.

Organization
------------
- TestBootstrapSE: bootstrap standard error estimation
- TestConfidenceInt: Cousineau-Morey within-subject confidence intervals
- TestPairedT: paired t-test wrapper
- TestConnectedAdjacency: adjacency matrix construction for cluster tests
- TestPerformStats: group-level statistical testing (perm/ttest/fdr)
"""

import warnings

import pytest
import numpy as np
from scipy.stats import ttest_rel, t as tdist

from open_dvm.stats.stats_utils import (
    bootstrap_SE,
    confidence_int,
    paired_t,
    connected_adjacency,
    perform_stats,
)


# ============================================================================
# bootstrap_SE
# ============================================================================

class TestBootstrapSE:
    @pytest.mark.unit
    def test_recovers_known_mean_and_theoretical_se(self):
        rng = np.random.default_rng(0)
        true_mean, true_std, n = 5.0, 2.0, 200
        X = rng.normal(true_mean, true_std, size=(n, 3))

        SE, avg_X = bootstrap_SE(X, nr_iter=2000)

        np.testing.assert_allclose(avg_X, true_mean, atol=0.2)
        theoretical_se = true_std / np.sqrt(n)
        np.testing.assert_allclose(SE, theoretical_se, rtol=0.25)

    @pytest.mark.unit
    def test_output_shapes_match_number_of_variables(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(50, 4))

        SE, avg_X = bootstrap_SE(X, nr_iter=500)

        assert SE.shape == (4,)
        assert avg_X.shape == (4,)


# ============================================================================
# confidence_int
# ============================================================================

class TestConfidenceInt:
    @pytest.mark.unit
    def test_removes_between_subject_variance(self):
        """Cousineau normalization should give small, condition-consistent
        CIs based only on within-subject noise, even when between-subject
        means differ enormously."""
        rng = np.random.default_rng(1)
        subj_means = np.array([10, 20, 30, 40]).reshape(4, 1)
        noise = rng.normal(0, 0.5, size=(4, 3))
        data = subj_means + noise

        ci = confidence_int(data, p_value=0.05, tail='two', morey=True)

        assert np.all(ci < 2.0)  # small: reflects only the 0.5-std noise

    @pytest.mark.unit
    def test_much_smaller_than_naive_non_normalized_ci(self):
        rng = np.random.default_rng(1)
        subj_means = np.array([10, 20, 30, 40]).reshape(4, 1)
        noise = rng.normal(0, 0.5, size=(4, 3))
        data = subj_means + noise

        ci = confidence_int(data, p_value=0.05, tail='two', morey=True)

        naive_std = np.std(data, axis=0, ddof=1)
        naive_ci = naive_std / np.sqrt(4) * abs(tdist.ppf(0.025, 3))
        assert np.all(ci < naive_ci / 5)

    @pytest.mark.unit
    def test_invalid_tail_warns_and_uses_default(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (10, 3))

        with pytest.warns(UserWarning, match='Incorrect argument'):
            ci_invalid = confidence_int(data, tail='bogus')

        ci_two = confidence_int(data, tail='two')
        np.testing.assert_allclose(ci_invalid, ci_two)


# ============================================================================
# paired_t
# ============================================================================

class TestPairedT:
    @pytest.mark.unit
    def test_matches_scipy_ttest_rel_statistic(self):
        rng = np.random.default_rng(0)
        a = rng.normal(5, 1, 20)
        b = a + rng.normal(2, 0.1, 20)

        result = paired_t(a, b)

        assert result == pytest.approx(ttest_rel(a, b)[0])


# ============================================================================
# connected_adjacency
# ============================================================================

class TestConnectedAdjacency:
    @staticmethod
    def _manual_adjacency(r, c, connect):
        expected = np.zeros((r * c, r * c), dtype=int)
        offsets = {
            '4': [(-1, 0), (1, 0), (0, -1), (0, 1)],
            '8': [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                  if not (di == 0 and dj == 0)],
        }[connect]
        for i in range(r):
            for j in range(c):
                idx = i * c + j
                for di, dj in offsets:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < r and 0 <= jj < c:
                        expected[idx, ii * c + jj] = 1
        return expected

    @pytest.mark.unit
    @pytest.mark.parametrize('connect', ['4', '8'])
    @pytest.mark.parametrize('shape', [(3, 3), (3, 4)])
    def test_matches_manual_adjacency_square_and_nonsquare(self, connect, shape):
        r, c = shape
        grid = np.zeros((r, c))

        adj = connected_adjacency(grid, connect).toarray().astype(int)

        np.testing.assert_array_equal(adj, self._manual_adjacency(r, c, connect))

    @pytest.mark.unit
    def test_patch_size_groups_into_coarser_grid(self):
        grid = np.zeros((4, 4))

        adj = connected_adjacency(grid, '4', patch_size=(2, 2)).toarray().astype(int)

        assert adj.shape == (4, 4)
        np.testing.assert_array_equal(adj, self._manual_adjacency(2, 2, '4'))

    @pytest.mark.unit
    def test_invalid_connect_raises(self):
        grid = np.zeros((3, 3))
        with pytest.raises(ValueError):
            connected_adjacency(grid, '6')

    @pytest.mark.unit
    def test_matrix_is_symmetric(self):
        grid = np.zeros((5, 4))
        adj = connected_adjacency(grid, '8').toarray()

        np.testing.assert_array_equal(adj, adj.T)


# ============================================================================
# perform_stats
# ============================================================================

class TestPerformStats:
    @pytest.mark.unit
    def test_ttest_detects_injected_effect(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=(20, 100))
        y[:, 40:60] += 1.5

        t_vals, sig_mask, p_vals = perform_stats(y, chance=0, stat_test='ttest')

        assert sig_mask[40:60].mean() > 0.8  # most of the true effect detected
        assert t_vals.shape == (100,)
        assert p_vals.shape == (100,)

    @pytest.mark.unit
    def test_fdr_detects_injected_effect(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=(20, 100))
        y[:, 40:60] += 1.5

        t_vals, sig_mask, p_vals = perform_stats(y, chance=0, stat_test='fdr')

        assert sig_mask[40:60].mean() > 0.8

    @pytest.mark.unit
    def test_fdr_handles_2d_timefreq_input(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=(20, 5, 30))
        y[:, 2, 10:15] += 3.0

        t_vals, sig_mask, p_vals = perform_stats(y, chance=0, stat_test='fdr')

        assert sig_mask.shape == (5, 30)
        assert sig_mask[2, 10:15].all()

    @pytest.mark.unit
    def test_perm_finds_precise_cluster_boundary(self):
        """Cluster-based permutation test should recover the injected
        effect's boundary with high precision (this is the actual
        statistical engine behind published significance claims)."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=(20, 100))
        y[:, 40:60] += 1.5

        t_vals, clusters, p_vals = perform_stats(y, chance=0, stat_test='perm')

        assert len(clusters) == 1
        assert p_vals[0] < 0.05
        # MNE's cluster format: a tuple containing an array of literal
        # sample indices (not a boolean mask)
        cluster_idx = clusters[0][0]
        assert cluster_idx.min() >= 38
        assert cluster_idx.max() <= 61

    @pytest.mark.unit
    def test_perm_no_effect_finds_no_significant_clusters(self):
        rng = np.random.default_rng(2)
        y = rng.normal(0, 1, size=(20, 50))  # pure noise, no effect

        t_vals, clusters, p_vals = perform_stats(y, chance=0, stat_test='perm')

        assert len(clusters) == 0

    @pytest.mark.unit
    def test_invalid_stat_test_raises(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, (10, 20))
        with pytest.raises(ValueError):
            perform_stats(y, stat_test='bogus')

    @pytest.mark.unit
    def test_noncallable_statfun_raises(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, (10, 20))
        with pytest.raises(TypeError):
            perform_stats(y, stat_test='perm', statfun='not_callable')

    @pytest.mark.unit
    def test_p_cluster_converts_to_threshold_consistently(self):
        """p_cluster=0.05 should produce the same result as passing the
        equivalent t-statistic threshold directly."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=(20, 60))
        y[:, 20:30] += 1.5
        n_subjects = y.shape[0]
        equivalent_threshold = tdist.ppf(1 - 0.05 / 2, n_subjects - 1)

        _, clusters_a, p_vals_a = perform_stats(
            y, stat_test='perm', p_cluster=0.05
        )
        _, clusters_b, p_vals_b = perform_stats(
            y, stat_test='perm', threshold=equivalent_threshold
        )

        assert len(clusters_a) == len(clusters_b)
        np.testing.assert_allclose(p_vals_a, p_vals_b)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
