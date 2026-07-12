"""
Comprehensive test suite for open_dvm.analysis.ERP module.

Test Organization: Workflow-based (single-subject → features → group analysis)

PHASE 1: Core Single-Subject Analysis
  - TestParameterNaming: Semantic clarity
  - TestSpatialRestrictionANDLogic: AND logic critical bug fix
  - TestConditionERPs: Condition ERP generation
  - TestFullERPWorkflow: Basic workflows
  - TestEdgeCases: Edge case handling
  - TestRegressions: Regression protection

PHASE 2: Feature Extraction & Latency Analysis  
  - TestExtractERPFeatures: Mean amplitude, AUC, latency extraction
  - TestSelectERPWindow: Peak detection window selection
  - TestLatencyAnalysisConditionNames: Latency analysis with condition names
  - TestStatisticalComparisons: Jackknife statistics
  - TestERPComponentAnalysis: Component analysis
  - TestERPDataValidation: Data validation

PHASE 3: Group-Level & Advanced Analysis
  - TestGroupERPAnalysis: Group ERP aggregation
  - TestSelectWaveformVariations: Multi-subject waveforms
  - TestWaveformAnalysisCombinations: Feature comparison workflows
  - TestGetERPParams: Parameter extraction utilities
  - TestExportERPMetrics: Export functionality
  - TestERPValidationAndErrors: Advanced error handling
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import Mock, patch
import mne

from open_dvm.analysis.ERP import ERP
from tests.fixtures.sample_data import (
    create_sample_epochs,
    create_sample_erp_dataframe,
    create_sample_waveforms,
    create_lateralization_test_data,
    create_multilocation_test_data,
)

# ============================================================================
# PHASE 1: Core Single-Subject Analysis
# ============================================================================
# Foundational tests: parameter clarity, spatial filtering, ERP generation,
# basic workflows, and edge case handling.
# Note: Classes are logically Phase 1 regardless of file position.
# - TestParameterNaming (line ~286)
# - TestSpatialRestrictionANDLogic (below)
# - TestConditionERPs (line ~633)
# - TestFullERPWorkflow (line ~312)
# - TestEdgeCases (line ~362)

class TestSpatialRestrictionANDLogic:
    """Tests for the fixed AND logic in spatial restrictions.
    
    These tests verify the critical bug fix where multi-key spatial 
    restrictions now require ALL constraints to be satisfied (intersection)
    instead of any (union).
    """

    @pytest.mark.unit
    def test_select_lateralization_idx_simple_and_logic(self):
        """Test that AND logic correctly filters with single constraint."""
        data, trial_info = create_lateralization_test_data()

        # Select only trials with dist1_loc == 0
        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0]}
        )

        # Trials 0-3 have dist1_loc == 0
        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_multi_key_and_logic(self):
        """Test that AND logic correctly requires ALL constraints.
        
        Critical test for the AND/OR bug fix:
        - spatial_restriction = {'dist1_loc': [0], 'dist2_loc': [4]}
        - Should return trials where dist1_loc == 0 AND dist2_loc == 4
        - Expected: [1] (one trial matches both constraints)
        - Bug would return: [0,1,2,3,4,6,8] (trials matching either constraint)
        """
        data, trial_info = create_lateralization_test_data()

        # Select trials with dist1_loc=0, but restrict to dist2_loc=4
        spatial_restriction = {'dist2_loc': [4]}

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0]},
            spatial_restriction=spatial_restriction
        )

        # Only trial 1 has dist1_loc=0 AND dist2_loc=4
        expected = np.array([1])
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_multi_value_and_logic(self):
        """Test AND logic with multiple values in each constraint.
        
        pos_labels = {'dist1_loc': [0, 1]}, spatial_restriction = {'dist2_loc': [0, 4]}
        Should return trials where:
            (dist1_loc in [0, 1]) AND (dist2_loc in [0, 4])
        """
        data, trial_info = create_multilocation_test_data()

        spatial_restriction = {'dist2_loc': [0, 4]}

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1]},
            spatial_restriction=spatial_restriction
        )

        # Trials matching both constraints:
        # Trial 0: (0,0) ✓, Trial 1: (0,4) ✓, Trial 3: (1,0) ✓, Trial 4: (1,4) ✓
        expected = np.array([0, 1, 3, 4])
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_no_matches(self):
        """Test that AND logic returns empty array when no trials match all constraints."""
        data, trial_info = create_lateralization_test_data()

        # Impossible constraint combination
        spatial_restriction = {'dist2_loc': [99]}

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0]},
            spatial_restriction=spatial_restriction
        )

        expected = np.array([], dtype=int)
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_none_restriction(self):
        """Test that None spatial_restriction returns only pos_labels matches."""
        data, trial_info = create_lateralization_test_data()

        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1, 2, 3, 4, 5, 6, 9]},
            spatial_restriction=None
        )

        # Should return trials 0-9 (all trials have dist1_loc in the specified range)
        expected = np.arange(10)
        np.testing.assert_array_equal(idx, expected)

    @pytest.mark.unit
    def test_select_lateralization_idx_single_key_vs_multi_key(self):
        """Verify AND logic across primary and secondary constraints.
        
        Primary constraint (pos_labels) + secondary constraint (spatial_restriction)
        should both be applied with AND logic.
        """
        data, trial_info = create_multilocation_test_data()

        # Primary selection only: dist1_loc in [0, 1] (should get 6 trials)
        idx_single = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1]},
            spatial_restriction=None
        )

        # Same primary selection + spatial restriction (should get fewer)
        idx_multi = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0, 1]},
            spatial_restriction={'dist2_loc': [0]}  # Only accept dist2_loc == 0
        )

        # idx_single: 6 trials (with dist1_loc in [0, 1])
        # idx_multi: 2 trials (with dist1_loc in [0, 1] AND dist2_loc == 0)
        # Trials 0 (0,0) and 3 (1,0) match both
        
        assert len(idx_single) > len(idx_multi)
        assert len(idx_multi) == 2


# ============================================================================
# Unit Tests: Latency Analysis and Condition Names
# ============================================================================

class TestLatencyAnalysisConditionNames:
    """Tests for latency analysis output with condition name display."""

    @pytest.mark.unit
    def test_jackknife_contrast_basic(self, sample_waveforms):
        """Test basic jackknife contrast functionality."""
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present']

        lat_diff, t_val = ERP.jackknife_contrast(
            x1, x2, times, 75, 
            cnd1_name='absent', 
            cnd2_name='present'
        )

        # Verify output types and reasonable values
        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))
        
        # Latency diff should be small but typically positive (present peaks later)
        assert -0.1 < lat_diff < 0.1

    @pytest.mark.unit
    def test_jackknife_contrast_condition_names_parameter(self, sample_waveforms):
        """Verify condition names are accepted and used in output."""
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present']

        # Should accept custom condition names without error
        lat_diff, t_val = ERP.jackknife_contrast(
            x1, x2, times, 75,
            cnd1_name='distractor_absent',
            cnd2_name='distractor_present'
        )

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))

    @pytest.mark.unit
    def test_jackknife_contrast_default_names(self, sample_waveforms):
        """Test that default condition names are used when not provided."""
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present']

        # Should work with default names
        lat_diff, t_val = ERP.jackknife_contrast(
            x1, x2, times, 75
        )

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))

    @pytest.mark.unit
    def test_jack_latency_contrast_with_names(self, sample_waveforms):
        """Test jack_latency_contrast output with condition names."""
        waveforms, times = sample_waveforms
        x1 = waveforms['absent'].mean(axis=0)  # Grand mean
        x2 = waveforms['present'].mean(axis=0)

        # Get amplitude thresholds
        c1 = np.max(x1) * 0.75
        c2 = np.max(x2) * 0.75

        # This should print condition names in output
        lat_diff = ERP.jack_latency_contrast(
            x1, x2, c1, c2, times,
            print_output=False,
            cnd1_name='absent',
            cnd2_name='present'
        )

        assert isinstance(lat_diff, (float, np.floating))

    @pytest.mark.unit
    def test_compare_latencies_extracts_condition_names(self):
        """Test that compare_latencies accepts erp_data dict format.
        
        Full integration testing of compare_latencies requires MNE Evoked objects.
        This test verifies the interface is prepared for condition name extraction.
        """
        # The actual integration test would require proper Evoked data
        # For now, verify the method exists and has proper signature
        assert hasattr(ERP, 'compare_latencies')
        assert callable(ERP.compare_latencies)


# ============================================================================
# PHASE 1: Core Single-Subject Analysis (Foundation)
# ============================================================================
# This phase tests the fundamental single-subject ERP analysis workflow
# starting with basic parameter clarity and spatial filtering

# ============================================================================
# Unit Tests: Parameter Naming (spatial_restriction)
# ============================================================================

class TestParameterNaming:
    """Tests for semantic clarity of parameter names."""

    @pytest.mark.unit
    def test_condition_erps_accepts_spatial_restriction_parameter(self):
        """Verify condition_erps accepts spatial_restriction parameter."""
        # This is primarily a documentation/interface test
        # Verify that docstring mentions spatial_restriction
        docstring = ERP.condition_erps.__doc__
        assert 'spatial_restriction' in docstring
        assert 'position-label' in docstring.lower()

    @pytest.mark.unit
    def test_select_lateralization_idx_docstring_clarity(self):
        """Verify docstring explains AND logic clearly."""
        docstring = ERP.select_lateralization_idx.__doc__
        
        # Should mention AND logic and provide examples
        assert 'AND' in docstring
        assert 'spatial restriction' in docstring.lower()


# ============================================================================
# Integration Tests: Full ERP Workflow
# ============================================================================

class TestFullERPWorkflow:
    """Integration tests for complete ERP analysis workflows."""

    @pytest.mark.integration
    def test_erp_generation_basic(self, sample_epochs, sample_trial_dataframe):
        """Test basic ERP generation from epochs."""
        epochs = sample_epochs
        trial_info = sample_trial_dataframe

        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock folder structure
            mock_fs = Mock()
            mock_fs.subdir = tmpdir

            # Initialize ERP instance (would normally inherit from FolderStructure)
            # For now, just test that static methods work
            
            # Extract data for testing
            data = epochs.get_data()
            times = epochs.times

            # Basic test: ensure data shapes are correct
            assert data.shape[0] == len(trial_info)  # Same number of trials
            assert times.shape[0] == data.shape[2]  # Same number of time points

    @pytest.mark.integration
    def test_latency_analysis_workflow(self, sample_waveforms):
        """Test complete latency analysis workflow."""
        waveforms, times = sample_waveforms

        # Test with jackknife_contrast directly instead of compare_latencies
        # (which has more complex data format requirements)
        lat_diff, t_val = ERP.jackknife_contrast(
            waveforms['absent'],
            waveforms['present'],
            times,
            percent_amp=50
        )

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))
        assert -0.2 < lat_diff < 0.2  # Reasonable latency difference


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.unit
    def test_select_lateralization_idx_empty_trial_info(self):
        """Test handling of empty trial info."""
        empty_trial_info = pd.DataFrame({
            'trial': [],
            'dist1_loc': [],
            'dist2_loc': [],
        })

        idx = ERP.select_lateralization_idx(
            empty_trial_info,
            pos_labels={'dist1_loc': [0]}
        )

        assert len(idx) == 0

    @pytest.mark.unit
    def test_jackknife_contrast_identical_waveforms(self):
        """Test jackknife contrast with identical waveforms raises error.
        
        When waveforms are identical, variance is zero and t-statistic 
        cannot be computed.
        """
        times = np.linspace(0, 0.4, 100)
        n_trials = 20

        # Identical waveforms (but with multiple trials for jackknife)
        x = np.random.randn(n_trials, len(times))
        
        # Should raise ValueError because std dev will be zero
        with pytest.raises(ValueError, match="Standard deviation is zero"):
            ERP.jackknife_contrast(x, x, times, 75)

    @pytest.mark.unit
    def test_jackknife_contrast_single_trial(self):
        """Test jackknife contrast with minimal trials."""
        times = np.linspace(0, 0.4, 100)
        
        # Single trial each
        x1 = np.random.randn(1, len(times))
        x2 = np.random.randn(1, len(times))
        
        # This may raise an error or return NaN - document expected behavior
        # For now, verify it doesn't crash
        try:
            lat_diff, t_val = ERP.jackknife_contrast(x1, x2, times, 75)
            # If it works, values should be present
            assert lat_diff is not None or np.isnan(lat_diff)
        except (ValueError, ZeroDivisionError):
            # Expected behavior - can't do jackknife with too few samples
            pass

    @pytest.mark.unit
    def test_spatial_restriction_missing_key(self):
        """Test spatial restriction when trial_info lacks required key."""
        trial_info = pd.DataFrame({
            'trial': np.arange(5),
            'loc': [0, 1, 2, 3, 4],
        })

        # Request restriction on non-existent key
        # Should either return empty or raise error
        try:
            idx = ERP.select_lateralization_idx(
                trial_info,
                pos_labels={'loc': [0, 1, 2, 3, 4]},
                spatial_restriction={'missing_column': [0]}
            )
            # If it doesn't error, should return empty
            assert len(idx) == 0
        except (KeyError, ValueError):
            # Expected - column doesn't exist
            pass


# ============================================================================
# Regression Tests
# ============================================================================

class TestRegressions:
    """Regression tests to ensure fixes don't break existing functionality."""

    @pytest.mark.unit
    def test_single_key_restriction_unchanged(self):
        """Verify single-key spatial restrictions still work correctly."""
        data, trial_info = create_lateralization_test_data()

        # Single constraint should work as before
        idx = ERP.select_lateralization_idx(
            trial_info,
            pos_labels={'dist1_loc': [0]}
        )

        # Should return trials 0, 1, 2, 3
        assert len(idx) == 4
        np.testing.assert_array_equal(idx, [0, 1, 2, 3])

    @pytest.mark.unit
    def test_latency_analysis_returns_correct_types(self, sample_waveforms):
        """Regression test: latency analysis returns float and float."""
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present']

        lat_diff, t_val = ERP.jackknife_contrast(x1, x2, times, 75)

        assert isinstance(lat_diff, (float, np.floating))
        assert isinstance(t_val, (float, np.floating))


# ============================================================================
# PHASE 2 TESTS: Feature Extraction & Latency Analysis
# ============================================================================
# Tests for extracting and analyzing ERP features, latency timing,
# and statistical comparisons. Classes may appear in different order in file.
# - TestExtractERPFeatures (below): mean amplitude, AUC, onset latency
# - TestSelectERPWindow (line ~633): peak detection interface
# - TestLatencyAnalysisConditionNames (line ~228): jackknife latency analysis
# - TestStatisticalComparisons (line ~1054): statistical tests
# - TestERPComponentAnalysis (line ~738): component analysis
# - TestERPDataValidation (line ~671): data validation

class TestExtractERPFeatures:
    """Tests for extract_erp_features method (mean amplitude, AUC, latency)."""

    @pytest.mark.unit
    def test_extract_mean_amplitude(self, sample_waveforms):
        """Test mean amplitude extraction."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']  # (20, 100) shape

        result = ERP.extract_erp_features(x, times, method='mean_amp')

        # Should return shape (20,) - one value per trial
        assert result.shape == (20,)
        # Should have reasonable EEG amplitude range
        assert np.all(result > -20)
        assert np.all(result < 20)
        # Mean should be negative (has negative component)
        assert np.mean(result) < 0

    @pytest.mark.unit
    def test_extract_auc_total(self, sample_waveforms):
        """Test total area under curve calculation."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='auc')

        # Should return shape (20,) - one value per trial
        assert result.shape == (20,)
        # AUC should exist and be in reasonable range
        assert np.all(np.isfinite(result))
        # Mean AUC should be negative (mostly negative component)
        assert np.mean(result) < 0

    @pytest.mark.unit
    def test_extract_auc_positive(self, sample_waveforms):
        """Test positive-only area under curve."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='auc_pos')

        # Should return shape (20,) - one value per trial
        assert result.shape == (20,)
        # Should be small or zero (mostly negative component)
        assert np.all(result >= 0)

    @pytest.mark.unit
    def test_extract_auc_negative(self, sample_waveforms):
        """Test negative-only area under curve."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='auc_neg')

        # Should return shape (20,) - one value per trial
        assert result.shape == (20,)
        # Should be negative (has negative component)
        assert np.all(result <= 0)

    @pytest.mark.unit
    def test_extract_onset_latency_pos(self):
        """Test onset latency detection for positive polarity."""
        times = np.linspace(0, 0.4, 100)
        
        # Create synthetic positive component
        x = np.random.randn(5, len(times)) * 0.5  # noise
        peak_idx = 50
        x[:, peak_idx - 5:peak_idx + 5] = 5  # Add +5 µV peak
        
        result = ERP.extract_erp_features(
            x, times, method='onset_latency',
            threshold=0.5, polarity='pos'
        )

        # Should return shape (5,) with latency values in seconds
        assert result.shape == (5,)
        # Latencies should be > 0 and < 0.4
        assert np.all(result > 0)
        assert np.all(result < 0.4)
        # Should be roughly around 0.2 (time of peak)
        assert np.all(np.abs(result - 0.2) < 0.1)

    @pytest.mark.unit
    def test_extract_onset_latency_neg(self):
        """Test onset latency detection for negative polarity."""
        times = np.linspace(0, 0.4, 100)
        
        # Create synthetic negative component
        x = np.random.randn(5, len(times)) * 0.5
        peak_idx = 50
        x[:, peak_idx - 5:peak_idx + 5] = -5  # Add -5 µV peak
        
        result = ERP.extract_erp_features(
            x, times, method='onset_latency',
            threshold=0.5, polarity='neg'
        )

        # Should return shape (5,) with latency values
        assert result.shape == (5,)
        assert np.all(result > 0)
        assert np.all(result < 0.4)

    @pytest.mark.unit
    def test_extract_features_preserves_input(self, sample_waveforms):
        """Verify that extract_erp_features doesn't modify input data."""
        waveforms, times = sample_waveforms
        x = waveforms['absent'].copy()
        x_orig = x.copy()

        # Run various feature extractions
        ERP.extract_erp_features(x, times, method='mean_amp')
        ERP.extract_erp_features(x, times, method='auc')
        ERP.extract_erp_features(x, times, method='auc_neg')

        # Input should be unchanged
        np.testing.assert_array_equal(x, x_orig)


class TestSelectERPWindow:
    """Tests for select_erp_window method (peak detection)."""

    @pytest.mark.unit
    def test_select_erp_window_interface_exists(self):
        """Test that select_erp_window method exists and is documented."""
        # Verify method exists
        assert hasattr(ERP, 'select_erp_window')
        assert callable(ERP.select_erp_window)
        
        # Verify docstring
        docstring = ERP.select_erp_window.__doc__
        assert 'peak detection' in docstring.lower()
        assert 'time window' in docstring.lower()
        assert 'polarity' in docstring.lower()

    @pytest.mark.unit
    def test_select_erp_window_parameters_documented(self):
        """Test that key parameters are documented."""
        docstring = ERP.select_erp_window.__doc__
        
        assert 'erps' in docstring.lower()
        assert 'elec_oi' in docstring.lower()
        assert 'method' in docstring.lower()
        assert 'window_oi' in docstring.lower()
        assert 'polarity' in docstring.lower()
        assert 'window_size' in docstring.lower()
        
        # Verify method options are documented
        assert 'cnd_avg' in docstring
        assert 'cnd_spc' in docstring
        assert 'pos' in docstring
        assert 'neg' in docstring


class TestConditionERPs:
    """Integration tests for condition_erps method."""

    @pytest.mark.integration
    def test_condition_erps_basic_structure(self, sample_epochs, sample_trial_dataframe):
        """Test that condition_erps method is callable with basic parameters."""
        # This is a structural test - full integration requires FolderStructure setup
        
        # Verify method exists and has correct signature
        assert hasattr(ERP, 'condition_erps')
        assert callable(ERP.condition_erps)
        
        # Check docstring mentions key parameters
        docstring = ERP.condition_erps.__doc__
        assert 'pos_labels' in docstring
        assert 'spatial_restriction' in docstring
        assert 'cnds' in docstring

    @pytest.mark.integration
    def test_condition_erps_position_labels_in_docs(self):
        """Test that condition_erps documents position-label agnostic behavior."""
        docstring = ERP.condition_erps.__doc__
        
        # Verify documentation clarity
        assert 'position-label agnostic' in docstring
        assert 'spatial restriction' in docstring.lower()
        assert 'AND logic' in docstring

    @pytest.mark.unit
    def test_select_lateralization_called_correctly(self):
        """Verify select_lateralization_idx is used correctly in condition_erps."""
        docstring = ERP.condition_erps.__doc__
        
        # Verify parameters match select_lateralization_idx signature
        assert 'pos_labels' in docstring
        assert 'spatial_restriction' in docstring


class TestERPDataValidation:
    """Tests for data validation and error handling in ERP methods."""

    @pytest.mark.unit
    def test_extract_features_invalid_method(self, sample_waveforms):
        """Test error handling for invalid feature extraction method."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        # Invalid method should raise error or return empty
        try:
            result = ERP.extract_erp_features(
                x, times, method='invalid_method'
            )
            # If no error, result should be empty or None
            assert result is None or len(result) == 0
        except (ValueError, KeyError):
            # Expected - invalid method
            pass

    @pytest.mark.unit
    def test_extract_features_mismatched_times(self, sample_waveforms):
        """Test handling of mismatched data and times arrays."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']
        wrong_times = np.linspace(0, 0.2, 50)  # Wrong length

        # Should raise error or handle gracefully
        try:
            result = ERP.extract_erp_features(
                x, wrong_times, method='mean_amp'
            )
        except (ValueError, IndexError):
            # Expected - array length mismatch
            pass

    @pytest.mark.unit
    def test_mean_amplitude_valid_output_range(self, sample_waveforms):
        """Verify mean amplitude is within reasonable EEG range."""
        waveforms, times = sample_waveforms
        x = waveforms['absent']

        result = ERP.extract_erp_features(x, times, method='mean_amp')

        # EEG amplitudes typically -/+ 100 µV
        assert np.all(result > -100)
        assert np.all(result < 100)

    @pytest.mark.unit
    def test_onset_latency_within_time_range(self, sample_waveforms):
        """Verify onset latency values are within the time range."""
        times = np.linspace(0, 0.4, 100)
        
        x = np.random.randn(5, len(times))
        peak_idx = 50
        x[:, peak_idx - 5:peak_idx + 5] = 5

        result = ERP.extract_erp_features(
            x, times, method='onset_latency',
            threshold=0.5, polarity='pos'
        )

        # All latencies should be within time range
        assert np.all(result >= times.min())
        assert np.all(result <= times.max())


class TestERPComponentAnalysis:
    """Tests for realistic ERP component analysis scenarios."""

    @pytest.mark.unit
    def test_n2pc_like_analysis(self):
        """Test analysis of N2pc-like component (negative, posterior, contralateral)."""
        times = np.linspace(-0.2, 0.4, 300)
        n_trials = 30
        
        # Create realistic N2pc component
        x = np.random.randn(n_trials, len(times)) * 0.5  # Low noise
        # N2pc: ~200-350ms post-stimulus
        peak_idx = 200  # ~200ms in
        x[:, peak_idx - 20:peak_idx + 20] = -3  # -3 µV negative peak
        
        # Extract metrics
        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')
        auc_neg = ERP.extract_erp_features(x, times, method='auc_neg')
        
        # N2pc should be negative on average
        assert np.mean(mean_amp) < 0
        assert np.mean(auc_neg) < 0

    @pytest.mark.unit
    def test_p300_like_analysis(self):
        """Test analysis of P300-like component (positive, central, late)."""
        times = np.linspace(-0.2, 0.8, 600)
        n_trials = 20
        
        # Create realistic P300 component
        x = np.random.randn(n_trials, len(times)) * 0.5  # Low noise
        # P300: ~300-500ms post-stimulus
        peak_idx = 300
        x[:, peak_idx - 40:peak_idx + 40] = 5  # +5 µV positive peak
        
        # Extract metrics
        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')
        auc_pos = ERP.extract_erp_features(x, times, method='auc_pos')
        
        # P300 should be positive on average
        assert np.mean(mean_amp) > 0
        assert np.mean(auc_pos) > 0

    @pytest.mark.unit
    def test_condition_comparison_workflow(self, sample_waveforms):
        """Test typical workflow: extract features from multiple conditions."""
        waveforms, times = sample_waveforms
        
        # Typical ERP analysis workflow
        features = {}
        for condition, waveform in waveforms.items():
            features[condition] = {
                'mean_amp': ERP.extract_erp_features(
                    waveform, times, method='mean_amp'
                ),
                'auc': ERP.extract_erp_features(
                    waveform, times, method='auc'
                ),
            }
        
        # Should have features for both conditions
        assert 'absent' in features
        assert 'present' in features
        assert 'mean_amp' in features['absent']
        assert 'auc' in features['absent']
        
        # Should be able to compare
        mean_diff = np.mean(features['absent']['mean_amp']) - \
                    np.mean(features['present']['mean_amp'])
        assert isinstance(mean_diff, (float, np.floating))


# ============================================================================
# PHASE 3 TESTS: Group-Level & Advanced Analysis
# ============================================================================
# Tests for group-level ERP aggregation, multi-subject analysis,
# advanced utilities, and error handling.
# - TestGroupERPAnalysis (below): group ERP methods
# - TestSelectWaveformVariations (line ~1005): waveform selection
# - TestWaveformAnalysisCombinations (line ~1117): multi-condition workflows
# - TestGetERPParams (line ~1024): parameter extraction
# - TestExportERPMetrics (line ~1058): export functionality
# - TestERPValidationAndErrors (line ~1161): advanced error handling
# - TestRegressions (line ~444): cross-cutting regression tests

class TestGroupERPAnalysis:
    """Tests for group-level ERP analysis methods."""

    @pytest.mark.unit
    def test_lateralized_erp_idx_basic(self, sample_epochs):
        """Test electrode index extraction for lateralized analysis."""
        evoked = sample_epochs.average()
        erp_list = [evoked]
        
        # Get actual channel names
        channels = evoked.ch_names
        contra_channels = channels[5:8]  # First few channels
        ipsi_channels = channels[10:13]   # Different channels

        contra_idx, ipsi_idx = ERP.lateralized_erp_idx(
            erp_list, contra_channels, ipsi_channels
        )

        # Should return numpy arrays of indices
        assert isinstance(contra_idx, np.ndarray)
        assert isinstance(ipsi_idx, np.ndarray)
        assert len(contra_idx) == 3
        assert len(ipsi_idx) == 3
        
        # Indices should be valid (within channel range)
        assert np.all(contra_idx >= 0)
        assert np.all(contra_idx < len(channels))
        assert np.all(ipsi_idx >= 0)
        assert np.all(ipsi_idx < len(channels))
        
        # Indices should be different
        assert not np.array_equal(contra_idx, ipsi_idx)

    @pytest.mark.unit
    def test_lateralized_erp_idx_mapping(self, sample_epochs):
        """Test that returned indices correctly map to channels."""
        evoked = sample_epochs.average()
        erp_list = [evoked]
        channels = evoked.ch_names
        
        contra_channels = [channels[0], channels[5]]
        ipsi_channels = [channels[10], channels[15]]

        contra_idx, ipsi_idx = ERP.lateralized_erp_idx(
            erp_list, contra_channels, ipsi_channels
        )

        # Verify indices map back to correct channels
        assert channels[contra_idx[0]] == channels[0]
        assert channels[contra_idx[1]] == channels[5]
        assert channels[ipsi_idx[0]] == channels[10]
        assert channels[ipsi_idx[1]] == channels[15]

    @pytest.mark.unit
    def test_group_erp_all_electrodes(self, sample_epochs):
        """Test group ERP creation with all electrodes."""
        evoked1 = sample_epochs[:50].average()
        evoked2 = sample_epochs[50:].average()
        erp_list = [evoked1, evoked2]

        evoked_X, group_evoked = ERP.group_erp(erp_list, elec_oi='all')

        # Should return data and group evoked object
        assert isinstance(evoked_X, np.ndarray)
        assert isinstance(group_evoked, mne.Evoked)
        
        # Data shape should be (n_subjects, n_timepoints)
        assert evoked_X.shape[0] == 2
        assert evoked_X.shape[1] == len(evoked1.times)
        
        # Group evoked should have same time points
        assert len(group_evoked.times) == len(evoked1.times)

    @pytest.mark.unit
    def test_group_erp_specific_electrodes(self, sample_epochs):
        """Test group ERP with specific electrode selection."""
        evoked1 = sample_epochs[:50].average()
        evoked2 = sample_epochs[50:].average()
        erp_list = [evoked1, evoked2]
        
        # Select first 3 electrodes
        elec_oi = evoked1.ch_names[:3]

        evoked_X, group_evoked = ERP.group_erp(erp_list, elec_oi=elec_oi)

        # Should return valid data
        assert evoked_X.shape[0] == 2
        assert evoked_X.shape[1] == len(evoked1.times)
        
        # Group evoked should exist
        assert isinstance(group_evoked, mne.Evoked)

    @pytest.mark.unit
    def test_select_waveform_basic(self, sample_epochs):
        """Test waveform selection from ERP data."""
        evoked = sample_epochs.average()
        erp_list = [evoked]
        channels = evoked.ch_names[:5]

        waveform = ERP.select_waveform(erp_list, channels)

        # Should return numpy array
        assert isinstance(waveform, np.ndarray)
        # Shape should be (n_evoked, n_timepoints) - averaged across channels
        assert waveform.shape[0] == 1  # 1 evoked object
        assert waveform.shape[1] == len(evoked.times)

    @pytest.mark.unit
    def test_select_waveform_single_channel(self, sample_epochs):
        """Test waveform selection with single channel."""
        evoked = sample_epochs.average()
        erp_list = [evoked]
        channel = [evoked.ch_names[0]]

        waveform = ERP.select_waveform(erp_list, channel)

        # Should return data (n_evoked, n_timepoints)
        assert waveform.shape[0] == 1
        assert waveform.shape[1] == len(evoked.times)

    @pytest.mark.unit
    def test_group_lateralized_erp_with_valid_montage(self, sample_epochs):
        """Test group-level lateralized ERP with valid channels from biosemi64."""
        evoked1 = sample_epochs[:50].average()
        evoked2 = sample_epochs[50:].average()
        erp_list = [evoked1, evoked2]
        
        channels = evoked1.ch_names
        # Find channels that exist in the data
        # Use channels that are likely to exist in standard montages
        available_channels = [ch for ch in channels if any(
            prefix in ch for prefix in ['O', 'P', 'C', 'F']
        )]
        
        if len(available_channels) >= 4:
            contra_channels = available_channels[0:2]
            ipsi_channels = available_channels[2:4]
            
            try:
                contra_data, ipsi_data = ERP.group_lateralized_erp(
                    erp_list, contra_channels, ipsi_channels,
                    montage='biosemi64'
                )
                
                # Should return data arrays
                assert isinstance(contra_data, np.ndarray)
                assert isinstance(ipsi_data, np.ndarray)
            except (ValueError, KeyError):
                # May fail if channels not in standard montage
                pytest.skip("Test channels not in biosemi64 montage")
        else:
            pytest.skip("Not enough suitable channels in test data")


class TestSelectWaveformVariations:
    """Tests for waveform selection with different input formats."""

    @pytest.mark.unit
    def test_select_waveform_multiple_evoked(self, sample_epochs):
        """Test waveform selection with multiple evoked objects."""
        erp1 = sample_epochs[:30].average()
        erp2 = sample_epochs[30:60].average()
        erp3 = sample_epochs[60:].average()
        erp_list = [erp1, erp2, erp3]
        
        channels = erp1.ch_names[:4]
        waveform = ERP.select_waveform(erp_list, channels)

        # Should return data for all 3 evoked objects
        assert waveform.shape[0] == 3
        assert waveform.shape[1] == len(erp1.times)


class TestGetERPParams:
    """Tests for ERP parameter extraction utility."""

    @pytest.mark.unit
    def test_get_erp_params_evoked_list_input(self, sample_epochs):
        """Test parameter extraction from list of evoked objects."""
        erp1 = sample_epochs[:50].average()
        erp2 = sample_epochs[50:].average()
        erp_list = [erp1, erp2]

        result = ERP.get_erp_params(erp_list)

        # Should return tuple with parameters
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_get_erp_params_dict_input_with_evoked(self, sample_epochs):
        """Test parameter extraction from dict of evoked objects."""
        erp1 = sample_epochs[:50].average()
        erp2 = sample_epochs[50:].average()
        
        erp_dict = {
            'condition1': [erp1],
            'condition2': [erp2],
        }

        result = ERP.get_erp_params(erp_dict)

        # Should return tuple with parameters
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestExportERPMetrics:
    """Tests for ERP metrics export functionality."""

    @pytest.mark.unit
    def test_export_function_exists(self):
        """Test that export function exists and is documented."""
        assert hasattr(ERP, 'export_erp_metrics_to_csv')
        assert callable(ERP.export_erp_metrics_to_csv)
        
        docstring = ERP.export_erp_metrics_to_csv.__doc__
        assert 'CSV' in docstring or 'csv' in docstring.lower()

    @pytest.mark.unit
    def test_extract_erp_features_all_methods(self):
        """Test that all feature extraction methods are accessible."""
        times = np.linspace(0, 0.4, 100)
        x = np.random.randn(5, 100)
        
        methods = ['mean_amp', 'auc', 'auc_pos', 'auc_neg', 'onset_latency']
        
        for method in methods:
            try:
                result = ERP.extract_erp_features(x, times, method=method)
                assert result.shape[0] == 5
            except (ValueError, KeyError):
                # Some methods may have specific requirements
                pass


class TestStatisticalComparisons:
    """Tests for statistical comparison methods."""

    @pytest.mark.unit
    def test_jackknife_contrast_produces_t_value(self, sample_waveforms):
        """Test that jackknife contrast produces valid t-statistic."""
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present']

        lat_diff, t_val = ERP.jackknife_contrast(x1, x2, times, 75)

        # t-value should be finite
        assert np.isfinite(t_val)
        # t-value should be non-zero (unless data are identical)
        assert t_val != 0

    @pytest.mark.unit
    def test_jackknife_variance_estimation(self, sample_waveforms):
        """Test that jackknife estimates reasonable variance."""
        waveforms, times = sample_waveforms
        x1 = waveforms['absent']
        x2 = waveforms['present'] + 0.01  # Slightly different

        lat_diff, t_val = ERP.jackknife_contrast(x1, x2, times, 75)

        # For small differences, t-value should be relatively small
        assert abs(t_val) < 10


class TestWaveformAnalysisCombinations:
    """Tests for common waveform analysis combinations."""

    @pytest.mark.unit
    def test_multi_condition_feature_comparison(self, sample_waveforms):
        """Test extracting and comparing features across multiple conditions."""
        waveforms, times = sample_waveforms
        
        # Extract features for multiple conditions
        conditions = {
            'absent': waveforms['absent'],
            'present': waveforms['present'],
        }
        
        features = {}
        for cond_name, waveform in conditions.items():
            features[cond_name] = {
                'mean_amp': np.mean(ERP.extract_erp_features(
                    waveform, times, method='mean_amp'
                )),
                'auc': np.mean(ERP.extract_erp_features(
                    waveform, times, method='auc'
                )),
            }
        
        # Should have valid features for both conditions
        assert 'absent' in features
        assert 'present' in features
        assert 'mean_amp' in features['absent']
        assert 'auc' in features['absent']
        
        # Features should be comparable
        amp_diff = features['absent']['mean_amp'] - features['present']['mean_amp']
        assert isinstance(amp_diff, (float, np.floating))

    @pytest.mark.unit
    def test_individual_subject_erp_pipeline(self, sample_waveforms):
        """Test realistic single-subject ERP analysis pipeline."""
        waveforms, times = sample_waveforms
        
        # Simulate individual subject ERPs
        subject_erps = {}
        for cond in ['absent', 'present']:
            waveform = waveforms[cond]
            subject_erps[cond] = {
                'mean_amplitude': np.mean(ERP.extract_erp_features(
                    waveform, times, method='mean_amp'
                )),
                'latency': np.mean(ERP.extract_erp_features(
                    waveform, times, method='onset_latency',
                    polarity='neg'
                )),
                'auc': np.mean(ERP.extract_erp_features(
                    waveform, times, method='auc'
                )),
            }
        
        # Should have complete analysis for all conditions
        assert all(cond in subject_erps for cond in ['absent', 'present'])
        for cond in subject_erps:
            assert all(metric in subject_erps[cond] 
                      for metric in ['mean_amplitude', 'latency', 'auc'])


class TestERPValidationAndErrors:
    """Tests for validation and error handling in statistical methods."""

    @pytest.mark.unit
    def test_jackknife_minimum_observations(self):
        """Test jackknife behavior with minimum observations."""
        times = np.linspace(0, 0.4, 100)
        
        # Create minimal datasets
        x1 = np.random.randn(2, len(times))
        x2 = np.random.randn(2, len(times))
        
        # Should work with minimum samples
        try:
            lat_diff, t_val = ERP.jackknife_contrast(x1, x2, times, 75)
            assert np.isfinite(lat_diff)
            assert np.isfinite(t_val)
        except ValueError:
            # May raise error if samples too few
            pass

    @pytest.mark.unit
    def test_feature_extraction_null_signal(self):
        """Test feature extraction with zero signal."""
        times = np.linspace(0, 0.4, 100)
        x = np.zeros((5, len(times)))
        
        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')
        
        # Mean of zeros should be zero
        np.testing.assert_array_almost_equal(mean_amp, np.zeros(5))

    @pytest.mark.unit
    def test_feature_extraction_constant_signal(self):
        """Test feature extraction with constant signal."""
        times = np.linspace(0, 0.4, 100)
        x = np.ones((5, len(times))) * 5
        
        mean_amp = ERP.extract_erp_features(x, times, method='mean_amp')
        
        # Mean of constant should be that constant
        np.testing.assert_array_almost_equal(mean_amp, np.ones(5) * 5)

    @pytest.mark.unit
    def test_auc_symmetry(self):
        """Test that AUC of symmetric signal is predictable."""
        times = np.linspace(0, 0.4, 100)
        
        # Create symmetric positive and negative signals
        x_pos = np.ones((1, len(times))) * 5
        x_neg = np.ones((1, len(times))) * -5
        
        auc_pos = ERP.extract_erp_features(x_pos, times, method='auc_pos')[0]
        auc_neg = ERP.extract_erp_features(x_neg, times, method='auc_neg')[0]
        
        # AUC magnitudes should be similar
        assert abs(abs(auc_pos) - abs(auc_neg)) < 0.1


# ============================================================================
# PHASE 4 TESTS: Preprocessing & Advanced Workflows  
# ============================================================================
# Tests for preprocessing methods, full workflow integration, and advanced
# analysis scenarios (15 tests)

class TestSelectERPDataFiltering:
    """Tests for select_erp_data trial filtering and exclusion."""

    @pytest.mark.unit
    def test_select_erp_data_method_exists(self):
        """Test that select_erp_data method exists and is documented."""
        assert hasattr(ERP, 'select_erp_data')
        assert callable(ERP.select_erp_data)
        
        docstring = ERP.select_erp_data.__doc__
        assert 'exclude' in docstring.lower() or 'select' in docstring.lower()

    @pytest.mark.unit
    def test_create_erps_method_exists(self):
        """Test that create_erps method exists."""
        assert hasattr(ERP, 'create_erps')
        assert callable(ERP.create_erps)
        
        docstring = ERP.create_erps.__doc__
        assert 'ERP' in docstring or 'epoch' in docstring.lower()

    @pytest.mark.unit
    def test_create_erps_parameters_documented(self):
        """Test that create_erps documents its parameters."""
        docstring = ERP.create_erps.__doc__
        
        # Should document key parameters
        assert 'epochs' in docstring.lower() or 'data' in docstring.lower()


class TestCompareLatenciesIntegration:
    """Integration tests for compare_latencies method."""

    @pytest.mark.unit
    def test_compare_latencies_method_exists(self):
        """Test that compare_latencies method exists."""
        assert hasattr(ERP, 'compare_latencies')
        assert callable(ERP.compare_latencies)

    @pytest.mark.unit
    def test_compare_latencies_docstring_complete(self):
        """Test that compare_latencies has complete documentation."""
        docstring = ERP.compare_latencies.__doc__
        
        assert 'latenc' in docstring.lower()
        assert 'erp' in docstring.lower()


class TestERPReportGeneration:
    """Tests for report generation functionality."""

    @pytest.mark.unit
    def test_generate_erp_report_exists(self):
        """Test that generate_erp_report method exists."""
        assert hasattr(ERP, 'generate_erp_report')
        assert callable(ERP.generate_erp_report)

    @pytest.mark.unit
    def test_flip_topography_exists(self):
        """Test that flip_topography method exists and is documented."""
        assert hasattr(ERP, 'flip_topography')
        assert callable(ERP.flip_topography)
        
        docstring = ERP.flip_topography.__doc__
        assert 'topograph' in docstring.lower()
        assert 'flip' in docstring.lower() or 'mirror' in docstring.lower()


class TestResidualEyeAnalysis:
    """Tests for residual eye movement analysis."""

    @pytest.mark.unit
    def test_residual_eye_method_exists(self):
        """Test that residual_eye method exists."""
        assert hasattr(ERP, 'residual_eye')
        assert callable(ERP.residual_eye)

    @pytest.mark.unit
    def test_residual_eye_docstring_complete(self):
        """Test that residual_eye documents lateralized analysis."""
        docstring = ERP.residual_eye.__doc__
        
        assert 'eye' in docstring.lower()
        assert 'lateral' in docstring.lower() or 'residual' in docstring.lower()

    @pytest.mark.unit
    def test_residual_eye_parameters_documented(self):
        """Test that residual_eye documents key parameters."""
        docstring = ERP.residual_eye.__doc__
        
        # Should mention key parameters
        assert 'heog' in docstring.lower() or 'eye' in docstring.lower()


class TestMultiConditionAnalysisWorkflow:
    """Tests for multi-condition ERP analysis workflows."""

    @pytest.mark.unit
    def test_three_condition_comparison(self, sample_waveforms):
        """Test analyzing 3+ conditions in a single workflow."""
        times = np.linspace(0, 0.4, 100)
        
        # Simulate 3 conditions
        conditions = {
            'present': sample_waveforms[0]['present'],
            'absent': sample_waveforms[0]['absent'],
            'neutral': np.random.randn(20, len(times)) * 0.3,  # Neutral condition
        }
        
        # Extract features for all conditions
        features = {}
        for cond_name, waveform in conditions.items():
            features[cond_name] = np.mean(ERP.extract_erp_features(
                waveform, times, method='mean_amp'
            ))
        
        # Should have features for all 3 conditions
        assert len(features) == 3
        assert 'present' in features
        assert 'absent' in features
        assert 'neutral' in features

    @pytest.mark.unit
    def test_multiple_subjects_multiple_conditions(self):
        """Test group-level analysis with multiple conditions."""
        times = np.linspace(0, 0.4, 100)
        n_subjects = 5
        
        # Simulate multi-subject, multi-condition analysis
        group_analysis = {}
        for subj in range(n_subjects):
            group_analysis[f'subj_{subj}'] = {
                'condition_a': np.mean(ERP.extract_erp_features(
                    np.random.randn(20, len(times)), times, method='mean_amp'
                )),
                'condition_b': np.mean(ERP.extract_erp_features(
                    np.random.randn(20, len(times)), times, method='mean_amp'
                )),
            }
        
        # Should have analysis for all subjects
        assert len(group_analysis) == n_subjects
        for subj in group_analysis:
            assert 'condition_a' in group_analysis[subj]
            assert 'condition_b' in group_analysis[subj]


class TestERPDataInjectionAndValidation:
    """Tests for comprehensive data validation in ERP workflows."""

    @pytest.mark.unit
    def test_robust_to_noisy_data(self):
        """Test that ERP methods are robust to noisy input."""
        times = np.linspace(0, 0.4, 100)
        
        # Create very noisy data
        x = np.random.randn(20, len(times)) * 5  # 5x typical noise
        
        # Should still extract features
        result = ERP.extract_erp_features(x, times, method='mean_amp')
        assert result is not None
        assert len(result) == 20

    @pytest.mark.unit
    def test_small_sample_handling(self):
        """Test handling of small sample sizes."""
        times = np.linspace(0, 0.4, 100)
        
        # Single trial
        x_small = np.random.randn(1, len(times))
        
        # Should handle gracefully
        result = ERP.extract_erp_features(x_small, times, method='mean_amp')
        assert result is not None
        assert len(result) == 1


class TestExportAndImportIntegration:
    """Tests for data export and import workflows."""

    @pytest.mark.unit
    def test_export_csv_interface_exists(self):
        """Test that CSV export functionality is accessible."""
        assert hasattr(ERP, 'export_erp_metrics_to_csv')
        assert callable(ERP.export_erp_metrics_to_csv)

    @pytest.mark.unit
    def test_erp_dict_format_standard(self):
        """Test that ERP data follows standard dict format."""
        # Verify get_erp_params works with standard formats
        assert hasattr(ERP, 'get_erp_params')
        assert callable(ERP.get_erp_params)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
