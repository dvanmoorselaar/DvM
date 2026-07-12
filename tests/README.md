# Tests for open_dvm

Comprehensive test suite for the open_dvm package using pytest.

## Test Organization: Workflow-Based Phases

Tests are organized by user workflow from single-subject analysis to group-level comparisons:

### **PHASE 1: Core Single-Subject Analysis** (21 tests)
Foundational tests for the basic ERP analysis workflow:
- **Parameter Naming**: Semantic clarity of API parameters (`spatial_restriction`)
- **Spatial Filtering**: AND logic bug fix for multi-constraint filtering
- **Condition ERP Generation**: Creating averaged ERPs per condition
- **Workflows**: Basic end-to-end analysis pipelines  
- **Edge Cases**: Handling empty data, single trials, missing columns
- **Regressions**: Ensuring bug fixes remain fixed

*Tests:* TestParameterNaming, TestSpatialRestrictionANDLogic, TestConditionERPs, TestFullERPWorkflow, TestEdgeCases (subset), TestRegressions

### **PHASE 2: Feature Extraction & Latency Analysis** (19 tests)
Core analysis methods for extracting ERP features and comparing conditions:
- **Feature Extraction**: Mean amplitude, area-under-curve (AUC), onset latency
- **Peak Detection**: Identifying ERP component windows
- **Latency Analysis**: Jackknife-based latency comparisons with condition names
- **Statistics**: Significance testing and variance estimation
- **Component Analysis**: N2pc, P300, and lateralized components
- **Validation**: Data type and range checking

*Tests:* TestExtractERPFeatures, TestSelectERPWindow, TestLatencyAnalysisConditionNames, TestStatisticalComparisons, TestERPComponentAnalysis, TestERPDataValidation

### **PHASE 3: Group-Level & Advanced Analysis** (19 tests)
Advanced methods for multi-subject analysis and utilities:
- **Group Aggregation**: Combining individual ERPs across subjects
- **Lateralized Analysis**: Contralateral-ipsilateral difference waves
- **Waveform Selection**: Extracting and averaging electrode subsets
- **Utility Functions**: Parameter extraction, data export
- **Advanced Validation**: Error handling for edge cases in group analysis

*Tests:* TestGroupERPAnalysis, TestSelectWaveformVariations, TestWaveformAnalysisCombinations, TestGetERPParams, TestExportERPMetrics, TestERPValidationAndErrors

## File Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── pytest.ini                  # Pytest settings (markers, output options)
├── README.md                   # This file (testing guide)
├── fixtures/                   # Sample data generators
│   ├── __init__.py
│   └── sample_data.py          # Synthetic EEG/ERP data generators
├── test_analysis/              # ERP/TFR analysis tests
│   ├── __init__.py
│   └── test_erp.py            # 60 tests across Phase 1, 2, 3
├── test_support/               # Support utilities (planned)
├── test_stats/                 # Statistical utilities (planned)
└── test_visualization/         # Plotting/visualization (planned)
```

## Test Categories by Type

### Unit Tests (`@pytest.mark.unit`)
Fast, isolated tests of individual functions. Expected to run in < 100ms each.

**Examples:**
- Spatial restriction AND logic verification
- Parameter name handling
- Output format correctness

### Integration Tests (`@pytest.mark.integration`)
Tests that verify multiple components work together correctly. May be slower.

**Examples:**
- Full ERP workflow from epochs to statistics
- Latency analysis pipeline
- Multi-step condition comparisons

### Slow Tests (`@pytest.mark.slow`)
Long-running tests (marked for optional skipping).

## Running Tests

### Install test dependencies:
```bash
pip install "open-dvm[dev]"
```

### Run all tests:
```bash
pytest
```

### Run only unit tests (fast):
```bash
pytest -m unit
```

### Run only integration tests:
```bash
pytest -m integration
```

### Skip slow tests:
```bash
pytest -m "not slow"
```

### Run with coverage report:
```bash
pytest --cov=open_dvm --cov-report=html
```

This generates `htmlcov/index.html` with coverage details.

### Run specific test file:
```bash
pytest tests/test_analysis/test_erp.py
```

### Run specific test class:
```bash
pytest tests/test_analysis/test_erp.py::TestSpatialRestrictionANDLogic
```

### Run specific test:
```bash
pytest tests/test_analysis/test_erp.py::TestSpatialRestrictionANDLogic::test_select_lateralization_idx_multi_key_and_logic
```

### Verbose output:
```bash
pytest -v
```

### Show print statements:
```bash
pytest -s
```

## Test Fixtures

Fixtures are defined in `conftest.py` and available to all tests:

- `sample_epochs` - Synthetic EEG epochs (100 trials, 32 channels)
- `sample_epochs_data` - Tuple of (epochs, raw_data)
- `sample_trial_dataframe` - Trial metadata with conditions and locations
- `sample_waveforms` - Synthetic N2pc-like waveforms for latency testing
- `lateralization_test_data` - Data designed for testing AND logic (10 trials)
- `multilocation_test_data` - Multi-constraint AND logic test data (12 trials)
- `temp_output_dir` - Temporary directory for test outputs
- `sample_times` - Standard time vector (500 Hz sampling)

### Usage Example:
```python
def test_something(sample_epochs, sample_trial_dataframe):
    """Access fixtures as function parameters."""
    epochs = sample_epochs
    trial_info = sample_trial_dataframe
    # ... test code ...
```

## Key Tests

### Spatial Restriction AND Logic (Critical Bug Fix)
**File:** `tests/test_analysis/test_erp.py::TestSpatialRestrictionANDLogic`

Tests verify that multi-key spatial restrictions use AND (intersection) logic:
- `{'dist1_loc': [0], 'dist2_loc': [4]}` selects trials where BOTH constraints are true
- NOT trials matching either constraint (union/OR logic)

**Critical test case:**
```python
def test_select_lateralization_idx_multi_key_and_logic(self):
    """Test that AND logic correctly requires ALL constraints."""
    spatial_restriction = {'dist1_loc': [0], 'dist2_loc': [4]}
    idx = ERP.select_lateralization_idx(trial_info, spatial_restriction)
    # Should return [1] (one trial matches both)
    # BUG would return [0,1,2,3,4,6,8] (matches either constraint)
```

### Latency Analysis with Condition Names
**File:** `tests/test_analysis/test_erp.py::TestLatencyAnalysisConditionNames`

Tests verify:
- Condition names are properly extracted from erp_data dict keys
- Jackknife contrast correctly displays condition names in output
- Default names work when not provided

### Semantic Clarity (Parameter Naming)
**File:** `tests/test_analysis/test_erp.py::TestParameterNaming`

Verifies `spatial_restriction` parameter replaces `midline`:
- Docstrings mention spatial_restriction
- Documentation explains position-label agnostic nature

## Adding New Tests

When adding new tests:

1. **Name tests clearly:** `test_what_should_happen()` or `test_error_when_invalid_input()`

2. **Mark appropriately:**
   ```python
   @pytest.mark.unit
   def test_something():
       """Brief description."""
       pass
   ```

3. **Use fixtures:**
   ```python
   def test_something(sample_epochs):
       data = sample_epochs.get_data()
       assert data.shape[0] > 0
   ```

4. **Test edge cases and errors:**
   ```python
   with pytest.raises(ValueError):
       invalid_function(bad_input)
   ```

5. **Add docstrings explaining the test:**
   ```python
   def test_multi_key_and_logic(self):
       """Test that AND logic correctly requires ALL constraints.
       
       Regression test for bug where {'a': [1], 'b': [2]} used OR instead of AND.
       """
   ```

## Continuous Integration

Tests can be run automatically on GitHub using Actions. Example workflow file would go in:
```
.github/workflows/tests.yml
```

This can test across Python versions (3.8, 3.9, 3.10, 3.11) and upload coverage reports.

## Troubleshooting

### Import errors:
```bash
# Ensure open_dvm is in PYTHONPATH or installed
pip install -e .
```

### Fixture not found:
- Check conftest.py is in correct location
- Ensure fixture name matches function parameter name

### Tests pass locally but fail in CI:
- Check Python version differences
- Verify all dependencies are specified
- Check for platform-specific issues (Windows vs Mac vs Linux)

## Coverage Goals

- **Target:** 80%+ coverage of core analysis functions
- **Focus:** Logic-critical paths, especially the AND/OR fix
- **OK to skip:** Visualization code, I/O operations, debug utilities

View coverage:
```bash
pytest --cov=open_dvm --cov-report=term-missing
```

## Next Steps

After ERP tests are solid, add comprehensive tests for:
1. `test_analysis/test_tfr.py` - Time-frequency analysis
2. `test_analysis/test_eye.py` - Eye-tracking analysis
3. `test_support/test_preprocessing.py` - Preprocessing utilities
4. `test_stats/test_stats_utils.py` - Statistical functions
5. `test_visualization/test_plot_utils.py` - Plotting utilities
