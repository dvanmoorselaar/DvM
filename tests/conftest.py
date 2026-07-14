"""
Pytest configuration and shared fixtures for open_dvm tests.

This file defines pytest fixtures and configuration used across all tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from tests.fixtures.sample_data import (
    create_sample_epochs,
    create_sample_erp_dataframe,
    create_sample_waveforms,
    create_lateralization_test_data,
    create_multilocation_test_data,
    create_lateralized_flip_epochs,
    create_peaked_erps,
    create_biosemi64_evoked_pair,
    create_residual_eye_data,
)


@pytest.fixture
def sample_epochs():
    """Fixture providing sample EEG epochs for testing."""
    epochs, data = create_sample_epochs(n_trials=100, n_channels=32)
    return epochs


@pytest.fixture
def sample_epochs_data():
    """Fixture providing both epochs object and raw data."""
    epochs, data = create_sample_epochs(n_trials=100, n_channels=32)
    return epochs, data


@pytest.fixture
def sample_trial_dataframe():
    """Fixture providing sample trial metadata."""
    return create_sample_erp_dataframe(n_trials=100)


@pytest.fixture
def sample_waveforms():
    """Fixture providing synthetic N2pc-like waveforms for latency testing."""
    waveforms, times = create_sample_waveforms()
    return waveforms, times


@pytest.fixture
def lateralization_test_data():
    """Fixture for testing spatial restriction AND logic."""
    data, trial_info = create_lateralization_test_data()
    return data, trial_info


@pytest.fixture
def multilocation_test_data():
    """Fixture for testing multi-constraint AND logic."""
    data, trial_info = create_multilocation_test_data()
    return data, trial_info


@pytest.fixture
def lateralized_flip_epochs():
    """Fixture providing deterministic epochs for flip_topography tests."""
    epochs, trial_info = create_lateralized_flip_epochs()
    return epochs, trial_info


@pytest.fixture
def peaked_erps():
    """Fixture providing two conditions with known peak locations."""
    return create_peaked_erps()


@pytest.fixture
def biosemi64_evoked_pair():
    """Fixture providing evoked data on a real biosemi64 montage."""
    return create_biosemi64_evoked_pair()


@pytest.fixture
def residual_eye_data():
    """Fixture providing epochs/trial_info/expected value for residual_eye."""
    return create_residual_eye_data()


@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_times():
    """Fixture providing standard time vector for 500 Hz sampling."""
    return np.linspace(-0.2, 0.4, 300)  # -200 to 400 ms at 500 Hz


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
