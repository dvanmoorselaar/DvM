"""
Statistical analysis utilities for EEG data.

Functions
---------
bootstrap_SE : Compute bootstrap standard errors
confidence_int : Computer confidence intervals
paired_t : Paired t-tests
connected_adjacency : Build adjacency matrices for cluster-based statistics
perform_stats : General statistical testing framework
"""

from open_dvm.stats.stats_utils import (
    bootstrap_SE,
    confidence_int,
    paired_t,
    connected_adjacency,
    perform_stats,
)

__all__ = [
    'bootstrap_SE',
    'confidence_int',
    'paired_t',
    'connected_adjacency',
    'perform_stats',
]
