"""
Otava Test Data - Test data generators for Apache Otava change point detection.

This package provides generators for creating synthetic time series data
with known change points for testing and benchmarking change point detection
algorithms.
"""

__version__ = "0.1.0"

from otava_test_data.generators.basic import (
    constant,
    noise_normal,
    noise_uniform,
    outlier,
    step_function,
    regression_fix,
)
from otava_test_data.generators.advanced import (
    banding,
    variance_change,
    phase_change,
    multiple_changes,
)
from otava_test_data.generators.combiner import combine, TimeSeries

__all__ = [
    "constant",
    "noise_normal",
    "noise_uniform",
    "outlier",
    "step_function",
    "regression_fix",
    "banding",
    "variance_change",
    "phase_change",
    "multiple_changes",
    "combine",
    "TimeSeries",
]
