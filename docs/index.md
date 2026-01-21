# Otava Test Data

Test data generators for Apache Otava change point detection.

## Overview

This package provides generators for creating synthetic time series data
with known change points for testing and benchmarking change point detection
algorithms.

## Installation

```bash
pip install otava-test-data
```

## Quick Start

```python
from otava_test_data import step_function, noise_normal, combine

# Generate a step function with noise
step = step_function(length=500, value_before=100, value_after=120)
noise = noise_normal(length=500, mean=0, sigma=5)
ts = combine(step, noise)

# Export to CSV for Otava
ts.to_csv("test_data.csv")

# Access change point information
for cp in ts.change_points:
    print(f"Change at index {cp.index}: {cp.description}")
```

## Contents

```{toctree}
:maxdepth: 2

generators
api
benchmark
```
