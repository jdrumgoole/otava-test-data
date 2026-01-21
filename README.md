# Otava Test Data

Test data generators and benchmarks for [Apache Otava](https://github.com/apache/otava) change point detection.

## Overview

This package provides Python generators for creating synthetic time series data with known change points. It enables:

1. **Testing Otava** - Verify Otava correctly detects change points in controlled scenarios
2. **Benchmarking** - Compare Otava against other change detection algorithms
3. **Education** - Understand what change point detection does with benchmark results

## Installation

```bash
pip install otava-test-data
```

Or with development dependencies:

```bash
pip install otava-test-data[dev,docs]
```

## Quick Start

```python
from otava_test_data import step_function, noise_normal, combine

# Generate a step function (single change point) with realistic noise
step = step_function(length=500, value_before=100, value_after=120)
noise = noise_normal(length=500, mean=0, sigma=5)
combined = combine(step, noise)

# Export to CSV for Otava analysis
combined.to_csv("test_data.csv")

# Access ground truth change point information
for cp in combined.change_points:
    print(f"Change at index {cp.index}: {cp.description}")
```

## Available Generators

### Basic Building Blocks

| Generator | Description |
|-----------|-------------|
| `constant` | Constant value: `S = x, x, x, x...` |
| `noise_normal` | Normal distribution: `S ~ N(mean, sigma)` |
| `noise_uniform` | Uniform distribution: `S ~ U(min, max)` |
| `outlier` | Single anomaly: `S = x, x, x', x, x...` |
| `step_function` | Single change point: `S = x1, x1, x2, x2...` |
| `regression_fix` | Temporary regression: `S = x1, x2, x1...` |

### Advanced Patterns

| Generator | Description |
|-----------|-------------|
| `banding` | Oscillation between two values |
| `variance_change` | Constant mean, changing variance |
| `phase_change` | Phase shift in periodic signal |
| `multiple_changes` | Multiple consecutive step changes |

## Generating Test Data

### Using the CLI

```bash
# Generate test suite
otava-gen generate --output-dir ./test_data --lengths 50 500 --seed 42

# List available generators
otava-gen list

# Get info about a generator
otava-gen info step_function
```

### Using Invoke Tasks

```bash
# Generate test data
inv generate --output-dir ./test_data

# Run tests
inv test

# Build documentation
inv docs-init
inv docs-build
```

## Web Visualizer

The package includes an interactive web visualizer for exploring test patterns and comparing Otava's detection results against ground truth.

```bash
# Start the web server
inv web-start

# Or run in foreground
inv web --port 8100

# Check server status
inv web-status

# Stop the server
inv web-stop
```

Then open http://127.0.0.1:8100 in your browser to:
- Generate and visualize any test pattern
- Run Otava detection with configurable parameters
- Compare ground truth vs detected change points
- View accuracy metrics (precision, recall, F1 score)

## Running Tests

```bash
# All tests
inv test

# Generator tests only (no Otava required)
inv test-generators

# Otava integration tests
inv test-otava
```

## Problem Domain

This project focuses on performance testing scenarios where:

- A performance test runs repeatedly, generating time series measurements
- Ideally produces constant values, but includes random noise
- We assume approximately normal distribution of noise
- Performance regressions/improvements appear as discrete steps

Things we do **NOT** model (not typical in perf testing):
- Trending (constant increase/decrease)
- Seasonality (time-of-day/week/year patterns)

## License

Apache License 2.0
