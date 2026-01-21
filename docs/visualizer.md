# Web Visualizer

The otava-test-data package includes an interactive web visualizer for exploring test patterns and comparing Otava's detection results against ground truth.

## Starting the Visualizer

```bash
# Start the web server in background
inv web-start

# Or run in foreground
inv web --port 8100

# Check server status
inv web-status

# Stop the server
inv web-stop
```

Then open http://127.0.0.1:8100 in your browser.

## Interface Overview

The visualizer provides:

- **Generator Selection**: Choose from 12 different test pattern generators
- **Parameter Controls**: Adjust generator-specific parameters
- **Otava Analysis**: Run change point detection and view results
- **Accuracy Metrics**: Compare detected vs ground truth change points

## Common Parameters

These parameters appear across multiple generators:

### Length

The total number of data points in the generated time series. Use longer series (500+) to give Otava more context for detection.

- **Range**: 50 - 1000
- **Default**: 200

### Seed

Random seed for reproducible generation. Using the same seed with the same parameters will produce identical results.

- **Default**: 42

### Sigma (Standard Deviation)

The standard deviation of Gaussian (normal) noise added to the signal. This is one of the most important parameters as it affects how easily Otava can detect change points.

**Low sigma (e.g., 2)**: Clean signal with minimal noise. Change points are easy to detect visually and statistically.

![Step function with sigma=2](_static/screenshots/step-function-sigma-2.png)

**High sigma (e.g., 15)**: Noisy signal. Change points become harder to detect, more similar to real-world performance data.

![Step function with sigma=15](_static/screenshots/step-function-sigma-15.png)

## Generator-Specific Parameters

### Step Function

Generates a single change point where the signal steps from one value to another.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Value Before | Signal value before the change point | 100 |
| Value After | Signal value after the change point | 120 |
| Change Point | Index where the step occurs | Middle of series |
| Sigma | Noise standard deviation | 5 |

**Use case**: Simulates a performance regression or improvement that persists.

### Variance Change

Signal with constant mean but changing variance (spread of noise).

| Parameter | Description | Default |
|-----------|-------------|---------|
| Mean | Constant signal mean | 100 |
| Sigma Before | Standard deviation before change | 2 |
| Sigma After | Standard deviation after change | 10 |

![Variance change example](_static/screenshots/variance-change.png)

**Use case**: Simulates a system becoming more unstable or erratic without mean shift.

### Multiple Changes

Multiple consecutive step changes creating a staircase pattern.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Sigma | Noise standard deviation | 5 |

![Multiple changes example](_static/screenshots/multiple-changes.png)

The generator creates 3 evenly-spaced step changes, each increasing by 10 units.

**Use case**: Simulates multiple successive performance regressions.

### Banding

Oscillation between two distinct values, creating a banded pattern.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Value1 | First band value | 100 |
| Value2 | Second band value | 105 |
| Sigma | Noise standard deviation | 2 |

![Banding example](_static/screenshots/banding.png)

**Use case**: Simulates bimodal performance (e.g., alternating between two configurations).

### Single Outlier

A single anomalous point in an otherwise constant signal.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Baseline | Normal signal value | 100 |
| Outlier Value | Value of the anomaly | 150 |
| Sigma | Noise standard deviation | 5 |

![Single outlier example](_static/screenshots/single-outlier.png)

**Use case**: Simulates a one-time spike or glitch in performance data.

### Phase Change

Periodic signal (sine wave) with a phase shift at the change point.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Amplitude | Wave amplitude | 10 |
| Baseline | Center value of oscillation | 100 |
| Period | Number of points per cycle | 20 |
| Sigma | Noise standard deviation | 2 |

![Phase change example](_static/screenshots/phase-change.png)

**Use case**: Simulates subtle timing changes in periodic behavior.

### Regression + Fix

Temporary regression that gets fixed - signal drops then recovers.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Value Before | Original performance | 100 |
| Regression Value | Value during regression | 80 |
| Value After | Value after fix | 100 |
| Sigma | Noise standard deviation | 5 |

**Use case**: Simulates a bug that causes regression and is later fixed.

### Constant

Constant value with no change points (baseline for comparison).

| Parameter | Description | Default |
|-----------|-------------|---------|
| Value | Constant signal value | 100 |

**Use case**: Verify that Otava doesn't produce false positives on stable data.

### Normal Noise

Pure Gaussian noise with no underlying signal.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Mean | Center of distribution | 100 |
| Sigma | Standard deviation | 10 |

**Use case**: Test behavior on random noise without change points.

### Uniform Noise

Uniformly distributed random values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Min | Minimum value | 90 |
| Max | Maximum value | 110 |

**Use case**: Test with non-Gaussian noise distribution.

## Otava Analysis Parameters

### Window Length

The sliding window size used by Otava for statistical analysis.

- **Default**: 30
- **Effect**: Smaller windows are more sensitive but prone to false positives. Larger windows are more robust but may miss closely-spaced changes.

### Max P-Value

Statistical threshold for change point significance.

- **Default**: 0.05
- **Effect**: Lower values make detection more strict (fewer false positives but may miss true changes).

### Match Tolerance

When comparing detected vs ground truth change points, how many indices apart can they be and still count as a match.

- **Default**: 5
- **Effect**: Allows for slight positional inaccuracy in detection.

## Accuracy Metrics

The visualizer calculates these metrics when comparing Otava's results to ground truth:

- **Precision**: Of all detected change points, what fraction are correct?
- **Recall**: Of all true change points, what fraction were detected?
- **F1 Score**: Harmonic mean of precision and recall
- **True Positives (TP)**: Correctly detected change points
- **False Positives (FP)**: Incorrectly flagged non-change-points
- **False Negatives (FN)**: Missed true change points

## Compare All Patterns

Click "Compare All Patterns" to run Otava analysis on all 12 generators simultaneously. This produces a summary table showing how well Otava performs across different types of change point patterns.
