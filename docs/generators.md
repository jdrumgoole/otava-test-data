# Time Series Generators

## Basic Building Blocks

These generators create fundamental time series patterns used in performance
testing scenarios.

### Constant

A constant time series: `S = x, x, x, x...`

Represents an ideal performance test with no variation.

### Noise (Normal)

Normally distributed noise: `S = x1, x2, x3...` where `X ~ N(mean, sigma)`

Represents typical performance test output with random variation.

### Noise (Uniform)

Uniformly distributed noise (white noise): `random(min, max)`

### Outlier

Single deviating point (anomaly): `S = x, x, x, x, x, x', x, x...`

### Step Function

Single change point: `S = x1, x1, x1, x2, x2, x2...`

Represents a performance regression or improvement that persists.

### Regression + Fix

Temporary regression: `S = x1, x1... x2, ...x2, x3, x3...`

## Advanced Phenomena

### Banding

Oscillation between two values: `S = x1, x2, x2, x1, x2, x1...`

### Variance Change

Constant mean, changing variance: `S = N(mean, sigma1)..., N(mean, sigma2)...`

### Phase Change

Constant mean and variance, but phase shifts: `S = cos(x)..., sin(x)...`

### Multiple Changes

Multiple consecutive changes: `S = x0, x0... x1, x2, ... xn, xn...`
