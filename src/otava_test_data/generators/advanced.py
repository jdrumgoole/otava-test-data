"""
Advanced time series generators for more complex phenomena.

These generators create patterns that represent more nuanced behaviors
seen in real performance testing scenarios.
"""

import numpy as np
from numpy.typing import NDArray

from otava_test_data.generators.basic import TimeSeries, ChangePoint


def banding(
    length: int,
    value1: float = 100.0,
    value2: float = 105.0,
    sigma: float = 0.0,
    seed: int | None = None,
) -> TimeSeries:
    """
    Generate a banding time series - oscillating randomly between two values.

    S = x1, x2, x2, x1, x2, x1, x1, x1, x2, x2, x1, x2, x2...

    Banding is a form of noise (unwanted change) where results oscillate
    randomly between two values. Typically:
    - abs(x2 - x1) << x1 (the band gap is small relative to values)
    - When random noise is mixed in: x1, x2 > std dev

    Args:
        length: Number of data points.
        value1: First band value.
        value2: Second band value.
        sigma: If > 0, add normal noise with this standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        TimeSeries with banding pattern (no explicit change points).
    """
    rng = np.random.default_rng(seed)

    # Randomly choose between the two values for each point
    choices = rng.choice([value1, value2], size=length)
    data = choices.astype(np.float64)

    if sigma > 0:
        data += rng.normal(0, sigma, length)

    # Banding doesn't have explicit change points - it's a noise pattern
    return TimeSeries(
        data=data,
        change_points=[],
        generator_name="banding",
        parameters={
            "length": length,
            "value1": value1,
            "value2": value2,
            "sigma": sigma,
            "seed": seed,
        },
    )


def variance_change(
    length: int,
    mean: float = 100.0,
    sigma_before: float = 2.0,
    sigma_after: float = 10.0,
    change_index: int | None = None,
    seed: int | None = None,
) -> TimeSeries:
    """
    Generate a time series with constant mean but changing variance.

    S = N(mean, sigma1)..., N(mean, sigma2)...

    The mean stays the same, but the spread of values changes at a point.
    This can indicate a change in test stability or environmental factors.

    Args:
        length: Number of data points.
        mean: Mean value (constant throughout).
        sigma_before: Standard deviation before change point.
        sigma_after: Standard deviation after change point.
        change_index: Position of the variance change. If None, placed at length//2.
        seed: Random seed for reproducibility.

    Returns:
        TimeSeries with one variance change point.
    """
    rng = np.random.default_rng(seed)

    if change_index is None:
        change_index = length // 2

    if change_index < 1 or change_index >= length:
        raise ValueError(f"change_index must be in [1, {length}), got {change_index}")

    data = np.empty(length, dtype=np.float64)
    data[:change_index] = rng.normal(mean, sigma_before, change_index)
    data[change_index:] = rng.normal(mean, sigma_after, length - change_index)

    change_points = [
        ChangePoint(
            index=change_index,
            change_type="variance",
            before_value=sigma_before,
            after_value=sigma_after,
            description=f"Variance change from sigma={sigma_before} to sigma={sigma_after}",
        )
    ]

    return TimeSeries(
        data=data,
        change_points=change_points,
        generator_name="variance_change",
        parameters={
            "length": length,
            "mean": mean,
            "sigma_before": sigma_before,
            "sigma_after": sigma_after,
            "change_index": change_index,
            "seed": seed,
        },
    )


def phase_change(
    length: int,
    amplitude: float = 10.0,
    baseline: float = 100.0,
    period: int = 20,
    change_index: int | None = None,
    phase_shift: float = np.pi / 2,  # Default: cos -> sin (90 degree shift)
    sigma: float = 0.0,
    seed: int | None = None,
) -> TimeSeries:
    """
    Generate a time series with constant mean and variance but phase changes.

    S = cos(x)..., sin(x)...

    The underlying periodic pattern shifts phase at a point. While periodic
    patterns are not typical in performance testing (mentioned in spec as
    things we do NOT encounter), this is included for completeness in testing
    detection algorithms.

    Args:
        length: Number of data points.
        amplitude: Amplitude of the oscillation.
        baseline: Baseline value around which oscillation occurs.
        period: Number of points per cycle.
        change_index: Position of the phase change. If None, placed at length//2.
        phase_shift: Amount of phase shift in radians (default pi/2 = cos to sin).
        sigma: If > 0, add normal noise with this standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        TimeSeries with one phase change point.
    """
    rng = np.random.default_rng(seed)

    if change_index is None:
        change_index = length // 2

    if change_index < 1 or change_index >= length:
        raise ValueError(f"change_index must be in [1, {length}), got {change_index}")

    # Generate x values for the cosine/sine
    x = np.arange(length) * 2 * np.pi / period

    data = np.empty(length, dtype=np.float64)
    # Before change: cos(x)
    data[:change_index] = baseline + amplitude * np.cos(x[:change_index])
    # After change: cos(x + phase_shift) = sin(x) when phase_shift = pi/2
    data[change_index:] = baseline + amplitude * np.cos(x[change_index:] + phase_shift)

    if sigma > 0:
        data += rng.normal(0, sigma, length)

    change_points = [
        ChangePoint(
            index=change_index,
            change_type="phase",
            before_value=0.0,
            after_value=phase_shift,
            description=f"Phase shift of {phase_shift:.3f} radians",
        )
    ]

    return TimeSeries(
        data=data,
        change_points=change_points,
        generator_name="phase_change",
        parameters={
            "length": length,
            "amplitude": amplitude,
            "baseline": baseline,
            "period": period,
            "change_index": change_index,
            "phase_shift": phase_shift,
            "sigma": sigma,
            "seed": seed,
        },
    )


def multiple_changes(
    length: int,
    values: list[float] | None = None,
    change_indices: list[int] | None = None,
    sigma: float = 0.0,
    seed: int | None = None,
) -> TimeSeries:
    """
    Generate a time series with multiple consecutive changes.

    S = x0, x0, x0... x1, x2, ... xn, xn, xn...

    Where x0 < x1 < x2 ... < xn (monotonically increasing) or any other
    sequence of distinct values.

    This represents multiple independent improvements or regressions merged
    back to back in performance testing.

    Args:
        length: Number of data points.
        values: List of values for each segment. If None, defaults to
                [100, 110, 120, 130] (three changes).
        change_indices: List of indices where changes occur. Must have
                       len(values) - 1 elements. If None, evenly distributed.
        sigma: If > 0, add normal noise with this standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        TimeSeries with multiple step change points.

    Example:
        >>> ts = multiple_changes(100, values=[100, 120, 140], change_indices=[30, 60])
        >>> len(ts.change_points)
        2
    """
    rng = np.random.default_rng(seed)

    if values is None:
        values = [100.0, 110.0, 120.0, 130.0]

    n_segments = len(values)
    n_changes = n_segments - 1

    if n_changes < 1:
        raise ValueError("values must have at least 2 elements")

    if change_indices is None:
        # Evenly distribute change points
        segment_length = length // n_segments
        change_indices = [segment_length * (i + 1) for i in range(n_changes)]

    if len(change_indices) != n_changes:
        raise ValueError(
            f"change_indices must have {n_changes} elements, got {len(change_indices)}"
        )

    # Validate change indices are in order and within bounds
    all_indices = [0] + list(change_indices) + [length]
    for i in range(len(all_indices) - 1):
        if all_indices[i] >= all_indices[i + 1]:
            raise ValueError("change_indices must be strictly increasing")

    data = np.empty(length, dtype=np.float64)

    # Fill each segment
    segment_starts = [0] + list(change_indices)
    segment_ends = list(change_indices) + [length]

    for i, (start, end, value) in enumerate(zip(segment_starts, segment_ends, values)):
        data[start:end] = value

    if sigma > 0:
        data += rng.normal(0, sigma, length)

    # Create change points
    change_points = []
    for i, idx in enumerate(change_indices):
        change_points.append(
            ChangePoint(
                index=idx,
                change_type="step",
                before_value=values[i],
                after_value=values[i + 1],
                description=f"Change {i + 1}: {values[i]} -> {values[i + 1]}",
            )
        )

    return TimeSeries(
        data=data,
        change_points=change_points,
        generator_name="multiple_changes",
        parameters={
            "length": length,
            "values": values,
            "change_indices": change_indices,
            "sigma": sigma,
            "seed": seed,
        },
    )


def multiple_outliers(
    length: int,
    baseline: float = 100.0,
    outlier_value: float = 150.0,
    outlier_indices: list[int] | None = None,
    n_outliers: int = 3,
    sigma: float = 0.0,
    seed: int | None = None,
) -> TimeSeries:
    """
    Generate a time series with multiple outliers.

    Extension of the single outlier case - multiple isolated anomalous points.

    Args:
        length: Number of data points.
        baseline: The normal/baseline value.
        outlier_value: The outlier value(s).
        outlier_indices: Specific indices for outliers. If None, randomly placed.
        n_outliers: Number of outliers if outlier_indices is None.
        sigma: If > 0, add normal noise with this standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        TimeSeries with multiple outlier change points.
    """
    rng = np.random.default_rng(seed)

    if outlier_indices is None:
        # Randomly place outliers, ensuring they're not adjacent
        available = list(range(1, length - 1))  # Avoid first and last
        if n_outliers > len(available):
            raise ValueError(f"Cannot place {n_outliers} outliers in length {length}")

        outlier_indices = sorted(rng.choice(available, size=n_outliers, replace=False))

    data = np.full(length, baseline, dtype=np.float64)

    if sigma > 0:
        data += rng.normal(0, sigma, length)

    for idx in outlier_indices:
        data[idx] = outlier_value

    change_points = [
        ChangePoint(
            index=idx,
            change_type="outlier",
            before_value=baseline,
            after_value=outlier_value,
            description=f"Outlier at index {idx}",
        )
        for idx in outlier_indices
    ]

    return TimeSeries(
        data=data,
        change_points=change_points,
        generator_name="multiple_outliers",
        parameters={
            "length": length,
            "baseline": baseline,
            "outlier_value": outlier_value,
            "outlier_indices": list(outlier_indices),
            "sigma": sigma,
            "seed": seed,
        },
    )


def multiple_variance_changes(
    length: int,
    mean: float = 100.0,
    sigmas: list[float] | None = None,
    change_indices: list[int] | None = None,
    seed: int | None = None,
) -> TimeSeries:
    """
    Generate a time series with multiple variance changes.

    Extension of variance_change - variance changes multiple times while
    mean stays constant.

    Args:
        length: Number of data points.
        mean: Constant mean value.
        sigmas: List of sigma values for each segment.
        change_indices: Indices where variance changes occur.
        seed: Random seed for reproducibility.

    Returns:
        TimeSeries with multiple variance change points.
    """
    rng = np.random.default_rng(seed)

    if sigmas is None:
        sigmas = [2.0, 8.0, 3.0, 10.0]

    n_segments = len(sigmas)
    n_changes = n_segments - 1

    if change_indices is None:
        segment_length = length // n_segments
        change_indices = [segment_length * (i + 1) for i in range(n_changes)]

    if len(change_indices) != n_changes:
        raise ValueError(
            f"change_indices must have {n_changes} elements, got {len(change_indices)}"
        )

    data = np.empty(length, dtype=np.float64)

    segment_starts = [0] + list(change_indices)
    segment_ends = list(change_indices) + [length]

    for start, end, sigma in zip(segment_starts, segment_ends, sigmas):
        segment_length = end - start
        data[start:end] = rng.normal(mean, sigma, segment_length)

    change_points = [
        ChangePoint(
            index=idx,
            change_type="variance",
            before_value=sigmas[i],
            after_value=sigmas[i + 1],
            description=f"Variance change: sigma {sigmas[i]} -> {sigmas[i + 1]}",
        )
        for i, idx in enumerate(change_indices)
    ]

    return TimeSeries(
        data=data,
        change_points=change_points,
        generator_name="multiple_variance_changes",
        parameters={
            "length": length,
            "mean": mean,
            "sigmas": sigmas,
            "change_indices": change_indices,
            "seed": seed,
        },
    )
