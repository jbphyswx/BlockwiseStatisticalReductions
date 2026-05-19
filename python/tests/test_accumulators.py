"""
Tests for Python accumulators with parallel merge.
"""

import numpy as np
import pytest
from blockwise_statistical_reductions._accumulators import (
    VarianceAccumulator,
    CovarianceAccumulator,
    RawMomentsAccumulator,
)


def test_variance_accumulator_basic():
    """Test basic variance accumulator operations."""
    acc = VarianceAccumulator()
    
    # Add values
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    for x in data:
        acc.fit(x)
    
    assert acc.count == 5
    assert acc.mean == pytest.approx(3.0)
    # Population variance
    assert acc.variance(corrected=False) == pytest.approx(2.0)
    # Sample variance (with Bessel's correction)
    assert acc.variance(corrected=True) == pytest.approx(2.5)


def test_variance_merge():
    """Test parallel variance merge (Chan's algorithm)."""
    # Split data into two blocks
    data = np.random.randn(100)
    mid = len(data) // 2
    
    acc1 = VarianceAccumulator()
    acc1.fit(data[:mid])
    
    acc2 = VarianceAccumulator()
    acc2.fit(data[mid:])
    
    # Merge
    merged = acc1.merge(acc2)
    
    # Compare to computing on full data
    full_acc = VarianceAccumulator()
    full_acc.fit(data)
    
    assert merged.mean == pytest.approx(full_acc.mean, abs=1e-10)
    assert merged.variance() == pytest.approx(full_acc.variance(), abs=1e-10)


def test_covariance_accumulator_basic():
    """Test basic covariance accumulator."""
    acc = CovarianceAccumulator()
    
    # Perfect linear relationship: y = 2x
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    for x, y in zip(xs, ys):
        acc.fit(x, y)
    
    assert acc.count == 5
    assert acc.mean_x == pytest.approx(3.0)
    assert acc.mean_y == pytest.approx(6.0)
    assert acc.covariance() > 0  # Positive covariance


def test_covariance_merge():
    """Test parallel covariance merge (Pebay's algorithm)."""
    n = 100
    xs = np.random.randn(n)
    ys = 2 * xs + np.random.randn(n) * 0.1  # Correlated
    
    mid = n // 2
    
    acc1 = CovarianceAccumulator()
    acc1.fit_arrays(xs[:mid], ys[:mid])
    
    acc2 = CovarianceAccumulator()
    acc2.fit_arrays(xs[mid:], ys[mid:])
    
    # Merge
    merged = acc1.merge(acc2)
    
    # Compare to full computation
    full_acc = CovarianceAccumulator()
    full_acc.fit_arrays(xs, ys)
    
    assert merged.mean_x == pytest.approx(full_acc.mean_x, abs=1e-10)
    assert merged.mean_y == pytest.approx(full_acc.mean_y, abs=1e-10)
    assert merged.covariance() == pytest.approx(full_acc.covariance(), abs=1e-10)


def test_raw_moments_accumulator():
    """Test raw moments accumulator."""
    acc = RawMomentsAccumulator(max_order=4)
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    for x in data:
        acc.fit(x)
    
    assert acc.count == 5
    assert len(acc.moments) == 4
    
    # First moment = mean
    assert acc.moments[0] == pytest.approx(np.mean(data))
    
    # Second moment = E[x²]
    assert acc.moments[1] == pytest.approx(np.mean([x**2 for x in data]))


def test_raw_moments_merge():
    """Test raw moments merge."""
    data = np.random.randn(100)
    mid = len(data) // 2
    
    acc1 = RawMomentsAccumulator(max_order=4)
    acc1.fit(data[:mid])
    
    acc2 = RawMomentsAccumulator(max_order=4)
    acc2.fit(data[mid:])
    
    # Merge
    merged = acc1.merge(acc2)
    
    # Compare to full
    full_acc = RawMomentsAccumulator(max_order=4)
    full_acc.fit(data)
    
    for i in range(4):
        assert merged.moments[i] == pytest.approx(full_acc.moments[i], abs=1e-10)


def test_variance_edge_cases():
    """Test edge cases for variance accumulator."""
    # Empty
    acc = VarianceAccumulator()
    assert np.isnan(acc.variance())
    
    # Single element
    acc.fit(5.0)
    assert acc.count == 1
    assert acc.mean == 5.0
    assert np.isnan(acc.variance(corrected=True))  # Bessel's correction gives NaN
    assert acc.variance(corrected=False) == 0.0  # Population variance is 0
