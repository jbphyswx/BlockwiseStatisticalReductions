"""
Internal accumulators for numerically stable parallel statistics.

These are implementation details - users should use the public API functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic
import numpy as np
from numba import njit

T = TypeVar('T', float, np.float32, np.float64)


@dataclass
class VarianceAccumulator:
    """
    Accumulator for mean and variance with numerically stable parallel merge.
    
    Uses Welford's online algorithm for incremental updates and Chan's algorithm 
    for parallel merge.
    """
    count: int
    mean: float
    sum_sq_dev: float  # Σ(xᵢ - mean)², also called M2
    
    def __init__(self, dtype: type = float):
        self.count = 0
        self.mean = dtype(0.0)
        self.sum_sq_dev = dtype(0.0)
    
    def fit(self, x: float | np.ndarray) -> None:
        """Add sample(s) to accumulator."""
        if isinstance(x, np.ndarray):
            for val in x.flat:
                self._fit_one(val)
        else:
            self._fit_one(x)
    
    def _fit_one(self, x: float) -> None:
        """Add single sample using Welford's algorithm."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.sum_sq_dev += delta * delta2
    
    def variance(self, corrected: bool = True) -> float:
        """Return variance. Uses Bessel's correction by default."""
        if self.count == 0:
            return np.nan
        if self.count == 1 and corrected:
            return np.nan
        denom = self.count - 1 if corrected else self.count
        return self.sum_sq_dev / denom
    
    def std(self, corrected: bool = True) -> float:
        """Return standard deviation."""
        return np.sqrt(self.variance(corrected))
    
    def merge(self, other: VarianceAccumulator) -> VarianceAccumulator:
        """
        Merge two accumulators using Chan's parallel algorithm.
        
        This allows hierarchical reductions where sub-blocks are computed
        independently then combined.
        """
        if self.count == 0:
            return VarianceAccumulator.__class__(type(self.mean))()
        if other.count == 0:
            return self
        
        # Chan's algorithm for parallel variance computation
        n1, n2 = self.count, other.count
        n = n1 + n2
        
        # Compute pooled mean
        pooled_mean = (n1 * self.mean + n2 * other.mean) / n
        
        # Compute pooled M2
        # M2_total = M2_1 + M2_2 + n1*(mean1 - pooled_mean)^2 + n2*(mean2 - pooled_mean)^2
        delta1 = self.mean - pooled_mean
        delta2 = other.mean - pooled_mean
        
        pooled_m2 = (
            self.sum_sq_dev + other.sum_sq_dev +
            n1 * delta1 * delta1 + n2 * delta2 * delta2
        )
        
        result = VarianceAccumulator(type(self.mean))
        result.count = n
        result.mean = pooled_mean
        result.sum_sq_dev = pooled_m2
        return result


@dataclass
class CovarianceAccumulator:
    """
    Accumulator for covariance with numerically stable parallel merge.
    
    Uses Pebay's extension of Chan's algorithm for parallel covariance computation.
    """
    count: int
    mean_x: float
    mean_y: float
    sum_cross_dev: float  # Σ(xᵢ - mean_x)(yᵢ - mean_y)
    
    def __init__(self, dtype: type = float):
        self.count = 0
        self.mean_x = dtype(0.0)
        self.mean_y = dtype(0.0)
        self.sum_cross_dev = dtype(0.0)
    
    def fit(self, x: float, y: float) -> None:
        """Add paired sample using online algorithm."""
        self.count += 1
        
        # Online update for means
        delta_x = x - self.mean_x
        delta_y = y - self.mean_y
        
        self.mean_x += delta_x / self.count
        self.mean_y += delta_y / self.count
        
        # Update cross-deviation sum
        self.sum_cross_dev += delta_x * (y - self.mean_y)
    
    def fit_arrays(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add paired samples from arrays."""
        for xi, yi in zip(x.flat, y.flat):
            self.fit(xi, yi)
    
    def covariance(self, corrected: bool = True) -> float:
        """Return covariance. Uses Bessel's correction by default."""
        if self.count == 0:
            return np.nan
        if self.count == 1 and corrected:
            return np.nan
        denom = self.count - 1 if corrected else self.count
        return self.sum_cross_dev / denom
    
    def merge(self, other: CovarianceAccumulator) -> CovarianceAccumulator:
        """
        Merge two accumulators using Pebay's parallel algorithm.
        """
        if self.count == 0:
            return CovarianceAccumulator.__class__(type(self.mean_x))()
        if other.count == 0:
            return self
        
        n1, n2 = self.count, other.count
        n = n1 + n2
        
        # Compute pooled means
        pooled_mean_x = (n1 * self.mean_x + n2 * other.mean_x) / n
        pooled_mean_y = (n1 * self.mean_y + n2 * other.mean_y) / n
        
        # Compute pooled cross-deviation sum
        delta_x1 = self.mean_x - pooled_mean_x
        delta_y1 = self.mean_y - pooled_mean_y
        delta_x2 = other.mean_x - pooled_mean_x
        delta_y2 = other.mean_y - pooled_mean_y
        
        pooled_cross_dev = (
            self.sum_cross_dev + other.sum_cross_dev +
            n1 * delta_x1 * delta_y1 + n2 * delta_x2 * delta_y2
        )
        
        result = CovarianceAccumulator(type(self.mean_x))
        result.count = n
        result.mean_x = pooled_mean_x
        result.mean_y = pooled_mean_y
        result.sum_cross_dev = pooled_cross_dev
        return result


@dataclass 
class RawMomentsAccumulator:
    """
    Accumulator for raw moments up to order N.
    
    Stores E[x], E[x²], ..., E[x^N] for arbitrary N.
    """
    count: int
    moments: tuple
    max_order: int
    
    def __init__(self, max_order: int = 4, dtype: type = float):
        self.count = 0
        self.max_order = max_order
        self.moments = tuple(dtype(0.0) for _ in range(max_order))
    
    def fit(self, x: float | np.ndarray) -> None:
        """Add sample(s) to accumulator."""
        if isinstance(x, np.ndarray):
            for val in x.flat:
                self._fit_one(val)
        else:
            self._fit_one(x)
    
    def _fit_one(self, x: float) -> None:
        """Add single sample."""
        self.count += 1
        
        # Update moments using online algorithm
        new_moments = []
        for k in range(self.max_order):
            order = k + 1  # 1-indexed
            # E[x^k] = (n-1)/n * E[x^k] + 1/n * x^k
            old_moment = self.moments[k]
            x_power = x ** order
            new_moment = ((self.count - 1) * old_moment + x_power) / self.count
            new_moments.append(new_moment)
        
        self.moments = tuple(new_moments)
    
    def merge(self, other: RawMomentsAccumulator) -> RawMomentsAccumulator:
        """Merge two accumulators."""
        if self.max_order != other.max_order:
            raise ValueError("Cannot merge accumulators with different max_order")
        
        if self.count == 0:
            return RawMomentsAccumulator(self.max_order, type(self.moments[0]))
        if other.count == 0:
            return self
        
        n1, n2 = self.count, other.count
        n = n1 + n2
        
        result = RawMomentsAccumulator(self.max_order, type(self.moments[0]))
        result.count = n
        
        # Weighted average of moments
        new_moments = []
        for k in range(self.max_order):
            merged = (n1 * self.moments[k] + n2 * other.moments[k]) / n
            new_moments.append(merged)
        
        result.moments = tuple(new_moments)
        return result


# Numba-accelerated batch merge functions

@njit(cache=True)
def _chan_merge_variance_numba(
    count1: int, mean1: float, m2_1: float,
    count2: int, mean2: float, m2_2: float
) -> tuple[int, float, float]:
    """
    Chan's algorithm for merging two variance accumulators.
    Returns (count, mean, m2) for merged result.
    """
    if count1 == 0:
        return count2, mean2, m2_2
    if count2 == 0:
        return count1, mean1, m2_1
    
    n1, n2 = count1, count2
    n = n1 + n2
    
    # Pooled mean
    pooled_mean = (n1 * mean1 + n2 * mean2) / n
    
    # Pooled M2
    delta1 = mean1 - pooled_mean
    delta2 = mean2 - pooled_mean
    pooled_m2 = m2_1 + m2_2 + n1 * delta1 * delta1 + n2 * delta2 * delta2
    
    return n, pooled_mean, pooled_m2


@njit(cache=True)
def _pebay_merge_covariance_numba(
    count1: int, mean_x1: float, mean_y1: float, c_1: float,
    count2: int, mean_x2: float, mean_y2: float, c_2: float
) -> tuple[int, float, float, float]:
    """
    Pebay's algorithm for merging two covariance accumulators.
    Returns (count, mean_x, mean_y, cross_dev) for merged result.
    """
    if count1 == 0:
        return count2, mean_x2, mean_y2, c_2
    if count2 == 0:
        return count1, mean_x1, mean_y1, c_1
    
    n1, n2 = count1, count2
    n = n1 + n2
    
    # Pooled means
    pooled_mean_x = (n1 * mean_x1 + n2 * mean_x2) / n
    pooled_mean_y = (n1 * mean_y1 + n2 * mean_y2) / n
    
    # Pooled cross-deviation
    delta_x1 = mean_x1 - pooled_mean_x
    delta_y1 = mean_y1 - pooled_mean_y
    delta_x2 = mean_x2 - pooled_mean_x
    delta_y2 = mean_y2 - pooled_mean_y
    
    pooled_c = c_1 + c_2 + n1 * delta_x1 * delta_y1 + n2 * delta_x2 * delta_y2
    
    return n, pooled_mean_x, pooled_mean_y, pooled_c
