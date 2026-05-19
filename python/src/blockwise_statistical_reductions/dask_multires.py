"""
Multi-resolution Dask graph optimization with automatic caching.

Builds Dask graphs that compute statistics at multiple scales,
reusing intermediate results when factors divide evenly.
"""

from __future__ import annotations

import numpy as np
import dask.array as da
from dask import delayed
from typing import Callable, Sequence

from ._accumulators import VarianceAccumulator, CovarianceAccumulator
from ._numba_kernels import _chan_merge_variance, _pebay_merge_covariance


def factor_sequence(start: int, targets: Sequence[int]) -> list[int]:
    """
    Find ordered sequence of factors from `start` to cover all `targets`,
    where each factor divides evenly into the next.
    
    This enables caching: if we compute at factor 2, we can reuse for 4, 8, etc.
    """
    sorted_targets = sorted(targets)
    result = [start]
    current = start
    
    for target in sorted_targets:
        if target <= current:
            continue
        
        # Find intermediate factors
        if target % current == 0:
            # Direct path exists
            result.append(target)
            current = target
        else:
            # Need intermediate steps
            # Find largest factor of target that divides current or is >= current
            for step in range(current + 1, target + 1):
                if target % step == 0 and step % current == 0:
                    result.append(step)
                    current = step
                    break
            if current != target:
                result.append(target)
                current = target
    
    # Remove duplicates while preserving order
    seen = set()
    unique_result = []
    for x in result:
        if x not in seen:
            seen.add(x)
            unique_result.append(x)
    
    return unique_result


class MultiResolutionDask:
    """
    Multi-resolution reduction with automatic caching for Dask arrays.
    
    Builds optimized Dask graphs that reuse intermediate results when
    reduction factors divide evenly.
    """
    
    def __init__(self, shape: tuple[int, ...], factors: Sequence[int]):
        """
        Initialize multi-resolution planner.
        
        Parameters
        ----------
        shape : Input array shape
        factors : Target reduction factors (e.g., [2, 4, 8, 16])
        """
        self.shape = shape
        self.factors = factor_sequence(1, factors)
        self._cache: dict[int, da.Array] = {}
    
    def build_graph(
        self,
        arr: da.Array,
        stat: str = 'mean',
        use_cache: bool = True
    ) -> dict[int, da.Array]:
        """
        Build Dask graph for multi-resolution computation.
        
        Parameters
        ----------
        arr : Input Dask array
        stat : Statistic to compute ('mean', 'variance', 'std')
        use_cache : Whether to reuse intermediate results
        
        Returns
        -------
        Dictionary mapping factor -> Dask array
        """
        results = {}
        prev_arr = arr
        prev_factor = 1
        
        for factor in self.factors:
            if factor == 1:
                continue  # Skip identity
            
            if use_cache and factor % prev_factor == 0 and prev_factor > 1:
                # Reduce from previous level
                relative_factor = factor // prev_factor
                result = self._reduce_from_cached(
                    self._cache[prev_factor],
                    relative_factor,
                    stat
                )
            else:
                # Compute from original
                result = self._reduce_from_original(arr, factor, stat)
            
            if use_cache:
                self._cache[factor] = result
            
            results[factor] = result
            prev_factor = factor
        
        return results
    
    def _reduce_from_cached(
        self,
        cached: da.Array,
        relative_factor: int,
        stat: str
    ) -> da.Array:
        """Reduce from cached intermediate result."""
        if stat == 'mean':
            return self._blockwise_mean_dask(cached, (relative_factor,) * cached.ndim)
        elif stat == 'variance':
            return self._blockwise_variance_dask(cached, (relative_factor,) * cached.ndim)
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    def _reduce_from_original(
        self,
        arr: da.Array,
        factor: int,
        stat: str
    ) -> da.Array:
        """Reduce from original array."""
        window_sizes = (factor,) * arr.ndim
        
        if stat == 'mean':
            return self._blockwise_mean_dask(arr, window_sizes)
        elif stat == 'variance':
            return self._blockwise_variance_dask(arr, window_sizes)
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    def _blockwise_mean_dask(
        self,
        arr: da.Array,
        window_sizes: tuple[int, ...]
    ) -> da.Array:
        """Blockwise mean using Dask's coarsen."""
        # Dask's coarsen is perfect for this
        axes = {i: w for i, w in enumerate(window_sizes)}
        return da.coarsen(np.mean, arr, axes, trim_excess=True)
    
    def _blockwise_variance_dask(
        self,
        arr: da.Array,
        window_sizes: tuple[int, ...]
    ) -> da.Array:
        """Blockwise variance using Dask with Welford."""
        # Use map_blocks with custom variance function
        from ._numba_kernels import blockwise_variance
        
        out_shape = tuple(s // w for s, w in zip(arr.shape, window_sizes))
        
        def variance_block(block):
            return blockwise_variance(block, window_sizes)
        
        return da.map_blocks(
            variance_block,
            arr,
            dtype=arr.dtype,
            chunks=tuple(c // w for c, w in zip(arr.chunks, window_sizes))
        )


def dask_multiresolution_stats(
    arr: da.Array,
    factors: Sequence[int],
    stats: Sequence[str] = ('mean',)
) -> dict[int, dict[str, da.Array]]:
    """
    Compute multiple statistics at multiple resolutions.
    
    High-level API for multi-resolution Dask computation.
    
    Parameters
    ----------
    arr : Input Dask array
    factors : Target reduction factors
    stats : Statistics to compute ('mean', 'variance', etc.)
    
    Returns
    -------
    Nested dict: results[factor][stat_name] = dask_array
    """
    planner = MultiResolutionDask(arr.shape, factors)
    
    results = {}
    for factor in factors:
        results[factor] = {}
    
    # Build graph for each stat
    for stat in stats:
        stat_results = planner.build_graph(arr, stat, use_cache=True)
        for factor, result in stat_results.items():
            results[factor][stat] = result
    
    return results


# Parallel merge operations for Dask

def _delayed_merge_variance(
    acc1: tuple[int, float, float],
    acc2: tuple[int, float, float]
) -> tuple[int, float, float]:
    """
    Delayed wrapper for Chan merge.
    Input/Output: (count, mean, m2)
    """
    n1, mean1, m2_1 = acc1
    n2, mean2, m2_2 = acc2
    return _chan_merge_variance(n1, mean1, m2_1, n2, mean2, m2_2)


def _delayed_merge_covariance(
    acc1: tuple[int, float, float, float],
    acc2: tuple[int, float, float, float]
) -> tuple[int, float, float, float]:
    """
    Delayed wrapper for Pebay merge.
    Input/Output: (count, mean_x, mean_y, cross_dev)
    """
    n1, mx1, my1, c1 = acc1
    n2, mx2, my2, c2 = acc2
    return _pebay_merge_covariance(n1, mx1, my1, c1, n2, mx2, my2, c2)


def dask_tree_reduce_variance(
    blocks: list[da.Array],
    axis: int = 0
) -> da.Array:
    """
    Tree reduction of variance accumulators across Dask blocks.
    
    Uses parallel merge for numerical stability.
    """
    # Convert blocks to accumulator format
    def to_accumulator(block):
        acc = VarianceAccumulator()
        acc.fit(block)
        return (acc.count, acc.mean, acc.sum_sq_dev)
    
    delayed_blocks = [delayed(to_accumulator)(b) for b in blocks]
    
    # Tree reduction
    while len(delayed_blocks) > 1:
        new_blocks = []
        for i in range(0, len(delayed_blocks), 2):
            if i + 1 < len(delayed_blocks):
                merged = delayed(_delayed_merge_variance)(
                    delayed_blocks[i], delayed_blocks[i + 1]
                )
                new_blocks.append(merged)
            else:
                new_blocks.append(delayed_blocks[i])
        delayed_blocks = new_blocks
    
    # Final result
    def from_accumulator(acc_tuple):
        n, mean, m2 = acc_tuple
        acc = VarianceAccumulator()
        acc.count = n
        acc.mean = mean
        acc.sum_sq_dev = m2
        return acc.variance()
    
    return delayed(from_accumulator)(delayed_blocks[0])
