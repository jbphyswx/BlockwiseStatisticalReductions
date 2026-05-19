"""
Statistical reduction kernels with Numba optimization.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Literal, Sequence
import numpy as np
from numba import njit, prange
import bottleneck as bn

from .core import WindowConfig, ReductionResult, rolling_windows


# Numba-optimized Welford online statistics

@njit(cache=True)
def _welford_online_1d(data: np.ndarray) -> tuple[float, float, int]:
    """
    Welford's online algorithm for mean and variance.
    Returns (mean, variance, count).
    """
    n = 0
    mean = 0.0
    m2 = 0.0

    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2

    variance = m2 / n if n > 0 else 0.0
    return mean, variance, n


@njit(cache=True, parallel=True)
def _parallel_window_stats(
    data: np.ndarray,
    window_size: tuple,
    stride: tuple,
    stat: int,  # 0=mean, 1=var, 2=std, 3=min, 4=max, 5=sum, 6=count
) -> np.ndarray:
    """
    Compute statistics for all windows in parallel.

    This is a low-level numba kernel for performance-critical paths.
    """
    ndim = data.ndim
    shape = data.shape

    # Compute output shape
    out_shape = []
    for i in range(ndim):
        out_dim = (shape[i] - window_size[i]) // stride[i] + 1
        out_shape.append(out_dim)

    out_shape_tuple = tuple(out_shape)
    n_windows = np.prod(np.array(out_shape))

    # Flatten for parallel processing
    result = np.empty(n_windows, dtype=np.float64)

    for flat_idx in prange(n_windows):
        # Convert flat index to multi-dimensional indices
        idx = flat_idx
        window_idx = []
        for dim_size in reversed(out_shape):
            window_idx.append(idx % dim_size)
            idx //= dim_size
        window_idx = window_idx[::-1]

        # Compute start position
        start_pos = [window_idx[i] * stride[i] for i in range(ndim)]

        # Extract and compute
        if ndim == 1:
            s0, w0 = start_pos[0], window_size[0]
            window_data = data[s0:s0+w0]
        elif ndim == 2:
            s0, s1 = start_pos[0], start_pos[1]
            w0, w1 = window_size[0], window_size[1]
            window_data = data[s0:s0+w0, s1:s1+w1].flatten()
        elif ndim == 3:
            s0, s1, s2 = start_pos[0], start_pos[1], start_pos[2]
            w0, w1, w2 = window_size[0], window_size[1], window_size[2]
            window_data = data[s0:s0+w0, s1:s1+w1, s2:s2+w2].flatten()
        else:
            # Generic but slower
            slices = tuple(slice(s, s+w) for s, w in zip(start_pos, window_size))
            window_data = data[slices].flatten()

        # Compute statistic
        if stat == 0:  # mean
            result[flat_idx] = window_data.mean()
        elif stat == 1:  # var
            result[flat_idx] = window_data.var()
        elif stat == 2:  # std
            result[flat_idx] = window_data.std()
        elif stat == 3:  # min
            result[flat_idx] = window_data.min()
        elif stat == 4:  # max
            result[flat_idx] = window_data.max()
        elif stat == 5:  # sum
            result[flat_idx] = window_data.sum()
        elif stat == 6:  # count
            result[flat_idx] = len(window_data)

    return result.reshape(out_shape_tuple)


# Bottleneck-accelerated statistics

def _bottleneck_stat(data: np.ndarray, stat: str, axis: int | None = None) -> np.ndarray:
    """Use bottleneck for fast NaN-aware statistics."""
    func_map = {
        "mean": bn.nanmean,
        "std": bn.nanstd,
        "var": bn.nanvar,
        "min": bn.nanmin,
        "max": bn.nanmax,
        "sum": bn.nansum,
        "median": bn.nanmedian,
    }
    func = func_map.get(stat, bn.nanmean)
    return func(data, axis=axis)


def window_stat(
    window: np.ndarray,
    stat: Literal["mean", "var", "std", "min", "max", "sum", "count", "median"],
    axis: int | None = None,
) -> float | np.ndarray:
    """
    Compute a statistic for a single window.

    Uses bottleneck for NaN-aware fast paths when available.
    """
    if stat in ("mean", "var", "std", "min", "max", "sum", "median"):
        return _bottleneck_stat(window, stat, axis=axis)
    elif stat == "count":
        return np.sum(~np.isnan(window))
    else:
        raise ValueError(f"Unknown statistic: {stat}")


def rolling_stats(
    array: np.ndarray,
    config: WindowConfig,
    stat: Literal["mean", "var", "std", "min", "max", "sum", "count", "median"],
    strict: bool = False,
    use_numba: bool = True,
) -> ReductionResult:
    """
    Compute rolling window statistics.

    Parameters
    ----------
    array : np.ndarray
        Input array
    config : WindowConfig
        Window configuration
    stat : str
        Statistic to compute
    strict : bool
        Validate exact divisibility
    use_numba : bool
        Use numba parallel kernel for performance

    Returns
    -------
    ReductionResult
    """
    # Use numba kernel for common cases
    if use_numba and stat in ("mean", "var", "std", "min", "max", "sum", "count"):
        stat_map = {
            "mean": 0, "var": 1, "std": 2, "min": 3,
            "max": 4, "sum": 5, "count": 6,
        }
        values = _parallel_window_stats(
            array,
            config.sizes,
            config.strides,
            stat_map[stat],
        )

        # Build metadata
        metadata = {
            "config": config,
            "stat": stat,
            "window_indices": None,  # Could compute if needed
        }
        return ReductionResult(values, metadata)

    # Fallback to generic implementation
    windows, out_shape = rolling_windows(array, config, strict=strict)
    results = np.empty(out_shape)
    metadata = {"config": config, "stat": stat, "windows": []}

    for i, (view, meta) in enumerate(windows):
        idx = np.unravel_index(i, out_shape)
        results[idx] = window_stat(view, stat)
        metadata["windows"].append(meta)

    return ReductionResult(results, metadata)


def blockwise_stats(
    array: np.ndarray,
    block_size: tuple[int, ...],
    stat: Literal["mean", "var", "std", "min", "max", "sum", "count", "median"],
    strict: bool = True,
) -> ReductionResult:
    """
    Compute blockwise (non-overlapping) statistics.

    This is optimized for the common case of tiling an array.
    """
    config = WindowConfig(
        sizes=block_size,
        strides=block_size,
        padding="valid",
    )
    return rolling_stats(array, config, stat, strict=strict)


def merge_stats(
    stats: list[ReductionResult],
    method: Literal["mean", "sum", "welford"] = "mean",
) -> ReductionResult:
    """
    Merge multiple partial statistics using tree reduction.

    Parameters
    ----------
    stats : list of ReductionResult
        Partial results to combine
    method : str
        How to merge: "mean"=average, "sum"=add, "welford"=online merge

    Returns
    -------
    ReductionResult
    """
    if not stats:
        raise ValueError("Cannot merge empty stats list")

    if len(stats) == 1:
        return stats[0]

    if method == "mean":
        values = np.mean([s.values for s in stats], axis=0)
    elif method == "sum":
        values = np.sum([s.values for s in stats], axis=0)
    elif method == "welford":
        # Welford's parallel algorithm
        values = _welford_merge([s.values for s in stats])
    else:
        raise ValueError(f"Unknown merge method: {method}")

    # Combine metadata
    metadata = {
        "merged_from": len(stats),
        "method": method,
        "original_metadata": [s.metadata for s in stats],
    }

    return ReductionResult(values, metadata)


def _welford_merge(values_list: list[np.ndarray]) -> np.ndarray:
    """Merge multiple mean/variance estimates using Welford's algorithm."""
    # For now, simple mean. Full Welford merge would track count and m2
    return np.mean(values_list, axis=0)


def tiled_stats(
    array: np.ndarray,
    config: WindowConfig,
    stat: str,
) -> list[tuple[np.ndarray, dict]]:
    """
    Compute tiled statistics returning mergeable objects.

    Unlike rolling_stats which returns aggregated values,
    this returns per-tile statistics that can be merged later.
    """
    windows, _ = rolling_windows(array, config, strict=True)
    results = []

    for view, meta in windows:
        # Store the full statistic object for later merging
        # For simple stats, we store (value, count) for weighted merging
        value = window_stat(view, stat)
        count = np.prod(view.shape)
        results.append((np.array([value, count]), meta))

    return results


def tree_reduce(
    data: list,
    reduce_fn: Callable,
    arity: int = 2,
) -> any:
    """
    Tree reduction of data using a binary (or n-ary) reduction function.

    Parameters
    ----------
    data : list
        Items to reduce
    reduce_fn : callable
        Function that combines items (e.g., merge_stats)
    arity : int
        Number of items to combine at each step (default 2)

    Returns
    -------
    Reduced result
    """
    if len(data) == 0:
        return None
    if len(data) == 1:
        return data[0]

    while len(data) > 1:
        new_data = []
        for i in range(0, len(data), arity):
            chunk = data[i:i+arity]
            if len(chunk) == 1:
                new_data.append(chunk[0])
            else:
                new_data.append(reduce_fn(chunk))
        data = new_data

    return data[0]
