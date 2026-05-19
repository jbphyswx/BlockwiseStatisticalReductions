"""
Xarray integration for labeled arrays with coordinate preservation.

This module provides:
- Rolling and blockwise statistics that preserve xarray metadata
- Coordinate range generation for window positions
- Units preservation via pint-xarray
- Integration with xarray's resample and groupby operations
"""

from __future__ import annotations

from typing import Any
import numpy as np
import xarray as xr
import pandas as pd

from .core import WindowConfig, ReductionResult
from .statistics import rolling_stats, blockwise_stats, merge_stats


def preserve_xr_metadata(
    da: xr.DataArray,
    result_values: np.ndarray,
    new_coords: dict[str, Any],
    window_config: WindowConfig | None = None,
) -> xr.DataArray:
    """
    Preserve xarray metadata when creating result DataArray.

    Preserves:
    - Units (via attrs["units"])
    - Attributes
    - Encoding

    Parameters
    ----------
    da : xr.DataArray
        Original DataArray
    result_values : np.ndarray
        Computed statistic values
    new_coords : dict
        New coordinate values for result dimensions
    window_config : WindowConfig, optional
        Window configuration for metadata

    Returns
    -------
    xr.DataArray
        Result with preserved metadata
    """
    # Create new DataArray
    result = xr.DataArray(
        result_values,
        coords=new_coords,
        dims=list(new_coords.keys()),
        attrs=da.attrs.copy(),
        name=f"{da.name}_stat" if da.name else "stat",
    )

    # Update units if doing mean (same units), variance (squared), etc.
    if "units" in result.attrs:
        units = result.attrs["units"]
        if window_config and hasattr(window_config, "stat"):
            if window_config.stat == "var":
                result.attrs["units"] = f"({units})^2"
            elif window_config.stat == "std":
                result.attrs["units"] = units

    # Copy encoding
    result.encoding = da.encoding.copy()

    # Add metadata about operation
    result.attrs["window_config"] = str(window_config)

    return result


def create_coordinate_ranges(
    original_coords: dict[str, np.ndarray],
    window_size: dict[str, int],
    stride: dict[str, int],
    padding: str = "valid",
) -> dict[str, np.ndarray]:
    """
    Create coordinate ranges for window centers.

    For rolling windows, computes the center position of each window.
    For blockwise, uses the block center.

    Parameters
    ----------
    original_coords : dict
        Original coordinate arrays {dim: values}
    window_size : dict
        Window size per dimension {dim: size}
    stride : dict
        Stride per dimension {dim: stride}
    padding : str
        Padding mode

    Returns
    -------
    dict
        New coordinate arrays for result dimensions
    """
    new_coords = {}

    for dim, orig_vals in original_coords.items():
        ws = window_size.get(dim, 1)
        st = stride.get(dim, ws)
        n_orig = len(orig_vals)

        if padding == "valid":
            n_out = max(0, (n_orig - ws) // st + 1)
            # Centers are at: ws/2, ws/2 + st, ws/2 + 2*st, ...
            centers = np.array([
                orig_vals[min(ws // 2 + i * st, n_orig - 1)]
                for i in range(n_out)
            ])
        elif padding == "same":
            n_out = (n_orig - 1) // st + 1
            centers = np.array([
                orig_vals[min(i * st, n_orig - 1)]
                for i in range(n_out)
            ])
        else:  # full
            n_out = (n_orig + ws - 2) // st + 1
            centers = np.array([
                orig_vals[max(0, min(i * st - ws // 2, n_orig - 1))]
                for i in range(n_out)
            ])

        new_coords[dim] = centers

    return new_coords


def xr_rolling_stats(
    da: xr.DataArray,
    window_dims: dict[str, int],
    stat: str,
    min_periods: int | None = None,
    center: bool = True,
) -> xr.DataArray:
    """
    Compute rolling statistics on xarray DataArray.

    Wrapper around xarray's rolling with our statistical functions.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray
    window_dims : dict
        Window size per dimension {dim: size}
    stat : str
        Statistic (mean, var, std, min, max, sum)
    min_periods : int, optional
        Minimum number of valid observations
    center : bool
        Center windows

    Returns
    -------
    xr.DataArray
        Result with preserved coordinates
    """
    # Use xarray's rolling
    rolling = da.rolling(window_dims, min_periods=min_periods, center=center)

    # Map to our statistics
    stat_map = {
        "mean": rolling.mean,
        "var": rolling.var,
        "std": rolling.std,
        "min": rolling.min,
        "max": rolling.max,
        "sum": rolling.sum,
        "count": rolling.count,
    }

    if stat not in stat_map:
        raise ValueError(f"Unknown statistic: {stat}")

    result = stat_map[stat]()

    # Preserve metadata
    result.attrs = da.attrs.copy()
    result.name = f"{da.name}_{stat}" if da.name else stat

    return result


def xr_blockwise_stats(
    da: xr.DataArray,
    block_dims: dict[str, int],
    stat: str,
    strict: bool = True,
) -> xr.DataArray:
    """
    Compute blockwise (coarsened) statistics on xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray
    block_dims : dict
        Block size per dimension {dim: size}
    stat : str
        Statistic to compute
    strict : bool
        Require exact divisibility

    Returns
    -------
    xr.DataArray
        Coarsened result with new coordinates
    """
    # Check divisibility
    if strict:
        for dim, size in block_dims.items():
            if da.sizes[dim] % size != 0:
                raise ValueError(
                    f"Dimension {dim} size {da.sizes[dim]} not divisible by block size {size}"
                )

    # Use xarray's coarsen
    coarsen = da.coarsen(block_dims)

    stat_map = {
        "mean": coarsen.mean,
        "var": coarsen.var,
        "std": coarsen.std,
        "min": coarsen.min,
        "max": coarsen.max,
        "sum": coarsen.sum,
    }

    if stat not in stat_map:
        # Fallback to our implementation
        config = WindowConfig(
            sizes=tuple(block_dims[d] for d in da.dims),
            strides=tuple(block_dims[d] for d in da.dims),
        )
        result_vals = rolling_stats(da.values, config, stat, strict=strict).values

        # Create new coordinates (centers of blocks)
        new_coords = {}
        for dim in da.dims:
            orig_coord = da.coords[dim].values
            block_size = block_dims[dim]
            n_blocks = len(orig_coord) // block_size
            # Take center of each block
            new_coords[dim] = [
                orig_coord[i * block_size + block_size // 2]
                for i in range(n_blocks)
            ]

        return preserve_xr_metadata(da, result_vals, new_coords, config)

    result = stat_map[stat]()
    result.attrs = da.attrs.copy()
    result.name = f"{da.name}_{stat}_blockwise" if da.name else f"{stat}_blockwise"

    return result


def xr_tree_reduce(
    dataarrays: list[xr.DataArray],
    reduce_fn: callable = None,
    dim: str | None = None,
) -> xr.DataArray:
    """
    Tree reduction over multiple xarray DataArrays.

    Parameters
    ----------
    dataarrays : list of xr.DataArray
        Arrays to reduce (must have compatible coordinates)
    reduce_fn : callable, optional
        Binary reduction function (default: merge_stats)
    dim : str, optional
        Dimension to reduce along (if applicable)

    Returns
    -------
    xr.DataArray
    """
    if reduce_fn is None:
        reduce_fn = lambda x, y: x + y  # Simple merge

    # Validate coordinates match
    first = dataarrays[0]
    for da in dataarrays[1:]:
        for dim_name in first.dims:
            if not np.allclose(da.coords[dim_name].values, first.coords[dim_name].values):
                raise ValueError(f"Coordinates don't match for dimension {dim_name}")

    # Tree reduce
    while len(dataarrays) > 1:
        new_arrays = []
        for i in range(0, len(dataarrays), 2):
            if i + 1 < len(dataarrays):
                merged = reduce_fn(dataarrays[i], dataarrays[i+1])
                new_arrays.append(merged)
            else:
                new_arrays.append(dataarrays[i])
        dataarrays = new_arrays

    result = dataarrays[0]
    result.attrs["tree_reduced_from"] = len(dataarrays)

    return result


def xr_resample_stats(
    da: xr.DataArray,
    time_dim: str,
    freq: str,
    stat: str,
) -> xr.DataArray:
    """
    Compute time-resampled statistics.

    Convenience wrapper around xarray's resample.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with time dimension
    time_dim : str
        Name of time dimension
    freq : str
        Pandas frequency string (e.g., "1D", "6H", "1M")
    stat : str
        Statistic to compute

    Returns
    -------
    xr.DataArray
    """
    resampled = da.resample({time_dim: freq})

    stat_map = {
        "mean": resampled.mean,
        "var": resampled.var,
        "std": resampled.std,
        "min": resampled.min,
        "max": resampled.max,
        "sum": resampled.sum,
        "count": resampled.count,
    }

    if stat not in stat_map:
        raise ValueError(f"Unknown statistic: {stat}")

    return stat_map[stat]()


def add_window_coordinates(
    result_da: xr.DataArray,
    original_da: xr.DataArray,
    window_config: WindowConfig,
) -> xr.DataArray:
    """
    Add window position coordinates to result DataArray.

    Creates new coordinates indicating the start/center/end of each window.

    Parameters
    ----------
    result_da : xr.DataArray
        Result DataArray
    original_da : xr.DataArray
        Original DataArray
    window_config : WindowConfig
        Window configuration

    Returns
    -------
    xr.DataArray
        Result with additional window coordinate variables
    """
    # Create window position coordinates
    for i, dim in enumerate(original_da.dims):
        if dim in result_da.dims:
            orig_coord = original_da.coords[dim].values
            n_out = result_da.sizes[dim]
            ws = window_config.sizes[i]
            st = window_config.strides[i]

            # Window start positions
            starts = [orig_coord[j * st] for j in range(n_out)]
            # Window centers
            centers = [orig_coord[min(j * st + ws // 2, len(orig_coord) - 1)]
                       for j in range(n_out)]
            # Window end positions
            ends = [orig_coord[min((j + 1) * st + ws - 1, len(orig_coord) - 1)]
                    for j in range(n_out)]

            # Add as coordinates
            result_da = result_da.assign_coords({
                f"{dim}_window_start": (dim, starts),
                f"{dim}_window_center": (dim, centers),
                f"{dim}_window_end": (dim, ends),
            })

    return result_da
