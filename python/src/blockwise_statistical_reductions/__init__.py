"""
BlockwiseStatisticalReductions: N-dimensional blockwise and rolling-window statistical reductions.

This package provides:
- Blockwise (tiled) statistical reductions
- Rolling window statistics
- Dask integration for parallel/distributed computing
- Xarray integration for labeled arrays with coordinates
- Numba/JAX optimized kernels
"""

__version__ = "0.1.0"

# Core exports
from .core import WindowConfig, ReductionPlan, ReductionResult
from .core import validate_window_config, rolling_windows, blockwise_windows
from .statistics import (
    window_stat,
    rolling_stats,
    blockwise_stats,
    merge_stats,
    tiled_stats,
    tree_reduce,
)

# Backend exports
from .backends import CPUBackend, DaskBackend, DaskDistributedBackend

# Dask integration
from .dask_integration import (
    dask_rolling_stats,
    dask_blockwise_stats,
    dask_tree_reduce,
    create_dask_plan,
    execute_dask_plan,
)

# Xarray integration
from .xarray_integration import (
    xr_rolling_stats,
    xr_blockwise_stats,
    xr_tree_reduce,
    preserve_xr_metadata,
    create_coordinate_ranges,
)

__all__ = [
    # Core
    "WindowConfig",
    "ReductionPlan",
    "ReductionResult",
    "validate_window_config",
    "rolling_windows",
    "blockwise_windows",
    # Statistics
    "window_stat",
    "rolling_stats",
    "blockwise_stats",
    "merge_stats",
    "tiled_stats",
    "tree_reduce",
    # Backends
    "CPUBackend",
    "DaskBackend",
    "DaskDistributedBackend",
    # Dask
    "dask_rolling_stats",
    "dask_blockwise_stats",
    "dask_tree_reduce",
    "create_dask_plan",
    "execute_dask_plan",
    # Xarray
    "xr_rolling_stats",
    "xr_blockwise_stats",
    "xr_tree_reduce",
    "preserve_xr_metadata",
    "create_coordinate_ranges",
]
