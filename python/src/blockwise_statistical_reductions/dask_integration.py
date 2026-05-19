"""
Dask integration for distributed/parallel blockwise reductions.

This module provides:
- Integration with Dask's task graph for lazy evaluation
- Flox-based groupby reductions for xarray integration
- Scatter/gather for distributed execution
- Tree reduction patterns using Dask's native scheduling
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Literal
import numpy as np
import dask.array as da
import dask.bag as db
from dask import delayed
from dask.distributed import Client, as_completed
import flox

from .core import WindowConfig, ReductionPlan
from .statistics import rolling_stats, blockwise_stats, merge_stats


def _make_dask_task_name(base: str, *indices: int) -> str:
    """Generate unique task name for Dask graph."""
    return f"{base}-{'-'.join(map(str, indices))}"


def dask_rolling_stats(
    darr: da.Array,
    config: WindowConfig,
    stat: str,
    split_every: int = 4,
) -> da.Array:
    """
    Compute rolling statistics on a Dask array.

    Uses overlap and map_blocks for out-of-core computation.

    Parameters
    ----------
    darr : dask.array.Array
        Input dask array
    config : WindowConfig
        Window configuration
    stat : str
        Statistic to compute (mean, var, std, min, max, sum)
    split_every : int
        Group size for tree reduction

    Returns
    -------
    dask.array.Array
        Lazy result
    """
    # For rolling windows, we need overlap
    depth = tuple((w // 2, w // 2) for w in config.sizes)

    def _compute_block(block, block_info=None):
        """Process a single block with overlap."""
        # block_info tells us where this block is in global coordinates
        result = rolling_stats(
            block,
            WindowConfig(sizes=config.sizes, strides=config.strides, padding="valid"),
            stat,
            strict=False,
        )
        return result.values

    # Use map_overlap for proper boundary handling
    result = darr.map_overlap(
        _compute_block,
        depth=depth,
        boundary="periodic",  # or "reflect", "nearest", etc.
        trim=True,
    )

    return result


def dask_blockwise_stats(
    darr: da.Array,
    block_size: tuple[int, ...],
    stat: str,
) -> da.Array:
    """
    Compute blockwise (tiled) statistics on Dask array.

    This is more efficient than rolling for non-overlapping blocks as it
    can use Dask's native rechunking.

    Parameters
    ----------
    darr : dask.array.Array
        Input array
    block_size : tuple
        Size of output blocks
    stat : str
        Statistic to compute

    Returns
    -------
    dask.array.Array
        Lazy result with shape (darr.shape / block_size)
    """
    # Rechunk to match block size
    darr = darr.rechunk(block_size)

    def _compute_block(block):
        """Compute statistic for a single block."""
        return blockwise_stats(block, block_size, stat, strict=False).values

    # Use map_blocks for elementwise computation
    # The output shape is 1 per block
    result = darr.map_blocks(
        _compute_block,
        chunks=tuple(1 for _ in block_size),
        drop_axis=list(range(darr.ndim)),
        dtype=np.float64,
    )

    return result


def dask_tree_reduce(
    futures: list,
    reduce_fn: Callable,
    split_every: int = 4,
) -> Any:
    """
    Tree reduction over Dask futures.

    Parameters
    ----------
    futures : list
        Dask futures to reduce
    reduce_fn : callable
        Binary reduction function
    split_every : int
        Fan-in at each level

    Returns
    -------
    Final reduced future
    """
    while len(futures) > 1:
        new_futures = []
        for i in range(0, len(futures), split_every):
            chunk = futures[i:i+split_every]
            if len(chunk) == 1:
                new_futures.append(chunk[0])
            else:
                # Submit tree reduction of this chunk
                reduced = delayed(_tree_reduce_chunk)(chunk, reduce_fn)
                new_futures.append(reduced)
        futures = new_futures

    return futures[0]


def _tree_reduce_chunk(chunk, reduce_fn):
    """Reduce a chunk of results."""
    result = chunk[0]
    for item in chunk[1:]:
        result = reduce_fn([result, item])
    return result


def create_dask_plan(
    input_shape: tuple,
    operations: list[dict],
) -> dict:
    """
    Create a Dask task graph from plan operations.

    Parameters
    ----------
    input_shape : tuple
        Shape of input array
    operations : list of dict
        Each dict has keys: "op", "config", "stat", etc.

    Returns
    -------
    dict
        Dask task graph {key: (func, *args)}
    """
    graph = {}
    prev_key = "input"
    graph[prev_key] = None  # Placeholder, filled at runtime

    for i, op in enumerate(operations):
        key = f"op-{i}"
        config = op.get("config")
        stat = op.get("stat")

        if op["op"] == "window":
            task = (_dask_window_task, prev_key, config)
        elif op["op"] == "stats":
            task = (_dask_stats_task, prev_key, stat)
        elif op["op"] == "merge":
            task = (_dask_merge_task, prev_key)
        else:
            raise ValueError(f"Unknown operation: {op['op']}")

        graph[key] = task
        prev_key = key

    return graph


def _dask_window_task(data, config):
    """Dask task for windowing."""
    from .core import rolling_windows
    windows, shape = rolling_windows(data, config, strict=False)
    return windows


def _dask_stats_task(windows, stat):
    """Dask task for statistics."""
    from .statistics import window_stat
    return [window_stat(w, stat) for w, _ in windows]


def _dask_merge_task(results):
    """Dask task for merging."""
    from .statistics import merge_stats
    return merge_stats(results)


def execute_dask_plan(
    plan: ReductionPlan,
    data: np.ndarray | da.Array,
    scheduler: str | Client = "threads",
) -> Any:
    """
    Execute a ReductionPlan using Dask.

    Parameters
    ----------
    plan : ReductionPlan
        Plan to execute
    data : array
        Input data (numpy or dask array)
    scheduler : str or Client
        Dask scheduler to use

    Returns
    -------
    Result
    """
    # Convert numpy to dask if needed
    if isinstance(data, np.ndarray):
        darr = da.from_array(data, chunks="auto")
    else:
        darr = data

    # Convert plan to Dask graph
    graph = plan.to_dask_graph()

    # Execute
    from dask.threaded import get as threaded_get
    from dask.multiprocessing import get as mp_get

    if scheduler == "threads":
        result = threaded_get(graph, plan.outputs[0])
    elif scheduler == "processes":
        result = mp_get(graph, plan.outputs[0])
    elif isinstance(scheduler, Client):
        # Distributed client
        future = scheduler.get(graph, plan.outputs[0])
        result = future.result()
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    return result


def flox_reduce(
    arr: da.Array,
    groupby_coords: dict[str, da.Array],
    reductions: dict[str, str],
) -> dict[str, da.Array]:
    """
    Use Flox for groupby reductions on Dask arrays.

    Flox provides optimized implementations of groupby reductions
    that are faster than Dask's native groupby.

    Parameters
    ----------
    arr : dask.array.Array
        Input array
    groupby_coords : dict
        Coordinate arrays for grouping (e.g., {"x": x_coords, "y": y_coords})
    reductions : dict
        Mapping of output name to reduction ("mean", "sum", "count", etc.)

    Returns
    -------
    dict of dask.array.Array
    """
    results = {}
    for name, reduction in reductions.items():
        # Use flox's xarray_reduce for efficient groupby
        result = flox.xarray_reduce(
            arr,
            **groupby_coords,
            func=reduction,
        )
        results[name] = result
    return results


def scatter_to_workers(
    client: Client,
    data: np.ndarray,
    n_partitions: int | None = None,
) -> list:
    """
    Scatter data chunks to distributed workers.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client
    data : np.ndarray
        Array to scatter
    n_partitions : int, optional
        Number of partitions (default: n_workers)

    Returns
    -------
    list of futures
    """
    if n_partitions is None:
        n_partitions = len(client.scheduler_info()["workers"])

    # Split array
    chunks = np.array_split(data, n_partitions)

    # Scatter to workers
    futures = client.scatter(chunks, broadcast=False)
    return futures


def gather_from_workers(
    client: Client,
    futures: list,
) -> list:
    """Gather results from distributed workers."""
    return client.gather(futures)
