"""
Core types and window operations for blockwise statistical reductions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple, TypeVar, Union
import numpy as np
from numba import njit

T = TypeVar("T")


@dataclass(frozen=True)
class WindowConfig:
    """
    Configuration for N-dimensional window operations.

    Parameters
    ----------
    sizes : tuple of int
        Window size in each dimension
    strides : tuple of int, optional
        Step size between windows. Default is same as sizes (non-overlapping).
    padding : {"valid", "same", "full"}
        Padding mode for edge handling
    """

    sizes: Tuple[int, ...]
    strides: Tuple[int, ...] | None = None
    padding: Literal["valid", "same", "full"] = "valid"

    def __post_init__(self):
        if self.strides is None:
            # Default to non-overlapping (blockwise)
            object.__setattr__(self, "strides", self.sizes)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.sizes)

    def is_blockwise(self) -> bool:
        """Check if windows are non-overlapping (stride == size)."""
        return all(s == sz for s, sz in zip(self.strides, self.sizes))


@dataclass
class ReductionResult:
    """
    Result of a reduction operation with metadata.

    Parameters
    ----------
    values : np.ndarray
        Computed statistic values
    metadata : dict
        Window indices, coordinates, etc.
    """

    values: np.ndarray
    metadata: dict

    def __getitem__(self, key):
        """Allow dict-like access to metadata."""
        return self.metadata[key]


class ReductionPlan:
    """
    Graph structure representing a tree of reduction operations.

    Similar to Dask's task graph but specialized for statistical reductions.
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, list[str]] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []

    def add_node(self, name: str, op: str, **kwargs) -> str:
        """Add a node to the plan."""
        self.nodes[name] = {"op": op, **kwargs}
        return name

    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)

    def to_dask_graph(self) -> dict:
        """Convert to Dask graph format."""
        # Dask graph is {key: (func, *args)} or {key: object}
        graph = {}
        for name, node in self.nodes.items():
            if node["op"] == "input":
                graph[name] = node["data"]
            else:
                # Build task tuple: (func, *deps)
                deps = self._get_dependencies(name)
                graph[name] = tuple([node["op"]] + deps)
        return graph

    def _get_dependencies(self, node_name: str) -> list[str]:
        """Get nodes that feed into this node."""
        deps = []
        for src, dsts in self.edges.items():
            if node_name in dsts:
                deps.append(src)
        return deps


def validate_window_config(
    array_shape: Tuple[int, ...],
    config: WindowConfig,
    strict: bool = False,
) -> bool:
    """
    Validate that window configuration divides evenly into array dimensions.

    For blockwise (non-overlapping) windows with "valid" padding:
    - array_shape[i] must be divisible by config.sizes[i]

    For rolling windows with overlap:
    - (array_shape[i] - window_size) must be divisible by stride

    Parameters
    ----------
    array_shape : tuple of int
        Shape of the input array
    config : WindowConfig
        Window configuration
    strict : bool
        If True, raise error for invalid configurations. Otherwise return False.

    Returns
    -------
    bool
        True if valid, False if invalid (only when strict=False)

    Raises
    ------
    ValueError
        If strict=True and dimensions don't divide evenly
    """
    if not strict:
        return True

    for i, (arr_dim, win_dim, stride) in enumerate(
        zip(array_shape, config.sizes, config.strides)
    ):
        if config.padding == "valid":
            if config.is_blockwise():
                # Blockwise: array_dim must be multiple of window_dim
                if arr_dim % win_dim != 0:
                    msg = (
                        f"Blockwise window: dimension {i} has size {arr_dim} "
                        f"which is not divisible by window size {win_dim}. "
                        f"Use padding='same' or 'full', or strict=False."
                    )
                    if strict:
                        raise ValueError(msg)
                    return False
            else:
                # Rolling: (arr_dim - win_dim) must be divisible by stride
                if (arr_dim - win_dim) % stride != 0:
                    msg = (
                        f"Rolling window: dimension {i} with size {arr_dim}, "
                        f"window {win_dim}, stride {stride} does not divide evenly. "
                        f"Final window would be partial."
                    )
                    if strict:
                        raise ValueError(msg)
                    return False
    return True


@njit(cache=True)
def _compute_window_indices(
    shape: Tuple[int, ...],
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: int,  # 0=valid, 1=same, 2=full
) -> Tuple[list, list]:
    """
    Numba-compiled window index computation.

    Returns (start_indices, output_shape)
    """
    ndim = len(shape)
    out_shape = []
    offsets = []

    for i in range(ndim):
        sz = shape[i]
        wsz = window_size[i]
        st = stride[i]

        if padding == 0:  # valid
            out_dim = max(0, (sz - wsz) // st + 1)
            offset = 0
        elif padding == 1:  # same
            out_dim = (sz - 1) // st + 1
            offset = (wsz - 1) // 2
        else:  # full
            out_dim = (sz + wsz - 2) // st + 1
            offset = -(wsz - 1)

        out_shape.append(out_dim)
        offsets.append(offset)

    # Generate all window start positions
    starts = []
    if ndim == 1:
        for i0 in range(out_shape[0]):
            start0 = offsets[0] + i0 * stride[0]
            starts.append((start0,))
    elif ndim == 2:
        for i0 in range(out_shape[0]):
            for i1 in range(out_shape[1]):
                start0 = offsets[0] + i0 * stride[0]
                start1 = offsets[1] + i1 * stride[1]
                starts.append((start0, start1))
    elif ndim == 3:
        for i0 in range(out_shape[0]):
            for i1 in range(out_shape[1]):
                for i2 in range(out_shape[2]):
                    start0 = offsets[0] + i0 * stride[0]
                    start1 = offsets[1] + i1 * stride[1]
                    start2 = offsets[2] + i2 * stride[2]
                    starts.append((start0, start1, start2))

    return starts, out_shape


def rolling_windows(
    array: np.ndarray,
    config: WindowConfig,
    strict: bool = False,
) -> Tuple[list, Tuple[int, ...]]:
    """
    Generate rolling window views with metadata.

    Parameters
    ----------
    array : np.ndarray
        Input array
    config : WindowConfig
        Window configuration
    strict : bool
        Validate exact divisibility

    Returns
    -------
    windows : list of (view, metadata)
        Each element is (window_view, {"indices": [...], "center": ...})
    output_shape : tuple
        Shape of the output grid
    """
    validate_window_config(array.shape, config, strict=strict)

    padding_map = {"valid": 0, "same": 1, "full": 2}
    padding_code = padding_map[config.padding]

    starts, out_shape = _compute_window_indices(
        array.shape,
        config.sizes,
        config.strides,
        padding_code,
    )

    if any(s == 0 for s in out_shape):
        return [], tuple(out_shape)

    windows = []
    for start in starts:
        # Extract window
        slices = tuple(
            slice(max(0, s), min(s + wsz, array.shape[i]))
            for i, (s, wsz) in enumerate(zip(start, config.sizes))
        )
        view = array[slices]

        # Compute center
        center = tuple(s + wsz // 2 for s, wsz in zip(start, config.sizes))

        metadata = {
            "indices": list(start),
            "slices": slices,
            "center": center,
            "shape": view.shape,
        }
        windows.append((view, metadata))

    return windows, tuple(out_shape)


def blockwise_windows(
    array: np.ndarray,
    block_size: Tuple[int, ...],
    strict: bool = True,
) -> Tuple[list, Tuple[int, ...]]:
    """
    Generate non-overlapping blockwise windows.

    Convenience function that creates a WindowConfig with stride=size.

    Parameters
    ----------
    array : np.ndarray
        Input array
    block_size : tuple of int
        Size of each block
    strict : bool
        If True, require exact divisibility (default for blockwise)

    Returns
    -------
    Same as rolling_windows
    """
    config = WindowConfig(
        sizes=block_size,
        strides=block_size,
        padding="valid",
    )
    return rolling_windows(array, config, strict=strict)
