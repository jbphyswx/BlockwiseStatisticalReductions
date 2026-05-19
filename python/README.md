# BlockwiseStatisticalReductions (Python)

N-dimensional blockwise and rolling-window statistical reductions with Dask and Xarray support.

## Features

- **Blockwise (Tiled) Reductions**: Non-overlapping window statistics optimized for large arrays
- **Rolling Window Statistics**: Overlapping windows with configurable stride
- **Dask Integration**: Distributed/parallel execution with native task graph support
- **Xarray Integration**: Preserves labeled coordinates and metadata
- **Numba Optimization**: JIT-compiled kernels for performance-critical paths
- **Flox Integration**: Accelerated groupby reductions for Xarray
- **Tree Reductions**: Hierarchical merge operations for parallel scalability

## Installation

### From Source (Development)

**pip:**
```bash
pip install -e .
pip install -e ".[dev]"  # With development dependencies
```

**uv:**
```bash
uv pip install -e .
uv pip install -e ".[dev]"
```

**conda (environment.yml):**
```yaml
# environment.yml
name: bsr-dev
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - numpy>=1.24
  - numba>=0.58
  - dask>=2024.1
  - distributed>=2024.1
  - xarray>=2024.1
  - flox>=0.9
  - bottleneck>=1.3
  - pandas>=2.0
  - pytest>=7.0
  - black>=23.0
  - ruff>=0.1
  - mypy>=1.0
```

**poetry:**
```bash
poetry install
poetry install --with dev
```

### Optional Features

**pip/uv:**
```bash
# JAX support (GPU/TPU)
pip install ".[jax]"
# or: uv pip install ".[jax]"

# All optional dependencies
pip install ".[dev,jax]"
```

**conda:**
```bash
# JAX from conda-forge
conda install -c conda-forge jax jaxlib

# Or modify environment.yml to include:
# - jax
# - jaxlib
```

**poetry:**
```bash
poetry install --extras jax
```

## Quick Start

### Basic Usage

```python
import numpy as np
from blockwise_statistical_reductions import WindowConfig, blockwise_stats, rolling_stats

# Create test data
data = np.random.randn(100, 100)

# Blockwise (non-overlapping) statistics
result = blockwise_stats(data, block_size=(10, 10), stat="mean", strict=True)
print(result.values.shape)  # (10, 10)

# Rolling (overlapping) statistics
config = WindowConfig(sizes=(10, 10), strides=(5, 5))  # 50% overlap
result = rolling_stats(data, config, stat="mean")
```

### Dask Integration

```python
import dask.array as da
from blockwise_statistical_reductions import dask_blockwise_stats

# Create large dask array
data = da.random.randn(10000, 10000, chunks=(1000, 1000))

# Compute blockwise mean (lazy evaluation)
result = dask_blockwise_stats(data, block_size=(100, 100), stat="mean")
computed = result.compute()  # Triggers computation
```

### Xarray Integration

```python
import xarray as xr
import numpy as np
from blockwise_statistical_reductions import xr_blockwise_stats

# Create DataArray with coordinates
da = xr.DataArray(
    np.random.randn(100, 100),
    dims=["lon", "lat"],
    coords={"lon": np.linspace(0, 360, 100), "lat": np.linspace(-90, 90, 100)},
    attrs={"units": "m/s"}
)

# Blockwise mean with coordinate preservation
result = xr_blockwise_stats(da, {"lon": 10, "lat": 10}, "mean", strict=True)
# result has shape (10, 10) with appropriate lon/lat coordinates
```

### Parallel Tree Reduction

```python
from blockwise_statistical_reductions import tree_reduce, blockwise_stats

# Compute partial statistics on chunks
chunks = [blockwise_stats(chunk, (10, 10), "mean") for chunk in data_chunks]

# Tree reduce for scalable merging
final = tree_reduce(chunks, merge_fn=lambda x: merge_stats(x, method="mean"))
```

## API Overview

### Core Types

- `WindowConfig`: Configuration for window sizes, strides, and padding
- `ReductionResult`: Computed statistics with metadata
- `ReductionPlan`: Graph structure for complex reduction pipelines

### Statistics Functions

- `window_stat()`: Single window statistic
- `rolling_stats()`: Rolling window statistics
- `blockwise_stats()`: Non-overlapping block statistics
- `merge_stats()`: Merge multiple partial statistics
- `tiled_stats()`: Blockwise with mergeable output
- `tree_reduce()`: Hierarchical reduction

### Backend Functions

- `CPUBackend`: Single/multi-threaded execution
- `DaskBackend`: Dask local scheduler
- `DaskDistributedBackend`: Dask distributed cluster
- `JAXBackend`: GPU/TPU execution (optional)

### Integration Functions

- `dask_rolling_stats()`: Rolling stats on Dask arrays
- `dask_blockwise_stats()`: Blockwise stats on Dask arrays
- `xr_rolling_stats()`: Rolling stats preserving Xarray metadata
- `xr_blockwise_stats()`: Blockwise stats with coordinate handling

## Testing

Run tests with pytest:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_core.py

# With coverage
pytest --cov=blockwise_statistical_reductions

# Parallel test execution
pytest -n auto
```

## Strict Validation

For scientific applications requiring exact window alignment:

```python
from blockwise_statistical_reductions import validate_window_config

# Raises error if dimensions don't divide evenly
validate_window_config(
    array_shape=(100, 100),
    config=WindowConfig(sizes=(10, 10)),
    strict=True
)

# Or use in operations
result = blockwise_stats(data, (10, 10), "mean", strict=True)  # Raises on mismatch
```

## Architecture

The package follows a layered architecture:

1. **Core Layer** (`core.py`): Window operations, validation, plans
2. **Statistics Layer** (`statistics.py`): Numba-optimized kernels
3. **Backends** (`backends.py`): Execution environment abstraction
4. **Dask Integration** (`dask_integration.py`): Distributed computing
5. **Xarray Integration** (`xarray_integration.py`): Labeled array support

## Dependencies

Required:
- numpy >= 1.24
- numba >= 0.58
- dask >= 2024.1
- xarray >= 2024.1
- flox >= 0.9
- bottleneck >= 1.3

Optional:
- jax/jaxlib (for GPU acceleration)
- cupy (for CUDA arrays)

## License

MIT License - see LICENSE file for details.
