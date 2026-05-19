# BlockwiseStatisticalReductions

N-dimensional blockwise and rolling-window statistical reductions for Python and Julia.

This repository contains two implementations:
- **`julia/`** - Julia package with OnlineStats, Dask-like plans, and GPU extensions
- **`python/`** - Python package with Numba, Dask, and Xarray integration

## Features (Both Implementations)

- **Blockwise (Tiled) Reductions**: Non-overlapping window statistics
- **Rolling Window Statistics**: Overlapping windows with configurable stride
- **Tree Reductions**: Hierarchical merge operations for parallel scalability
- **Exact Divisibility Validation**: Strict mode for scientific applications
- **Deduplication**: Cache identical operations in plan graphs
- **Parallel Backends**: CPU, distributed (Dask/Distributed), GPU (CUDA/JAX)

## Julia Package

See [`julia/README.md`](julia/README.md) for details.

### Quick Start
```julia
using BlockwiseStatisticalReductions

# Blockwise mean
data = rand(100, 100)
config = WindowConfig((10, 10))
result = blockwise_stats(data, (10, 10), :mean, strict=true)

# DAG plan with branching
builder = build_plan((100, 100))
branches = fork(builder, 2)  # Horizontal and vertical reductions
# ... configure branches ...
merge_branches!(builder, branches, merge_fn)
plan = finalize_plan(builder)
```

### Key Features
- `OnlineStats.jl` integration for mergeable streaming statistics
- `fork()` / `merge_branches!()` for DAG plan structures
- `tiled_stats()` with tree reduction via `OnlineStats.merge!`
- `validate_window_config()` with strict exact divisibility
- Cache deduplication via semantic (not node ID) hashing

## Python Package

See [`python/README.md`](python/README.md) for details.

### Quick Start
```python
from blockwise_statistical_reductions import blockwise_stats, WindowConfig
import numpy as np

# Blockwise mean
data = np.random.randn(100, 100)
result = blockwise_stats(data, (10, 10), "mean", strict=True)

# Dask integration
import dask.array as da
darr = da.from_array(data, chunks=(50, 50))
result = dask_blockwise_stats(darr, (10, 10), "mean").compute()

# Xarray with coordinate preservation
import xarray as xr
da = xr.DataArray(data, dims=["x", "y"])
result = xr_blockwise_stats(da, {"x": 10, "y": 10}, "mean", strict=True)
```

### Key Features
- Numba JIT-compiled kernels with parallel `prange`
- Dask task graph integration (lazy evaluation)
- Xarray metadata/coordinate preservation
- Flox-accelerated groupby reductions
- Bottleneck fast NaN-aware operations

## Repository Structure

```
BlockwiseStatisticalReductions/
├── julia/                          # Julia implementation
│   ├── src/                        # Source code
│   ├── test/                       # Test suite
│   ├── ext/                        # Package extensions (CUDA, JLD2, OhMyThreads)
│   ├── Project.toml                # Julia package manifest
│   └── README.md                   # Julia-specific docs
├── python/                         # Python implementation
│   ├── src/blockwise_statistical_reductions/
│   ├── tests/                      # pytest test suite
│   ├── pyproject.toml              # Python package config
│   └── README.md                   # Python-specific docs
└── README.md                       # This file
```

## Development

### Julia
```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using BlockwiseStatisticalReductions'
```

### Python
```bash
cd python

# pip
pip install -e ".[dev]"

# uv
uv pip install -e ".[dev]"

# conda (see python/environment.yml)
conda env create -f environment.yml
conda activate bsr-dev

# poetry
poetry install --with dev

pytest
```

## License

MIT License - see individual package directories for details.

