# BlockwiseStatisticalReductions

N-dimensional blockwise and rolling-window statistical reductions, over **many window sizes at once**.

Two independent implementations live here:

- **`julia/BlockwiseStatisticalReductions.jl/`** — a planned tree of reductions that touches the data as
  close to once as possible, on CPU, GPU, MPI and Distributed.
- **`python/`** — Numba kernels with Dask and Xarray integration.

## Julia Package

![Tile means at four scales, from one pass over the field](julia/BlockwiseStatisticalReductions.jl/docs/src/assets/scales.png)

```julia
using BlockwiseStatisticalReductions

x = randn(4096, 4096)
r = blockstats(x, [2, 4, 8, 16, 32, 64]; stats = (Mean(), Var()))

r[(8, 8)].mean        # 512×512 array of tile means
r[(64, 64)].var       # 64×64 array of tile variances
```

Every statistic is a mergeable monoid over sufficient statistics, so a coarse window is the merge of finer
ones and the sizes you ask for can be built from each other instead of from the input. The planner searches
that space and picks the cheapest tree, inventing intermediate sizes where they let targets share work:

![The plan for five tile sizes, against one pass each](julia/BlockwiseStatisticalReductions.jl/docs/src/assets/plan.png)

Cost then stops growing with the number of scales:

![Cost against the number of scales requested](julia/BlockwiseStatisticalReductions.jl/docs/src/assets/cost.png)

Tiles, overlapping windows at any stride, dense windows and windows at hand-picked anchors are all the
same geometry object, in any number of dimensions and with a different size per axis:

![Tiles, strided windows, anchors and anisotropic windows](julia/BlockwiseStatisticalReductions.jl/docs/src/assets/windows.png)

Count, sum, mean, variance, extrema, raw and central moments, skewness, kurtosis, covariance, correlation
and product mean — statistics of different fields fuse into one pass. Labelled axes, physical window
sizes, weights, NaN skipping, and `DimArray`/NetCDF/Zarr inputs are all supported.

See [`julia/BlockwiseStatisticalReductions.jl/README.md`](julia/BlockwiseStatisticalReductions.jl/README.md)
for details, [`examples/`](julia/BlockwiseStatisticalReductions.jl/examples) for runnable scripts, and
[`MIGRATION.md`](julia/BlockwiseStatisticalReductions.jl/MIGRATION.md) if you used version 0.1.

```julia
import Pkg
Pkg.add(url = "https://github.com/jbphyswx/BlockwiseStatisticalReductions", subdir = "julia/BlockwiseStatisticalReductions.jl")
```

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
├── julia/
│   └── BlockwiseStatisticalReductions.jl/
│       ├── src/                    # statistics · geometry · scales · planner · kernels · execute · api
│       ├── ext/                    # OhMyThreads, KernelAbstractions, CUDA, DimensionalData,
│       │                           #   NCDatasets, Zarr, MPI, Distributed
│       ├── test/                   # one file per layer, plus backend and I/O parity
│       ├── benchmark/              # roofline-relative performance gates
│       ├── docs/                   # Documenter site
│       ├── examples/               # runnable scripts
│       └── gpu/                    # hardware-gated CUDA tests
├── python/
│   ├── src/blockwise_statistical_reductions/
│   ├── tests/                      # pytest test suite
│   └── pyproject.toml
└── README.md                       # this file
```

## Development

### Julia
```bash
cd julia/BlockwiseStatisticalReductions.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'

julia --project=docs docs/make.jl                                   # documentation
julia --project=benchmark benchmark/gates.jl --only=kernels,planner  # performance gates
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

