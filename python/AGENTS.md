# BlockwiseStatisticalReductions (Python) - Agent Guide

## Import Convention (CRITICAL)

**Use absolute imports within the package.**

### Correct Pattern:
```python
# Within the package, use absolute imports
from blockwise_statistical_reductions.core import WindowConfig
from blockwise_statistical_reductions.statistics import rolling_stats
from blockwise_statistical_reductions.dask_integration import dask_rolling_stats
```

### Forbidden Patterns:
```python
# Avoid relative imports
from .core import WindowConfig  # Use absolute instead
from ..statistics import rolling_stats  # Never use

# Avoid circular imports
# If two modules need each other, refactor common code to a third module
```

## Code Style

- **Type hints**: Use `from __future__ import annotations` for forward references
- **Docstrings**: NumPy-style docstrings with Parameters/Returns sections
- **Line length**: 100 characters (Black default)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes

## Dependencies

### Required:
- `numpy` - Core array operations
- `numba` - JIT compilation for performance
- `dask` / `distributed` - Parallel/distributed computing
- `xarray` - Labeled multi-dimensional arrays
- `flox` - Optimized groupby for xarray
- `bottleneck` - Fast NaN-aware operations
- `pandas` - Time series handling for xarray

### Optional:
- `jax` / `jaxlib` - GPU/TPU acceleration
- `cupy` - CUDA array support

## Testing

- Use `pytest` for all tests
- Tests should be in `tests/` directory
- Use fixtures in `conftest.py` for common test data
- Mark slow tests with `@pytest.mark.slow`
- Use `@pytest.importorskip` for optional dependencies

## Dask Integration Guidelines

1. **Lazy evaluation**: Functions should return Dask arrays, not compute
2. **Chunks**: Respect input chunking; rechunk only when necessary
3. **Graph optimization**: Use `dask.delayed` for custom operations
4. **Flox**: Use `flox.xarray_reduce` for groupby operations when available

## Xarray Integration Guidelines

1. **Metadata preservation**: Always preserve `attrs`, `encoding`, `name`
2. **Coordinate handling**: Update coordinates for reduced dimensions
3. **Units**: Track units in attrs; square for variance, keep for std
4. **Dask backends**: Return DataArrays with Dask arrays for lazy ops

## Performance Guidelines

1. **Numba**: Use `@njit(cache=True)` for pure functions
2. **Parallel`: Use `@njit(parallel=True, prange)` for embarassingly parallel
3. **Bottleneck**: Use for NaN-aware fast paths
4. **Avoid**: Python loops over large arrays; use vectorization or Numba

## Architecture Rules

1. **Separation of concerns**: Core is NumPy-only; Dask/Xarray in separate modules
2. **Backend abstraction**: All execution goes through `ExecutionBackend` subclasses
3. **Validation**: Strict validation (exact divisibility) is opt-in via `strict=True`
4. **Error messages**: Be specific about dimension mismatches and suggest fixes
