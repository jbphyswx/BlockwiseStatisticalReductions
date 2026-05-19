# BlockwiseStatisticalReductions.jl

Blockwise statistical reductions for Julia with support for multi-resolution computation, numerically stable parallel merges, memory-efficient fused operations, and hardware acceleration.

## Overview

This package provides efficient blockwise and rolling-window statistical reductions for N-dimensional arrays, with focus on:

- **Mergeable statistics**: Numerically stable variance and covariance via Welford/Chan/Pebay algorithms
- **Multi-resolution execution**: Compute statistics at multiple scales with automatic caching
- **Product coarsening**: Compute `<x*y>` and joint moments without intermediate allocations
- **Plan-based execution**: Build and execute reduction DAGs with fork/merge support
- **Hardware acceleration**: SIMD vectorization and GPU kernels (CUDA)
- **Distributed computing**: Multi-node reductions with optimized scheduling
- **Python parity**: Numba kernels, Dask integration, Xarray accessors

## Core Features

### 1. Mergeable Statistics Accumulators

Online statistics accumulators that support numerically stable parallel merges for hierarchical reductions:

```julia
using BlockwiseStatisticalReductions

# Variance accumulator with Welford's algorithm
acc = VarianceAccumulator{Float64}()
fit!(acc, data)
mean_val = Statistics.mean(acc)
var_val = Statistics.var(acc)

# Parallel merge (Chan's algorithm)
acc1 = VarianceAccumulator{Float64}()
fit!(acc1, data1)
acc2 = VarianceAccumulator{Float64}()
fit!(acc2, data2)
merged = merge(acc1, acc2)  # Same as computing on [data1; data2]

# Covariance accumulator (Pebay's algorithm)
acc_xy = CovarianceAccumulator{Float64}()
fit!(acc_xy, x, y)
cov_val = Statistics.cov(acc_xy)

# Raw moments (arbitrary order)
acc_moments = RawMomentsAccumulator{Float64,4}()  # Up to 4th moment
fit!(acc_moments, data)
m1, m2, m3, m4 = acc_moments.moments
```

### 2. Product Coarsening

Compute statistics of products without materializing intermediate arrays:

```julia
# Compute <x*y> without allocating x.*y
result = product_mean(x, y, window_config)

# Compute joint moments (mean, variance, covariance) in one fused pass
moments = product_moments(x, y, window_config)
mean_x = moments.mean_x
var_y = moments.var_y
cov_xy = moments.cov_xy
```

### 3. Multi-Resolution Execution

Compute statistics at multiple scales with automatic caching:

```julia
# Compute variance at 2x, 4x, 8x, 10x reductions
results = execute_cached_multilevel(data, [2, 4, 8, 10], [:variance])

# Access results
var_2x = results[2].data
var_4x = results[4].data

# Or use the high-level API
results = multiresolution_stats(data, [2, 4, 8], stats=[:mean, :variance])
mean_2x = results[2].data[:mean]
var_4x = results[4].data[:variance]
```

### 4. Buffer Pool for Zero-Allocation Reductions

Preallocate reusable buffers for hierarchical reductions:

```julia
# Create pool for multi-resolution factors
pool = create_buffer_pool_for_factors(Float64, (100, 100, 50), [2, 4, 8, 10])

# Acquire buffer for factor 4 level
buf = acquire_level!(pool, 4)  # Shape: (25, 25, 12)

# Use for computation
compute_variance!(buf, data)

# Release back to pool
release_level!(pool, 4, buf)

# Or use with automatic cleanup
result = with_buffer!((100, 100, 50), pool) do buf
    compute_something!(buf, data)
    return sum(buf)
end
```

### 4. Public API Convenience Functions

High-level wrappers for common blockwise operations:

```julia
# Single statistic
means = blockwise_mean(data, (10, 10, 5))
variances = blockwise_variance(data, (10, 10, 5), corrected=true)
stds = blockwise_std(data, (10, 10, 5))

# Multiple statistics in one pass
results = blockwise_stats(data, (10, 10, 5), stats=[:mean, :variance, :min, :max])
mean_result = results[:mean]
var_result = results[:variance]

# Covariance between two fields
covs = blockwise_covariance(ql, w, (10, 10, 5))

# Raw moments up to any order
moments = blockwise_moments(data, (10, 10), 4)  # Up to 4th moment
mean_val = moments[i, j][1]  # First moment
```

### 5. Hybrid Mode (Blockwise + Sliding)

Combine blockwise coarsening with sliding window analysis:

```julia
# Blockwise coarsening followed by sliding analysis on coarsened data
result = hybrid_reduction(data,
    block_sizes=(10, 10, 5),      # 10x10x5 blocks
    sliding_sizes=(3, 3, 3),       # 3x3x3 sliding window
    block_stats=[:mean],            # Mean of each block
    sliding_stats=[:variance]       # Variance in sliding windows
)

# Access results
coarsened = result.block_result.data       # Shape: (10, 10, 10)
sliding_analysis = result.sliding_result.data  # On coarsened data
```

### 4. Window Configurations and Plans

Define reduction windows and build execution plans:

```julia
# Blockwise window (non-overlapping)
window = WindowConfig((10, 10, 5), (10, 10, 5), :valid)

# Rolling window (overlapping)
rolling = WindowConfig((5, 5, 5), (1, 1, 1), :same)

# Build reduction plan
plan = build_plan(window)
plan = add_stats!(plan, :mean)
result = execute(plan, data)
```

## API Reference

### Types

- `VarianceAccumulator{T}` - Mean and variance with parallel merge
- `CovarianceAccumulator{T}` - Covariance with parallel merge  
- `RawMomentsAccumulator{T,N}` - Raw moments up to order N
- `JointMomentsResult` - Container for joint moment computations
- `WindowConfig` - Window specification for reductions
- `ReductionPlan` - DAG-based reduction execution plan
- `BufferPool{T,N}` - Reusable buffer pool
- `LevelBufferPool{T,N}` - Multi-level buffer pool
- `HybridReductionSpec` - Hybrid mode specification
- `HybridReductionResult` - Hybrid mode result container

### Functions

**Accumulators:**
- `fit!(acc, x)` / `fit!(acc, x, y)` - Add samples to accumulator
- `merge(acc1, acc2)` - Parallel merge of accumulators
- `merge_many(counts, means, sum_sq_devs)` - Batch merge
- `Statistics.mean(acc)`, `Statistics.var(acc)`, `Statistics.cov(acc)` - Extract statistics

**Product Coarsening:**
- `product_mean(x, y, window)` - Mean of products without allocation
- `product_moments(x, y, window)` - Joint moments in one pass
- `covariance_from_moments(mean_x, mean_y, mean_xy)` - Covariance from moments identity
- `variance_from_moments(mean, mean_sq)` - Variance from moments identity

**Multi-Resolution:**
- `factor_sequence(start, targets)` - Build factor sequence for caching
- `execute_cached_multilevel(data, factors, stats)` - Execute with caching
- `multiresolution_stats(data, factors; stats=[:mean])` - High-level API

**Public API:**
- `blockwise_mean(data, window_sizes)` - Blockwise mean
- `blockwise_variance(data, window_sizes; corrected=true)` - Blockwise variance
- `blockwise_std(data, window_sizes; corrected=true)` - Blockwise standard deviation
- `blockwise_stats(data, window_sizes; stats=[:mean])` - Multiple statistics
- `blockwise_covariance(x, y, window_sizes)` - Blockwise covariance
- `blockwise_moments(data, window_sizes, max_order)` - Raw moments

**Buffer Pool:**
- `BufferPool{T,N}(max_buffers)` - Create reusable buffer pool
- `acquire!(pool, shape)` - Get buffer from pool (allocates if needed)
- `release!(pool, buffer)` - Return buffer to pool
- `with_buffer!(shape, pool) do ... end` - Automatic buffer cleanup
- `LevelBufferPool` - Multi-level pool for hierarchical reductions
- `create_buffer_pool_for_factors(T, input_shape, factors)` - Configured pool

**Hybrid Mode:**
- `execute_hybrid(data, spec)` - Blockwise + sliding execution
- `hybrid_reduction(data; block_sizes, sliding_sizes, ...)` - Convenience API
- `HybridReductionSpec` - Specification for hybrid workflow

**Hardware Acceleration:**
- `best_blockwise_mean(data, window_sizes)` - Auto-select CPU/GPU
- `simd_blockwise_mean!(out, data, window_sizes)` - SIMD vectorized
- `simd_product_moments!(...)` - Fused joint moments with SIMD
- `cuda_blockwise_mean!`, `cuda_blockwise_variance!` - GPU kernels (requires CUDA.jl)

**Distributed Computing:**
- `distributed_multiresolution_stats(data, factors, stats)` - Multi-node reductions
- `DistributedMultiResScheduler` - Optimized task scheduling
- `create_shared_cache(shape, T)` - Node-local caching with SharedArrays

## Benchmarks

Run the benchmark suite to verify performance meets success criteria:

```julia
include("benchmark/benchmark_suite.jl")
run_benchmarks()
check_success_criteria()
```

**Target performance**: <2x overhead vs hand-optimized for 500×250×127 grids.

## Design Principles

- **No backwards compatibility baggage**: Clean, modern API without legacy cruft
- **Fully qualified imports**: All external package functions use `Package.function()` pattern
- **Zero-allocation hot paths**: Preallocate buffers and fuse loops where possible
- **Numerical stability**: Use Welford/Chan/Pebay algorithms for variance/covariance
- **Test-driven**: Comprehensive unit tests with mathematical invariants

## Implementation Status

**Completed:**
- ✅ Phase 1: Core statistics (Variance/Covariance/RawMoments accumulators)
- ✅ Phase 2: Advanced features (Multi-resolution, Buffer Pool, Hybrid Mode)
- ✅ Phase 3: Public API (blockwise_*, product_*, convenience functions)
- ✅ Phase 4: Python parity (Numba kernels, Dask, Xarray)
- ✅ Phase 5: Testing (797+ Julia tests, Python tests, Integration tests)
- ✅ Phase 6: Optimization (SIMD kernels, GPU extension stubs, Distributed scheduling)

**TODO

TODO: Add NaN support via extension (do we use NaNStatistics? idk the cannonical julia solution these days)
TODO: Consider add note to README to consider use of FixedSizedArrays once it is more advanced.
