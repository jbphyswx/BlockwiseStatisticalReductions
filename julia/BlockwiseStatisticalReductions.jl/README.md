# BlockwiseStatisticalReductions.jl

Blockwise statistical reductions for N-dimensional arrays in Julia, with
multi-resolution DAG planning, numerically stable parallel merges,
memory-efficient fused operations, and hardware acceleration.

## Overview

```julia
using BlockwiseStatisticalReductions

# Quick start: blockwise mean of 10×10×5 blocks
means = blockwise_mean(data, (10, 10, 5))

# Multi-resolution tower: compute means at 2×, 4×, 8× in one DAG
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8], [:mean])
results = execute(plan, data)
```

### Key capabilities

- **Mergeable statistics** — Numerically stable variance and covariance via
  Welford/Chan/Pebay algorithms with `O(1)` parallel merge
- **Multi-resolution DAG planning** — `build_tower_plan` constructs a BFS
  lattice of output shapes and wires a DAG that maximizes intermediate reuse,
  with per-dimension factor sets and floor constraints
- **Product coarsening** — Compute `⟨x·y⟩` and joint moments without
  materializing intermediate arrays
- **Zero-allocation execution** — Pre-allocate buffers with `allocate_buffers`,
  then call `execute!` repeatedly with no GC pressure
- **Hardware acceleration** — SIMD kernels, GPU extension stubs (CUDA.jl),
  and distributed multi-node scheduling
- **Hybrid mode** — Compose blockwise coarsening with sliding-window analysis

---

## 1. Public API — Convenience Functions

High-level wrappers for common blockwise operations.  These accept raw arrays
and return results directly — no plan construction needed.

```julia
# Single statistic
means     = blockwise_mean(data, (10, 10, 5))
variances = blockwise_variance(data, (10, 10, 5); corrected=true)
stds      = blockwise_std(data, (10, 10, 5))

# Multiple statistics in one pass (fused mean+variance kernel)
results = blockwise_stats(data, (10, 10, 5); stats=[:mean, :variance, :min, :max])
results[:mean]      # Array
results[:variance]  # Array

# Covariance between two co-located fields
covs = blockwise_covariance(ql, w, (10, 10, 5))

# Raw moments up to any order
moments = blockwise_moments(data, (10, 10), 4)   # NTuple{4,Float64} per block
moments[i, j][1]  # first moment (mean)
moments[i, j][4]  # fourth moment
```

---

## 2. Mergeable Statistics Accumulators

Online accumulators that support numerically stable `O(1)` parallel merge,
suitable for hierarchical reductions, distributed computing, and streaming.

```julia
# Variance accumulator (Welford's algorithm)
acc = VarianceAccumulator{Float64}()
fit!(acc, data)
Statistics.mean(acc)
Statistics.var(acc)

# Parallel merge (Chan's algorithm) — exact as computing on concatenated data
merged = merge(acc1, acc2)

# Covariance accumulator (Pebay's algorithm)
acc_xy = CovarianceAccumulator{Float64}()
fit!(acc_xy, x, y)
Statistics.cov(acc_xy)

# Raw moments (arbitrary order)
acc_m = RawMomentsAccumulator{Float64,4}()
fit!(acc_m, data)
m1, m2, m3, m4 = acc_m.moments
```

---

## 3. Product Coarsening

Compute statistics of products without materializing intermediate arrays.

```julia
# ⟨x·y⟩ without allocating x .* y
result = product_mean(x, y, window_config)

# Joint moments (mean_x, mean_y, var_x, var_y, cov_xy) in one fused pass
jm = product_moments(x, y, window_config)
jm.mean_x; jm.var_y; jm.cov_xy

# Identities for deriving statistics from raw moments
cov_xy = covariance_from_moments(mean_x, mean_y, mean_xy)
var_x  = variance_from_moments(mean_x, mean_x_sq)
```

---

## 4. Multi-Resolution Planning (Tower DAG)

The tower plan builder is the canonical way to compute statistics at multiple
spatial scales.  It uses BFS over the N-dimensional output-shape lattice to
discover all reachable resolutions, then wires a DAG that reuses every
intermediate computation.

### 4.1 Uniform factors across selected dims

```julia
# 128×128×8 input, reduce by 2×, 4×, 8× in dims 1 and 2
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8], [:mean]; dims=(1, 2))
results = execute(plan, data)   # Vector{ReductionResult}
```

### 4.2 Tower plan with explicit base block

```julia
# 6000×6000×8 data, 100×100 base blocks, tower by 2s and 3s
plan = build_tower_plan((6000, 6000, 8);
    base_block  = (100, 100, 1),
    tower_factors = [2, 3],
    stats = [:mean],
    dims  = (1, 2))
# Produces: 60×60×8, 30×30×8, 20×20×8, 10×10×8, 5×5×8, ...
```

### 4.3 Per-dimension factors and floor constraints

Different dimensions can scale with different factor sets and have independent
minimum output size constraints — essential for data with non-square physical
grids (e.g., equal x/y but limited z aggregation).

```julia
# 6000×6000×64 data, base 100×100×4 blocks
# x,y scale by factors of 2 and 3; z only halves; z never goes below 4
plan = build_tower_plan((6000, 6000, 64);
    base_block    = (100, 100, 4),
    tower_factors = ([2, 3], [2, 3], [2]),   # per-dimension factor sets
    min_output_size = (1, 1, 4),             # z floor at 4 cells
    dims = (1, 2, 3))
# Produces shapes like 60×60×16, 30×30×8, 20×20×8, 10×10×4, ...
# z naturally stops reducing once it hits 4
```

### 4.4 Building from target output shapes

```julia
plan = build_tower_plan_from_outputs((600, 600, 8),
    [(60, 60, 8), (30, 30, 8), (10, 10, 8)];
    dims=(1, 2))
```

### 4.5 Factor schedule generation

Build factor towers from seed ladders (useful for experiment sweeps):

```julia
build_factor_schedule(128; seeds=(1,))         # [1, 2, 4, 8, 16, 32, 64, 128]
build_factor_schedule(128; seeds=(1, 3))       # [1, 2, 3, 4, 6, 8, 12, ..., 128]
build_factor_schedule(60; seeds=(1,), min_factor=2, include_full=true)  # [2, 4, 8, 16, 32, 60]
```

### 4.6 High-level convenience

```julia
# Build + execute in one call
results = multiresolution_stats(data, [2, 4, 8]; stats=[:mean])
```

---

## 5. Zero-Allocation Execution

For tight loops or real-time pipelines, pre-allocate all output buffers once
and reuse across calls:

```julia
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8], [:mean])
bufs = allocate_buffers(plan, data)

# First call warms up; subsequent calls allocate nothing
outputs = execute!(plan, bufs, data)

# Feed new data through the same plan+buffers
outputs = execute!(plan, bufs, new_data)
```

---

## 6. Buffer Pool

Manage reusable buffer pools for hierarchical reductions without per-call
allocation:

```julia
pool = create_buffer_pool_for_factors(Float64, (100, 100, 50), [2, 4, 8])

buf = acquire_level!(pool, 4)   # Shape: (25, 25, 12)
# ... use buf ...
release_level!(pool, 4, buf)

# Or with automatic cleanup
result = with_buffer!((100, 100, 50), pool) do buf
    compute_something!(buf, data)
end
```

---

## 7. Hybrid Mode (Blockwise + Sliding)

Compose blockwise coarsening with sliding-window analysis:

```julia
result = hybrid_reduction(data;
    block_sizes   = (10, 10, 5),
    sliding_sizes = (3, 3, 3),
    block_stats   = [:mean],
    sliding_stats = [:variance])

result.block_result.data      # 10×10×10 coarsened
result.sliding_result.data    # sliding variance on coarsened data
```

---

## 8. In-Place Canonical Kernels

Low-level kernels that write directly into pre-allocated output arrays.
These form the hot inner loops that all higher-level APIs call.

```julia
# Blockwise reductions (non-overlapping windows)
blockwise_mean!(out, data, (10, 10, 5))
blockwise_variance!(out, data, (10, 10, 5))
blockwise_mean_variance!(means, variances, data, (10, 10, 5))
blockwise_min!(out, data, (10, 10, 5))
blockwise_max!(out, data, (10, 10, 5))
blockwise_product_mean!(out, x, y, (10, 10, 5))
blockwise_joint_moments!(mean_x, mean_y, mean_xy, x, y, (10, 10, 5))

# Merge kernels (hierarchical sufficient-statistics composition)
blockwise_mean_M2!(mean_out, M2_out, data, window_sizes)
blockwise_merge_mean_M2!(mean_out, M2_out, means, M2s, counts, window_sizes)
blockwise_merge_covariance!(cov_out, mean_x, mean_y, C, counts, window_sizes)
blockwise_merge_raw_moments!(out_moments, moment_arrs, window_sizes)
```

---

## 9. Window Configurations

```julia
# Blockwise (non-overlapping, no padding)
window = WindowConfig((10, 10, 5), (10, 10, 5), :valid)

# Rolling (sliding with stride 1, same-size output)
rolling = WindowConfig((5, 5, 5), (1, 1, 1), :same)

# Strided
strided = WindowConfig((8, 8, 4), (4, 4, 2), :valid)
```

---

## 10. Hardware Acceleration

### SIMD kernels (CPU)

```julia
simd_blockwise_mean!(out, data, window_sizes)
simd_product_moments!(out_mx, out_my, out_mxy, x, y, window_sizes)
```

### GPU kernels (requires CUDA.jl extension)

```julia
cuda_blockwise_mean!(out, data, window_sizes)
cuda_blockwise_variance!(out, data, window_sizes)
```

### Distributed computing

```julia
distributed_multiresolution_stats(data, factors, stats; workers=nworkers())
```

---

## API Reference

### Types

| Type | Description |
|------|-------------|
| `WindowConfig{D}` | D-dimensional window specification (sizes, strides, padding) |
| `ReductionPlan` | DAG of reduction nodes with pre-compiled execution sequence |
| `ReductionResult{T}` | Computed result with shape metadata |
| `ExecutionBuffers{T,N}` | Pre-allocated buffer set for zero-allocation `execute!` |
| `VarianceAccumulator{T}` | Online mean + variance (Welford) with `O(1)` merge |
| `CovarianceAccumulator{T}` | Online covariance (Pebay) with `O(1)` merge |
| `RawMomentsAccumulator{T,K}` | Online raw moments up to order K |
| `JointMomentsResult` | Container for fused joint moment computations |
| `BufferPool{T,N}` | Reusable buffer pool |
| `LevelBufferPool{T,N}` | Multi-level buffer pool for hierarchical reductions |
| `HybridReductionSpec` | Specification for hybrid blockwise + sliding workflow |
| `HybridReductionResult` | Result container for hybrid mode |
| `AbstractExecutionBackend` | Base type (`CPUBackend`, `GPUBackend`, `DistributedBackend`) |
| `AbstractStorage` | Base type (`MemoryStorage`, `DiskStorage`) |
| `PlanCache` | Cache wrapper for plan intermediate results |

### Functions — Public API

| Function | Description |
|----------|-------------|
| `blockwise_mean(data, window_sizes)` | Blockwise mean |
| `blockwise_variance(data, window_sizes; corrected)` | Blockwise variance |
| `blockwise_std(data, window_sizes; corrected)` | Blockwise standard deviation |
| `blockwise_stats(data, window_sizes; stats)` | Multiple statistics (fused) |
| `blockwise_covariance(x, y, window_sizes)` | Blockwise covariance |
| `blockwise_moments(data, window_sizes, max_order)` | Raw moments per block |

### Functions — Multi-Resolution

| Function | Description |
|----------|-------------|
| `build_tower_plan(input_shape; base_block, tower_factors, ...)` | Canonical BFS+DAG builder with per-dim support |
| `build_optimal_multires_plan(input_shape, factors, stats; dims)` | Factor-based wrapper (delegates to `build_tower_plan`) |
| `build_tower_plan_from_outputs(input_shape, targets; ...)` | Build from target output shapes |
| `multiresolution_stats(data, factors; stats)` | Build + execute in one call |
| `seed_factor_ladder(n, seed; min_factor)` | Generate `seed × 2^k` tower |
| `build_factor_schedule(n; seeds, min_factor, include_full)` | Merged multi-seed tower |
| `execute(plan, data; backend, cache)` | Execute a `ReductionPlan` |
| `allocate_buffers(plan, data)` | Pre-allocate output buffers |
| `execute!(plan, buffers, data)` | Zero-allocation execution |

### Functions — Accumulators

| Function | Description |
|----------|-------------|
| `fit!(acc, x)` / `fit!(acc, x, y)` | Add samples |
| `merge(acc1, acc2)` | Parallel merge (Chan/Pebay) |
| `merge_many(counts, means, M2s)` | Batch merge |
| `Statistics.mean(acc)`, `Statistics.var(acc)` | Extract statistics |

### Functions — Product Coarsening

| Function | Description |
|----------|-------------|
| `product_mean(x, y, window)` | `⟨x·y⟩` without intermediate allocation |
| `product_moments(x, y, window)` | Joint moments in one fused pass |
| `covariance_from_moments(mean_x, mean_y, mean_xy)` | Covariance identity |
| `variance_from_moments(mean, mean_sq)` | Variance identity |

---

## Design Principles

- **Fully qualified imports** — All external calls use `Package.function()`;
  no bare `using` or `import` anywhere in source code
- **Zero-allocation hot paths** — `execute!` + `ExecutionBuffers` for
  pre-allocated repeated execution
- **Numerical stability** — Welford/Chan/Pebay algorithms for variance,
  covariance, and higher moments
- **DAG-based planning** — Build once, execute many times; the DAG is
  pre-compiled into a flat execution sequence with no runtime graph traversal
- **Per-dimension generality** — Tower factors, floor constraints, and dims
  selection are all per-dimension; no assumption of square or isotropic grids
- **Test-driven** — 1300+ tests covering mathematical invariants, edge cases,
  and allocation bounds

---

## Roadmap

See [`docs/README.md`](docs/README.md) for the full TODO list and roadmap.
