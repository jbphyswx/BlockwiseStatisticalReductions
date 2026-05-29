# BlockwiseStatisticalReductions.jl

**Blockwise statistical reductions for N-dimensional arrays** — with
multi-resolution DAG planning, numerically stable parallel merges,
memory-efficient fused operations, and hardware acceleration.

## What is this package?

BlockwiseStatisticalReductions.jl computes statistics (mean, variance,
covariance, higher moments) over non-overlapping blocks of N-dimensional
arrays.  It goes far beyond a simple loop:

- **Multi-resolution DAGs** — compute statistics at many scales in one pass,
  reusing intermediate results via a directed acyclic graph
- **Per-dimension control** — different factor sets and floor constraints per
  dimension (e.g., aggressive x/y scaling but capped z aggregation)
- **Numerically stable merges** — Welford/Chan/Pebay algorithms allow correct
  hierarchical combination of block statistics
- **Zero-allocation hot paths** — pre-allocate once, execute repeatedly with
  no GC pressure
- **Product coarsening** — compute `⟨x·y⟩` and joint moments without
  materializing intermediate arrays

## Quick Start

```julia
using BlockwiseStatisticalReductions

# Simple blockwise mean
data = randn(Float32, 128, 128, 64)
means = blockwise_mean(data, (4, 4, 4))  # → 32×32×16 array

# Multiple stats in one pass
results = blockwise_stats(data, (8, 8, 8); stats=[:mean, :variance])
results[:mean]      # 16×16×8
results[:variance]  # 16×16×8

# Multi-resolution: compute means at 2×, 4×, 8× coarsening
plan = build_optimal_multires_plan(size(data), [2, 4, 8], [:mean])
outputs = execute(plan, data)
# outputs[1].data is 64×64×32 (2× coarsened)
# outputs[2].data is 32×32×16 (4× coarsened)
# outputs[3].data is 16×16×8  (8× coarsened)
```

## Package structure

```
BlockwiseStatisticalReductions.jl
├── Public API          blockwise_mean, blockwise_stats, multiresolution_stats, ...
├── Plan Building       build_tower_plan, build_optimal_multires_plan, ...
├── Execution Engine    execute, execute!, allocate_buffers, ...
├── Canonical Kernels   blockwise_mean!, blockwise_variance!, merge kernels, ...
├── Accumulators        VarianceAccumulator, CovarianceAccumulator, ...
├── Buffer Pool         BufferPool, LevelBufferPool, acquire!, release!, ...
└── Hybrid Mode         hybrid_reduction, HybridReductionSpec, ...
```

## Next steps

- **[Getting Started](@ref)** — Installation, first reduction, understanding output shapes
- **[Concepts](@ref)** — Windows, plans, towers, numerical stability
- **[How-To Guides](@ref)** — Task-oriented recipes for common workflows
- **[API Reference](@ref)** — Complete function and type documentation
