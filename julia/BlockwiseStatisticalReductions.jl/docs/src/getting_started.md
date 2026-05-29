# Getting Started

## Installation

BlockwiseStatisticalReductions.jl is not yet registered.  Install from the
repository:

```julia
using Pkg
Pkg.add(url="https://github.com/jbphyswx/BlockwiseStatisticalReductions.jl")
```

Or in development mode:

```julia
Pkg.develop(path="/path/to/BlockwiseStatisticalReductions.jl")
```

## Your first reduction

```julia
using BlockwiseStatisticalReductions

# Create some 3D data (e.g., 128×128 horizontal, 8 vertical levels)
data = randn(Float32, 128, 128, 8)

# Compute the mean over 4×4×1 blocks (coarsen x,y by 4, keep z intact)
means = blockwise_mean(data, (4, 4, 1))
# Result: 32×32×8 array
```

The block size `(4, 4, 1)` means: take 4×4×1 non-overlapping tiles and compute
the mean of each tile.  The output shape is `input_shape .÷ block_size`.

## Understanding output shapes

The fundamental invariant:

```
output_size[d] = input_size[d] ÷ block_size[d]
```

This requires `block_size[d]` to evenly divide `input_size[d]` for every
dimension `d`.  If it doesn't divide evenly, the excess elements at the
boundary are dropped (`:valid` padding mode).

| Input shape | Block size | Output shape |
|---|---|---|
| `(128, 128, 8)` | `(4, 4, 1)` | `(32, 32, 8)` |
| `(128, 128, 8)` | `(4, 4, 8)` | `(32, 32, 1)` |
| `(6000, 6000, 64)` | `(100, 100, 4)` | `(60, 60, 16)` |

## Multiple statistics in one pass

When you need both mean and variance, computing them together is faster than
two separate passes because the fused kernel reads data only once:

```julia
results = blockwise_stats(data, (4, 4, 1); stats=[:mean, :variance])
results[:mean]      # 32×32×8
results[:variance]  # 32×32×8
```

Available statistics: `:mean`, `:variance`, `:std`, `:min`, `:max`.

## Multi-resolution in one DAG

If you need statistics at multiple scales (e.g., 2× and 4× and 8×
coarsening), building a plan is far more efficient than running three
independent reductions because the 4× result reuses the 2× intermediate:

```julia
plan = build_optimal_multires_plan(size(data), [2, 4, 8], [:mean])
results = execute(plan, data)

# results is a Vector{ReductionResult}
# results[1].data → 64×64×4  (2× coarsened in all dims except last*)
# results[2].data → 32×32×2  (4×)
# results[3].data → 16×16×1  (8×)
```

!!! note
    By default, `build_optimal_multires_plan` reduces all dimensions except the
    last.  Pass `dims=(1, 2)` to only reduce specific dimensions.

## Controlling which dimensions are reduced

The `dims` keyword specifies which dimensions participate in the reduction:

```julia
# Only reduce x and y (dims 1 and 2), leave z (dim 3) intact
plan = build_optimal_multires_plan((128, 128, 8), [2, 4], [:mean]; dims=(1, 2))
results = execute(plan, data)
# results[1].data → 64×64×8  (z preserved)
```

## What to read next

- **[Window Configurations](concepts/windows.md)** — understand sizes, strides, and padding
- **[Multi-Resolution Towers](concepts/tower.md)** — the BFS lattice and DAG construction
- **[Basic Reductions](howto/basic_reductions.md)** — detailed recipes
- **[Per-Dimension Scaling](howto/per_dimension.md)** — different factors per dimension
