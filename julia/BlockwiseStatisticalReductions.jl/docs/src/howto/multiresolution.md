# How-To: Multi-Resolution Analysis

## Why multi-resolution?

Climate and turbulence data exhibit structure at multiple scales.  Computing
statistics at 2×, 4×, 8×, 16× coarsening reveals how variability changes with
scale — essential for sub-grid parameterization, scale analysis, and
structural error quantification.

## Quick approach: `multiresolution_stats`

One-liner that builds and executes a plan:

```julia
using BlockwiseStatisticalReductions

data = randn(Float32, 128, 128, 8)
results = multiresolution_stats(data, [2, 4, 8]; stats=[:mean])
# results is a Vector{ReductionResult}
# results[1].data → 64×64×4
# results[2].data → 32×32×2
# results[3].data → 16×16×1
```

## Build a plan for repeated use

If you process many frames with the same resolution structure, build the plan
once:

```julia
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8, 16], [:mean])

# Execute on different data
for frame in frames
    results = execute(plan, frame)
    # process results...
end
```

## Control which dimensions are reduced

```julia
# Only reduce dims 1 and 2, preserve dim 3
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8], [:mean]; dims=(1, 2))
results = execute(plan, data)
# results[1].data → 64×64×8  (z preserved)
# results[2].data → 32×32×8
# results[3].data → 16×16×8
```

## Use explicit tower construction

For full control over the DAG structure:

```julia
plan = build_tower_plan((6000, 6000, 8);
    base_block  = (100, 100, 1),     # Finest coarsening: 100×100 blocks
    tower_factors = [2, 3],          # Build coarser levels by 2 and 3
    stats = [:mean],
    dims  = (1, 2),
    include_base = true,             # Include the 60×60 base level
    include_full = true)             # Include full-domain reduction if reachable
```

## Generate factor schedules for experiment sweeps

If you want a dense set of scales:

```julia
factors = build_factor_schedule(128; seeds=(1, 3), min_factor=2)
# [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

plan = build_optimal_multires_plan((128, 128, 8), factors, [:mean]; dims=(1, 2))
```

## Build from target output shapes

If you know exactly what output shapes you want:

```julia
plan = build_tower_plan_from_outputs((600, 600, 8),
    [(60, 60, 8), (30, 30, 8), (10, 10, 8)];
    stats=[:mean],
    dims=(1, 2))
```

## Inspect plan structure

```julia
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8], [:mean])

length(plan.nodes)               # Number of reduction nodes
length(plan.execution_sequence)  # Number of execution steps
length(plan.output_indices)      # Number of outputs
```

## Performance notes

- The DAG ensures each intermediate is computed only once.  Computing 2×, 4×,
  8× via the DAG does 3 reduction steps; independently it would be
  3 full passes over the data.
- For large arrays, the savings are proportional to the number of levels.
- The first `execute` call after plan construction compiles the execution
  sequence; subsequent calls are fast.
