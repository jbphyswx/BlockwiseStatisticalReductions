# How-To: Hybrid Mode (Blockwise + Sliding)

## The problem

Sometimes you want to:
1. Coarsen data via blockwise reduction (reduce resolution)
2. Then analyze spatial structure on the coarsened data via sliding windows

For example: compute the mean over 10×10 blocks, then compute the local
variance in a 3×3 sliding window on the coarsened grid.

## Solution: `hybrid_reduction`

```julia
using BlockwiseStatisticalReductions

data = randn(Float32, 100, 100, 50)

result = hybrid_reduction(data;
    block_sizes   = (10, 10, 5),     # Phase 1: 10×10×5 blockwise coarsening
    sliding_sizes = (3, 3, 3),       # Phase 2: 3×3×3 sliding window
    block_stats   = [:mean],         # Compute mean in each block
    sliding_stats = [:variance])     # Compute variance in sliding window

# Access results
result.block_result.data      # 10×10×10 (coarsened means)
result.sliding_result.data    # Sliding variance on the 10×10×10 grid
```

## Using the spec-based API

For more control, construct a `HybridReductionSpec`:

```julia
spec = HybridReductionSpec(
    WindowConfig((10, 10, 5), (10, 10, 5), :valid),   # Block window
    WindowConfig((3, 3, 3), (1, 1, 1), :same),        # Sliding window
    [:mean],                                            # Block stats
    [:variance]                                         # Sliding stats
)

result = execute_hybrid(data, spec)
```

## Result structure

```julia
struct HybridReductionResult
    block_result::ReductionResult     # Output of phase 1
    sliding_result::ReductionResult   # Output of phase 2 (on coarsened data)
end
```

Both phases produce a `ReductionResult` with `.data` (the array) and `.shape`
(the input shape for that phase).

## Use cases

| Workflow | block_stats | sliding_stats |
|----------|------------|---------------|
| Local variability analysis | `:mean` | `:variance` |
| Edge detection on coarsened data | `:mean` | `:max` - `:min` |
| Smoothed gradient estimation | `:mean` | custom |
| Scale-dependent structure | `:variance` | `:mean` |

## Combining with multi-resolution

You can apply hybrid mode at each level of a multi-resolution tower by
building the tower first, then applying sliding analysis to each output:

```julia
plan = build_optimal_multires_plan((100, 100, 50), [2, 5, 10], [:mean]; dims=(1, 2))
results = execute(plan, data)

# Apply sliding analysis to each level
for r in results
    sliding = blockwise_stats(r.data, (3, 3, 1); stats=[:variance])
    # ... use sliding[:variance] ...
end
```
