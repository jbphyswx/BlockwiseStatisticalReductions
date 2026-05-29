# DAG-Based Planning

## Why plans?

If you need statistics at 2×, 4×, and 8× coarsening, the naïve approach is
three independent reductions from the full input.  But notice: the 4× result
is just the 2× result reduced by another factor of 2.  A **plan** encodes this
dependency as a DAG, so intermediate results are computed once and reused.

```
Input (128×128×8)
 └─ blockwise_mean ÷2 → (64×64×4)    [output 1]
     └─ blockwise_mean ÷2 → (32×32×2)  [output 2]
         └─ blockwise_mean ÷2 → (16×16×1)  [output 3]
```

Three outputs, but only three reduction steps (not three full passes over the
original data).

## Plan structure

A `ReductionPlan` has:

- **nodes** — `Vector{AbstractPlanNode}`: the operations (usually `WindowNode`s)
- **edges** — `Dict{UInt64, Vector{UInt64}}`: parent → children
- **inputs** — root node IDs (nodes that read from the original data)
- **outputs** — which nodes produce final results
- **execution_sequence** — pre-compiled flat execution order
- **output_indices** — indices into the execution_sequence for output extraction

## Building plans

### Low-level (manual)

```julia
builder = build_plan((128, 128, 8))
# Add a 2× reduction node
window = WindowConfig((2, 2, 2), (2, 2, 2), :valid)
node = WindowNode(window, next_id!(builder))
add_node!(builder, node)
plan = finalize_plan(builder)
```

### High-level (tower builder)

```julia
plan = build_tower_plan((128, 128, 8);
    base_block=(2, 2, 1),
    tower_factors=[2],
    dims=(1, 2))
```

This constructs a multi-level DAG automatically via BFS.

### Convenience (factor-based)

```julia
plan = build_optimal_multires_plan((128, 128, 8), [2, 4, 8], [:mean])
```

## Execution model

After construction, `finalize_plan(builder)` compiles the DAG into a flat
`execution_sequence`:

```julia
struct ExecutionStep
    node::AbstractPlanNode       # What to compute
    input_indices::Vector{Int}   # Where inputs come from (indices into results)
    result_index::Int            # Where to store output
end
```

At runtime, `execute(plan, data)` walks this flat vector in order — no
topological sort, no hash lookups, no graph traversal.  Each step either reads
from the original array (root nodes) or from a previous step's output.

## Zero-allocation execution

For repeated execution (e.g., processing a time series frame by frame):

```julia
plan = build_optimal_multires_plan((128, 128, 8), [2, 4], [:mean])
bufs = allocate_buffers(plan, data)

# Subsequent calls allocate nothing
for frame in frames
    outputs = execute!(plan, bufs, frame)
    process(outputs)
end
```

`allocate_buffers` pre-allocates one output array per execution step.
`execute!` writes into these buffers and returns views — no allocation.

## Node types

| Type | Purpose |
|------|---------|
| `WindowNode` | Blockwise/rolling reduction |
| `StatsNode{S}` | Statistical reduction (mean, var, etc.) |
| `TreeNode` | Tree/pairwise merge |
| `UserNode{F}` | User-supplied function |
| `MergeNode{F}` | Merge multiple branches |
