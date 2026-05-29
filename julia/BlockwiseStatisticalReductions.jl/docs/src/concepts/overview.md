# Concepts Overview

BlockwiseStatisticalReductions.jl is built around a small number of composable
abstractions.  Understanding these makes the entire API predictable.

## The reduction pipeline

Every operation in this package follows the same conceptual flow:

```
Input Array → Window Configuration → Reduction Kernel → Output Array
```

For simple cases (one statistic, one scale), you call a convenience function
and the pipeline is invisible.  For complex cases (many scales, many stats,
repeated execution), you explicitly construct a **plan** that wires multiple
steps into a DAG.

## Core abstractions

### 1. WindowConfig

A `WindowConfig{D}` defines *how* to tile the input:

- **sizes** — the block dimensions (e.g., `(4, 4, 1)`)
- **strides** — step between consecutive blocks (== sizes for non-overlapping,
  < sizes for overlapping/rolling)
- **padding** — `:valid` (drop remainder), `:same` (pad to preserve size), `:full`

For blockwise reductions, strides always equal sizes.

### 2. ReductionPlan (DAG)

A `ReductionPlan` is a directed acyclic graph of `WindowNode`s.  Each node
represents one blockwise reduction step.  Edges encode data flow: the output
of one node becomes the input of the next.

The plan is **compiled** once (`finalize_plan`) into a flat
`execution_sequence` — an ordered vector of `ExecutionStep`s with pre-resolved
input indices.  At runtime, no graph traversal or Dict lookup occurs.

### 3. Tower (multi-resolution lattice)

`build_tower_plan` constructs a DAG by BFS-expanding from a base output shape.
At each step it divides each participating dimension by allowed factors,
subject to floor constraints.  The result is a lattice of reachable output
shapes, wired as a DAG that maximizes intermediate reuse.

### 4. Accumulators (mergeable statistics)

`VarianceAccumulator`, `CovarianceAccumulator`, and `RawMomentsAccumulator`
are online statistics that support `O(1)` parallel merge.  This enables
hierarchical reductions: compute statistics on sub-blocks independently, then
merge them — producing the exact same result as computing on the full data.

### 5. Kernels

The lowest level: in-place functions like `blockwise_mean!(out, data,
window_sizes)` that iterate over tiles and write results directly into
pre-allocated output arrays.  All higher-level APIs eventually call these.

## Relationship between layers

```
multiresolution_stats(data, [2,4,8])     ← High-level API
    │
    ├── build_optimal_multires_plan(...)  ← Plan construction
    │       │
    │       └── build_tower_plan(...)     ← BFS + DAG wiring
    │
    └── execute(plan, data)              ← Execution engine
            │
            └── blockwise_mean!(...)     ← Canonical kernel
```

Each layer can be used independently.  You can build plans without executing
them, execute with pre-allocated buffers, or call kernels directly for maximum
control.
