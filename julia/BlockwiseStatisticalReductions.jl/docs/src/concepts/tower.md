# Multi-Resolution Towers

The tower plan builder (`build_tower_plan`) is the canonical engine for
constructing multi-resolution DAGs.  It uses BFS over the N-dimensional
output-shape lattice to discover all reachable resolutions, then wires a DAG
that maximizes intermediate reuse.

## Core idea

Starting from a base output shape (determined by `base_block`), the builder
asks: "what coarser shapes can I reach by dividing any dimension by any
allowed factor?"  It repeats this question (BFS) until no further divisions
are possible (either the dimension hits the floor constraint or no factor
divides evenly).

The result is a **lattice** of reachable output shapes.  Each shape becomes a
node in the DAG, with edges from parent (finer) to child (coarser).

## BFS lattice expansion

```
Base output: (60, 60, 16)
    factors for dims 1,2: [2, 3]
    factors for dim 3:    [2]
    min_output_size:      (1, 1, 4)

BFS discovers:
    (60, 60, 16) → base
    (30, 60, 16) → dim 1 ÷ 2
    (60, 30, 16) → dim 2 ÷ 2
    (30, 30, 16) → dims 1,2 ÷ 2 simultaneously
    (60, 60, 8)  → dim 3 ÷ 2
    (30, 30, 8)  → dims 1,2 ÷ 2 then dim 3 ÷ 2
    (20, 20, 8)  → from (60,60,8) by ÷3 in dims 1,2
    (10, 10, 4)  → from (20,20,8) by ÷2 in dims 1,2 and dim 3
    ...
    z stops at 4 due to min_output_size constraint
```

## DAG wiring

After BFS, the builder sorts shapes from finest to coarsest (by total cell
count) and wires each shape to its **best parent** — the finest shape that
can reach it via a single factor application.  This maximizes reuse: instead
of re-reducing from the input every time, each coarser level builds on the
finest available intermediate.

## Per-dimension control

### Different factor sets

```julia
tower_factors = ([2, 3], [2, 3], [2])
```

Dims 1 and 2 can be divided by 2 or 3; dim 3 can only be halved.  This models
data where horizontal resolution can be aggressively coarsened but vertical
aggregation should be conservative.

### Floor constraints

```julia
min_output_size = (1, 1, 4)
```

Dim 3 will never be reduced below 4 cells.  The BFS simply skips any child
shape where `child[d] < min_output_size[d]`.

### Combined steps

The BFS generates both single-dimension and multi-dimension steps.  For
example, if both dim 1 and dim 2 can be halved, it generates:
- `(30, 60, 16)` — only dim 1 halved
- `(60, 30, 16)` — only dim 2 halved
- `(30, 30, 16)` — both halved simultaneously (one WindowNode with `(2,2,1)`)

The combined step is crucial for square grids: halving x and y together in one
kernel call is faster than doing them sequentially.

## Selecting outputs

Not every reachable shape needs to be a plan output.  The
`target_output_sizes` parameter controls which shapes appear in
`plan.outputs`:

```julia
# Only expose specific shapes as outputs; intermediates stay for reuse
plan = build_tower_plan((128, 128, 8);
    base_block=(2, 2, 1),
    tower_factors=[2],
    dims=(1, 2),
    target_output_sizes=[(32, 32, 8), (16, 16, 8)])
```

`build_optimal_multires_plan` uses this internally: it converts user-specified
factors to target output shapes and passes them through.

## Factor schedules

`build_factor_schedule` generates structured sets of reduction factors from
seed values:

```julia
build_factor_schedule(128; seeds=(1,))
# [1, 2, 4, 8, 16, 32, 64, 128]  — powers of 2

build_factor_schedule(128; seeds=(1, 3))
# [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]  — interleaved

build_factor_schedule(60; seeds=(1,), min_factor=4)
# [4, 8, 16, 32, 60]
```

Each seed generates a ladder `seed × 2^k`.  Multiple seeds are merged and
deduplicated.  This is useful for experiment sweeps where you want a dense set
of scales without manually listing every factor.

## Relationship to other builders

```
build_tower_plan          ← The engine (BFS + DAG)
    ▲
    │ delegates to
    │
build_optimal_multires_plan  ← Convenience (factors → tower)
    ▲
    │ calls
    │
multiresolution_stats     ← One-liner (build + execute)
build_tower_plan_from_outputs  ← From target shapes
```
