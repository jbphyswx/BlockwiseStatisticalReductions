# How-To: Per-Dimension Scaling

## The problem

Real-world data often has non-square physical grids where dimensions should
scale differently:

- **Climate columns**: 128×128 horizontal × 64 vertical — you want aggressive
  x/y coarsening but never want to aggregate entire vertical extents
- **Long channels**: 1024×64×64 — the streamwise direction might scale by
  large factors while cross-stream stays moderate
- **Satellite swaths**: 2048×512 — along-track and cross-track have different
  resolutions and should be coarsened independently

## Solution: per-dimension tower factors and floor constraints

`build_tower_plan` accepts tower factors as an NTuple of vectors (one per
dimension) and `min_output_size` as an NTuple of minimum allowed output sizes.

### Example: Climate data with capped z

```julia
using BlockwiseStatisticalReductions

# 6000×6000 horizontal, 64 vertical levels
input_shape = (6000, 6000, 64)

plan = build_tower_plan(input_shape;
    base_block    = (100, 100, 4),        # Base: 100×100 in x,y; 4 in z
    tower_factors = ([2, 3], [2, 3], [2]),  # x,y scale by 2 and 3; z only halves
    min_output_size = (1, 1, 4),          # z never goes below 4 cells
    dims = (1, 2, 3),                     # All dims participate
    stats = [:mean])

results = execute(plan, data)
```

This produces output shapes like:
- `(60, 60, 16)` — base output
- `(30, 30, 8)` — x,y halved, z halved
- `(20, 20, 8)` — x,y reduced by 3, z stays at 8
- `(10, 10, 4)` — further coarsening, z hits floor
- `(5, 5, 4)` — x,y continue, z stays at 4 (floor reached)

### Example: Equal x and y, no z reduction

```julia
plan = build_tower_plan((6000, 6000, 64);
    base_block    = (100, 100, 1),      # z block = 1 means no z reduction
    tower_factors = [2, 3],             # Uniform factors for x,y
    dims = (1, 2),                      # Only reduce x and y
    stats = [:mean])
```

Since dim 3 is not in `dims`, it's completely untouched — every output has
z = 64.

### Example: Asymmetric channel flow

```julia
# 1024 streamwise × 64 spanwise × 64 wall-normal
plan = build_tower_plan((1024, 64, 64);
    base_block    = (8, 4, 4),
    tower_factors = ([2, 4], [2], [2]),  # Streamwise scales faster
    min_output_size = (1, 2, 2),         # Keep at least 2 cells cross-stream
    dims = (1, 2, 3),
    stats = [:mean, :variance])
```

## How the BFS handles per-dimension factors

The BFS expands from the base output shape by trying each factor in each
dimension independently, plus combined multi-dimension steps for shared
factors.  For the climate example above with base output `(60, 60, 16)`:

**Single-dimension steps:**
- Dim 1 ÷ 2 → `(30, 60, 16)`
- Dim 1 ÷ 3 → `(20, 60, 16)`
- Dim 2 ÷ 2 → `(60, 30, 16)`
- Dim 3 ÷ 2 → `(60, 60, 8)`

**Combined steps (factor 2 shared by dims 1, 2, 3):**
- Dims 1,2 ÷ 2 → `(30, 30, 16)`
- Dims 1,2,3 ÷ 2 → `(30, 30, 8)`
- Dims 1,3 ÷ 2 → `(30, 60, 8)`
- Dims 2,3 ÷ 2 → `(60, 30, 8)`

Combined steps produce single-kernel reductions (e.g., a `(2,2,2)` window)
which are faster than sequential single-dim reductions.

## Floor constraint enforcement

If `min_output_size = (1, 1, 4)` and the BFS tries to reduce dim 3 from 4 to
2, the step is **skipped** — that child shape is never added to the queue.
All downstream shapes that would depend on it are also unreachable.

This means the floor is a hard bound: no output in the plan will ever have
fewer than `min_output_size[d]` cells in dimension `d`.

## Selecting only specific outputs

If the BFS generates more shapes than you need, filter with
`target_output_sizes`:

```julia
plan = build_tower_plan((6000, 6000, 64);
    base_block    = (100, 100, 4),
    tower_factors = ([2, 3], [2, 3], [2]),
    min_output_size = (1, 1, 4),
    dims = (1, 2, 3),
    target_output_sizes = [(30, 30, 8), (10, 10, 4)])
# Only these two shapes appear as plan outputs;
# intermediates stay in the DAG for reuse but aren't exposed
```
