# Migrating from 0.1

Version 0.2 is a ground-up rebuild. Nothing from 0.1 is deprecated — it is gone — so this note maps the
old surface onto the new one. The concepts survived; the names, the layering and the planner did not.

## Entry points

| 0.1 | 0.2 |
|---|---|
| `reduce_stats(x, scales; stats, backend)` | `blockstats(x, scales; stats, backend)` |
| `reduce_stats(x, y, scales; stats)` | `blockstats((x = x, y = y), scales; stats = (Cov(:x, :y),))` |
| `prepare(x, scales; …)` → `PreparedReduction` | `prepare(x, scales; …)` → `Prepared` |
| `reduce_stats!(p, x)` | `blockstats!(p, x)` |
| `materialize(result)` / `materialize!` | not needed — results are already arrays |
| `MultiResResult` | `ScaleResults` |
| `factors(r)` | `scales(r)` |
| `shapes(r)` | `shapes(r)` |
| `stat_name(tag)` | name the statistics yourself: `stats = (mean_u = Mean(:u),)` |

`ScaleResults` indexes by size tuple as before (`r[(8, 8)]`), and also by the full window and by position.
It adds `windows(r)`, `statnames(r)`, `geometry(r, key)`, `dimnames(r)` and `spacing(r)`.

## Asking for scales

0.1 had three separate mechanisms. 0.2 has one: every request resolves to a list of windows.

| 0.1 | 0.2 |
|---|---|
| `[2, 4, 8]` (isotropic factors) | `[2, 4, 8]` — unchanged |
| `Tower(; base_factor = 2, steps = [2], maxfactor = 64)` | `Dyadic(; min = 2, max = 64)`, or just `[2, 4, 8, 16, 32, 64]` |
| `Ladder` / `scale_ladder(…)` | `ScaleSet(Smooth((2, 3)); …)`, or `Subsample(gen, budget)` to thin any generator |
| `Sliding(w; stride, origin)` | `ScaleSet(Sizes([w]); placement = Stride(stride))` |
| — | `placement = Tiled() / Stride(k) / Overlap(frac) / Dense() / Anchors(v) / Spread(k)` |
| — | `edge = Truncate() / Partial() / Centered() / Strict()` |
| — | sizes in physical units: `Dyadic(; max = Length(500.0))` with `spacing` |

A sliding window is no longer a separate code path: tiles are the `stride == size` case of the same
geometry, and the planner spans both from one derivation graph.

## Statistics

The tags kept their names — `Count`, `Sum`, `Mean`, `Var`, `Std`, `Cov`, `Min`, `Max`, `Moments` — and
gained `Extrema`, `Corr`, `ProductMean`, `CentralMoments`, `Skewness`, `Kurtosis` and `Component`.

`Component(tag, field)` replaces the custom tags 0.1 users wrote to reach raw accumulator fields:
`Component(Var(), :M2)` is the variance numerator, `Component(Cov(:u, :w), :C)` the covariance numerator.

Statistics bind fields by name or position, so one pass computes `Var(:u)`, `Var(:w)` and `Cov(:u, :w)`
together instead of three passes.

## Accumulator interface

| 0.1 | 0.2 |
|---|---|
| `empty_acc(A)` | `neutral(A)` |
| `inverse_merge(ab, b)` | `unmerge(ab, b)` |
| `result_value(tag, a, T)` | `finalize(tag, a, T)` |
| `default_output_eltype(tag, T)` | `result_eltype(tag, T)` |
| `stat_arity(tag)` / `arity(A)` | `arity(A)` |
| `CompositeAccumulator{Tuple{…}}` | `Composite{Members,Bindings}` |
| `Array{Acc,N}` storage | `AccumulatorArray` (struct-of-arrays; constant components cost no memory) |

New in the interface: the two-phase `combine` protocol (`phases`, `p1lift`, `mid`, `p2lift`, `finish`)
that lets a composite of any number of members cost at most two passes over a box, and `unshift` /
`shiftable` for shifted accumulation.

## Backends

0.1 defined its own backend types. 0.2 dispatches on
[ComputationalBackends.jl](https://github.com/jbphyswx/ComputationalBackends.jl):

```julia
using ComputationalBackends: ComputationalBackends as CB
blockstats(x, [8]; stats = (Mean(),), backend = CB.ThreadedBackend())
```

`CB.AutoBackend()` is the default and picks a GPU backend for device arrays, threads when more than one
is available, else serial.

MPI changed shape: 0.1 had every rank hold the whole input and `Allgatherv!` the results. 0.2 takes a
partitioned tensor — each rank passes its own slab as `Partitioned(slab; axis = d)` — and results stay
partitioned by owning rank.

## Things that no longer exist

- `ReductionPlan` / `ReductionStep` / `tower_plan` / `solver_plan`: the planner is `plan(shape, targets; …)`
  and produces a `Plan` of derivations (`Base_`, `Coarsen`, `Compose`, `Restride`, `Scan`).
- `plan_work` / `naive_work` / `n_base_passes` / `total_work`: `explain(p)` reports passes, bytes, merges
  and peak memory.
- `TowerBuffers` / `allocate_tower`: `Workspace` with liveness-based buffer reuse.
- `sliding_reduce`: sliding windows go through the same `blockstats` as everything else.
