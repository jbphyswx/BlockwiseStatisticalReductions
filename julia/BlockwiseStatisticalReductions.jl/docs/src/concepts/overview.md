```@meta
CurrentModule = BlockwiseStatisticalReductions
```

# Overview & Theory

## Statistics as mergeable monoids

The whole package rests on one idea: every statistic is a **mergeable monoid over sufficient
statistics**. An *accumulator* is an immutable, isbits struct carrying the sufficient statistics of
a set of observations, with:

- `empty_acc(Acc)` — the identity (empty set),
- `lift(Acc, x)` — the accumulator of a single observation,
- `merge(a, b)` — combine two accumulators (**associative + commutative**),
- `result_value(stat, acc, Tout)` — finalize into the reported statistic.

For example a variance accumulator stores `(n, mean, M2)` and merges by Chan's algorithm; covariance
stores `(n, μx, μy, C)` and merges by Pebay's. Because `merge` carries the *numerators* `M2`/`C`
(not finalized variances), combining sub-blocks is **exact** — identical to computing on the union.

Two consequences make everything else work:

1. **Tree reductions.** A block reduction is a fold of `lift` over its elements; "binary/trinary
   reductions for efficient summation" are just merge trees. The kernels use a pairwise
   divide-and-conquer fold for numerical stability.
2. **Exact cross-scale reuse.** A coarse non-overlapping block is the `merge` of its finer child
   blocks — so coarser scales reuse the finer accumulators rather than re-reading the data.

Numerically, accumulation widens automatically: Float32 input accumulates in Float64
(`accumulation_eltype`) and narrows back to the input eltype on output.

`CompositeAccumulator` packs several statistics over the same field into one product accumulator
that still satisfies the interface, so all requested statistics are produced in a single pass.

## Two regimes

- **Blockwise (non-overlapping):** `stride == window`. Coarse blocks are disjoint unions of finer
  blocks, so the full multi-scale tower (next page) reuses intermediates exactly.
- **Sliding (overlapping):** `stride < window`. Reuse across overlapping windows uses a separable
  two-stack Sliding-Window Aggregation (SWAG): it combines any monoid over a window in O(1)
  amortized per step — no inverse, no double-counting — applied one axis at a time for O(|X|·N),
  independent of window size, and numerically as stable as `merge`. With `stride == window` it
  reduces to the blockwise case.

## Backends

Execution is selected by an [`AbstractExecutionBackend`](@ref), along two orthogonal axes:

- **Local compute:** `SerialBackend`, `ThreadedBackend` (OhMyThreads), `GPUBackend{B}` (planned).
- **Distribution wrapper**, parametric over the inner local backend: `DistributedBackend{Inner}`,
  `MPIBackend{Inner}`.

All backends parallelize over independent output cells (and, for distribution, disjoint output
slabs), so results are **identical to serial** — associativity/commutativity guarantee it. The base
pass (the expensive, O(|X|) part) is what gets parallelized; the coarsening tower is cheap.

## Extensibility

A user adds a statistic by defining an isbits accumulator and a handful of methods, with no changes
to the package — the generic kernels and the whole tower specialize on the new type. `check_monoid`
validates that an accumulator obeys the monoid (and, if claimed, group-inverse) laws.
