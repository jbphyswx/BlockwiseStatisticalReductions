# BlockwiseStatisticalReductions.jl — Agent Guide

Purpose-agnostic engine for statistics over N-D data at many coarser scales, reusing intermediates
to touch the data as few times as possible. **No domain concepts ever** (no meteorology / tabular /
masking / IO) — it operates on `AbstractArray`s and accumulators only.

## Conventions (CRITICAL)

- **Imports:** no bare `using`/`import`. Use `using X: X as XX` (or `using X: X`) and qualify every
  call (`OMT.tforeach`). Same in examples/tests.
- **Extension naming:** every extension is `BlockwiseStatisticalReductions<Trigger>Ext` (file =
  module = key in `[extensions]`). This is a Julia requirement — unprefixed names collide across
  packages in one session.
- **Naming:** name things by function; avoid vague catch-alls ("bundle", "slab"). The composite
  accumulator is `CompositeAccumulator`, not `Bundle`.
- **Type hygiene:** parametrize the data path; keep accumulators immutable + isbits; `NTuple{N,Int}`
  not `Tuple`; no `Union{T,Nothing}` state. Don't over-concretize containers (`AbstractArray` at
  boundaries, concrete isbits at the element).

## Architecture (the layers, finest → coarsest)

1. **Accumulator algebra** (`src/accumulators/`). The whole package rests on the mergeable-monoid
   idea: a statistic = `empty_acc` / `lift` / `merge` (+ optional `inverse_merge`, `value`).
   - `interface.jl`: `AbstractAccumulator`, `AbstractStatistic`, traits, `accumulation_eltype`
     (Float32→Float64 widening rule), `check_monoid` (verifies the laws — use it on new accumulators).
   - `builtin.jl`: `CountAcc`/`SumAcc`/`MeanAcc`/`VarAcc`/`CovAcc`/`MinAcc`/`MaxAcc`/`RawMomentsAcc`
     and their tags (`Count`/`Sum`/`Mean`/`Var`/`Std`/`Cov`/`Min`/`Max`/`Moments`). Welford lift,
     Chan (variance) / Pebay (covariance) merge, additive power sums (exact). `subsumes` lets one
     accumulator serve several stats (a `VarAcc` gives mean/sum/count too).
   - `composite.jl`: `CompositeAccumulator` — a product accumulator (several stats over the same
     field) that itself satisfies the accumulator interface, so kernels stay generic.
2. **Kernels** (`src/kernels/`). Generic over `Acc`, zero-alloc, type-stable.
   - `fold.jl`: `treefold` — pairwise (divide-and-conquer) merge over a box (stable + SIMD-able).
   - `block.jl`: `blockreduce!` (base pass from data) and `coarsen!` (merge finer accumulators).
     Both dispatch the cell loop to `_drive_base!`/`_drive_merge!(…, backend)`; serial lives here,
     other backends add methods in extensions.
   - `sliding.jl`: overlapping windows via separable two-stack SWAG (`sliding_reduce`) — any monoid,
     O(|X|·N), no inverse, exact; `stride == window` reproduces blockwise.
3. **Planner** (`src/lattice.jl`, `src/planner.jl`). Work in *factor* (block-size) space: reuse
   `p→c` iff `f_p | f_c`; merge window `f_c ./ f_p`; floor-division composes so reuse is exact.
   `tower_plan` enumerates a tower; `solver_plan` builds a min-work DAG for arbitrary targets with
   Steiner (gcd-closure) sharing. Optimal parent = largest materialized divisor. `ReductionPlan` is
   pure geometry (independent of the statistic/eltype).
4. **Execution** (`src/buffers.jl`, `src/execute.jl`). `allocate_tower(plan, Acc)` →
   `TowerBuffers`; `run!(buf, plan, inputs, backend)` walks the topo-ordered steps (base ⇒
   `blockreduce!`, else `coarsen!`). Reuse `buf` across calls for 0 allocations. `materialize`
   finalizes accumulator arrays into statistic arrays.
5. **API** (`src/api.jl`). `reduce_stats(data[, y], scales; stats, backend)` →
   `MultiResResult` keyed by output factor. Scales: `Tower`, vector of factor tuples / ints, single
   factor, or `Sliding`. Assembles the composite accumulator + routing, executes behind a
   type-stable barrier.
6. **Backends** (`src/backends.jl`). `AbstractExecutionBackend`: `SerialBackend`,
   `ThreadedBackend` (OhMyThreads ext), `GPUBackend{B}` (KA/CUDA ext, planned),
   `DistributedBackend{Inner}`/`MPIBackend{Inner}` (Distributed ext), `AutoBackend`. Two axes: local
   compute × distribution wrapper. Add a backend by adding `_drive_base!`/`_drive_merge!` methods
   (local) or a `run!` method (distribution) in an extension.

## Invariants to preserve

- Base reduction and cross-scale coarsening call the **same** `merge` ⇒ multi-scale results are
  bit-for-bit equal to a direct reduction, and parallel/distributed paths can't diverge.
- Variance/covariance flow as `M2`/`C` (numerators) through the tower; finalize (÷n or ÷(n−1)) only
  at requested outputs. Never average child variances.
- `run!` is allocation-free at steady state for every accumulator (assert with `@allocated == 0`).
- Results are independent of backend and of merge order (associative + commutative).

## Testing

`Pkg.test()` (add `julia_args=["-t4"]` to exercise threading). Tests live in `test/` with shared
helpers in `testutils.jl`; every statistic is checked against a brute-force reference across eltypes,
dimensions, truncation, and edge cases, plus type-stability (`@inferred`) and zero-alloc.
