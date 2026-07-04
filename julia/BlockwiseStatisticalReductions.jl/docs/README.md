# BlockwiseStatisticalReductions.jl — Status & Roadmap

A purpose-agnostic engine for computing statistics over N-dimensional data at many coarser scales
efficiently, by treating every statistic as a **mergeable monoid** and reusing intermediate results
across scales via a divisor-lattice DAG. See `docs/src/` for the user guide and
`docs/DESIGN_OPTIMIZATION.md` for the performance design.

## Implemented

- **Accumulator algebra** (`src/accumulators/`): isbits accumulators with `empty_acc`/`lift`/`merge`;
  built-ins Count, Sum, Mean (Welford), Var/Std (Chan), Cov (Pébay), Min, Max, RawMoments;
  `CompositeAccumulator` packs several stats over one field binding into a single pass. User statistics
  add a struct + a few methods (see `examples/custom_statistic.jl`).
- **Public API** (`src/api.jl`): `reduce_stats(data, scales; stats)` and the arity-2
  `reduce_stats(x, y, scales; stats)` for covariance. `scales` is a `Tower`, a `Ladder`, a factor
  vector/tuple, or a single factor. Result is a `MultiResResult` keyed by output factor.
- **Gap-filling scale schedules** (`src/schedule.jl`): `scale_ladder(n; seeds, steps, …)` and `Ladder`
  — the multiplicative closure under `steps` (e.g. `steps=[2,3]` → the full 2,3-smooth reduction tree
  1,2,3,4,6,8,9,12,…), handed to the optimal shared DAG.
- **Fast kernels** (`src/kernels/`): a sequential contiguous `blockfold` fallback plus specialized
  whole-block `reduce_block`/`coarsen_block` bulk kernels (public hooks) for the built-ins — Sum/Mean
  match or beat `reshape`+`mean`; two-pass Var/Cov. Sliding-window (SWAG) reductions; non-overlapping
  sliding dispatches to the blockwise bulk path.
- **Divisor-lattice DAG planner** (`src/lattice.jl`, `src/planner.jl`): minimum-work shared DAG
  (gcd-closure Steiner sharing), work-optimal in element-touches; `plan_work`/`naive_work` diagnostics.
- **Prepared reductions** (`src/prepared.jl`): `prepare` + `reduce_stats!` reuse plan + buffers +
  result arrays for an **allocation-free** hot loop (pipelines).
- **Backends**: Serial; Threaded (OhMyThreads ext); Distributed (ext, honors its inner backend);
  GPU via **KernelAbstractions** (ext) — a per-cell `@kernel` validated bit-identically on `CPU()` and
  transpiling to CUDA/ROCm/…; **MPI** (ext, `MPIBackend{Inner}`) — SPMD `Allgatherv` of isbits
  accumulator slabs, validated under `mpiexec`. Backends are bit-identical to serial (independent
  output cells); GPU differs only by floating-point order.
- **Visualization**: `plan_dot(plan)` emits Graphviz DOT of the reduction DAG (dependency-free).
- **Tests**: correctness vs brute force (incl. non-divisible/truncation), type-stability (`@inferred`,
  incl. mixed non-subsuming stats), zero-allocation steady state, and backend parity
  (serial/threaded/distributed bit-identical; GPU-KA on `CPU()`; MPI via `mpiexec`).

## Roadmap / not yet done

- **Real-CUDA end-to-end**: device-resident tower buffers + GPU-side finalize, and a shared-memory /
  warp-shuffle tree kernel for the few-large-blocks regime. Needs GPU hardware to validate (the KA
  kernel logic is already verified on the KA `CPU()` backend).
- **Mixed block + sliding hierarchical plans** (some tower levels overlapping).
- **Anchor-subsampled multi-resolution sliding** as a first-class convenience.
