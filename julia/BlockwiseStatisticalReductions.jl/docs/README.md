# BlockwiseStatisticalReductions.jl — Roadmap & TODOs

## Completed

- [x] Core statistics accumulators (Variance, Covariance, RawMoments)
- [x] Canonical in-place blockwise kernels (`blockwise_mean!`, `blockwise_variance!`, etc.)
- [x] Merge kernels for hierarchical sufficient-statistics composition
- [x] Public API convenience functions (`blockwise_mean`, `blockwise_stats`, etc.)
- [x] Product coarsening (`product_mean`, `product_moments`, joint moments)
- [x] Buffer pool and level buffer pool for zero-allocation reductions
- [x] Zero-allocation execution path (`allocate_buffers` + `execute!`)
- [x] Multi-resolution DAG builder (`build_tower_plan`) with BFS lattice
- [x] Per-dimension tower factors and `min_output_size` floor constraints
- [x] Factor schedule generation from seed ladders
- [x] Convenience wrappers (`build_optimal_multires_plan`, `multiresolution_stats`)
- [x] Hybrid mode (blockwise + sliding window)
- [x] SIMD kernel variants
- [x] GPU extension stubs (CUDA.jl)
- [x] Distributed scheduling
- [x] 1300+ tests (Aqua, correctness, allocation bounds)
- [x] Consolidated multi-resolution builders (removed legacy `plan_multires.jl`)
- [x] Thorough README with examples for all features

---

## Priority 1 — API Quality & Type Ergonomics

These are blocking for production use and clean external API.

- [ ] **Remove hardcoded collection type annotations**
  Replace `Vector{Int}`, `Vector{Symbol}`, `NTuple{N,Int}` in public signatures
  with `AbstractVector{Int}`, `AbstractVector{Symbol}`, etc. where the concrete
  type is not required. This breaks composability with generators, tuples passed
  as vectors, and other iterable types. Audit all exported functions.

- [ ] **Replace Symbol dispatch with function objects / Val types**
  - `stats=[:mean]` should accept `stats=[Statistics.mean]` or at least
    `stats=[Val(:mean)]` for type-stable dispatch
  - `padding::Symbol=:valid` should be a type (`Valid()`, `Same()`, `Full()`)
    or at minimum `Val{:valid}()` — symbols are not isbits and prevent
    type-stable dispatch
  - This also unblocks user-supplied reduction functions (see Priority 2)
  - Audit all `::Symbol` fields and arguments across the package

- [ ] **Plan pretty-print methods**
  `show()`, `display()`, `summary()` for `ReductionPlan`, `WindowConfig`,
  `ExecutionBuffers`. A plan should print its DAG structure, number of nodes,
  output shapes, and estimated memory footprint. Example:
  ```
  ReductionPlan: 5 nodes, 3 outputs
    root: (128,128,8) → blockwise_mean (2,2,1) → (64,64,8)
    ├─ blockwise_mean (2,2,1) → (32,32,8)
    │  └─ blockwise_mean (2,2,1) → (16,16,8)  [output]
    ├─ [output]
    └─ [output]
  ```

- [ ] **Attach metadata to outputs**
  `ReductionResult` should carry coordinate/label metadata: the effective block
  size, the reduction factor, the dimension mapping, and optionally
  user-supplied coordinate arrays. This enables downstream code to know what
  each result represents without external bookkeeping.

---

## Priority 2 — Extensibility & Composability

These unlock new use cases and interop with external code.

- [ ] **Support user-supplied kernel reductions**
  Allow `build_tower_plan(..., kernel=my_reduction_fn)` where the function
  satisfies an output-shape contract (either inferred or user-declared, similar
  to LinearAlgebra's `similar` pattern or xarray/dask output contracts). The
  existing `UserNode{F}` type already exists but is not wired into the tower
  builder or public API.

- [ ] **Mixed block + rolling reductions**
  Support plans where some levels use non-overlapping blocks and others use
  overlapping sliding windows. The `WindowConfig` already supports arbitrary
  strides; the gap is in the plan builder which assumes stride == size.

- [ ] **Fast options structs**
  Replace scattered keyword arguments with composable options structs
  (`ReductionOptions`, `TowerOptions`, etc.) for controlling behavior across
  the package. Should be `isbits`-compatible. Similar to
  `SolverOptions` patterns in DifferentialEquations.jl.

---

## Priority 3 — Advanced Features

Larger features that require design work.

- [ ] **Reduce to N-D histograms**
  Blockwise reduction that bins data into fixed-edge or fixed-count
  `n₁ × n₂ × ... × nₖ` histograms per block. Requires a new node type and
  merge kernel for histogram accumulation.

- [ ] **Streaming data support**
  Accept streaming input from Zarr, NetCDF, or any `IO`/channel source.
  Stream outputs to disk incrementally. Requires chunked execution and
  integration with the buffer pool for memory-bounded processing.

- [ ] **Rules for keeping/dropping data + plan pruning**
  Support predicates like "drop blocks that are all-zero" or "keep only blocks
  where variance > threshold". Prune the plan DAG accordingly to skip
  unnecessary computation. Requires a predicate node type and DAG optimizer.

- [ ] **Automatic ideal buffer size/shape from plan**
  Given a `ReductionPlan`, compute the minimum buffer set that allows full
  execution with maximum reuse. Currently the user calls `allocate_buffers`
  which allocates one buffer per step; an optimizer could share buffers across
  non-overlapping execution steps.

- [ ] **Plan optimizations**
  Analyze the DAG for redundant computations (e.g., two paths computing the
  same intermediate) and fuse or eliminate them. Allow user control over
  the compute-vs-memory tradeoff (recompute cheap intermediates vs. cache them).

---

## Priority 4 — Observability & UX

Nice-to-have for production monitoring and debugging.

- [ ] **Verbose plan progress logging**
  Support Julia's `@debug`/`@info`/`@warn` log levels. Custom loggers for
  long-running computations. Custom exception types for plan validation errors,
  shape mismatches, etc.

- [ ] **Fancy progress logging (REPL)**
  Progress bars and ETA for long-running multi-resolution computations.
  Plain-text fallback for non-interactive contexts. Integrate with
  `ProgressMeter.jl` or similar.

---

## Open Questions

- **NaN handling**: Add NaN support via extension? Use `NaNStatistics.jl`?
  Need to decide the canonical Julia approach for NaN-aware reductions.
- **FixedSizeArrays**: Consider using `FixedSizeArrays.jl` for window configs
  and small tuple-like storage once the package matures.
