# Roadmap

```@contents
Pages = ["roadmap.md"]
Depth = 2
```

See the full roadmap with prioritized TODOs in the repository:
[`docs/README.md`](https://github.com/jbphyswx/BlockwiseStatisticalReductions.jl/blob/main/docs/README.md)

## Priority 1 — API Quality & Type Ergonomics

- Remove hardcoded collection type annotations (`Vector{Int}` → `AbstractVector`)
- Replace `Symbol` dispatch with function objects / `Val` types for type stability
- Add `show()`, `display()`, `summary()` methods for `ReductionPlan`
- Attach metadata (coordinates, factor info) to `ReductionResult`

## Priority 2 — Extensibility & Composability

- Support user-supplied kernel reductions via output-shape contracts
- Mixed block + rolling reductions in a single plan
- Fast `isbits` options structs replacing scattered keyword arguments

## Priority 3 — Advanced Features

- N-D histogram reductions
- Streaming data support (Zarr, NetCDF)
- Data keep/drop rules with plan DAG pruning
- Automatic buffer sharing across non-overlapping execution steps
- Plan DAG optimization (redundancy elimination, compute/memory tradeoff)

## Priority 4 — Observability & UX

- Verbose plan progress logging with Julia log levels
- Progress bars for long-running computations (ProgressMeter.jl)

## Open Questions

- NaN handling: extension-based or built-in?
- FixedSizeArrays.jl adoption for window configs
