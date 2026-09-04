# BlockwiseStatisticalReductions.jl — Agent Guide

Purpose-agnostic engine: mergeable statistics (count, sum, mean, variance, covariance, extrema, moments)
of an N-D tensor over many window sizes at once, computed by a planned tree of reductions so the data is
touched as close to once as possible. It operates on `AbstractArray`s and accumulators only. No domain
vocabulary anywhere in this package: axes are axes, windows are windows, fields are fields.

## Conventions

- Imports: `using X: X as XX` (or `using X: X`) and qualify every call. No bare `using`. The backend
  vocabulary is `using ComputationalBackends: ComputationalBackends as CB`; methods dispatch on the
  abstract types (`CB.AbstractSerialBackend`, …), never on the concrete ones.
- Extensions are named `BlockwiseStatisticalReductions<Trigger>Ext` (file = module = key in `[extensions]`).
- Accumulators are immutable isbits structs; storage is struct-of-arrays (`AccumulatorArray`), one
  component array per accumulator field, allocated in the input's array type.
- Name == behavior. A function that is not implemented does not exist or throws; it never computes a
  plausible substitute.
- Comments are one line and describe the code that is there.

## Layers (`src/`, finest to coarsest)

| layer | owns |
|---|---|
| `statistics/` | accumulator algebra (`neutral`, `lift`, `merge`, k-ary `combine`, `unmerge`), statistic tags, composite over a field set, monoid checks |
| `storage/` | `AccumulatorArray` (SoA), uniform components, `Adapt` rules |
| `geometry/` | `AxisWindow(size, Positions)`, `Window{N}`, edge policies, derivation predicates, axis spacing and physical sizes |
| `scales/` | user-facing size generators, placements, combination and filters → resolved `Window`s |
| `planner/` | derivation DAG (base / coarsen / compose / restride), roofline cost model, candidates, liveness, fusion groups, `explain` |
| `kernels/` | box fold (two-phase protocol over boxes, static shapes), compose, sliding scan, fused pyramid, finalize (serial methods; backends add methods in `ext/`) |
| `execute/` | `Workspace` buffers and `run!` |
| `api/` | `blockstats`, `prepare`, `blockstats!`, `ScaleResults`, `show` |
| `backends.jl` | `CB` import, auto-resolution, kernel limits per backend, throwing stubs for extension-only backends |

An empty file in this tree is a layer that has not landed yet.

## Testing and performance

- Correctness: brute-force references in `test/testutils.jl`; accuracy against `BigFloat`; cross-backend
  parity (serial == threaded bit-identical; GPU approximate).
- Performance: `benchmark/gates.jl` measures ratios to a streaming `sum` over the same bytes in the same
  process and fails on threshold violations. Development uses a warm Revise REPL and targeted scripts;
  the full suite and the gates are release checks, run once per deliverable.
- `Pkg.test()` runs `test/runtests.jl`; add `julia_args=["-t4"]` to exercise threading.
