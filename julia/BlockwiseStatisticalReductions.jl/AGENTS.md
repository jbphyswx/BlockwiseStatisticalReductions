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
- A core fallback must be **strictly less specific** than the extension method that replaces it. Two
  methods with the same signature is method overwriting, which is illegal during precompilation and
  silently drops the extension to loading uncompiled.
- Accumulators are immutable isbits structs; storage is struct-of-arrays (`AccumulatorArray`), one
  component array per accumulator field, allocated in the input's array type.
- Name == behavior. A function that is not implemented does not exist or throws; it never computes a
  plausible substitute.
- Comments are one line and describe the code that is there.

## Layers (`src/`, finest to coarsest)

| layer | owns |
|---|---|
| `statistics/` | accumulator algebra (`neutral`, `lift`, `merge`, k-ary `combine`, `unmerge`), the two-phase combine protocol, statistic tags, weighted accumulators, composite over a field set, monoid checks |
| `storage/` | `AccumulatorArray` (SoA), uniform components, `Adapt` rules |
| `geometry/` | `AxisWindow(size, Positions)`, `Window{N}`, edge policies, derivation predicates, axis spacing and physical sizes, `Slab`/`partition` for split tensors |
| `scales/` | size generators, placements, combination and filters → resolved `Window`s; `Resolved` for targets a caller worked out |
| `planner/` | derivation DAG (base / coarsen / compose / restride / scan), roofline cost model, candidates, liveness, `explain` |
| `kernels/` | box fold (two-phase protocol over boxes, static shapes), compose, sliding scan, finalize (serial methods; backends add methods in `ext/`) |
| `execute/` | `Workspace` buffers and `run!` |
| `api/` | `blockstats`, `prepare`/`prepare_on`, `blockstats!`, `ScaleResults`, `Partitioned`/`slab_request` for split tensors, `show` |
| `backends.jl` | `CB` import, auto-resolution, kernel limits per backend, throwing stubs for extension-only backends |

**Fusion was tried and does not pay — do not re-derive this.** Computing several tiled levels per cache
tile looks like a 33 % traffic saving on a six-scale tower (402 MB → 267 MB), but measured it is 1.05× at
best in 2-D and 0.88× in 3-D. The bytes it removes are perfectly sequential and already prefetched, while
the blocking needed to remove them damages the input read, which is the whole cost. Blocking the
contiguous axis is 1.8× slower outright. In N dimensions a level shrinks by `kᴺ`, so the higher the
dimension the less there is to save and the worse the trade. The implementation and its hooks
(`Plan.fusion`, `KernelLimits.fusion_tile`) were removed rather than left as surface with nothing behind
them; see the performance page for the numbers.

## Extensions (`ext/`)

| trigger | adds |
|---|---|
| OhMyThreads | threaded kernel methods, threaded `kernel_limits` |
| KernelAbstractions | device kernels, `is_gpu_array`, device `field_shift` |
| CUDA | launch configuration from device attributes |
| DimensionalData | `DimArray`/`DimStack` in and out, `Intervals(Center())` output lookups |
| NCDatasets / Zarr | labelled variables read straight from a file or store |
| MPI | requests over a `Partitioned` tensor, halo exchange |
| Distributed | scatter of last-axis slabs to workers |

## Testing

`test/runtests.jl` runs one file per layer plus the backend and I/O parity files. `Pkg.test()` is what CI
runs; add `julia_args = ["-t4"]` to exercise threading.

- Correctness: brute-force references in `test/testutils.jl`; accuracy against `BigFloat`; cross-backend
  parity (serial == threaded == KA.CPU bit-identical for one plan; GPU and cross-process approximate).
- `test_mpi.jl` shells out to `mpiexec -n 2` and `-n 3`; `test_distributed.jl` starts its own workers.
  Together those are about 2.5 minutes of the ~6-minute suite, nearly all of it process startup.
- Development uses a warm Revise REPL and targeted scripts. The full suite is a release check, run once
  per deliverable — not a feedback tool.

## Performance

`benchmark/gates.jl` measures ratios to a reference measured in the same process and fails on threshold
violations:

```bash
julia --project=benchmark benchmark/gates.jl --only=kernels,kernels-f32,planner,executor
julia -t8 --project=benchmark benchmark/gates.jl --only=threaded
```

Run each group in the process it describes: the serial groups measure 20–70 % higher inside a
multithreaded process.

**The gates are deliberately not in CI.** They are throughput ratios and a shared runner cannot reproduce
them. What CI checks is the behaviour — the suite asserts that every kernel and every prepared request
allocates nothing, on every backend.

A gate whose two terms are only a few milliseconds does not repeat, because the task pool's own spawn and
join are the same order as the work: the threaded base-pass gates measured 1.09–1.42 for identical code at
4096² and repeat to a few percent at 8192². Size a new gate so both terms take tens of milliseconds.

## Documentation

- `docs/` — Documenter site; `julia --project=docs docs/make.jl`.
- `docs/generate_assets/` — a separate environment that writes the figures into `docs/src/assets/`. Run it
  by hand and commit the images; the docs build itself takes no measurements and needs no plotting stack.
- `examples/` — runnable scripts, a workspace member, executed by CI.
- `MIGRATION.md` — the 0.1 → 0.2 map, copied into the docs at build time.
