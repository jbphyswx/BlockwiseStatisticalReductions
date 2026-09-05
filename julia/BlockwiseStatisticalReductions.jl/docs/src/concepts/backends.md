# Backends

Every entry point takes a `backend` keyword, dispatched on the abstract types of
[ComputationalBackends.jl](https://github.com/jbphyswx/ComputationalBackends.jl):

```julia
using ComputationalBackends: ComputationalBackends as CB

blockstats(x, [8]; stats = (Mean(),), backend = CB.SerialBackend())
blockstats(x, [8]; stats = (Mean(),), backend = CB.ThreadedBackend())
blockstats(x, [8]; stats = (Mean(),), backend = CB.GPUBackend(KernelAbstractions.CPU()))
```

`CB.AutoBackend()` is the default: a GPU backend when the fields are device arrays and the
KernelAbstractions extension is loaded, else threads when OhMyThreads is loaded and more than one is
available, else serial.

A backend whose extension is not loaded says so rather than failing obscurely.

## Serial

The reference implementation. Box loops run innermost along the contiguous axis; axes whose windows have
uniform length ≤ 8 get compile-time lengths, so boxes of at most 32 cells unroll into one balanced
expression tree and longer rows unroll per row.

## Threaded — `using OhMyThreads`

`tforeach` over contiguous chunks of output cells, four chunks per thread. No cell is written by two
tasks and the merge order inside a cell is unchanged, so a threaded run of a given plan is **bitwise
identical** to a serial run of the same plan.

Threading does not make bandwidth-bound work arbitrarily faster. On a 96-core machine a threaded
streaming `sum` is only 2.75× a serial one at 8 threads, so that — not the thread count — is the ceiling.

## GPU — `using KernelAbstractions` (plus `CUDA` for launch tuning)

One workitem per output cell, reducing its box in registers with the same `combine` the CPU runs. Nothing
is shared between workitems, so there is no local memory and no barrier.

Accumulator storage is struct-of-arrays in the input's own array type, so a plan allocated from a device
array is already device-resident: results stay on the device and nothing transfers except finalized
scalars. The workspace copies any index data the kernels dereference into the storage's array type once,
at allocate time.

A device reports `scan_ok = false` — the two-stack sliding scan carries a dependency along its axis — so
the planner composes instead. A per-cell scan kernel exists for a caller who builds that step by hand.

Parity is tested on `KernelAbstractions.CPU()` and JLArrays on every run; real CUDA hardware is a separate
gated job in `gpu/`.

## Across processes

MPI and Distributed have their own page: [partitioned tensors](distributed.md).
