# Partitioned tensors

Two backends compute over a tensor no single process holds. They share everything below the transport —
the same slab geometry, the same two plans, the same kernels — and differ only in how the halo arrives.

## The shape of the problem

Split a tensor along one axis. A window belongs to the slab its **first** cell falls in, so every window
reaches forward only and one halo, on one side, is enough.

Each rank's windows then divide in two:

- those its own slab covers, computed straight off its array;
- those that reach past its end, computed over a small **staging region** holding just the cells they read.

Both write disjoint slices of one set of result arrays, so nothing is copied to stitch them together.

What travels is raw input cells, not accumulators. An accumulator is wider than an input element and the
missing cells are the same either way, so for dense and strided windows the halo is strictly cheaper; for
large pure tiles accumulators would be cheaper, but they cannot express the leading, clipped pieces that
overlapping windows produce. One halo, exchanged once before any compute, handles every window family.

## MPI — `using MPI`

SPMD: each rank already holds its slab and passes it as `Partitioned`.

```julia
using MPI, ComputationalBackends
MPI.Init()

r = blockstats(Partitioned(my_slab; axis = 3), [4, 8, 16];
               stats = (Mean(), Var()),
               backend = CB.MPIBackend(CB.SerialBackend(), MPI.COMM_WORLD))
```

`Partitioned(fields; axis)` says only which axis is split; the global tensor is the concatenation of every
rank's fields along it, in rank order, and an `Allgather` of the local extents supplies the rest. Passing
`comm = nothing` to `MPIBackend` means `MPI.COMM_WORLD`.

Results stay partitioned: each rank gets the ordinary `ScaleResults` for the windows it owns, with the
windows in global coordinates so `geometry` reports where they sit in the whole tensor.

## Distributed — `using Distributed`

One process owns the tensor and scatters slabs of the last axis to the workers.

```julia
using Distributed
addprocs(4)
@everywhere using BlockwiseStatisticalReductions

r = blockstats(x, [4, 8, 16]; stats = (Mean(), Var()),
               backend = CB.DistributedBackend(CB.SerialBackend()))
release!(p)   # give the worker buffers back when done with a prepared handle
```

Every worker needs `@everywhere using BlockwiseStatisticalReductions`; the request says so plainly if one
does not have it.

A worker keeps its buffers and its prepared plan between calls, so what crosses a process boundary per
call is the slab's cells out and its share of the results back — never a plan, and never the whole tensor
at once. That transfer is inherent to a single owner; if the data is already on the workers, MPI is the
better fit.

## Weights

Per-element weights are partitioned like the data and travel in the same halo. Per-axis weight factors
index the **global** axis, so every rank slices its own share out of the same vector — a factor along an
axis is a property of the axis, not of who happens to hold which cells.

## Accuracy

A rank derives its windows from a different plan than a single process does, so cross-process results
agree to rounding rather than bit for bit — the same rule as across backends.
