# A tensor split across worker processes. Run as-is; it starts its own workers.
using Distributed

nworkers() == 1 && addprocs(2; exeflags = `--project=$(Base.active_project())`)
@everywhere using BlockwiseStatisticalReductions

using BlockwiseStatisticalReductions
using ComputationalBackends: ComputationalBackends as CB
using Random

Random.seed!(8)
x = randn(128, 128, 64)

# The owner holds the whole tensor and scatters slabs of the last axis to the workers.
r = blockstats(x, [(8, 8, 4), (16, 16, 8)]; stats = (Mean(), Var()),
               backend = CB.DistributedBackend(CB.SerialBackend()))

want = blockstats(x, [(8, 8, 4), (16, 16, 8)]; stats = (Mean(), Var()), backend = CB.SerialBackend())
println("workers: ", nworkers())
println("sizes:   ", scales(r))
println("matches a single process: ", all(r[i].mean ≈ want[i].mean for i in 1:length(windows(r))))

# For MPI the shape is different: each rank already holds its slab and passes it as
# `Partitioned(slab; axis = 3)` with `backend = CB.MPIBackend(CB.SerialBackend(), comm)`.
# See test/mpi_parity.jl for a worked SPMD example.
