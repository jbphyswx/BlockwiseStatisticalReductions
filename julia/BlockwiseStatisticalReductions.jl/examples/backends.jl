# Backends: serial / threaded / distributed give identical results.
#
#   julia -t4 --project=examples examples/backends.jl

using BlockwiseStatisticalReductions
using OhMyThreads: OhMyThreads          # enables ThreadedBackend
using Random: Random

Random.seed!(4)
data = randn(512, 512)
scales = [4, 8, 16, 32]

rser = reduce_stats(data, scales; stats = (Mean(), Var()), backend = SerialBackend())
rthr = reduce_stats(data, scales; stats = (Mean(), Var()), backend = ThreadedBackend())

println("Julia threads: ", Threads.nthreads())
println("threaded == serial (bit-identical): ",
        all(rthr[f].var == rser[f].var for f in factors(rser)))
println("AutoBackend resolves to: ", resolve_backend(AutoBackend()))

# Distributed (uncomment to run across worker processes):
#
#   using Distributed, SharedArrays
#   addprocs(4)
#   @everywhere using BlockwiseStatisticalReductions
#   rdist = reduce_stats(data, scales; stats = (Mean(), Var()), backend = DistributedBackend())
#   @assert all(rdist[f].var == rser[f].var for f in factors(rser))   # also bit-identical

# Zero-allocation repeated execution (e.g. streaming frames): build plan + buffers once, reuse.
plan = tower_plan(size(data); base_factor = (2, 2), steps = ([2], [2]), maxfactor = (32, 32))
buf = allocate_tower(plan, VarAcc{Float64})
run!(buf, plan, data)                                  # warmup
allocs = @allocated run!(buf, plan, data)
println("steady-state allocations per run!: ", allocs, " bytes")
