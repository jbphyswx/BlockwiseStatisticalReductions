# Benchmarks: multi-scale tower vs naive independent reductions, threading speedup, zero-alloc.
#
#   julia -t8 --project=benchmark benchmark/benchmarks.jl
#
# Demonstrates that the optimal DAG touches the data ~once (work ratio ≈ 1/(#scales)) and that the
# variance tower runs allocation-free at steady state.

using BlockwiseStatisticalReductions
const BSR = BlockwiseStatisticalReductions
using OhMyThreads
using Random

best(f, n = 5) = minimum(begin
                             GC.gc()
                             @elapsed f()
                         end for _ in 1:n)

# Naive baseline: compute each scale independently from the data (one full pass each).
function naive_multiscale(data, factors)
    out = Dict()
    for f in factors
        out[f] = blockreduce(VarAcc{Float64}, data, ntuple(_ -> f, ndims(data)))
    end
    return out
end

function main()
    Random.seed!(0)
    println("Julia threads = ", Threads.nthreads())
    for sz in ((2048, 2048), (4096, 4096))
        data = randn(sz...)
        factors = [2, 4, 8, 16, 32, 64]
        reduce_stats(data, factors; stats = (Mean(), Var()))            # warmup
        naive_multiscale(data, factors)

        t_tower = best(() -> reduce_stats(data, factors; stats = (Mean(), Var()), backend = SerialBackend()))
        t_naive = best(() -> naive_multiscale(data, factors))
        t_thread = best(() -> reduce_stats(data, factors; stats = (Mean(), Var()), backend = ThreadedBackend()))
        plan = BSR._plan_for(sz, factors)

        println("\n=== $(sz), mean+var at scales $factors ===")
        println("  ", plan)
        println("  tower (serial):    ", round(t_tower * 1e3; digits = 2), " ms")
        println("  naive independent: ", round(t_naive * 1e3; digits = 2), " ms  (",
                round(t_naive / t_tower; digits = 2), "× slower)")
        println("  tower (threaded):  ", round(t_thread * 1e3; digits = 2), " ms  (",
                round(t_tower / t_thread; digits = 2), "× vs serial tower)")
        println("  plan work / naive work = ", round(plan_work(plan) / naive_work(plan); digits = 3))
    end

    data = randn(1024, 1024)
    plan = tower_plan((1024, 1024); base_factor = (2, 2), steps = ([2], [2]), maxfactor = (128, 128))
    buf = allocate_tower(plan, VarAcc{Float64})
    run!(buf, plan, data)
    println("\nsteady-state allocations, variance tower run!: ", (@allocated run!(buf, plan, data)), " bytes")
end

main()
