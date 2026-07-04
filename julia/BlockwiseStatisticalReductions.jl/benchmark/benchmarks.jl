# Benchmarks for BlockwiseStatisticalReductions.
#
#   julia -t8 --project=benchmark benchmark/benchmarks.jl
#
# Three things are demonstrated:
#   1. LIMIT / OVERHEAD — a single dyadic (binary) reduction is essentially one optimal pass over the
#      data; the DAG/planner overhead over a hand-written reshape+reduce is tiny.
#   2. BLAZING FAST vs PURE-ARRAY — for mean/var/cov, at single and multi scale, BSR matches or beats
#      the pure-Julia reductions a consumer would otherwise hand-write (reshape+reduce, manual loop).
#   3. FILLS A RANGE OF SCALES — a whole Ladder of scales is produced in ≈ one pass (plan_work ≈ prod(X)),
#      far below independent reductions; threading and steady-state zero-alloc are also checked.

using BlockwiseStatisticalReductions
const BSR = BlockwiseStatisticalReductions
using OhMyThreads
using Random
using Statistics: mean, var

best(f, n = 5) = minimum(begin
                             GC.gc()
                             @elapsed f()
                         end for _ in 1:n)

ratio(a, b) = round(a / b; digits = 2)
ms(t) = round(t * 1e3; digits = 3)

# ── Pure-array baselines a consumer would hand-write (no BSR) ──────────────────

# Block mean via reshape + reduce (the strong idiomatic competitor; requires divisible sizes).
function reshape_block_mean(A::AbstractArray{T,2}, f::Int) where {T}
    m, n = size(A, 1) ÷ f, size(A, 2) ÷ f
    return dropdims(mean(reshape(A, f, m, f, n); dims = (1, 3)); dims = (1, 3))
end
function reshape_block_var(A::AbstractArray{T,2}, f::Int) where {T}
    m, n = size(A, 1) ÷ f, size(A, 2) ÷ f
    R = reshape(A, f, m, f, n)
    # population/sample variance over the (1,3) block axes, per (m,n) cell
    return dropdims(var(R; dims = (1, 3), corrected = true); dims = (1, 3))
end
# Horizontal (f,f,1) block mean of a 3-D array via reshape + reduce.
function reshape_block_mean3(A::AbstractArray{T,3}, f::Int) where {T}
    m, n, p = size(A, 1) ÷ f, size(A, 2) ÷ f, size(A, 3)
    return dropdims(mean(reshape(A, f, m, f, n, p); dims = (1, 3)); dims = (1, 3))
end

# Manual @inbounds block-mean loop (what one writes without reshape tricks).
function loop_block_mean(A::AbstractArray{T,2}, f::Int) where {T}
    m, n = size(A, 1) ÷ f, size(A, 2) ÷ f
    out = Array{Float64}(undef, m, n)
    @inbounds for j in 1:n, i in 1:m
        s = 0.0
        for jj in 1:f, ii in 1:f
            s += A[(i - 1) * f + ii, (j - 1) * f + jj]
        end
        out[i, j] = s / (f * f)
    end
    return out
end

# Independent multiscale via reshape (no cross-scale reuse) — the naive multiscale baseline.
naive_reshape_multiscale(A, factors) = Dict(f => reshape_block_mean(A, f) for f in factors)

# ── 1. LIMIT: single dyadic reduction ≈ one optimal pass, tiny graph overhead ──
function bench_limit()
    println("\n" * "="^72)
    println("1. LIMIT — single dyadic (2×) reduction: BSR vs optimal pure-array pass")
    println("="^72)
    for sz in ((2048, 2048), (4096, 4096))
        A = randn(sz...)
        reduce_stats(A, 2; stats = (Mean(),))                    # warmup
        reshape_block_mean(A, 2)
        t_bsr = best(() -> reduce_stats(A, 2; stats = (Mean(),)))
        t_reshape = best(() -> reshape_block_mean(A, 2))
        t_loop = best(() -> loop_block_mean(A, 2))
        plan = BSR._plan_for(sz, [(2, 2)])
        println("  $sz  mean @2×:")
        println("    BSR reduce_stats:  $(ms(t_bsr)) ms")
        println("    reshape+mean:      $(ms(t_reshape)) ms   (BSR $(ratio(t_bsr, t_reshape))× the optimal pass)")
        println("    manual loop:       $(ms(t_loop)) ms   (BSR $(ratio(t_bsr, t_loop))×)")
        println("    plan_work/prod(X): $(round(plan_work(plan) / prod(sz); digits = 3))   (→ ~1.0 means one pass, negligible graph overhead)")
    end
end

# ── 2 & 3. Multi-scale mean/var: BSR (shared DAG) vs independent pure-array reductions ──
function bench_multiscale()
    println("\n" * "="^72)
    println("2/3. MULTI-SCALE — fill a range of scales in ≈ one pass; BSR vs independent pure-array")
    println("="^72)
    for sz in ((2048, 2048), (4096, 4096))
        A = randn(sz...)
        factors = [2, 4, 8, 16, 32, 64]
        reduce_stats(A, factors; stats = (Mean(), Var()))        # warmup
        naive_reshape_multiscale(A, factors)

        t_bsr = best(() -> reduce_stats(A, factors; stats = (Mean(), Var()), backend = SerialBackend()))
        t_bsr_thr = best(() -> reduce_stats(A, factors; stats = (Mean(), Var()), backend = ThreadedBackend()))
        t_naive_reshape = best(() -> naive_reshape_multiscale(A, factors))
        plan = BSR._plan_for(sz, factors)

        println("\n  $sz  mean+var at $(length(factors)) scales $factors:")
        println("    BSR (serial DAG):        $(ms(t_bsr)) ms")
        println("    BSR (threaded DAG):      $(ms(t_bsr_thr)) ms   ($(ratio(t_bsr, t_bsr_thr))× vs serial)")
        println("    independent reshape+mean:$(ms(t_naive_reshape)) ms   (reshape here computes MEAN ONLY at each scale; BSR computes mean+var at all scales in ≈$(round(plan_work(plan)/prod(sz); digits=2)) passes)")
        println("    plan_work/naive_work =   $(round(plan_work(plan) / naive_work(plan); digits = 3))   ($(length(factors)) scales in ≈ $(round(plan_work(plan)/prod(sz); digits=2)) passes)")
    end
end

# ── Full 2,3-smooth reduction tree: a dense Ladder of scales, still ≈ one pass ──
function bench_full_tree()
    println("\n" * "="^72)
    println("   Full reduction tree (Ladder steps=[2,3]) on a LES-shaped 3-D slab")
    println("="^72)
    A = randn(Float32, 480, 128, 128)
    l = Ladder(seeds = [1], steps = [2, 3], maxfactor = (64, 64, 1))   # horizontal, z unreduced
    reduce_stats(A, l; stats = (Mean(),))                              # warmup
    t = best(() -> reduce_stats(A, l; stats = (Mean(),)))
    plan = BSR._plan_for(size(A), l)
    r = reduce_stats(A, l; stats = (Mean(),))
    println("  size $(size(A)) Float32, isotropic-horizontal full tree:")
    println("    scales produced: $(length(factors(r)))  → $(sort(unique(f[1] for f in factors(r))))")
    println("    time: $(ms(t)) ms")
    println("    plan_work/naive_work = $(round(plan_work(plan)/naive_work(plan); digits=3))  (all $(length(factors(r))) scales in ≈ $(round(plan_work(plan)/prod(size(A)); digits=2)) passes)")
end

# ── Covariance: BSR vs pure-array ⟨xy⟩−⟨x⟩⟨y⟩ at a single scale ──
function bench_cov()
    println("\n" * "="^72)
    println("   Covariance @8× — BSR (Pebay, fused) vs pure-array ⟨xy⟩−⟨x⟩⟨y⟩")
    println("="^72)
    sz = (2048, 2048)
    X = randn(sz...); Y = randn(sz...)
    covpa(X, Y, f) = reshape_block_mean(X .* Y, f) .- reshape_block_mean(X, f) .* reshape_block_mean(Y, f)
    reduce_stats(X, Y, 8; stats = (Cov(; corrected = false),)); covpa(X, Y, 8)  # warmup
    t_bsr = best(() -> reduce_stats(X, Y, 8; stats = (Cov(; corrected = false),)))
    t_pa = best(() -> covpa(X, Y, 8))
    println("  $sz cov @8×:  BSR $(ms(t_bsr)) ms   pure-array $(ms(t_pa)) ms   (BSR $(ratio(t_bsr, t_pa))×; pure-array also materializes X.*Y)")
end

function bench_zero_alloc()
    println("\n" * "="^72)
    println("   Steady-state allocations (variance tower run!)")
    println("="^72)
    A = randn(1024, 1024)
    plan = tower_plan((1024, 1024); base_factor = (2, 2), steps = ([2], [2]), maxfactor = (128, 128))
    buf = allocate_tower(plan, VarAcc{Float64})
    run!(buf, plan, A)
    println("  variance tower run!: $(@allocated run!(buf, plan, A)) bytes")
end

# ── Prepared handle: one-shot reduce_stats (rebuilds plan+buffers+results) vs prepare/reduce_stats! ──
function bench_prepared()
    println("\n" * "="^72)
    println("   Prepared handle — per-call allocation in a repeated (pipeline) loop")
    println("="^72)
    for sz in ((2048, 2048), (4096, 4096))
        A = randn(sz...)
        factors = [2, 4, 8, 16, 32, 64]
        reduce_stats(A, factors; stats = (Mean(), Var()))
        a1 = @allocated reduce_stats(A, factors; stats = (Mean(), Var()))
        p = prepare(sz, factors; stats = (Mean(), Var()), Tin = Float64)
        reduce_stats!(p, A)
        a2 = @allocated reduce_stats!(p, A)
        t1 = best(() -> reduce_stats(A, factors; stats = (Mean(), Var())))
        t2 = best(() -> reduce_stats!(p, A))
        println("  $sz  6-scale mean+var:")
        println("    one-shot reduce_stats:  $(round(a1/1e6; digits=1)) MB/call,  $(ms(t1)) ms")
        println("    prepared reduce_stats!: $(a2) bytes/call,  $(ms(t2)) ms  ($(ratio(t1, t2))× faster, allocation-free)")
    end
end

function main()
    Random.seed!(0)
    println("Julia threads = ", Threads.nthreads())
    bench_limit()
    bench_multiscale()
    bench_full_tree()
    bench_cov()
    bench_prepared()
    bench_zero_alloc()
    println("\nDone.")
end

main()
