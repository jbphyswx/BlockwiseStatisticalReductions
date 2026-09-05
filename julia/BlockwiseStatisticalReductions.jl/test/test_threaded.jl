using Test: Test
using Random: Random
using Statistics: Statistics
using OhMyThreads: OhMyThreads
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB
include("testutils.jl")

Random.seed!(31)
const THREADED = CB.ThreadedBackend()
const SERIAL_ = CB.SerialBackend()

Test.@testset "threaded" begin
    Test.@test BSR.THREADS_LOADED[]

    Test.@testset "kernels are bit-identical to serial" begin
        x = randn(96, 72); y = randn(96, 72)
        for (stats, fields) in (((BSR.Mean(), BSR.Var()), (x = x,)),
                                ((BSR.Mean(), BSR.Min(), BSR.Max()), (x = x,)),
                                ((BSR.Cov(:x, :y), BSR.Corr(:x, :y)), (x = x, y = y)))
            for sizes in ((8, 8), (6, 9), (5, 5))
                w = BSR.tiled((96, 72), sizes, BSR.Truncate())
                C, _, _, _ = BSR.assemble(stats, keys(fields), Float64, Float64)
                s = BSR.boxfold!(BSR.AccumulatorArray(C, x, BSR.shape(w)), fields, w, SERIAL_)
                t = BSR.boxfold!(BSR.AccumulatorArray(C, x, BSR.shape(w)), fields, w, THREADED)
                Test.@test collect(s) == collect(t)
            end
        end
        # coarsen, compose and scan on the same accumulators
        w1 = BSR.tiled((96, 72), (2, 2), BSR.Truncate())
        fine = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, BSR.shape(w1)), (x,), w1, SERIAL_)
        grid = BSR.tiled((48, 36), (2, 2), BSR.Truncate())
        cs = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (24, 18)), fine, grid, SERIAL_)
        ct = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (24, 18)), fine, grid, THREADED)
        Test.@test collect(cs) == collect(ct)
        ps = BSR.compose!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (40, 36)), fine, fine, 1, 1:40, 9:48, SERIAL_)
        pt = BSR.compose!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (40, 36)), fine, fine, 1, 1:40, 9:48, THREADED)
        Test.@test collect(ps) == collect(pt)
        for (axis, sz, partial) in ((1, 5, false), (2, 4, false), (1, 3, true))
            m = partial ? size(fine, axis) : size(fine, axis) - sz + 1
            osz = ntuple(d -> d == axis ? m : size(fine, d), 2)
            scr = BSR.ScanScratch(BSR.VarAcc{Float64}, sz)
            ss = BSR.scan!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, osz), fine, axis, sz, partial, scr, SERIAL_)
            st = BSR.scan!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, osz), fine, axis, sz, partial, scr, THREADED)
            Test.@test collect(ss) == collect(st)
        end
        # finalize
        dst_s = Array{Float64}(undef, size(fine)); dst_t = similar(dst_s)
        BSR.finalize!(dst_s, fine, BSR.Var(), SERIAL_)
        BSR.finalize!(dst_t, fine, BSR.Var(), (0.0,), THREADED)
        Test.@test dst_s == dst_t
    end

    Test.@testset "one plan run on either backend is bit-identical" begin
        # The kernels differ only in which thread evaluates a cell, so a fixed plan gives identical
        # results. Across *requests* the plans themselves differ: `kernel_limits` is backend-dependent,
        # so the planner picks derivations for the backend it is given (see the next testset).
        x = randn(200, 150)
        C, _, _, _ = BSR.assemble((BSR.Mean(), BSR.Var()), (:x,), Float64, Float64)
        for scales in ([4, 8, 16], BSR.ScaleSet(BSR.Sizes([3, 4, 5]); placement = BSR.Dense()))
            p = BSR.plan((200, 150), BSR.resolve(scales, (200, 150)); acc_bytes = sizeof(C))
            wss = BSR.allocate(p, C, x); wst = BSR.allocate(p, C, x)
            BSR.run!(wss, p, (x,), SERIAL_)
            BSR.run!(wst, p, (x,), THREADED)
            for k in p.outputs
                Test.@test collect(BSR.node_storage(wss, k)) == collect(BSR.node_storage(wst, k))
            end
        end
    end

    Test.@testset "whole requests agree to rounding across backends" begin
        x = randn(200, 150); y = randn(200, 150)
        for scales in ([4, 8, 16], [(6, 4), (12, 8)], BSR.ScaleSet(BSR.Sizes([5, 7]); placement = BSR.Stride(3)))
            rs = BSR.blockstats((x = x, y = y), scales;
                                stats = (BSR.Mean(:x), BSR.Var(:x), BSR.Min(:x), BSR.Cov(:x, :y)), backend = SERIAL_)
            rt = BSR.blockstats((x = x, y = y), scales;
                                stats = (BSR.Mean(:x), BSR.Var(:x), BSR.Min(:x), BSR.Cov(:x, :y)), backend = THREADED)
            Test.@test BSR.scales(rs) == BSR.scales(rt)
            for w in BSR.windows(rs), k in keys(rs[w])
                Test.@test rs[w][k] ≈ rt[w][k] rtol = 1e-12
            end
        end
        # dense windows exercise the scan under threads
        rs = BSR.blockstats(x, BSR.ScaleSet(BSR.Sizes([3, 4, 5]); placement = BSR.Dense()); stats = (BSR.Mean(),), backend = SERIAL_)
        rt = BSR.blockstats(x, BSR.ScaleSet(BSR.Sizes([3, 4, 5]); placement = BSR.Dense()); stats = (BSR.Mean(),), backend = THREADED)
        for w in BSR.windows(rs)
            Test.@test rs[w].mean ≈ rt[w].mean rtol = 1e-12
        end
        # and against brute force, so "identical to serial" cannot mean "identically wrong"
        w = BSR.windows(rt)[1]
        Test.@test rt[w].mean ≈ brute(Statistics.mean, x, w)
    end

    Test.@testset "AutoBackend prefers threads once they are available" begin
        Test.@test BSR.resolve_backend(CB.AutoBackend(), (randn(4),)) ===
                   (Threads.nthreads() > 1 ? CB.ThreadedBackend() : CB.SerialBackend())
        Test.@test BSR.resolve_backend(CB.SerialBackend(), (randn(4),)) === CB.SerialBackend()
    end

    Test.@testset "planner limits scale with the thread count" begin
        st = BSR.kernel_limits(THREADED, 2)
        sr = BSR.kernel_limits(SERIAL_, 2)
        Test.@test st.merge_rate == sr.merge_rate * Threads.nthreads()
        Test.@test st.min_cells >= sr.min_cells
        Test.@test st.max_tile_elements == sr.max_tile_elements
    end

    Test.@testset "a prepared threaded request repeats without growing its allocation" begin
        x = randn(256, 256)
        p = BSR.prepare(x, [4, 8]; stats = (BSR.Mean(), BSR.Var()), backend = THREADED)
        BSR.blockstats!(p, x)
        a1 = @allocated BSR.blockstats!(p, x)
        a2 = @allocated BSR.blockstats!(p, x)
        Test.@test a1 == a2
        Test.@test p.result[(8, 8)].var ≈ brute(Statistics.var, x, BSR.windows(p.result)[2])
    end
end
