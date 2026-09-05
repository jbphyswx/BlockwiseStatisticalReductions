using Test: Test
using Random: Random
using Statistics: Statistics
using KernelAbstractions: KernelAbstractions as KA
using JLArrays: JLArray
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB
include("testutils.jl")

Random.seed!(41)
const SERIAL_G = CB.SerialBackend()
# KA's CPU backend runs the same kernels on host arrays; JLArray is a device array type whose storage,
# launches and adaptions behave like a GPU's without needing one. Between them these cover the kernels
# and the device-resident plumbing; a CUDA device only adds real hardware.
const DEVICES = (("KA.CPU", KA.CPU(), identity), ("JLArray", KA.get_backend(JLArray(zeros(1))), JLArray))

# JLArrays 0.3.3 ships a KernelAbstractions backend without a `synchronize` method. Its kernels run to
# completion on the host before the launch returns, so the missing method is a no-op; it is defined here
# rather than in the package, which must not add methods to types it does not own.
KA.synchronize(::typeof(DEVICES[2][2])) = nothing

Test.@testset "gpu" begin
    Test.@test BSR.GPU_LOADED[]

    Test.@testset "$name: kernels match serial" for (name, dev, move) in DEVICES
        gpu = CB.GPUBackend(dev)
        x = randn(48, 36); y = randn(48, 36)
        gx, gy = move(x), move(y)
        for (stats, hostf, devf) in (((BSR.Mean(), BSR.Var()), (x = x,), (x = gx,)),
                                     ((BSR.Mean(), BSR.Min(), BSR.Max()), (x = x,), (x = gx,)),
                                     ((BSR.Cov(:x, :y), BSR.Corr(:x, :y)), (x = x, y = y), (x = gx, y = gy)))
            for sizes in ((6, 6), (4, 9))
                w = BSR.tiled((48, 36), sizes, BSR.Truncate())
                C, _, _, _ = BSR.assemble(stats, keys(hostf), Float64, Float64)
                h = BSR.boxfold!(BSR.AccumulatorArray(C, x, BSR.shape(w)), hostf, w, SERIAL_G)
                g = BSR.boxfold!(BSR.AccumulatorArray(C, gx, BSR.shape(w)), devf, w, gpu)
                # storage stayed where the data is (compare the container, not the component eltype)
                Test.@test Base.typename(typeof(BSR._prototype(g))) === Base.typename(typeof(move(zeros(1, 1))))
                Test.@test host_accumulators(h) == host_accumulators(g)
            end
        end
        # a window with explicit origins carries an index vector that has to travel with the data
        aw = (BSR.anchored(48, 7, [0, 10, 41]), BSR.anchored(36, 5, [0, 31]))
        C, _, _, _ = BSR.assemble((BSR.Mean(),), (:x,), Float64, Float64)
        h = BSR.boxfold!(BSR.AccumulatorArray(C, x, BSR.shape(aw)), (x = x,), aw, SERIAL_G)
        g = BSR.boxfold!(BSR.AccumulatorArray(C, gx, BSR.shape(aw)), (x = gx,), BSR._like(gx, aw), gpu)
        Test.@test host_accumulators(h) == host_accumulators(g)
        # coarsen, compose, scan and finalize
        fw = BSR.tiled((48, 36), (2, 2), BSR.Truncate())
        hf = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, BSR.shape(fw)), (x,), fw, SERIAL_G)
        gf = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, gx, BSR.shape(fw)), (gx,), fw, gpu)
        grid = BSR.tiled((24, 18), (2, 2), BSR.Truncate())
        hc = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (12, 9)), hf, grid, SERIAL_G)
        gc = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, gx, (12, 9)), gf, grid, gpu)
        Test.@test host_accumulators(hc) == host_accumulators(gc)
        hp = BSR.compose!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (20, 18)), hf, hf, 1, 1:20, 5:24, SERIAL_G)
        gp = BSR.compose!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, gx, (20, 18)), gf, gf, 1,
                          BSR._like(gx, collect(1:20)), BSR._like(gx, collect(5:24)), gpu)
        Test.@test host_accumulators(hp) == host_accumulators(gp)
        scr = BSR.ScanScratch(BSR.VarAcc{Float64}, 4)
        hs = BSR.scan!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (21, 18)), hf, 1, 4, false, scr, SERIAL_G)
        gs = BSR.scan!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, gx, (21, 18)), gf, 1, 4, false, scr, gpu)
        Test.@test values_of(hs, BSR.Var()) ≈ values_of(Adapt.adapt(Array, gs), BSR.Var()) rtol = 1e-12
        hd = Array{Float64}(undef, size(hf)); gd = move(Array{Float64}(undef, size(gf)))
        BSR.finalize!(hd, hf, BSR.Var(), SERIAL_G)
        BSR.finalize!(gd, gf, BSR.Var(), (0.0,), gpu)
        Test.@test hd == collect(gd)
    end

    Test.@testset "$name: whole requests stay on the device" for (name, dev, move) in DEVICES
        gpu = CB.GPUBackend(dev)
        x = move(randn(96, 72))
        r = BSR.blockstats(x, [4, 8]; stats = (BSR.Mean(), BSR.Var()), backend = gpu)
        Test.@test typeof(r[(4, 4)].mean) === typeof(x)          # results are device arrays, never copied back
        h = BSR.blockstats(collect(x), [4, 8]; stats = (BSR.Mean(), BSR.Var()), backend = SERIAL_G)
        for w in BSR.windows(r), k in keys(r[w])
            Test.@test collect(r[w][k]) ≈ h[w][k] rtol = 1e-12
        end
        Test.@test collect(r[(8, 8)].mean) ≈ brute(Statistics.mean, collect(x), BSR.windows(r)[2])
        # a prepared device request repeats without copying to the host
        p = BSR.prepare(x, [4, 8]; stats = (BSR.Mean(),), backend = gpu)
        BSR.blockstats!(p, x)
        Test.@test typeof(BSR.blockstats!(p, x)[(4, 4)].mean) === typeof(x)
    end

    Test.@testset "AutoBackend picks the device its data is on" begin
        Test.@test CB.is_gpu_array(JLArray(zeros(4)))
        Test.@test !CB.is_gpu_array(zeros(4))
        auto = BSR.resolve_backend(CB.AutoBackend(), (JLArray(zeros(4)),))
        Test.@test auto isa CB.AbstractGPUBackend
        Test.@test !(BSR.resolve_backend(CB.AutoBackend(), (zeros(4),)) isa CB.AbstractGPUBackend)
    end

    Test.@testset "planner limits for a device" begin
        gpu = CB.GPUBackend(KA.CPU())
        l = BSR.kernel_limits(gpu, 2)
        Test.@test !l.scan_ok                       # the two-stack scan carries a dependency; a device composes instead
        Test.@test l.min_cells > BSR.kernel_limits(SERIAL_G, 2).min_cells
        Test.@test l.max_tile_elements <= BSR.kernel_limits(SERIAL_G, 2).max_tile_elements
        # with scans unavailable the planner still reaches dense targets, by composition or base passes
        targets = BSR.resolve(BSR.ScaleSet(BSR.Sizes([3, 4, 5]); placement = BSR.Dense()), (128, 128))
        p = BSR.plan((128, 128), targets; backend = gpu)
        BSR.check(p)
        Test.@test !any(h -> h isa BSR.Scan, p.how)
        Test.@test length(p.outputs) == length(targets)
    end
end
