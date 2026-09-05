# CUDA tests. Run on a machine with a device:
#
#   julia --project=gpu gpu/runtests.jl
#
# The kernels themselves are covered without hardware by test/test_gpu.jl, which runs them on
# KernelAbstractions' CPU backend and on JLArray. What only a device can check is that the plan is
# allocated in device memory, that results never leave it, and that the kernels are worth launching.

using Test: Test
using Random: Random
using Statistics: Statistics
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB

if !CUDA.functional()
    @warn "no CUDA device; skipping" CUDA.functional()
    exit(0)
end

CUDA.versioninfo()
Random.seed!(51)
const GPU = CB.GPUBackend(CUDA.CUDABackend())
const HOST = CB.SerialBackend()
best(f, n = 5) = minimum((CUDA.@sync f(); @elapsed CUDA.@sync f()) for _ in 1:n)

Test.@testset "CUDA" begin
    Test.@testset "results match the host and stay on the device" begin
        x = randn(512, 384); y = randn(512, 384)
        gx, gy = CUDA.CuArray(x), CUDA.CuArray(y)
        stats = (BSR.Mean(), BSR.Var(), BSR.Min(), BSR.Max())
        for scales in ([4, 8, 16], [(6, 4), (12, 8)])
            h = BSR.blockstats(x, scales; stats = stats, backend = HOST)
            g = BSR.blockstats(gx, scales; stats = stats, backend = GPU)
            Test.@test BSR.scales(h) == BSR.scales(g)
            for w in BSR.windows(h), k in keys(h[w])
                Test.@test g[w][k] isa CUDA.CuArray
                Test.@test Array(g[w][k]) ≈ h[w][k] rtol = 1e-12
            end
        end
        hc = BSR.blockstats((x = x, y = y), [8]; stats = (BSR.Cov(:x, :y), BSR.Corr(:x, :y)), backend = HOST)
        gc = BSR.blockstats((x = gx, y = gy), [8]; stats = (BSR.Cov(:x, :y), BSR.Corr(:x, :y)), backend = GPU)
        Test.@test Array(gc[(8, 8)].cov_x_y) ≈ hc[(8, 8)].cov_x_y rtol = 1e-12
        Test.@test Array(gc[(8, 8)].corr_x_y) ≈ hc[(8, 8)].corr_x_y rtol = 1e-12
    end

    Test.@testset "the workspace is device memory" begin
        gx = CUDA.CuArray(randn(256, 256))
        p = BSR.prepare(gx, [4, 8]; stats = (BSR.Mean(), BSR.Var()), backend = GPU)
        for k in p.plan.order
            Test.@test BSR._prototype(BSR.node_storage(p.workspace, k)) isa CUDA.CuArray
        end
        r = BSR.blockstats!(p, gx)
        Test.@test r[(4, 4)].mean isa CUDA.CuArray
        # a device request must not fall back to scalar indexing on the host
        CUDA.allowscalar(false)
        BSR.blockstats!(p, gx)
        CUDA.allowscalar(true)
    end

    Test.@testset "AutoBackend selects CUDA for device arrays" begin
        gx = CUDA.CuArray(randn(64, 64))
        Test.@test CB.is_gpu_array(gx)
        Test.@test BSR.resolve_backend(CB.AutoBackend(), (gx,)) == GPU
        r = BSR.blockstats(gx, [8]; stats = (BSR.Mean(),))
        Test.@test r[(8, 8)].mean isa CUDA.CuArray
    end

    Test.@testset "planner limits come from the device" begin
        l = BSR.kernel_limits(GPU, 2)
        Test.@test l.bandwidth > BSR.kernel_limits(HOST, 2).bandwidth
        Test.@test l.min_cells > BSR.kernel_limits(HOST, 2).min_cells
        Test.@test !l.scan_ok
    end

    Test.@testset "throughput" begin
        n = 4096
        gx = CUDA.CuArray(randn(n, n))
        roofline = best(() -> sum(gx))
        for (label, scales, stats, threshold) in (("mean 8x", [8], (BSR.Mean(),), 2.0),
                                                  ("mean+var 8x", [8], (BSR.Mean(), BSR.Var()), 4.0),
                                                  ("6 scales mean+var", [2, 4, 8, 16, 32, 64], (BSR.Mean(), BSR.Var()), 12.0))
            p = BSR.prepare(gx, scales; stats = stats, backend = GPU)
            BSR.blockstats!(p, gx)
            t = best(() -> BSR.blockstats!(p, gx))
            ratio = t / roofline
            println("  ", rpad(label, 20), "  ", round(t * 1e3; digits = 3), " ms   ", round(ratio; digits = 2),
                    "x a device-wide sum (threshold ", threshold, ")")
            Test.@test ratio <= threshold
        end
    end
end
