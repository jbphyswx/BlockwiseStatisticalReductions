"""
Tests for canonical blockwise kernels (src/kernels/blockwise_kernels.jl).

Verifies correctness, zero-allocation for in-place variants, and SIMD dispatch.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "blockwise_mean! correctness" begin
    # 2D
    data2d = reshape(Float64.(1:100), 10, 10)
    out2d = similar(data2d, 2, 2)
    BlockwiseStatisticalReductions.blockwise_mean!(out2d, data2d, (5, 5))
    Test.@test out2d[1, 1] ≈ Statistics.mean(data2d[1:5, 1:5])
    Test.@test out2d[2, 1] ≈ Statistics.mean(data2d[6:10, 1:5])
    Test.@test out2d[1, 2] ≈ Statistics.mean(data2d[1:5, 6:10])
    Test.@test out2d[2, 2] ≈ Statistics.mean(data2d[6:10, 6:10])

    # 3D (exercises SIMD path for large arrays)
    data3d = randn(Float32, 128, 128, 8)
    out3d = similar(data3d, 64, 64, 8)
    BlockwiseStatisticalReductions.blockwise_mean!(out3d, data3d, (2, 2, 1))
    Test.@test out3d[1, 1, 1] ≈ Statistics.mean(data3d[1:2, 1:2, 1])
    Test.@test out3d[64, 64, 8] ≈ Statistics.mean(data3d[127:128, 127:128, 8])

    # 3D with z-reduction
    out3d_z = similar(data3d, 64, 64, 4)
    BlockwiseStatisticalReductions.blockwise_mean!(out3d_z, data3d, (2, 2, 2))
    Test.@test out3d_z[1, 1, 1] ≈ Statistics.mean(data3d[1:2, 1:2, 1:2])
end

Test.@testset "blockwise_mean! zero allocations" begin
    data = randn(Float32, 256, 256, 8)
    out = similar(data, 128, 128, 8)
    # Warmup
    BlockwiseStatisticalReductions.blockwise_mean!(out, data, (2, 2, 1))
    alloc = Test.@allocated BlockwiseStatisticalReductions.blockwise_mean!(out, data, (2, 2, 1))
    Test.@test alloc == 0
end

Test.@testset "blockwise_mean_kernel allocating" begin
    data = randn(Float32, 64, 64, 4)
    result = BlockwiseStatisticalReductions.blockwise_mean_kernel(data, (4, 4, 2))
    Test.@test size(result) == (16, 16, 2)
    Test.@test result[1, 1, 1] ≈ Statistics.mean(data[1:4, 1:4, 1:2])
end

Test.@testset "blockwise_variance! correctness" begin
    data = randn(Float64, 20, 20)
    out = similar(data, 2, 2)

    # Corrected (Bessel's)
    BlockwiseStatisticalReductions.blockwise_variance!(out, data, (10, 10); corrected=true)
    Test.@test out[1, 1] ≈ Statistics.var(vec(data[1:10, 1:10]); corrected=true)
    Test.@test out[2, 2] ≈ Statistics.var(vec(data[11:20, 11:20]); corrected=true)

    # Population variance
    BlockwiseStatisticalReductions.blockwise_variance!(out, data, (10, 10); corrected=false)
    Test.@test out[1, 1] ≈ Statistics.var(vec(data[1:10, 1:10]); corrected=false)
end

Test.@testset "blockwise_mean_variance! single pass" begin
    data = randn(Float64, 30, 30, 6)
    out_mean = similar(data, 3, 3, 2)
    out_var = similar(data, 3, 3, 2)

    BlockwiseStatisticalReductions.blockwise_mean_variance!(out_mean, out_var, data, (10, 10, 3))

    block = data[1:10, 1:10, 1:3]
    Test.@test out_mean[1, 1, 1] ≈ Statistics.mean(block)
    Test.@test out_var[1, 1, 1] ≈ Statistics.var(vec(block); corrected=true)
end

Test.@testset "blockwise_min! / blockwise_max!" begin
    data = randn(Float32, 40, 40, 10)
    out_min = similar(data, 4, 4, 2)
    out_max = similar(data, 4, 4, 2)

    BlockwiseStatisticalReductions.blockwise_min!(out_min, data, (10, 10, 5))
    BlockwiseStatisticalReductions.blockwise_max!(out_max, data, (10, 10, 5))

    Test.@test out_min[1, 1, 1] ≈ minimum(data[1:10, 1:10, 1:5])
    Test.@test out_max[1, 1, 1] ≈ maximum(data[1:10, 1:10, 1:5])
    Test.@test all(out_min .<= out_max)
end

Test.@testset "blockwise_product_mean!" begin
    x = randn(Float32, 20, 20, 4)
    y = randn(Float32, 20, 20, 4)
    out = similar(x, 2, 2, 2)

    BlockwiseStatisticalReductions.blockwise_product_mean!(out, x, y, (10, 10, 2))

    expected = Statistics.mean(x[1:10, 1:10, 1:2] .* y[1:10, 1:10, 1:2])
    Test.@test out[1, 1, 1] ≈ expected
end

Test.@testset "blockwise_joint_moments!" begin
    x = randn(Float64, 20, 20)
    y = 2.0 .* x .+ randn(20, 20) * 0.1
    mx = similar(x, 2, 2)
    my = similar(x, 2, 2)
    vx = similar(x, 2, 2)
    vy = similar(x, 2, 2)
    cxy = similar(x, 2, 2)

    BlockwiseStatisticalReductions.blockwise_joint_moments!(mx, my, vx, vy, cxy, x, y, (10, 10))

    block_x = vec(x[1:10, 1:10])
    block_y = vec(y[1:10, 1:10])
    Test.@test mx[1, 1] ≈ Statistics.mean(block_x)
    Test.@test my[1, 1] ≈ Statistics.mean(block_y)
    Test.@test vx[1, 1] ≈ Statistics.var(block_x; corrected=true)
    Test.@test vy[1, 1] ≈ Statistics.var(block_y; corrected=true)
    Test.@test cxy[1, 1] ≈ Statistics.cov(block_x, block_y)
end
