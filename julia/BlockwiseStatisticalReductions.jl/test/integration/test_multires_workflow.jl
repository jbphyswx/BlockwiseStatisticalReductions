"""
Integration tests for multi-resolution workflows.

Real workflow tests with synthetic data matching actual use cases:
- 3D block variance at multiple scales
- Product moments <ql*ql>, <ql*w>, <w*w> for TKE-style computation
- Full reduce → export pipeline
- Numerical stability stress test
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

# Use a shape where many common factors divide evenly in all dims
const TEST_SHAPE = (120, 60, 30)
const FACTORS = [2, 3, 5, 6, 10]

Test.@testset "3D block variance at multiple scales" begin
    data = randn(TEST_SHAPE)

    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, FACTORS; stats=[:mean, :variance], dims=(1, 2, 3)
    )

    for factor in FACTORS
        Test.@test haskey(results, factor)
        expected_shape = div.(TEST_SHAPE, factor)

        Test.@test size(results[factor][:mean]) == expected_shape
        Test.@test size(results[factor][:variance]) == expected_shape

        # All variances should be non-negative
        Test.@test all(results[factor][:variance] .>= 0)
    end
end

Test.@testset "Product moments for TKE computation" begin
    ql = randn(TEST_SHAPE)
    w = randn(TEST_SHAPE)

    window = BlockwiseStatisticalReductions.WindowConfig(
        (10, 10, 10), (10, 10, 10), :valid
    )

    moments = BlockwiseStatisticalReductions.product_moments(ql, w, window)

    out_shape = div.(TEST_SHAPE, 10)
    Test.@test size(moments.mean_x) == out_shape
    Test.@test size(moments.mean_y) == out_shape
    Test.@test size(moments.cov_xy) == out_shape
    Test.@test size(moments.var_x) == out_shape
    Test.@test size(moments.var_y) == out_shape

    # All variances should be non-negative
    Test.@test all(moments.var_x .>= 0)
    Test.@test all(moments.var_y .>= 0)
end

Test.@testset "Full pipeline: reduce → export" begin
    data = randn(TEST_SHAPE)

    factors = [2, 3, 5, 6]
    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, factors; stats=[:mean, :variance], dims=(1, 2, 3)
    )

    # Extract statistics for export
    export_data = Dict{String, AbstractArray}()
    for factor in factors
        export_data["mean_$(factor)x"] = results[factor][:mean]
        export_data["var_$(factor)x"] = results[factor][:variance]
    end

    # Verify all exports are present
    for factor in factors
        Test.@test haskey(export_data, "mean_$(factor)x")
        Test.@test haskey(export_data, "var_$(factor)x")
    end

    total_mean = sum(sum(export_data["mean_$(factor)x"]) for factor in factors)
    Test.@test isfinite(total_mean)
end

Test.@testset "Memory efficiency with buffer pool" begin
    factors = [2, 3, 5, 6]

    pool = BlockwiseStatisticalReductions.create_buffer_pool_for_factors(
        Float64, TEST_SHAPE, factors
    )

    # Acquire and release multiple times
    for _ in 1:10
        buf = BlockwiseStatisticalReductions.acquire_level!(pool, 3)
        buf .= randn(size(buf)...)
        BlockwiseStatisticalReductions.release_level!(pool, 3, buf)
    end

    Test.@test haskey(pool.level_shapes, 3)
end

Test.@testset "Numerical stability stress test" begin
    # Large values with small variance (catastrophic cancellation)
    large_mean = 1e8
    data = large_mean .+ randn(TEST_SHAPE) .* 1e-3

    variance = BlockwiseStatisticalReductions.blockwise_variance(data, (10, 10, 10))

    # Welford should handle this — variance should be positive and small
    Test.@test all(variance .> 0)
    Test.@test all(variance .< 1e-2)
end
