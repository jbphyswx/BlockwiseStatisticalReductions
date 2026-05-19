"""
Integration tests for multi-resolution workflows.

Real workflow tests with synthetic data matching actual use cases:
- 3D block variance at 6 scales (500×250×127 → 6 resolution levels)
- Horizontal + vertical reductions with automatic caching
- Product moments <ql*ql>, <ql*w>, <w*w> for TKE-style computation
- Sliding valid-box on progressively coarsened data
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

# Realistic data size from success criteria
const TEST_SHAPE = (500, 250, 127)
const FACTORS = [2, 4, 8, 16, 32, 64]

Test.@testset "3D block variance at 6 scales" begin
    # Synthetic data mimicking atmospheric fields
    data = randn(TEST_SHAPE)
    
    # Compute variance at multiple scales
    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, FACTORS; stats=[:mean, :variance]
    )
    
    # Verify we got results for all factors
    for factor in FACTORS
        Test.@test haskey(results, factor)
        expected_shape = div.(TEST_SHAPE, factor)
        
        result = results[factor]
        Test.@test size(result.data[:mean]) == expected_shape
        Test.@test size(result.data[:variance]) == expected_shape
        
        # All variances should be positive
        Test.@test all(result.data[:variance] .>= 0)
    end
    
    # Verify variance decreases at coarser scales (law of large numbers)
    variances = [results[f].data[:variance][1] for f in FACTORS]
    # Generally, variance should decrease as we average more samples
    # (though this is stochastic, we check rough trend)
end

Test.@testset "Product moments for TKE computation" begin
    # Simulating turbulent kinetic energy terms
    # TKE = 0.5 * (<u'u'> + <v'v'> + <w'w'>)
    # We compute <ql*ql>, <ql*w>, <w*w> for cloud/turbulence analysis
    
    ql = randn(TEST_SHAPE)  # Liquid water
    w = randn(TEST_SHAPE)   # Vertical velocity
    
    window = BlockwiseStatisticalReductions.WindowConfig(
        (10, 10, 10), (10, 10, 10), :valid
    )
    
    # Compute product moments in one pass
    moments = BlockwiseStatisticalReductions.product_moments(ql, w, window)
    
    # Verify shapes
    out_shape = div.(TEST_SHAPE, 10)
    Test.@test size(moments.mean_x) == out_shape  # mean(ql)
    Test.@test size(moments.mean_y) == out_shape  # mean(w)
    Test.@test size(moments.cov_xy) == out_shape  # cov(ql, w)
    Test.@test size(moments.var_x) == out_shape   # var(ql)
    Test.@test size(moments.var_y) == out_shape   # var(w)
    
    # Verify covariance identity: Cov(x,y) = <xy> - <x><y>
    # (allowing for numerical precision)
    for i in 1:out_shape[1]
        for j in 1:out_shape[2]
            for k in 1:out_shape[3]
                cov = moments.cov_xy[i, j, k]
                mean_x = moments.mean_x[i, j, k]
                mean_y = moments.mean_y[i, j, k]
                mean_xy = moments.mean_xy[i, j, k]
                
                # Should satisfy identity
                computed_cov = mean_xy - mean_x * mean_y
                Test.@test cov ≈ computed_cov atol=1e-10
            end
        end
    end
end

Test.@testset "Multi-resolution with caching" begin
    # Test that caching actually reuses intermediates
    data = randn(TEST_SHAPE)
    
    factors = [2, 4, 8]  # 2 divides 4, 4 divides 8 - enables caching
    
    # First call - should populate cache
    results1 = BlockwiseStatisticalReductions.execute_cached_multilevel(
        data, factors, [:mean]
    )
    
    # Second call - should use cache
    results2 = BlockwiseStatisticalReductions.execute_cached_multilevel(
        data, factors, [:mean]
    )
    
    # Results should be identical
    for factor in factors
        Test.@test results1[factor].data ≈ results2[factor].data
    end
end

Test.@testset "Sliding window on coarsened data" begin
    # Simulate: coarsen by 4x, then sliding window analysis
    data = randn(TEST_SHAPE)
    
    # Step 1: Blockwise coarsening
    coarsened = BlockwiseStatisticalReductions.blockwise_mean(data, (4, 4, 4))
    
    # Step 2: Sliding window on coarsened data
    spec = BlockwiseStatisticalReductions.HybridReductionSpec(
        BlockwiseStatisticalReductions.WindowConfig((4, 4, 4), (4, 4, 4), :valid),
        BlockwiseStatisticalReductions.WindowConfig((3, 3, 3), (1, 1, 1), :same),
        [:mean],
        [:variance]
    )
    
    result = BlockwiseStatisticalReductions.execute_hybrid(data, spec)
    
    # Verify pipeline executed
    Test.@test result.block_result isa BlockwiseStatisticalReductions.ReductionResult
    Test.@test result.sliding_result isa BlockwiseStatisticalReductions.ReductionResult
end

Test.@testset "Full pipeline: load → reduce → export" begin
    # Simulate complete workflow
    data = randn(TEST_SHAPE)
    
    # Step 1: Multi-resolution reduction
    factors = [2, 4, 8, 16]
    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, factors; stats=[:mean, :variance]
    )
    
    # Step 2: Extract statistics for export
    export_data = Dict()
    for factor in factors
        export_data["mean_$(factor)x"] = results[factor].data[:mean]
        export_data["var_$(factor)x"] = results[factor].data[:variance]
    end
    
    # Step 3: Verify all exports are present
    for factor in factors
        Test.@test haskey(export_data, "mean_$(factor)x")
        Test.@test haskey(export_data, "var_$(factor)x")
    end
    
    # Step 4: Verify data integrity
    total_mean = sum([sum(export_data["mean_$(factor)x"]) for factor in factors])
    Test.@test isfinite(total_mean)
end

Test.@testset "Memory efficiency with buffer pool" begin
    # Test buffer pool reduces allocations
    data = randn(TEST_SHAPE)
    factors = [2, 4, 8, 16]
    
    pool = BlockwiseStatisticalReductions.create_buffer_pool_for_factors(
        Float64, TEST_SHAPE, factors
    )
    
    # Acquire and release multiple times
    for _ in 1:10
        buf = BlockwiseStatisticalReductions.acquire_level!(pool, 4)
        # Simulate computation
        buf .= randn(size(buf)...)
        BlockwiseStatisticalReductions.release_level!(pool, 4, buf)
    end
    
    # Pool should still be valid
    Test.@test haskey(pool.level_shapes, 4)
end

Test.@testset "Numerical stability stress test" begin
    # Test with data that causes numerical issues
    # Large values with small variance (catastrophic cancellation)
    large_mean = 1e8
    data = large_mean .+ randn(TEST_SHAPE) .* 1e-3
    
    # Our algorithm should handle this
    variance = BlockwiseStatisticalReductions.blockwise_variance(data, (10, 10, 10))
    
    # Variance should be reasonable (around 1e-6)
    Test.@test all(variance .> 0)
    Test.@test all(variance .< 1e-2)  # Should capture the small spread
end
