"""
Tests for hybrid reduction mode (blockwise + sliding).
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "HybridReductionSpec construction" begin
    spec = BlockwiseStatisticalReductions.HybridReductionSpec(
        BlockwiseStatisticalReductions.WindowConfig((10, 10), (10, 10), :valid),
        BlockwiseStatisticalReductions.WindowConfig((3, 3), (1, 1), :same),
        [:mean],
        [:variance]
    )
    
    Test.@test spec.block_window.sizes == (10, 10)
    Test.@test spec.sliding_window.sizes == (3, 3)
    Test.@test spec.block_stats == [:mean]
    Test.@test spec.sliding_stats == [:variance]
end

Test.@testset "execute_hybrid basic" begin
    # Create 20x20 test data
    data = reshape(1.0:400.0, 20, 20)
    
    # Hybrid: 10x10 block then 3x3 sliding
    spec = BlockwiseStatisticalReductions.HybridReductionSpec(
        BlockwiseStatisticalReductions.WindowConfig((10, 10), (10, 10), :valid),
        BlockwiseStatisticalReductions.WindowConfig((3, 3), (1, 1), :same),
        [:mean],
        [:mean]
    )
    
    result = BlockwiseStatisticalReductions.execute_hybrid(data, spec)
    
    # Check block result
    Test.@test result.block_result isa BlockwiseStatisticalReductions.ReductionResult
    Test.@test size(result.block_result.data) == (2, 2)  # 20/10 = 2
    
    # Block (1,1): mean([1,2,21,22; 3,4,23,24]) = mean(1:4, 21:24)
    # Actually columns first in Julia: [1,21,2,22; 3,23,4,24; ...]
    # 10x10 block from (1,1) to (10,10) in 20x20
    
    # Check sliding result (should be same size as block output for :same padding)
    Test.@test result.sliding_result isa BlockwiseStatisticalReductions.ReductionResult
    Test.@test size(result.sliding_result.data) == (2, 2)
end

Test.@testset "hybrid_reduction convenience function" begin
    data = rand(50, 50, 10)
    
    result = BlockwiseStatisticalReductions.hybrid_reduction(data,
        block_sizes=(10, 10, 5),
        sliding_sizes=(3, 3, 3),
        block_stats=[:mean],
        sliding_stats=[:mean]
    )
    
    # Block output: (5, 5, 2)
    Test.@test size(result.block_result.data) == (5, 5, 2)
    
    # Sliding output on coarsened data
    Test.@test size(result.sliding_result.data) == (5, 5, 2)
end

Test.@testset "hybrid workflow variance" begin
    data = randn(40, 40)
    
    spec = BlockwiseStatisticalReductions.HybridReductionSpec(
        BlockwiseStatisticalReductions.WindowConfig((10, 10), (10, 10), :valid),
        BlockwiseStatisticalReductions.WindowConfig((3, 3), (1, 1), :same),
        [:mean],
        [:mean]
    )
    
    result = BlockwiseStatisticalReductions.execute_hybrid(data, spec)
    
    # Blockwise mean of each 10x10 block
    block_means = result.block_result.data
    Test.@test size(block_means) == (4, 4)
    
    # Sliding mean on the coarsened 4x4 grid
    sliding_means = result.sliding_result.data
    Test.@test size(sliding_means) == (4, 4)
    
    # Verify values are reasonable (finite, not NaN)
    Test.@test all(isfinite.(block_means))
    Test.@test all(isfinite.(sliding_means))
end

Test.@testset "hybrid preserves metadata" begin
    data = rand(20, 20)
    
    spec = BlockwiseStatisticalReductions.HybridReductionSpec(
        BlockwiseStatisticalReductions.WindowConfig((10, 10), (10, 10), :valid),
        BlockwiseStatisticalReductions.WindowConfig((3, 3), (1, 1), :same),
        [:mean],
        [:mean]
    )
    
    result = BlockwiseStatisticalReductions.execute_hybrid(data, spec)
    
    # Check metadata in results
    Test.@test result.block_result.metadata[:phase] == :blockwise
    Test.@test result.sliding_result.metadata[:phase] == :sliding
    Test.@test result.block_result.metadata[:input_shape] == (20, 20)
end

Test.@testset "hybrid 3D" begin
    # 3D data
    data = rand(30, 30, 30)
    
    result = BlockwiseStatisticalReductions.hybrid_reduction(data,
        block_sizes=(10, 10, 10),
        sliding_sizes=(3, 3, 3),
        block_stats=[:mean],
        sliding_stats=[:mean]
    )
    
    # Block: 30/10 = 3 per dim -> (3, 3, 3)
    Test.@test size(result.block_result.data) == (3, 3, 3)
    
    # Sliding on coarsened
    Test.@test size(result.sliding_result.data) == (3, 3, 3)
end

Test.@testset "blockwise variance in hybrid" begin
    data = randn(40, 40)
    
    spec = BlockwiseStatisticalReductions.HybridReductionSpec(
        BlockwiseStatisticalReductions.WindowConfig((10, 10), (10, 10), :valid),
        BlockwiseStatisticalReductions.WindowConfig((3, 3), (1, 1), :same),
        [:variance],  # Blockwise variance
        [:mean]        # Sliding mean on variances
    )
    
    result = BlockwiseStatisticalReductions.execute_hybrid(data, spec)
    
    # Check we got variance from block phase
    block_var = result.block_result.data
    Test.@test size(block_var) == (4, 4)
    Test.@test all(block_var .>= 0)  # Variances are non-negative
end
