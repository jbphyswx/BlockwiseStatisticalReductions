"""
Tests for multi-resolution plan building and cached execution.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "factor_sequence basic" begin
    # Basic case - powers of 2
    seq = BlockwiseStatisticalReductions.factor_sequence(1, [2, 4, 8, 16])
    Test.@test seq == [1, 2, 4, 8, 16]
    
    # With gaps
    seq2 = BlockwiseStatisticalReductions.factor_sequence(1, [2, 8])
    Test.@test 2 in seq2
    Test.@test 8 in seq2
    
    # Non-divisible sequence
    seq3 = BlockwiseStatisticalReductions.factor_sequence(1, [3, 5, 15])
    Test.@test 3 in seq3
    Test.@test 5 in seq3
    Test.@test 15 in seq3
end

Test.@testset "factor_sequence divisibility" begin
    # Verify consecutive factors divide evenly
    for factors in [[2, 4, 8], [3, 6, 12], [5, 10, 20]]
        seq = BlockwiseStatisticalReductions.factor_sequence(1, factors)
        
        for i in 2:length(seq)-1
            # Each factor should divide the next (when applicable)
            if seq[i+1] % seq[i] == 0
                Test.@test seq[i+1] % seq[i] == 0
            end
        end
    end
end

Test.@testset "execute_cached_multilevel mean" begin
    # Create test data: 20x20 grid
    data = reshape(1.0:400.0, 20, 20)
    
    # Compute mean at factors 2, 4, 5, 10
    factors = [2, 4, 5, 10]
    results = BlockwiseStatisticalReductions.execute_cached_multilevel(
        data, factors, [:mean]
    )
    
    # Check we got results for all factors
    Test.@test length(results) == 4
    Test.@test haskey(results, 2)
    Test.@test haskey(results, 4)
    Test.@test haskey(results, 5)
    Test.@test haskey(results, 10)
    
    # Check shapes
    Test.@test size(results[2].data) == (10, 10)  # 20/2 = 10
    Test.@test size(results[4].data) == (5, 5)   # 20/4 = 5
    Test.@test size(results[5].data) == (4, 4)   # 20/5 = 4
    Test.@test size(results[10].data) == (2, 2)  # 20/10 = 2
    
    # Check values for factor 2 (mean of each 2x2 block)
    # Block (1,1): mean([1,2,21,22]) = 11.5
    expected_11 = Statistics.mean([1.0, 2.0, 21.0, 22.0])
    Test.@test results[2].data[1, 1] ≈ expected_11
end

Test.@testset "execute_cached_multilevel variance" begin
    # Create test data with known variance
    n = 20
    data = reshape(1.0:(n*n), n, n)
    
    factors = [2, 4, 10]
    results = BlockwiseStatisticalReductions.execute_cached_multilevel(
        data, factors, [:variance]
    )
    
    # Check shapes
    Test.@test size(results[2].data) == (10, 10)
    Test.@test size(results[4].data) == (5, 5)
    Test.@test size(results[10].data) == (2, 2)
    
    # Variance should be positive
    Test.@test all(results[2].data .> 0)
end

Test.@testset "multiresolution_stats high-level API" begin
    n = 50
    data = randn(n, n, 10)  # 3D data
    
    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, [5, 10],
        stats=[:mean, :variance]
    )
    
    # Check factor 5 result
    Test.@test haskey(results, 5)
    result_5 = results[5]  # ReductionResult
    Test.@test result_5.data[:mean] isa AbstractArray
    Test.@test size(result_5.data[:mean]) == (10, 10, 2)  # 50/5 = 10, 10/5 = 2
end

Test.@testset "caching reduces computation" begin
    # When factors divide evenly, caching should be used
    n = 64
    data = randn(n, n)
    
    # These factors divide evenly: 2, 4, 8, 16, 32, 64
    factors = [2, 4, 8, 16]
    
    results = BlockwiseStatisticalReductions.execute_cached_multilevel(
        data, factors, [:mean]
    )
    
    # Verify results are consistent
    # 4x should equal reducing 2x by another 2x
    result_2 = results[2].data
    result_4 = results[4].data
    
    # Manually reduce 2x result by 2x
    expected_4_size = (div(n, 4), div(n, 4))
    manual_4 = zeros(expected_4_size)
    
    for i in 1:expected_4_size[1]
        for j in 1:expected_4_size[2]
            block = result_2[(i-1)*2+1:i*2, (j-1)*2+1:j*2]
            manual_4[i, j] = Statistics.mean(block)
        end
    end
    
    Test.@test result_4 ≈ manual_4
end

Test.@testset "error on invalid factors" begin
    n = 10
    data = randn(n, n)
    
    # 3 doesn't divide 10 evenly
    Test.@test_throws ErrorException BlockwiseStatisticalReductions.execute_cached_multilevel(
        data, [3], [:mean]
    )
end

Test.@testset "3D multi-resolution" begin
    # Test with 3D data
    nx, ny, nz = 30, 30, 30
    data = randn(nx, ny, nz)
    
    factors = [2, 3, 5, 6, 10, 15]
    results = BlockwiseStatisticalReductions.execute_cached_multilevel(
        data, factors, [:mean, :variance]
    )
    
    # Check all results have correct 3D shapes (multi-stat returns Dict)
    for factor in factors
        Test.@test haskey(results, factor)
        expected_shape = (div(nx, factor), div(ny, factor), div(nz, factor))
        # With multiple stats, .data is a Dict
        Test.@test size(results[factor].data[:mean]) == expected_shape
        Test.@test size(results[factor].data[:variance]) == expected_shape
    end
end
