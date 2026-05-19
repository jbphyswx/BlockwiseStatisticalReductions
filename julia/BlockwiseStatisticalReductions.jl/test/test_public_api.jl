"""
Tests for public API convenience functions.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "blockwise_mean basic" begin
    data = reshape(1.0:400.0, 20, 20)
    
    means = BlockwiseStatisticalReductions.blockwise_mean(data, (10, 10))
    
    Test.@test size(means) == (2, 2)
    
    # Block (1,1): mean of 1:10, 21:30, ..., 191:200
    # Actually: rows 1:10, cols 1:10 -> values 1,2,...,10,21,22,...,30,...
    expected_11 = Statistics.mean(data[1:10, 1:10])
    Test.@test means[1, 1] ≈ expected_11
    
    expected_22 = Statistics.mean(data[11:20, 11:20])
    Test.@test means[2, 2] ≈ expected_22
end

Test.@testset "blockwise_variance" begin
    data = randn(40, 40)
    
    vars = BlockwiseStatisticalReductions.blockwise_variance(data, (10, 10))
    
    Test.@test size(vars) == (4, 4)
    Test.@test all(vars .>= 0)  # Variances are non-negative
end

Test.@testset "blockwise_std" begin
    data = randn(40, 40)
    
    stds = BlockwiseStatisticalReductions.blockwise_std(data, (10, 10))
    vars = BlockwiseStatisticalReductions.blockwise_variance(data, (10, 10))
    
    Test.@test size(stds) == (4, 4)
    Test.@test all(stds .≈ sqrt.(vars))
end

Test.@testset "blockwise_covariance" begin
    # Perfect linear relationship: y = 2x
    x = reshape(1.0:400.0, 20, 20)
    y = 2 .* x
    
    covs = BlockwiseStatisticalReductions.blockwise_covariance(x, y, (10, 10))
    
    Test.@test size(covs) == (2, 2)
    Test.@test all(covs .> 0)  # Positive covariance for linear relationship
end

Test.@testset "blockwise_stats multiple" begin
    data = randn(20, 20, 10)
    
    results = BlockwiseStatisticalReductions.blockwise_stats(data, (10, 10, 5), 
                                                           stats=[:mean, :variance, :min, :max])
    
    Test.@test haskey(results, :mean)
    Test.@test haskey(results, :variance)
    Test.@test haskey(results, :min)
    Test.@test haskey(results, :max)
    
    Test.@test size(results[:mean]) == (2, 2, 2)
    Test.@test size(results[:variance]) == (2, 2, 2)
    
    # Variance should be positive
    Test.@test all(results[:variance] .>= 0)
    
    # Min should be <= max
    Test.@test all(results[:min] .<= results[:max])
end

Test.@testset "blockwise_moments" begin
    data = reshape(1.0:100.0, 10, 10)
    
    moments = BlockwiseStatisticalReductions.blockwise_moments(data, (5, 5), 4)
    
    Test.@test size(moments) == (2, 2)
    
    # Check first block moments
    m = moments[1, 1]
    Test.@test length(m) == 4
    
    # First moment = mean
    expected_mean = Statistics.mean(data[1:5, 1:5])
    Test.@test m[1] ≈ expected_mean
    
    # Second moment = E[x²]
    expected_m2 = Statistics.mean(x^2 for x in data[1:5, 1:5])
    Test.@test m[2] ≈ expected_m2
end

Test.@testset "blockwise_stats error on invalid window" begin
    data = randn(10, 10)
    
    # 3 doesn't divide 10 evenly
    Test.@test_throws ErrorException BlockwiseStatisticalReductions.blockwise_mean(data, (3, 3))
end

Test.@testset "blockwise_stats 3D" begin
    data = randn(30, 30, 30)
    
    means = BlockwiseStatisticalReductions.blockwise_mean(data, (10, 10, 10))
    
    Test.@test size(means) == (3, 3, 3)
end
