"""
Tests for online statistics accumulators: VarianceAccumulator, CovarianceAccumulator, RawMomentsAccumulator
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "VarianceAccumulator basic" begin
    acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    Test.@test acc.count == 0
    Test.@test acc.mean == 0.0
    Test.@test acc.sum_sq_dev == 0.0
    
    # Single value
    BlockwiseStatisticalReductions.fit!(acc, 5.0)
    Test.@test acc.count == 1
    Test.@test acc.mean == 5.0
    Test.@test iszero(acc.sum_sq_dev)  # No variance with 1 sample
    
    # More values
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    acc2 = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc2, data)
    
    Test.@test acc2.count == 5
    Test.@test Statistics.mean(acc2) ≈ 3.0
    Test.@test Statistics.var(acc2) ≈ 2.5  # Sample variance (n-1 denominator)
    Test.@test Statistics.var(acc2; corrected=false) ≈ 2.0  # Population variance
    Test.@test sqrt(Statistics.var(acc2)) ≈ sqrt(2.5)  # std = sqrt(var)
end

Test.@testset "VarianceAccumulator vs Statistics.var" begin
    # Test against stdlib
    for n in [2, 10, 100, 1000]
        data = randn(n)
        
        acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc, data)
        
        Test.@test Statistics.mean(acc) ≈ Statistics.mean(data)
        Test.@test Statistics.var(acc) ≈ Statistics.var(data)  # Both use corrected (n-1)
        Test.@test Statistics.var(acc; corrected=false) ≈ Statistics.var(data; corrected=false)
        Test.@test sqrt(Statistics.var(acc)) ≈ Statistics.std(data)
    end
end

Test.@testset "CovarianceAccumulator basic" begin
    acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
    
    # Perfect linear relationship: y = 2x
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    for (x, y) in zip(xs, ys)
        BlockwiseStatisticalReductions.fit!(acc, x, y)
    end
    
    Test.@test acc.count == 5
    Test.@test Statistics.mean(acc)[1] ≈ 3.0
    Test.@test Statistics.mean(acc)[2] ≈ 6.0
    Test.@test Statistics.cov(acc) > 0  # Positive covariance for linear relationship
    
    # Should match Statistics.cov
    Test.@test Statistics.cov(acc; corrected=false) ≈ Statistics.cov(xs, ys; corrected=false)
end

Test.@testset "CovarianceAccumulator vs Statistics.cov" begin
    for n in [2, 10, 100]
        xs = randn(n)
        ys = randn(n)
        
        acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc, xs, ys)
        
        Test.@test Statistics.mean(acc)[1] ≈ Statistics.mean(xs)
        Test.@test Statistics.mean(acc)[2] ≈ Statistics.mean(ys)
        Test.@test Statistics.cov(acc) ≈ Statistics.cov(xs, ys)
        Test.@test Statistics.cov(acc; corrected=false) ≈ Statistics.cov(xs, ys; corrected=false)
    end
end

Test.@testset "RawMomentsAccumulator" begin
    acc = BlockwiseStatisticalReductions.RawMomentsAccumulator{Float64,4}()
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    BlockwiseStatisticalReductions.fit!(acc, data)
    
    Test.@test acc.count == 5
    
    # First moment = mean
    Test.@test acc.moments[1] ≈ Statistics.mean(data)
    
    # Second moment = E[x²]
    expected_m2 = Statistics.mean(x^2 for x in data)
    Test.@test acc.moments[2] ≈ expected_m2
    
    # Third and fourth moments
    expected_m3 = Statistics.mean(x^3 for x in data)
    expected_m4 = Statistics.mean(x^4 for x in data)
    Test.@test acc.moments[3] ≈ expected_m3
    Test.@test acc.moments[4] ≈ expected_m4
end

Test.@testset "VarianceAccumulator edge cases" begin
    # Empty accumulator
    acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    Test.@test isnan(Statistics.var(acc))
    Test.@test isnan(sqrt(Statistics.var(acc)))
    
    # Single element
    BlockwiseStatisticalReductions.fit!(acc, 5.0)
    Test.@test isnan(Statistics.var(acc))  # Default (corrected=true) gives NaN with 1 sample
    Test.@test isnan(Statistics.var(acc; corrected=true))  # Bessel's correction gives NaN with 1 sample
    Test.@test Statistics.var(acc; corrected=false) == 0.0  # Population variance is 0
end

Test.@testset "CovarianceAccumulator edge cases" begin
    # Empty
    acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
    Test.@test isnan(Statistics.cov(acc))
    
    # Single element
    BlockwiseStatisticalReductions.fit!(acc, 1.0, 2.0)
    Test.@test Statistics.cov(acc; corrected=false) == 0.0
    Test.@test isnan(Statistics.cov(acc; corrected=true))
end

Test.@testset "Precision promotion" begin
    # Test Float32 inputs with Float64 accumulators
    data_f32 = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    
    acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc, data_f32)
    
    Test.@test Statistics.mean(acc) isa Float64
    Test.@test Statistics.var(acc) isa Float64
    Test.@test Statistics.mean(acc) ≈ 3.0
end
