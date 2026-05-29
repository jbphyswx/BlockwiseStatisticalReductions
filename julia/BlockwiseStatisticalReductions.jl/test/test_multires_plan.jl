"""
Tests for multi-resolution statistics via multiresolution_stats.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "multiresolution_stats mean" begin
    # Create test data: 20x20 grid
    data = reshape(collect(1.0:400.0), 20, 20)

    factors = [2, 4, 5, 10]
    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, factors; stats=[:mean], dims=(1, 2)
    )

    # Check we got results for all factors
    Test.@test length(results) == 4
    for f in factors
        Test.@test haskey(results, f)
    end

    # Check shapes
    Test.@test size(results[2][:mean]) == (10, 10)
    Test.@test size(results[4][:mean]) == (5, 5)
    Test.@test size(results[5][:mean]) == (4, 4)
    Test.@test size(results[10][:mean]) == (2, 2)

    # Check values for factor 2 (mean of each 2×2 block)
    # Block (1,1): mean([1,2,21,22]) = 11.5
    expected_11 = Statistics.mean([1.0, 2.0, 21.0, 22.0])
    Test.@test results[2][:mean][1, 1] ≈ expected_11
end

Test.@testset "multiresolution_stats variance" begin
    n = 20
    data = reshape(collect(1.0:(n * n)), n, n)

    factors = [2, 4, 10]
    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, factors; stats=[:variance], dims=(1, 2)
    )

    # Check shapes
    Test.@test size(results[2][:variance]) == (10, 10)
    Test.@test size(results[4][:variance]) == (5, 5)
    Test.@test size(results[10][:variance]) == (2, 2)

    # Variance should be positive
    Test.@test all(results[2][:variance] .> 0)
end

Test.@testset "multiresolution_stats high-level API" begin
    n = 50
    data = randn(n, n, 10)  # 3D data, dims default to (1,2)

    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, [5, 10]; stats=[:mean, :variance]
    )

    # Check factor 5 result
    Test.@test haskey(results, 5)
    Test.@test results[5][:mean] isa AbstractArray
    Test.@test size(results[5][:mean]) == (10, 10, 10)  # 50/5=10, 50/10 not applied to dim3
    Test.@test results[5][:variance] isa AbstractArray
end

Test.@testset "multiresolution_stats mean composability" begin
    # Mean-of-means equals direct mean (composable)
    n = 64
    data = randn(n, n)

    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, [2, 4]; stats=[:mean], dims=(1, 2)
    )

    result_2 = results[2][:mean]
    result_4 = results[4][:mean]

    # Manually reduce 2× result by 2× — should equal direct 4×
    expected_4_size = (div(n, 4), div(n, 4))
    manual_4 = zeros(expected_4_size)
    for i in 1:expected_4_size[1]
        for j in 1:expected_4_size[2]
            block = result_2[(i - 1) * 2 + 1:i * 2, (j - 1) * 2 + 1:j * 2]
            manual_4[i, j] = Statistics.mean(block)
        end
    end

    Test.@test result_4 ≈ manual_4
end

Test.@testset "multiresolution_stats 3D multi-stat" begin
    nx, ny, nz = 30, 30, 30
    data = randn(nx, ny, nz)

    factors = [2, 3, 5, 6, 10, 15]
    results = BlockwiseStatisticalReductions.multiresolution_stats(
        data, factors; stats=[:mean, :variance], dims=(1, 2, 3)
    )

    for factor in factors
        Test.@test haskey(results, factor)
        expected_shape = (div(nx, factor), div(ny, factor), div(nz, factor))
        Test.@test size(results[factor][:mean]) == expected_shape
        Test.@test size(results[factor][:variance]) == expected_shape
    end
end
