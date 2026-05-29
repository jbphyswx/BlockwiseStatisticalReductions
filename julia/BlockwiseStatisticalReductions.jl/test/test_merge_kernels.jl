"""
Tests for hierarchical merge kernels (src/kernels/merge_kernels.jl).
Verifies that merging sufficient statistics up a tower reproduces the
same results as computing directly on the full block.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "blockwise_mean_M2! correctness" begin
    data = randn(Float64, 60, 60)
    window = (10, 10)
    out_sz = (6, 6)
    out_mean = zeros(Float64, out_sz...)
    out_M2 = zeros(Float64, out_sz...)

    BlockwiseStatisticalReductions.blockwise_mean_M2!(out_mean, out_M2, data, window)

    # Check against reference
    for j in 1:6, i in 1:6
        block = data[(i-1)*10+1:i*10, (j-1)*10+1:j*10]
        Test.@test out_mean[i,j] ≈ Statistics.mean(block)
        Test.@test out_M2[i,j] ≈ sum((block .- Statistics.mean(block)).^2)
    end
end

Test.@testset "blockwise_merge_mean_M2! correctness" begin
    # Create 60×60 data, compute (mean, M2) at 10×10 blocks → 6×6
    # Then merge 2×2 blocks of the 6×6 → 3×3
    # Result should match computing (mean, M2) directly at 20×20 blocks → 3×3
    data = randn(Float64, 60, 60)
    window_base = (10, 10)
    n_base = prod(window_base)  # 100 samples per base block

    # Base level: 6×6 of (mean, M2)
    mean_6 = zeros(Float64, 6, 6)
    M2_6 = zeros(Float64, 6, 6)
    BlockwiseStatisticalReductions.blockwise_mean_M2!(mean_6, M2_6, data, window_base)

    # Merge 2×2: 6×6 → 3×3
    mean_3 = zeros(Float64, 3, 3)
    M2_3 = zeros(Float64, 3, 3)
    BlockwiseStatisticalReductions.blockwise_merge_mean_M2!(mean_3, M2_3, mean_6, M2_6, n_base, (2, 2))

    # Reference: compute directly at 20×20 blocks
    for j in 1:3, i in 1:3
        block = data[(i-1)*20+1:i*20, (j-1)*20+1:j*20]
        Test.@test mean_3[i,j] ≈ Statistics.mean(block)
        Test.@test M2_3[i,j] ≈ sum((block .- Statistics.mean(block)).^2)
    end
end

Test.@testset "variance tower 3 levels" begin
    # Full tower: 120×120 → base=10 → merge 2×2 → merge 3×3
    # Should equal direct computation at 60×60 blocks
    data = randn(Float64, 120, 120)
    n_base = 100  # 10×10 blocks

    # Level 1: 12×12
    m1 = zeros(Float64, 12, 12)
    M2_1 = zeros(Float64, 12, 12)
    BlockwiseStatisticalReductions.blockwise_mean_M2!(m1, M2_1, data, (10, 10))

    # Level 2: merge 2×2 → 6×6 (effective 20×20 blocks, n=400)
    m2 = zeros(Float64, 6, 6)
    M2_2 = zeros(Float64, 6, 6)
    BlockwiseStatisticalReductions.blockwise_merge_mean_M2!(m2, M2_2, m1, M2_1, n_base, (2, 2))

    # Level 3: merge 3×3 → 2×2 (effective 60×60 blocks, n=3600)
    m3 = zeros(Float64, 2, 2)
    M2_3 = zeros(Float64, 2, 2)
    BlockwiseStatisticalReductions.blockwise_merge_mean_M2!(m3, M2_3, m2, M2_2, n_base * 4, (3, 3))

    # Reference: direct 60×60
    for j in 1:2, i in 1:2
        block = data[(i-1)*60+1:i*60, (j-1)*60+1:j*60]
        Test.@test m3[i,j] ≈ Statistics.mean(block) atol=1e-10
        expected_M2 = sum((block .- Statistics.mean(block)).^2)
        Test.@test M2_3[i,j] ≈ expected_M2 rtol=1e-10
    end

    # Variance derived from M2
    total_count = 60 * 60
    for j in 1:2, i in 1:2
        block = data[(i-1)*60+1:i*60, (j-1)*60+1:j*60]
        expected_var = Statistics.var(vec(block); corrected=true)
        got_var = BlockwiseStatisticalReductions.variance_from_M2(M2_3[i,j], total_count; corrected=true)
        Test.@test got_var ≈ expected_var rtol=1e-10
    end
end

Test.@testset "blockwise_mean_M2_M3! correctness" begin
    data = randn(Float64, 30, 30)
    window = (10, 10)
    out_mean = zeros(Float64, 3, 3)
    out_M2 = zeros(Float64, 3, 3)
    out_M3 = zeros(Float64, 3, 3)

    BlockwiseStatisticalReductions.blockwise_mean_M2_M3!(out_mean, out_M2, out_M3, data, window)

    for j in 1:3, i in 1:3
        block = data[(i-1)*10+1:i*10, (j-1)*10+1:j*10]
        μ = Statistics.mean(block)
        Test.@test out_mean[i,j] ≈ μ
        Test.@test out_M2[i,j] ≈ sum((block .- μ).^2)
        Test.@test out_M3[i,j] ≈ sum((block .- μ).^3)
    end
end

Test.@testset "blockwise_merge_mean_M2_M3! correctness" begin
    # 60×60, base blocks 10×10 → 6×6, merge 2×2 → 3×3
    # M3 at 3×3 should match direct 20×20 computation
    data = randn(Float64, 60, 60)
    n_base = 100

    m1 = zeros(Float64, 6, 6)
    M2_1 = zeros(Float64, 6, 6)
    M3_1 = zeros(Float64, 6, 6)
    BlockwiseStatisticalReductions.blockwise_mean_M2_M3!(m1, M2_1, M3_1, data, (10, 10))

    m2 = zeros(Float64, 3, 3)
    M2_2 = zeros(Float64, 3, 3)
    M3_2 = zeros(Float64, 3, 3)
    BlockwiseStatisticalReductions.blockwise_merge_mean_M2_M3!(m2, M2_2, M3_2, m1, M2_1, M3_1, n_base, (2, 2))

    for j in 1:3, i in 1:3
        block = data[(i-1)*20+1:i*20, (j-1)*20+1:j*20]
        μ = Statistics.mean(block)
        Test.@test m2[i,j] ≈ μ atol=1e-10
        Test.@test M2_2[i,j] ≈ sum((block .- μ).^2) rtol=1e-10
        Test.@test M3_2[i,j] ≈ sum((block .- μ).^3) rtol=1e-8
    end
end

Test.@testset "blockwise_mean_C! and merge covariance" begin
    x = randn(Float64, 60, 60)
    y = randn(Float64, 60, 60)
    n_base = 100

    # Base: 6×6
    mx1 = zeros(Float64, 6, 6)
    my1 = zeros(Float64, 6, 6)
    C1 = zeros(Float64, 6, 6)
    BlockwiseStatisticalReductions.blockwise_mean_C!(mx1, my1, C1, x, y, (10, 10))

    # Verify base
    for j in 1:6, i in 1:6
        bx = x[(i-1)*10+1:i*10, (j-1)*10+1:j*10]
        by = y[(i-1)*10+1:i*10, (j-1)*10+1:j*10]
        μx = Statistics.mean(bx)
        μy = Statistics.mean(by)
        Test.@test mx1[i,j] ≈ μx
        Test.@test my1[i,j] ≈ μy
        Test.@test C1[i,j] ≈ sum((bx .- μx) .* (by .- μy))
    end

    # Merge 2×2 → 3×3
    mx2 = zeros(Float64, 3, 3)
    my2 = zeros(Float64, 3, 3)
    C2 = zeros(Float64, 3, 3)
    BlockwiseStatisticalReductions.blockwise_merge_covariance!(mx2, my2, C2, mx1, my1, C1, n_base, (2, 2))

    # Should match direct 20×20 computation
    for j in 1:3, i in 1:3
        bx = x[(i-1)*20+1:i*20, (j-1)*20+1:j*20]
        by = y[(i-1)*20+1:i*20, (j-1)*20+1:j*20]
        μx = Statistics.mean(bx)
        μy = Statistics.mean(by)
        expected_C = sum((bx .- μx) .* (by .- μy))
        Test.@test mx2[i,j] ≈ μx atol=1e-10
        Test.@test my2[i,j] ≈ μy atol=1e-10
        Test.@test C2[i,j] ≈ expected_C rtol=1e-10
    end

    # Derive covariance
    total_count = 20 * 20
    for j in 1:3, i in 1:3
        bx = x[(i-1)*20+1:i*20, (j-1)*20+1:j*20]
        by = y[(i-1)*20+1:i*20, (j-1)*20+1:j*20]
        expected_cov = sum((bx .- Statistics.mean(bx)) .* (by .- Statistics.mean(by))) / (total_count - 1)
        got_cov = BlockwiseStatisticalReductions.covariance_from_C(C2[i,j], total_count; corrected=true)
        Test.@test got_cov ≈ expected_cov rtol=1e-10
    end
end

Test.@testset "blockwise_sum! composability" begin
    data = randn(Float64, 60, 60)
    # Sum at 10×10 → 6×6
    s1 = zeros(Float64, 6, 6)
    BlockwiseStatisticalReductions.blockwise_sum!(s1, data, (10, 10))
    # Sum of sums at 2×2 → 3×3
    s2 = zeros(Float64, 3, 3)
    BlockwiseStatisticalReductions.blockwise_sum!(s2, s1, (2, 2))

    # Should equal direct sum at 20×20
    for j in 1:3, i in 1:3
        block = data[(i-1)*20+1:i*20, (j-1)*20+1:j*20]
        Test.@test s2[i,j] ≈ sum(block)
    end
end

Test.@testset "3D arrays" begin
    data = randn(Float32, 60, 60, 8)
    m = zeros(Float32, 6, 6, 8)
    M2 = zeros(Float32, 6, 6, 8)
    BlockwiseStatisticalReductions.blockwise_mean_M2!(m, M2, data, (10, 10, 1))

    # Merge 2×2×1 → 3×3×8
    m2 = zeros(Float32, 3, 3, 8)
    M2_2 = zeros(Float32, 3, 3, 8)
    BlockwiseStatisticalReductions.blockwise_merge_mean_M2!(m2, M2_2, m, M2, 100, (2, 2, 1))

    # Spot check
    block = data[1:20, 1:20, 1]
    Test.@test m2[1,1,1] ≈ Statistics.mean(block) atol=1e-4
    Test.@test M2_2[1,1,1] ≈ sum((block .- Statistics.mean(block)).^2) rtol=1e-4
end

Test.@testset "skewness tower" begin
    # Verify skewness derivation from M2, M3
    data = randn(Float64, 60, 60)
    n_base = 100

    m1 = zeros(Float64, 6, 6)
    M2_1 = zeros(Float64, 6, 6)
    M3_1 = zeros(Float64, 6, 6)
    BlockwiseStatisticalReductions.blockwise_mean_M2_M3!(m1, M2_1, M3_1, data, (10, 10))

    # Merge 6×6 → 1×1 (full domain)
    m_full = zeros(Float64, 1, 1)
    M2_full = zeros(Float64, 1, 1)
    M3_full = zeros(Float64, 1, 1)
    BlockwiseStatisticalReductions.blockwise_merge_mean_M2_M3!(
        m_full, M2_full, M3_full, m1, M2_1, M3_1, n_base, (6, 6))

    # Compare to direct computation on all data
    μ = Statistics.mean(data)
    expected_M3 = sum((data .- μ).^3)
    Test.@test M3_full[1,1] ≈ expected_M3 rtol=1e-8

    # Skewness
    n_total = 60 * 60
    got_skew = BlockwiseStatisticalReductions.skewness_from_M2_M3(M2_full[1,1], M3_full[1,1], n_total)
    expected_skew = (expected_M3 / n_total) / (sum((data .- μ).^2) / n_total)^1.5
    Test.@test got_skew ≈ expected_skew rtol=1e-8
end
