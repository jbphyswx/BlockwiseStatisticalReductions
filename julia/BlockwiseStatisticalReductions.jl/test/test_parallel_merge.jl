"""
Tests for parallel merge algorithms (Chan's variance merge, Pebay's covariance merge).

These tests verify that merging accumulators from sub-blocks gives the same result
as computing on the full dataset - this is the key invariant for hierarchical reductions.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "Variance merge: two blocks" begin
    # Split data into two blocks, compute separately, merge
    data = randn(100)
    mid = length(data) ÷ 2
    
    block1 = data[1:mid]
    block2 = data[mid+1:end]
    
    # Individual accumulators
    acc1 = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc1, block1)
    
    acc2 = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc2, block2)
    
    # Merge
    merged = BlockwiseStatisticalReductions.merge(acc1, acc2)
    
    # Compare to full computation
    full_acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(full_acc, data)
    
    Test.@test Statistics.mean(merged) ≈ Statistics.mean(full_acc) atol=1e-10
    Test.@test Statistics.var(merged) ≈ Statistics.var(full_acc) atol=1e-10
    Test.@test merged.count == full_acc.count
end

Test.@testset "Variance merge: many blocks" begin
    # Split into many small blocks
    data = randn(1000)
    n_blocks = 10
    block_size = length(data) ÷ n_blocks
    
    accs = Vector{BlockwiseStatisticalReductions.VarianceAccumulator{Float64}}()
    for i in 1:n_blocks
        block = data[(i-1)*block_size+1:i*block_size]
        acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc, block)
        push!(accs, acc)
    end
    
    # Merge all
    merged = BlockwiseStatisticalReductions.merge_all(accs)
    
    # Compare to full
    full_acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(full_acc, data)
    
    Test.@test Statistics.mean(merged) ≈ Statistics.mean(full_acc) rtol=1e-10
    Test.@test Statistics.var(merged) ≈ Statistics.var(full_acc) rtol=1e-10
end

Test.@testset "Variance merge: empty and single-element blocks" begin
    data = [1.0, 2.0, 3.0]
    
    acc_nonempty = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc_nonempty, data)
    
    acc_empty = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    
    # Merging with empty should give same result
    merged = BlockwiseStatisticalReductions.merge(acc_nonempty, acc_empty)
    Test.@test Statistics.mean(merged) ≈ Statistics.mean(acc_nonempty)
    Test.@test Statistics.var(merged) ≈ Statistics.var(acc_nonempty)
    
    # Reverse order
    merged2 = BlockwiseStatisticalReductions.merge(acc_empty, acc_nonempty)
    Test.@test Statistics.mean(merged2) ≈ Statistics.mean(acc_nonempty)
    Test.@test Statistics.var(merged2) ≈ Statistics.var(acc_nonempty)
end

Test.@testset "Variance merge_many" begin
    data = randn(100)
    n_blocks = 5
    block_size = length(data) ÷ n_blocks
    
    counts = Vector{Int}()
    means = Vector{Float64}()
    sum_sq_devs = Vector{Float64}()
    
    for i in 1:n_blocks
        block = data[(i-1)*block_size+1:i*block_size]
        acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc, block)
        push!(counts, acc.count)
        push!(means, acc.mean)
        push!(sum_sq_devs, acc.sum_sq_dev)
    end
    
    # Merge using vectorized function
    merged = BlockwiseStatisticalReductions.merge_many(counts, means, sum_sq_devs)
    
    # Compare to full
    full_acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(full_acc, data)
    
    Test.@test Statistics.mean(merged) ≈ Statistics.mean(full_acc) rtol=1e-10
    Test.@test Statistics.var(merged) ≈ Statistics.var(full_acc) rtol=1e-10
end

Test.@testset "Covariance merge: two blocks" begin
    xs = randn(100)
    ys = 2 .* xs .+ randn(100) .* 0.1  # Correlated with noise
    
    mid = length(xs) ÷ 2
    
    # Two blocks
    acc1 = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc1, xs[1:mid], ys[1:mid])
    
    acc2 = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc2, xs[mid+1:end], ys[mid+1:end])
    
    merged = BlockwiseStatisticalReductions.merge(acc1, acc2)
    
    # Full computation
    full_acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(full_acc, xs, ys)
    
    Test.@test Statistics.mean(merged)[1] ≈ Statistics.mean(full_acc)[1] atol=1e-10
    Test.@test Statistics.mean(merged)[2] ≈ Statistics.mean(full_acc)[2] atol=1e-10
    Test.@test Statistics.cov(merged) ≈ Statistics.cov(full_acc) atol=1e-10
end

Test.@testset "Covariance merge: many blocks" begin
    n = 1000
    xs = randn(n)
    ys = randn(n)
    n_blocks = 10
    block_size = n ÷ n_blocks
    
    accs = Vector{BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}}()
    for i in 1:n_blocks
        r = (i-1)*block_size+1:i*block_size
        acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc, xs[r], ys[r])
        push!(accs, acc)
    end
    
    merged = BlockwiseStatisticalReductions.merge_all(accs)
    
    full_acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(full_acc, xs, ys)
    
    Test.@test Statistics.cov(merged) ≈ Statistics.cov(full_acc) rtol=1e-10
end

Test.@testset "Covariance merge_many" begin
    n = 100
    xs = randn(n)
    ys = randn(n)
    n_blocks = 4
    block_size = n ÷ n_blocks
    
    counts = Vector{Int}()
    means_x = Vector{Float64}()
    means_y = Vector{Float64}()
    sum_cross_devs = Vector{Float64}()
    
    for i in 1:n_blocks
        r = (i-1)*block_size+1:i*block_size
        acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc, xs[r], ys[r])
        push!(counts, acc.count)
        push!(means_x, acc.mean_x)
        push!(means_y, acc.mean_y)
        push!(sum_cross_devs, acc.sum_cross_dev)
    end
    
    merged = BlockwiseStatisticalReductions.merge_many(counts, means_x, means_y, sum_cross_devs)
    
    full_acc = BlockwiseStatisticalReductions.CovarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(full_acc, xs, ys)
    
    Test.@test Statistics.cov(merged) ≈ Statistics.cov(full_acc) rtol=1e-10
end

Test.@testset "RawMoments merge" begin
    data = randn(100)
    mid = length(data) ÷ 2
    
    acc1 = BlockwiseStatisticalReductions.RawMomentsAccumulator{Float64,4}()
    BlockwiseStatisticalReductions.fit!(acc1, data[1:mid])
    
    acc2 = BlockwiseStatisticalReductions.RawMomentsAccumulator{Float64,4}()
    BlockwiseStatisticalReductions.fit!(acc2, data[mid+1:end])
    
    merged = BlockwiseStatisticalReductions.merge(acc1, acc2)
    
    full_acc = BlockwiseStatisticalReductions.RawMomentsAccumulator{Float64,4}()
    BlockwiseStatisticalReductions.fit!(full_acc, data)
    
    # All moments should match
    for i in 1:4
        Test.@test merged.moments[i] ≈ full_acc.moments[i] rtol=1e-10
    end
end

Test.@testset "RawMoments merge: many blocks" begin
    data = randn(1000)
    n_blocks = 10
    block_size = length(data) ÷ n_blocks
    
    accs = Vector{BlockwiseStatisticalReductions.RawMomentsAccumulator{Float64,3}}()
    for i in 1:n_blocks
        acc = BlockwiseStatisticalReductions.RawMomentsAccumulator{Float64,3}()
        BlockwiseStatisticalReductions.fit!(acc, data[(i-1)*block_size+1:i*block_size])
        push!(accs, acc)
    end
    
    merged = BlockwiseStatisticalReductions.merge_all(accs)
    
    full_acc = BlockwiseStatisticalReductions.RawMomentsAccumulator{Float64,3}()
    BlockwiseStatisticalReductions.fit!(full_acc, data)
    
    for i in 1:3
        Test.@test merged.moments[i] ≈ full_acc.moments[i] rtol=1e-10
    end
end

Test.@testset "Merge precision promotion" begin
    # Test that Float32 + Float64 merges promote correctly
    acc1 = BlockwiseStatisticalReductions.VarianceAccumulator{Float32}()
    BlockwiseStatisticalReductions.fit!(acc1, Float32[1.0, 2.0, 3.0])
    
    acc2 = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(acc2, [4.0, 5.0, 6.0])
    
    merged = BlockwiseStatisticalReductions.merge(acc1, acc2)
    Test.@test merged isa BlockwiseStatisticalReductions.VarianceAccumulator{Float64}
    
    # Should match full Float64 computation
    full_acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
    BlockwiseStatisticalReductions.fit!(full_acc, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    Test.@test Statistics.mean(merged) ≈ Statistics.mean(full_acc) rtol=1e-6
end

Test.@testset "Mathematical invariants" begin
    # Key invariant: BlockwiseStatisticalReductions.merge(acc1, acc2) should equal compute([data1; data2])
    # Test this many times with random data
    for _ in 1:100
        n1 = rand(1:50)
        n2 = rand(1:50)
        
        d1 = randn(n1)
        d2 = randn(n2)
        
        acc1 = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc1, d1)
        
        acc2 = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(acc2, d2)
        
        merged = BlockwiseStatisticalReductions.merge(acc1, acc2)
        
        full_acc = BlockwiseStatisticalReductions.VarianceAccumulator{Float64}()
        BlockwiseStatisticalReductions.fit!(full_acc, [d1; d2])
        
        Test.@test Statistics.mean(merged) ≈ Statistics.mean(full_acc) rtol=1e-10
        Test.@test Statistics.var(merged) ≈ Statistics.var(full_acc) rtol=1e-10
        Test.@test merged.sum_sq_dev ≈ full_acc.sum_sq_dev rtol=1e-10
    end
end
