using OnlineStats: OnlineStats
using StatsBase: StatsBase

Test.@testset "create_stat" begin
    # Mean
    m = BlockwiseStatisticalReductions.create_stat(:mean, Float64)
    Test.@test m isa OnlineStats.Mean{Float64}
    
    # Variance
    v = BlockwiseStatisticalReductions.create_stat(:var, Float64)
    Test.@test v isa OnlineStats.Variance{Float64}
    
    # Std
    s = BlockwiseStatisticalReductions.create_stat(:std, Float64)
    Test.@test s isa OnlineStats.Series
    
    # Moments
    mo = BlockwiseStatisticalReductions.create_stat(:moments, Float64)
    Test.@test mo isa OnlineStats.Series
    
    # Extrema
    e = BlockwiseStatisticalReductions.create_stat(:min, Float64)
    Test.@test e isa OnlineStats.Extrema{Float64}
    
    # Sum
    su = BlockwiseStatisticalReductions.create_stat(:sum, Float64)
    Test.@test su isa OnlineStats.Sum{Float64}
    
    # Multiple stats
    ms = BlockwiseStatisticalReductions.create_stat([:mean, :var], Float64)
    Test.@test ms isa OnlineStats.Series
    
    # Error on unknown
    Test.@test_throws ErrorException BlockwiseStatisticalReductions.create_stat(:unknown)
end

using Statistics: Statistics

Test.@testset "fit_window!" begin
    arr = rand(100)
    
    # Fit mean
    m = OnlineStats.Mean(Float64)
    BlockwiseStatisticalReductions.fit_window!(m, arr)
    Test.@test OnlineStats.value(m) ≈ Statistics.mean(arr)
    
    # Fit variance
    v = OnlineStats.Variance(Float64; weight=OnlineStats.EqualWeight())
    BlockwiseStatisticalReductions.fit_window!(v, arr)
    Test.@test OnlineStats.value(v) ≈ Statistics.var(arr; corrected=true)
end

Test.@testset "window_stat" begin
    arr = reshape(1:100, (10, 10))
    
    # Single stat
    m = BlockwiseStatisticalReductions.window_stat(arr, :mean)
    Test.@test m ≈ Statistics.mean(arr)
    
    # Multiple stats
    ms = BlockwiseStatisticalReductions.window_stat(arr, [:mean, :var])
    Test.@test length(ms) == 2
    Test.@test ms[1] ≈ Statistics.mean(arr)
    Test.@test ms[2] ≈ Statistics.var(arr)
end

Test.@testset "rolling_stats" begin
    arr = reshape(1:100, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    
    # Rolling mean
    results = BlockwiseStatisticalReductions.rolling_stats(arr, cfg, :mean)
    Test.@test length(results) == 4  # 2x2 grid
    
    # Check first result
    val, meta = results[1]
    Test.@test val isa Real
    Test.@test haskey(meta, :indices)
    Test.@test haskey(meta, :center)
end

Test.@testset "rolling_histogram" begin
    arr = reshape(1:100, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    edges = 0:25:100
    
    results = BlockwiseStatisticalReductions.rolling_histogram(arr, cfg, edges)
    Test.@test length(results) == 4
    
    # Check histogram properties
    h, meta = results[1]
    Test.@test h isa StatsBase.Histogram
    Test.@test length(h.weights) == 4  # 4 bins
end

Test.@testset "adaptive_rolling_histogram" begin
    arr = reshape(1:100, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    
    results = BlockwiseStatisticalReductions.adaptive_rolling_histogram(arr, cfg, 4)
    Test.@test length(results) == 4
    
    h, meta = results[1]
    Test.@test h isa StatsBase.Histogram
end

Test.@testset "merge_stats" begin
    # Create two partial statistics
    arr1 = 1:50
    arr2 = 51:100
    
    v1 = OnlineStats.Variance(Float64; weight=OnlineStats.EqualWeight())
    v2 = OnlineStats.Variance(Float64; weight=OnlineStats.EqualWeight())
    
    BlockwiseStatisticalReductions.fit_window!(v1, arr1)
    BlockwiseStatisticalReductions.fit_window!(v2, arr2)
    
    # Merge
    merged = BlockwiseStatisticalReductions.merge_stats(v1, v2)
    
    # Compare to full array
    v_full = OnlineStats.Variance(Float64; weight=OnlineStats.EqualWeight())
    BlockwiseStatisticalReductions.fit_window!(v_full, 1:100)
    
    Test.@test OnlineStats.value(merged) ≈ OnlineStats.value(v_full)
end

Test.@testset "OnlineStats accuracy" begin
    # Compare OnlineStats results to Statistics stdlib
    arr = rand(1000)
    
    # Mean
    os_mean = OnlineStats.Mean(Float64)
    BlockwiseStatisticalReductions.fit_window!(os_mean, arr)
    Test.@test OnlineStats.value(os_mean) ≈ Statistics.mean(arr)
    
    # Variance
    os_var = OnlineStats.Variance(Float64; weight=OnlineStats.EqualWeight())
    BlockwiseStatisticalReductions.fit_window!(os_var, arr)
    Test.@test OnlineStats.value(os_var) ≈ Statistics.var(arr)
    
    # Std
    os_std = BlockwiseStatisticalReductions.create_stat(:std, Float64)
    BlockwiseStatisticalReductions.fit_window!(os_std, arr)
    Test.@test OnlineStats.value(os_std.stats[2]) ≈ Statistics.std(arr)
end
