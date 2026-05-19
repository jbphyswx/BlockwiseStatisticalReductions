using OnlineStats: OnlineStats
using StatsBase: StatsBase

@testset "create_stat" begin
    # Mean
    m = BlockwiseStatisticalReductions.create_stat(:mean, Float64)
    @test m isa OnlineStats.Mean{Float64}
    
    # Variance
    v = BlockwiseStatisticalReductions.create_stat(:var, Float64)
    @test v isa OnlineStats.Variance{Float64}
    
    # Std
    s = BlockwiseStatisticalReductions.create_stat(:std, Float64)
    @test s isa OnlineStats.Series
    
    # Moments
    mo = BlockwiseStatisticalReductions.create_stat(:moments, Float64)
    @test mo isa OnlineStats.Series
    
    # Extrema
    e = BlockwiseStatisticalReductions.create_stat(:min, Float64)
    @test e isa OnlineStats.Extrema{Float64}
    
    # Sum
    su = BlockwiseStatisticalReductions.create_stat(:sum, Float64)
    @test su isa OnlineStats.Sum{Float64}
    
    # Multiple stats
    ms = BlockwiseStatisticalReductions.create_stat([:mean, :var], Float64)
    @test ms isa OnlineStats.Series
    
    # Error on unknown
    @test_throws ErrorException BlockwiseStatisticalReductions.create_stat(:unknown)
end

using Statistics: Statistics

@testset "fit_window!" begin
    arr = rand(100)
    
    # Fit mean
    m = OnlineStats.Mean(Float64)
    BlockwiseStatisticalReductions.fit_window!(m, arr)
    @test OnlineStats.value(m) ≈ Statistics.mean(arr)
    
    # Fit variance
    v = OnlineStats.Variance(Float64; weight=OnlineStats.EqualWeight())
    BlockwiseStatisticalReductions.fit_window!(v, arr)
    @test OnlineStats.value(v) ≈ Statistics.var(arr; corrected=true)
end

@testset "window_stat" begin
    arr = reshape(1:100, (10, 10))
    
    # Single stat
    m = BlockwiseStatisticalReductions.window_stat(arr, :mean)
    @test m ≈ Statistics.mean(arr)
    
    # Multiple stats
    ms = BlockwiseStatisticalReductions.window_stat(arr, [:mean, :var])
    @test length(ms) == 2
    @test ms[1] ≈ Statistics.mean(arr)
    @test ms[2] ≈ Statistics.var(arr)
end

@testset "rolling_stats" begin
    arr = reshape(1:100, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    
    # Rolling mean
    results = BlockwiseStatisticalReductions.rolling_stats(arr, cfg, :mean)
    @test length(results) == 4  # 2x2 grid
    
    # Check first result
    val, meta = results[1]
    @test val isa Real
    @test haskey(meta, :indices)
    @test haskey(meta, :center)
end

@testset "rolling_histogram" begin
    arr = reshape(1:100, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    edges = 0:25:100
    
    results = BlockwiseStatisticalReductions.rolling_histogram(arr, cfg, edges)
    @test length(results) == 4
    
    # Check histogram properties
    h, meta = results[1]
    @test h isa StatsBase.Histogram
    @test length(h.weights) == 4  # 4 bins
end

@testset "adaptive_rolling_histogram" begin
    arr = reshape(1:100, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    
    results = BlockwiseStatisticalReductions.adaptive_rolling_histogram(arr, cfg, 4)
    @test length(results) == 4
    
    h, meta = results[1]
    @test h isa StatsBase.Histogram
end

@testset "merge_stats" begin
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
    
    @test OnlineStats.value(merged) ≈ OnlineStats.value(v_full)
end

@testset "OnlineStats accuracy" begin
    # Compare OnlineStats results to Statistics stdlib
    arr = rand(1000)
    
    # Mean
    os_mean = OnlineStats.Mean(Float64)
    BlockwiseStatisticalReductions.fit_window!(os_mean, arr)
    @test OnlineStats.value(os_mean) ≈ Statistics.mean(arr)
    
    # Variance
    os_var = OnlineStats.Variance(Float64; weight=OnlineStats.EqualWeight())
    BlockwiseStatisticalReductions.fit_window!(os_var, arr)
    @test OnlineStats.value(os_var) ≈ Statistics.var(arr)
    
    # Std
    os_std = BlockwiseStatisticalReductions.create_stat(:std, Float64)
    BlockwiseStatisticalReductions.fit_window!(os_std, arr)
    @test OnlineStats.value(os_std.stats[2]) ≈ Statistics.std(arr)
end
