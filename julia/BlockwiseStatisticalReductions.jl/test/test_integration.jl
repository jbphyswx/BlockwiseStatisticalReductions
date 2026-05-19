Test.@testset "end-to-end: window -> stats" begin
    # Create test data
    arr = reshape(1.0:10000.0, (100, 100))
    
    # Build and execute plan
    results = build_plan((100, 100)) |>
              rolling_window((10, 10), (10, 10)) |>
              stats(:mean) |>
              execute(arr)
    
    # Should have 10x10 = 100 windows
    Test.@test results[1] isa Vector
    Test.@test length(results[1]) == 100
    
    # Check results are reasonable
    means = [r.data for r in results[1]]
    Test.@test all(m -> 1.0 <= m <= 10000.0, means)
end

Test.@testset "end-to-end: window -> stats -> tree reduce" begin
    arr = reshape(1.0:10000.0, (100, 100))
    
    results = build_plan((100, 100)) |>
              rolling_window((10, 10), (10, 10)) |>
              stats(:var) |>
              tree_reduce(2) |>
              execute(arr)
    
    # Should have single result from tree reduction
    Test.@test results[1] isa ReductionResult
    
    # The variance should be positive
    Test.@test results[1].data > 0
end

Test.@testset "multi-level reduction" begin
    arr = reshape(1.0:1000000.0, (100, 100, 100))
    
    # Large 3D array with multi-stage reduction
    results = build_plan((100, 100, 100)) |>
              rolling_window((20, 20, 20), (20, 20, 20)) |>
              stats([:mean, :var]) |>
              tree_reduce(2) |>
              execute(arr)
    
    Test.@test results[1] isa ReductionResult
end

Test.@testset "disk spillover" begin
    arr = rand(100, 100)
    
    mktempdir() do dir
        results = build_plan((100, 100)) |>
                  rolling_window((10, 10), (10, 10)) |>
                  stats(:mean) |>
                  execute(arr; disk_spill=true, disk_dir=dir)
        
        Test.@test results isa Vector
        Test.@test length(results) == 1
        
        # Check that files were created
        files = readdir(dir)
        Test.@test length(files) > 0
    end
end

Test.@testset "streaming vs batch equivalence" begin
    arr = rand(1000)
    
    # Batch computation
    batch_mean = mean(arr)
    batch_var = var(arr)
    
    # Streaming computation via OnlineStats
    m = Mean(Float64)
    v = Variance(Float64; weight=EqualWeight())
    for x in arr
        fit!(m, x)
        fit!(v, x)
    end
    
    Test.@test value(m) ≈ batch_mean
    Test.@test value(v) ≈ batch_var
end

Test.@testset "tree reduction commutativity" begin
    # Test that merge order doesn't matter for OnlineStats
    arr1 = rand(100)
    arr2 = rand(100)
    arr3 = rand(100)
    arr4 = rand(100)
    
    # Different merge orders
    v1 = Variance(Float64; weight=EqualWeight())
    v2 = Variance(Float64; weight=EqualWeight())
    v3 = Variance(Float64; weight=EqualWeight())
    v4 = Variance(Float64; weight=EqualWeight())
    
    fit_window!(v1, arr1)
    fit_window!(v2, arr2)
    fit_window!(v3, arr3)
    fit_window!(v4, arr4)
    
    # Merge in different orders
    result_a = merge!(merge!(deepcopy(v1), v2), merge!(deepcopy(v3), v4))
    result_b = merge!(merge!(deepcopy(v1), v3), merge!(deepcopy(v2), v4))
    result_c = merge!(merge!(deepcopy(v1), v4), merge!(deepcopy(v2), v3))
    
    Test.@test value(result_a) ≈ value(result_b) ≈ value(result_c)
end

Test.@testset "memory stays bounded with disk spill" begin
    # Create moderately large array
    arr = rand(1000, 1000)
    
    mktempdir() do dir
        # Execute with disk spill
        # Memory usage should not grow with number of windows
        # (in a real implementation, we'd use memory profiling)
        
        results = build_plan((1000, 1000)) |>
                  rolling_window((100, 100), (100, 100)) |>
                  stats(:mean) |>
                  execute(arr; disk_spill=true, disk_dir=dir)
        
        # Should complete without OOM
        Test.@test results isa Vector
        Test.@test length(results) == 1
        Test.@test length(results[1]) == 100  # 10x10 grid
    end
end

Test.@testset "custom user function integration" begin
    arr = reshape(1.0:10000.0, (100, 100))
    
    # Custom reduction: range (max - min)
    results = build_plan((100, 100)) |>
              rolling_window((10, 10), (10, 10)) |>
              user_reduce(x -> maximum(x) - minimum(x), Float64) |>
              execute(arr)
    
    Test.@test results[1] isa Vector
    ranges = [r.data for r in results[1]]
    Test.@test all(r -> r > 0, ranges)
end

Test.@testset "different padding modes" begin
    arr = reshape(1.0:100.0, (10, 10))
    
    # Valid padding
    valid_results = build_plan((10, 10)) |>
                    rolling_window((5, 5); padding=:valid) |>
                    stats(:mean) |>
                    execute(arr)
    Test.@test length(valid_results[1]) == 36  # 6x6 windows
    
    # Same padding
    same_results = build_plan((10, 10)) |>
                   rolling_window((5, 5); padding=:same) |>
                   stats(:mean) |>
                   execute(arr)
    Test.@test length(same_results[1]) == 100  # 10x10 windows
end
