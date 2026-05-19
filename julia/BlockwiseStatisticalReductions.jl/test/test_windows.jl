Test.@testset "rolling_views" begin
    # 1D array
    arr1d = 1:100
    
    # Valid padding (default)
    cfg1 = WindowConfig((10,))
    iter1 = rolling_views(arr1d, cfg1)
    results1 = collect(iter1)
    Test.@test length(results1) == 91  # 100 - 10 + 1
    
    # Check first window
    view1, meta1 = results1[1]
    Test.@test vec(view1) == 1:10
    Test.@test meta1[:indices] == (1:10,)
    Test.@test meta1[:center] == (5,)
    
    # Check last window
    view_last, meta_last = results1[end]
    Test.@test vec(view_last) == 91:100
    Test.@test meta1[:out_index] == (1,)
    
    # 2D array
    arr2d = reshape(1:100, (10, 10))
    cfg2 = WindowConfig((3, 3))
    iter2 = rolling_views(arr2d, cfg2)
    results2 = collect(iter2)
    Test.@test length(results2) == 64  # (10-3+1)^2
    
    # With stride
    cfg_stride = WindowConfig((5, 5), (5, 5), :valid)
    iter_stride = rolling_views(arr2d, cfg_stride)
    results_stride = collect(iter_stride)
    Test.@test length(results_stride) == 4  # 2x2 grid
    
    # Check metadata
    view_s, meta_s = results_stride[1]
    Test.@test meta_s[:window_size] == (5, 5)
end

Test.@testset "tiled_blocks" begin
    arr2d = reshape(1:100, (10, 10))
    
    # 2x2 tiles
    iter = tiled_blocks(arr2d, (5, 5))
    results = collect(iter)
    Test.@test length(results) == 4
    
    # Check tile contents
    Test.@test vec(results[1][1]) == 1:5:46  # First tile (column-major)
    Test.@test vec(results[4][1]) == 10:5:55  # Last tile
    
    # Different tile sizes (NxM, not just NxN)
    iter2 = tiled_blocks(arr2d, (2, 5))
    results2 = collect(iter2)
    Test.@test length(results2) == 10  # 5 x 2
end

Test.@testset "window edge cases" begin
    # Window larger than array
    arr = 1:5
    cfg = WindowConfig((10,), (1,), :valid)
    iter = rolling_views(arr, cfg)
    Test.@test length(iter) == 0
    
    # Same padding
    arr = 1:10
    cfg_same = WindowConfig((5,), (1,), :same)
    iter_same = rolling_views(arr, cfg_same)
    results_same = collect(iter_same)
    Test.@test length(results_same) == 10
    
    # Full padding
    cfg_full = WindowConfig((3,), (1,), :full)
    iter_full = rolling_views(arr, cfg_full)
    results_full = collect(iter_full)
    Test.@test length(results_full) == 12  # 10 + 3 - 1
end

Test.@testset "apply_windowed" begin
    arr = reshape(1:100, (10, 10))
    cfg = WindowConfig((3, 3), (3, 3), :valid)
    
    # Simple application without combine
    results = apply_windowed(arr, cfg) do view, meta
        mean(view)
    end
    
    Test.@test length(results) == 9  # 3x3 grid of 3x3 windows
    Test.@test all(r isa Real for r in results)
    
    # With tree reduction
    total = apply_windowed(arr, cfg; combine=+) do view, meta
        sum(view)
    end
    Test.@test total isa Real
    Test.@test total > 0
end

Test.@testset "tree_reduce_impl" begin
    items = collect(1:8)
    
    # Sum reduction
    result = BlockwiseStatisticalReductions.tree_reduce_impl(items, +, CPUBackend(1))
    Test.@test result == 36  # sum(1:8)
    
    # With multi-threading
    if Threads.nthreads() > 1
        result_mt = BlockwiseStatisticalReductions.tree_reduce_impl(items, +, CPUBackend())
        Test.@test result_mt == 36
    end
    
    # Single element
    single = [42]
    result_single = BlockwiseStatisticalReductions.tree_reduce_impl(single, +, CPUBackend())
    Test.@test result_single == 42
    
    # Empty
    empty_arr = Int[]
    result_empty = BlockwiseStatisticalReductions.tree_reduce_impl(empty_arr, +, CPUBackend())
    Test.@test result_empty === nothing
end
