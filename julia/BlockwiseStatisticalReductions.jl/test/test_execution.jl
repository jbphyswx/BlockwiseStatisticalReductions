Test.@testset "simple execution" begin
    arr = reshape(1:100, (10, 10))
    
    # Execute simple plan
    plan = BlockwiseStatisticalReductions.build_plan((10, 10)) |>
           BlockwiseStatisticalReductions.rolling_window((5, 5), (5, 5)) |>
           BlockwiseStatisticalReductions.stats(:mean) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    results = BlockwiseStatisticalReductions.execute(plan, arr)
    Test.@test length(results) >= 1
    Test.@test results[1] isa BlockwiseStatisticalReductions.ReductionResult
end

Test.@testset "execute from builder" begin
    arr = reshape(1:100, (10, 10))
    
    plan = BlockwiseStatisticalReductions.build_plan((10, 10)) |>
              BlockwiseStatisticalReductions.rolling_window((5, 5), (5, 5)) |>
              BlockwiseStatisticalReductions.stats(:mean) |>
              BlockwiseStatisticalReductions.finalize_plan
    results = BlockwiseStatisticalReductions.execute(plan, arr)
    
    Test.@test results isa Tuple || results isa Vector
end

Test.@testset "execution with tree reduce" begin
    arr = reshape(1:100, (10, 10))
    
    plan = BlockwiseStatisticalReductions.build_plan((10, 10)) |>
           BlockwiseStatisticalReductions.rolling_window((5, 5), (5, 5)) |>
           BlockwiseStatisticalReductions.stats(:var) |>
           BlockwiseStatisticalReductions.tree_reduce(2) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    results = BlockwiseStatisticalReductions.execute(plan, arr)
    Test.@test length(results) == 1
    Test.@test results[1] isa BlockwiseStatisticalReductions.ReductionResult
end

Test.@testset "execution with user function" begin
    arr = reshape(1:100, (10, 10))
    
    plan = BlockwiseStatisticalReductions.build_plan((10, 10)) |>
           BlockwiseStatisticalReductions.rolling_window((5, 5), (5, 5)) |>
           BlockwiseStatisticalReductions.user_reduce(x -> maximum(x) - minimum(x), Float64) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    results = BlockwiseStatisticalReductions.execute(plan, arr)
    Test.@test length(results) == 1
end

Test.@testset "execute_node for WindowNode" begin
    arr = reshape(1:100, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5))
    node = BlockwiseStatisticalReductions.WindowNode(cfg, UInt64(1))
    
    cache = BlockwiseStatisticalReductions.PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, arr, BlockwiseStatisticalReductions.CPUBackend(), cache)
    
    Test.@test result isa BlockwiseStatisticalReductions.ReductionResult
end

Test.@testset "execute_node for ReductionNode" begin
    arr = reshape(1.0:100.0, (10, 10))
    cfg = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    node = BlockwiseStatisticalReductions.ReductionNode(BlockwiseStatisticalReductions.blockwise_mean!, cfg, (2, 2), UInt64(1))
    
    cache = BlockwiseStatisticalReductions.PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, arr, BlockwiseStatisticalReductions.CPUBackend(), cache)
    
    Test.@test result isa BlockwiseStatisticalReductions.ReductionResult
    Test.@test size(result.data) == (2, 2)
end

Test.@testset "execute_node for TreeNode" begin
    items = [BlockwiseStatisticalReductions.ReductionResult(1.0, (1,)), BlockwiseStatisticalReductions.ReductionResult(2.0, (1,)),
             BlockwiseStatisticalReductions.ReductionResult(3.0, (1,)), BlockwiseStatisticalReductions.ReductionResult(4.0, (1,))]
    
    node = BlockwiseStatisticalReductions.TreeNode(2, UInt64(1))
    cache = BlockwiseStatisticalReductions.PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, items, BlockwiseStatisticalReductions.CPUBackend(), cache)
    
    Test.@test result isa BlockwiseStatisticalReductions.ReductionResult
    # Tree reduction of [1,2,3,4] with (+): (1+2)=3, (3+4)=7, then 3+7=10
    Test.@test result.data == 10.0
end

Test.@testset "execute_node for UserNode" begin
    arr = [1.0, 2.0, 3.0]
    node = BlockwiseStatisticalReductions.UserNode{typeof(sum)}(sum, Float64, UInt64(1))
    
    cache = BlockwiseStatisticalReductions.PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, arr, BlockwiseStatisticalReductions.CPUBackend(), cache)
    
    Test.@test result isa BlockwiseStatisticalReductions.ReductionResult
    Test.@test result.data == 6.0
end
