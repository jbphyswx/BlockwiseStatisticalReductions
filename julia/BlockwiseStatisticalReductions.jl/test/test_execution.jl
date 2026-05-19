@testset "simple execution" begin
    arr = reshape(1:100, (10, 10))
    
    # Execute simple plan
    plan = build_plan((10, 10)) |>
           rolling_window((5, 5), (5, 5)) |>
           stats(:mean) |>
           finalize_plan
    
    results = execute(plan, arr)
    @test length(results) == 1
    @test results[1] isa Vector  # Multiple window results
    @test length(results[1]) == 4  # 2x2 grid of windows
end

@testset "execute from builder" begin
    arr = reshape(1:100, (10, 10))
    
    results = build_plan((10, 10)) |>
              rolling_window((5, 5), (5, 5)) |>
              stats(:mean) |>
              execute(arr)
    
    @test results isa Vector
end

@testset "execution with tree reduce" begin
    arr = reshape(1:100, (10, 10))
    
    plan = build_plan((10, 10)) |>
           rolling_window((5, 5), (5, 5)) |>
           stats(:var) |>
           tree_reduce(2) |>
           finalize_plan
    
    results = execute(plan, arr)
    @test length(results) == 1
    @test results[1] isa ReductionResult
end

@testset "execution with user function" begin
    arr = reshape(1:100, (10, 10))
    
    plan = build_plan((10, 10)) |>
           rolling_window((5, 5), (5, 5)) |>
           user_reduce(x -> maximum(x) - minimum(x), Float64) |>
           finalize_plan
    
    results = execute(plan, arr)
    @test length(results) == 1
end

@testset "execute_node for WindowNode" begin
    arr = reshape(1:100, (10, 10))
    cfg = WindowConfig((5, 5), (5, 5))
    node = WindowNode(cfg, UInt64(1))
    
    cache = PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, arr, CPUBackend(), cache)
    
    @test result isa Vector
    @test length(result) == 4  # 2x2 windows
    @test result[1] isa ReductionResult
end

@testset "execute_node for StatsNode" begin
    arr = reshape(1.0:100.0, (10, 10))
    stat = Mean{Float64}()
    node = StatsNode{typeof(stat)}(stat, :, UInt64(1))
    
    cache = PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, arr, CPUBackend(), cache)
    
    @test result isa ReductionResult
    @test result.data ≈ mean(arr)
end

@testset "execute_node for TreeNode" begin
    items = [ReductionResult(1.0, Dict()), ReductionResult(2.0, Dict()),
             ReductionResult(3.0, Dict()), ReductionResult(4.0, Dict())]
    
    node = TreeNode(2, UInt64(1))
    cache = PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, items, CPUBackend(), cache)
    
    @test result isa ReductionResult
    # Tree reduction of [1,2,3,4] with (+): (1+2)=3, (3+4)=7, then 3+7=10
    @test result.data == 10.0
end

@testset "execute_node for UserNode" begin
    arr = [1.0, 2.0, 3.0]
    node = UserNode{typeof(sum)}(sum, Float64, UInt64(1))
    
    cache = PlanCache()
    result = BlockwiseStatisticalReductions.execute_node(node, arr, CPUBackend(), cache)
    
    @test result isa ReductionResult
    @test result.data == 6.0
end
