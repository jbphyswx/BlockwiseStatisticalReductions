@testset "build_plan" begin
    # Basic plan construction
    builder = build_plan((100, 100, 100))
    @test builder isa ReductionPlanBuilder
    @test builder.input_shape == (100, 100, 100)
end

@testset "plan construction API" begin
    # Simple plan: window -> stats
    plan = build_plan((100, 100)) |> 
           rolling_window((10, 10)) |>
           stats(:mean) |>
           finalize_plan
    
    @test plan isa ReductionPlan
    @test length(plan.nodes) == 2
    @test length(plan.inputs) == 1
    @test length(plan.outputs) == 1
end

@testset "plan validation" begin
    # Valid plan
    plan = build_plan((100, 100)) |> 
           rolling_window((10, 10)) |>
           stats(:mean) |>
           finalize_plan
    
    @test validate(plan)
    
    # Check node types
    @test plan.nodes[1] isa WindowNode
    @test plan.nodes[2] isa StatsNode
end

@testset "plan with tree reduce" begin
    plan = build_plan((100, 100)) |> 
           rolling_window((10, 10)) |>
           stats(:var) |>
           tree_reduce(2) |>
           finalize_plan
    
    @test length(plan.nodes) == 3
    @test plan.nodes[3] isa TreeNode
    @test plan.nodes[3].arity == 2
end

@testset "plan with user function" begin
    my_reduce(x) = sum(x) / length(x)
    
    plan = build_plan((100, 100)) |> 
           rolling_window((5, 5)) |>
           user_reduce(my_reduce, Float64) |>
           finalize_plan
    
    @test plan.nodes[2] isa UserNode
end

@testset "complex plan" begin
    # Multi-stage plan
    plan = build_plan((100, 100, 100)) |> 
           rolling_window((10, 10, 10)) |>
           stats([:mean, :var]) |>
           tree_reduce(2) |>
           finalize_plan
    
    @test validate(plan)
    @test length(plan.nodes) == 3
end

@testset "plan hashing" begin
    plan1 = build_plan((100, 100)) |> 
            rolling_window((10, 10)) |>
            stats(:mean) |>
            finalize_plan
    
    plan2 = build_plan((100, 100)) |> 
            rolling_window((10, 10)) |>
            stats(:mean) |>
            finalize_plan
    
    # Same plan structure should have same hash
    h1 = BlockwiseStatisticalReductions.plan_hash(plan1)
    h2 = BlockwiseStatisticalReductions.plan_hash(plan2)
    @test h1 == h2
    
    # Different plan should have different hash
    plan3 = build_plan((100, 100)) |> 
            rolling_window((5, 5)) |>
            stats(:mean) |>
            finalize_plan
    
    h3 = BlockwiseStatisticalReductions.plan_hash(plan3)
    @test h1 != h3
end

@testset "node id generation" begin
    builder = ReductionPlanBuilder()
    
    id1 = BlockwiseStatisticalReductions.next_id!(builder)
    id2 = BlockwiseStatisticalReductions.next_id!(builder)
    id3 = BlockwiseStatisticalReductions.next_id!(builder)
    
    @test id1 == 1
    @test id2 == 2
    @test id3 == 3
    @test id1 != id2 != id3
end

@testset "find_node" begin
    plan = build_plan((10, 10)) |>
           rolling_window((3, 3)) |>
           stats(:mean) |>
           finalize_plan
    
    node1 = BlockwiseStatisticalReductions.find_node(plan, plan.nodes[1].id)
    @test node1 isa WindowNode
    
    node2 = BlockwiseStatisticalReductions.find_node(plan, plan.nodes[2].id)
    @test node2 isa StatsNode
    
    @test_throws ErrorException BlockwiseStatisticalReductions.find_node(plan, UInt64(999))
end

@testset "topological sort" begin
    plan = build_plan((10, 10)) |>
           rolling_window((3, 3)) |>
           stats(:mean) |>
           tree_reduce(2) |>
           finalize_plan
    
    order = BlockwiseStatisticalReductions.topological_sort(plan)
    
    # Should have all nodes
    @test length(order) == length(plan.nodes)
    
    # Input should come first
    @test order[1] == plan.inputs[1]
    
    # Output should be last
    @test order[end] == plan.outputs[1]
end
