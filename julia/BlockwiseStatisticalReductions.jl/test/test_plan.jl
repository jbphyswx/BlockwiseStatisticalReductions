Test.@testset "build_plan" begin
    # Basic plan construction
    builder = build_plan((100, 100, 100))
    Test.@test builder isa ReductionPlanBuilder
    Test.@test builder.input_shape == (100, 100, 100)
end

Test.@testset "plan construction API" begin
    # Simple plan: window -> stats
    plan = build_plan((100, 100)) |> 
           rolling_window((10, 10)) |>
           stats(:mean) |>
           finalize_plan
    
    Test.@test plan isa ReductionPlan
    Test.@test length(plan.nodes) == 2
    Test.@test length(plan.inputs) == 1
    Test.@test length(plan.outputs) == 1
end

Test.@testset "plan validation" begin
    # Valid plan
    plan = build_plan((100, 100)) |> 
           rolling_window((10, 10)) |>
           stats(:mean) |>
           finalize_plan
    
    Test.@test validate(plan)
    
    # Check node types
    Test.@test plan.nodes[1] isa WindowNode
    Test.@test plan.nodes[2] isa StatsNode
end

Test.@testset "plan with tree reduce" begin
    plan = build_plan((100, 100)) |> 
           rolling_window((10, 10)) |>
           stats(:var) |>
           tree_reduce(2) |>
           finalize_plan
    
    Test.@test length(plan.nodes) == 3
    Test.@test plan.nodes[3] isa TreeNode
    Test.@test plan.nodes[3].arity == 2
end

Test.@testset "plan with user function" begin
    my_reduce(x) = sum(x) / length(x)
    
    plan = build_plan((100, 100)) |> 
           rolling_window((5, 5)) |>
           user_reduce(my_reduce, Float64) |>
           finalize_plan
    
    Test.@test plan.nodes[2] isa UserNode
end

Test.@testset "complex plan" begin
    # Multi-stage plan
    plan = build_plan((100, 100, 100)) |> 
           rolling_window((10, 10, 10)) |>
           stats([:mean, :var]) |>
           tree_reduce(2) |>
           finalize_plan
    
    Test.@test validate(plan)
    Test.@test length(plan.nodes) == 3
end

Test.@testset "plan hashing" begin
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
    Test.@test h1 == h2
    
    # Different plan should have different hash
    plan3 = build_plan((100, 100)) |> 
            rolling_window((5, 5)) |>
            stats(:mean) |>
            finalize_plan
    
    h3 = BlockwiseStatisticalReductions.plan_hash(plan3)
    Test.@test h1 != h3
end

Test.@testset "node id generation" begin
    builder = ReductionPlanBuilder()
    
    id1 = BlockwiseStatisticalReductions.next_id!(builder)
    id2 = BlockwiseStatisticalReductions.next_id!(builder)
    id3 = BlockwiseStatisticalReductions.next_id!(builder)
    
    Test.@test id1 == 1
    Test.@test id2 == 2
    Test.@test id3 == 3
    Test.@test id1 != id2 != id3
end

Test.@testset "find_node" begin
    plan = build_plan((10, 10)) |>
           rolling_window((3, 3)) |>
           stats(:mean) |>
           finalize_plan
    
    node1 = BlockwiseStatisticalReductions.find_node(plan, plan.nodes[1].id)
    Test.@test node1 isa WindowNode
    
    node2 = BlockwiseStatisticalReductions.find_node(plan, plan.nodes[2].id)
    Test.@test node2 isa StatsNode
    
    Test.@test_throws ErrorException BlockwiseStatisticalReductions.find_node(plan, UInt64(999))
end

Test.@testset "topological sort" begin
    plan = build_plan((10, 10)) |>
           rolling_window((3, 3)) |>
           stats(:mean) |>
           tree_reduce(2) |>
           finalize_plan
    
    order = BlockwiseStatisticalReductions.topological_sort(plan)
    
    # Should have all nodes
    Test.@test length(order) == length(plan.nodes)
    
    # Input should come first
    Test.@test order[1] == plan.inputs[1]
    
    # Output should be last
    Test.@test order[end] == plan.outputs[1]
end
