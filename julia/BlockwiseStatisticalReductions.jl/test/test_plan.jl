Test.@testset "build_plan" begin
    # Basic plan construction
    builder = BlockwiseStatisticalReductions.build_plan((100, 100, 100))
    Test.@test builder isa BlockwiseStatisticalReductions.ReductionPlanBuilder
    Test.@test builder.input_shape == (100, 100, 100)
end

Test.@testset "plan construction API" begin
    # Simple plan: window -> stats
    plan = BlockwiseStatisticalReductions.build_plan((100, 100)) |> 
           BlockwiseStatisticalReductions.rolling_window((10, 10)) |>
           BlockwiseStatisticalReductions.stats(:mean) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    Test.@test plan isa BlockwiseStatisticalReductions.ReductionPlan
    Test.@test length(plan.nodes) == 2
    Test.@test length(plan.inputs) == 1
    Test.@test length(plan.outputs) == 1
end

Test.@testset "plan validation" begin
    # Valid plan
    plan = BlockwiseStatisticalReductions.build_plan((100, 100)) |> 
           BlockwiseStatisticalReductions.rolling_window((10, 10)) |>
           BlockwiseStatisticalReductions.stats(:mean) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    Test.@test BlockwiseStatisticalReductions.validate(plan)
    
    # Check node types
    Test.@test plan.nodes[1] isa BlockwiseStatisticalReductions.WindowNode
    Test.@test plan.nodes[2] isa BlockwiseStatisticalReductions.ReductionNode
end

Test.@testset "plan with tree reduce" begin
    plan = BlockwiseStatisticalReductions.build_plan((100, 100)) |> 
           BlockwiseStatisticalReductions.rolling_window((10, 10)) |>
           BlockwiseStatisticalReductions.stats(:var) |>
           BlockwiseStatisticalReductions.tree_reduce(2) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    Test.@test length(plan.nodes) == 3
    Test.@test plan.nodes[3] isa BlockwiseStatisticalReductions.TreeNode
    Test.@test plan.nodes[3].arity == 2
end

Test.@testset "plan with user function" begin
    my_reduce(x) = sum(x) / length(x)
    
    plan = BlockwiseStatisticalReductions.build_plan((100, 100)) |> 
           BlockwiseStatisticalReductions.rolling_window((5, 5)) |>
           BlockwiseStatisticalReductions.user_reduce(my_reduce, Float64) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    Test.@test plan.nodes[2] isa BlockwiseStatisticalReductions.UserNode
end

Test.@testset "complex plan" begin
    # Multi-stage plan
    plan = BlockwiseStatisticalReductions.build_plan((100, 100, 100)) |> 
           BlockwiseStatisticalReductions.rolling_window((10, 10, 10)) |>
           BlockwiseStatisticalReductions.stats([:mean, :var]) |>
           BlockwiseStatisticalReductions.tree_reduce(2) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    Test.@test BlockwiseStatisticalReductions.validate(plan)
    Test.@test length(plan.nodes) == 3
end

Test.@testset "plan hashing" begin
    plan1 = BlockwiseStatisticalReductions.build_plan((100, 100)) |> 
            BlockwiseStatisticalReductions.rolling_window((10, 10)) |>
            BlockwiseStatisticalReductions.stats(:mean) |>
            BlockwiseStatisticalReductions.finalize_plan
    
    plan2 = BlockwiseStatisticalReductions.build_plan((100, 100)) |> 
            BlockwiseStatisticalReductions.rolling_window((10, 10)) |>
            BlockwiseStatisticalReductions.stats(:mean) |>
            BlockwiseStatisticalReductions.finalize_plan
    
    # Same plan structure should have same hash
    h1 = BlockwiseStatisticalReductions.plan_hash(plan1)
    h2 = BlockwiseStatisticalReductions.plan_hash(plan2)
    Test.@test h1 == h2
    
    # Different plan should have different hash
    plan3 = BlockwiseStatisticalReductions.build_plan((100, 100)) |> 
            BlockwiseStatisticalReductions.rolling_window((5, 5)) |>
            BlockwiseStatisticalReductions.stats(:mean) |>
            BlockwiseStatisticalReductions.finalize_plan
    
    h3 = BlockwiseStatisticalReductions.plan_hash(plan3)
    Test.@test h1 != h3
end

Test.@testset "node id generation" begin
    builder = BlockwiseStatisticalReductions.ReductionPlanBuilder()
    
    id1 = BlockwiseStatisticalReductions.next_id!(builder)
    id2 = BlockwiseStatisticalReductions.next_id!(builder)
    id3 = BlockwiseStatisticalReductions.next_id!(builder)
    
    Test.@test id1 == 1
    Test.@test id2 == 2
    Test.@test id3 == 3
    Test.@test id1 != id2 != id3
end

Test.@testset "find_node" begin
    plan = BlockwiseStatisticalReductions.build_plan((10, 10)) |>
           BlockwiseStatisticalReductions.rolling_window((3, 3)) |>
           BlockwiseStatisticalReductions.stats(:mean) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    node1 = BlockwiseStatisticalReductions.find_node(plan, plan.nodes[1].id)
    Test.@test node1 isa BlockwiseStatisticalReductions.WindowNode
    
    node2 = BlockwiseStatisticalReductions.find_node(plan, plan.nodes[2].id)
    Test.@test node2 isa BlockwiseStatisticalReductions.ReductionNode
    
    Test.@test_throws ErrorException BlockwiseStatisticalReductions.find_node(plan, UInt64(999))
end

Test.@testset "topological sort" begin
    plan = BlockwiseStatisticalReductions.build_plan((10, 10)) |>
           BlockwiseStatisticalReductions.rolling_window((3, 3)) |>
           BlockwiseStatisticalReductions.stats(:mean) |>
           BlockwiseStatisticalReductions.tree_reduce(2) |>
           BlockwiseStatisticalReductions.finalize_plan
    
    order = BlockwiseStatisticalReductions.topological_sort(plan)
    
    # Should have all nodes
    Test.@test length(order) == length(plan.nodes)
    
    # Input should come first
    Test.@test order[1] == plan.inputs[1]
    
    # Output should be last
    Test.@test order[end] == plan.outputs[1]
end
