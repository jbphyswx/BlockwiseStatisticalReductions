"""
Tests for tower construction utilities (src/tower.jl).
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "seed_factor_ladder" begin
    Test.@test BlockwiseStatisticalReductions.seed_factor_ladder(128, 1) == (1, 2, 4, 8, 16, 32, 64, 128)
    Test.@test BlockwiseStatisticalReductions.seed_factor_ladder(128, 3) == (3, 6, 12, 24, 48, 96)
    Test.@test BlockwiseStatisticalReductions.seed_factor_ladder(60, 1) == (1, 2, 4, 8, 16, 32)
    Test.@test BlockwiseStatisticalReductions.seed_factor_ladder(128, 1; min_factor=8) == (8, 16, 32, 64, 128)
    Test.@test BlockwiseStatisticalReductions.seed_factor_ladder(10, 20) == ()  # seed > n
    Test.@test_throws ArgumentError BlockwiseStatisticalReductions.seed_factor_ladder(0, 1)
end

Test.@testset "build_factor_schedule" begin
    sched = BlockwiseStatisticalReductions.build_factor_schedule(128; seeds=(1,))
    Test.@test sched == [1, 2, 4, 8, 16, 32, 64, 128]

    sched2 = BlockwiseStatisticalReductions.build_factor_schedule(60; seeds=(1, 3))
    Test.@test issorted(sched2)
    Test.@test 3 in sched2
    Test.@test 6 in sched2
    Test.@test 60 in sched2  # include_full default

    sched3 = BlockwiseStatisticalReductions.build_factor_schedule(60; seeds=(1,), include_full=false)
    Test.@test !(60 in sched3)  # 60 not a power of 2
end

Test.@testset "build_tower_plan basic" begin
    plan = BlockwiseStatisticalReductions.build_tower_plan((600, 600, 8);
        base_block=(10, 10, 1),
        tower_factors=[2, 3],
        stats=[:mean],
        dims=(1, 2))

    Test.@test length(plan.execution_sequence) > 0
    Test.@test length(plan.output_indices) >= 3

    # Verify DAG chains properly (each step feeds next)
    for (i, step) in enumerate(plan.execution_sequence)
        if i > 1
            # Every non-root step should reference a previous step
            Test.@test !isempty(step.input_indices)
            Test.@test all(idx -> idx < step.result_index, step.input_indices)
        end
    end
end

Test.@testset "build_tower_plan execution correctness" begin
    data = randn(Float32, 120, 120, 4)
    plan = BlockwiseStatisticalReductions.build_tower_plan((120, 120, 4);
        base_block=(10, 10, 1),
        tower_factors=[2, 3],
        stats=[:mean],
        dims=(1, 2))

    results = BlockwiseStatisticalReductions.execute(plan, data)

    # Base level: 120/10 = 12×12
    base_result = first(r for r in results if size(r.data) == (12, 12, 4))
    expected_base = BlockwiseStatisticalReductions.blockwise_mean(data, (10, 10, 1))
    Test.@test base_result.data ≈ expected_base

    # Second level: 12/2 = 6×6 (effective block = 20×20)
    level2 = first(r for r in results if size(r.data) == (6, 6, 4))
    expected_l2 = BlockwiseStatisticalReductions.blockwise_mean(expected_base, (2, 2, 1))
    Test.@test level2.data ≈ expected_l2

    # Third level: 6/3 = 2×2 (effective block = 60×60)
    level3 = first(r for r in results if size(r.data) == (2, 2, 4))
    expected_l3 = BlockwiseStatisticalReductions.blockwise_mean(expected_l2, (3, 3, 1))
    Test.@test level3.data ≈ expected_l3
end

Test.@testset "build_tower_plan z-dimension preserved" begin
    plan = BlockwiseStatisticalReductions.build_tower_plan((120, 120, 8);
        base_block=(10, 10, 1),
        tower_factors=[2],
        stats=[:mean],
        dims=(1, 2))

    data = randn(Float32, 120, 120, 8)
    results = BlockwiseStatisticalReductions.execute(plan, data)
    for r in results
        Test.@test size(r.data, 3) == 8
    end
end

Test.@testset "build_tower_plan_from_outputs" begin
    plan = BlockwiseStatisticalReductions.build_tower_plan_from_outputs((600, 600, 8),
        [(60, 60, 8), (30, 30, 8), (10, 10, 8)];
        dims=(1, 2))

    Test.@test length(plan.output_indices) == 3

    data = randn(Float32, 600, 600, 8)
    results = BlockwiseStatisticalReductions.execute(plan, data)

    sizes = sort([size(r.data) for r in results], by=prod, rev=true)
    Test.@test (60, 60, 8) in sizes
    Test.@test (30, 30, 8) in sizes
    Test.@test (10, 10, 8) in sizes
end

Test.@testset "build_tower_plan_from_outputs non-even division" begin
    # Non-even division is allowed (truncation/valid padding)
    plan = BlockwiseStatisticalReductions.build_tower_plan_from_outputs(
        (600, 600, 8), [(7, 7, 8)]; dims=(1, 2))
    Test.@test length(plan.output_indices) >= 1
end

Test.@testset "build_tower_plan with non-even base_block" begin
    # 7 doesn't divide 120 evenly but should still work (truncation)
    plan = BlockwiseStatisticalReductions.build_tower_plan((120, 120, 8);
        base_block=(7, 7, 1), tower_factors=[2], dims=(1, 2))
    Test.@test length(plan.output_indices) >= 1
    
    # But block larger than input is still an error
    Test.@test_throws ArgumentError BlockwiseStatisticalReductions.build_tower_plan((120, 120, 8);
        base_block=(200, 200, 1), tower_factors=[2], dims=(1, 2))
end
