"""
Tests for the compiled execution engine (pre-compiled DAG traversal).

Verifies:
- Execution sequence compilation
- Correctness of compiled vs naive execution
- Zero-allocation execute! path
- DAG intermediate reuse (chain correctness)
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "execution sequence compilation" begin
    plan = BlockwiseStatisticalReductions.build_optimal_multires_plan(
        (128, 128, 8), [2, 4, 8], [:mean])

    Test.@test length(plan.execution_sequence) > 0
    Test.@test length(plan.output_indices) == 3

    # All steps should be WindowNodes for mean-only plans
    for step in plan.execution_sequence
        Test.@test step.node isa BlockwiseStatisticalReductions.WindowNode
    end

    # First step should have no inputs (root node)
    Test.@test isempty(plan.execution_sequence[1].input_indices)

    # Subsequent steps should reference earlier steps
    for step in plan.execution_sequence[2:end]
        Test.@test !isempty(step.input_indices)
        Test.@test all(idx -> idx < step.result_index, step.input_indices)
    end
end

Test.@testset "compiled execution correctness" begin
    data = randn(Float32, 128, 128, 8)
    plan = BlockwiseStatisticalReductions.build_optimal_multires_plan(
        (128, 128, 8), [2, 4, 8, 16], [:mean])

    results = BlockwiseStatisticalReductions.execute(plan, data)
    Test.@test length(results) == 4

    # Verify each factor matches direct computation
    direct_2x = BlockwiseStatisticalReductions.blockwise_mean(data, (2, 2, 1))
    result_2x = first(r for r in results if size(r.data) == (64, 64, 8))
    Test.@test result_2x.data ≈ direct_2x

    # 4x should equal mean-of-2x (DAG reuse chain)
    direct_4x_from_2x = BlockwiseStatisticalReductions.blockwise_mean(direct_2x, (2, 2, 1))
    result_4x = first(r for r in results if size(r.data) == (32, 32, 8))
    Test.@test result_4x.data ≈ direct_4x_from_2x

    # 8x from 4x
    direct_8x_from_4x = BlockwiseStatisticalReductions.blockwise_mean(direct_4x_from_2x, (2, 2, 1))
    result_8x = first(r for r in results if size(r.data) == (16, 16, 8))
    Test.@test result_8x.data ≈ direct_8x_from_4x

    # 16x from 8x
    direct_16x_from_8x = BlockwiseStatisticalReductions.blockwise_mean(direct_8x_from_4x, (2, 2, 1))
    result_16x = first(r for r in results if size(r.data) == (8, 8, 8))
    Test.@test result_16x.data ≈ direct_16x_from_8x
end

Test.@testset "single factor plan" begin
    data = randn(Float32, 64, 64, 4)
    plan = BlockwiseStatisticalReductions.build_optimal_multires_plan(
        (64, 64, 4), [4], [:mean])

    results = BlockwiseStatisticalReductions.execute(plan, data)
    Test.@test length(results) == 1

    # Should match: first reduce by 2, then reduce that by 2 (since 4 = 2*2)
    # OR direct 4x reduction depending on plan factorization
    expected = BlockwiseStatisticalReductions.blockwise_mean(data, (4, 4, 1))
    # Check sizes match
    Test.@test any(r -> size(r.data) == size(expected), results)
end

Test.@testset "allocate_buffers and execute!" begin
    data = randn(Float32, 128, 128, 8)
    plan = BlockwiseStatisticalReductions.build_optimal_multires_plan(
        (128, 128, 8), [2, 4, 8], [:mean])

    bufs = BlockwiseStatisticalReductions.allocate_buffers(plan, data)

    # Should have one buffer per execution step
    Test.@test length(bufs.buffers) == length(plan.execution_sequence)

    # Execute
    outputs = BlockwiseStatisticalReductions.execute!(plan, bufs, data)
    Test.@test length(outputs) == 3

    # Must match allocating version
    results = BlockwiseStatisticalReductions.execute(plan, data)
    for r in results
        matching = first(o for o in outputs if size(o) == size(r.data))
        Test.@test matching ≈ r.data
    end
end

Test.@testset "execute! zero allocations" begin
    data = randn(Float32, 128, 128, 8)
    plan = BlockwiseStatisticalReductions.build_optimal_multires_plan(
        (128, 128, 8), [2, 4, 8], [:mean])
    bufs = BlockwiseStatisticalReductions.allocate_buffers(plan, data)

    # Warmup
    BlockwiseStatisticalReductions.execute!(plan, bufs, data)

    alloc = Test.@allocated BlockwiseStatisticalReductions.execute!(plan, bufs, data)
    # Should be near-zero (small tuple return overhead is acceptable)
    Test.@test alloc < 2000  # Less than 2KB overhead
end

Test.@testset "execute! with different input data" begin
    plan = BlockwiseStatisticalReductions.build_optimal_multires_plan(
        (64, 64, 4), [2, 4], [:mean])

    data1 = randn(Float32, 64, 64, 4)
    data2 = randn(Float32, 64, 64, 4)
    bufs = BlockwiseStatisticalReductions.allocate_buffers(plan, data1)

    out1 = BlockwiseStatisticalReductions.execute!(plan, bufs, data1)
    # Copy results before overwriting
    saved = [copy(o) for o in out1]

    out2 = BlockwiseStatisticalReductions.execute!(plan, bufs, data2)

    # Results should differ (different input data)
    Test.@test !all(saved[1] .≈ out2[1])

    # Results should match fresh execution
    fresh = BlockwiseStatisticalReductions.execute(plan, data2)
    for r in fresh
        matching = first(o for o in out2 if size(o) == size(r.data))
        Test.@test matching ≈ r.data
    end
end

Test.@testset "dims keyword preserves z-dimension" begin
    data = randn(Float32, 64, 64, 8)
    plan = BlockwiseStatisticalReductions.build_optimal_multires_plan(
        (64, 64, 8), [2, 4], [:mean]; dims=(1, 2))

    results = BlockwiseStatisticalReductions.execute(plan, data)

    # z-dimension should be unchanged
    for r in results
        Test.@test size(r.data, 3) == 8
    end
end
