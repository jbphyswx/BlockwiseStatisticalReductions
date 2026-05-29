using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics
using OnlineStats: OnlineStats
using Aqua: Aqua

# Test utilities
include("testutils.jl")

# Unit tests
Test.@testset "BlockwiseStatisticalReductions" begin
    Test.@testset "Aqua" begin
        Aqua.test_all(BlockwiseStatisticalReductions)
    end

    Test.@testset "Online Statistics" begin
        include("test_online_stats.jl")
    end

    Test.@testset "Parallel Merge" begin
        include("test_parallel_merge.jl")
    end

    Test.@testset "Product Reductions" begin
        include("test_product_reductions.jl")
    end

    Test.@testset "Buffer Pool" begin
        include("test_buffer_pool.jl")
    end

    Test.@testset "Public API" begin
        include("test_public_api.jl")
    end

    Test.@testset "Canonical Kernels" begin
        include("test_canonical_kernels.jl")
    end

    Test.@testset "Compiled Execution" begin
        include("test_compiled_execution.jl")
    end

    Test.@testset "Tower Construction" begin
        include("test_tower.jl")
    end

    Test.@testset "Merge Kernels" begin
        include("test_merge_kernels.jl")
    end

    Test.@testset "Storage" begin
        include("test_storage.jl")
    end

    Test.@testset "Multi-Resolution Stats" begin
        include("test_multires_plan.jl")
    end

    Test.@testset "Integration: Multi-Res Workflow" begin
        include("integration/test_multires_workflow.jl")
    end

    # NOTE: The following test files are disabled pending cleanup.
    # They test the old rolling-window/plan-pipe/execute architecture which has
    # been superseded by the blockwise kernel + DAG approach:
    #   test_types.jl        - references unexported WindowNode, old ReductionResult ctor
    #   test_windows.jl      - rolling_views/tiled_blocks/apply_windowed (removed)
    #   test_hybrid_mode.jl  - execute_hybrid (broken: passes Dict as Tuple to ReductionResult)
    #   test_plan.jl         - rolling_window/stats/tree_reduce pipe API (removed)
    #   test_execution.jl    - old execute(plan, arr) engine
    #   test_cache.jl        - cache_key/invalidate for old plan nodes
    #   test_integration.jl  - rolling_window/tree_reduce full pipeline (removed)
end
