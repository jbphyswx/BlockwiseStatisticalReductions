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
    
    Test.@testset "Types" begin
        include("test_types.jl")
    end
    
    Test.@testset "Windows" begin
        include("test_windows.jl")
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
    
    Test.@testset "Multi-Resolution Plan" begin
        include("test_multires_plan.jl")
    end
    
    Test.@testset "Buffer Pool" begin
        include("test_buffer_pool.jl")
    end
    
    Test.@testset "Hybrid Mode" begin
        include("test_hybrid_mode.jl")
    end
    
    Test.@testset "Public API" begin
        include("test_public_api.jl")
    end
    
    Test.@testset "Integration - Multi-Resolution Workflows" begin
        include("integration/test_multires_workflow.jl")
    end
    
    Test.@testset "Plan" begin
        include("test_plan.jl")
    end
    
    Test.@testset "Execution" begin
        include("test_execution.jl")
    end
    
    Test.@testset "Storage" begin
        include("test_storage.jl")
    end
    
    Test.@testset "Cache" begin
        include("test_cache.jl")
    end
    
    Test.@testset "Integration" begin
        include("test_integration.jl")
    end
end
