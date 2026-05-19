using Test
using BlockwiseStatisticalReductions
using Statistics
using OnlineStats
using Aqua

# Test utilities
include("testutils.jl")

# Unit tests
@testset "BlockwiseStatisticalReductions" begin
    @testset "Aqua" begin
        Aqua.test_all(BlockwiseStatisticalReductions)
    end
    
    @testset "Types" begin
        include("test_types.jl")
    end
    
    @testset "Windows" begin
        include("test_windows.jl")
    end
    
    @testset "Statistics" begin
        include("test_statistics.jl")
    end
    
    @testset "Plan" begin
        include("test_plan.jl")
    end
    
    @testset "Execution" begin
        include("test_execution.jl")
    end
    
    @testset "Storage" begin
        include("test_storage.jl")
    end
    
    @testset "Cache" begin
        include("test_cache.jl")
    end
    
    @testset "Integration" begin
        include("test_integration.jl")
    end
end
