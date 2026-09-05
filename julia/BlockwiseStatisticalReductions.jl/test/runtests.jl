using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions

Test.@testset "BlockwiseStatisticalReductions" begin
    include("test_quality.jl")
    include("test_statistics.jl")
    include("test_geometry.jl")
    include("test_scales.jl")
    include("test_kernels.jl")
    include("test_planner.jl")
    include("test_execute.jl")
    include("test_api.jl")
    include("test_threaded.jl")
    include("test_gpu.jl")
    include("test_weights.jl")
    include("test_labeled.jl")
    include("test_io.jl")
end