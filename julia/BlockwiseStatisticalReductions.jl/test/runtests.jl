include("testutils.jl")

@testset "BlockwiseStatisticalReductions" begin
    include("test_accumulators.jl")
    include("test_kernels.jl")
    include("test_planner.jl")
    include("test_execute.jl")
    include("test_api.jl")
    include("test_sliding.jl")
    include("test_threading.jl")
    include("test_distributed.jl")
end
