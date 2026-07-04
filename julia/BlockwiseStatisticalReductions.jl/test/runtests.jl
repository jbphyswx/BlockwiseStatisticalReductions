include("testutils.jl")

@testset "BlockwiseStatisticalReductions" begin
    include("test_quality.jl")
    include("test_accumulators.jl")
    include("test_kernels.jl")
    include("test_planner.jl")
    include("test_execute.jl")
    include("test_api.jl")
    include("test_schedule.jl")
    include("test_prepared.jl")
    include("test_sliding.jl")
    include("test_threading.jl")
    include("test_gpu_ka.jl")
    include("test_distributed.jl")
    include("test_mpi.jl")
end
