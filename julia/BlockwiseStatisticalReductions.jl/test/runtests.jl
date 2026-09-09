using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions

# The shared brute-force references, once for every file below. To run one file on its own, include this
# first: `include("test/testutils.jl"); include("test/test_api.jl")`.
include("testutils.jl")

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
    include("test_partition.jl")
    include("test_labeled.jl")
    include("test_io.jl")
    include("test_mpi.jl")
    include("test_distributed.jl")
end