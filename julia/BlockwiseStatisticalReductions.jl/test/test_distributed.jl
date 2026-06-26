using Distributed
using SharedArrays   # with Distributed, activates the DistributedExt
Random.seed!(9)

@testset "distributed" begin
    added = false
    if nworkers() == 1
        addprocs(2)          # workers inherit the master's --project
        added = true
    end
    @everywhere using BlockwiseStatisticalReductions
    try
        data = randn(120, 96)
        rser = reduce_stats(data, [4, 8, 12]; stats = (Mean(), Var(), Max()), backend = SerialBackend())
        rdist = reduce_stats(data, [4, 8, 12]; stats = (Mean(), Var(), Max()), backend = DistributedBackend())
        for f in factors(rser)
            @test rdist[f].mean == rser[f].mean    # bit-identical (disjoint output slabs)
            @test rdist[f].var == rser[f].var
            @test rdist[f].max == rser[f].max
        end
        x = randn(100, 80); y = randn(100, 80)
        cser = reduce_stats(x, y, [8]; stats = (Cov(),), backend = SerialBackend())
        cdist = reduce_stats(x, y, [8]; stats = (Cov(),), backend = DistributedBackend())
        @test cdist[(8, 8)].cov == cser[(8, 8)].cov
    finally
        added && rmprocs(workers())
    end
end
