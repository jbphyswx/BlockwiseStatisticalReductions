Random.seed!(11)

@testset "prepared" begin
    @testset "reduce_stats! matches reduce_stats (arity 1)" begin
        data = randn(120, 96)
        for scales in ([4, 8, 16], Tower(steps = [2], maxfactor = 32), Ladder(seeds = [1, 3], maxfactor = 24))
            p = prepare((120, 96), scales; stats = (Mean(), Var()), Tin = Float64)
            rp = reduce_stats!(p, data)
            r = reduce_stats(data, scales; stats = (Mean(), Var()))
            @test Set(factors(rp)) == Set(factors(r))
            for f in factors(r)
                @test rp(f, Mean()) ≈ r(f, Mean())
                @test rp(f, Var()) ≈ r(f, Var())
            end
        end
    end

    @testset "reduce_stats! matches reduce_stats (arity 2 covariance)" begin
        x = randn(64, 64); y = randn(64, 64)
        p = prepare((64, 64), [8]; stats = (Cov(),), Tin = Float64)
        rp = reduce_stats!(p, x, y)
        @test rp((8, 8), Cov()) ≈ reduce_stats(x, y, [8]; stats = (Cov(),))((8, 8), Cov())
    end

    @testset "steady-state allocation-free + result reuse" begin
        data = randn(256, 256)
        p = prepare((256, 256), [4, 8, 16]; stats = (Mean(), Var()), Tin = Float64)
        reduce_stats!(p, data)                       # warmup (compile)
        @test (@allocated reduce_stats!(p, data)) == 0
        # new data reuses the same buffers/result arrays and reflects the update
        data2 = fill(2.0, 256, 256)
        rp = reduce_stats!(p, data2)
        @test all(≈(2.0), rp((4, 4), Mean()))
        @test all(≈(0.0; atol = 1e-9), rp((4, 4), Var()))
    end
end
