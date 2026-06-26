Random.seed!(6)

@testset "reduce_stats API" begin
    @testset "isotropic factors, multi-stat" begin
        data = randn(120, 120)
        r = reduce_stats(data, [4, 8, 12]; stats = (Mean(), Var(), Min(), Max()))
        @test Set(factors(r)) == Set([(4, 4), (8, 8), (12, 12)])
        for f in factors(r)
            nt = r[f]
            @test nt.mean ≈ brute(mean, data, f)
            @test nt.var ≈ brute(x -> var(x; corrected = true), data, f)
            @test nt.min ≈ brute(minimum, data, f)
            @test nt.max ≈ brute(maximum, data, f)
            @test r(f, Mean()) === nt.mean
        end
    end

    @testset "anisotropic / partial-dim factors" begin
        data = randn(96, 48, 24)
        r = reduce_stats(data, [(4, 4, 1), (8, 8, 1), (8, 4, 1)]; stats = (Mean(),))
        for f in factors(r)
            @test r[f].mean ≈ brute(mean, data, f)
        end
    end

    @testset "tower spec" begin
        data = randn(96, 72)
        r = reduce_stats(data, Tower(base_factor = 2, steps = [2, 3], maxfactor = 24); stats = (Mean(), Std()))
        @test length(factors(r)) > 3
        for f in factors(r)
            @test r[f].mean ≈ brute(mean, data, f)
            @test r[f].std ≈ brute(x -> std(x; corrected = true), data, f)
        end
    end

    @testset "covariance two fields" begin
        x = randn(100, 100); y = randn(100, 100)
        r = reduce_stats(x, y, [5, 10, 25]; stats = (Cov(corrected = true),))
        for f in factors(r)
            @test r[f].cov ≈ brute_cov(x, y, f)
        end
    end

    @testset "Float32 narrowing; Count Int; subsumption" begin
        data = randn(Float32, 100, 100)
        r = reduce_stats(data, [10, 20]; stats = (Mean(), Var()))
        @test eltype(r[(10, 10)].mean) === Float32
        @test r[(10, 10)].mean ≈ brute(mean, data, (10, 10))
        r2 = reduce_stats(randn(60, 60), [10]; stats = (Count(), Sum(), Mean(), Var()))
        @test eltype(r2[(10, 10)].count) === Int && all(r2[(10, 10)].count .== 100)
        C, routing, _, _ = BSR._assemble((Count(), Sum(), Mean(), Var()), Float64)
        @test C === CompositeAccumulator{Tuple{VarAcc{Float64}}} && all(==(1), routing)
    end

    @testset "arity mismatch errors" begin
        data = randn(20, 20)
        @test_throws ArgumentError reduce_stats(data, [4]; stats = (Cov(),))
        @test_throws ArgumentError reduce_stats(data, data, [4]; stats = (Mean(),))
    end

    @testset "multi-scale agrees with single-scale" begin
        data = randn(128, 128)
        multi = reduce_stats(data, [2, 4, 8, 16, 32]; stats = (Mean(), Var()))
        for f in [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
            single = reduce_stats(data, f; stats = (Mean(), Var()))
            @test multi[f].mean ≈ single[f].mean && multi[f].var ≈ single[f].var
        end
    end
end
