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
        # routing encodes the member index in the type (`Val`) for type-stable per-cell finalize
        @test C === CompositeAccumulator{Tuple{VarAcc{Float64}}} && all(r -> r === Val(1), routing)
    end

    @testset "mixed non-subsuming finalize is type-stable (no per-cell boxing)" begin
        data = randn(64, 64)
        r = reduce_stats(data, [8]; stats = (Mean(), Min(), Max()))
        @test keys(r[(8, 8)]) == (:mean, :min, :max)
        # Build the heterogeneous composite (members MeanAcc, MinAcc, MaxAcc) and check that
        # Val-indexed member extraction infers to a concrete Array (the audit-#4 regression guard).
        plan = BSR._plan_for((64, 64), [8])
        C, _, _, _ = BSR._assemble((Mean(), Min(), Max()), Float64)
        @test C === CompositeAccumulator{Tuple{MeanAcc{Float64},MinAcc{Float64},MaxAcc{Float64}}}
        buf = BSR.allocate_tower(plan, C)
        BSR.run!(buf, plan, (data,))
        accs = BSR.step_result(buf, plan.output_steps[1])
        @test (@inferred BSR.materialize(accs, Val(1), Mean(), Float64)) isa Array{Float64}
        @test (@inferred BSR.materialize(accs, Val(2), Min(), Float64)) isa Array{Float64}
        @test (@inferred BSR.materialize(accs, Val(3), Max(), Float64)) isa Array{Float64}
    end

    @testset "plan_dot (DAG visualization)" begin
        plan = solver_plan((120, 96), [(4, 4), (8, 8), (12, 12)])
        d = plan_dot(plan)
        @test occursin("digraph", d)
        @test occursin("input (120, 96)", d)
        @test occursin("->", d)                              # has edges
        @test occursin("(8, 8)", d) && occursin("(12, 12)", d)   # factors labeled
        @test count("->", d) == length(plan.steps)           # one incoming edge per node
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
