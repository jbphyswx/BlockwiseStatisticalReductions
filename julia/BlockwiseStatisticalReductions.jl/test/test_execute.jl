Random.seed!(5)

@testset "executor" begin
    @testset "run! correctness across outputs" begin
        data = randn(96, 72)
        plan = tower_plan((96, 72); base_factor = (2, 2), steps = ([2, 3], [2, 3]), maxfactor = (24, 24))
        buf = allocate_tower(plan, VarAcc{Float64})
        run!(buf, plan, data)
        for (i, s) in enumerate(plan.steps)
            s.is_output || continue
            @test materialize(step_result(buf, i), Var(), Float64) ≈ brute(x -> var(x; corrected = true), data, s.factor)
            @test materialize(step_result(buf, i), Mean(), Float64) ≈ brute(mean, data, s.factor)
        end
    end

    @testset "execute + covariance + composite" begin
        x = randn(120, 80); y = randn(120, 80)
        plan = solver_plan((120, 80), [(4, 4), (8, 8), (20, 20)])
        buf = execute(plan, CovAcc{Float64}, (x, y))
        for (i, s) in enumerate(plan.steps)
            s.is_output || continue
            @test materialize(step_result(buf, i), Cov(), Float64) ≈ brute_cov(x, y, s.factor)
        end
        C = CompositeAccumulator{Tuple{VarAcc{Float64},MinAcc{Float64},MaxAcc{Float64}}}
        data = randn(64, 64)
        plan2 = tower_plan((64, 64); base_factor = (2, 2), steps = ([2], [2]), maxfactor = (16, 16))
        buf2 = execute(plan2, C, data)
        for (i, s) in enumerate(plan2.steps)
            s.is_output || continue
            @test materialize(step_result(buf2, i), 1, Var(), Float64) ≈ brute(x -> var(x; corrected = true), data, s.factor)
            @test materialize(step_result(buf2, i), 2, Min(), Float64) ≈ brute(minimum, data, s.factor)
            @test materialize(step_result(buf2, i), 3, Max(), Float64) ≈ brute(maximum, data, s.factor)
        end
    end

    @testset "type stability + zero-alloc reuse" begin
        data = randn(128, 128)
        plan = tower_plan((128, 128); base_factor = (2, 2), steps = ([2], [2]), maxfactor = (32, 32))
        buf = allocate_tower(plan, VarAcc{Float64})
        @test (@inferred run!(buf, plan, (data,))) === buf
        run!(buf, plan, (data,))
        @test (@allocated run!(buf, plan, (data,))) == 0
        x = randn(128, 128); y = randn(128, 128)
        bufc = allocate_tower(plan, CovAcc{Float64}); run!(bufc, plan, (x, y))
        @test (@allocated run!(bufc, plan, (x, y))) == 0
        C = CompositeAccumulator{Tuple{VarAcc{Float64},MinAcc{Float64}}}
        bufk = allocate_tower(plan, C); run!(bufk, plan, (data,))
        @test (@allocated run!(bufk, plan, (data,))) == 0
    end
end
