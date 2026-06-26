Random.seed!(4)

# manual plan executor independent of the package executor (cross-check the DAG itself)
function run_plan_manual(plan, ::Type{Acc}, data) where {Acc}
    res = Vector{Array{Acc}}(undef, length(plan.steps))
    for (i, s) in enumerate(plan.steps)
        out = allocate_accumulators(Acc, s.shape)
        s.source == 0 ? blockreduce!(out, (data,), s.window) : coarsen!(out, res[s.source], s.window)
        res[i] = out
    end
    return res
end

@testset "planner" begin
    @testset "lattice helpers" begin
        @test factor_shape((100, 100), (4, 4)) == (25, 25)
        @test factor_shape((101, 100), (4, 4)) == (25, 25)
        @test divides((2, 2), (4, 6)) && !divides((4, 4), (4, 6))
        @test BSR.factor_window((2, 3), (4, 6)) == (2, 2)
        @test Set(reachable_factors((1, 1), ([2], [2]), (8, 8))) ==
              Set([(a, b) for a in (1, 2, 4, 8) for b in (1, 2, 4, 8)])
    end

    @testset "plan validity + optimal parent" begin
        plan = tower_plan((96, 96); base_factor = (2, 2), steps = ([2, 3], [2, 3]), maxfactor = (48, 48))
        present = Set(s.factor for s in plan.steps)
        for (i, s) in enumerate(plan.steps)
            @test s.source < i
            @test s.shape == factor_shape((96, 96), s.factor)
            if s.source == 0
                @test s.window == s.factor
            else
                p = plan.steps[s.source]
                @test divides(p.factor, s.factor) && prod(p.factor) < prod(s.factor)
                @test s.window == BSR.factor_window(p.factor, s.factor)
                bp = maximum(prod(q) for q in present if q != s.factor && divides(q, s.factor))
                @test prod(p.factor) == bp     # largest-divisor (cheapest) parent
            end
        end
    end

    @testset "work bounds and beats naive" begin
        chain1d = tower_plan((4096,); base_factor = (2,), steps = ([2],), maxfactor = (1024,))
        @test plan_work(chain1d) < 2 * 4096
        X = (256, 256)
        plan = tower_plan(X; base_factor = (1, 1), steps = ([2], [2]), maxfactor = (64, 64))
        @test n_base_passes(plan) == 1
        @test plan_work(plan) < naive_work(plan) ÷ 3
    end

    @testset "Steiner sharing optimal on small cases" begin
        X = (360, 360)
        M = BSR._augment_steiner(X, [(4, 4), (6, 6)]; cap = 4096)
        @test (2, 2) in M && total_work(X, M) < total_work(X, [(4, 4), (6, 6)])
        targets = [(4, 4), (6, 6), (9, 9)]
        cands = setdiff(BSR.gcd_closure(targets), targets)
        brute_min = minimum(
            total_work(X, vcat(targets, [cands[j] for j in 1:length(cands) if (mask >> (j - 1)) & 1 == 1]))
            for mask in 0:(2^length(cands) - 1))
        @test total_work(X, BSR._augment_steiner(X, targets; cap = 4096)) == brute_min
    end

    @testset "end-to-end DAG correctness" begin
        data = randn(96, 72)
        plan = tower_plan((96, 72); base_factor = (2, 2), steps = ([2, 3], [2, 3]), maxfactor = (24, 24))
        res = run_plan_manual(plan, VarAcc{Float64}, data)
        for (i, s) in enumerate(plan.steps)
            s.is_output || continue
            @test vals(Var(), res[i], Float64) ≈ brute(x -> var(x; corrected = true), data, s.factor)
        end
        data2 = randn(360, 240)
        plan2 = solver_plan((360, 240), [(4, 6), (6, 4), (12, 12)]; allow_steiner = true)
        res2 = run_plan_manual(plan2, VarAcc{Float64}, data2)
        for (i, s) in enumerate(plan2.steps)
            s.is_output || continue
            @test vals(Var(), res2[i], Float64) ≈ brute(x -> var(x; corrected = true), data2, s.factor)
        end
    end
end
