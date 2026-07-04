Random.seed!(7)

@testset "schedule" begin
    @testset "scale_ladder semantics" begin
        @test scale_ladder(128) == [1, 2, 4, 8, 16, 32, 64, 128]                        # dyadic
        @test scale_ladder(128; seeds = [1, 3]) == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]  # merged dyadic (no 9)
        # full 2,3-smooth reduction tree: fills 9, 18, 27, 36, 54, 72, 81, 108
        @test scale_ladder(128; steps = [2, 3]) ==
              [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108, 128]
        @test scale_ladder(81; steps = [3]) == [1, 3, 9, 27, 81]
        @test scale_ladder(60; minfactor = 2, include_full = true) == [2, 4, 8, 16, 32, 60]
        @test scale_ladder(1) == [1]
        @test scale_ladder(128; seeds = [200]) == []          # seed past the ceiling ⇒ empty
        @test scale_ladder(100; maxfactor = 16) == [1, 2, 4, 8, 16]
        @test issorted(scale_ladder(1000; steps = [2, 3, 5]))
        @test allunique(scale_ladder(1000; steps = [2, 4]))   # overlapping multipliers dedup
        # the isotropic full tree equals the 1-D reachable-factor closure the planner uses
        @test scale_ladder(128; steps = [2, 3]) == sort(first.(BSR.reachable_factors((1,), ([2, 3],), (128,))))
        @test_throws ArgumentError scale_ladder(0)
        @test_throws ArgumentError scale_ladder(16; steps = [1])
        @test_throws ArgumentError scale_ladder(16; seeds = [0])
    end

    @testset "Ladder → target factors" begin
        X = (128, 128)
        # isotropic: [(v,v) for v in the 1-D ladder], identity dropped
        iso = BSR._ladder_targets(Ladder(seeds = [1, 3], maxfactor = 64), X)
        @test Set(iso) == Set((v, v) for v in scale_ladder(64; seeds = [1, 3]) if v != 1)
        # isotropic full tree: (9,9),(18,18),(27,27) present
        full = BSR._ladder_targets(Ladder(steps = [2, 3], maxfactor = 64), X)
        @test Set(full) == Set((v, v) for v in scale_ladder(64; steps = [2, 3]) if v != 1)
        @test (9, 9) in full && (18, 18) in full && (27, 27) in full
        # excluding a dimension via maxfactor=1 leaves it unreduced
        excl = BSR._ladder_targets(Ladder(seeds = [1], maxfactor = (32, 1)), X)
        @test all(f -> f[2] == 1, excl) && Set(f[1] for f in excl) == Set([2, 4, 8, 16, 32])
        # product: full Cartesian of per-dim ladders
        prod_t = BSR._ladder_targets(Ladder(seeds = [1], maxfactor = (4, 8), combine = :product), X)
        @test Set(prod_t) == Set((a, b) for a in [1, 2, 4] for b in [1, 2, 4, 8] if !(a == 1 && b == 1))
        @test_throws ArgumentError BSR._ladder_targets(Ladder(combine = :bogus), X)
        # per-dimension seeds (tuple of vectors)
        pd = BSR._ladder_targets(Ladder(seeds = ([1], [1, 3]), maxfactor = (8, 8), combine = :product), X)
        @test Set(f[1] for f in pd) ⊆ Set([1, 2, 4, 8]) && (3 in Set(f[2] for f in pd))
    end

    @testset "plan is inferrable and reuses work" begin
        X = (256, 256)
        l = Ladder(seeds = [1, 3], maxfactor = 64)
        plan = @inferred BSR._plan_for(X, l)
        @test plan isa ReductionPlan{2}
        @test plan_work(plan) < naive_work(plan)           # DAG reuse beats independent reductions
        @test Set(s.factor for s in plan.steps if s.is_output) == Set(BSR._ladder_targets(l, X))
        # a single-seed dyadic ladder is one root ⇒ exactly one base pass, rest are merges
        @test n_base_passes(BSR._plan_for(X, Ladder(seeds = [1], maxfactor = 64))) == 1
        # two coprime seeds (2,3) each root a base pass (2 ∤ 3) — still far below naive
        @test n_base_passes(plan) == 2
    end

    @testset "end-to-end reduce_stats(Ladder) vs brute" begin
        data = randn(120, 96)
        l = Ladder(seeds = [1, 3], maxfactor = 24)
        r = reduce_stats(data, l; stats = (Mean(), Var()))
        @test Set(factors(r)) == Set(BSR._ladder_targets(l, (120, 96)))
        for f in factors(r)
            @test r(f, Mean()) ≈ brute(mean, data, f)
            @test r(f, Var()) ≈ brute(x -> var(x; corrected = true), data, f)
        end
    end
end
