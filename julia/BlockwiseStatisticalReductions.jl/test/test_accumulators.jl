Random.seed!(1)

@testset "accumulators" begin
    @testset "isbits" begin
        for T in (CountAcc, SumAcc{Float64}, MeanAcc{Float64}, VarAcc{Float64}, CovAcc{Float64},
                  MinAcc{Float32}, MaxAcc{Float32}, RawMomentsAcc{4,Float64},
                  CompositeAccumulator{Tuple{VarAcc{Float64},MinAcc{Float64}}})
            @test isbitstype(T)
        end
    end

    @testset "monoid / group laws" begin
        d = randn(64)
        pairs = [(randn(), randn()) for _ in 1:64]
        @test check_monoid(CountAcc; samples = d)
        @test check_monoid(SumAcc{Float64}; samples = d)
        @test check_monoid(MeanAcc{Float64}; samples = d)
        @test check_monoid(VarAcc{Float64}; samples = d)
        @test check_monoid(MinAcc{Float64}; samples = d)
        @test check_monoid(MaxAcc{Float64}; samples = d)
        @test check_monoid(RawMomentsAcc{4,Float64}; samples = d)
        @test check_monoid(CovAcc{Float64}; samples = pairs)
        @test check_monoid(CompositeAccumulator{Tuple{VarAcc{Float64},MinAcc{Float64},MaxAcc{Float64}}}; samples = d)
    end

    @testset "correctness vs brute force" begin
        x = randn(1000); y = randn(1000)
        mx, my = mean(x), mean(y)
        accV = foldl(merge, [lift(VarAcc{Float64}, xi) for xi in x])
        @test result_value(Mean(), accV, Float64) ≈ mx
        @test result_value(Var(corrected = true), accV, Float64) ≈ var(x; corrected = true)
        @test result_value(Var(corrected = false), accV, Float64) ≈ var(x; corrected = false)
        @test result_value(Std(), accV, Float64) ≈ std(x; corrected = true)
        @test result_value(Sum(), accV, Float64) ≈ sum(x)
        @test result_value(Count(), accV, Int) == length(x)
        accC = foldl(merge, [lift(CovAcc{Float64}, x[i], y[i]) for i in eachindex(x)])
        @test result_value(Cov(corrected = true), accC, Float64) ≈ cov(x, y; corrected = true)
        accM = foldl(merge, [lift(RawMomentsAcc{3,Float64}, xi) for xi in x])
        m = result_value(Moments(3), accM, Float64)
        @test m[1] ≈ mean(x) && m[2] ≈ mean(x .^ 2) && m[3] ≈ mean(x .^ 3)
    end

    @testset "widening Float32 -> Float64" begin
        @test accumulation_eltype(Float32) === Float64
        @test accumulation_eltype(Float16) === Float32
        @test accumulation_eltype(Int) === Float64
        @test accumulator_type(Var(), Float32) === VarAcc{Float64}
        @test accumulator_type(Min(), Float32) === MinAcc{Float32}
    end

    @testset "inference / type stability" begin
        a = lift(VarAcc{Float64}, 1.0); b = lift(VarAcc{Float64}, 2.0)
        @test @inferred(merge(a, b)) isa VarAcc{Float64}
        @test @inferred(lift(VarAcc{Float64}, 3.0f0)) isa VarAcc{Float64}
        @test @inferred(result_value(Var(), a, Float64)) isa Float64
        ca = lift(CovAcc{Float64}, 1.0, 2.0); cb = lift(CovAcc{Float64}, 3.0, 4.0)
        @test @inferred(merge(ca, cb)) isa CovAcc{Float64}
        B = CompositeAccumulator{Tuple{VarAcc{Float64},MinAcc{Float64}}}
        @test @inferred(merge(lift(B, 1.0), lift(B, 2.0))) isa B
    end

    @testset "subsumption + routing" begin
        @test subsumes(VarAcc{Float64}, MeanAcc{Float64})
        @test subsumes(MeanAcc{Float64}, SumAcc{Float64})
        @test subsumes(RawMomentsAcc{4,Float64}, RawMomentsAcc{2,Float64})
        @test !subsumes(RawMomentsAcc{2,Float64}, RawMomentsAcc{4,Float64})
        @test Set(BSR.minimal_accumulator_set([MeanAcc{Float64}, VarAcc{Float64}, CountAcc])) == Set([VarAcc{Float64}])
        @test BSR.member_for(Mean(), (VarAcc{Float64}, MinAcc{Float64}), Float64) == 1
        @test BSR.member_for(Min(), (VarAcc{Float64}, MinAcc{Float64}), Float64) == 2
    end

    @testset "exact merge-tree reuse" begin
        x = randn(4096)
        fine = [foldl(merge, [lift(VarAcc{Float64}, x[4(b - 1) + j]) for j in 1:4]) for b in 1:1024]
        via_merge = [foldl(merge, [fine[4(c - 1) + j] for j in 1:4]) for c in 1:256]
        direct = [foldl(merge, [lift(VarAcc{Float64}, x[16(c - 1) + j]) for j in 1:16]) for c in 1:256]
        for c in 1:256
            @test result_value(Var(), via_merge[c], Float64) ≈ result_value(Var(), direct[c], Float64)
        end
    end
end
