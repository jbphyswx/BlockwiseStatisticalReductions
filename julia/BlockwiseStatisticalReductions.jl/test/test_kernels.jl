Random.seed!(3)

@testset "kernels" begin
    @testset "blockreduce correctness (1D/2D/3D, truncation)" begin
        for (sz, win) in [((100,), (5,)), ((101,), (5,)),
                          ((24, 24), (2, 3)), ((25, 26), (4, 4)),
                          ((12, 8, 6), (2, 2, 3)), ((13, 9, 7), (3, 3, 3))]
            data = randn(sz...)
            mA = blockreduce(VarAcc{Float64}, data, win)
            @test vals(Mean(), mA, Float64) ≈ brute(mean, data, win)
            @test vals(Var(corrected = true), mA, Float64) ≈ brute(x -> var(x; corrected = true), data, win)
            @test vals(Min(), blockreduce(MinAcc{Float64}, data, win), Float64) ≈ brute(minimum, data, win)
            @test vals(Max(), blockreduce(MaxAcc{Float64}, data, win), Float64) ≈ brute(maximum, data, win)
        end
    end

    @testset "coarsen == direct (exact hierarchical reuse)" begin
        data = randn(48, 36)
        fine = blockreduce(VarAcc{Float64}, data, (2, 2))
        coarse = coarsen!(allocate_accumulators(VarAcc{Float64}, (8, 6)), fine, (3, 3))
        direct = blockreduce(VarAcc{Float64}, data, (6, 6))
        @test vals(Var(), coarse, Float64) ≈ vals(Var(), direct, Float64)
        d2 = randn(64, 64)
        l1 = blockreduce(VarAcc{Float64}, d2, (2, 2))
        l2 = coarsen!(allocate_accumulators(VarAcc{Float64}, (16, 16)), l1, (2, 2))
        l3 = coarsen!(allocate_accumulators(VarAcc{Float64}, (8, 8)), l2, (2, 2))
        @test vals(Var(), l3, Float64) ≈ vals(Var(), blockreduce(VarAcc{Float64}, d2, (8, 8)), Float64)
    end

    @testset "covariance (arity 2) + composite" begin
        x = randn(40, 30); y = randn(40, 30)
        c = blockreduce(CovAcc{Float64}, (x, y), (4, 5))
        @test vals(Cov(corrected = true), c, Float64) ≈ brute_cov(x, y, (4, 5))
        C = CompositeAccumulator{Tuple{VarAcc{Float64},MinAcc{Float64},MaxAcc{Float64}}}
        data = randn(30, 20)
        arr = blockreduce(C, data, (3, 4))
        @test map(a -> result_value(Mean(), members(a)[1], Float64), arr) ≈ brute(mean, data, (3, 4))
        @test map(a -> result_value(Min(), members(a)[2], Float64), arr) ≈ brute(minimum, data, (3, 4))
    end

    @testset "Float32 in -> Float64 accumulate" begin
        data = randn(Float32, 100, 100)
        arr = blockreduce(accumulator_type(Var(), Float32), data, (10, 10))
        @test eltype(arr) === VarAcc{Float64}
        @test vals(Var(), arr, Float64) ≈ brute(x -> var(Float64.(x); corrected = true), data, (10, 10))
    end

    @testset "type stability + zero allocation" begin
        data = randn(64, 64)
        out = allocate_accumulators(VarAcc{Float64}, (8, 8))
        @test (@inferred blockreduce!(out, (data,), (8, 8))) === out
        fine = blockreduce(VarAcc{Float64}, data, (2, 2))
        cout = allocate_accumulators(VarAcc{Float64}, (8, 8))
        @test (@inferred coarsen!(cout, fine, (4, 4))) === cout
        blockreduce!(out, (data,), (8, 8)); coarsen!(cout, fine, (4, 4))
        @test (@allocated blockreduce!(out, (data,), (8, 8))) == 0
        @test (@allocated coarsen!(cout, fine, (4, 4))) == 0
        C = CompositeAccumulator{Tuple{VarAcc{Float64},MinAcc{Float64}}}
        cobuf = allocate_accumulators(C, (8, 8)); blockreduce!(cobuf, (data,), (8, 8))
        @test (@allocated blockreduce!(cobuf, (data,), (8, 8))) == 0
    end
end
