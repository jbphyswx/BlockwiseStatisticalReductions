Random.seed!(8)

@testset "sliding (SWAG)" begin
    @testset "1D sliding, any window" begin
        v = randn(200)
        for w in (1, 2, 3, 7, 50)
            for (Acc, st, bf) in [(VarAcc{Float64}, Var(corrected = false), x -> var(x; corrected = false)),
                                  (MeanAcc{Float64}, Mean(), mean),
                                  (MinAcc{Float64}, Min(), minimum),
                                  (MaxAcc{Float64}, Max(), maximum)]
                accs = sliding_reduce(Acc, v, (w,), (1,))
                @test vals(st, accs, Float64) ≈ brute_sliding(bf, v, (w,), (1,), (1,))
            end
        end
    end

    @testset "2D overlapping windows, stride, origin" begin
        data = randn(60, 48)
        for (w, s, o) in [((4, 4), (1, 1), (1, 1)), ((8, 6), (2, 3), (1, 1)),
                          ((5, 5), (3, 2), (2, 1)), ((10, 10), (5, 5), (1, 1))]
            accs = sliding_reduce(VarAcc{Float64}, data, w, s; origin = o)
            @test vals(Mean(), accs, Float64) ≈ brute_sliding(mean, data, w, s, o)
            @test vals(Var(), accs, Float64) ≈ brute_sliding(x -> var(x; corrected = true), data, w, s, o)
        end
    end

    @testset "stride==window reproduces blockwise" begin
        data = randn(64, 48)
        for w in [(2, 2), (4, 4), (8, 6), (4, 3)]
            sld = sliding_reduce(VarAcc{Float64}, data, w, w)
            blk = blockreduce(VarAcc{Float64}, data, w)
            @test vals(Var(), sld, Float64) ≈ vals(Var(), blk, Float64)
            @test vals(Mean(), sld, Float64) ≈ vals(Mean(), blk, Float64)
        end
    end

    @testset "reduce_stats Sliding API + covariance + 3D partial-dim" begin
        data = randn(80, 80)
        r = reduce_stats(data, [Sliding((8, 8); stride = (2, 2)), Sliding((16, 16); stride = (4, 4))];
                         stats = (Mean(), Var(), Max()))
        @test r[(8, 8)].mean ≈ brute_sliding(mean, data, (8, 8), (2, 2), (1, 1))
        @test r[(16, 16)].var ≈ brute_sliding(x -> var(x; corrected = true), data, (16, 16), (4, 4), (1, 1))
        r2 = reduce_stats(data, Sliding(10; stride = 5); stats = (Mean(),))
        @test r2[(10, 10)].mean ≈ brute_sliding(mean, data, (10, 10), (5, 5), (1, 1))

        x = randn(50, 40); y = randn(50, 40)
        rc = reduce_stats(x, y, Sliding((6, 6); stride = (2, 2)); stats = (Cov(corrected = true),))
        bc = brute_sliding(_ -> 0.0, x, (6, 6), (2, 2), (1, 1))
        for I in CartesianIndices(bc)
            p = ntuple(d -> 1 + (I[d] - 1) * 2, 2)
            bc[I] = cov(vec(x[p[1]:p[1]+5, p[2]:p[2]+5]), vec(y[p[1]:p[1]+5, p[2]:p[2]+5]); corrected = true)
        end
        @test rc[(6, 6)].cov ≈ bc

        d3 = randn(20, 16, 12)
        accs = sliding_reduce(MeanAcc{Float64}, d3, (4, 4, 1), (2, 2, 1))
        @test vals(Mean(), accs, Float64) ≈ brute_sliding(mean, d3, (4, 4, 1), (2, 2, 1), (1, 1, 1))
    end
end
