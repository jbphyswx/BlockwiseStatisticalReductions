using Test: Test
using Random: Random
using Statistics: Statistics
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB
include("testutils.jl")

Random.seed!(21)

Test.@testset "api" begin
    Test.@testset "one field, every builtin, against brute force" begin
        for (shape, scales) in (((60,), [4, 5]), ((48, 36), [4, 6]), ((24, 18, 10), [(4, 3, 2), (8, 9, 5)]))
            x = randn(shape...)
            r = BSR.blockstats(x, scales; stats = (BSR.Count(), BSR.Sum(), BSR.Mean(), BSR.Var(), BSR.Std(),
                                                   BSR.Min(), BSR.Max(), BSR.Moments(2), BSR.Skewness()))
            for w in BSR.windows(r)
                nt = r[w]
                Test.@test nt.count == round.(Int, brute(length, x, w))
                Test.@test nt.sum ≈ brute(sum, x, w)
                Test.@test nt.mean ≈ brute(Statistics.mean, x, w)
                Test.@test nt.var ≈ brute(Statistics.var, x, w)
                Test.@test nt.std ≈ brute(Statistics.std, x, w)
                Test.@test nt.min ≈ brute(minimum, x, w)
                Test.@test nt.max ≈ brute(maximum, x, w)
                Test.@test map(t -> t[2], nt.moments2) ≈ brute(v -> Statistics.mean(v .^ 2), x, w)
                Test.@test nt.skewness ≈ brute(v -> (d = v .- Statistics.mean(v); Statistics.mean(d .^ 3) / Statistics.mean(d .^ 2)^1.5), x, w)
            end
        end
    end

    Test.@testset "edge policies and non-divisible extents" begin
        x = randn(50, 27)
        for edge in (BSR.Truncate(), BSR.Partial(), BSR.Centered())
            r = BSR.blockstats(x, [8]; stats = (BSR.Mean(), BSR.Count()), edge = edge)
            w = BSR.windows(r)[1]
            Test.@test r[w].mean ≈ brute(Statistics.mean, x, w)
            Test.@test r[w].count == round.(Int, brute(length, x, w))
        end
        # Truncate drops the tail; Partial keeps it with a smaller count
        rt = BSR.blockstats(x, [8]; stats = (BSR.Count(),), edge = BSR.Truncate())
        rp = BSR.blockstats(x, [8]; stats = (BSR.Count(),), edge = BSR.Partial())
        Test.@test size(rt[(8, 8)].count) == (6, 3)
        Test.@test size(rp[(8, 8)].count) == (7, 4)
        Test.@test all(==(64), rt[(8, 8)].count)
        Test.@test rp[(8, 8)].count[end, end] == 2 * 3
        Test.@test_throws ArgumentError BSR.blockstats(x, [8]; stats = (BSR.Mean(),), edge = BSR.Strict())
    end

    Test.@testset "several fields in one pass" begin
        u = randn(40, 40); v = randn(40, 40)
        r = BSR.blockstats((u = u, v = v), [8, 10];
                           stats = (BSR.Mean(:u), BSR.Var(:u), BSR.Mean(:v), BSR.Cov(:u, :v), BSR.Corr(:u, :v), BSR.ProductMean(:u, :v)))
        for w in BSR.windows(r)
            nt = r[w]
            Test.@test nt.mean_u ≈ brute(Statistics.mean, u, w)
            Test.@test nt.mean_v ≈ brute(Statistics.mean, v, w)
            Test.@test nt.var_u ≈ brute(Statistics.var, u, w)
            Test.@test nt.cov_u_v ≈ brute2(Statistics.cov, u, v, w)
            Test.@test nt.corr_u_v ≈ brute2(Statistics.cor, u, v, w)
            Test.@test nt.product_mean_u_v ≈ brute2((a, b) -> Statistics.mean(a .* b), u, v, w)
        end
        # the fused pass agrees with computing each field separately
        ru = BSR.blockstats(u, [8]; stats = (BSR.Mean(), BSR.Var()))
        Test.@test r[(8, 8)].mean_u == ru[(8, 8)].mean && r[(8, 8)].var_u == ru[(8, 8)].var
        # positional binding on a plain tuple of fields
        rp = BSR.blockstats((u, v), [8]; stats = (BSR.Cov(1, 2),))
        Test.@test rp[(8, 8)].cov ≈ brute2(Statistics.cov, u, v, BSR.windows(rp)[1])
    end

    Test.@testset "raw numerators through Component" begin
        x = randn(32, 32); y = randn(32, 32)
        r = BSR.blockstats((x = x, y = y), [8];
                           stats = (BSR.Component(BSR.Var(:x), :M2), BSR.Component(BSR.Cov(:x, :y), :C), BSR.Count(:x)))
        w = BSR.windows(r)[1]
        Test.@test r[w].M2_x ≈ brute(v -> sum(abs2, v .- Statistics.mean(v)), x, w)
        Test.@test r[w].C_x_y ≈ brute2((a, b) -> sum((a .- Statistics.mean(a)) .* (b .- Statistics.mean(b))), x, y, w)
        Test.@test all(==(64), r[w].count_x)
    end

    Test.@testset "scale specifications reach the API" begin
        x = randn(120, 96)
        # every generator's lower bound defaults to 1, so the identity window is included unless excluded
        r = BSR.blockstats(x, BSR.ScaleSet(BSR.Smooth((2, 3); max = 24)); stats = (BSR.Mean(),))
        Test.@test Set(first.(BSR.scales(r))) == Set([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24])
        r2 = BSR.blockstats(x, BSR.ScaleSet(BSR.Smooth((2, 3); min = 2, max = 24)); stats = (BSR.Mean(),))
        Test.@test Set(first.(BSR.scales(r2))) == Set([2, 3, 4, 6, 8, 9, 12, 16, 18, 24])
        for w in BSR.windows(r)
            Test.@test r[w].mean ≈ brute(Statistics.mean, x, w)
        end
        # overlapping windows go through the same entry point
        rs = BSR.blockstats(x, BSR.ScaleSet(BSR.Sizes([8, 12]); placement = BSR.Stride(4)); stats = (BSR.Mean(), BSR.Var()))
        for w in BSR.windows(rs)
            Test.@test rs[w].mean ≈ brute(Statistics.mean, x, w)
            Test.@test rs[w].var ≈ brute(Statistics.var, x, w)
        end
        rd = BSR.blockstats(x, BSR.ScaleSet(BSR.Sizes([4, 5, 6]); placement = BSR.Dense()); stats = (BSR.Mean(),))
        Test.@test size(rd[(5, 5)].mean) == (116, 92)
        Test.@test rd[(5, 5)].mean ≈ brute(Statistics.mean, x, BSR.windows(rd)[2])
    end

    Test.@testset "eltypes" begin
        x = randn(Float32, 32, 32)
        r = BSR.blockstats(x, [8]; stats = (BSR.Mean(), BSR.Var(), BSR.Count()))
        Test.@test eltype(r[(8, 8)].mean) === Float32 && eltype(r[(8, 8)].count) === Int
        Test.@test r[(8, 8)].mean ≈ Float32.(brute(Statistics.mean, Float64.(x), BSR.windows(r)[1])) rtol = 1e-5
        r64 = BSR.blockstats(x, [8]; stats = (BSR.Mean(),), out_eltype = Float64)
        Test.@test eltype(r64[(8, 8)].mean) === Float64
        # integer input: a mean is a ratio and widens, a sum stays exact
        ri = BSR.blockstats(rand(1:9, 40, 40), [8]; stats = (BSR.Mean(), BSR.Sum(), BSR.Min()))
        Test.@test eltype(ri[(8, 8)].mean) === Float64
        Test.@test eltype(ri[(8, 8)].sum) === Int && eltype(ri[(8, 8)].min) === Int
        r32 = BSR.blockstats(randn(Float32, 32, 32), [8]; stats = (BSR.Mean(),), acc_eltype = Float32)
        Test.@test eltype(r32[(8, 8)].mean) === Float32
    end

    Test.@testset "prepared handle: reuse, zero allocation, aliasing" begin
        x = randn(64, 64)
        p = BSR.prepare(x, [4, 8]; stats = (BSR.Mean(), BSR.Var()))
        r1 = BSR.blockstats!(p, x)
        Test.@test r1[(4, 4)].mean ≈ brute(Statistics.mean, x, BSR.windows(r1)[1])
        Test.@test (Test.@inferred BSR.blockstats!(p, x)) isa BSR.ScaleResults
        Test.@test (@allocated BSR.blockstats!(p, x)) == 0
        # the result aliases the handle and follows new data
        y = fill(2.0, 64, 64)
        r2 = BSR.blockstats!(p, y)
        Test.@test r2 === r1
        Test.@test all(≈(2.0), r2[(8, 8)].mean) && all(≈(0.0; atol = 1e-12), r2[(8, 8)].var)
        # a one-shot call matches the prepared one
        Test.@test BSR.blockstats(x, [4, 8]; stats = (BSR.Mean(), BSR.Var()))[(8, 8)].var ≈ BSR.blockstats!(p, x)[(8, 8)].var
        pm = BSR.prepare((a = x, b = x), [8]; stats = (BSR.Cov(:a, :b),))
        BSR.blockstats!(pm, (a = x, b = x))
        Test.@test (@allocated BSR.blockstats!(pm, (a = x, b = x))) == 0
    end

    Test.@testset "results container" begin
        x = randn(64, 48)
        r = BSR.blockstats(x, [4, 8, 16]; stats = (BSR.Mean(),))
        Test.@test length(r) == 3
        Test.@test BSR.scales(r) == [(4, 4), (8, 8), (16, 16)]
        Test.@test BSR.shapes(r) == [(16, 12), (8, 6), (4, 3)]
        Test.@test collect(keys(r)) == BSR.scales(r)
        Test.@test r[(8, 8)] === r[2] === r[BSR.windows(r)[2]]
        Test.@test BSR.haskey(r, (8, 8)) && !BSR.haskey(r, (7, 7))
        Test.@test_throws KeyError r[(7, 7)]
        g = BSR.geometry(r, (8, 8))
        Test.@test g.window == BSR.windows(r)[2]
        Test.@test g.origins[1] == 0:8:56 && g.ranges[1][2] == 9:16
        Test.@test length(collect(r)) == 3
        # a size shared by two different placements is ambiguous and says so
        amb = BSR.blockstats(x, [(BSR.tiled(64, 8, BSR.Truncate()), BSR.tiled(48, 8, BSR.Truncate())),
                                 (BSR.strided(64, 8, 4, BSR.Truncate()), BSR.strided(48, 8, 4, BSR.Truncate()))];
                             stats = (BSR.Mean(),))
        Test.@test_throws ArgumentError amb[(8, 8)]
        Test.@test amb[BSR.windows(amb)[1]] isa NamedTuple
        s = sprint(show, MIME"text/plain"(), r)
        Test.@test occursin("ScaleResults over input (64, 48)", s) && occursin("mean", s)
        Test.@test occursin("ScaleResults{2}", sprint(show, r))
    end

    Test.@testset "explain and show for a prepared request" begin
        p = BSR.prepare(randn(128, 128), [4, 8]; stats = (BSR.Mean(), BSR.Var()))
        s = sprint(io -> BSR.explain(io, p))
        Test.@test occursin("2 requested", s) && occursin("input passes", s) && occursin("peak storage", s)
        Test.@test occursin("input passes", sprint(io -> BSR.explain(io, BSR.blockstats!(p, randn(128, 128)))))
        Test.@test occursin("Prepared request over input (128, 128)", sprint(show, MIME"text/plain"(), p))
        Test.@test occursin("Prepared{2}", sprint(show, p))
    end

    Test.@testset "shifted accumulation stays at the accumulation eltype's epsilon" begin
        # A large mean with a small spread is the case that breaks narrow accumulation: every difference
        # the moment kernels take cancels. Shifting removes the offset, so the error stops depending on it.
        w = BSR.tiled((128, 128), (8, 8), BSR.Truncate())
        for offset in (0.0, 1e2, 1e4)
            x = Float32.(offset .+ 1e-3 .* randn(128, 128))
            ref = Array{Float64}(undef, BSR.shape(w))
            for I in CartesianIndices(ref)
                v = BigFloat.(vec(x[BSR.window_range(w[1], I[1]), BSR.window_range(w[2], I[2])]))
                m = sum(v) / length(v)
                ref[I] = Float64(sum((v .- m) .^ 2) / (length(v) - 1))
            end
            err(r) = maximum(abs.(r[(8, 8)].var .- ref) ./ ref)
            shifted = BSR.blockstats(x, [8]; stats = (BSR.Var(),), out_eltype = Float64)
            Test.@test err(shifted) < 1e-6                       # at Float32's epsilon, whatever the offset
            plain = BSR.blockstats(x, [8]; stats = (BSR.Var(),), shift = false, acc_eltype = Float32, out_eltype = Float64)
            offset >= 1e4 && Test.@test err(plain) > 1e-3        # unshifted Float32 degrades with the offset
        end
        # Float32 input shifts and accumulates narrow by default; Float64 input does neither.
        p32 = BSR.prepare(randn(Float32, 32, 32), [8]; stats = (BSR.Var(),))
        Test.@test BSR.is_shifting(p32) && BSR.shifts(p32) isa Tuple{Float32}
        p64 = BSR.prepare(randn(32, 32), [8]; stats = (BSR.Var(),))
        Test.@test !BSR.is_shifting(p64)
        Test.@test !BSR.is_shifting(BSR.prepare(randn(Float32, 32, 32), [8]; stats = (BSR.Var(),), shift = false))
        Test.@test BSR.shifts(BSR.prepare(randn(32, 32), [8]; stats = (BSR.Var(),), shift = true)) isa Tuple{Float64}
        # a statistic of raw (uncentred) moments cannot be un-shifted, so such a request does not shift
        Test.@test !BSR.is_shifting(BSR.prepare(randn(Float32, 32, 32), [8]; stats = (BSR.Sum(),)))
        # shifting does not change what is reported, for either eltype
        x = randn(Float32, 64, 64); y = randn(Float32, 64, 64)
        a = BSR.blockstats((x = x, y = y), [8]; stats = (BSR.Mean(:x), BSR.Var(:x), BSR.Min(:x), BSR.Cov(:x, :y)), out_eltype = Float64)
        b = BSR.blockstats((x = x, y = y), [8]; stats = (BSR.Mean(:x), BSR.Var(:x), BSR.Min(:x), BSR.Cov(:x, :y)),
                           shift = false, acc_eltype = Float64, out_eltype = Float64)
        Test.@test a[(8, 8)].mean_x ≈ b[(8, 8)].mean_x rtol = 1e-6
        Test.@test a[(8, 8)].var_x ≈ b[(8, 8)].var_x rtol = 1e-5
        Test.@test a[(8, 8)].min_x == b[(8, 8)].min_x            # exact: the shift cancels
        Test.@test a[(8, 8)].cov_x_y ≈ b[(8, 8)].cov_x_y rtol = 1e-5
        # a prepared request re-centres on each new input
        p = BSR.prepare(randn(Float32, 64, 64), [8]; stats = (BSR.Var(),), out_eltype = Float64)
        far = Float32.(1e4 .+ 1e-3 .* randn(64, 64))
        r = BSR.blockstats!(p, far)
        Test.@test BSR.shifts(p)[1] ≈ 1e4 rtol = 1e-3
        Test.@test all(v -> 1e-7 < v < 1e-5, r[(8, 8)].var)      # ≈ (1e-3)^2, not swamped by the offset
        # wrapping each field with its shift costs a few bytes per field per call; what matters is that
        # it does not grow with the input
        small = BSR.prepare(Float32.(1e4 .+ randn(Float32, 64, 64)), [8]; stats = (BSR.Var(),))
        big = BSR.prepare(Float32.(1e4 .+ randn(Float32, 256, 256)), [8]; stats = (BSR.Var(),))
        xs = randn(Float32, 64, 64); xb = randn(Float32, 256, 256)
        BSR.blockstats!(small, xs); BSR.blockstats!(big, xb)
        Test.@test (@allocated BSR.blockstats!(small, xs)) == (@allocated BSR.blockstats!(big, xb)) <= 64
    end

    Test.@testset "errors" begin
        x = randn(32, 32)
        Test.@test_throws ArgumentError BSR.blockstats(x, [8]; stats = ())
        Test.@test_throws ArgumentError BSR.blockstats(x, [8]; stats = (BSR.Cov(),))          # arity 2 on one field
        Test.@test_throws ArgumentError BSR.blockstats((a = x, b = x), [8]; stats = (BSR.Mean(:c),))
        Test.@test_throws ArgumentError BSR.blockstats(x, [8]; stats = (BSR.Mean(), BSR.Mean()))
        Test.@test_throws DimensionMismatch BSR.blockstats((a = x, b = randn(16, 16)), [8]; stats = (BSR.Cov(:a, :b),))
        Test.@test_throws ArgumentError BSR.blockstats(x, [64]; stats = (BSR.Mean(),))        # window exceeds the extent
        p = BSR.prepare(x, [8]; stats = (BSR.Mean(),))
        Test.@test_throws DimensionMismatch BSR.blockstats!(p, randn(16, 16))
        Test.@test_throws ArgumentError BSR.blockstats(x, [8]; stats = (BSR.Mean(),), backend = CB.ThreadedBackend())
    end
end
