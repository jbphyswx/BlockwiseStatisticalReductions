using Test: Test
using Random: Random
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
include("testutils.jl")

Random.seed!(31)

# Weighted references in BigFloat over the elements of every window; `f` takes (values, weights).
function wbrute(f, x::AbstractArray{T,N}, w::AbstractArray, win::BSR.Window{N}) where {T,N}
    out = Array{Float64,N}(undef, BSR.shape(win))
    for I in CartesianIndices(out)
        rngs = ntuple(d -> BSR.window_range(win[d], I[d]), N)
        out[I] = Float64(f(BigFloat.(vec(collect(view(x, rngs...)))), BigFloat.(vec(collect(view(w, rngs...))))))
    end
    return out
end
wmean(v, ω) = sum(ω .* v) / sum(ω)
wM2(v, ω) = (m = wmean(v, ω); sum(ω .* (v .- m) .^ 2))
wC(v, u, ω) = (sum(ω .* (v .- wmean(v, ω)) .* (u .- wmean(u, ω))))

Test.@testset "weights" begin
    Test.@testset "accumulator laws" begin
        obs1 = [(randn(), rand() + 0.5) for _ in 1:32]
        obs2 = [(randn(), randn(), rand() + 0.5) for _ in 1:32]
        Test.@test BSR.check_monoid(BSR.WMeanAcc{Float64}; samples = obs1)
        Test.@test BSR.check_monoid(BSR.WVarAcc{Float64}; samples = obs1)
        Test.@test BSR.check_monoid(BSR.WCovAcc{Float64}; samples = obs2)
        Test.@test BSR.check_monoid(BSR.WCorrAcc{Float64}; samples = obs2)
        for A in (BSR.WMeanAcc{Float64}, BSR.WVarAcc{Float64}, BSR.WCovAcc{Float64}, BSR.WCorrAcc{Float64})
            Test.@test isbitstype(A)
            Test.@test BSR.shiftable(A)
            Test.@test BSR.arity(A) == (A <: Union{BSR.WMeanAcc,BSR.WVarAcc} ? 2 : 3)
        end
        # A zero weight contributes nothing at all.
        z = BSR.lift(BSR.WVarAcc{Float64}, (1e6, 0.0))
        a = BSR.lift(BSR.WVarAcc{Float64}, (2.0, 3.0))
        Test.@test merge(a, z) === a
        Test.@test BSR.subsumes(BSR.WVarAcc{Float64}, BSR.WMeanAcc{Float64})
        Test.@test BSR.subsumes(BSR.WCorrAcc{Float64}, BSR.WCovAcc{Float64})
    end

    Test.@testset "values against BigFloat" begin
        for (shape, scale) in (((60,), (5,)), ((32, 24), (4, 6)), ((16, 12, 8), (4, 3, 2)))
            x = randn(shape...)
            y = randn(shape...) .* 2 .+ 1
            ω = rand(shape...) .+ 0.5
            st = (mean = BSR.Mean(:x), sum = BSR.Sum(:x), W = BSR.Component(BSR.Mean(:x), :W),
                  vfr = BSR.Var(:x), vpo = BSR.Var(:x; corrected = false),
                  vre = BSR.Var(:x; corrected = :reliability), sd = BSR.Std(:x),
                  cov = BSR.Cov(:x, :y), corr = BSR.Corr(:x, :y), pm = BSR.ProductMean(:x, :y))
            r = BSR.blockstats((x = x, y = y), [scale]; stats = st, weights = ω, backend = CB.SerialBackend())
            win = only(BSR.windows(r))
            nt = r[win]
            Test.@test nt.mean ≈ wbrute(wmean, x, ω, win)
            Test.@test nt.sum ≈ wbrute((v, o) -> sum(o .* v), x, ω, win)
            Test.@test nt.W ≈ wbrute((v, o) -> sum(o), x, ω, win)
            Test.@test nt.vfr ≈ wbrute((v, o) -> wM2(v, o) / (sum(o) - 1), x, ω, win)
            Test.@test nt.vpo ≈ wbrute((v, o) -> wM2(v, o) / sum(o), x, ω, win)
            Test.@test nt.vre ≈ wbrute((v, o) -> wM2(v, o) / (sum(o) - sum(o .^ 2) / sum(o)), x, ω, win)
            Test.@test nt.sd ≈ sqrt.(nt.vfr)
            xy = (I, f) -> begin
                out = Array{Float64}(undef, BSR.shape(win))
                for J in CartesianIndices(out)
                    rs = ntuple(d -> BSR.window_range(win[d], J[d]), length(shape))
                    out[J] = Float64(f(BigFloat.(vec(collect(view(x, rs...)))), BigFloat.(vec(collect(view(y, rs...)))),
                                       BigFloat.(vec(collect(view(ω, rs...))))))
                end
                out
            end
            Test.@test nt.cov ≈ xy(0, (v, u, o) -> wC(v, u, o) / (sum(o) - 1))
            Test.@test nt.corr ≈ xy(0, (v, u, o) -> wC(v, u, o) / sqrt(wM2(v, o) * wM2(u, o)))
            Test.@test nt.pm ≈ xy(0, (v, u, o) -> sum(o .* v .* u) / sum(o))
            # The two corrections are genuinely different formulas for non-unit weights.
            Test.@test !(nt.vfr ≈ nt.vre)
        end
    end

    Test.@testset "unit weights reproduce the unweighted request" begin
        x = randn(40, 32)
        st = (m = BSR.Mean(), v = BSR.Var(), vp = BSR.Var(; corrected = false),
              vr = BSR.Var(; corrected = :reliability), s = BSR.Sum(), n = BSR.Count())
        a = BSR.blockstats(x, [(4, 4), (8, 8)]; stats = st, weights = ones(size(x)), backend = CB.SerialBackend())
        b = BSR.blockstats(x, [(4, 4), (8, 8)]; stats = st, backend = CB.SerialBackend())
        for win in BSR.windows(a), k in keys(st)
            Test.@test a[win][k] == b[win][k]
        end
        # `:reliability` and `:frequency` coincide at unit weights, corrected or not.
        Test.@test a[(8, 8)].v == a[(8, 8)].vr
        Test.@test !(a[(8, 8)].v ≈ a[(8, 8)].vp)
    end

    Test.@testset "separable weights equal the materialized product" begin
        x = randn(24, 16, 8)
        wa, wb, wc = rand(24) .+ 0.5, rand(16) .+ 0.5, rand(8) .+ 0.5
        st = (m = BSR.Mean(), v = BSR.Var(), s = BSR.Sum())
        scale = [(4, 4, 2)]
        for (factors, full) in (((wa, wb, wc), [wa[i] * wb[j] * wc[k] for i in 1:24, j in 1:16, k in 1:8]),
                                ((nothing, wb, nothing), [wb[j] for i in 1:24, j in 1:16, k in 1:8]),
                                ((wa, nothing, wc), [wa[i] * wc[k] for i in 1:24, j in 1:16, k in 1:8]))
            sep = BSR.blockstats(x, scale; stats = st, weights = factors, backend = CB.SerialBackend())
            mat = BSR.blockstats(x, scale; stats = st, weights = full, backend = CB.SerialBackend())
            for k in keys(st)
                Test.@test sep[(4, 4, 2)][k] ≈ mat[(4, 4, 2)][k]
            end
        end
        # Named factors resolve against `dimnames`, in any order.
        named = BSR.blockstats(x, scale; stats = st, weights = (c = wc, a = wa), dimnames = (:a, :b, :c),
                               backend = CB.SerialBackend())
        pos = BSR.blockstats(x, scale; stats = st, weights = (wa, nothing, wc), backend = CB.SerialBackend())
        Test.@test named[(4, 4, 2)].m == pos[(4, 4, 2)].m
        # Separable weights carry no per-element stream, so they cost no modelled input bytes.
        Test.@test BSR.SeparableWeights((wa, nothing, wc), (24, 16, 8)) isa BSR.WeightSource
        Test.@test eltype(BSR.SeparableWeights((wa, nothing, wc), (24, 16, 8))) === Float64
    end

    Test.@testset "skipnan" begin
        x = randn(32, 24)
        xn = copy(x)
        xn[3, 3] = NaN; xn[9, 7] = Inf; xn[1, 1] = NaN; xn[17, 20] = -Inf
        st = (n = BSR.Count(), m = BSR.Mean(), v = BSR.Var(), s = BSR.Sum(), mn = BSR.Min())
        r = BSR.blockstats(xn, [(4, 4)]; stats = st, skipnan = true, backend = CB.SerialBackend())
        win = only(BSR.windows(r))
        finite(v) = filter(isfinite, v)
        Test.@test r[win].n == round.(Int, brute(v -> length(finite(v)), xn, win))
        Test.@test r[win].m ≈ brute(v -> sum(finite(v)) / length(finite(v)), xn, win)
        Test.@test r[win].s ≈ brute(v -> sum(finite(v)), xn, win)
        Test.@test r[win].mn ≈ brute(v -> minimum(finite(v)), xn, win)
        Test.@test r[win].v ≈ brute(v -> (u = finite(v); sum((u .- sum(u) / length(u)) .^ 2) / (length(u) - 1)), xn, win)
        # Without it the same input poisons every window it touches.
        bad = BSR.blockstats(xn, [(4, 4)]; stats = (m = BSR.Mean(),), backend = CB.SerialBackend())
        Test.@test count(!isfinite, bad[win].m) == 3
        # A window with no finite observation has no mean and no observations.
        allnan = fill(NaN, 4, 4)
        e = BSR.blockstats(allnan, [(4, 4)]; stats = (n = BSR.Count(), m = BSR.Mean()), skipnan = true,
                           backend = CB.SerialBackend())[(4, 4)]
        Test.@test e.n == [0;;]
        Test.@test isnan(only(e.m))
        # Counts vary per cell now, so they cannot be stored once for the whole node.
        p = BSR.prepare(x, [(4, 4)]; stats = (n = BSR.Count(),), skipnan = true, backend = CB.SerialBackend())
        pu = BSR.prepare(x, [(4, 4)]; stats = (n = BSR.Count(),), backend = CB.SerialBackend())
        counts(q) = BSR.component(BSR.member_array(BSR.node_storage(q.workspace, only(q.plan.outputs)), Val(1)), :n)
        Test.@test !(counts(p) isa BSR.Uniform)
        Test.@test counts(pu) isa BSR.Uniform
    end

    Test.@testset "skipnan is per statistic, not per element" begin
        x = randn(16, 16); y = randn(16, 16)
        x[2, 2] = NaN
        r = BSR.blockstats((x = x, y = y), [(4, 4)]; stats = (nx = BSR.Count(:x), ny = BSR.Count(:y),
                                                             mx = BSR.Mean(:x), my = BSR.Mean(:y),
                                                             c = BSR.Cov(:x, :y)),
                           skipnan = true, backend = CB.SerialBackend())[(4, 4)]
        Test.@test r.nx[1, 1] == 15
        Test.@test r.ny[1, 1] == 16                     # y has no gap of its own
        Test.@test r.my ≈ brute(v -> sum(v) / length(v), y, (BSR.tiled(16, 4, BSR.Truncate()), BSR.tiled(16, 4, BSR.Truncate())))
        Test.@test !isnan(r.c[1, 1])                    # the pair drops only the element the gap is in
    end

    Test.@testset "weights and skipnan together, over many scales" begin
        x = randn(48, 32)
        x[5, 5] = NaN
        ω = rand(48, 32) .+ 0.5
        st = (m = BSR.Mean(), v = BSR.Var(), W = BSR.Component(BSR.Mean(), :W))
        many = BSR.blockstats(x, [(2, 2), (4, 4), (8, 8), (16, 8)]; stats = st, weights = ω, skipnan = true,
                              backend = CB.SerialBackend())
        for win in BSR.windows(many)
            one = BSR.blockstats(x, [map(aw -> aw.size, win)]; stats = st, weights = ω, skipnan = true,
                                 backend = CB.SerialBackend())
            for k in keys(st)
                Test.@test approx_nan(many[win][k], one[win][k]; rtol = 1e-12)
            end
        end
        # A shared plan reaches the coarse scales by merging, not by re-reading the input.
        Test.@test count(d -> d isa BSR.Base_, many.plan.how) == 1
    end

    Test.@testset "shifted accumulation leaves the weight alone" begin
        x32 = Float32.(randn(32, 32) .+ 1.0f5)
        ω32 = Float32.(rand(32, 32) .+ 0.5)
        p = BSR.prepare(x32, [(4, 4)]; stats = (m = BSR.Mean(), v = BSR.Var()), weights = ω32,
                        backend = CB.SerialBackend())
        r = BSR.blockstats!(p, x32)
        Test.@test BSR.is_shifting(p)
        Test.@test BSR.shifts(p)[1] > 9.0f4            # the data field is centred
        Test.@test BSR.shifts(p)[2] == 0               # the weight field is not
        win = only(BSR.windows(r))
        Test.@test r[win].m ≈ wbrute(wmean, x32, ω32, win) rtol = 1e-5
        Test.@test r[win].v ≈ wbrute((v, o) -> wM2(v, o) / (sum(o) - 1), x32, ω32, win) rtol = 1e-4
    end

    Test.@testset "backend parity" begin
        x = randn(48, 40); ω = rand(48, 40) .+ 0.5; wv = rand(48) .+ 0.5
        x[7, 7] = NaN
        st = (m = BSR.Mean(), v = BSR.Var(), s = BSR.Sum())
        for (weights, skipnan) in ((ω, false), ((wv, nothing), true), (ω, true))
            ser = BSR.blockstats(x, [(4, 4), (8, 8)]; stats = st, weights = weights, skipnan = skipnan,
                                 backend = CB.SerialBackend())
            thr = BSR.blockstats(x, [(4, 4), (8, 8)]; stats = st, weights = weights, skipnan = skipnan,
                                 backend = CB.ThreadedBackend())
            ka = BSR.blockstats(x, [(4, 4), (8, 8)]; stats = st, weights = weights, skipnan = skipnan,
                                backend = CB.GPUBackend(KA.CPU()))
            for win in BSR.windows(ser), k in keys(st)
                Test.@test approx_nan(ser[win][k], thr[win][k]; rtol = 0.0)
                Test.@test approx_nan(ser[win][k], ka[win][k]; rtol = 0.0)
            end
        end
    end

    # `@allocated` at a call site whose argument types are not known boxes the arguments; the barrier
    # keeps the measurement about the call itself.
    measure(p, x) = @allocated BSR.blockstats!(p, x)

    Test.@testset "prepared requests stay allocation-free" begin
        x = randn(64, 64); ω = rand(64, 64) .+ 0.5; wv = rand(64) .+ 0.5
        st = (m = BSR.Mean(), v = BSR.Var())
        for kw in ((weights = ω,), (weights = (wv, nothing),), (skipnan = true,), (weights = ω, skipnan = true))
            p = BSR.prepare(x, [(4, 4), (8, 8)]; stats = st, backend = CB.SerialBackend(), kw...)
            BSR.blockstats!(p, x)
            Test.@test measure(p, x) == 0
            Test.@test BSR.blockstats!(p, x) isa BSR.ScaleResults
        end
        p = BSR.prepare(x, [(4, 4)]; stats = st, weights = ω, backend = CB.SerialBackend())
        Test.@test (Test.@inferred BSR.blockstats!(p, x)) isa BSR.ScaleResults
    end

    Test.@testset "rejected requests" begin
        x = randn(16, 12); ω = rand(16, 12) .+ 0.5
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Min(),), weights = ω)
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Extrema(),), weights = ω)
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Skewness(),), weights = ω)
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Moments(2),), weights = ω)
        Test.@test_throws DimensionMismatch BSR.blockstats(x, [(4, 4)]; stats = (BSR.Mean(),), weights = rand(4, 4))
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Mean(),), weights = (rand(16),))
        Test.@test_throws DimensionMismatch BSR.blockstats(x, [(4, 4)]; stats = (BSR.Mean(),),
                                                           weights = (rand(9), nothing))
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Mean(),), weights = (a = rand(16),))
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Mean(),), weights = (z = rand(16),),
                                                       dimnames = (:a, :b))
        Test.@test_throws ArgumentError BSR.blockstats(x, [(4, 4)]; stats = (BSR.Mean(),),
                                                       weights = (nothing, nothing))
        Test.@test_throws ArgumentError BSR.Var(; corrected = :nope)
        Test.@test_throws ArgumentError BSR.Cov(; corrected = :nope)
        Test.@test_throws ArgumentError BSR.Std(; corrected = :nope)
    end
end
