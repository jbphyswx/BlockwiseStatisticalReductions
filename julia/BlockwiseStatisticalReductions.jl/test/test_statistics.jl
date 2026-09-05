using Test: Test
using Random: Random
using Statistics: Statistics
using Adapt: Adapt
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

Random.seed!(1)

const ALL_ACCS = (
    BSR.CountAcc, BSR.SumAcc{Float64}, BSR.MeanAcc{Float64}, BSR.VarAcc{Float64}, BSR.CentralMomentsAcc{Float64},
    BSR.RawMomentsAcc{4,Float64}, BSR.MinAcc{Float64}, BSR.MaxAcc{Float32}, BSR.ExtremaAcc{Float64},
    BSR.ProductSumAcc{Float64}, BSR.CovAcc{Float64}, BSR.CorrAcc{Float64},
    BSR.Composite{Tuple{BSR.VarAcc{Float64},BSR.MinAcc{Float64},BSR.CovAcc{Float64}},((1,), (1,), (1, 2))},
)

samples_for(::Type{A}) where {A} = BSR.arity(A) == 1 ? randn(64) : [(randn(), randn()) for _ in 1:64]

# BigFloat reference of the population moments of `x` (and cross moment with `y`).
function reference(x::AbstractVector, y::AbstractVector = x)
    bx, by = BigFloat.(x), BigFloat.(y)
    n = length(x)
    mx, my = sum(bx) / n, sum(by) / n
    dx, dy = bx .- mx, by .- my
    return (n = n, mean = mx, M2 = sum(dx .^ 2), M3 = sum(dx .^ 3), M4 = sum(dx .^ 4), C = sum(dx .* dy), mean2 = my, M2y = sum(dy .^ 2))
end

tree_fold(::Type{A}, xs) where {A} = length(xs) == 1 ? BSR.lift(A, xs[1]) :
    merge(tree_fold(A, xs[1:end÷2]), tree_fold(A, xs[end÷2+1:end]))
tree_fold_accs(parts) = length(parts) == 1 ? parts[1] :
    merge(tree_fold_accs(parts[1:end÷2]), tree_fold_accs(parts[end÷2+1:end]))

Test.@testset "statistics algebra" begin
    Test.@testset "isbits and monoid laws" begin
        for A in ALL_ACCS
            Test.@test isbitstype(A)
            Test.@test BSR.check_monoid(A; samples = samples_for(A))
        end
    end

    # Exact (BigFloat-rounded) partial accumulators of `xs` (and `ys`) over consecutive tiles of `w`.
    function exact_tiles(::Type{A}, xs, ys, w) where {A}
        parts = A[]
        for lo in 1:w:length(xs)
            r = reference(xs[lo:min(lo + w - 1, end)], ys[lo:min(lo + w - 1, end)])
            f = Float64
            push!(parts, A === BSR.VarAcc{Float64} ? BSR.VarAcc(r.n, f(r.mean), f(r.M2)) :
                         A === BSR.CentralMomentsAcc{Float64} ? BSR.CentralMomentsAcc(r.n, f(r.mean), f(r.M2), f(r.M3), f(r.M4)) :
                         A === BSR.CovAcc{Float64} ? BSR.CovAcc(r.n, f(r.mean), f(r.mean2), f(r.C)) :
                         BSR.CorrAcc(r.n, f(r.mean), f(r.mean2), f(r.M2), f(r.M2y), f(r.C)))
        end
        return parts
    end

    # Accuracy limit of hierarchical merging: means are stored in Float64, so second moments carry a
    # relative error of order ulp(|mean|)/scale (1e-16·1e4/1e-3 ≈ 1e-9 at offset 1e4; ≈ 1e-5 at 1e8).
    Test.@testset "accuracy vs BigFloat" begin
        n = 20_000
        for (offset, rtol) in ((0.0, 1e-12), (1e4, 1e-8), (1e8, 1e-4))
            zx = randn(n)
            x = offset .+ 1e-3 .* zx
            y = offset .+ 1e-3 .* (0.8 .* zx .+ 0.6 .* randn(n))
            ref = reference(x, y)
            obs = collect(zip(x, y))
            for A in (BSR.VarAcc{Float64}, BSR.CentralMomentsAcc{Float64})
                for acc in (foldl(merge, (BSR.lift(A, (xi,)) for xi in x)), tree_fold(A, [(xi,) for xi in x]))
                    Test.@test acc.n == n
                    Test.@test acc.mean ≈ Float64(ref.mean) atol = 100 * eps(max(1.0, offset))
                    Test.@test acc.M2 ≈ Float64(ref.M2) rtol = rtol
                    if A === BSR.CentralMomentsAcc{Float64}
                        Test.@test acc.M3 ≈ Float64(ref.M3) atol = rtol * Float64(ref.M2)^1.5 / sqrt(n)
                        Test.@test acc.M4 ≈ Float64(ref.M4) rtol = 10rtol
                    end
                end
            end
            for A in (BSR.CovAcc{Float64}, BSR.CorrAcc{Float64})
                for acc in (foldl(merge, (BSR.lift(A, o) for o in obs)), tree_fold(A, obs))
                    Test.@test acc.mean1 ≈ Float64(ref.mean) atol = 100 * eps(max(1.0, offset))
                    Test.@test acc.C ≈ Float64(ref.C) rtol = rtol
                end
            end
            # Merging exact tile partials (what the planner does after a two-pass base tile).
            for A in (BSR.VarAcc{Float64}, BSR.CentralMomentsAcc{Float64}, BSR.CovAcc{Float64}, BSR.CorrAcc{Float64})
                parts = exact_tiles(A, x, y, 64)
                for acc in (foldl(merge, parts), BSR.combine(A, parts), tree_fold_accs(parts))
                    Test.@test acc.n == n
                    if A === BSR.CovAcc{Float64} || A === BSR.CorrAcc{Float64}
                        Test.@test acc.C ≈ Float64(ref.C) rtol = rtol
                    else
                        Test.@test acc.M2 ≈ Float64(ref.M2) rtol = rtol
                    end
                    A === BSR.CentralMomentsAcc{Float64} && Test.@test acc.M4 ≈ Float64(ref.M4) rtol = 10rtol
                    A === BSR.CorrAcc{Float64} && Test.@test BSR.finalize(BSR.Corr(), acc, Float64) ≈ Float64(ref.C / sqrt(ref.M2 * ref.M2y)) rtol = rtol
                end
            end
        end
    end

    Test.@testset "k-ary merge equals pairwise merge" begin
        for A in (BSR.MeanAcc{Float64}, BSR.VarAcc{Float64}, BSR.CovAcc{Float64}, BSR.SumAcc{Float64}, BSR.CountAcc,
                  BSR.Composite{Tuple{BSR.VarAcc{Float64},BSR.CovAcc{Float64}},((1,), (1, 2))})
            parts = [foldl(merge, (BSR.lift(A, BSR.arity(A) == 1 ? (x,) : (x, 2x + randn())) for x in randn(k) .+ 1e3)) for k in (3, 17, 1, 40, 8)]
            kary = BSR.combine(A, parts)
            pair = foldl(merge, parts)
            Test.@test BSR._approx(kary, pair, 1e-13 * 1e3, 1e-13)
            Test.@test BSR.combine(A, BSR.AbstractAccumulator[]) == BSR.neutral(A)
        end
    end

    Test.@testset "finalize values vs Statistics" begin
        x = randn(500) .* 3 .+ 7
        y = 0.5 .* x .+ randn(500)
        v = foldl(merge, (BSR.lift(BSR.VarAcc{Float64}, (xi,)) for xi in x))
        Test.@test BSR.finalize(BSR.Mean(), v, Float64) ≈ Statistics.mean(x)
        Test.@test BSR.finalize(BSR.Var(), v, Float64) ≈ Statistics.var(x)
        Test.@test BSR.finalize(BSR.Var(; corrected = false), v, Float64) ≈ Statistics.var(x; corrected = false)
        Test.@test BSR.finalize(BSR.Std(), v, Float32) ≈ Float32(Statistics.std(x))
        Test.@test BSR.finalize(BSR.Sum(), v, Float64) ≈ sum(x)
        Test.@test BSR.finalize(BSR.Count(), v, Int) == 500
        c = foldl(merge, (BSR.lift(BSR.CovAcc{Float64}, o) for o in zip(x, y)))
        Test.@test BSR.finalize(BSR.Cov(), c, Float64) ≈ Statistics.cov(x, y)
        Test.@test BSR.finalize(BSR.ProductMean(), c, Float64) ≈ Statistics.mean(x .* y)
        r = foldl(merge, (BSR.lift(BSR.CorrAcc{Float64}, o) for o in zip(x, y)))
        Test.@test BSR.finalize(BSR.Corr(), r, Float64) ≈ Statistics.cor(x, y)
        m = foldl(merge, (BSR.lift(BSR.RawMomentsAcc{3,Float64}, (xi,)) for xi in x))
        Test.@test collect(BSR.finalize(BSR.Moments(3), m, NTuple{3,Float64})) ≈ [Statistics.mean(x .^ k) for k in 1:3]
        Test.@test collect(BSR.finalize(BSR.Moments(2), m, NTuple{2,Float32})) ≈ [Statistics.mean(x .^ k) for k in 1:2]
        cm = foldl(merge, (BSR.lift(BSR.CentralMomentsAcc{Float64}, (xi,)) for xi in x))
        dx = x .- Statistics.mean(x)
        Test.@test collect(BSR.finalize(BSR.CentralMoments(4), cm, NTuple{3,Float64})) ≈ [Statistics.mean(dx .^ k) for k in 2:4]
        Test.@test BSR.finalize(BSR.CentralMoments(2), cm, NTuple{1,Float64})[1] ≈ Statistics.var(x; corrected = false)
        Test.@test BSR.finalize(BSR.Skewness(), cm, Float64) ≈ Statistics.mean(dx .^ 3) / Statistics.mean(dx .^ 2)^1.5
        Test.@test BSR.finalize(BSR.Kurtosis(), cm, Float64) ≈ Statistics.mean(dx .^ 4) / Statistics.mean(dx .^ 2)^2 - 3
        Test.@test BSR.finalize(BSR.Kurtosis(; excess = false), cm, Float64) ≈ Statistics.mean(dx .^ 4) / Statistics.mean(dx .^ 2)^2
        e = foldl(merge, (BSR.lift(BSR.ExtremaAcc{Float64}, (xi,)) for xi in x))
        Test.@test BSR.finalize(BSR.Extrema(), e, Tuple{Float64,Float64}) == extrema(x)
        Test.@test BSR.finalize(BSR.Min(), e, Float64) == minimum(x) && BSR.finalize(BSR.Max(), e, Float64) == maximum(x)
        Test.@test BSR.finalize(BSR.Component(BSR.Var(), :M2), v, Float64) == v.M2
        Test.@test BSR.finalize(BSR.Component(BSR.Cov(:u, :w), :C), c, Float32) == Float32(c.C)
    end

    Test.@testset "eltype rules and names" begin
        Test.@test BSR.accumulation_eltype(Float32) === Float64
        Test.@test BSR.accumulation_eltype(Float16) === Float32
        Test.@test BSR.accumulation_eltype(Int) === Float64
        Test.@test BSR.accumulator_type(BSR.Var(), Float32, Float64) === BSR.VarAcc{Float64}
        Test.@test BSR.accumulator_type(BSR.Min(), Float32, Float64) === BSR.MinAcc{Float32}
        Test.@test BSR.result_eltype(BSR.Count(), Float32) === Int
        Test.@test BSR.result_eltype(BSR.Moments(3), Float32) === NTuple{3,Float32}
        Test.@test BSR.name(BSR.Var()) == :var && BSR.name(BSR.Var(:u)) == :var_u && BSR.name(BSR.Var(2)) == :var_2
        Test.@test BSR.name(BSR.Cov(:u, :w)) == :cov_u_w && BSR.name(BSR.Cov()) == :cov
        Test.@test BSR.name(BSR.Component(BSR.Var(:u), :M2)) == :M2_u
        Test.@test BSR.name(BSR.Component(BSR.Cov(), :C)) == :C
        Test.@test BSR.bindings(BSR.Cov(:u, :w)) == (:u, :w) && BSR.bindings(BSR.Mean()) == (1,)
    end

    Test.@testset "assemble: subsumption and routing" begin
        C, routing, names, outs = BSR.assemble((BSR.Count(), BSR.Sum(), BSR.Mean(), BSR.Var()), (:x,), Float32, Float64)
        Test.@test C === BSR.Composite{Tuple{BSR.VarAcc{Float64}},((1,),)}
        Test.@test routing == (Val(1), Val(1), Val(1), Val(1))
        Test.@test names == (:count, :sum, :mean, :var) && outs == (Int, Float32, Float32, Float32)

        C, routing, names, _ = BSR.assemble((BSR.Mean(), BSR.Min(), BSR.Max()), (:x,), Float64, Float64)
        Test.@test C === BSR.Composite{Tuple{BSR.MeanAcc{Float64},BSR.MinAcc{Float64},BSR.MaxAcc{Float64}},((1,), (1,), (1,))}
        Test.@test routing == (Val(1), Val(2), Val(3))

        stats = (BSR.Var(:u), BSR.Cov(:u, :w), BSR.Mean(:w), BSR.ProductMean(:u, :w), BSR.Count(:u))
        C, routing, names, _ = BSR.assemble(stats, (:u, :w), Float32, Float64)
        Test.@test C === BSR.Composite{Tuple{BSR.VarAcc{Float64},BSR.CovAcc{Float64},BSR.MeanAcc{Float64}},((1,), (1, 2), (2,))}
        Test.@test routing == (Val(1), Val(2), Val(3), Val(2), Val(1))
        Test.@test names == (:var_u, :cov_u_w, :mean_w, :product_mean_u_w, :count_u)
        Test.@test BSR.arity(C) == 2

        Test.@test_throws ArgumentError BSR.assemble((BSR.Var(:q),), (:u, :w), Float64, Float64)
        Test.@test_throws ArgumentError BSR.assemble((BSR.Cov(1, 3),), (:u, :w), Float64, Float64)
        Test.@test_throws ArgumentError BSR.assemble((BSR.Var(), BSR.Var()), (:u,), Float64, Float64)
        Test.@test_throws ArgumentError BSR.assemble((), (:u,), Float64, Float64)

        obs = (1.5f0, -2.0f0)
        c = BSR.lift(C, obs)
        Test.@test BSR.members(c)[1] == BSR.VarAcc(1, 1.5, 0.0)
        Test.@test BSR.members(c)[2] == BSR.CovAcc(1, 1.5, -2.0, 0.0)
        Test.@test BSR.members(c)[3] == BSR.MeanAcc(1, -2.0)
    end

    Test.@testset "AccumulatorArray storage" begin
        A = BSR.VarAcc{Float64}
        aa = BSR.AccumulatorArray(A, zeros(Float32, 2, 2), (4, 3))
        Test.@test size(aa) == (4, 3) && eltype(aa) === A
        Test.@test BSR.component(aa, :mean) isa Array{Float64,2} && BSR.component(aa, :n) isa Array{Int,2}
        vals = [BSR.VarAcc(i, Float64(i) / 3, Float64(i)^2) for i in 1:12]
        for (i, v) in enumerate(vals)
            aa[i] = v
        end
        Test.@test collect(aa) == reshape(vals, 4, 3)
        Test.@test aa[2, 3] == vals[10]
        Test.@test BSR.component(aa, :M2)[3, 1] == 9.0
        Test.@test Test.@inferred(aa[5]) == vals[5]

        au = BSR.AccumulatorArray(A, zeros(Float32, 1), (5,); uniform = (n = 64,))
        Test.@test BSR.component(au, :n) === BSR.Uniform(64)
        au[2] = BSR.VarAcc(64, 1.0, 2.0)
        au[3] = BSR.VarAcc(7, 3.0, 4.0)
        Test.@test au[2] == BSR.VarAcc(64, 1.0, 2.0) && au[3] == BSR.VarAcc(64, 3.0, 4.0)

        C = BSR.Composite{Tuple{BSR.VarAcc{Float64},BSR.MinAcc{Float32},BSR.CovAcc{Float64}},((1,), (1,), (1, 2))}
        ac = BSR.AccumulatorArray(C, zeros(Float32, 1), (3,); uniform = (n = 9,))
        Test.@test BSR.component(ac, :members, :m1, :n) === BSR.Uniform(9) && BSR.component(ac, :members, :m3, :n) === BSR.Uniform(9)
        Test.@test BSR.component(ac, :members, :m2, :m) isa Vector{Float32}
        x = BSR.lift(C, (2.0f0, 5.0f0))
        ac[1] = x
        Test.@test BSR.members(ac[1])[2] == BSR.MinAcc(2.0f0) && BSR.members(ac[1])[3] == BSR.CovAcc(9, 2.0, 5.0, 0.0)
        Test.@test Test.@inferred(ac[1]) isa C

        m = BSR.AccumulatorArray(BSR.RawMomentsAcc{3,Float64}, zeros(1), (2,))
        m[1] = BSR.lift(BSR.RawMomentsAcc{3,Float64}, (2.0,))
        Test.@test m[1].S == (2.0, 4.0, 8.0)
        Test.@test BSR.component(m, :S, :m2) isa Vector{Float64} && BSR.component(m, :S, :m2)[1] == 4.0

        s = similar(aa)
        Test.@test size(s) == size(aa) && BSR.component(s, :mean) !== BSR.component(aa, :mean)
        Test.@test Adapt.adapt(Array, au) isa typeof(au)
        Test.@test_throws BoundsError aa[13]
    end

    Test.@testset "inference" begin
        a = BSR.lift(BSR.VarAcc{Float64}, (1.0,))
        Test.@test Test.@inferred(merge(a, a)) isa BSR.VarAcc{Float64}
        Test.@test Test.@inferred(BSR.lift(BSR.CovAcc{Float64}, (1.0f0, 2))) isa BSR.CovAcc{Float64}
        C = BSR.Composite{Tuple{BSR.VarAcc{Float64},BSR.MinAcc{Float64}},((1,), (1,))}
        c = BSR.lift(C, (1.0,))
        Test.@test Test.@inferred(merge(c, c)) isa C
        Test.@test Test.@inferred(BSR.combine(C, [c, c, c])) isa C
        Test.@test Test.@inferred(BSR.finalize(BSR.Var(), a, Float32)) isa Float32
        Test.@test Test.@inferred(BSR.neutral(C)) isa C
    end
end
