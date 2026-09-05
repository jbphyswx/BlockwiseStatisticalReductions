using Test: Test
using Random: Random
using Statistics: Statistics
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB

include("testutils.jl")

Random.seed!(4)
const SERIAL = CB.SerialBackend()

pop_var(v) = Statistics.var(v; corrected = false)
skew(v) = (d = v .- Statistics.mean(v); Statistics.mean(d .^ 3) / Statistics.mean(d .^ 2)^1.5)

Test.@testset "kernels" begin
    Test.@testset "base pass vs brute force" begin
        cases = [
            ((100,), BSR.tiled((100,), (5,), BSR.Truncate())),
            ((101,), BSR.tiled((101,), (5,), BSR.Truncate())),
            ((101,), BSR.tiled((101,), (5,), BSR.Partial())),
            ((26, 25), BSR.tiled((26, 25), (4, 4), BSR.Centered())),
            ((24, 24), BSR.tiled((24, 24), (2, 3), BSR.Truncate())),
            ((25, 26), BSR.tiled((25, 26), (4, 4), BSR.Partial())),
            ((13, 9, 7), BSR.tiled((13, 9, 7), (3, 3, 3), BSR.Truncate())),
            ((13, 9, 7), BSR.tiled((13, 9, 7), (3, 2, 1), BSR.Partial())),
            ((20, 15), (BSR.strided(20, 5, 1, BSR.Truncate()), BSR.strided(15, 4, 2, BSR.Partial()))),
            ((20, 15), (BSR.anchored(20, 6, [0, 3, 14]), BSR.anchored(15, 5, [0, 10]))),
        ]
        for T in (Float64, Float32), (shape, w) in cases
            x = randn(T, shape...)
            y = randn(T, shape...) .+ x
            xr, yr = Float64.(x), Float64.(y)
            for (stat, f) in ((BSR.Mean(), Statistics.mean), (BSR.Var(), Statistics.var), (BSR.Var(; corrected = false), pop_var),
                              (BSR.Std(), Statistics.std), (BSR.Min(), minimum), (BSR.Max(), maximum), (BSR.Sum(), sum),
                              (BSR.Count(), length), (BSR.Skewness(), skew),
                              (BSR.Kurtosis(), v -> (d = v .- Statistics.mean(v); Statistics.mean(d .^ 4) / Statistics.mean(d .^ 2)^2 - 3)))
                any(BSR.window_length(w[d], i) < 3 for d in 1:length(w) for i in 1:BSR.nwindows(w[d])) && stat isa Union{BSR.Skewness,BSR.Kurtosis} && continue
                C, routing, _, _ = BSR.assemble((stat,), (:x,), T, BSR.accumulation_eltype(T))
                out = BSR.boxfold!(allocate(C, w, x), (x,), w, SERIAL)
                got, want = values_of(out, stat, routing[1]), brute(f, xr, w)
                ok = approx_nan(got, want; rtol = (T === Float32 ? 1e-4 : 1e-9))
                ok || println("mismatch: ", T, " ", shape, " ", stat, " max|Δ| = ", maximum(filter(!isnan, abs.(got .- want)); init = 0.0))
                Test.@test ok
            end
            rtol = T === Float32 ? 1e-4 : 1e-9
            C, routing, _, _ = BSR.assemble((BSR.Cov(), BSR.Corr(), BSR.ProductMean()), (:x, :y), T, BSR.accumulation_eltype(T))
            out = BSR.boxfold!(allocate(C, w, x), (x = x, y = y), w, SERIAL)
            Test.@test approx_nan(values_of(out, BSR.Cov(), routing[1]), brute2(Statistics.cov, xr, yr, w); rtol = rtol)
            Test.@test approx_nan(values_of(out, BSR.Corr(), routing[2]), brute2(Statistics.cor, xr, yr, w); rtol = rtol)
            Test.@test approx_nan(values_of(out, BSR.ProductMean(), routing[3]), brute2((a, b) -> Statistics.mean(a .* b), xr, yr, w); rtol = rtol)
            m = BSR.boxfold!(allocate(BSR.RawMomentsAcc{3,Float64}, w, x), (x,), w, SERIAL)
            Test.@test map(t -> t[3], collect(BSR.finalize!(Array{NTuple{3,Float64}}(undef, size(m)), m, BSR.Moments(3), SERIAL))) ≈ brute(v -> Statistics.mean(v .^ 3), xr, w) rtol = rtol
        end
    end

    Test.@testset "multi-field composite in one pass" begin
        u = randn(64, 48)
        v = 0.3 .* u .+ randn(64, 48)
        w = BSR.tiled((64, 48), (8, 6), BSR.Truncate())
        stats = (BSR.Mean(:u), BSR.Var(:u), BSR.Mean(:v), BSR.Var(:v), BSR.Cov(:u, :v), BSR.Min(:v), BSR.Count(:u))
        C, routing, names, _ = BSR.assemble(stats, (:u, :v), Float64, Float64)
        Test.@test fieldcount(C.parameters[1]) == 4
        out = BSR.boxfold!(allocate(C, w, u), (u = u, v = v), w, SERIAL)
        Test.@test values_of(out, BSR.Mean(:u), routing[1]) ≈ brute(Statistics.mean, u, w)
        Test.@test values_of(out, BSR.Var(:v), routing[4]) ≈ brute(Statistics.var, v, w)
        Test.@test values_of(out, BSR.Cov(:u, :v), routing[5]) ≈ brute2(Statistics.cov, u, v, w)
        Test.@test values_of(out, BSR.Min(:v), routing[6]) ≈ brute(minimum, v, w)
        Test.@test all(==(48), values_of(out, BSR.Count(:u), routing[7]))
        Test.@test BSR.component(BSR.member_array(out, routing[1]), :mean) === BSR.component(out, :members, :m1, :mean)
    end

    Test.@testset "coarsen equals direct" begin
        x = randn(96, 72)
        for policy in (BSR.Truncate(), BSR.Partial())
            fine_w = BSR.tiled((96, 72), (2, 3), policy)
            fine = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, fine_w, x), (x,), fine_w, SERIAL)
            coarse_w = BSR.tiled((96, 72), (6, 6), policy)
            parent_w = ntuple(d -> BSR.coarsen_result(fine_w[d], coarse_w[d].size ÷ fine_w[d].size), 2)
            Test.@test all(d -> parent_w[d] == coarse_w[d], 1:2)
            grid_w = ntuple(d -> BSR.tiled(BSR.nwindows(fine_w[d]), coarse_w[d].size ÷ fine_w[d].size, policy), 2)
            coarse = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, coarse_w, x), fine, grid_w, SERIAL)
            direct = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, coarse_w, x), (x,), coarse_w, SERIAL)
            Test.@test values_of(coarse, BSR.Var()) ≈ values_of(direct, BSR.Var()) rtol = 1e-12
            Test.@test values_of(coarse, BSR.Count()) == values_of(direct, BSR.Count())
        end
        C, routing, _, _ = BSR.assemble((BSR.Mean(), BSR.Min(), BSR.Max()), (:x,), Float64, Float64)
        fine_w = BSR.tiled((96, 72), (2, 2), BSR.Truncate())
        fine = BSR.boxfold!(allocate(C, fine_w, x), (x,), fine_w, SERIAL)
        grid_w = BSR.tiled((48, 36), (4, 4), BSR.Truncate())
        coarse = BSR.boxfold!(allocate(C, BSR.tiled((96, 72), (8, 8), BSR.Truncate()), x), fine, grid_w, SERIAL)
        Test.@test values_of(coarse, BSR.Max(), routing[3]) == brute(maximum, x, BSR.tiled((96, 72), (8, 8), BSR.Truncate()))
        Test.@test values_of(coarse, BSR.Mean(), routing[1]) ≈ brute(Statistics.mean, x, BSR.tiled((96, 72), (8, 8), BSR.Truncate()))
    end

    Test.@testset "compose equals direct" begin
        x = randn(40, 30)
        a_w = (BSR.strided(40, 2, 1, BSR.Truncate()), BSR.tiled(30, 3, BSR.Truncate()))
        a = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, a_w, x), (x,), a_w, SERIAL)
        child_w = (BSR.compose_result(a_w[1], a_w[1]), a_w[2])
        Test.@test child_w[1] == BSR.strided(40, 4, 1, BSR.Truncate())
        out = allocate(BSR.VarAcc{Float64}, child_w, x)
        BSR.compose!(out, a, a, 1, 1:37, 3:39, SERIAL)
        direct = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, child_w, x), (x,), child_w, SERIAL)
        Test.@test values_of(out, BSR.Var()) ≈ values_of(direct, BSR.Var()) rtol = 1e-12
        anchors = (BSR.anchored(40, 4, [0, 7, 36]), a_w[2])
        Test.@test BSR.can_compose(a_w[1], a_w[1], anchors[1])
        outa = BSR.compose!(allocate(BSR.VarAcc{Float64}, anchors, x), a, a, 1, [1, 8, 37], [3, 10, 39], SERIAL)
        Test.@test values_of(outa, BSR.Var()) ≈ brute(Statistics.var, x, anchors) rtol = 1e-12
        Test.@test_throws DimensionMismatch BSR.compose!(out, a, a, 1, 1:37, 3:38, SERIAL)
        Test.@test_throws BoundsError BSR.compose!(out, a, a, 1, 1:37, 4:40, SERIAL)
    end

    Test.@testset "scan equals brute sliding" begin
        x = randn(50, 20)
        lifted = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, BSR.tiled((50, 20), (1, 1), BSR.Truncate()), x), (x,),
                              BSR.tiled((50, 20), (1, 1), BSR.Truncate()), SERIAL)
        for size in (1, 2, 7, 50), axis in (1, 2)
            size > Base.size(x, axis) && continue
            for partial in (false, true)
                w = ntuple(d -> d == axis ? BSR.strided(Base.size(x, d), size, 1, partial ? BSR.Partial() : BSR.Truncate()) :
                                            BSR.tiled(Base.size(x, d), 1, BSR.Truncate()), 2)
                out = allocate(BSR.VarAcc{Float64}, w, x)
                BSR.scan!(out, lifted, axis, size, partial, BSR.ScanScratch(BSR.VarAcc{Float64}, size), SERIAL)
                Test.@test values_of(out, BSR.Mean()) ≈ brute(Statistics.mean, x, w)
                Test.@test values_of(out, BSR.Count()) == brute(length, x, w)
                size > 1 && Test.@test values_of(out, BSR.Var(; corrected = false)) ≈ brute(pop_var, x, w) rtol = 1e-10 atol = 1e-12
            end
        end
        w = (BSR.strided(50, 60, 1, BSR.Partial()), BSR.tiled(20, 1, BSR.Truncate()))
        out = BSR.scan!(allocate(BSR.VarAcc{Float64}, w, x), lifted, 1, 60, true, BSR.ScanScratch(BSR.VarAcc{Float64}, 60), SERIAL)
        Test.@test values_of(out, BSR.Mean()) ≈ brute(Statistics.mean, x, w)
        m = BSR.boxfold!(allocate(BSR.MinAcc{Float64}, BSR.tiled((50, 20), (1, 1), BSR.Truncate()), x), (x,),
                         BSR.tiled((50, 20), (1, 1), BSR.Truncate()), SERIAL)
        w = (BSR.strided(50, 5, 1, BSR.Truncate()), BSR.tiled(20, 1, BSR.Truncate()))
        outm = BSR.scan!(allocate(BSR.MinAcc{Float64}, w, x), m, 1, 5, false, BSR.ScanScratch(BSR.MinAcc{Float64}, 5), SERIAL)
        Test.@test values_of(outm, BSR.Min()) == brute(minimum, x, w)
    end

    Test.@testset "finalize eltypes" begin
        x = randn(Float32, 32, 32)
        w = BSR.tiled((32, 32), (4, 4), BSR.Truncate())
        out = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, w, x), (x,), w, SERIAL)
        dst = BSR.finalize!(Array{Float32}(undef, 8, 8), out, BSR.Var(), SERIAL)
        Test.@test dst ≈ Float32.(brute(Statistics.var, x, w))
        Test.@test BSR.finalize!(Array{Int}(undef, 8, 8), out, BSR.Count(), SERIAL) == fill(16, 8, 8)
        Test.@test_throws DimensionMismatch BSR.finalize!(Array{Float32}(undef, 8, 7), out, BSR.Var(), SERIAL)
    end

    Test.@testset "backend stubs" begin
        x = randn(8, 8)
        w = BSR.tiled((8, 8), (2, 2), BSR.Truncate())
        out = allocate(BSR.MeanAcc{Float64}, w, x)
        # MPI has no extension loaded in this session; AutoBackend must be resolved before kernels run.
        Test.@test_throws ArgumentError BSR.boxfold!(out, (x,), w, CB.MPIBackend())
        Test.@test_throws ArgumentError BSR.boxfold!(out, (x,), w, CB.AutoBackend())
        Test.@test_throws ArgumentError BSR.missing_extension(CB.ThreadedBackend())
    end

    Test.@testset "inference and zero allocation" begin
        x = randn(256, 256)
        y = randn(256, 256)
        w = BSR.tiled((256, 256), (8, 8), BSR.Truncate())
        C, _, _, _ = BSR.assemble((BSR.Mean(), BSR.Var(), BSR.Min(), BSR.Cov(:x, :y)), (:x, :y), Float64, Float64)
        out = allocate(C, w, x)
        S = Val(BSR.static_shape(w))
        Test.@test S === Val((8, 8))
        Test.@test Test.@inferred(BSR.boxfold!(out, (x = x, y = y), w, S, SERIAL)) === out
        Test.@test (@allocated BSR.boxfold!(out, (x = x, y = y), w, S, SERIAL)) == 0
        uni = BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, BSR.shape(w); uniform = (n = 64,))
        Test.@test Test.@inferred(BSR.boxfold!(uni, (x,), w, S, SERIAL)) === uni
        Test.@test (@allocated BSR.boxfold!(uni, (x,), w, S, SERIAL)) == 0
        Test.@test BSR.boxfold!(uni, (x,), w, SERIAL) === uni
        big = BSR.tiled((256, 256), (16, 32), BSR.Truncate())
        Test.@test BSR.static_shape(big) == (0, 0)
        outb = allocate(BSR.VarAcc{Float64}, big, x)
        Test.@test (@allocated BSR.boxfold!(outb, (x,), big, Val((0, 0)), SERIAL)) == 0
        Test.@test_throws ArgumentError BSR.boxfold!(outb, (x,), big, Val((8, 0)), SERIAL)
        Test.@test BSR.boxfold!(outb, (x,), big, Val((16, 0)), SERIAL) === outb
        grid = BSR.tiled((32, 32), (2, 2), BSR.Truncate())
        coarse = allocate(C, BSR.tiled((256, 256), (16, 16), BSR.Truncate()), x)
        Test.@test Test.@inferred(BSR.boxfold!(coarse, out, grid, Val((2, 2)), SERIAL)) === coarse
        Test.@test (@allocated BSR.boxfold!(coarse, out, grid, Val((2, 2)), SERIAL)) == 0
        a_w = (BSR.strided(256, 4, 1, BSR.Truncate()), BSR.tiled(256, 1, BSR.Truncate()))
        a = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, a_w, x), (x,), a_w, SERIAL)
        c_w = (BSR.compose_result(a_w[1], a_w[1]), a_w[2])
        c = allocate(BSR.VarAcc{Float64}, c_w, x)
        Test.@test Test.@inferred(BSR.compose!(c, a, a, 1, 1:249, 5:253, SERIAL)) === c
        Test.@test (@allocated BSR.compose!(c, a, a, 1, 1:249, 5:253, SERIAL)) == 0
        scratch = BSR.ScanScratch(BSR.VarAcc{Float64}, 4)
        lifted = BSR.boxfold!(allocate(BSR.VarAcc{Float64}, BSR.tiled((256, 256), (1, 1), BSR.Truncate()), x), (x,),
                              BSR.tiled((256, 256), (1, 1), BSR.Truncate()), SERIAL)
        Test.@test Test.@inferred(BSR.scan!(a, lifted, 1, 4, false, scratch, SERIAL)) === a
        Test.@test (@allocated BSR.scan!(a, lifted, 1, 4, false, scratch, SERIAL)) == 0
        dst = Array{Float64}(undef, size(out))
        Test.@test (@allocated BSR.finalize!(dst, BSR.member_array(out, Val(1)), BSR.Var(), SERIAL)) == 0
    end
end
