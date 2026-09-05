using Test: Test
using Random: Random
using Statistics: Statistics
using DimensionalData: DimensionalData as DD
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB
include("testutils.jl")

Random.seed!(61)

Test.@testset "labeled axes" begin
    Test.@testset "core: names and spacing without any labelled array" begin
        x = randn(60, 40)
        sp = (BSR.Regular(0.5), BSR.Regular(2.0; first = 100.0))
        r = BSR.blockstats(x, BSR.ScaleSet((u = BSR.Sizes([4, 6]), v = BSR.Fixed(2)));
                           stats = (BSR.Mean(),), dimnames = (:u, :v), spacing = sp)
        Test.@test Set(BSR.scales(r)) == Set([(4, 2), (6, 2)])
        Test.@test BSR.dimnames(r) == (:u, :v)
        g = BSR.geometry(r, (4, 2))
        Test.@test g.names == (:u, :v)
        Test.@test g.bounds[1][1] == (0.0, 2.0) && g.bounds[1][2] == (2.0, 4.0)   # 4 cells of 0.5
        Test.@test g.centers[2][1] == 102.0                                        # 2 cells of 2.0 from 100.0
        Test.@test g.ranges[1][2] == 5:8
        # a size in physical units resolves against the spacing
        r2 = BSR.blockstats(x, BSR.ScaleSet((u = BSR.Every(min = BSR.Length(1.4), max = BSR.Length(2.1)), v = BSR.Fixed(1)));
                            stats = (BSR.Mean(),), dimnames = (:u, :v), spacing = sp)
        Test.@test Set(first.(BSR.scales(r2))) == Set([3, 4])                       # ceil(1.4/0.5) .. floor(2.1/0.5)
        Test.@test_throws ArgumentError BSR.blockstats(x, [4]; stats = (BSR.Mean(),), dimnames = (:u, :v, :w))
        Test.@test_throws ArgumentError BSR.blockstats(x, [4]; stats = (BSR.Mean(),), dimnames = (:u, :u))
        # a Length with no spacing cannot be resolved, and says so
        Test.@test_throws ArgumentError BSR.blockstats(x, BSR.ScaleSet(BSR.Dyadic(max = BSR.Length(2.0))); stats = (BSR.Mean(),))
    end

    Test.@testset "spacing read off the dims" begin
        DDExt = Base.get_extension(BSR, :BlockwiseStatisticalReductionsDimensionalDataExt)
        # regularly spaced points become cells centred on them
        pts = DD.DimArray(randn(8), (DD.X(0.0:0.5:3.5),))
        sp = DDExt._spacing(DD.dims(pts, 1))
        Test.@test sp isa BSR.Regular && sp.step == 0.5 && sp.first == -0.25
        # irregular points become cells whose edges are the midpoints
        irr = DD.DimArray(randn(4), (DD.X([0.0, 1.0, 3.0, 4.0]),))
        spi = DDExt._spacing(DD.dims(irr, 1))
        Test.@test spi isa BSR.Edges && spi.edges ≈ [-0.5, 0.5, 2.0, 3.5, 4.5]
        # an interval lookup states its own bounds
        m = [0.0 1.0 2.5; 1.0 2.5 4.0]
        expl = DD.DimArray(randn(3), (DD.X(DD.Sampled([0.5, 1.75, 3.25]; order = DD.ForwardOrdered(),
                                                      span = DD.Explicit(m), sampling = DD.Intervals(DD.Center()))),))
        spe = DDExt._spacing(DD.dims(expl, 1))
        Test.@test spe isa BSR.Edges && spe.edges ≈ [0.0, 1.0, 2.5, 4.0]
    end

    Test.@testset "a dimensional array in, dimensional stacks out" begin
        A = DD.DimArray(randn(64, 48), (DD.X(0.0:0.5:31.5), DD.Y(0.0:2.0:94.0)); name = :f)
        r = BSR.blockstats(A, [4, 8]; stats = (BSR.Mean(), BSR.Var()))
        Test.@test length(r) == 2
        w8 = BSR.tiled((64, 48), (8, 8), BSR.Truncate())
        st = r[w8]
        Test.@test st isa DD.AbstractDimStack && keys(st) == (:mean, :var)
        Test.@test size(st) == (8, 6)
        Test.@test parent(st.mean) ≈ brute(Statistics.mean, parent(A), w8)
        Test.@test parent(st.var) ≈ brute(Statistics.var, parent(A), w8)
        # the coarsened axes are intervals spanning exactly the cells each window covered
        Test.@test map(DD.dim2key, DD.dims(st)) == (:X, :Y)
        Test.@test DD.bounds(st, 1) == (-0.25, 31.75)          # the input's own extent, cells centred on points
        Test.@test DD.sampling(DD.lookup(st, 1)) isa DD.Intervals
        xb = DD.val(DD.span(DD.lookup(st, 1)))
        Test.@test xb[1, 1] == -0.25 && xb[2, 1] ≈ 3.75        # first window covers 8 cells of 0.5
        Test.@test collect(DD.val(DD.lookup(st, 1)))[1] ≈ 1.75  # its centre
        # truncation shows up as a shorter axis, not a wrong one
        rt = BSR.blockstats(A, [7]; stats = (BSR.Mean(),))
        st7 = rt[(7, 7)]
        Test.@test size(st7) == (9, 6)
        Test.@test DD.bounds(st7, 1)[2] ≈ -0.25 + 9 * 7 * 0.5
    end

    Test.@testset "axes addressed by name, sizes in physical units" begin
        A = DD.DimArray(randn(64, 48), (DD.X(0.0:0.5:31.5), DD.Y(0.0:2.0:94.0)))
        r = BSR.blockstats(A, BSR.ScaleSet((X = BSR.Dyadic(min = BSR.Length(2.0), max = BSR.Length(9.0)), Y = BSR.Fixed(1)));
                           stats = (BSR.Mean(),))
        Test.@test sort(BSR.scales(r)) == [(4, 1), (8, 1), (16, 1)]   # 2.0 and 9.0 units at 0.5 per cell
    end

    Test.@testset "a stack of layers reduces in one pass" begin
        u = DD.DimArray(randn(32, 24), (DD.X(1:32), DD.Y(1:24)))
        v = DD.DimArray(randn(32, 24), (DD.X(1:32), DD.Y(1:24)))
        s = DD.DimStack((u = u, v = v))
        r = BSR.blockstats(s, [8]; stats = (BSR.Mean(:u), BSR.Var(:v), BSR.Cov(:u, :v)))
        st = r[(8, 8)]
        Test.@test BSR.statnames(r) == (:mean_u, :var_v, :cov_u_v)
        Test.@test BSR.dimnames(r) == (:X, :Y)
        Test.@test keys(st) == (:mean_u, :var_v, :cov_u_v)
        w = BSR.tiled((32, 24), (8, 8), BSR.Truncate())
        Test.@test parent(st.mean_u) ≈ brute(Statistics.mean, parent(u), w)
        Test.@test parent(st.cov_u_v) ≈ brute2(Statistics.cov, parent(u), parent(v), w)
    end

    Test.@testset "a prepared dimensional request repeats" begin
        A = DD.DimArray(randn(48, 32), (DD.X(0.0:0.5:23.5), DD.Y(1:32)))
        p = BSR.prepare(A, [8]; stats = (BSR.Mean(),), backend = CB.SerialBackend())
        r1 = BSR.blockstats!(p, A)
        B = DD.DimArray(fill(3.0, 48, 32), DD.dims(A))
        r2 = BSR.blockstats!(p, B)
        Test.@test all(≈(3.0), parent(r2[(8, 8)].mean))
        Test.@test DD.bounds(r2[(8, 8)], 1) == DD.bounds(r1[(8, 8)], 1)
        Test.@test BSR.geometry(r2, (8, 8)).bounds[1][1] == (-0.25, 3.75)
        Test.@test occursin("input passes", sprint(io -> BSR.explain(io, p)))
    end
end
