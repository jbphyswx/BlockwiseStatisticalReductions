using Test: Test
using Random: Random
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB
include("testutils.jl")

Random.seed!(57)

# The split a distributed backend runs, exercised in one process: slice the array into slabs by hand,
# feed each slab's own plan and its staging region, and check the pieces are the single-process answer.
# `fields` is an array or a NamedTuple of them, as the API takes; slicing keeps that shape.
_slice_fields(x::AbstractArray, axis, r) = collect(BSR.axis_slice(x, axis, r))
_slice_fields(x::NamedTuple, axis, r) = map(f -> collect(BSR.axis_slice(f, axis, r)), x)

function split_parity(x, scales, axis, bounds; stats, edge = BSR.Truncate(), kw...)
    G = size(first(BSR._fieldtuple(x)))
    whole = BSR.blockstats(x, scales; stats = stats, edge = edge, backend = CB.SerialBackend(), kw...)
    targets = BSR.resolve(scales, G; edge = edge)
    owned = [Int[] for _ in targets]
    for (lo, n) in bounds
        slab = BSR.Slab(axis, lo, n, G[axis])
        part = BSR.partition(targets, slab)
        local_x = _slice_fields(x, axis, (lo + 1):(lo + n))
        stage_x = _slice_fields(x, axis, (first(part.staging) + 1):(last(part.staging) + 1))
        sr = BSR.slab_request(part, local_x, stage_x, slab, G; stats = stats, edge = edge,
                              backend = CB.SerialBackend(), kw...)
        r = BSR.run_slab!(sr, local_x)
        Test.@test length(BSR.windows(r)) == length(targets)
        for (i, w) in enumerate(BSR.windows(r))
            gw = BSR.windows(whole)[i]
            Test.@test map(aw -> aw.size, w) == map(aw -> aw.size, gw)
            idx = [findfirst(==(o), collect(BSR.origins(gw[axis]))) for o in BSR.origins(w[axis])]
            Test.@test all(!isnothing, idx)
            append!(owned[i], idx)
            for k in BSR.statnames(r)
                Test.@test approx_nan(collect(r[i][k]), collect(BSR.axis_slice(whole[i][k], axis, idx)); rtol = 1e-12)
            end
        end
    end
    # Between them the slabs own every window exactly once.
    for (i, seen) in enumerate(owned)
        Test.@test sort(seen) == collect(1:BSR.shape(BSR.windows(whole)[i])[axis])
    end
    return nothing
end

Test.@testset "partition" begin
    Test.@testset "slab geometry" begin
        aw = BSR.tiled(32, 5, BSR.Truncate())                 # origins 0,5,…,25
        s = BSR.Slab(1, 8, 10, 32)                            # cells 8..17
        interior, boundary = BSR.split_origins(aw, s)
        Test.@test interior == [10]                           # [10,15) ends inside
        Test.@test boundary == [15]                           # [15,20) reaches past 18
        Test.@test BSR.window_stop(aw, 25) == 30
        Test.@test BSR.window_stop(BSR.tiled(32, 5, BSR.Partial()), 30) == 32
        part = BSR.partition([(aw, BSR.tiled(4, 2, BSR.Truncate()))], BSR.Slab(1, 8, 10, 32))
        Test.@test part.staging == 15:19
        Test.@test part.split == [1]
        Test.@test collect(BSR.origins(only(part.owned)[1])) == [10, 15]
        Test.@test collect(BSR.origins(only(part.interior)[1])) == [2]     # 10 - 8
        Test.@test collect(BSR.origins(only(part.boundary)[1])) == [0]     # 15 - 15
        Test.@test only(part.interior)[1].extent == 10
        Test.@test only(part.boundary)[1].extent == 5
        # A slab that owns nothing still produces a well-formed empty window.
        empty = BSR.partition([(aw, BSR.tiled(4, 2, BSR.Truncate()))], BSR.Slab(1, 31, 1, 32))
        Test.@test isempty(BSR.origins(only(empty.owned)[1]))
        Test.@test BSR.shape(only(empty.owned))[1] == 0
        Test.@test isempty(empty.staging)
        Test.@test_throws ArgumentError BSR.Slab(1, 20, 20, 32)
        Test.@test_throws DimensionMismatch BSR.partition([(aw, BSR.tiled(4, 2, BSR.Truncate()))], BSR.Slab(1, 0, 4, 40))
    end

    Test.@testset "explicit targets" begin
        shape = (16, 12)
        w1 = (BSR.tiled(16, 4, BSR.Truncate()), BSR.tiled(12, 4, BSR.Truncate()))
        w2 = (BSR.tiled(16, 8, BSR.Truncate()), BSR.tiled(12, 4, BSR.Truncate()))
        Test.@test BSR.resolve(BSR.Resolved([w2, w1]), shape) == [w2, w1]   # order is the caller's
        Test.@test_throws ArgumentError BSR.resolve(BSR.Resolved([w1, w1]), shape)
        Test.@test_throws DimensionMismatch BSR.resolve(BSR.Resolved([w1]), (8, 12))
    end

    Test.@testset "results written into the caller's arrays" begin
        x = randn(16, 12)
        into = [(mean = zeros(4, 3),)]
        w = (BSR.tiled(16, 4, BSR.Truncate()), BSR.tiled(12, 4, BSR.Truncate()))
        p = BSR.prepare(x, BSR.Resolved([w]); stats = (mean = BSR.Mean(),), into = into,
                        backend = CB.SerialBackend())
        r = BSR.blockstats!(p, x)
        Test.@test r[1].mean === into[1].mean
        Test.@test into[1].mean ≈ brute(v -> sum(v) / length(v), x, w)
        Test.@test_throws ArgumentError BSR.prepare(x, BSR.Resolved([w]); stats = (mean = BSR.Mean(),),
                                                    into = [(wrong = zeros(4, 3),)], backend = CB.SerialBackend())
        Test.@test_throws DimensionMismatch BSR.prepare(x, BSR.Resolved([w]); stats = (mean = BSR.Mean(),),
                                                        into = [(mean = zeros(3, 3),)], backend = CB.SerialBackend())
        Test.@test_throws ArgumentError BSR.prepare(x, BSR.Resolved([w]); stats = (mean = BSR.Mean(),),
                                                    into = [(mean = zeros(Float32, 4, 3),)], backend = CB.SerialBackend())
    end

    Test.@testset "a slab's share equals the whole" begin
        x = randn(24, 32)
        st = (m = BSR.Mean(), v = BSR.Var(), n = BSR.Count())
        split_parity(x, [4], 2, [(0, 16), (16, 16)]; stats = st)
        split_parity(x, [4], 2, [(0, 10), (10, 12), (22, 10)]; stats = st)
        split_parity(x, [4, 8, 16], 2, [(0, 10), (10, 12), (22, 10)]; stats = st)
        split_parity(x, [4, 8, 16], 1, [(0, 7), (7, 9), (16, 8)]; stats = st)
        split_parity(x, [(3, 5)], 2, [(0, 13), (13, 19)]; stats = st)
        split_parity(x, [4, 8, 16], 2, [(0, 5), (5, 27)]; stats = st)          # a window wider than a slab
        split_parity(x, [4, 8, 16], 2, [(0, 32), (32, 0)]; stats = st)         # a slab that owns nothing
        split_parity(x, [4], 2, [(0, 16), (16, 16)]; stats = st, edge = BSR.Partial())
        split_parity(x, [(3, 5)], 2, [(0, 13), (13, 19)]; stats = st, edge = BSR.Partial())
        y = randn(24, 32)
        split_parity((x = x, y = y), [4, 8], 2, [(0, 10), (10, 22)];
                     stats = (c = BSR.Cov(:x, :y), mx = BSR.Mean(:x)))
        xn = copy(x); xn[3, 7] = NaN
        split_parity(xn, [4, 8], 2, [(0, 10), (10, 22)]; stats = st, skipnan = true)
        split_parity(randn(12, 10, 20), [(3, 5, 4), (6, 5, 8)], 3, [(0, 7), (7, 6), (13, 7)]; stats = st)
    end

    Test.@testset "a partitioned request needs a distributed backend" begin
        x = randn(8, 8)
        Test.@test_throws ArgumentError BSR.blockstats(BSR.Partitioned(x; axis = 2), [4];
                                                       stats = (BSR.Mean(),), backend = CB.SerialBackend())
        Test.@test_throws ArgumentError BSR.Partitioned(x; axis = 0)
    end
end
