using Test: Test
using Random: Random
using Statistics: Statistics
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB

include("testutils.jl")

Random.seed!(5)
const SER = CB.SerialBackend()

# Finalize `tag` from member `k` of the storage of node `i`.
function node_values(ws, p, i, tag, ::Val{k}) where {k}
    src = BSR.member_array(BSR.node_storage(ws, i), Val(k))
    return BSR.finalize!(Array{Float64}(undef, size(src)), src, tag, SER)
end

Test.@testset "execute" begin
    Test.@testset "planned towers match direct reductions" begin
        x = randn(120, 96)
        for edge in (BSR.Truncate(), BSR.Partial())
            targets = BSR.resolve([2, 3, 4, 6, 12, 24], (120, 96); edge)
            p = BSR.plan((120, 96), targets)
            BSR.check(p)
            C, routing, _, _ = BSR.assemble((BSR.Mean(), BSR.Var(), BSR.Max()), (:x,), Float64, Float64)
            ws = BSR.allocate(p, C, x)
            BSR.run!(ws, p, (x,), SER)
            for (t, k) in zip(targets, p.outputs)
                Test.@test node_values(ws, p, k, BSR.Mean(), routing[1]) ≈ brute(Statistics.mean, x, t)
                Test.@test approx_nan(node_values(ws, p, k, BSR.Var(), routing[2]), brute(Statistics.var, x, t); rtol = 1e-11)
                Test.@test node_values(ws, p, k, BSR.Max(), routing[3]) == brute(maximum, x, t)
            end
        end
    end

    Test.@testset "multi-field composite through a plan" begin
        u = randn(64, 48, 6)
        v = 0.5 .* u .+ randn(64, 48, 6)
        targets = BSR.resolve([(2, 2, 1), (4, 4, 1), (8, 8, 2), (16, 16, 2)], (64, 48, 6))
        p = BSR.plan((64, 48, 6), targets)
        stats = (BSR.Mean(:u), BSR.Var(:u), BSR.Cov(:u, :v), BSR.Mean(:v), BSR.Min(:v))
        C, routing, _, _ = BSR.assemble(stats, (:u, :v), Float64, Float64)
        ws = BSR.allocate(p, C, u)
        BSR.run!(ws, p, (u = u, v = v), SER)
        for (t, k) in zip(targets, p.outputs)
            Test.@test node_values(ws, p, k, BSR.Var(:u), routing[2]) ≈ brute(Statistics.var, u, t)
            Test.@test node_values(ws, p, k, BSR.Cov(:u, :v), routing[3]) ≈ brute2(Statistics.cov, u, v, t)
            Test.@test node_values(ws, p, k, BSR.Mean(:v), routing[4]) ≈ brute(Statistics.mean, v, t)
            Test.@test node_values(ws, p, k, BSR.Min(:v), routing[5]) == brute(minimum, v, t)
        end
        Test.@test BSR.component(BSR.node_storage(ws, p.outputs[1]), :members, :m1, :n) isa BSR.Uniform
    end

    Test.@testset "dense sizes and anchored windows" begin
        x = randn(300, 20)
        dense = [(BSR.strided(300, s, 1, BSR.Truncate()), BSR.tiled(20, 4, BSR.Truncate())) for s in (2, 3, 5, 8)]
        anchored = [(BSR.anchored(300, 6, [0, 17, 100, 294]), BSR.tiled(20, 4, BSR.Truncate()))]
        p = BSR.plan((300, 20), vcat(dense, anchored))
        BSR.check(p)
        ws = BSR.allocate(p, BSR.VarAcc{Float64}, x)
        BSR.run!(ws, p, (x,), SER)
        for (t, k) in zip(vcat(dense, anchored), p.outputs)
            got = BSR.finalize!(Array{Float64}(undef, BSR.shape(t)), BSR.node_storage(ws, k), BSR.Var(), SER)
            Test.@test got ≈ brute(Statistics.var, x, t)
        end
    end

    Test.@testset "every derivation kind executes (hand-built plans)" begin
        x = randn(48, 30)
        dense4 = (BSR.strided(48, 4, 1, BSR.Truncate()), BSR.tiled(30, 3, BSR.Truncate()))
        dense2 = (BSR.strided(48, 2, 1, BSR.Truncate()), BSR.tiled(30, 3, BSR.Truncate()))
        tile4 = (BSR.tiled(48, 4, BSR.Truncate()), BSR.tiled(30, 3, BSR.Truncate()))
        tile8 = (BSR.tiled(48, 8, BSR.Truncate()), BSR.tiled(30, 3, BSR.Truncate()))
        ident = (BSR.strided(48, 1, 1, BSR.Truncate()), BSR.tiled(30, 3, BSR.Truncate()))
        scan5 = (BSR.strided(48, 5, 1, BSR.Partial()), BSR.tiled(30, 3, BSR.Truncate()))
        nodes = [BSR.Node(dense2, false), BSR.Node(dense4, true), BSR.Node(tile4, true), BSR.Node(tile8, true),
                 BSR.Node(ident, false), BSR.Node(scan5, true)]
        how = BSR.Derivation[
            BSR.Base_(),
            BSR.Compose(1, 1, 1, BSR.index_map(dense2[1].pos, dense4[1].pos, 0), BSR.index_map(dense2[1].pos, dense4[1].pos, 2)),
            BSR.Restride{2}(2, (BSR.index_map(dense4[1].pos, tile4[1].pos, 0), 1:10)),
            BSR.Coarsen{2}(3, (2, 1)),
            BSR.Base_(),
            BSR.Scan(5, 1, 5, true),
        ]
        p = BSR.Plan{2}((48, 30), nodes, how, [1, 5, 2, 4, 6], [1, 2, 0, 3, 4, 5], [2, 3, 4, 6])
        BSR.check(p)
        ws = BSR.allocate(p, BSR.VarAcc{Float64}, x)
        BSR.run!(ws, p, (x,), SER)
        for (w, k) in ((dense4, 2), (tile4, 3), (tile8, 4), (scan5, 6))
            got = BSR.finalize!(Array{Float64}(undef, BSR.shape(w)), BSR.node_storage(ws, k), BSR.Var(), SER)
            Test.@test got ≈ brute(Statistics.var, x, w)
        end
        Test.@test BSR.node_storage(ws, 3) isa BSR.AccumulatorArray
        Test.@test_throws ErrorException BSR.check(BSR.Plan{2}((48, 30), nodes, how, [1, 5, 2, 4, 6], [1, 2, 0, 3, 4, 5], [2, 3, 4, 6]) |>
                                                     q -> BSR.Plan{2}(q.input_shape, q.nodes, BSR.Derivation[q.how[1:3]..., BSR.Coarsen{2}(2, (2, 1)), q.how[5:6]...], q.order, q.buffer, q.outputs))
    end

    Test.@testset "zero allocation after warmup" begin
        x = randn(512, 512)
        y = randn(512, 512)
        p = BSR.plan((512, 512), BSR.resolve([2, 4, 8, 16, 32, 64], (512, 512)))
        C, _, _, _ = BSR.assemble((BSR.Mean(), BSR.Var(), BSR.Cov(:x, :y)), (:x, :y), Float64, Float64)
        ws = BSR.allocate(p, C, x)
        BSR.run!(ws, p, (x, y), SER)
        Test.@test (@allocated BSR.run!(ws, p, (x, y), SER)) == 0
        pd = BSR.plan((512, 512), [(BSR.strided(512, s, 1, BSR.Truncate()), BSR.tiled(512, 1, BSR.Truncate())) for s in (3, 4, 7)])
        wsd = BSR.allocate(pd, BSR.MeanAcc{Float64}, x)
        BSR.run!(wsd, pd, (x,), SER)
        Test.@test (@allocated BSR.run!(wsd, pd, (x,), SER)) == 0
    end
end
