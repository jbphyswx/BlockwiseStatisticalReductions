using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB

kinds(p) = map(typeof, p.how)
requested_hows(p) = [p.how[k] for k in p.outputs]

Test.@testset "planner" begin
    Test.@testset "divisor chain telescopes into one base pass" begin
        shape = (256, 256)
        targets = BSR.resolve([2, 4, 8, 16, 32], shape)
        p = BSR.plan(shape, targets)
        BSR.check(p)
        Test.@test length(BSR.base_nodes(p)) == 1
        Test.@test BSR.sizes(p.nodes[BSR.base_nodes(p)[1]]) == (2, 2)
        Test.@test all(h -> h isa BSR.Coarsen, [p.how[k] for k in p.outputs[2:end]])
        Test.@test BSR.input_passes(p) ≈ 1
        c = BSR.total_cost(p, 8, 24)
        Test.@test c.merges < 1.5 * prod(shape)
        Test.@test length(p.order) == 5 && all(k -> p.buffer[k] > 0, p.order)
        Test.@test Set(BSR.sizes(p.nodes[k]) for k in p.outputs) == Set([(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)])
    end

    Test.@testset "gcd sharing adds a finer intermediate" begin
        shape = (360, 360)
        p = BSR.plan(shape, BSR.resolve([4, 6, 12], shape))
        BSR.check(p)
        Test.@test length(BSR.base_nodes(p)) == 1
        Test.@test BSR.sizes(p.nodes[BSR.base_nodes(p)[1]]) == (2, 2)
        Test.@test !p.nodes[BSR.base_nodes(p)[1]].requested
        Test.@test all(h -> h isa BSR.Coarsen, requested_hows(p))
        Test.@test BSR.input_passes(p) ≈ 1
        # the cheapest parent of 12 is 6 (or 4), never 2
        k12 = p.outputs[3]
        Test.@test BSR.sizes(p.nodes[p.how[k12].parent]) in ((6, 6), (4, 4))
    end

    Test.@testset "anisotropic and partial targets" begin
        shape = (100, 64, 10)
        targets = BSR.resolve([(2, 2, 1), (4, 4, 1), (8, 4, 1), (8, 8, 2)], shape; edge = BSR.Partial())
        p = BSR.plan(shape, targets)
        BSR.check(p)
        Test.@test length(BSR.base_nodes(p)) == 1
        Test.@test length(p.outputs) == 4
        # `partial` survives only where a window really is clipped (100 is not a multiple of 8)
        clipped = p.nodes[p.outputs[3]].window
        Test.@test clipped[1].partial && !BSR.uniform_length(clipped[1])
        Test.@test !p.nodes[p.outputs[1]].window[1].partial
        Test.@test all(k -> BSR.uniform_count(p.nodes[k]) == BSR.uniform_length(p.nodes[k].window), eachindex(p.nodes))
    end

    Test.@testset "dense sizes share work instead of one pass each" begin
        # Sharing is measured against the same request planned without composition candidates, and
        # against one base pass per target; both are computed here rather than pinned to a constant.
        for (shape, mk, share) in (((512,), s -> (BSR.strided(512, s, 1, BSR.Truncate()),), 0.8),
                                   ((512, 512), s -> (BSR.strided(512, s, 1, BSR.Truncate()), BSR.strided(512, s, 1, BSR.Truncate())), 0.5))
            targets = [mk(s) for s in 2:9]
            p = BSR.plan(shape, targets)
            BSR.check(p)
            limits = BSR.kernel_limits(CB.SerialBackend(), length(shape))
            modelled(q) = sum(BSR.seconds(BSR.cost(q.how[k], q.nodes[k], q.nodes, 8, 24), limits) for k in q.order)
            planned = modelled(p)
            allbase = sum(BSR.seconds(BSR.cost(BSR.Base_(), BSR.Node(w, true), p.nodes, 8, 24), limits) for w in targets)
            Test.@test planned < share * allbase
            Test.@test planned <= modelled(BSR.plan(shape, targets; chains = false))
            Test.@test length(BSR.base_nodes(p)) < length(targets)
            Test.@test any(h -> h isa Union{BSR.Scan,BSR.Compose}, requested_hows(p))
            Test.@test count(k -> BSR.sizes(p.nodes[k]) == ntuple(_ -> 1, length(shape)), eachindex(p.nodes)) <= 1
        end
    end

    Test.@testset "base boxes respect the tile limit" begin
        shape = (1024, 1024)
        limits = BSR.kernel_limits(CB.SerialBackend(), 2)
        p = BSR.plan(shape, BSR.resolve([128], shape))
        BSR.check(p)
        k = p.outputs[1]
        Test.@test p.how[k] isa BSR.Coarsen
        Test.@test prod(BSR.sizes(p.nodes[BSR.base_nodes(p)[1]])) <= limits.max_tile_elements
        Test.@test BSR.input_passes(p) ≈ 1
    end

    Test.@testset "anchored and centered windows" begin
        shape = (100, 40)
        w = (BSR.anchored(100, 7, [0, 30, 93]), BSR.tiled(40, 5, BSR.Centered()))
        p = BSR.plan(shape, [w])
        BSR.check(p)
        Test.@test p.nodes[p.outputs[1]].window == w
        Test.@test p.how[p.outputs[1]] isa BSR.Base_
    end

    Test.@testset "buffers reuse dead intermediates, never outputs" begin
        shape = (1024, 1024)
        p = BSR.plan(shape, BSR.resolve([4, 6, 12, 24], shape))
        BSR.check(p)
        Test.@test length(unique(p.buffer[p.outputs])) == length(p.outputs)
        Test.@test BSR.peak_bytes(p, 24) <= 24 * sum(BSR.cells(n) for n in p.nodes)
        Test.@test_throws ArgumentError BSR.plan(shape, BSR.resolve([4, 6, 12, 24], shape); memory_limit = 1)
    end

    Test.@testset "errors and reports" begin
        shape = (64, 64)
        Test.@test_throws ArgumentError BSR.plan(shape, BSR.Window{2}[])
        Test.@test_throws DimensionMismatch BSR.plan(shape, BSR.resolve([4], (32, 32)))
        p = BSR.plan(shape, BSR.resolve([4, 8], shape))
        s = sprint(io -> BSR.explain(io, p))
        Test.@test occursin("2 requested", s) && occursin("base pass", s) && occursin("input passes", s)
        d = BSR.dot(p)
        Test.@test occursin("digraph", d) && count("->", d) == length(p.nodes)
        Test.@test_throws ArgumentError BSR.plan(shape, BSR.resolve([4], shape); backend = CB.ThreadedBackend())
    end
end
