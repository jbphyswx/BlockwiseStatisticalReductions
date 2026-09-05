using Test: Test
using Random: Random
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

Random.seed!(2)

covered(aw::BSR.AxisWindow, i) = Set(BSR.window_range(aw, i))
windows(aw::BSR.AxisWindow) = [BSR.window_range(aw, i) for i in 1:BSR.nwindows(aw)]

# Direct enumeration of the origins a placement should produce.
function brute_origins(extent, size, stride, policy)
    if policy isa BSR.Partial
        return [o for o in 0:stride:extent-1]
    end
    full = [o for o in 0:extent-1 if o + size <= extent]
    if policy isa BSR.Centered
        n = extent < size ? 0 : (extent - size) ÷ stride + 1
        off = extent < size ? 0 : ((extent - size) % stride) ÷ 2
        return [off + stride * j for j in 0:n-1]
    end
    return [o for o in full if o % stride == 0]
end

Test.@testset "geometry" begin
    Test.@testset "positions" begin
        p = BSR.Progression(3, 4, 5)
        Test.@test collect(BSR.origins(p)) == [3, 7, 11, 15, 19]
        Test.@test BSR.nwindows(p) == 5
        o = BSR.Origins([0, 4, 9])
        Test.@test BSR.origins(o) == [0, 4, 9]
        Test.@test BSR.Progression(0, 2, 3) == BSR.Origins([0, 2, 4]) && hash(BSR.Progression(0, 2, 3)) == hash(BSR.Origins([0, 2, 4]))
        Test.@test_throws ArgumentError BSR.Origins([2, 2])
        Test.@test_throws ArgumentError BSR.Origins([-1, 2])
        Test.@test_throws ArgumentError BSR.Progression(0, 0, 3)
        for _ in 1:300
            p = BSR.Progression(rand(0:5), rand(1:4), rand(0:8))
            q = rand(Bool) ? BSR.Progression(rand(0:5), rand(1:4), rand(0:6)) : BSR.Origins(sort!(unique!(rand(0:30, rand(0:6)))))
            shift = rand(-3:12)
            expected = all(x -> (x + shift) in collect(BSR.origins(p)), BSR.origins(q))
            Test.@test BSR.contains(p, q, shift) == expected
            Test.@test BSR.contains(BSR.Origins(collect(BSR.origins(p))), q, shift) == expected
        end
    end

    Test.@testset "placements vs enumeration" begin
        for extent in 1:20, size in 1:extent+2, stride in 1:size+2
            for policy in (BSR.Truncate(), BSR.Partial(), BSR.Centered())
                aw = BSR.strided(extent, size, stride, policy)
                Test.@test collect(BSR.origins(aw)) == brute_origins(extent, size, stride, policy)
                for i in 1:BSR.nwindows(aw)
                    r = BSR.window_range(aw, i)
                    Test.@test first(r) == BSR.origins(aw)[i] + 1 && last(r) <= extent
                    Test.@test length(r) == (policy isa BSR.Partial ? min(size, extent - BSR.origins(aw)[i]) : size)
                end
                Test.@test BSR.uniform_length(aw) == all(o -> o + size <= extent, BSR.origins(aw))
            end
            if extent >= size && (extent - size) % stride == 0
                Test.@test collect(BSR.origins(BSR.strided(extent, size, stride, BSR.Strict()))) == brute_origins(extent, size, stride, BSR.Truncate())
            else
                Test.@test_throws ArgumentError BSR.strided(extent, size, stride, BSR.Strict())
            end
        end
        for extent in 1:24, size in 1:extent+1
            for policy in (BSR.Truncate(), BSR.Partial(), BSR.Centered())
                aw = BSR.tiled(extent, size, policy)
                Test.@test aw == BSR.strided(extent, size, size, policy)
                Test.@test BSR.is_tiled(aw)
                cov = reduce(union, (covered(aw, i) for i in 1:BSR.nwindows(aw)); init = Set{Int}())
                policy isa BSR.Partial && Test.@test cov == Set(1:extent)
                policy isa BSR.Truncate && Test.@test cov == Set(1:(extent÷size)*size)
                Test.@test sum(BSR.window_length(aw, i) for i in 1:BSR.nwindows(aw); init = 0) == length(cov)
            end
            extent % size == 0 ? Test.@test(BSR.tiled(extent, size, BSR.Strict()) == BSR.tiled(extent, size, BSR.Truncate())) :
                                 Test.@test_throws(ArgumentError, BSR.tiled(extent, size, BSR.Strict()))
        end
        w = BSR.tiled((10, 7, 3), (2, 3, 1), BSR.Truncate())
        Test.@test BSR.shape(w) == (5, 2, 3) && BSR.volume(w) == 6 && BSR.uniform_length(w)
        a = BSR.anchored(10, 3, [0, 4, 7])
        Test.@test windows(a) == [1:3, 5:7, 8:10]
        Test.@test_throws ArgumentError BSR.anchored(10, 3, [0, 8])
        Test.@test windows(BSR.anchored(10, 3, [0, 8]; partial = true)) == [1:3, 9:10]
        Test.@test_throws ArgumentError BSR.anchored(10, 3, [10])
    end

    Test.@testset "exactness identities" begin
        for X in 1:40, p in 1:8, k in 1:5
            fine = BSR.tiled(X, p, BSR.Truncate())
            coarse = BSR.coarsen_result(fine, k)
            Test.@test coarse == BSR.tiled(X, k * p, BSR.Truncate())
            Test.@test (X ÷ p) ÷ k == X ÷ (k * p)
            finep = BSR.tiled(X, p, BSR.Partial())
            coarsep = BSR.coarsen_result(finep, k)
            Test.@test coarsep == BSR.tiled(X, k * p, BSR.Partial())
            Test.@test cld(cld(X, p), k) == cld(X, k * p)
        end
        for g in 1:6, s1 in 1:12
            Test.@test BSR.contains(BSR.Progression(0, g, 40), BSR.Progression(0, g, 10), s1) == (s1 % g == 0)
        end
    end

    Test.@testset "derivation predicates vs covered sets" begin
        # coarsen: child window == union of k parent windows
        for X in 1:14, p in 1:5, k in 1:4, policy in (BSR.Truncate(), BSR.Partial())
            parent = BSR.tiled(X, p, policy)
            child = BSR.coarsen_result(parent, k)
            Test.@test BSR.can_coarsen(parent, child)
            for i in 1:BSR.nwindows(child)
                o = BSR.origins(child)[i]
                parts = [covered(parent, j) for j in 1:BSR.nwindows(parent) if o <= BSR.origins(parent)[j] < o + k * p]
                Test.@test covered(child, i) == reduce(union, parts; init = Set{Int}())
            end
            # a shifted child cannot be coarsened from a parent with tiles wider than one cell
            if X > k * p + 1 && p > 1
                shifted = BSR.AxisWindow(X, k * p, BSR.Progression(1, k * p, (X - 1) ÷ (k * p)), false)
                Test.@test !BSR.can_coarsen(parent, shifted)
            end
            Test.@test !BSR.can_coarsen(BSR.strided(X, p, 1, policy), child) || p == 1
        end
        # compose: child window == a window ++ b window at origin + a.size
        for X in 1:14, sa in 1:5, sb in 1:5, g in 1:4, policy in (BSR.Truncate(), BSR.Partial())
            a = BSR.strided(X, sa, g, policy)
            b = BSR.strided(X, sb, g, policy)
            child = BSR.compose_result(a, b)
            Test.@test BSR.can_compose(a, b, child)
            Test.@test child.size == sa + sb
            for i in 1:BSR.nwindows(child)
                o = BSR.origins(child)[i]
                ia = findfirst(==(o), collect(BSR.origins(a)))
                ib = findfirst(==(o + sa), collect(BSR.origins(b)))
                expect = covered(a, ia)
                ib === nothing || (expect = union(expect, covered(b, ib)))
                Test.@test covered(child, i) == expect
            end
            direct = BSR.strided(X, sa + sb, g, policy)
            Test.@test BSR.can_compose(a, b, direct) == (sa % g == 0 || BSR.nwindows(direct) == 0 || policy isa BSR.Partial && all(o -> o + sa >= X || BSR._has_origin(b.pos, o + sa), BSR.origins(direct)))
            if sa % g == 0
                Test.@test child == direct
                Test.@test BSR.can_compose(a, b, direct)
            end
        end
        # restride: child windows are a subset of parent windows
        for X in 1:14, s in 1:5, g in 1:3, k in 1:3
            parent = BSR.strided(X, s, g, BSR.Truncate())
            n = BSR.nwindows(parent)
            child = BSR.AxisWindow(X, s, BSR.Progression(parent.pos.offset, k * g, cld(n, k)), false)
            Test.@test BSR.can_restride(parent, child)
            Test.@test all(i -> covered(child, i) == covered(parent, findfirst(==(BSR.origins(child)[i]), collect(BSR.origins(parent)))), 1:BSR.nwindows(child))
            n > 1 && g > 1 && Test.@test !BSR.can_restride(parent, BSR.AxisWindow(X, s, BSR.Progression(parent.pos.offset + 1, g, 1), true))
            Test.@test !BSR.can_restride(parent, BSR.AxisWindow(X + 1, s, parent.pos, true))
        end
        anchors = BSR.anchored(20, 4, [0, 3, 9, 16])
        dense = BSR.strided(20, 4, 1, BSR.Truncate())
        Test.@test BSR.can_restride(dense, anchors) && !BSR.can_restride(anchors, dense)
        Test.@test BSR.compose_result(BSR.strided(20, 2, 1, BSR.Truncate()), BSR.strided(20, 2, 1, BSR.Truncate())) == BSR.strided(20, 4, 1, BSR.Truncate())
        Test.@test BSR.can_compose(BSR.strided(20, 2, 1, BSR.Truncate()), BSR.strided(20, 2, 1, BSR.Truncate()), anchors)
    end

    Test.@testset "coordinates" begin
        r = BSR.Regular(0.5)
        Test.@test BSR.mean_step(r, 10) == 0.5
        Test.@test BSR.cells_at_least(r, 10, BSR.Length(1.2)) == 3 && BSR.cells_at_most(r, 10, BSR.Length(1.2)) == 2
        Test.@test BSR.cells_at_least(r, 10, BSR.Length(0.0)) == 1
        aw = BSR.tiled(10, 4, BSR.Partial())
        Test.@test BSR.cell_bounds(r, aw) == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]
        Test.@test BSR.cell_centers(r, aw) == [1.0, 3.0, 4.5]
        Test.@test BSR.cell_bounds(BSR.Regular(2; first = 10), BSR.tiled(4, 2, BSR.Truncate())) == [(10, 14), (14, 18)]
        e = BSR.Edges([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
        Test.@test BSR.mean_step(e, 5) == 3.0
        Test.@test BSR.cell_bounds(e, BSR.tiled(5, 2, BSR.Partial())) == [(0.0, 3.0), (3.0, 10.0), (10.0, 15.0)]
        Test.@test BSR.cell_spans(e, BSR.tiled(5, 2, BSR.Truncate())) == [3.0, 7.0]
        Test.@test_throws DimensionMismatch BSR.cell_bounds(e, BSR.tiled(6, 2, BSR.Truncate()))
        Test.@test_throws ArgumentError BSR.Edges([0.0, 0.0, 1.0])
        Test.@test_throws ArgumentError BSR.Regular(0.0)
    end
end
