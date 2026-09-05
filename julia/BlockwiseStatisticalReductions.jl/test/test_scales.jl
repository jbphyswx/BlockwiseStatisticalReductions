using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

sizes_of(ws) = [map(aw -> aw.size, w) for w in ws]

Test.@testset "scales" begin
    Test.@testset "generators" begin
        Test.@test BSR.generate(BSR.Dyadic(), 100, nothing) == [1, 2, 4, 8, 16, 32, 64]
        Test.@test BSR.generate(BSR.Dyadic(; min = 3, max = 40), 100, nothing) == [4, 8, 16, 32]
        Test.@test BSR.generate(BSR.Smooth(), 40, nothing) == [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
        Test.@test BSR.generate(BSR.Smooth((2, 5); min = 4), 60, nothing) == [4, 5, 8, 10, 16, 20, 25, 32, 40, 50]
        Test.@test BSR.generate(BSR.Every(; min = 3, max = 6), 100, nothing) == [3, 4, 5, 6]
        Test.@test BSR.generate(BSR.Every(), 4, nothing) == [1, 2, 3, 4]
        Test.@test BSR.generate(BSR.Divisors(), 12, nothing) == [1, 2, 3, 4, 6, 12]
        Test.@test BSR.generate(BSR.Divisors(; min = 2, max = 6), 12, nothing) == [2, 3, 4, 6]
        Test.@test BSR.generate(BSR.Sizes([8, 2, 2, 64, 5]), 32, nothing) == [2, 5, 8]
        Test.@test BSR.generate(BSR.Fixed(3), 10, nothing) == [3] && BSR.generate(BSR.Fixed(11), 10, nothing) == Int[]
        Test.@test BSR.generate(BSR.Subsample(BSR.Every(), 4), 10, nothing) == [1, 4, 7, 10]
        Test.@test BSR.generate(BSR.Subsample(BSR.Every(), 1), 10, nothing) == [5]
        Test.@test BSR.generate(BSR.Subsample(BSR.Every(), 20), 10, nothing) == collect(1:10)
        Test.@test BSR.generate(BSR.Every(; min = 7, max = 3), 10, nothing) == Int[]
        Test.@test_throws ArgumentError BSR.generate(BSR.Smooth((1,)), 10, nothing)
        Test.@test_throws ArgumentError BSR.generate(BSR.Subsample(BSR.Every(), 0), 10, nothing)
    end

    Test.@testset "physical bounds" begin
        r = BSR.Regular(0.5)
        Test.@test BSR.generate(BSR.Every(; min = BSR.Length(1.2), max = BSR.Length(2.6)), 100, r) == [3, 4, 5]
        Test.@test BSR.generate(BSR.Dyadic(; min = BSR.Length(1.0)), 100, r) == [2, 4, 8, 16, 32, 64]
        e = BSR.Edges(collect(0.0:2.0:20.0))
        Test.@test BSR.generate(BSR.Every(; max = BSR.Length(5.0)), 10, e) == [1, 2]
        Test.@test_throws ArgumentError BSR.generate(BSR.Every(; min = BSR.Length(1.0)), 10, nothing)
    end

    Test.@testset "combination" begin
        shape = (64, 48, 5)
        iso = BSR.resolve(BSR.ScaleSet((BSR.Dyadic(), BSR.Dyadic(), BSR.Fixed(1))), shape)
        Test.@test sizes_of(iso) == [(1, 1, 1), (2, 2, 1), (4, 4, 1), (8, 8, 1), (16, 16, 1), (32, 32, 1)]
        Test.@test all(w -> BSR.uniform_length(w) && BSR.is_tiled(w[1]), iso)
        prod = BSR.resolve(BSR.ScaleSet((BSR.Sizes([2, 4]), BSR.Sizes([3, 6]), 1); combine = BSR.Product()), shape)
        Test.@test Set(sizes_of(prod)) == Set([(2, 3, 1), (2, 6, 1), (4, 3, 1), (4, 6, 1)])
        zipped = BSR.resolve(BSR.ScaleSet((BSR.Sizes([2, 4]), BSR.Sizes([3, 6]), 1); combine = BSR.Zip()), shape)
        Test.@test sizes_of(zipped) == [(2, 3, 1), (4, 6, 1)]
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((BSR.Sizes([2, 4]), BSR.Sizes([3]), 1); combine = BSR.Zip()), shape)
        full = BSR.resolve(BSR.ScaleSet((BSR.Dyadic(; min = 16), BSR.Dyadic(; min = 16), 1); include_full = true), shape)
        Test.@test sizes_of(full) == [(16, 16, 1), (32, 32, 1), (64, 48, 1)]
        Test.@test BSR.nwindows(full[end][1]) == 1 && BSR.nwindows(full[end][2]) == 1
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((BSR.Sizes([7]), BSR.Sizes([5]), 1)), shape)
        one_axis = BSR.resolve(BSR.ScaleSet(BSR.Dyadic(; min = 2)), (16,))
        Test.@test sizes_of(one_axis) == [(2,), (4,), (8,), (16,)]
        all_axes = BSR.resolve(BSR.ScaleSet(BSR.Sizes([2, 3])), (12, 12))
        Test.@test sizes_of(all_axes) == [(2, 2), (3, 3)]
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((BSR.Dyadic(), BSR.Dyadic())), shape)
    end

    Test.@testset "filters and element bounds" begin
        shape = (64, 64)
        ws = BSR.resolve(BSR.ScaleSet(BSR.Every(); min_elements = 9, max_elements = 36), shape)
        Test.@test sizes_of(ws) == [(3, 3), (4, 4), (5, 5), (6, 6)]
        odd = BSR.resolve(BSR.ScaleSet(BSR.Every(; max = 9); filter = w -> isodd(w[1].size)), shape)
        Test.@test sizes_of(odd) == [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]
        aspect = BSR.resolve(BSR.ScaleSet((BSR.Sizes([2, 4]), BSR.Sizes([2, 4, 8])); combine = BSR.Product(),
                                          filter = w -> w[2].size <= 2 * w[1].size), shape)
        Test.@test Set(sizes_of(aspect)) == Set([(2, 2), (2, 4), (4, 2), (4, 4), (4, 8)])
    end

    Test.@testset "placements and edge policies" begin
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); placement = BSR.Stride(2)), (10, 10))
        Test.@test collect(BSR.origins(ws[1][1])) == [0, 2, 4, 6] && BSR.shape(ws[1]) == (4, 4)
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); placement = BSR.Overlap(1 // 2)), (10, 10))
        Test.@test collect(BSR.origins(ws[1][1])) == [0, 2, 4, 6]
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); placement = BSR.Dense()), (10, 10))
        Test.@test collect(BSR.origins(ws[1][1])) == 0:6
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); placement = BSR.Anchors([0, 5, 6])), (10, 10))
        Test.@test BSR.origins(ws[1][2]) == [0, 5, 6]
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([3]); placement = BSR.Spread(3)), (10, 10))
        Test.@test BSR.origins(ws[1][1]) == [0, 3, 7]
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([3]); placement = BSR.Spread(20)), (5,))
        Test.@test BSR.origins(ws[1][1]) == [0, 1, 2]
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([3]); placement = (BSR.Tiled(), BSR.Dense())), (9, 9))
        Test.@test BSR.is_tiled(ws[1][1]) && collect(BSR.origins(ws[1][2])) == 0:6
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); edge = BSR.Partial()), (10,))
        Test.@test ws[1][1].partial && BSR.nwindows(ws[1][1]) == 3 && !BSR.uniform_length(ws[1])
        ws = BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); edge = BSR.Centered()), (10,))
        Test.@test collect(BSR.origins(ws[1][1])) == [1, 5]
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); edge = BSR.Strict()), (10,))
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); placement = (BSR.Tiled(),)), (10, 10))
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet(BSR.Sizes([4]); placement = BSR.Spread(0)), (10,))
    end

    Test.@testset "named axes" begin
        shape = (32, 16, 8)
        names = (:u, :v, :t)
        ws = BSR.resolve(BSR.ScaleSet((u = BSR.Dyadic(; min = 4), v = BSR.Dyadic(; min = 4))), shape; dimnames = names)
        Test.@test sizes_of(ws) == [(4, 4, 1), (8, 8, 1), (16, 16, 1)]
        ws = BSR.resolve(BSR.ScaleSet((t = BSR.Every(; min = 2, max = 4),)), shape; dimnames = names)
        Test.@test sizes_of(ws) == [(1, 1, 2), (1, 1, 3), (1, 1, 4)]
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((q = BSR.Dyadic(),)), shape; dimnames = names)
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((u = BSR.Dyadic(),)), shape)
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((u = BSR.Dyadic(),)), shape; dimnames = (:u, :v))
        r = (BSR.Regular(2.0), BSR.Regular(2.0), BSR.Edges(collect(0.0:10.0:80.0)))
        ws = BSR.resolve(BSR.ScaleSet((u = BSR.Dyadic(; min = BSR.Length(7.0)), v = BSR.Dyadic(; min = BSR.Length(7.0)),
                                       t = BSR.Every(; max = BSR.Length(25.0))); combine = BSR.Isotropic((:u, :v))), shape;
                         dimnames = names, spacing = r)
        Test.@test Set(sizes_of(ws)) == Set([(a, a, c) for a in (4, 8, 16) for c in 1:2])
        ws2 = BSR.resolve(BSR.ScaleSet((BSR.Dyadic(; min = 4), BSR.Dyadic(; min = 4), BSR.Every(; max = 2)); combine = BSR.Isotropic((1, 2))), shape)
        Test.@test Set(sizes_of(ws2)) == Set([(a, a, c) for a in (4, 8, 16) for c in 1:2])
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((BSR.Dyadic(), BSR.Dyadic(), 1); combine = BSR.Isotropic((:u,))), shape)
        Test.@test_throws ArgumentError BSR.resolve(BSR.ScaleSet((BSR.Dyadic(), BSR.Dyadic(), 1); combine = BSR.Isotropic((4,))), shape)
    end

    Test.@testset "convenience forms" begin
        shape = (24, 18)
        Test.@test sizes_of(BSR.resolve([2, 3, 6], shape)) == [(2, 2), (3, 3), (6, 6)]
        Test.@test sizes_of(BSR.resolve(6, shape)) == [(6, 6)]
        Test.@test sizes_of(BSR.resolve((6, 3), shape)) == [(6, 3)]
        Test.@test sizes_of(BSR.resolve([(6, 3), (2, 2), (6, 3)], shape)) == [(2, 2), (6, 3)]
        Test.@test_throws ArgumentError BSR.resolve([(30, 3)], shape)
        w = BSR.tiled(shape, (4, 6), BSR.Truncate())
        Test.@test BSR.resolve(w, shape) == [w]
        Test.@test BSR.resolve([w, w], shape) == [w]
        Test.@test_throws DimensionMismatch BSR.resolve(w, (20, 18))
        ws = BSR.resolve([4], shape; edge = BSR.Partial())
        Test.@test ws[1][1].partial && BSR.nwindows(ws[1][2]) == 5
    end
end
