using Test: Test
using Random: Random
using Statistics: Statistics
using NCDatasets: NCDatasets as NC
using Zarr: Zarr
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
include("testutils.jl")

Random.seed!(71)

# One dataset in both formats: axes `x` (regular, 0.5 apart) and `y` (irregular), two fields.
const NX, NY = 32, 12
const XC = collect(0.0:0.5:0.5 * (NX - 1))
const YC = Float64[0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36]
const F = reshape(collect(1.0:(NX * NY)), NX, NY)
const G = randn(NX, NY)

function write_netcdf(path)
    NC.NCDataset(path, "c") do ds
        NC.defDim(ds, "x", NX); NC.defDim(ds, "y", NY)
        NC.defVar(ds, "x", Float64, ("x",))[:] = XC
        NC.defVar(ds, "y", Float64, ("y",))[:] = YC
        NC.defVar(ds, "f", Float64, ("x", "y"))[:, :] = F
        NC.defVar(ds, "g", Float64, ("x", "y"))[:, :] = G
    end
    return path
end

function write_zarr(path)
    g = Zarr.zgroup(path)
    dims = Dict("_ARRAY_DIMENSIONS" => ["x", "y"])
    Zarr.zcreate(Float64, g, "f", NX, NY; chunks = (8, 4), attrs = dims)[:, :] = F
    Zarr.zcreate(Float64, g, "g", NX, NY; chunks = (8, 4), attrs = dims)[:, :] = G
    Zarr.zcreate(Float64, g, "x", NX; chunks = (NX,), attrs = Dict("_ARRAY_DIMENSIONS" => ["x"]))[:] = XC
    Zarr.zcreate(Float64, g, "y", NY; chunks = (NY,), attrs = Dict("_ARRAY_DIMENSIONS" => ["y"]))[:] = YC
    return path
end

Test.@testset "labelled files" begin
    dir = mktempdir()
    ncpath = write_netcdf(joinpath(dir, "d.nc"))
    zpath = write_zarr(joinpath(dir, "d.zarr"))
    w8 = BSR.tiled((NX, NY), (8, 4), BSR.Truncate())

    Test.@testset "NetCDF" begin
        NC.NCDataset(ncpath) do ds
            r = BSR.blockstats(ds["f"], [(8, 4)]; stats = (BSR.Mean(), BSR.Var()))
            Test.@test BSR.dimnames(r) == (:x, :y)
            Test.@test BSR.scales(r) == [(8, 4)]
            Test.@test r[(8, 4)].mean ≈ brute(Statistics.mean, F, w8)
            Test.@test r[(8, 4)].var ≈ brute(Statistics.var, F, w8)
            # the regular axis becomes a Regular spacing, the irregular one explicit edges
            g = BSR.geometry(r, (8, 4))
            Test.@test g.bounds[1][1] == (-0.25, 3.75)                  # 8 cells of 0.5 from -0.25
            Test.@test g.bounds[2][1][1] == -0.5                        # y's first edge, reflected
            Test.@test g.centers[1][2] ≈ 5.75          # second window covers [3.75, 7.75]
            # an axis addressed by name, with a size in physical units
            rn = BSR.blockstats(ds["f"], BSR.ScaleSet((x = BSR.Every(min = BSR.Length(1.4), max = BSR.Length(2.1)), y = BSR.Fixed(1)));
                                stats = (BSR.Mean(),))
            Test.@test Set(first.(BSR.scales(rn))) == Set([3, 4])
            # several variables of one dataset, in a single pass
            rc = BSR.blockstats(ds, (:f, :g), [(8, 4)]; stats = (BSR.Mean(:f), BSR.Cov(:f, :g)))
            Test.@test rc[(8, 4)].mean_f ≈ brute(Statistics.mean, F, w8)
            Test.@test rc[(8, 4)].cov_f_g ≈ brute2(Statistics.cov, F, G, w8)
            # a prepared request re-reads the variable each call
            p = BSR.prepare(ds["f"], [(8, 4)]; stats = (BSR.Mean(),))
            Test.@test BSR.blockstats!(p, ds["f"])[(8, 4)].mean ≈ brute(Statistics.mean, F, w8)
        end
    end

    Test.@testset "Zarr" begin
        g = Zarr.zopen(zpath)
        r = BSR.blockstats(g["f"], [(8, 4)]; stats = (BSR.Mean(), BSR.Var()), group = g)
        Test.@test BSR.dimnames(r) == (:x, :y)
        Test.@test r[(8, 4)].mean ≈ brute(Statistics.mean, F, w8)
        Test.@test BSR.geometry(r, (8, 4)).bounds[1][1] == (-0.25, 3.75)
        # without the group there are no coordinates, so the axes are named but unplaced
        rp = BSR.blockstats(g["f"], [(8, 4)]; stats = (BSR.Mean(),))
        Test.@test BSR.dimnames(rp) == (:x, :y)
        Test.@test all(isnothing, BSR.geometry(rp, (8, 4)).bounds)
        Test.@test rp[(8, 4)].mean ≈ brute(Statistics.mean, F, w8)
        # several arrays of one group, and axes by name
        rc = BSR.blockstats(g, (:f, :g), [(8, 4)]; stats = (BSR.Mean(:f), BSR.Cov(:f, :g)))
        Test.@test rc[(8, 4)].cov_f_g ≈ brute2(Statistics.cov, F, G, w8)
        rn = BSR.blockstats(g, (:f,), BSR.ScaleSet((x = BSR.Dyadic(max = BSR.Length(4.1)), y = BSR.Fixed(2)));
                            stats = (BSR.Mean(:f),))
        Test.@test Set(first.(BSR.scales(rn))) == Set([1, 2, 4, 8])     # 4.1 units at 0.5 per cell
        p = BSR.prepare(g["f"], [(8, 4)]; stats = (BSR.Mean(),), group = g)
        Test.@test BSR.blockstats!(p, g["f"])[(8, 4)].mean ≈ brute(Statistics.mean, F, w8)
        # an array with no axis names says so rather than guessing
        bare = Zarr.zcreate(Float64, Zarr.zgroup(joinpath(dir, "bare.zarr")), "h", 8, 8; chunks = (4, 4))
        bare[:, :] = randn(8, 8)
        Test.@test_throws ArgumentError BSR.blockstats(bare, [4]; stats = (BSR.Mean(),))
        Test.@test BSR.blockstats(bare, [4]; stats = (BSR.Mean(),), dimnames = (:a, :b)) isa BSR.ScaleResults
    end

    Test.@testset "both formats agree with each other and with the array" begin
        NC.NCDataset(ncpath) do ds
            zg = Zarr.zopen(zpath)
            plain = BSR.blockstats(F, [(8, 4)]; stats = (BSR.Mean(), BSR.Var()))
            fromnc = BSR.blockstats(ds["f"], [(8, 4)]; stats = (BSR.Mean(), BSR.Var()))
            fromz = BSR.blockstats(zg["f"], [(8, 4)]; stats = (BSR.Mean(), BSR.Var()), group = zg)
            Test.@test plain[(8, 4)].mean == fromnc[(8, 4)].mean == fromz[(8, 4)].mean
            Test.@test plain[(8, 4)].var == fromnc[(8, 4)].var == fromz[(8, 4)].var
            Test.@test BSR.geometry(fromnc, (8, 4)).bounds == BSR.geometry(fromz, (8, 4)).bounds
        end
    end
end
