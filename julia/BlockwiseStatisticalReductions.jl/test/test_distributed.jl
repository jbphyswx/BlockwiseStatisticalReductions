using Test: Test
using Random: Random
using Distributed: Distributed
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB
include("testutils.jl")

# Workers of our own when the process was not started with any, so this file runs either way.
const ADDED = Distributed.nworkers() == 1 && Distributed.workers() == [Distributed.myid()] ?
    Distributed.addprocs(2; exeflags = `--project=$(Base.active_project())`) : Int[]
Distributed.@everywhere using BlockwiseStatisticalReductions

Random.seed!(83)

_slice(x::AbstractArray, axis, r) = collect(BSR.axis_slice(x, axis, r))

function dist_parity(label, x, scales; stats, rtol = 1e-12, kw...)
    got = BSR.blockstats(x, scales; stats = stats, backend = CB.DistributedBackend(CB.SerialBackend()), kw...)
    want = BSR.blockstats(x, scales; stats = stats, backend = CB.SerialBackend(), kw...)
    Test.@testset "$label" begin
        Test.@test BSR.windows(got) == BSR.windows(want)
        Test.@test BSR.statnames(got) == BSR.statnames(want)
        for i in 1:length(BSR.windows(got)), k in BSR.statnames(got)
            Test.@test approx_nan(got[i][k], want[i][k]; rtol = rtol)
        end
    end
    return nothing
end

Test.@testset "distributed" begin
    x = randn(18, 31)
    y = randn(18, 31)
    w = rand(18, 31) .+ 0.5
    wz = rand(31) .+ 0.5
    st = (m = BSR.Mean(), v = BSR.Var(), n = BSR.Count())

    dist_parity("tiles", x, [4]; stats = st)
    dist_parity("several scales", x, [2, 4, 8]; stats = st)
    dist_parity("anisotropic sizes", x, [(3, 5), (6, 10)]; stats = st)
    dist_parity("partial edges", x, [(3, 5)]; stats = st, edge = BSR.Partial())
    dist_parity("a window wider than a slab", x, [(4, 24)]; stats = st)
    dist_parity("overlapping windows", x, BSR.ScaleSet((BSR.Sizes([4]), BSR.Sizes([6]));
                combine = BSR.Product(), placement = BSR.Stride(2)); stats = st)
    dist_parity("two fields and their covariance", (x = x, y = y), [4];
                stats = (c = BSR.Cov(:x, :y), mx = BSR.Mean(:x), my = BSR.Mean(:y)))
    dist_parity("element weights", x, [4, 8]; stats = (m = BSR.Mean(), v = BSR.Var()), weights = w)
    dist_parity("per-axis weight factors", x, [4]; stats = (m = BSR.Mean(),), weights = (nothing, wz))
    xn = copy(x); xn[3, 7] = NaN; xn[11, 20] = Inf
    dist_parity("skipnan", xn, [4, 8]; stats = st, skipnan = true)
    dist_parity("weights and skipnan", xn, [4]; stats = st, weights = w, skipnan = true)
    dist_parity("float32 with a large offset", Float32.(x .+ 1.0f5), [4, 8];
                stats = (m = BSR.Mean(), v = BSR.Var()), rtol = 1e-4)
    dist_parity("three dimensions", randn(8, 6, 19), [(2, 3, 4), (4, 3, 8)]; stats = st)

    Test.@testset "prepared reuse" begin
        st2 = (m = BSR.Mean(), v = BSR.Var())
        p = BSR.prepare(x, [4, 8]; stats = st2, backend = CB.DistributedBackend(CB.SerialBackend()))
        snap(r) = [map(copy, r[i]) for i in 1:length(BSR.windows(p))]
        z = 2 .* x .+ 1
        first_run = snap(BSR.blockstats!(p, x))
        second = snap(BSR.blockstats!(p, z))
        again = snap(BSR.blockstats!(p, x))
        for i in 1:length(BSR.windows(p)), k in keys(st2)
            want = BSR.blockstats(z, [4, 8]; stats = st2, backend = CB.SerialBackend())[i][k]
            Test.@test second[i][k] ≈ want
            Test.@test again[i][k] == first_run[i][k]
            Test.@test first_run[i][k] != second[i][k]
        end
        Test.@test BSR.release!(p) === nothing
        Test.@test BSR.release!(BSR.prepare(x, [4]; stats = st2, backend = CB.SerialBackend())) === nothing
    end

    Test.@testset "rejected requests" begin
        Test.@test_throws DimensionMismatch BSR.blockstats!(
            BSR.prepare(x, [4]; stats = (m = BSR.Mean(),), backend = CB.DistributedBackend(CB.SerialBackend())),
            randn(9, 31))
        # An MPI backend wants a partitioned input, not a whole tensor.
        Test.@test_throws ArgumentError BSR.prepare(x, [4]; stats = (m = BSR.Mean(),),
                                                    backend = CB.MPIBackend(CB.SerialBackend(), nothing))
    end
end

isempty(ADDED) || Distributed.rmprocs(ADDED)
