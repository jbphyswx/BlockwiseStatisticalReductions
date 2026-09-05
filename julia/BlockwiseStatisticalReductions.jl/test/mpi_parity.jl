# Run under `mpiexec -n N`. Every rank builds the same global tensor, keeps only its own slab, and checks
# that the partitioned request gives exactly what a single process gives for the windows it owns.
using Test: Test
using Random: Random
using MPI: MPI
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR
using ComputationalBackends: ComputationalBackends as CB

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

# Deliberately uneven slabs: the last rank gets the remainder, so no window size divides the partition.
function slab_bounds(extent::Int, n::Int)
    base = extent ÷ n
    lo = 0
    out = Tuple{Int,Int}[]
    for r in 1:n
        len = r < n ? base : extent - lo
        push!(out, (lo, len))
        lo += len
    end
    return out
end

axis_slice(a, axis, r) = BSR.axis_slice(a, axis, r)

# A rank derives its windows from a different plan than a single process does, so the merge order — and
# with it the last bit — differs. Values agree to rounding, and undefined cells are `NaN` on both sides.
function same(a, b; rtol)
    size(a) == size(b) || return false
    return all(i -> (isnan(a[i]) && isnan(b[i])) || isapprox(a[i], b[i]; rtol = rtol), eachindex(a, b))
end

function parity(label, x, scales, axis; stats, weights = nothing, rtol = 1e-12, kw...)
    G = x isa NamedTuple ? size(first(values(x))) : size(x)
    lo, len = slab_bounds(G[axis], NRANKS)[RANK+1]
    rows = (lo + 1):(lo + len)
    local_fields = x isa NamedTuple ? map(f -> collect(axis_slice(f, axis, rows)), x) :
                                      collect(axis_slice(x, axis, rows))
    local_w = weights isa AbstractArray ? collect(axis_slice(weights, axis, rows)) : weights
    got = BSR.blockstats(BSR.Partitioned(local_fields; axis = axis), scales;
                         stats = stats, weights = local_w, backend = CB.MPIBackend(CB.SerialBackend(), COMM), kw...)
    want = BSR.blockstats(x, scales; stats = stats, weights = weights, backend = CB.SerialBackend(), kw...)
    Test.@testset "$label (rank $RANK of $NRANKS)" begin
        Test.@test length(BSR.windows(got)) == length(BSR.windows(want))
        for (i, w) in enumerate(BSR.windows(got))
            gw = BSR.windows(want)[i]
            Test.@test map(aw -> aw.size, w) == map(aw -> aw.size, gw)
            idx = [findfirst(==(o), collect(BSR.origins(gw[axis]))) for o in BSR.origins(w[axis])]
            Test.@test all(!isnothing, idx)
            for k in BSR.statnames(got)
                Test.@test same(got[i][k], axis_slice(want[i][k], axis, idx); rtol = rtol)
            end
        end
        # Together the ranks own every window exactly once.
        counts = MPI.Allgather(length(BSR.origins(BSR.windows(got)[1][axis])), COMM)
        Test.@test sum(counts) == length(BSR.origins(BSR.windows(want)[1][axis]))
    end
    return nothing
end

Random.seed!(4321)
x = randn(20, 33)
y = randn(20, 33)
w = rand(20, 33) .+ 0.5
wx = rand(20) .+ 0.5
wz = rand(33) .+ 0.5

Test.@testset "mpi parity" begin
    stats2 = (m = BSR.Mean(), v = BSR.Var(), n = BSR.Count())
    parity("tiles, partitioned on the last axis", x, [4], 2; stats = stats2)
    parity("several scales", x, [2, 4, 8], 2; stats = stats2)
    parity("partitioned on the contiguous axis", x, [2, 4, 5], 1; stats = stats2)
    parity("anisotropic sizes", x, [(3, 5), (6, 10)], 2; stats = stats2)
    parity("partial edges", x, [(3, 5)], 2; stats = stats2, edge = BSR.Partial())
    parity("a window wider than a slab", x, [(4, 30)], 2; stats = stats2)
    parity("the whole axis in one window", x, [(20, 33)], 2; stats = stats2, edge = BSR.Partial())
    parity("overlapping windows", x, BSR.ScaleSet((BSR.Sizes([4]), BSR.Sizes([6])); combine = BSR.Product(), placement = BSR.Stride(2)), 2;
           stats = stats2)
    parity("two fields and their covariance", (x = x, y = y), [4], 2;
           stats = (c = BSR.Cov(:x, :y), mx = BSR.Mean(:x), my = BSR.Mean(:y)))
    parity("element weights", x, [4, 8], 2; stats = (m = BSR.Mean(), v = BSR.Var()), weights = w)
    parity("per-axis weight factors", x, [4, 8], 2; stats = (m = BSR.Mean(), v = BSR.Var()), weights = (wx, wz))
    parity("weights on the partitioned axis only", x, [(4, 6)], 2;
           stats = (m = BSR.Mean(),), weights = (nothing, wz))
    xn = copy(x); xn[3, 7] = NaN; xn[11, 20] = Inf
    parity("skipnan", xn, [4, 8], 2; stats = stats2, skipnan = true)
    parity("weights and skipnan", xn, [4], 2; stats = stats2, weights = w, skipnan = true)
    parity("float32 with a large offset", Float32.(x .+ 1.0f5), [4, 8], 2;
           stats = (m = BSR.Mean(), v = BSR.Var()), rtol = 1e-4)
    parity("three dimensions", randn(8, 6, 21), [(2, 3, 4), (4, 3, 8)], 3; stats = stats2)

    # A prepared handle runs again on new data without rebuilding the plan or the exchange. The second
    # input is derived from the first rather than drawn, since `@testset` reseeds the global RNG.
    Test.@testset "prepared reuse (rank $RANK)" begin
        z = 2 .* x .+ 1
        lo, len = slab_bounds(size(x, 2), NRANKS)[RANK+1]
        rows = (lo + 1):(lo + len)
        xs = collect(BSR.axis_slice(x, 2, rows))
        zs = collect(BSR.axis_slice(z, 2, rows))
        stats = (m = BSR.Mean(), v = BSR.Var())
        p = BSR.prepare(BSR.Partitioned(xs; axis = 2), [4, 8]; stats = stats,
                        backend = CB.MPIBackend(CB.SerialBackend(), COMM))
        # Each result aliases the handle's arrays, so snapshot before running again.
        snap(r) = [map(copy, r[i]) for i in 1:length(BSR.windows(p))]
        first_run = snap(BSR.blockstats!(p, xs))
        second = snap(BSR.blockstats!(p, zs))
        again = snap(BSR.blockstats!(p, xs))
        for (i, w) in enumerate(BSR.windows(p))
            want = BSR.blockstats(z, [4, 8]; stats = stats, backend = CB.SerialBackend())[i]
            gw = BSR.windows(BSR.blockstats(z, [4, 8]; stats = stats, backend = CB.SerialBackend()))[i]
            idx = [findfirst(==(o), collect(BSR.origins(gw[2]))) for o in BSR.origins(w[2])]
            for k in keys(stats)
                Test.@test same(second[i][k], BSR.axis_slice(want[k], 2, idx); rtol = 1e-12)
                Test.@test again[i][k] == first_run[i][k]      # the same input gives the same bits back
                Test.@test first_run[i][k] != second[i][k]
            end
        end
    end
end

MPI.Barrier(COMM)
MPI.Finalize()
