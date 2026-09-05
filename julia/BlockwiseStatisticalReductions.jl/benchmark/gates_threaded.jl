using OhMyThreads: OhMyThreads as OMT

# Threaded gates. The reference is a *threaded* streaming sum, not a serial one: this work is
# bandwidth-bound, and on this machine 8 threads take a streaming read only about 2.75x faster than one
# thread. Measuring against the serial sum would report a speedup the memory system cannot give, so each
# ratio asks the same question the serial gates ask — how close is the kernel to just moving the bytes.

const THREADED = CB.ThreadedBackend()

roofline_threaded(A::AbstractArray) = (OMT.treduce(+, A); best(() -> OMT.treduce(+, A)))

function threaded_gate(label, shape, scales, stats, threshold; nfields = 1)
    gate("threaded", "$label (vs threaded sum)", threshold) do
        names = ntuple(i -> Symbol(:f, i), nfields)
        fields = NamedTuple{names}(ntuple(_ -> randn(shape...), nfields))
        p = prepare(fields, scales; stats = stats, backend = THREADED)
        BSR.blockstats!(p, fields)
        t = best(() -> BSR.blockstats!(p, fields))
        return t / sum(roofline_threaded, values(fields))
    end
end

threaded_gate("base pass mean 8x 4096²", (4096, 4096), [8], (BSR.Mean(),), 1.3)
threaded_gate("base pass mean+var 8x 4096²", (4096, 4096), [8], (BSR.Mean(), BSR.Var()), 2.5)
threaded_gate("cov 8x 4096²", (4096, 4096), [8], (BSR.Cov(:f1, :f2),), 3.0; nfields = 2)
threaded_gate("6 scales [2..64] mean+var 4096²", (4096, 4096), [2, 4, 8, 16, 32, 64], (BSR.Mean(), BSR.Var()), 5.5)
threaded_gate("tile sizes 2..64 mean 4096²", (4096, 4096), collect(2:64), (BSR.Mean(),), 25.0)

# Threading must actually pay on a request big enough to fill the threads. Registered only when there is
# more than one thread, so a single-threaded run still exercises the gates above.
if Threads.nthreads() > 1
    gate("threaded", "tile sizes 2..64 mean 4096²: threaded/serial time", 0.7) do
        x = randn(4096, 4096)
        scales = collect(2:64)
        ps = prepare(x, scales; stats = (BSR.Mean(),), backend = CB.SerialBackend())
        pt = prepare(x, scales; stats = (BSR.Mean(),), backend = THREADED)
        BSR.blockstats!(ps, x); BSR.blockstats!(pt, x)
        return best(() -> BSR.blockstats!(pt, x)) / best(() -> BSR.blockstats!(ps, x))
    end
    gate("threaded", "15 dense sizes 2..16 mean 1024²: threaded/serial time", 0.7) do
        x = randn(1024, 1024)
        spec = BSR.ScaleSet(BSR.Sizes(collect(2:16)); placement = BSR.Dense())
        ps = prepare(x, spec; stats = (BSR.Mean(),), backend = CB.SerialBackend())
        pt = prepare(x, spec; stats = (BSR.Mean(),), backend = THREADED)
        BSR.blockstats!(ps, x); BSR.blockstats!(pt, x)
        return best(() -> BSR.blockstats!(pt, x), 3) / best(() -> BSR.blockstats!(ps, x), 3)
    end
end
