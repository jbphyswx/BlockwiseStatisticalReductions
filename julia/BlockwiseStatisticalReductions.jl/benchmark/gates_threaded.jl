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
        p = BSR.prepare(fields, scales; stats = stats, backend = THREADED)
        BSR.blockstats!(p, fields)
        t = best(() -> BSR.blockstats!(p, fields))
        return t / sum(roofline_threaded, values(fields))
    end
end

# A base pass over 4096² is a few milliseconds at 8 threads, which is the same order as the task pool's
# own spawn and join, so the ratio of two such measurements does not repeat: identical code measured
# 1.09-1.42 across processes there, with the kernel term alone swinging 2.70-3.45 ms. At 8192² both terms
# are long enough to drown that out and the same ratios repeat to within a few percent, so the base-pass
# gates run there. The multi-scale gates below already take tens of milliseconds and stay at 4096².
threaded_gate("base pass mean 8x 8192²", (8192, 8192), [8], (BSR.Mean(),), 1.4)
threaded_gate("base pass mean+var 8x 8192²", (8192, 8192), [8], (BSR.Mean(), BSR.Var()), 2.5)
threaded_gate("cov 8x 8192²", (8192, 8192), [8], (BSR.Cov(:f1, :f2),), 2.0; nfields = 2)
threaded_gate("6 scales [2..64] mean+var 4096²", (4096, 4096), [2, 4, 8, 16, 32, 64], (BSR.Mean(), BSR.Var()), 5.5)
threaded_gate("tile sizes 2..64 mean 4096²", (4096, 4096), collect(2:64), (BSR.Mean(),), 25.0)

# Threading must actually pay on a request big enough to fill the threads. Registered only when there is
# more than one thread, so a single-threaded run still exercises the gates above.
if Threads.nthreads() > 1
    gate("threaded", "tile sizes 2..64 mean 4096²: threaded/serial time", 0.7) do
        x = randn(4096, 4096)
        scales = collect(2:64)
        ps = BSR.prepare(x, scales; stats = (BSR.Mean(),), backend = CB.SerialBackend())
        pt = BSR.prepare(x, scales; stats = (BSR.Mean(),), backend = THREADED)
        BSR.blockstats!(ps, x); BSR.blockstats!(pt, x)
        return best(() -> BSR.blockstats!(pt, x)) / best(() -> BSR.blockstats!(ps, x))
    end
    gate("threaded", "15 dense sizes 2..16 mean 1024²: threaded/serial time", 0.7) do
        x = randn(1024, 1024)
        spec = BSR.ScaleSet(BSR.Sizes(collect(2:16)); placement = BSR.Dense())
        ps = BSR.prepare(x, spec; stats = (BSR.Mean(),), backend = CB.SerialBackend())
        pt = BSR.prepare(x, spec; stats = (BSR.Mean(),), backend = THREADED)
        BSR.blockstats!(ps, x); BSR.blockstats!(pt, x)
        return best(() -> BSR.blockstats!(pt, x), 3) / best(() -> BSR.blockstats!(ps, x), 3)
    end
end
