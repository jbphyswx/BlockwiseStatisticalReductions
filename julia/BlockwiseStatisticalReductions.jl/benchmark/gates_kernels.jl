using ComputationalBackends: ComputationalBackends as CB

# Serial kernel gates: ratio of kernel time to a streaming sum over the same input, accumulated in Float64.
#
# Float32 inputs form their own group and run the path the API takes by default: shifted observations
# accumulated in Float32, which keeps full-width lanes and stays at Float32's own epsilon at any offset
# (see `Shifted`). The per-element subtraction is not free, so these still sit above their Float64
# counterparts; the thresholds guard the measured behaviour of that default path.

const SERIAL = CB.SerialBackend()

roofline64(A::AbstractArray) = (sum(Float64, A); best(() -> sum(Float64, A)))

# Float64 input accumulates in Float64 against a Float64-accumulating roofline; Float32 input runs the
# default shifted-narrow path against a Float32-accumulating one, so each ratio compares a kernel with
# the streaming sum it could at best match.
function base_gate(stats, sizes, T, threshold; label, nfields = 1)
    shifted = T !== Float64
    gate(shifted ? "kernels-f32" : "kernels", "$label $(T) tile $(sizes[1])x", threshold) do
        n = 4096
        fields = ntuple(_ -> randn(T, n, n), nfields)
        names = ntuple(i -> Symbol(:f, i), nfields)
        w = BSR.tiled((n, n), sizes, BSR.Truncate())
        Tacc = shifted ? T : BSR.accumulation_eltype(T)
        C, _, _, _ = BSR.assemble(stats, names, T, Tacc)
        out = BSR.AccumulatorArray(C, fields[1], BSR.shape(w); uniform = (n = prod(sizes),))
        raw = NamedTuple{names}(fields)
        src = shifted ? NamedTuple{names}(map(f -> BSR.Shifted(f, BSR.field_shift(f, T, SERIAL)), fields)) : raw
        BSR.boxfold!(out, src, w, SERIAL)
        t = best(() -> BSR.boxfold!(out, src, w, SERIAL))
        base = shifted ? sum(f -> (sum(T, f); best(() -> sum(T, f))), fields) : sum(roofline64, fields)
        return t / base
    end
end

# (Float64 acceptance threshold, Float32 guard on the default shifted-narrow path) per case.
for (T, mean8, mean2, meanvar, meanminmax, cov) in ((Float64, 1.25, 2.0, 1.5, 1.5, 1.3),
                                                    (Float32, 1.5, 2.0, 3.1, 3.0, 3.0))
    base_gate((BSR.Mean(),), (8, 8), T, mean8; label = "mean")
    base_gate((BSR.Mean(),), (2, 2), T, mean2; label = "mean")
    base_gate((BSR.Mean(), BSR.Var()), (8, 8), T, meanvar; label = "mean+var")
    base_gate((BSR.Mean(), BSR.Min(), BSR.Max()), (8, 8), T, meanminmax; label = "mean+min+max")
    base_gate((BSR.Cov(:f1, :f2),), (8, 8), T, cov; label = "cov", nfields = 2)
end

gate("kernels", "coarsen 2x of a 2048² VarAcc array vs streaming its components", 1.3) do
    fine = BSR.AccumulatorArray(BSR.VarAcc{Float64}, zeros(1), (2048, 2048))
    fill!(BSR.component(fine, :n), 4)
    Random.randn!(BSR.component(fine, :mean))
    Random.rand!(BSR.component(fine, :M2))
    grid = BSR.tiled((2048, 2048), (2, 2), BSR.Truncate())
    out = BSR.AccumulatorArray(BSR.VarAcc{Float64}, zeros(1), (1024, 1024))
    BSR.boxfold!(out, fine, grid, SERIAL)
    t = best(() -> BSR.boxfold!(out, fine, grid, SERIAL))
    base = roofline(BSR.component(fine, :n)) + roofline(BSR.component(fine, :mean)) + roofline(BSR.component(fine, :M2))
    return t / base
end

# The two-stack scan does about three merges per cell; `compose!` does one. Both are merge-bound.
gate("kernels", "scan window 16 along axis 1 of 1024² VarAcc vs compose! (1 merge/cell)", 4.0) do
    x = randn(1024, 1024)
    lifted_w = BSR.tiled((1024, 1024), (1, 1), BSR.Truncate())
    lifted = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (1024, 1024); uniform = (n = 1,)), (x,), lifted_w, SERIAL)
    out = BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (1009, 1024))
    scratch = BSR.ScanScratch(BSR.VarAcc{Float64}, 16)
    BSR.scan!(out, lifted, 1, 16, false, scratch, SERIAL)
    t = best(() -> BSR.scan!(out, lifted, 1, 16, false, scratch, SERIAL))
    c = BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (1009, 1024))
    BSR.compose!(c, lifted, lifted, 1, 1:1009, 16:1024, SERIAL)
    tc = best(() -> BSR.compose!(c, lifted, lifted, 1, 1:1009, 16:1024, SERIAL))
    return t / tc
end

gate("kernels", "zero allocation: boxfold!/compose!/scan!/finalize! (bytes)", 0.0) do
    x = randn(512, 512)
    w = BSR.tiled((512, 512), (8, 8), BSR.Truncate())
    C, _, _, _ = BSR.assemble((BSR.Mean(), BSR.Var(), BSR.Min()), (:x,), Float64, Float64)
    out = BSR.AccumulatorArray(C, x, BSR.shape(w))
    BSR.boxfold!(out, (x,), w, SERIAL)
    a_w = (BSR.strided(512, 4, 1, BSR.Truncate()), BSR.tiled(512, 1, BSR.Truncate()))
    a = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, BSR.shape(a_w)), (x,), a_w, SERIAL)
    c = BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (505, 512))
    BSR.compose!(c, a, a, 1, 1:505, 5:509, SERIAL)
    scratch = BSR.ScanScratch(BSR.VarAcc{Float64}, 4)
    lifted = BSR.boxfold!(BSR.AccumulatorArray(BSR.VarAcc{Float64}, x, (512, 512)), (x,), BSR.tiled((512, 512), (1, 1), BSR.Truncate()), SERIAL)
    BSR.scan!(a, lifted, 1, 4, false, scratch, SERIAL)
    dst = Array{Float64}(undef, size(out))
    BSR.finalize!(dst, BSR.member_array(out, Val(1)), BSR.Var(), SERIAL)
    S = Val(BSR.static_shape(w))
    bytes = @allocated(BSR.boxfold!(out, (x,), w, S, SERIAL)) + @allocated(BSR.compose!(c, a, a, 1, 1:505, 5:509, SERIAL)) +
            @allocated(BSR.scan!(a, lifted, 1, 4, false, scratch, SERIAL)) +
            @allocated(BSR.finalize!(dst, BSR.member_array(out, Val(1)), BSR.Var(), SERIAL))
    return Float64(bytes)
end
