module BlockwiseStatisticalReductionsKernelAbstractionsExt

# Device kernels, written once for every KernelAbstractions backend. One workitem per output cell: a
# cell's box is reduced in registers by the same `combine` the CPU runs, so nothing is shared between
# workitems and no barrier or local memory is needed. The accumulators are isbits and their storage is
# struct-of-arrays in the input's own array type, so a plan allocated from a device array is already
# device-resident and the kernels index it directly.

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index
using ComputationalBackends: ComputationalBackends as CB
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

function __init__()
    BSR.GPU_LOADED[] = true
    return nothing
end

# `get_backend` throws for an array type no backend claims, so ask whether one does before believing it.
CB.is_gpu_array(x::AbstractArray) =
    hasmethod(KA.get_backend, Tuple{typeof(x)}) && !(KA.get_backend(x) isa KA.CPU)
BSR.gpu_backend(fields::Tuple) = CB.GPUBackend(KA.get_backend(fields[1]))

@inline _device(b::CB.AbstractGPUBackend) = b.backend

# Device arrays do not index on the host, so the shift's strided sample is summed on the device.
function BSR.field_shift(field::AbstractArray, ::Type{T}, ::CB.AbstractGPUBackend) where {T}
    n = length(field)
    n == 0 && return zero(T)
    sample = view(vec(field), BSR.shift_sample(n))
    m = sum(sample) / length(sample)
    return isfinite(m) ? T(m) : zero(T)
end

# ── Kernels ──────────────────────────────────────────────────────────────────────

@kernel function _boxfold_kernel!(out, src, w, ::Val{S}) where {S}
    I = @index(Global, Cartesian)
    @inbounds out[I] = BSR.combine(eltype(out), BSR.Box(src, BSR._box_ranges(w, I, Val(S))))
end

@kernel function _compose_kernel!(out, a, b, axis, amap, bmap)
    I = @index(Global, Cartesian)
    @inbounds out[I] = BSR.compose_cell(a, b, I, axis, amap, bmap)
end

@kernel function _finalize_kernel!(dst, src, tag, shifts)
    I = @index(Global, Cartesian)
    @inbounds dst[I] = BSR.finalize(tag, BSR.unshift(src[I], shifts), eltype(dst))
end

# A dense sliding window, reduced independently per output cell. The CPU's two-stack scan is O(1)
# amortized but carries a dependency along the axis; a device has the throughput to redo the window
# instead, and every cell stays independent. `kernel_limits` reports `scan_ok = false` so the planner
# prefers composition and only reaches this kernel if a caller builds the step directly.
@kernel function _scan_kernel!(out, src, axis, size, extent)
    I = @index(Global, Cartesian)
    j = @inbounds I[axis]
    acc = @inbounds src[BSR._at(I, axis, j)]
    @inbounds for r in (j + 1):min(j + size - 1, extent)
        acc = merge(acc, src[BSR._at(I, axis, r)])
    end
    @inbounds out[I] = acc
end

# ── Kernel methods ───────────────────────────────────────────────────────────────

BSR.boxfold!(out::BSR.AccumulatorArray{A,N}, src, w::BSR.Window{N}, b::CB.AbstractGPUBackend) where {A,N} =
    BSR.boxfold!(out, src, w, Val(BSR.static_shape(w)), b)
function BSR.boxfold!(out::BSR.AccumulatorArray{A,N}, src, w::BSR.Window{N}, ::Val{S},
                      b::CB.AbstractGPUBackend) where {A,N,S}
    source = BSR._source(A, src)
    BSR._check_boxfold(out, source, w)
    BSR._check_shape(w, S)
    isempty(out) && return out
    dev = _device(b)
    _boxfold_kernel!(dev)(out, source, w, Val(S); ndrange = size(out))
    KA.synchronize(dev)
    return out
end

function BSR.compose!(out::BSR.AccumulatorArray{A,N}, a::BSR.AccumulatorArray{A,N}, b::BSR.AccumulatorArray{A,N},
                      axis::Int, amap::AbstractVector{Int}, bmap::AbstractVector{Int},
                      bk::CB.AbstractGPUBackend) where {A,N}
    BSR.check_compose(out, a, b, axis, amap, bmap)
    isempty(out) && return out
    dev = _device(bk)
    _compose_kernel!(dev)(out, a, b, axis, amap, bmap; ndrange = size(out))
    KA.synchronize(dev)
    return out
end

function BSR.scan!(out::BSR.AccumulatorArray{A,N}, src::BSR.AccumulatorArray{A,N}, axis::Int, size::Int,
                   partial::Bool, scratch::BSR.ScanScratch{A}, b::CB.AbstractGPUBackend) where {A,N}
    BSR.scan_lines(out, src, axis, size, partial, scratch)
    isempty(out) && return out
    dev = _device(b)
    _scan_kernel!(dev)(out, src, axis, size, Base.size(src, axis); ndrange = Base.size(out))
    KA.synchronize(dev)
    return out
end

function BSR.finalize!(dst::AbstractArray{Tout,N}, src::BSR.AccumulatorArray{A,N}, tag::BSR.AbstractStatistic,
                       shifts::Tuple, b::CB.AbstractGPUBackend) where {Tout,A,N}
    size(dst) == size(src) || throw(DimensionMismatch("destination $(size(dst)) does not match accumulators $(size(src))"))
    isempty(dst) && return dst
    dev = _device(b)
    _finalize_kernel!(dev)(dst, src, tag, shifts; ndrange = size(dst))
    KA.synchronize(dev)
    return dst
end

# A device wants far more cells per launch than a core does, and reduces a cell in registers, so base
# boxes stay small. Rates are placeholders until measured on real hardware — a device's bandwidth and
# merge throughput are both an order above a socket's, and `scan_ok = false` keeps the planner on
# composition, whose cells are independent.
BSR.kernel_limits(::CB.AbstractGPUBackend, N::Int) = BSR.KernelLimits(1 << 16, 256, 500e9, 5e10, 5e10, false)

end
