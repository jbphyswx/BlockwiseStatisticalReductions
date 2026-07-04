module BlockwiseStatisticalReductionsKernelAbstractionsExt

# Device-agnostic GPU (`GPUBackend{B}`) execution via KernelAbstractions.jl (CUDA / ROCm / oneAPI /
# Metal, and `CPU()` for testing). One workitem per OUTPUT CELL: each cell reduces its own block via
# the SAME `reduce_block`/`coarsen_block` the CPU path uses, so built-in stats get their specialized
# bulk kernels and — because cells are independent — the result matches the serial one exactly (up to
# floating point). This "one cell per workitem, reduce in registers" mapping is the recommended path
# for the many-small-cells regime: no shared memory, no `@synchronize`, no barrier fragility. (A
# shared-memory / warp-shuffle tree for the few-large-blocks regime is a later optimization.)
#
# Accumulator arrays and inputs are expected to already live on the KA backend's device (regular
# `Array`s for `CPU()`, `CuArray`s for CUDA, …); the executor runs one DAG layer per kernel launch.

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const, synchronize
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

@kernel function _base_kernel!(out, @Const(A), window)
    I = @index(Global, Cartesian)
    @inbounds out[I] = BSR.reduce_block(eltype(out), (A,), BSR._block_lo(I, window), BSR._block_hi(I, window))
end

@kernel function _base_kernel2!(out, @Const(X), @Const(Y), window)
    I = @index(Global, Cartesian)
    @inbounds out[I] = BSR.reduce_block(eltype(out), (X, Y), BSR._block_lo(I, window), BSR._block_hi(I, window))
end

@kernel function _merge_kernel!(out, @Const(fine), window)
    I = @index(Global, Cartesian)
    @inbounds out[I] = BSR.coarsen_block(fine, BSR._block_lo(I, window), BSR._block_hi(I, window))
end

function BSR._drive_base!(out::AbstractArray{Acc,N}, inputs::Tuple, window::NTuple{N,Int},
                         b::BSR.GPUBackend) where {Acc,N}
    dev = b.backend
    if length(inputs) == 1
        _base_kernel!(dev)(out, inputs[1], window; ndrange = size(out))
    else
        _base_kernel2!(dev)(out, inputs[1], inputs[2], window; ndrange = size(out))
    end
    synchronize(dev)
    return nothing
end

function BSR._drive_merge!(out::AbstractArray{Acc,N}, fine::AbstractArray{Acc,N}, window::NTuple{N,Int},
                          b::BSR.GPUBackend) where {Acc,N}
    _merge_kernel!(b.backend)(out, fine, window; ndrange = size(out))
    synchronize(b.backend)
    return nothing
end

end # module
