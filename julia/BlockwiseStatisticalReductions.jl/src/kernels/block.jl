# ─────────────────────────────────────────────────────────────────────────────
# Block kernels: the base reduction and the cross-scale merge
# ─────────────────────────────────────────────────────────────────────────────
#
# Two generic methods drive the whole non-overlapping engine:
#
#   blockreduce!(out, inputs, window)  — base pass: fold raw data over non-overlapping blocks
#   coarsen!(out, fine, window)        — merge step: combine already-computed accumulators
#
# Both write into a preallocated `Array{Acc}` (one method each, generic over `Acc`), use the
# sequential `blockfold`, and allocate nothing beyond `out`. There is no `is_base` branch: the base
# pass dispatches on a data array, the merge on an accumulator array. With non-overlapping blocks
# and low-index truncation, the high-index remainder is simply not covered (size `out`
# accordingly); no padding.

# Resolve one observation into an accumulator, routing 1 or 2 input fields by input-tuple arity.
@inline _leaf(::Type{Acc}, inputs::Tuple{<:AbstractArray}, J) where {Acc} =
    lift(Acc, @inbounds inputs[1][J])
@inline _leaf(::Type{Acc}, inputs::Tuple{<:AbstractArray,<:AbstractArray}, J) where {Acc} =
    lift(Acc, (@inbounds inputs[1][J]), (@inbounds inputs[2][J]))

# Block box [lo, hi] for output cell I under non-overlapping `window` (low-index origin).
@inline _block_lo(I::CartesianIndex{N}, window::NTuple{N,Int}) where {N} =
    CartesianIndex(ntuple(i -> (I[i] - 1) * window[i] + 1, Val(N)))
@inline _block_hi(I::CartesianIndex{N}, window::NTuple{N,Int}) where {N} =
    CartesianIndex(ntuple(i -> I[i] * window[i], Val(N)))

"""
    blockreduce!(out::AbstractArray{Acc,N}, inputs, window) -> out

Base pass: fold the raw data over non-overlapping `window`-sized blocks into the preallocated
accumulator array `out`. `inputs` is one array (arity-1 statistics) or a 2-tuple of arrays
(arity-2 statistics such as covariance); a single array is accepted directly. `out` must be sized
`size(data) .÷ window` (the high-index remainder is truncated away).
"""
function blockreduce!(out::AbstractArray{Acc,N}, inputs::Tuple, window::NTuple{N,Int},
                      backend::AbstractExecutionBackend = SerialBackend()) where {Acc,N}
    @boundscheck _check_block_inputs(out, inputs, window)
    _drive_base!(out, inputs, window, backend)
    return out
end
blockreduce!(out::AbstractArray, data::AbstractArray, window::NTuple,
             backend::AbstractExecutionBackend = SerialBackend()) =
    blockreduce!(out, (data,), window, backend)

"""
    reduce_block(::Type{Acc}, inputs::Tuple, lo::CartesianIndex, hi::CartesianIndex) -> Acc

Reduce the raw-data block `[lo, hi]` (over `inputs`) into one accumulator of type `Acc`. This is a
public, overridable hook: the default is the generic sequential [`blockfold`], but a statistic may
specialize it on `Acc` to reduce a whole block by any means (SIMD `sum`, two-pass, BLAS, …). The
whole-*array* bulk path (`_drive_base!` specialized on `Acc`) is the reshape-style fast route BSR uses
for its built-ins; `reduce_block` is the per-cell route also used under threading.
"""
@inline reduce_block(::Type{Acc}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {Acc,N} =
    blockfold(J -> _leaf(Acc, inputs, J), lo, hi)

# Serial cell driver: fill every output cell via the per-cell `reduce_block` hook. Other backends
# (Threaded) add `_drive_base!` methods in their extensions; a statistic may add a whole-array bulk
# `_drive_base!` specialized on its `Acc` (see kernels/bulk.jl for BSR's built-in bulk kernels).
function _drive_base!(out::AbstractArray{Acc,N}, inputs::Tuple, window::NTuple{N,Int}, ::SerialBackend) where {Acc,N}
    @inbounds for I in CartesianIndices(out)
        out[I] = reduce_block(Acc, inputs, _block_lo(I, window), _block_hi(I, window))
    end
    return nothing
end

"""
    coarsen!(out::AbstractArray{Acc,N}, fine::AbstractArray{Acc,N}, window) -> out

Merge step: combine the already-computed finer-scale accumulators `fine` over non-overlapping
`window`-sized blocks into the coarser accumulator array `out`. Because this uses the same
`merge` as the base pass, the result is bit-for-bit identical to reducing the raw data directly at
the coarse scale (exact hierarchical reuse). `out` must be sized `size(fine) .÷ window`.
"""
function coarsen!(out::AbstractArray{Acc,N}, fine::AbstractArray{Acc,N}, window::NTuple{N,Int},
                  backend::AbstractExecutionBackend = SerialBackend()) where {Acc,N}
    @boundscheck _check_coarsen_inputs(out, fine, window)
    _drive_merge!(out, fine, window, backend)
    return out
end

"""
    coarsen_block(fine::AbstractArray{Acc}, lo::CartesianIndex, hi::CartesianIndex) -> Acc

Combine the already-computed finer accumulators in the block `[lo, hi]` of `fine` into one coarser
accumulator. Public, overridable hook (the coarsen-side twin of [`reduce_block`]); default is the
generic sequential [`blockfold`] over `fine`. A statistic may specialize it (or the whole-array
`_drive_merge!`) to combine a whole block by vectorized reductions over the accumulators' fields.
"""
@inline coarsen_block(fine::AbstractArray{Acc,N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {Acc,N} =
    blockfold(J -> (@inbounds fine[J]), lo, hi)

function _drive_merge!(out::AbstractArray{Acc,N}, fine::AbstractArray{Acc,N}, window::NTuple{N,Int}, ::SerialBackend) where {Acc,N}
    @inbounds for I in CartesianIndices(out)
        out[I] = coarsen_block(fine, _block_lo(I, window), _block_hi(I, window))
    end
    return nothing
end

# ── Allocating conveniences (the planner/executor preallocate instead; these aid testing) ──

"Allocate an uninitialized accumulator array of element type `Acc` and the given `shape`."
allocate_accumulators(::Type{Acc}, shape::NTuple{N,Int}) where {Acc,N} = Array{Acc,N}(undef, shape)

"Output shape of a non-overlapping `window` reduction of an array of size `insize` (floor / truncate)."
@inline reduced_shape(insize::NTuple{N,Int}, window::NTuple{N,Int}) where {N} =
    ntuple(i -> insize[i] ÷ window[i], Val(N))

"""
    blockreduce(::Type{Acc}, inputs, window) -> Array{Acc}

Allocating base reduction: size and fill a fresh accumulator array. `inputs` is one array or a
2-tuple of arrays.
"""
function blockreduce(::Type{Acc}, inputs::Tuple, window::NTuple{N,Int}) where {Acc,N}
    out = allocate_accumulators(Acc, reduced_shape(size(inputs[1]), window))
    return blockreduce!(out, inputs, window)
end
blockreduce(::Type{Acc}, data::AbstractArray, window::NTuple) where {Acc} = blockreduce(Acc, (data,), window)

# ── Bounds checks (only run under @boundscheck) ────────────────────────────────

@inline function _check_block_inputs(out::AbstractArray{<:Any,N}, inputs::Tuple, window::NTuple{N,Int}) where {N}
    for arr in inputs
        ndims(arr) == N || throw(DimensionMismatch("input has $(ndims(arr)) dims, expected $N"))
        size(arr) == size(inputs[1]) || throw(DimensionMismatch("inputs must share size"))
        for i in 1:N
            size(out, i) * window[i] <= size(arr, i) ||
                throw(DimensionMismatch("block $window at out size $(size(out)) exceeds input size $(size(arr)) in dim $i"))
        end
    end
    return nothing
end

@inline function _check_coarsen_inputs(out::AbstractArray{<:Any,N}, fine::AbstractArray{<:Any,N}, window::NTuple{N,Int}) where {N}
    for i in 1:N
        size(out, i) * window[i] <= size(fine, i) ||
            throw(DimensionMismatch("coarsen $window at out size $(size(out)) exceeds fine size $(size(fine)) in dim $i"))
    end
    return nothing
end
