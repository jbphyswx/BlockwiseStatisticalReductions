"""
    finalize!(dst, src::AccumulatorArray, tag, [shifts,] backend) -> dst

Write `finalize(tag, src[i], eltype(dst))` into every cell of `dst`, undoing `shifts` (one per field the
accumulator binds, zero by default) first. Select a composite member with [`member_array`](@ref) first.
"""
function finalize!(dst::AbstractArray{Tout,N}, src::AccumulatorArray{A,N}, tag::AbstractStatistic,
                   shifts::Tuple, ::CB.AbstractSerialBackend) where {Tout,A,N}
    size(dst) == size(src) || throw(DimensionMismatch("destination $(size(dst)) does not match accumulators $(size(src))"))
    @inbounds for i in eachindex(dst, src)
        dst[i] = finalize(tag, unshift(src[i], shifts), Tout)
    end
    return dst
end
finalize!(dst::AbstractArray, src::AccumulatorArray, tag::AbstractStatistic, backend::CB.AbstractExecutionBackend) =
    finalize!(dst, src, tag, _no_shift(src), backend)
finalize!(dst::AbstractArray{Tout,N}, src::AccumulatorArray{A,N}, tag::AbstractStatistic, shifts::Tuple,
          b::CB.AbstractExecutionBackend) where {Tout,A,N} = missing_extension(b)

_no_shift(::AccumulatorArray{A}) where {A} = ntuple(_ -> NoShift(), Val(arity(A)))
