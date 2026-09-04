"""
    finalize!(dst, src::AccumulatorArray, tag, backend) -> dst

Write `finalize(tag, src[i], eltype(dst))` into every cell of `dst`. Select a composite member with
[`member_array`](@ref) first.
"""
function finalize!(dst::AbstractArray{Tout,N}, src::AccumulatorArray{A,N}, tag::AbstractStatistic, ::CB.AbstractSerialBackend) where {Tout,A,N}
    size(dst) == size(src) || throw(DimensionMismatch("destination $(size(dst)) does not match accumulators $(size(src))"))
    @inbounds for i in eachindex(dst, src)
        dst[i] = finalize(tag, src[i], Tout)
    end
    return dst
end
finalize!(dst::AbstractArray{Tout,N}, src::AccumulatorArray{A,N}, tag::AbstractStatistic, b::CB.AbstractExecutionBackend) where {Tout,A,N} =
    missing_extension(b)
