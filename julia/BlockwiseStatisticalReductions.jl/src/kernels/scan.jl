"Two-stack scratch for [`scan!`](@ref); stacks hold `size` accumulators each."
struct ScanScratch{A, VA <: AbstractVector{A}}
    values::VA
    back::VA
    front::VA
end
ScanScratch(::Type{A}, size::Integer) where {A} =
    ScanScratch(Vector{A}(undef, size), Vector{A}(undef, size), Vector{A}(undef, size))

"""
    scan!(out, src, axis, size, partial, scratch, backend) -> out

Dense sliding windows of `size` cells along `axis`: `out` cell `j` is the combination of `src` cells
`j : j+size-1` along `axis`. With `partial`, `out` has the extent of `src` and trailing windows are
clipped; otherwise it has `extent - size + 1` cells along `axis`. Any monoid; no inverse is used.
"""
function scan!(out::AccumulatorArray{A,N}, src::AccumulatorArray{A,N}, axis::Int, size::Int, partial::Bool,
               scratch::ScanScratch{A}, ::CB.AbstractSerialBackend) where {A,N}
    1 <= axis <= N || throw(ArgumentError("axis $axis out of range"))
    size >= 1 || throw(ArgumentError("window size must be ≥ 1"))
    n = Base.size(src, axis)
    m = partial ? n : max(n - size + 1, 0)
    Base.size(out, axis) == m || throw(DimensionMismatch("output has $(Base.size(out, axis)) cells along axis $axis, expected $m"))
    all(d -> d == axis || Base.size(out, d) == Base.size(src, d), 1:N) ||
        throw(DimensionMismatch("output $(Base.size(out)) does not match source $(Base.size(src)) off axis $axis"))
    length(scratch.values) >= size || throw(ArgumentError("scratch holds $(length(scratch.values)) < $size accumulators"))
    lines = CartesianIndices(ntuple(d -> d == axis ? (1:1) : (1:Base.size(src, d)), Val(N)))
    for L in lines
        _scan_line!(out, src, axis, L, size, partial, scratch)
    end
    return out
end
scan!(out::AccumulatorArray{A,N}, src::AccumulatorArray{A,N}, axis::Int, size::Int, partial::Bool, scratch::ScanScratch{A},
      b::CB.AbstractExecutionBackend) where {A,N} = missing_extension(b)

@inline _at(L::CartesianIndex{N}, axis::Int, j::Int) where {N} = CartesianIndex(Base.setindex(Tuple(L), j, axis))

# Two-stack queue aggregation along one line: push each cell on the back stack, emit the window
# aggregate once `size` cells are queued, then evict the oldest cell (flipping back → front when the
# front is empty). With `partial`, keep evicting after the last push to emit the clipped tail windows.
function _scan_line!(out::AccumulatorArray{A,N}, src::AccumulatorArray{A,N}, axis::Int, L::CartesianIndex{N},
                     size::Int, partial::Bool, scratch::ScanScratch{A}) where {A,N}
    values, back, front = scratch.values, scratch.back, scratch.front
    n = Base.size(src, axis)
    btop = 0
    ftop = 0
    queued = 0
    j = 1
    @inbounds for r in 1:n
        x = src[_at(L, axis, r)]
        btop += 1
        values[btop] = x
        back[btop] = btop == 1 ? x : merge(back[btop-1], x)
        queued += 1
        if queued == size
            out[_at(L, axis, j)] = ftop == 0 ? back[btop] : merge(front[ftop], back[btop])
            j += 1
            ftop, btop = _evict!(values, back, front, ftop, btop)
            queued -= 1
        end
    end
    if partial
        @inbounds while queued > 0
            out[_at(L, axis, j)] = ftop == 0 ? back[btop] : (btop == 0 ? front[ftop] : merge(front[ftop], back[btop]))
            j += 1
            ftop, btop = _evict!(values, back, front, ftop, btop)
            queued -= 1
        end
    end
    return nothing
end

# Remove the oldest queued cell: refill the front stack from the back stack when it is empty.
@inline function _evict!(values, back, front, ftop::Int, btop::Int)
    if ftop == 0
        @inbounds while btop > 0
            x = values[btop]
            btop -= 1
            ftop += 1
            front[ftop] = ftop == 1 ? x : merge(x, front[ftop-1])
        end
    end
    return ftop - 1, btop
end