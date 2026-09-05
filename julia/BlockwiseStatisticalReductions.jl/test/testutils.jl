using Statistics: Statistics
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

# Brute-force reference: `f` over the elements of every window of `w` in `data`.
function brute(f, data::AbstractArray{T,N}, w::BSR.Window{N}) where {T,N}
    out = Array{Float64,N}(undef, BSR.shape(w))
    for I in CartesianIndices(out)
        rngs = ntuple(d -> BSR.window_range(w[d], I[d]), N)
        out[I] = f(vec(collect(view(data, rngs...))))
    end
    return out
end

# Brute-force reference for a two-field statistic.
function brute2(f, x::AbstractArray{T,N}, y::AbstractArray{T,N}, w::BSR.Window{N}) where {T,N}
    out = Array{Float64,N}(undef, BSR.shape(w))
    for I in CartesianIndices(out)
        rngs = ntuple(d -> BSR.window_range(w[d], I[d]), N)
        out[I] = f(vec(collect(view(x, rngs...))), vec(collect(view(y, rngs...))))
    end
    return out
end

# Elementwise approximate equality that treats NaN == NaN (undefined statistics of tiny windows).
function approx_nan(a::AbstractArray, b::AbstractArray; rtol = 1e-9, atol = 0.0)
    size(a) == size(b) || return false
    return all(i -> (isnan(a[i]) && isnan(b[i])) || isapprox(a[i], b[i]; rtol = rtol, atol = atol), eachindex(a, b))
end

# Allocate storage for accumulator `A` over the cells of window `w`, matching `proto`'s array type.
allocate(::Type{A}, w::BSR.Window, proto::AbstractArray) where {A} = BSR.AccumulatorArray(A, proto, BSR.shape(w))

# Finalize `tag` from member `k` of composite storage into a fresh Float64 array.
function values_of(accs::BSR.AccumulatorArray, tag, ::Val{k}) where {k}
    src = BSR.member_array(accs, Val(k))
    return BSR.finalize!(Array{Float64}(undef, size(src)), src, tag, BSR.CB.SerialBackend())
end
values_of(accs::BSR.AccumulatorArray, tag) = BSR.finalize!(Array{Float64}(undef, size(accs)), accs, tag, BSR.CB.SerialBackend())

using Adapt: Adapt

# An accumulator array with its components copied to the host in bulk. Comparing two device accumulator
# arrays directly would index them one cell at a time, which on a device array is both slow and, under
# `allowscalar(false)`, an error.
host_accumulators(aa::BSR.AccumulatorArray) = collect(Adapt.adapt(Array, aa))
