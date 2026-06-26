# ─────────────────────────────────────────────────────────────────────────────
# Sliding (overlapping) window reductions
# ─────────────────────────────────────────────────────────────────────────────
#
# For overlapping windows (stride < size) the disjoint-merge tower no longer applies, but the same
# accumulator algebra still does. We use the two-stack Sliding-Window Aggregation (SWAG): it
# combines any monoid over a sliding window in O(1) amortized per step with NO inverse and NO
# double-counting (so it is correct and numerically stable for variance/covariance, not just
# idempotent min/max). An N-D box reduction factors into one 1-D SWAG pass per axis (associativity),
# giving O(|X|·N) regardless of window size. Strided / offset outputs are a subsample of the dense
# result. When stride == window and origin == 1 this reproduces the non-overlapping blockwise
# reduction exactly — sliding windows generalize blockwise reduction.

# One dense (stride-1) SWAG pass over a line `v`, window `w`, writing length-(n-w+1) `out`.
# `bv`/`ba`/`fa` are reusable scratch stacks (length ≥ w): back values, back aggregates, front
# aggregates of the classic two-stack queue aggregation.
function _swag_line!(out::AbstractVector{Acc}, v::AbstractVector{Acc}, w::Int,
                     bv::Vector{Acc}, ba::Vector{Acc}, fa::Vector{Acc}) where {Acc}
    btop = 0
    ftop = 0
    oi = firstindex(out)
    cnt = 0
    @inbounds for r in eachindex(v)
        # push v[r] onto the back stack
        btop += 1
        bv[btop] = v[r]
        ba[btop] = btop == 1 ? v[r] : merge(ba[btop - 1], v[r])
        cnt += 1
        if cnt >= w
            # query aggregate of the current window
            q = ftop == 0 ? ba[btop] : (btop == 0 ? fa[ftop] : merge(fa[ftop], ba[btop]))
            out[oi] = q
            oi += 1
            # evict the oldest element (front top); flip back→front first if the front is empty
            if ftop == 0
                while btop > 0
                    x = bv[btop]
                    btop -= 1
                    ftop += 1
                    fa[ftop] = ftop == 1 ? x : merge(x, fa[ftop - 1])
                end
            end
            ftop -= 1
        end
    end
    return out
end

# Dense (stride-1) sliding reduction of an accumulator array along dimension `d`, window `w`.
function sliding_axis(A::AbstractArray{Acc,N}, d::Int, w::Int) where {Acc,N}
    n = size(A, d)
    m = n - w + 1
    m >= 1 || throw(ArgumentError("sliding window $w exceeds extent $n along dim $d"))
    out = Array{Acc,N}(undef, ntuple(i -> i == d ? m : size(A, i), Val(N)))
    bv = Vector{Acc}(undef, w)
    ba = Vector{Acc}(undef, w)
    fa = Vector{Acc}(undef, w)
    if N == 1
        _swag_line!(out, A, w, bv, ba, fa)
    else
        od = Tuple(i for i in 1:N if i != d)
        for (so, si) in zip(eachslice(out; dims = od), eachslice(A; dims = od))
            _swag_line!(so, si, w, bv, ba, fa)
        end
    end
    return out
end

# Lift the data into accumulators, then apply a dense sliding pass per reduced axis.
function sliding_dense(::Type{Acc}, inputs::Tuple, window::NTuple{N,Int}) where {Acc,N}
    A = Array{Acc,N}(undef, size(inputs[1]))
    @inbounds for I in CartesianIndices(A)
        A[I] = _leaf(Acc, inputs, I)
    end
    for d in 1:N
        window[d] > 1 && (A = sliding_axis(A, d, window[d]))
    end
    return A     # size: (size(data,d) - window[d] + 1) per reduced dim
end

"""
    sliding_reduce(::Type{Acc}, inputs, window, stride; origin) -> Array{Acc}

Overlapping-window reduction: the accumulator of every window of size `window`, placed at output
positions `origin .+ (k .- 1) .* stride` that fit inside the data. `inputs` is one array or a tuple
of arrays (arity-2). With `stride == window` and `origin == 1` this equals the non-overlapping
blockwise reduction.
"""
function sliding_reduce(::Type{Acc}, inputs::Tuple, window::NTuple{N,Int}, stride::NTuple{N,Int};
                        origin::NTuple{N,Int} = ntuple(_ -> 1, Val(N))) where {Acc,N}
    dense = sliding_dense(Acc, inputs, window)
    md = size(dense)
    for d in 1:N
        1 <= origin[d] <= md[d] || throw(ArgumentError("origin $origin out of range (valid positions $md) in dim $d"))
        stride[d] >= 1 || throw(ArgumentError("stride must be ≥ 1"))
    end
    osz = ntuple(d -> (md[d] - origin[d]) ÷ stride[d] + 1, Val(N))
    out = Array{Acc,N}(undef, osz)
    @inbounds for I in CartesianIndices(out)
        src = CartesianIndex(ntuple(d -> origin[d] + (I[d] - 1) * stride[d], Val(N)))
        out[I] = dense[src]
    end
    return out
end
sliding_reduce(::Type{Acc}, data::AbstractArray, window::NTuple, stride::NTuple; kw...) where {Acc} =
    sliding_reduce(Acc, (data,), window, stride; kw...)
