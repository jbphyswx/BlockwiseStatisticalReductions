# ─────────────────────────────────────────────────────────────────────────────
# Whole-block per-cell kernels for BSR's built-in single-member accumulators
# ─────────────────────────────────────────────────────────────────────────────
#
# These specialize the public per-cell `reduce_block` hook (kernels/block.jl) for the built-in
# statistics: reduce one block into a register with the simplest fast arithmetic, avoiding the
# generic `merge`/`lift` fold's per-element Welford cost. They are zero-allocation (accumulate in
# registers, one write to the output cell), are used identically by every backend (the serial and
# threaded/distributed drivers all call `reduce_block` per cell), and are therefore bit-identical
# across backends. Arbitrary user monoids and multi-member composites keep the generic fold.
#
# Numerics: Mean/Var/Cov use the stable two-pass form (mean first, then centered moments) — what
# `Statistics`/`conv3d` do — never the cancellation-prone Σx² − (Σx)²/n. Blocks are small and
# cache-resident, so the extra pass is cheap and avoids a division/guard per element.
#
# (A whole-array *separable* reduction that matches reshape-and-reduce at very coarse factors needs
# preallocated intermediate buffers to stay allocation-free; that lands with the prepared-handle work
# in §4 of docs/DESIGN_OPTIMIZATION.md. These per-cell kernels are the allocation-free core.)

@inline _wrap1(m) = CompositeAccumulator((m,))
const _Comp1{M} = CompositeAccumulator{Tuple{M}}

@inline _blocklen(lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {N} =
    prod(ntuple(d -> hi[d] - lo[d] + 1, Val(N)))

# ── Count: O(1) — a full block has exactly `prod(window)` observations ──
@inline reduce_block(::Type{_Comp1{CountAcc}}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {N} =
    _wrap1(CountAcc(_blocklen(lo, hi)))

# ── Sum ──
@inline function reduce_block(::Type{_Comp1{SumAcc{S}}}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    A = inputs[1]
    s = zero(S)
    @inbounds for J in lo:hi
        s += S(A[J])
    end
    return _wrap1(SumAcc(s))
end

# ── Mean: block sum ÷ n (no per-element division) ──
@inline function reduce_block(::Type{_Comp1{MeanAcc{S}}}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    A = inputs[1]
    s = zero(S)
    @inbounds for J in lo:hi
        s += S(A[J])
    end
    n = _blocklen(lo, hi)
    return _wrap1(MeanAcc(n, s / S(n)))
end

# ── Var/Std: two-pass — sum→mean, then Σ(x-mean)² ──
@inline function reduce_block(::Type{_Comp1{VarAcc{S}}}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    A = inputs[1]
    s = zero(S)
    @inbounds for J in lo:hi
        s += S(A[J])
    end
    n = _blocklen(lo, hi)
    mean = s / S(n)
    m2 = zero(S)
    @inbounds for J in lo:hi
        d = S(A[J]) - mean
        m2 += d * d
    end
    return _wrap1(VarAcc(n, mean, m2))
end

# ── Cov: two-pass over the field pair — means, then Σ(x-x̄)(y-ȳ) ──
@inline function reduce_block(::Type{_Comp1{CovAcc{S}}}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    X, Y = inputs[1], inputs[2]
    sx = zero(S); sy = zero(S)
    @inbounds for J in lo:hi
        sx += S(X[J]); sy += S(Y[J])
    end
    n = _blocklen(lo, hi)
    mx = sx / S(n); my = sy / S(n)
    c = zero(S)
    @inbounds for J in lo:hi
        c += (S(X[J]) - mx) * (S(Y[J]) - my)
    end
    return _wrap1(CovAcc(n, mx, my, c))
end

# ── Min / Max ──
@inline function reduce_block(::Type{_Comp1{MinAcc{S}}}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    A = inputs[1]
    m = typemax(S)
    @inbounds for J in lo:hi
        x = S(A[J]); m = ifelse(x < m, x, m)
    end
    return _wrap1(MinAcc(m))
end
@inline function reduce_block(::Type{_Comp1{MaxAcc{S}}}, inputs::Tuple, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    A = inputs[1]
    m = typemin(S)
    @inbounds for J in lo:hi
        x = S(A[J]); m = ifelse(x > m, x, m)
    end
    return _wrap1(MaxAcc(m))
end

# ── Coarsen side (symmetric): combine a block of fine accumulators by vectorized reductions over
#    their fields — the stable parallel-combine (weighted-mean + δ² correction), not per-cell Chan. ──

@inline function coarsen_block(fine::AbstractArray{_Comp1{CountAcc},N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {N}
    n = 0
    @inbounds for J in lo:hi
        n += members(fine[J])[1].n
    end
    return _wrap1(CountAcc(n))
end

@inline function coarsen_block(fine::AbstractArray{_Comp1{SumAcc{S}},N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    s = zero(S)
    @inbounds for J in lo:hi
        s += members(fine[J])[1].s
    end
    return _wrap1(SumAcc(s))
end

@inline function coarsen_block(fine::AbstractArray{_Comp1{MeanAcc{S}},N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    n = 0; ws = zero(S)                         # ws = Σ nᵢ·meanᵢ (weighted, stable for unequal counts)
    @inbounds for J in lo:hi
        a = members(fine[J])[1]; n += a.n; ws += S(a.n) * a.mean
    end
    return _wrap1(MeanAcc(n, ws / S(n)))
end

@inline function coarsen_block(fine::AbstractArray{_Comp1{VarAcc{S}},N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    n = 0; ws = zero(S)
    @inbounds for J in lo:hi
        a = members(fine[J])[1]; n += a.n; ws += S(a.n) * a.mean
    end
    mean = ws / S(n)
    m2 = zero(S)
    @inbounds for J in lo:hi
        a = members(fine[J])[1]; d = a.mean - mean; m2 += a.M2 + S(a.n) * d * d
    end
    return _wrap1(VarAcc(n, mean, m2))
end

@inline function coarsen_block(fine::AbstractArray{_Comp1{CovAcc{S}},N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    n = 0; wsx = zero(S); wsy = zero(S)
    @inbounds for J in lo:hi
        a = members(fine[J])[1]; n += a.n; wsx += S(a.n) * a.meanx; wsy += S(a.n) * a.meany
    end
    mx = wsx / S(n); my = wsy / S(n)
    c = zero(S)
    @inbounds for J in lo:hi
        a = members(fine[J])[1]; c += a.C + S(a.n) * (a.meanx - mx) * (a.meany - my)
    end
    return _wrap1(CovAcc(n, mx, my, c))
end

@inline function coarsen_block(fine::AbstractArray{_Comp1{MinAcc{S}},N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    m = typemax(S)
    @inbounds for J in lo:hi
        x = members(fine[J])[1].m; m = ifelse(x < m, x, m)
    end
    return _wrap1(MinAcc(m))
end
@inline function coarsen_block(fine::AbstractArray{_Comp1{MaxAcc{S}},N}, lo::CartesianIndex{N}, hi::CartesianIndex{N}) where {S,N}
    m = typemin(S)
    @inbounds for J in lo:hi
        x = members(fine[J])[1].m; m = ifelse(x > m, x, m)
    end
    return _wrap1(MaxAcc(m))
end
