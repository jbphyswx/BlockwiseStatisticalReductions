using BlockwiseStatisticalReductions
using Test, Random, Statistics

const BSR = BlockwiseStatisticalReductions

# Brute-force non-overlapping block reduction of `f` over `data` (reference for correctness tests).
function brute(f, data::AbstractArray{T,N}, window::NTuple{N,Int}) where {T,N}
    osh = ntuple(i -> size(data, i) ÷ window[i], N)
    out = Array{Float64,N}(undef, osh)
    for I in CartesianIndices(out)
        lo = ntuple(i -> (I[i] - 1) * window[i] + 1, N)
        hi = ntuple(i -> I[i] * window[i], N)
        out[I] = f(vec(collect(@view data[ntuple(i -> lo[i]:hi[i], N)...])))
    end
    return out
end

# Brute-force block covariance of `x,y`.
function brute_cov(x::AbstractArray{T,N}, y::AbstractArray{T,N}, window::NTuple{N,Int}; corrected=true) where {T,N}
    osh = ntuple(i -> size(x, i) ÷ window[i], N)
    out = Array{Float64,N}(undef, osh)
    for I in CartesianIndices(out)
        lo = ntuple(i -> (I[i] - 1) * window[i] + 1, N)
        hi = ntuple(i -> I[i] * window[i], N)
        rng = ntuple(i -> lo[i]:hi[i], N)
        out[I] = cov(vec(collect(@view x[rng...])), vec(collect(@view y[rng...])); corrected = corrected)
    end
    return out
end

# Finalize an array of accumulators into a statistic (single-accumulator arrays).
vals(stat, accs, ::Type{Tout}) where {Tout} = map(a -> result_value(stat, a, Tout), accs)

# Brute-force overlapping (sliding) window reduction of `f` over `data`.
function brute_sliding(f, data::AbstractArray{T,N}, w::NTuple{N,Int}, s::NTuple{N,Int}, o::NTuple{N,Int}) where {T,N}
    md = ntuple(d -> size(data, d) - w[d] + 1, N)
    osz = ntuple(d -> (md[d] - o[d]) ÷ s[d] + 1, N)
    out = Array{Float64,N}(undef, osz)
    for I in CartesianIndices(out)
        p = ntuple(d -> o[d] + (I[d] - 1) * s[d], N)
        rng = ntuple(d -> p[d]:p[d] + w[d] - 1, N)
        out[I] = f(vec(collect(@view data[rng...])))
    end
    return out
end
