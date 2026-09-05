"""
    check_monoid(::Type{A}; samples, rtol = 1e-8) -> Bool

Verify identity, commutativity, associativity, fold/k-ary agreement and, when claimed, the group inverse
for accumulator type `A` on `samples` (raw observations; tuples for arity > 1). Throws on the first
violated law.
"""
function check_monoid(::Type{A}; samples = randn(64), rtol::Real = 1e-8) where {A<:AbstractAccumulator}
    lifted = [lift(A, _observation(s)) for s in samples]
    length(lifted) >= 3 || throw(ArgumentError("need at least 3 samples"))
    e = neutral(A)
    scale = max(1.0, maximum(_magnitude, lifted))
    ≈(p, q) = _approx(p, q, rtol * scale, rtol)
    for a in lifted
        ≈(merge(e, a), a) || error("check_monoid($A): left identity failed")
        ≈(merge(a, e), a) || error("check_monoid($A): right identity failed")
    end
    for a in lifted, b in lifted
        ≈(merge(a, b), merge(b, a)) || error("check_monoid($A): commutativity failed")
    end
    a, b, c = lifted[1], lifted[2], lifted[3]
    ≈(merge(merge(a, b), c), merge(a, merge(b, c))) || error("check_monoid($A): associativity failed")
    full = foldl(merge, lifted)
    half = length(lifted) ÷ 2
    ≈(full, merge(foldl(merge, lifted[1:half]), foldl(merge, lifted[half+1:end]))) ||
        error("check_monoid($A): fold grouping changed the result")
    ≈(full, combine(A, lifted)) || error("check_monoid($A): combine disagrees with the fold")
    if is_invertible(A)
        ≈(unmerge(merge(a, b), b), a) || error("check_monoid($A): unmerge(merge(a, b), b) != a")
    end
    return true
end

_observation(s::Tuple) = s
_observation(s) = (s,)

_magnitude(x::AbstractFloat) = abs(Float64(x))
_magnitude(x::Integer) = 0.0
_magnitude(x::Tuple) = isempty(x) ? 0.0 : maximum(_magnitude, x)
_magnitude(a::AbstractAccumulator) = maximum(f -> _magnitude(getfield(a, f)), fieldnames(typeof(a)); init = 0.0)

_approx(x::Integer, y::Integer, atol, rtol) = x == y
_approx(x::Tuple, y::Tuple, atol, rtol) = all(_approx(p, q, atol, rtol) for (p, q) in zip(x, y))
_approx(x::AbstractAccumulator, y::AbstractAccumulator, atol, rtol) =
    all(_approx(getfield(x, f), getfield(y, f), atol, rtol) for f in fieldnames(typeof(x)))
_approx(x, y, atol, rtol) = isapprox(x, y; atol = atol, rtol = rtol)
