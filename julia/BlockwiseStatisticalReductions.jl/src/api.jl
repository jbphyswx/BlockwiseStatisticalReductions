# ─────────────────────────────────────────────────────────────────────────────
# Public API: reduce_stats
# ─────────────────────────────────────────────────────────────────────────────
#
# Turn a user request — some statistic tags, a field (or field pair), and a set of scales — into a
# planned, executed, finalized result keyed by resolution. The statistics requested over one field
# binding are packed into a single `CompositeAccumulator` (subsumption drops redundant members,
# e.g. `Mean` when `Var` is present), so all of them are produced in one pass over the data and one
# multi-scale tower. The heavy execution runs behind a function barrier on the concrete composite
# type, keeping the hot loop type-stable and allocation-free even though the type is assembled from
# the (runtime) request.

"""
    stat_name(stat::AbstractStatistic) -> Symbol

The short key under which a statistic appears in a [`MultiResResult`](@ref)'s NamedTuples
(`:mean`, `:var`, `:cov`, …). Define a method for a custom statistic to give it a result key.
"""
stat_name(::Count) = :count
stat_name(::Sum) = :sum
stat_name(::Mean) = :mean
stat_name(::Var) = :var
stat_name(::Std) = :std
stat_name(::Cov) = :cov
stat_name(::Min) = :min
stat_name(::Max) = :max
stat_name(::Moments) = :moments

# ── Result container ──────────────────────────────────────────────────────────

"""
    MultiResResult{N,NT}

Results of [`reduce_stats`](@ref): for each requested output factor (block size), a `NamedTuple`
of result arrays — one entry per statistic. Index by factor to get the NamedTuple, call with a
factor and a statistic tag to get a single array, or iterate [`factors`](@ref).

```julia
r[(4,4)]            # NamedTuple (mean = …, var = …)
r((4,4), Mean())    # the mean array at factor (4,4)
factors(r)          # the output factors, finest first
```
"""
struct MultiResResult{N,NT}
    input_shape::NTuple{N,Int}
    results::Dict{NTuple{N,Int},NT}
    order::Vector{NTuple{N,Int}}              # output factors, in finest-first order
    shapes::Dict{NTuple{N,Int},NTuple{N,Int}}
end

Base.getindex(r::MultiResResult{N}, factor::NTuple{N,Int}) where {N} = r.results[factor]
(r::MultiResResult)(factor::NTuple, stat::AbstractStatistic) = getproperty(r.results[factor], stat_name(stat))
"Output factors (block sizes) present in the result, finest first."
factors(r::MultiResResult) = r.order
"Output shapes present in the result, finest first."
shapes(r::MultiResResult) = [r.shapes[f] for f in r.order]
Base.keys(r::MultiResResult) = r.order
Base.haskey(r::MultiResResult{N}, f::NTuple{N,Int}) where {N} = haskey(r.results, f)
Base.length(r::MultiResResult) = length(r.order)

# ── Scale specifications ──────────────────────────────────────────────────────

"""
    Tower(; base_factor=2, steps=[2], maxfactor=nothing)

A multi-scale tower specification (per the divisor lattice). `base_factor` is the finest block,
`steps` the allowed per-level multipliers, `maxfactor` the coarsest block (defaults to the full
input extent per dimension). Each may be an `Int`/`Vector` (applied to every dimension) or an
`NTuple`/tuple-of-vectors (per dimension). Use factor `1` in a dimension to leave it unreduced.
"""
struct Tower{B,S,M}
    base_factor::B
    steps::S
    maxfactor::M
end
Tower(; base_factor = 2, steps = [2], maxfactor = nothing) = Tower(base_factor, steps, maxfactor)

# Normalize a per-dimension spec to NTuple{N}; scalars broadcast, tuples pass through.
_perdim(x::Integer, ::Val{N}) where {N} = ntuple(_ -> Int(x), Val(N))
_perdim(x::NTuple{N,Integer}, ::Val{N}) where {N} = map(Int, x)
_perdim(x::AbstractVector{<:Integer}, ::Val{N}) where {N} =
    (length(x) == N || throw(ArgumentError("per-dimension spec length $(length(x)) ≠ ndims $N")); ntuple(i -> Int(x[i]), Val(N)))
_perdim_steps(x::AbstractVector{<:Integer}, ::Val{N}) where {N} = ntuple(_ -> collect(Int, x), Val(N))
_perdim_steps(x::Tuple, ::Val{N}) where {N} =
    (length(x) == N || throw(ArgumentError("per-dimension steps length $(length(x)) ≠ ndims $N")); ntuple(i -> collect(Int, x[i]), Val(N)))

function _resolve_tower(t::Tower, X::NTuple{N,Int}) where {N}
    base = _perdim(t.base_factor, Val(N))
    steps = _perdim_steps(t.steps, Val(N))
    maxf = t.maxfactor === nothing ? X : _perdim(t.maxfactor, Val(N))
    return base, steps, maxf
end

"""
    Sliding(window; stride=window, origin=1)

An overlapping-window specification. `window` is the window size, `stride` the step between window
placements (defaults to `window`, i.e. non-overlapping/blockwise), and `origin` the first window
position. Each may be an `Int` (every dimension) or an `NTuple` (per dimension); use window `1` in
a dimension to leave it unreduced. Results are keyed by `window`.
"""
struct Sliding{W,S,O}
    window::W
    stride::S
    origin::O
end
Sliding(window; stride = nothing, origin = 1) = Sliding(window, stride, origin)

# ── Composite accumulator + routing assembly (build-time; behind a barrier for execution) ──

# Build the composite accumulator type and the per-statistic (member, Tout) routing for one field
# binding of input eltype `Tin`. Runtime build-time work (small); execution is behind a barrier.
function _assemble(stats::Tuple, ::Type{Tin}) where {Tin}
    acc_types = [accumulator_type(s, Tin) for s in stats]
    members_kept = minimal_accumulator_set(acc_types)
    C = CompositeAccumulator{Tuple{members_kept...}}
    routing = ntuple(k -> member_for(stats[k], members_kept, Tin), length(stats))
    any(==(0), routing) && error("internal: a statistic could not be routed to a composite member")
    names = map(stat_name, stats)
    touts = map(s -> default_output_eltype(s, Tin), stats)
    return C, routing, names, touts
end

# Finalize one output node's accumulators into a NamedTuple of result arrays.
@inline function _finalize_node(accs::AbstractArray, stats::Tuple, routing::Tuple, names::Tuple, touts::Tuple)
    vals = map((s, m, T) -> materialize(accs, m, s, T), stats, routing, touts)
    return NamedTuple{names}(vals)
end

# Execute the plan with composite type `C` and finalize all outputs (type-stable barrier).
function _execute_finalize(::Type{C}, plan::ReductionPlan{N}, inputs::Tuple,
                           backend::AbstractExecutionBackend,
                           stats::Tuple, routing::Tuple, names::NMS, touts::Tuple) where {C,N,NMS}
    buf = allocate_tower(plan, C)
    run!(buf, plan, inputs, backend)
    NT = typeof(_finalize_node(step_result(buf, plan.output_steps[1]), stats, routing, names, touts))
    results = Dict{NTuple{N,Int},NT}()
    shapes = Dict{NTuple{N,Int},NTuple{N,Int}}()
    order = NTuple{N,Int}[]
    for i in plan.output_steps
        f = plan.steps[i].factor
        results[f] = _finalize_node(step_result(buf, i), stats, routing, names, touts)
        shapes[f] = plan.steps[i].shape
        push!(order, f)
    end
    return MultiResResult{N,NT}(plan.input_shape, results, order, shapes)
end

# ── reduce_stats ──────────────────────────────────────────────────────────────

"""
    reduce_stats(data, scales; stats, backend=SerialBackend()) -> MultiResResult
    reduce_stats(x, y, scales; stats, backend=SerialBackend()) -> MultiResResult   # arity-2 stats

Compute `stats` over `data` (or the field pair `x, y` for covariance-like statistics) at every
scale in `scales`, reusing intermediate accumulators across scales. `scales` is one of:

  * a [`Tower`](@ref) — a full multi-scale tower;
  * a vector of factor tuples (`[(4,4), (8,8)]`) — explicit per-dimension block sizes;
  * a vector of integers (`[4, 8]`) — isotropic block sizes over all dimensions;
  * a single integer or factor tuple — one scale.

`stats` is a tuple of statistic tags, e.g. `(Mean(), Var())`. Result is keyed by output factor.
"""
function reduce_stats(data::AbstractArray{Tin,N}, scales; stats::Tuple,
                      backend::AbstractExecutionBackend = SerialBackend()) where {Tin,N}
    _check_arity(stats, 1)
    plan = _plan_for(size(data), scales)
    C, routing, names, touts = _assemble(stats, Tin)
    return _execute_finalize(C, plan, (data,), resolve_backend(backend), stats, routing, names, touts)
end

function reduce_stats(x::AbstractArray{Tx,N}, y::AbstractArray{Ty,N}, scales; stats::Tuple,
                      backend::AbstractExecutionBackend = SerialBackend()) where {Tx,Ty,N}
    _check_arity(stats, 2)
    size(x) == size(y) || throw(DimensionMismatch("x and y must have the same shape"))
    plan = _plan_for(size(x), scales)
    Tin = promote_type(Tx, Ty)
    C, routing, names, touts = _assemble(stats, Tin)
    return _execute_finalize(C, plan, (x, y), resolve_backend(backend), stats, routing, names, touts)
end

# ── Sliding (overlapping) windows ─────────────────────────────────────────────

function reduce_stats(data::AbstractArray{Tin,N}, slidings::AbstractVector{<:Sliding}; stats::Tuple,
                      backend::AbstractExecutionBackend = SerialBackend()) where {Tin,N}
    _check_arity(stats, 1)
    C, routing, names, touts = _assemble(stats, Tin)
    return _sliding_finalize(C, (data,), size(data), collect(slidings), stats, routing, names, touts)
end
reduce_stats(data::AbstractArray, s::Sliding; kwargs...) = reduce_stats(data, [s]; kwargs...)

function reduce_stats(x::AbstractArray{Tx,N}, y::AbstractArray{Ty,N}, slidings::AbstractVector{<:Sliding};
                      stats::Tuple, backend::AbstractExecutionBackend = SerialBackend()) where {Tx,Ty,N}
    _check_arity(stats, 2)
    size(x) == size(y) || throw(DimensionMismatch("x and y must have the same shape"))
    C, routing, names, touts = _assemble(stats, promote_type(Tx, Ty))
    return _sliding_finalize(C, (x, y), size(x), collect(slidings), stats, routing, names, touts)
end
reduce_stats(x::AbstractArray, y::AbstractArray, s::Sliding; kwargs...) = reduce_stats(x, y, [s]; kwargs...)

# Resolve one Sliding spec to (window, stride, origin) NTuples.
function _resolve_sliding(s::Sliding, ::Val{N}) where {N}
    w = _perdim(s.window, Val(N))
    st = s.stride === nothing ? w : _perdim(s.stride, Val(N))
    o = _perdim(s.origin, Val(N))
    return w, st, o
end

# Execute every sliding spec and collect results keyed by window (type-stable barrier on `C`).
function _sliding_finalize(::Type{C}, inputs::Tuple, X::NTuple{N,Int}, slidings::Vector,
                           stats::Tuple, routing::Tuple, names::NMS, touts::Tuple) where {C,N,NMS}
    w1, s1, o1 = _resolve_sliding(slidings[1], Val(N))
    accs1 = sliding_reduce(C, inputs, w1, s1; origin = o1)
    nt1 = _finalize_node(accs1, stats, routing, names, touts)
    NT = typeof(nt1)
    results = Dict{NTuple{N,Int},NT}(w1 => nt1)
    shapes = Dict{NTuple{N,Int},NTuple{N,Int}}(w1 => size(accs1))
    order = NTuple{N,Int}[w1]
    for k in 2:length(slidings)
        w, s, o = _resolve_sliding(slidings[k], Val(N))
        accs = sliding_reduce(C, inputs, w, s; origin = o)
        results[w] = _finalize_node(accs, stats, routing, names, touts)
        shapes[w] = size(accs)
        push!(order, w)
    end
    return MultiResResult{N,NT}(X, results, order, shapes)
end

function _check_arity(stats::Tuple, want::Int)
    for s in stats
        stat_arity(s) == want ||
            throw(ArgumentError("statistic $(s) has arity $(stat_arity(s)); this call expects arity-$want statistics " *
                                (want == 1 ? "(use reduce_stats(x, y, …) for covariance)" : "(pass two fields)")))
    end
    return nothing
end

# Build a ReductionPlan from a scale specification.
function _plan_for(X::NTuple{N,Int}, t::Tower) where {N}
    base, steps, maxf = _resolve_tower(t, X)
    return tower_plan(X; base_factor = base, steps = steps, maxfactor = maxf)
end
_plan_for(X::NTuple{N,Int}, factors::AbstractVector{<:Tuple}) where {N} =
    solver_plan(X, [_perdim(f, Val(N)) for f in factors])
_plan_for(X::NTuple{N,Int}, factors::AbstractVector{<:Integer}) where {N} =
    solver_plan(X, [_perdim(f, Val(N)) for f in factors])
_plan_for(X::NTuple{N,Int}, factor::Integer) where {N} = solver_plan(X, [_perdim(factor, Val(N))])
_plan_for(X::NTuple{N,Int}, factor::NTuple{N,Integer}) where {N} = solver_plan(X, [map(Int, factor)])
