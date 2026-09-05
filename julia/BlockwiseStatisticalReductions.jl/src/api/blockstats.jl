# The user-facing entry points. `prepare` does everything that depends only on the request (resolve the
# windows, assemble the composite, build the plan, allocate the workspace and the result arrays);
# `blockstats!` then feeds new data through it without allocating. `blockstats` is the one-shot.

"""
    Prepared

A resolved request: plan, workspace, routing and result arrays, reusable across inputs of the same
shape and element type. Build with [`prepare`](@ref), run with [`blockstats!`](@ref).
"""
# One prepared finalize: a statistic, the member storage it reads and the array it writes. Built once,
# like the workspace's kernel steps, so running a prepared request constructs nothing.
struct FinalizeStep{D<:AbstractArray,S<:AccumulatorArray,T<:AbstractStatistic,K}
    dst::D
    src::S
    tag::T
    binds::NTuple{K,Int}
end
# Not `@inline`: the steps are held as `Any`, so this call must stay a function barrier that specializes
# on the concrete step type. Inlining it would leave the field loads dynamic and box every one.
run_step!(s::FinalizeStep, shifts::Tuple, backend) =
    (finalize!(s.dst, s.src, s.tag, map(i -> shifts[i], s.binds), backend); nothing)

function _run_finalizers!(steps::Vector{Any}, shifts::Tuple, backend)
    for s in steps
        run_step!(s, shifts, backend)
    end
    return nothing
end

struct Prepared{N,C<:AbstractAccumulator,ST<:Tuple,RT<:Tuple,BK,WS,R<:ScaleResults{N},SH}
    plan::Plan{N}
    workspace::WS
    stats::ST
    routing::RT
    backend::BK
    result::R
    finalizers::Vector{Any}
    shift::SH
    fieldnames::NTuple{<:Any,Symbol}
    input_shape::NTuple{N,Int}
    in_bytes::Int
end

_fieldnames(::AbstractArray) = (:data,)
_fieldnames(f::NamedTuple) = keys(f)
_fieldnames(f::Tuple) = ntuple(i -> Symbol(:f, i), length(f))
_fieldtuple(f::AbstractArray) = (f,)
_fieldtuple(f::NamedTuple) = values(f)
_fieldtuple(f::Tuple) = f

function _check_fields(fields::Tuple)
    isempty(fields) && throw(ArgumentError("at least one input field is required"))
    for f in fields
        size(f) == size(fields[1]) || throw(DimensionMismatch("input fields have different shapes: $(size(f)) and $(size(fields[1]))"))
    end
    return nothing
end

"""
    prepare(fields, scales; stats, edge = Truncate(), backend = CB.AutoBackend(),
            acc_eltype = nothing, out_eltype = nothing, memory_limit = typemax(Int)) -> Prepared

Resolve a request against inputs shaped like `fields` (one array, or a `NamedTuple`/`Tuple` of
same-shaped arrays) and allocate everything it needs. `scales` is anything [`resolve`](@ref) accepts.
`stats` is a tuple of statistic tags; a tag bound to a field name or position selects which input it
reads. Accumulation defaults to [`accumulation_eltype`](@ref) of the promoted input eltype and results
to each tag's [`result_eltype`](@ref). `dimnames` names the axes, so a scale specification may address
them by name; `spacing` gives their coordinates, so sizes may be given as a [`Length`](@ref) and results
report the physical extent of every output cell.
"""
function prepare(fields, scales; stats::Tuple, edge::EdgePolicy = Truncate(),
                 backend::CB.AbstractExecutionBackend = CB.AutoBackend(), acc_eltype = nothing,
                 out_eltype = nothing, shift = :auto, dimnames = nothing, spacing = nothing,
                 memory_limit::Int = typemax(Int))
    fs = _fieldtuple(fields)
    _check_fields(fs)
    names = _fieldnames(fields)
    shape = size(fs[1])
    N = length(shape)
    Tin = promote_type(map(eltype, fs)...)
    bk = resolve_backend(backend, fs)
    _check_dimnames(dimnames, N)
    targets = resolve(scales, shape; edge = edge, dimnames = dimnames, spacing = spacing)
    # Shifting keeps the offset out of every difference the moment kernels take, so the accumulation
    # eltype only has to resolve the spread of the data rather than its magnitude. That is what lets a
    # narrow input accumulate in its own eltype; `:auto` pays for it only where it buys something, which
    # is exactly when the accumulation would otherwise have to widen.
    Cprobe, _, _, _ = assemble(stats, names, Tin, accumulation_eltype(Tin))
    shifting = _shifting(shift, Tin, Cprobe, acc_eltype)
    Tacc = acc_eltype !== nothing ? acc_eltype : (shifting ? Tin : accumulation_eltype(Tin))
    C, routing, statnames, outs = assemble(stats, names, Tin, Tacc)
    router = _bind_router(fs, C)
    in_bytes = _in_bytes(C, fs)
    p = plan(shape, targets; backend = bk, in_bytes, acc_bytes = sizeof(C), memory_limit)
    ws = allocate(p, C, fs[1])
    eltypes = out_eltype === nothing ? outs : ntuple(_ -> out_eltype, length(stats))
    result = _allocate_results(p, targets, fs[1], statnames, eltypes, dimnames, spacing)
    sh = _shift_state(shift, shifting, Tin, length(_bound_fields(C)))
    finalizers = _finalize_steps(p, ws, result, stats, routing, C, names)
    return Prepared{N,C,typeof(stats),typeof(routing),typeof(bk),typeof(ws),typeof(result),typeof(sh)}(
        p, ws, stats, routing, bk, result, finalizers, sh, names, shape, in_bytes)
end

# Bytes of raw input one observation reads: only the fields the composite actually binds.
_in_bytes(::Type{C}, fs::Tuple) where {C} = sum(i -> sizeof(eltype(fs[i])), _bound_fields(C); init = 0)
_bound_fields(::Type{Composite{M,B}}) where {M,B} = sort!(unique!(collect(Iterators.flatten(B))))
# Kernels read fields positionally in binding order, so hand them the fields the composite binds. The
# selection is fixed by the composite type, so it is generated rather than rebuilt on every call.
@generated function _bind_router(fs::Tuple, ::Type{C}) where {C}
    picks = [:(fs[$i]) for i in _bound_fields(C)]
    return quote
        Base.@_inline_meta
        ($(picks...),)
    end
end

function _allocate_results(p::Plan{N}, targets, proto::AbstractArray, statnames::Tuple, eltypes::Tuple,
                           dimnames, spacing) where {N}
    results = [NamedTuple{statnames}(ntuple(k -> similar(proto, eltypes[k], shape(w)), length(statnames))) for w in targets]
    return ScaleResults(p.input_shape, collect(targets), results, p, statnames, dimnames, spacing)
end

_check_dimnames(::Nothing, N::Int) = nothing
function _check_dimnames(names::Tuple, N::Int)
    length(names) == N || throw(ArgumentError("$(length(names)) axis name(s) for $N axes"))
    allunique(names) || throw(ArgumentError("axis names must be unique, got $names"))
    return nothing
end

"""
    blockstats!(p::Prepared, fields) -> ScaleResults

Run the prepared request on new input. Allocation-free at steady state. The returned result aliases
`p`'s arrays and is overwritten by the next call — copy what you need before reusing `p`.
"""
function blockstats!(p::Prepared, fields)
    fs = _fieldtuple(fields)
    _check_fields(fs)
    size(fs[1]) == p.input_shape ||
        throw(DimensionMismatch("input shape $(size(fs[1])) does not match the prepared shape $(p.input_shape)"))
    length(fs) == length(p.fieldnames) ||
        throw(ArgumentError("$(length(fs)) field(s) given for a request prepared with $(length(p.fieldnames))"))
    bound = _bind_router(fs, _composite(p))
    sh = _update_shift!(p.shift, bound)
    run!(p.workspace, p.plan, _apply_shift(bound, sh), p.backend)
    _run_finalizers!(p.finalizers, sh, p.backend)
    return p.result
end
_composite(::Prepared{N,C}) where {N,C} = C

# `:auto` shifts exactly when it buys accuracy: a floating input whose accumulation would otherwise have
# to widen. An input already accumulating at its own width is well conditioned without it, and the
# per-element subtraction is not free, so `:auto` leaves it alone; `true` asks for it regardless (worth
# it when the data carries an offset far larger than its spread).
_shifting(shift::Symbol, ::Type{Tin}, ::Type{C}, acc_eltype) where {Tin,C} =
    shift === :auto ? (Tin <: AbstractFloat && shiftable(C) && acc_eltype === nothing &&
                       accumulation_eltype(Tin) !== Tin) :
    throw(ArgumentError("shift must be :auto, true, false or a value, got :$shift"))
_shifting(shift::Bool, ::Type{Tin}, ::Type{C}, acc_eltype) where {Tin,C} =
    shift && (Tin <: AbstractFloat && shiftable(C))
_shifting(shift::Union{Real,Tuple}, ::Type{Tin}, ::Type{C}, acc_eltype) where {Tin,C} = true

# A tuple of `NoShift` marks a request that does not shift: the fields pass through untouched.
_update_shift!(r::Base.RefValue{<:Tuple{NoShift,Vararg{NoShift}}}, fields::Tuple) = r[]
function _update_shift!(r::Base.RefValue{S}, fields::Tuple) where {S<:Tuple{Vararg{Real}}}
    r[] = ntuple(i -> field_shift(fields[i], eltype(S)), Val(fieldcount(S)))
    return r[]
end
@inline _apply_shift(fields::Tuple, ::Tuple{NoShift,Vararg{NoShift}}) = fields
@inline _apply_shift(fields::Tuple, sh::Tuple{Vararg{Real}}) = map(Shifted, fields, sh)

# One step per (requested window, statistic), resolving the composite member, the output array and the
# positions of the tag's fields within the request's shift tuple.
function _finalize_steps(p::Plan, ws::Workspace, result::ScaleResults, stats::Tuple, routing::Tuple,
                         ::Type{C}, names::Tuple) where {C}
    bound = _bound_fields(C)
    steps = Any[]
    for (j, k) in enumerate(p.outputs)
        accs = node_storage(ws, k)
        dst = result.results[j]
        map(stats, routing, values(dst)) do tag, r, out
            fields = resolve_bindings(tag, names)
            binds = ntuple(i -> findfirst(==(fields[i]), bound), length(fields))
            push!(steps, FinalizeStep(out, member_array(accs, r), tag, binds))
        end
    end
    return steps
end

# Shifts live in a mutable holder so a prepared request re-centres on every new input.
_shift_state(shift::Union{Bool,Symbol}, shifting::Bool, ::Type{Tin}, nbound::Int) where {Tin} =
    shifting ? Ref(ntuple(_ -> zero(Tin), nbound)) : Ref(ntuple(_ -> NoShift(), nbound))
_shift_state(shift::Union{Real,Tuple}, shifting::Bool, ::Type{Tin}, nbound::Int) where {Tin} =
    Ref(shift isa Tuple ? map(Tin, shift) : ntuple(_ -> Tin(shift), nbound))

"Largest sample `field_shift` averages over."
const SHIFT_SAMPLES = 4096

"""
    field_shift(field, ::Type{T}) -> T

A value near the middle of `field`, in `T`: the mean of an evenly strided sample of at most
[`SHIFT_SAMPLES`](@ref) elements. Only the magnitude matters — the shift cancels exactly in the centred
moments — so a sample suffices; a non-finite result falls back to zero.
"""
function field_shift(field::AbstractArray, ::Type{T}) where {T}
    n = length(field)
    n == 0 && return zero(T)
    step = max(1, n ÷ SHIFT_SAMPLES)
    total = 0.0
    taken = 0
    @inbounds for i in 1:step:n
        total += Float64(field[i])
        taken += 1
    end
    m = total / taken
    return isfinite(m) ? T(m) : zero(T)
end

"""
    blockstats(fields, scales; stats, kwargs...) -> ScaleResults

Statistics of `fields` over every window `scales` asks for, computed in one shared plan. `fields` is one
array or a `NamedTuple`/`Tuple` of same-shaped arrays; statistics bound to different fields (a covariance
of two of them, say) are still produced in a single pass. Keywords are [`prepare`](@ref)'s.

```julia
r = blockstats(x, [2, 4, 8]; stats = (Mean(), Var()))
r[(4, 4)].mean

r = blockstats((u = u, w = w), [8]; stats = (Var(:u), Var(:w), Cov(:u, :w)))
r[(8, 8)].cov_u_w
```

For many inputs of the same shape, [`prepare`](@ref) once and call [`blockstats!`](@ref).
"""
function blockstats(fields, scales; kwargs...)
    p = prepare(fields, scales; kwargs...)
    return blockstats!(p, fields)
end

"Plan report for a prepared request or a result (see [`explain`](@ref) on a plan)."
explain(io::IO, p::Prepared; kw...) = explain(io, p.plan; in_bytes = p.in_bytes, acc_bytes = sizeof(_composite(p)), kw...)

"The per-field shifts a prepared request last accumulated against ([`NoShift`](@ref) when it does not)."
shifts(p::Prepared) = p.shift[]

"`true` when a prepared request accumulates shifted observations."
is_shifting(p::Prepared) = !(shifts(p) isa Tuple{Vararg{NoShift}})
explain(io::IO, r::ScaleResults; kw...) = explain(io, r.plan; kw...)
explain(x::Union{Prepared,ScaleResults}; kw...) = explain(stdout, x; kw...)
