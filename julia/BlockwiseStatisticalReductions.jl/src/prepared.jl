# ─────────────────────────────────────────────────────────────────────────────
# Prepared reductions: reuse plan + buffers + result arrays across calls
# ─────────────────────────────────────────────────────────────────────────────
#
# `reduce_stats(data, scales; stats)` is the one-shot convenience: it (re)builds the plan, assembles
# the composite accumulator type, allocates the tower buffers, runs, and allocates fresh result
# arrays — every call. For a hot loop over many same-shape inputs (e.g. a data-generation pipeline),
# that per-call setup + buffer allocation dominates (measured ~15 MB single-scale, ~290 MB for a
# 6-scale mean+var on 4096², mostly the accumulator tower). `prepare` does that work ONCE and returns
# a handle; `reduce_stats!(handle, data)` then reuses the plan, the buffers, AND the result arrays, so
# steady-state execution is allocation-free.
#
# Note: the returned `MultiResResult` aliases the handle's preallocated result arrays; they are
# overwritten by the next `reduce_stats!` call. Consume (or copy) results before reusing the handle —
# the intended pattern for a streaming pipeline.

"""
    PreparedReduction

A reusable plan + buffers + result arrays for repeated reductions of same-shape inputs. Build with
[`prepare`](@ref); execute with [`reduce_stats!`](@ref). Allocation-free at steady state.
"""
struct PreparedReduction{N,Cacc,ST,RT,BK,MR}
    plan::ReductionPlan{N}
    buffers::TowerBuffers{Cacc,N}
    stats::ST
    routing::RT
    backend::BK
    result::MR      # a MultiResResult whose result arrays are updated in place; returned each call
end

"""
    prepare(input_shape, scales; stats, Tin=Float64, backend=SerialBackend()) -> PreparedReduction

Precompute everything reusable for reducing inputs of shape `input_shape` (element type `Tin`) at
`scales` with `stats`: the plan, the accumulator tower buffers, and the result arrays. Feed inputs
with [`reduce_stats!`](@ref). `scales` accepts the same specs as [`reduce_stats`](@ref) (`Tower`,
`Ladder`, factor vector/tuple/int). Use for repeated same-shape reductions (pipelines).
"""
function prepare(input_shape::NTuple{N,Int}, scales; stats::Tuple,
                 Tin::Type = Float64, backend::AbstractExecutionBackend = SerialBackend()) where {N}
    plan = _plan_for(input_shape, scales)
    C, routing, names, touts = _assemble(stats, Tin)
    buffers = allocate_tower(plan, C)
    K = length(stats)
    order = NTuple{N,Int}[plan.steps[i].factor for i in plan.output_steps]
    shapes = NTuple{N,Int}[plan.steps[i].shape for i in plan.output_steps]
    results = [NamedTuple{names}(ntuple(k -> Array{touts[k],N}(undef, sh), K)) for sh in shapes]
    result = MultiResResult(input_shape, order, shapes, results)
    return PreparedReduction(plan, buffers, stats, routing, resolve_backend(backend), result)
end

# In-place finalize: write statistic `stat` from composite member `Val{M}` into `out` (no allocation).
function materialize!(out::AbstractArray, accs::AbstractArray{<:CompositeAccumulator}, ::Val{M},
                      stat::AbstractStatistic) where {M}
    Tout = eltype(out)
    @inbounds for i in eachindex(out, accs)
        out[i] = result_value(stat, getfield(members(accs[i]), M), Tout)
    end
    return out
end

@inline function _finalize_into!(res::NamedTuple, accs::AbstractArray, stats::Tuple, routing::Tuple)
    map((stat, r, arr) -> materialize!(arr, accs, r, stat), stats, routing, values(res))
    return nothing
end

"""
    reduce_stats!(p::PreparedReduction, data) -> MultiResResult
    reduce_stats!(p::PreparedReduction, x, y) -> MultiResResult

Execute the prepared reduction `p` on new input(s), reusing its plan, buffers, and result arrays
(allocation-free at steady state). The returned result aliases `p`'s result arrays — consume or copy
before the next call.
"""
function reduce_stats!(p::PreparedReduction, inputs::Tuple)
    _check_arity(p.stats, length(inputs))
    run!(p.buffers, p.plan, inputs, p.backend)
    res = p.result
    @inbounds for (j, i) in enumerate(p.plan.output_steps)
        _finalize_into!(res.results[j], step_result(p.buffers, i), p.stats, p.routing)
    end
    return res   # its result arrays were updated in place (alias — consume/copy before next call)
end
reduce_stats!(p::PreparedReduction, data::AbstractArray) = reduce_stats!(p, (data,))
reduce_stats!(p::PreparedReduction, x::AbstractArray, y::AbstractArray) = reduce_stats!(p, (x, y))
