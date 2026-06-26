# ─────────────────────────────────────────────────────────────────────────────
# Executing a plan: fill the buffers, then finalize requested outputs
# ─────────────────────────────────────────────────────────────────────────────
#
# `run!` walks the plan's topologically-ordered steps, computing each node either from the input
# (base pass) or by coarsening its parent's already-filled buffer. The loop is type-stable (steps
# are concrete `ReductionStep{N}`, buffers concrete `Array{Acc,N}`) and allocation-free at steady
# state. The `backend` is positional so distribution wrappers can add their own `run!` method
# (partition → local `run!` per worker → merge); local backends (serial/threaded/GPU) flow through
# to the kernels.

"""
    run!(buf::TowerBuffers, plan::ReductionPlan, inputs, backend=SerialBackend()) -> buf

Execute `plan` into the preallocated `buf`. `inputs` is one array (arity-1 statistics) or a tuple
of arrays (arity-2, e.g. covariance); a single array is accepted directly. Reusing `buf` across
calls is allocation-free.
"""
function run!(buf::TowerBuffers{Acc,N}, plan::ReductionPlan{N}, inputs::Tuple,
              backend::AbstractExecutionBackend = SerialBackend()) where {Acc,N}
    @boundscheck length(buf.arrays) == length(plan.steps) ||
        throw(DimensionMismatch("buffers ($(length(buf.arrays))) do not match plan ($(length(plan.steps)) steps)"))
    for i in eachindex(plan.steps)
        s = plan.steps[i]
        out = buf.arrays[i]
        if s.source == 0
            blockreduce!(out, inputs, s.window, backend)
        else
            coarsen!(out, buf.arrays[s.source], s.window, backend)
        end
    end
    return buf
end
run!(buf::TowerBuffers, plan::ReductionPlan, data::AbstractArray,
     backend::AbstractExecutionBackend = SerialBackend()) = run!(buf, plan, (data,), backend)

"""
    execute(plan, ::Type{Acc}, inputs, backend=SerialBackend()) -> TowerBuffers

Allocate buffers for `plan` and accumulator type `Acc`, run it, and return the filled buffers.
Allocating convenience; for repeated execution preallocate once with [`allocate_tower`](@ref) and
call [`run!`](@ref).
"""
function execute(plan::ReductionPlan{N}, ::Type{Acc}, inputs,
                 backend::AbstractExecutionBackend = SerialBackend()) where {Acc,N}
    buf = allocate_tower(plan, Acc)
    return run!(buf, plan, inputs isa Tuple ? inputs : (inputs,), backend)
end

# ── Finalization ──────────────────────────────────────────────────────────────

"""
    materialize(accs::AbstractArray, stat, ::Type{Tout}) -> Array{Tout}

Finalize an array of accumulators into the statistic `stat`, elementwise, in output eltype `Tout`.
"""
materialize(accs::AbstractArray, stat::AbstractStatistic, ::Type{Tout}) where {Tout} =
    map(a -> result_value(stat, a, Tout), accs)

"""
    materialize(accs::AbstractArray{<:CompositeAccumulator}, member::Integer, stat, ::Type{Tout})

Finalize statistic `stat` from member `member` of an array of composite accumulators.
"""
materialize(accs::AbstractArray{<:CompositeAccumulator}, member::Integer,
            stat::AbstractStatistic, ::Type{Tout}) where {Tout} =
    map(a -> result_value(stat, members(a)[member], Tout), accs)

# Worker-side helper for the Distributed extension: compute one base node on a data slab. Defined in
# core so it is available on every process that has loaded the package (no @everywhere needed).
_compute_base_slab(::Type{Acc}, inputs::Tuple, window::NTuple) where {Acc} = blockreduce(Acc, inputs, window)
