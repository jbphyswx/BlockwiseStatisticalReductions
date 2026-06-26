# ─────────────────────────────────────────────────────────────────────────────
# Preallocated accumulator buffers for a plan
# ─────────────────────────────────────────────────────────────────────────────
#
# One `Array{Acc}` per plan step, sized from the step's output shape. Allocated once and reused
# across `run!` calls (same plan + accumulator, new data) so steady-state execution allocates
# nothing — including for variance/covariance, whose M2/C arrays are part of the same buffer set
# (the central bug of the previous implementation, where the variance path silently fell back to
# allocation). A future optimization can share physical arrays across non-overlapping lifetimes;
# v1 keeps one buffer per node (correct, still zero-alloc across calls).

"""
    TowerBuffers{Acc,N}

Preallocated accumulator arrays for every step of a [`ReductionPlan`], with accumulator element
type `Acc`. Build with [`allocate_tower`](@ref); fill with [`run!`](@ref).
"""
struct TowerBuffers{Acc,N}
    arrays::Vector{Array{Acc,N}}
end

"""
    allocate_tower(plan, ::Type{Acc}) -> TowerBuffers{Acc,N}

Allocate one accumulator array per step of `plan` (sized to each step's output shape). Reuse the
returned buffers across many [`run!`](@ref) calls for allocation-free repeated execution.
"""
function allocate_tower(plan::ReductionPlan{N}, ::Type{Acc}) where {Acc,N}
    arrays = Vector{Array{Acc,N}}(undef, length(plan.steps))
    for (i, s) in enumerate(plan.steps)
        arrays[i] = Array{Acc,N}(undef, s.shape)
    end
    return TowerBuffers{Acc,N}(arrays)
end

"The accumulator array produced by step `i` (after [`run!`](@ref))."
@inline step_result(buf::TowerBuffers, i::Integer) = buf.arrays[i]
