# ─────────────────────────────────────────────────────────────────────────────
# Display methods
# ─────────────────────────────────────────────────────────────────────────────
#
# Two-method pattern per type: a compact `show(io, x)` (used inline / in containers) and a rich
# `show(io, ::MIME"text/plain", x)` (used when a value is displayed at the REPL).

# ── Statistic tags (compact; the parametric ones otherwise print as `Var{true}()`) ──
Base.show(io::IO, ::Var{C}) where {C} = print(io, "Var(corrected=", C, ")")
Base.show(io::IO, ::Std{C}) where {C} = print(io, "Std(corrected=", C, ")")
Base.show(io::IO, ::Cov{C}) where {C} = print(io, "Cov(corrected=", C, ")")
Base.show(io::IO, ::Moments{K}) where {K} = print(io, "Moments(", K, ")")

# ── Scale specs ───────────────────────────────────────────────────────────────
Base.show(io::IO, t::Tower) =
    print(io, "Tower(base_factor=", t.base_factor, ", steps=", t.steps, ", maxfactor=", t.maxfactor, ")")
function Base.show(io::IO, s::Sliding)
    print(io, "Sliding(", s.window)
    s.stride === nothing || print(io, "; stride=", s.stride)
    s.origin == 1 || print(io, ", origin=", s.origin)
    print(io, ")")
end

# ── TowerBuffers ──────────────────────────────────────────────────────────────
Base.show(io::IO, b::TowerBuffers{Acc,N}) where {Acc,N} =
    print(io, "TowerBuffers{", Acc, ",", N, "}(", length(b.arrays), " buffers)")

# ── ReductionPlan: compact + rich ─────────────────────────────────────────────
function Base.show(io::IO, plan::ReductionPlan{N}) where {N}
    print(io, "ReductionPlan{", N, "}(", plan.input_shape, ": ", length(plan.steps), " nodes, ",
          length(plan.output_steps), " outputs, ", n_base_passes(plan), " base pass(es))")
end

function Base.show(io::IO, ::MIME"text/plain", plan::ReductionPlan{N}) where {N}
    nw, nn = plan_work(plan), naive_work(plan)
    println(io, "ReductionPlan over input ", plan.input_shape, ":")
    println(io, "  ", length(plan.steps), " nodes, ", length(plan.output_steps), " outputs, ",
            n_base_passes(plan), " base pass(es)")
    println(io, "  work ", nw, " vs naive ", nn, "  (", round(nw / max(nn, 1); digits = 3), "× the data)")
    nshow = min(length(plan.steps), 24)
    for i in 1:nshow
        s = plan.steps[i]
        src = s.source == 0 ? "input" : string("step ", s.source)
        println(io, s.is_output ? "  * [" : "    [", i, "] factor ", s.factor, " → shape ", s.shape,
                "  ⇐ ", src, " ×", s.window)
    end
    nshow < length(plan.steps) && println(io, "    … ", length(plan.steps) - nshow, " more")
    print(io, "  (* = requested output)")
end

# ── MultiResResult: compact + rich ────────────────────────────────────────────
_nt_names(::Type{NT}) where {NT} = NT isa DataType ? fieldnames(NT) : ()

function Base.show(io::IO, r::MultiResResult{N,NT}) where {N,NT}
    print(io, "MultiResResult{", N, "}(", r.input_shape, ": ", length(r.order),
          " resolution(s), stats ", _nt_names(NT), ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::MultiResResult{N,NT}) where {N,NT}
    println(io, "MultiResResult over input ", r.input_shape, ":")
    println(io, "  statistics: ", join(_nt_names(NT), ", "))
    println(io, "  ", length(r.order), " resolution(s) (factor → shape):")
    for f in r.order
        println(io, "    ", f, " → ", r.shapes[f])
    end
    print(io, "  index by factor, e.g. r[", first(r.order), "].", first(_nt_names(NT)))
end

# ── summary: short one-line headers (for explicit `summary` / nested / error contexts) ──
Base.summary(io::IO, r::MultiResResult{N,NT}) where {N,NT} =
    print(io, length(r.order), "-resolution MultiResResult{", N, "} with stats ", _nt_names(NT))
Base.summary(io::IO, plan::ReductionPlan{N}) where {N} =
    print(io, length(plan.steps), "-node ReductionPlan{", N, "} over ", plan.input_shape)
Base.summary(io::IO, b::TowerBuffers{Acc,N}) where {Acc,N} =
    print(io, length(b.arrays), "-buffer TowerBuffers{", Acc, ",", N, "}")
