"Total cost of the plan's computed nodes."
function total_cost(p::Plan{N}, in_bytes::Int, acc_bytes::Int) where {N}
    c = zero(Cost)
    for k in p.order
        c += cost(p.how[k], p.nodes[k], p.nodes, in_bytes, acc_bytes)
    end
    return c
end

"Passes over the raw input the plan makes (windows of every base node, in elements ÷ input elements)."
input_passes(p::Plan) = sum(k -> cells(p.nodes[k]) * volume(p.nodes[k].window), base_nodes(p); init = 0) / prod(p.input_shape)

_describe(::Base_) = "base pass over the fields"
_describe(d::Coarsen) = "coarsen node $(d.parent) by $(d.k)"
_describe(d::Compose) = "compose nodes $(d.a) ⊕ $(d.b) along axis $(d.axis)"
_describe(d::Restride) = "view of node $(d.parent)"
_describe(d::Scan) = "scan node $(d.parent) along axis $(d.axis), window $(d.size)"
_describe_axis(aw::AxisWindow) =
    aw.pos isa Progression ? "$(aw.size)@$(aw.pos.offset):$(aw.pos.stride)" : "$(aw.size)@[$(nwindows(aw)) origins]"

"""
    explain([io,] plan; in_bytes = 8, acc_bytes = 24)

Print every node (sizes, positions, shape, derivation) and the plan's totals: passes over the input,
bytes moved, merges, peak storage.
"""
explain(p::Plan; kw...) = explain(stdout, p; kw...)
function explain(io::IO, p::Plan{N}; in_bytes::Int = 8, acc_bytes::Int = 24) where {N}
    println(io, "Plan over input ", p.input_shape, ": ", length(p.nodes), " nodes, ", length(p.outputs), " requested, ",
            length(base_nodes(p)), " base pass(es)")
    for k in eachindex(p.nodes)
        n = p.nodes[k]
        mark = n.requested ? "*" : " "
        println(io, "  ", mark, lpad(k, 3), "  ", join(map(_describe_axis, n.window), " × "), "  → ", n.shape, "  ⇐ ", _describe(p.how[k]))
    end
    c = total_cost(p, in_bytes, acc_bytes)
    println(io, "  input passes ", round(input_passes(p); digits = 3), ", bytes ", round(c.bytes / 1e6; digits = 2), " MB, merges ",
            round(c.merges / 1e6; digits = 2), " M (", round(c.serial_merges / 1e6; digits = 2), " M serial), peak storage ",
            round(peak_bytes(p, acc_bytes) / 1e6; digits = 2), " MB")
    return nothing
end

"Graphviz DOT text of the plan's DAG."
function dot(p::Plan{N}; name::AbstractString = "plan") where {N}
    io = IOBuffer()
    println(io, "digraph \"", name, "\" {")
    println(io, "  rankdir=LR; node [fontname=\"monospace\", shape=box];")
    println(io, "  input [label=\"input ", p.input_shape, "\", shape=ellipse];")
    for k in eachindex(p.nodes)
        style = p.nodes[k].requested ? ", style=bold" : (is_view(p.how[k]) ? ", style=dashed" : "")
        println(io, "  n", k, " [label=\"", join(map(_describe_axis, p.nodes[k].window), " × "), " → ", p.nodes[k].shape, "\"", style, "];")
        ps = parents(p.how[k])
        isempty(ps) && println(io, "  input -> n", k, ";")
        for q in ps
            println(io, "  n", q, " -> n", k, ";")
        end
    end
    println(io, "}")
    return String(take!(io))
end
