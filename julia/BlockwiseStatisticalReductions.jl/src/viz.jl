# ─────────────────────────────────────────────────────────────────────────────
# Visualization: render the reduction DAG (dependency-free Graphviz DOT)
# ─────────────────────────────────────────────────────────────────────────────
#
# `plan_dot` emits a Graphviz DOT description of a `ReductionPlan`: the input, one node per reduction
# factor (labeled `factor → shape`), and edges from each node's optimal parent labeled with the merge
# window. Base-pass nodes (read directly from the input) and requested-output nodes are marked. It is
# pure string generation — no plotting dependency — so it is always available and testable; render
# with `dot -Tpng plan.dot` (Graphviz) or paste into any DOT viewer. This visualizes both the
# "tree of reductions" structure (how coarse scales reuse finer ones) and the set of output
# resolutions in one picture.

"""
    plan_dot(plan::ReductionPlan; name="reduction") -> String

A Graphviz DOT string for the reduction DAG of `plan`. Nodes are factors (label `factor → shape`);
edges go from each node's optimal parent, labeled with the merge window (`×window`); an ellipse marks
the input, boxes the reduction nodes, and bold boxes the requested outputs. Render with, e.g.,
`open("plan.dot","w") do io; print(io, plan_dot(plan)); end` then `dot -Tsvg plan.dot -o plan.svg`.
"""
function plan_dot(plan::ReductionPlan{N}; name::AbstractString = "reduction") where {N}
    io = IOBuffer()
    println(io, "digraph \"", name, "\" {")
    println(io, "  rankdir=LR;")
    println(io, "  node [fontname=\"monospace\", shape=box];")
    println(io, "  input [label=\"input ", plan.input_shape, "\", shape=ellipse, style=filled, fillcolor=\"#dddddd\"];")
    for (i, s) in enumerate(plan.steps)
        style = s.is_output ? ", style=bold" : ""
        println(io, "  n", i, " [label=\"", s.factor, " → ", s.shape, "\"", style, "];")
        src = s.source == 0 ? "input" : string("n", s.source)
        println(io, "  ", src, " -> n", i, " [label=\"×", s.window, "\"];")
    end
    println(io, "}")
    return String(take!(io))
end
