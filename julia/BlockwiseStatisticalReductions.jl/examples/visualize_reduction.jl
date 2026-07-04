# Visualize a multi-scale reduction: the DAG (as Graphviz DOT) and the set of output resolutions.
#
#   julia --project=examples examples/visualize_reduction.jl
#
# Writes `reduction_dag.dot` (render with `dot -Tsvg reduction_dag.dot -o dag.svg`) and, if CairoMakie
# is available, `reduction_resolutions.png` showing each output scale's grid size.

using BlockwiseStatisticalReductions
const BSR = BlockwiseStatisticalReductions

# A dense, gap-filled ladder of scales over a 2-D field (the full 2,3-smooth reduction tree).
X = (512, 512)
ladder = Ladder(seeds = [1], steps = [2, 3], maxfactor = 64)
plan = BSR._plan_for(X, ladder)

println("Reduction over input $X with $(length(plan.output_steps)) output scales:")
show(stdout, MIME"text/plain"(), plan)
println()

# 1) DAG visualization → Graphviz DOT (dependency-free). Shared finer intermediates that several
#    coarser scales reuse are visible as nodes with multiple outgoing edges.
dotfile = joinpath(@__DIR__, "reduction_dag.dot")
open(dotfile, "w") do io
    print(io, plan_dot(plan; name = "smooth_tree"))
end
println("\nWrote DAG to $dotfile  (render: dot -Tsvg $dotfile -o dag.svg)")

# 2) Output-resolution overview.
r = reduce_stats(randn(X...), ladder; stats = (Mean(), Var()))
println("\nOutput resolutions (factor → shape):")
for (f, sh) in zip(factors(r), shapes(r))
    println("  ", f, " → ", sh)
end

# 3) Optional CairoMakie figure of the output grid sizes across scales.
try
    @eval using CairoMakie
    facs = [f[1] for f in factors(r)]
    ncell = [prod(sh) for sh in shapes(r)]
    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "block factor", ylabel = "# output cells",
                         yscale = log10, xscale = log10, title = "Output resolutions of the reduction tree")
    CairoMakie.scatterlines!(ax, facs, ncell)
    pngfile = joinpath(@__DIR__, "reduction_resolutions.png")
    CairoMakie.save(pngfile, fig)
    println("\nWrote resolution figure to $pngfile")
catch err
    println("\n(CairoMakie not available — skipped the figure: $(sprint(showerror, err) |> x -> first(x, 60)))")
end
