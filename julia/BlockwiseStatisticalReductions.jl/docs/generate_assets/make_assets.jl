# Figures for the README and the documentation. Run with
#
#     julia --project=docs/generate_assets docs/generate_assets/make_assets.jl
#
# and commit what lands in docs/src/assets/. Kept out of the docs build so a docs run needs no plotting
# stack and no measurement time.

using BlockwiseStatisticalReductions
const BSR = BlockwiseStatisticalReductions
using CairoMakie
using Printf
using Random
using Statistics: Statistics

const OUT = normpath(joinpath(@__DIR__, "..", "src", "assets"))
mkpath(OUT)
save_fig(fig, name) = (save(joinpath(OUT, name), fig); println("wrote ", joinpath(OUT, name)))

best(f, n = 5) = minimum((GC.gc(); @elapsed f()) for _ in 1:n)

# ── 1. What a multi-scale request looks like ────────────────────────────────────
# Coarsening averages the fine structure away, so each panel gets its own symmetric colour range —
# a shared one would leave every panel but the first blank.
function scales_figure()
    Random.seed!(11)
    n = 256
    x = sum(k -> repeat(randn(n ÷ k, n ÷ k), inner = (k, k)), (1, 4, 16, 64)) ./ 2
    r = blockstats(x, [1, 4, 16, 64]; stats = (Mean(),))

    fig = Figure(size = (1040, 320))
    for (j, s) in enumerate(scales(r))
        m = r[s].mean
        lim = maximum(abs, m)
        ax = Axis(fig[1, j]; title = "$(s[1])×$(s[2]) windows  →  $(size(m, 1))×$(size(m, 2))",
                  aspect = 1, xticksvisible = false, yticksvisible = false,
                  xticklabelsvisible = false, yticklabelsvisible = false)
        heatmap!(ax, m; colorrange = (-lim, lim), colormap = :balance)
    end
    Label(fig[0, :], "Tile means of one 256×256 field, all from a single pass over it";
          fontsize = 17, font = :bold)
    rowgap!(fig.layout, 5)
    return fig
end

# ── 2. Cost against the number of scales ────────────────────────────────────────
function cost_figure()
    Random.seed!(12)
    x = randn(2048, 2048)
    roofline = (sum(x); best(() -> sum(x)))
    ks = 1:6
    sizes = [2^k for k in ks]
    shared = Float64[]
    apart = Float64[]
    for k in ks
        want = sizes[1:k]
        p = prepare(x, want; stats = (Mean(), Var()))
        blockstats!(p, x)
        push!(shared, best(() -> blockstats!(p, x)) / roofline)
        ps = [prepare(x, [s]; stats = (Mean(), Var())) for s in want]
        foreach(q -> blockstats!(q, x), ps)
        push!(apart, best(() -> foreach(q -> blockstats!(q, x), ps)) / roofline)
    end

    fig = Figure(size = (620, 420))
    ax = Axis(fig[1, 1]; xlabel = "scales requested (2, 4, 8, …)", ylabel = "time / one streaming sum(x)",
              title = "Mean and variance at many scales, 2048² Float64")
    lines!(ax, collect(ks), apart; label = "one pass each", linewidth = 3)
    scatter!(ax, collect(ks), apart; markersize = 12)
    lines!(ax, collect(ks), shared; label = "one shared plan", linewidth = 3)
    scatter!(ax, collect(ks), shared; markersize = 12)
    hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
    text!(ax, 1.05, 1.05; text = "reading the array once", color = :gray, fontsize = 12)
    axislegend(ax; position = :lt)
    ylims!(ax, 0, nothing)
    return fig
end

# ── 3. Why a narrow accumulation needs the shift ────────────────────────────────
# Both curves accumulate in Float32; the only difference is whether the data is centred first. The
# reference is the exact variance of the *same* Float32 data, so what is plotted is accumulation error
# alone and not the quantization the offset already imposed on the input.
function shift_figure()
    Random.seed!(13)
    n = 512
    spread = 1.0f0
    ratios = Float32[1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    base = randn(Float32, n, n) .* spread

    shifted = Float64[]
    plain = Float64[]
    for rr in ratios
        x = base .+ rr * spread
        truth = blockstats(Float64.(x), [8]; stats = (Var(),))[(8, 8)].var
        err(r) = Statistics.mean(abs.(r[(8, 8)].var .- truth) ./ truth)
        push!(plain, err(blockstats(x, [8]; stats = (Var(),), shift = false, acc_eltype = Float32)))
        push!(shifted, err(blockstats(x, [8]; stats = (Var(),), shift = true)))
    end

    fig = Figure(size = (620, 420))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "offset / spread of the data",
              ylabel = "relative error of the tile variance",
              title = "Float32 accumulation of 8×8 tile variances")
    floor_ = 1e-9
    lines!(ax, Float64.(ratios), max.(plain, floor_); label = "accumulated as given", linewidth = 3)
    scatter!(ax, Float64.(ratios), max.(plain, floor_); markersize = 12)
    lines!(ax, Float64.(ratios), max.(shifted, floor_); label = "shifted before accumulating", linewidth = 3)
    scatter!(ax, Float64.(ratios), max.(shifted, floor_); markersize = 12)
    axislegend(ax; position = :lt)
    return fig
end

# ── 4. The computation graph the planner builds ─────────────────────────────────
# Levels come from the longest path to the input, which is what makes the two panels comparable: the
# naive one is one level deep six times over, the planned one is a single chain.
function node_levels(p)
    lev = fill(-1, length(p.nodes))
    while true
        changed = false
        for k in eachindex(p.nodes)
            ps = BSR.parents(p.how[k])
            l = isempty(ps) ? 0 : (all(q -> lev[q] >= 0, ps) ? 1 + maximum(q -> lev[q], ps) : -1)
            if l >= 0 && l != lev[k]
                lev[k] = l
                changed = true
            end
        end
        changed || return lev
    end
end

const REQUESTED = RGBf(0.20, 0.45, 0.72)
const INTERMEDIATE = RGBf(0.80, 0.55, 0.15)

# `plans` is one plan, or several independent ones drawn together; either way a node sits at x = its
# level and the nodes sharing a level are spread symmetrically about y = 0, so the input sits at (0, 0).
function draw_dag!(ax, plans, title)
    keys_at = Dict{Int,Vector{Tuple{Int,Int}}}()
    levels = Dict{Tuple{Int,Int},Int}()
    for (pi, p) in enumerate(plans)
        lev = node_levels(p)
        for k in eachindex(p.nodes)
            levels[(pi, k)] = lev[k]
            push!(get!(keys_at, lev[k], Tuple{Int,Int}[]), (pi, k))
        end
    end
    src = Point2f(0, 0)
    at = Dict{Tuple{Int,Int},Point2f}()
    # Place level by level and sort each level by where its parents already sit, so edges do not cross
    # and the labels on them stay legible.
    for l in sort(collect(keys(keys_at)))
        ks = sort(keys_at[l]; by = key -> begin
            ps = BSR.parents(plans[key[1]].how[key[2]])
            isempty(ps) ? (0.0, key[2]) : (-Statistics.mean(at[(key[1], q)][2] for q in ps), key[2])
        end)
        for (j, key) in enumerate(ks)
            at[key] = Point2f(l + 1, (length(ks) + 1) / 2 - j)
        end
    end

    for (pi, p) in enumerate(plans), k in eachindex(p.nodes)
        ps = BSR.parents(p.how[k])
        from = isempty(ps) ? [src] : [at[(pi, q)] for q in ps]
        for f in from
            lines!(ax, [f, at[(pi, k)]]; color = (:gray30, 0.6), linewidth = 2)
            lab = _edge_label(p.how[k])
            isempty(lab) || text!(ax, (f + at[(pi, k)]) / 2; text = lab, align = (:center, :bottom),
                                  fontsize = 11, color = :gray25, offset = (0, 3))
        end
    end
    for (pi, p) in enumerate(plans), k in eachindex(p.nodes)
        n = p.nodes[k]
        scatter!(ax, [at[(pi, k)]]; markersize = 32, color = n.requested ? REQUESTED : INTERMEDIATE,
                 strokecolor = :white, strokewidth = 1.5)
        text!(ax, at[(pi, k)]; text = string(n.window[1].size), align = (:center, :center),
              color = :white, fontsize = 12, font = :bold)
    end
    scatter!(ax, [src]; markersize = 36, color = :gray25, marker = :rect)
    text!(ax, src; text = "x", align = (:center, :center), color = :white, fontsize = 13, font = :bold)

    ax.title = title
    hidedecorations!(ax)
    hidespines!(ax)
    return maximum(values(levels)) + 1
end

# The edge label is what the kernel actually does to get from parent to child.
_edge_label(d::BSR.Coarsen) = all(==(first(d.k)), d.k) ? "×$(first(d.k))" : "×$(d.k)"
_edge_label(d::BSR.Compose) = "⊕"
_edge_label(d) = ""

function plan_figure()
    shape = (480, 480)
    sizes = [6, 8, 12, 20, 24]
    # The real accumulator of this request, so the cost model sees what it would actually store.
    stats = (mean_u = BSR.Mean(:u), var_u = BSR.Var(:u), flux = BSR.Cov(:u, :w))
    C, _, _, _ = BSR.assemble(stats, (:u, :w), Float64, Float64)
    kw = (; in_bytes = 16, acc_bytes = sizeof(C))

    targets = BSR.resolve(sizes, shape)
    apart = [BSR.plan(shape, [w]; kw...) for w in targets]
    shared = BSR.plan(shape, targets; kw...)

    fig = Figure(size = (1000, 360))
    ax1 = Axis(fig[1, 1]; limits = (-0.6, 4.4, -3.2, 3.2))
    ax2 = Axis(fig[1, 2]; limits = (-0.6, 4.4, -3.2, 3.2))
    draw_dag!(ax1, apart, "one pass each: $(length(sizes)) passes over the fields")
    draw_dag!(ax2, [shared], "one shared plan: $(length(BSR.base_nodes(shared))) passes, then merges")
    Label(fig[0, :], "Mean and variance of u, and its covariance with w, at tile sizes $(join(sizes, ", "))";
          fontsize = 16, font = :bold)
    Label(fig[2, :], rich(
              rich("■", color = :gray25), " the fields      ",
              rich("●", color = REQUESTED), " a size you asked for      ",
              rich("●", color = INTERMEDIATE), " one the planner added to share the work      ",
              rich("×k", color = :gray25), " merges k×k of them");
          fontsize = 12, color = :gray30)
    rowgap!(fig.layout, 4)
    return fig
end

# ── 5. The window families one request can mix ──────────────────────────────────
# Every window is outlined; two are filled so the placement — and, where there is any, the overlap —
# is visible rather than washing the whole panel in one colour.
function windows_figure()
    extent = 24
    cases = (("tiles, 6×6", BSR.ScaleSet(BSR.Sizes([6])), (1, 2)),
             ("every 3 cells, 6×6", BSR.ScaleSet(BSR.Sizes([6]); placement = BSR.Stride(3)), (2, 3)),
             ("at chosen anchors, 8×8", BSR.ScaleSet(BSR.Sizes([8]); placement = BSR.Anchors([0, 5, 16])), (1, 2)),
             ("anisotropic, 12×4", BSR.ScaleSet((BSR.Sizes([12]), BSR.Sizes([4])); combine = BSR.Product()), (1, 2)))
    fills = (RGBf(0.20, 0.45, 0.72), RGBf(0.80, 0.35, 0.15))

    fig = Figure(size = (1040, 320))
    for (j, (label, spec, pick)) in enumerate(cases)
        w = only(BSR.resolve(spec, (extent, extent)))
        ox = collect(BSR.origins(w[1]))
        oy = collect(BSR.origins(w[2]))
        ax = Axis(fig[1, j]; title = label, aspect = 1, limits = (-0.5, extent + 0.5, -0.5, extent + 0.5))
        for i in 0:extent
            lines!(ax, [i, i], [0, extent]; color = (:gray, 0.22), linewidth = 0.5)
            lines!(ax, [0, extent], [i, i]; color = (:gray, 0.22), linewidth = 0.5)
        end
        for a in ox, b in oy
            poly!(ax, Rect2f(a, b, w[1].size, w[2].size);
                  color = (:white, 0.0), strokecolor = (:gray30, 0.45), strokewidth = 1)
        end
        for (c, i) in zip(fills, pick)
            i <= min(length(ox), length(oy)) || continue
            poly!(ax, Rect2f(ox[i], oy[i], w[1].size, w[2].size);
                  color = (c, 0.35), strokecolor = c, strokewidth = 2.5)
        end
        text!(ax, 0.5, extent - 0.2; text = "$(length(ox) * length(oy)) windows", align = (:left, :top),
              fontsize = 12, color = :gray30)
        hidedecorations!(ax)
    end
    Label(fig[0, :], "Tiles, overlapping windows and hand-picked anchors are all the same object";
          fontsize = 17, font = :bold)
    rowgap!(fig.layout, 5)
    return fig
end

save_fig(plan_figure(), "plan.png")
save_fig(windows_figure(), "windows.png")
save_fig(scales_figure(), "scales.png")
save_fig(cost_figure(), "cost.png")
save_fig(shift_figure(), "shift.png")
