# Frontline showcase: a structured multi-scale ("turbulent") field, its block-mean pyramid, and how
# its statistics vary with scale — the core thing BSR is for, in one figure.
#
#   julia --project=examples examples/multiscale_showcase.jl
#
# Saves docs/src/assets/multiscale_showcase.png.

using BlockwiseStatisticalReductions
using FFTW: FFTW
using CairoMakie: CairoMakie as MK
using Random: Random
using Statistics: Statistics

# A real 2-D field with a power-law spatial spectrum P(k) ∝ k^slope — structure at every scale,
# the canonical "turbulent field" look (not white noise).
function turbulent_field(n::Int; slope::Float64 = -3.0, seed::Int = 1)
    rng = Random.MersenneTwister(seed)
    white = randn(rng, n, n)
    F = FFTW.fft(white)
    k = FFTW.fftfreq(n) .* n
    @inbounds for j in 1:n, i in 1:n
        kk = sqrt(k[i]^2 + k[j]^2)
        F[i, j] *= kk == 0 ? 0.0 : kk^(slope / 2)        # shape the amplitude (power ∝ k^slope)
    end
    f = real.(FFTW.ifft(F))
    return (f .- Statistics.mean(f)) ./ Statistics.std(f) # standardize for display
end

function main()
    n = 512
    field = turbulent_field(n; slope = -3.0)

    # Block-mean pyramid at a few scales, plus a fine scan of scales for the variance curve.
    pyramid_scales = [4, 16, 64]
    scan = [2, 4, 8, 16, 32, 64, 128]
    r = reduce_stats(field, vcat(pyramid_scales, scan) |> unique; stats = (Mean(), Var()))

    fig = MK.Figure(; size = (1280, 760))
    MK.Label(fig[0, 1:5], "BlockwiseStatisticalReductions.jl — multi-scale statistics in one pass";
             fontsize = 22, font = :bold)

    crange = (-3, 3)
    local hm
    ax1 = MK.Axis(fig[1, 1]; title = "field  ($(n)×$(n))", aspect = 1)
    hm = MK.heatmap!(ax1, field; colorrange = crange, colormap = :balance)
    for (j, s) in enumerate(pyramid_scales)
        m = r[(s, s)].mean
        ax = MK.Axis(fig[1, j + 1]; title = "block-mean  $(s)×  ($(size(m, 1))×$(size(m, 2)))", aspect = 1)
        MK.heatmap!(ax, m; colorrange = crange, colormap = :balance)
    end
    MK.Colorbar(fig[1, 5], hm; label = "standardized value")

    # How much variance lives at each scale: spatial-mean of the per-block variance vs block size.
    bsz = sort(scan)
    meanvar = [Statistics.mean(r[(s, s)].var) for s in bsz]
    ax2 = MK.Axis(fig[2, 1:5]; xlabel = "block size (coarsening factor)", ylabel = "mean within-block variance",
                  xscale = log2, yscale = log10, title = "within-block variance grows with scale (power-law field)")
    MK.scatterlines!(ax2, Float64.(bsz), meanvar; markersize = 12)

    out = joinpath(@__DIR__, "..", "docs", "src", "assets", "multiscale_showcase.png")
    mkpath(dirname(out))
    MK.save(out, fig)
    println("wrote ", normpath(out))
    return out
end

main()
