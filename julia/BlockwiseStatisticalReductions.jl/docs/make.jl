using Documenter: Documenter
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions

# The migration note lives at the repository root, where someone upgrading will look for it.
cp(joinpath(@__DIR__, "..", "MIGRATION.md"), joinpath(@__DIR__, "src", "migration.md"); force = true)

Documenter.makedocs(;
    modules = [BlockwiseStatisticalReductions],
    sitename = "BlockwiseStatisticalReductions.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jbphyswx.github.io/BlockwiseStatisticalReductions/stable",
        assets = String[],
    ),
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "Concepts" => [
            "Statistics" => "concepts/statistics.md",
            "Windows" => "concepts/windows.md",
            "Scale specifications" => "concepts/scales.md",
            "The planner" => "concepts/planner.md",
            "Numerics" => "concepts/numerics.md",
            "Weights and gaps" => "concepts/weights.md",
            "Labelled axes" => "concepts/labeled.md",
            "Backends" => "concepts/backends.md",
            "Partitioned tensors" => "concepts/distributed.md",
        ],
        "Performance" => "performance.md",
        "Migrating" => "migration.md",
        "API reference" => "api.md",
    ],
)

get(ENV, "CI", "false") == "true" && Documenter.deploydocs(;
    repo = "github.com/jbphyswx/BlockwiseStatisticalReductions",
    devbranch = "main",
    push_preview = true,
)
