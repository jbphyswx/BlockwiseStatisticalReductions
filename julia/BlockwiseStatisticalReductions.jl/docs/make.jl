using Documenter: Documenter
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions

Documenter.makedocs(;
    modules = [BlockwiseStatisticalReductions],
    sitename = "BlockwiseStatisticalReductions.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/jbphyswx/BlockwiseStatisticalReductions.jl",
    ),
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Concepts" => [
            "Overview & Theory" => "concepts/overview.md",
            "The Tower & Lattice" => "concepts/tower.md",
        ],
        "API Reference" => "api/public.md",
    ],
)
