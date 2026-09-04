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
    pages = ["Home" => "index.md"],
)
