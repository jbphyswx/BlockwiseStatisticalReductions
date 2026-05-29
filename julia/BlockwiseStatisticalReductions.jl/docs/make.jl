using Documenter: Documenter
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions

Documenter.makedocs(;
    modules=[BlockwiseStatisticalReductions],
    sitename="BlockwiseStatisticalReductions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/jbphyswx/BlockwiseStatisticalReductions.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Concepts" => [
            "Overview" => "concepts/overview.md",
            "Window Configurations" => "concepts/windows.md",
            "DAG-Based Planning" => "concepts/plans.md",
            "Multi-Resolution Towers" => "concepts/tower.md",
            "Numerical Stability" => "concepts/numerical_stability.md",
        ],
        "How-To Guides" => [
            "Basic Blockwise Reductions" => "howto/basic_reductions.md",
            "Multi-Resolution Analysis" => "howto/multiresolution.md",
            "Per-Dimension Scaling" => "howto/per_dimension.md",
            "Zero-Allocation Execution" => "howto/zero_alloc.md",
            "Product Coarsening" => "howto/product_coarsening.md",
            "Hybrid Mode" => "howto/hybrid_mode.md",
        ],
        "API Reference" => [
            "Public API" => "api/public.md",
            "Plan Building" => "api/planning.md",
            "Execution" => "api/execution.md",
            "Accumulators" => "api/accumulators.md",
            "Kernels" => "api/kernels.md",
        ],
        "Roadmap" => "roadmap.md",
    ],
)
