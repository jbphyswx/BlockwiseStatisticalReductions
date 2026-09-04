using Test: Test
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions

# Aqua's persistent-task check builds a throwaway package that depends on this one and precompiles it,
# which needs this package to be instantiable from its Project.toml alone; it is not, because
# ComputationalBackends comes from a git `[sources]` entry. The property it checks is verified below.
Test.@testset "Aqua quality" begin
    Aqua.test_all(BlockwiseStatisticalReductions; persistent_tasks = false)
end

Test.@testset "no background work at load time" begin
    src = joinpath(pkgdir(BlockwiseStatisticalReductions), "src")
    files = [joinpath(root, f) for (root, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl")]
    Test.@test !isempty(files)
    for f in files
        text = read(f, String)
        Test.@test !occursin("@async", text)
        Test.@test !occursin("Threads.@spawn", text)
        Test.@test !occursin("__init__", text)
    end
end

Test.@testset "ExplicitImports" begin
    Test.@test ExplicitImports.check_no_implicit_imports(BlockwiseStatisticalReductions) === nothing
    Test.@test ExplicitImports.check_no_stale_explicit_imports(BlockwiseStatisticalReductions) === nothing
    Test.@test ExplicitImports.check_all_explicit_imports_via_owners(BlockwiseStatisticalReductions) === nothing
    Test.@test ExplicitImports.check_all_qualified_accesses_via_owners(BlockwiseStatisticalReductions) === nothing
end
