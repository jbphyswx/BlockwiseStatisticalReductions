using Aqua
using ExplicitImports

@testset "Aqua quality" begin
    Aqua.test_all(BlockwiseStatisticalReductions)
end

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(BlockwiseStatisticalReductions) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(BlockwiseStatisticalReductions) === nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(BlockwiseStatisticalReductions) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(BlockwiseStatisticalReductions) === nothing
end
