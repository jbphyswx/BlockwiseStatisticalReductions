using Test: Test
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using ComputationalBackends: ComputationalBackends as CB
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions as BSR

# Aqua's persistent-task check builds a throwaway package that depends on this one and precompiles it,
# which needs this package to be instantiable from its Project.toml alone; it is not, because
# ComputationalBackends comes from a git `[sources]` entry. The property it checks is verified below.
Test.@testset "Aqua quality" begin
    Aqua.test_all(BSR; persistent_tasks = false)
end

Test.@testset "no background work at load time" begin
    src = joinpath(pkgdir(BSR), "src")
    files = [joinpath(root, f) for (root, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl")]
    Test.@test !isempty(files)
    for f in files
        text = read(f, String)
        Test.@test !occursin("@async", text)
        Test.@test !occursin("Threads.@spawn", text)
        Test.@test !occursin("__init__", text)
    end
end

function abstract_fields!(out, ::Type{T}, path, seen, depth = 0) where {T}
    (T in seen || depth > 6 || !isstructtype(T)) && return out
    push!(seen, T)
    for k in 1:fieldcount(T)
        FT, p = fieldtype(T, k), string(path, ".", fieldname(T, k))
        isconcretetype(FT) || push!(out, string(p, "::", FT))
        abstract_fields!(out, FT, p, seen, depth + 1)
    end
    return out
end

# Heterogeneous collections are read behind a function barrier (`run_step!`, the workspace's setup-only
# buffers); a field whose declared type is abstract makes every load of it dynamic instead.
Test.@testset "no abstract field types in a running request" begin
    u = randn(24, 16); v = randn(24, 16)
    p = BSR.prepare((u = u, v = v), [(8, 8)];
                    stats = (m = BSR.Mean(:u), c = BSR.Cov(:u, :v)), backend = CB.SerialBackend())
    r = BSR.blockstats!(p, (u = u, v = v))
    for x in (p, p.workspace, p.plan, r, first(p.finalizers))
        Test.@test abstract_fields!(String[], typeof(x), string(nameof(typeof(x))), Set{Type}()) == String[]
    end
end

Test.@testset "ExplicitImports" begin
    Test.@test ExplicitImports.check_no_implicit_imports(BSR) === nothing
    Test.@test ExplicitImports.check_no_stale_explicit_imports(BSR) === nothing
    Test.@test ExplicitImports.check_all_explicit_imports_via_owners(BSR) === nothing
    # `CommonDataModel` is reached through NCDatasets on purpose: NCDatasets implements that interface and
    # re-exports the module, so the extension can use the data model without a second package having to be
    # loaded. Every other name must still come from its owner.
    Test.@test ExplicitImports.check_all_qualified_accesses_via_owners(BSR;
                                                                      ignore = (:CommonDataModel,)) === nothing
end
