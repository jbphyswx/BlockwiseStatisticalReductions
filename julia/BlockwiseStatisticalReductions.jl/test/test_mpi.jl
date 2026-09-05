using Test: Test
using MPI: MPI, mpiexec

# The MPI backend can only be exercised across processes. Two ranks cover a halo that reaches the next
# slab; three cover one that reaches past it, into a rank that is not a neighbour.
Test.@testset "mpi" begin
    script = joinpath(@__DIR__, "mpi_parity.jl")
    project = Base.active_project()
    for n in (2, 3)
        Test.@testset "$n ranks" begin
            Test.@test success(mpiexec(cmd -> run(`$cmd -n $n $(Base.julia_cmd()) --project=$project $script`)))
        end
    end
end
