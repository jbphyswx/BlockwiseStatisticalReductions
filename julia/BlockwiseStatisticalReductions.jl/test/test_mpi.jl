using MPI: MPI

# The MPI backend is SPMD, so it is tested by launching `mpi_parity.jl` under `mpiexec` with 2 ranks
# (MPI.jl bundles its own `mpiexec`) and asserting the run succeeds — the sub-process checks that the
# MPI result is bit-identical to serial and exits nonzero otherwise.
@testset "mpi (2 ranks via mpiexec)" begin
    script = joinpath(@__DIR__, "mpi_parity.jl")
    proj = Base.active_project()
    cmd = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()[1]) --project=$proj $script`
    @test success(run(ignorestatus(cmd)))
end
