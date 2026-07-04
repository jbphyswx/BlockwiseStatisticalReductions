# SPMD parity check, launched under `mpiexec` by test_mpi.jl. Every rank computes with MPIBackend and
# with SerialBackend on identical input and checks bit-for-bit agreement; exits nonzero on mismatch.
using MPI, BlockwiseStatisticalReductions, Random
const BSR = BlockwiseStatisticalReductions

MPI.Init()
Random.seed!(1234)                          # identical input on every rank

A = randn(120, 96)
rm = reduce_stats(A, [4, 8]; stats = (Mean(), Var()), backend = MPIBackend())
rs = reduce_stats(A, [4, 8]; stats = (Mean(), Var()), backend = SerialBackend())
ok = Set(factors(rm)) == Set(factors(rs)) &&
     all(f -> rm(f, Mean()) == rs(f, Mean()) && rm(f, Var()) == rs(f, Var()), factors(rs))

X = randn(64, 64); Y = randn(64, 64)
cm = reduce_stats(X, Y, [8]; stats = (Cov(),), backend = MPIBackend())
cs = reduce_stats(X, Y, [8]; stats = (Cov(),), backend = SerialBackend())
ok &= cm((8, 8), Cov()) == cs((8, 8), Cov())

MPI.Finalize()
ok || error("MPI backend parity with serial failed")
