using KernelAbstractions: CPU

# The GPU backend (KernelAbstractions) is exercised on the KA `CPU()` backend here — it runs the same
# `@kernel` that transpiles to CUDA/ROCm/… on real hardware, so this validates the kernel logic and
# backend parity without a GPU. Each output cell is reduced by the same per-cell `reduce_block`/
# `coarsen_block` as the serial path, so the result is bit-for-bit identical (cells are independent).
@testset "gpu (KernelAbstractions CPU backend)" begin
    gpu = GPUBackend(CPU())
    Random.seed!(9)

    @testset "base + coarsen parity vs serial (arity 1)" begin
        data = randn(120, 96)
        for stats in ((Mean(),), (Sum(),), (Count(),), (Var(),), (Min(), Max()), (Mean(), Var()))
            rs = reduce_stats(data, [4, 8, 16]; stats = stats, backend = SerialBackend())
            rg = reduce_stats(data, [4, 8, 16]; stats = stats, backend = gpu)
            @test Set(factors(rs)) == Set(factors(rg))
            for f in factors(rs), s in stats
                @test rs(f, s) == rg(f, s)      # bit-identical: same per-cell reduce_block
            end
        end
    end

    @testset "covariance parity vs serial (arity 2)" begin
        x = randn(64, 64); y = randn(64, 64)
        cs = reduce_stats(x, y, [4, 8]; stats = (Cov(),), backend = SerialBackend())
        cg = reduce_stats(x, y, [4, 8]; stats = (Cov(),), backend = gpu)
        for f in factors(cs)
            @test cs(f, Cov()) == cg(f, Cov())
        end
    end

    @testset "3-D + correctness vs brute" begin
        data = randn(24, 20, 16)
        rg = reduce_stats(data, (4, 4, 2); stats = (Mean(), Var()), backend = gpu)
        @test rg((4, 4, 2), Mean()) ≈ brute(mean, data, (4, 4, 2))
        @test rg((4, 4, 2), Var()) ≈ brute(x -> var(x; corrected = true), data, (4, 4, 2))
    end
end
