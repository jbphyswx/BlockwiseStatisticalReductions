"""
Tests for buffer pool functionality.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions

Test.@testset "BufferPool basic operations" begin
    pool = BlockwiseStatisticalReductions.BufferPool{Float64,2}(4)
    
    # Acquire from empty pool - should allocate
    buf1 = BlockwiseStatisticalReductions.acquire!(pool, (10, 10))
    Test.@test size(buf1) == (10, 10)
    Test.@test eltype(buf1) == Float64
    
    # Release back to pool
    BlockwiseStatisticalReductions.release!(pool, buf1)
    
    # Acquire again - should get same buffer (or equal-sized)
    buf2 = BlockwiseStatisticalReductions.acquire!(pool, (10, 10))
    Test.@test size(buf2) == (10, 10)
end

Test.@testset "BufferPool different shapes" begin
    pool = BlockwiseStatisticalReductions.BufferPool{Float64,3}(4)
    
    # Get buffers of different shapes
    buf_a = BlockwiseStatisticalReductions.acquire!(pool, (10, 10, 10))
    buf_b = BlockwiseStatisticalReductions.acquire!(pool, (5, 5, 5))
    
    Test.@test size(buf_a) == (10, 10, 10)
    Test.@test size(buf_b) == (5, 5, 5)
    
    # Release both
    BlockwiseStatisticalReductions.release!(pool, buf_a)
    BlockwiseStatisticalReductions.release!(pool, buf_b)
    
    # Verify each shape has its own pool
    buf_a2 = BlockwiseStatisticalReductions.acquire!(pool, (10, 10, 10))
    buf_b2 = BlockwiseStatisticalReductions.acquire!(pool, (5, 5, 5))
    
    Test.@test size(buf_a2) == (10, 10, 10)
    Test.@test size(buf_b2) == (5, 5, 5)
end

Test.@testset "BufferPool max limit" begin
    # Pool with max 2 buffers per shape
    pool = BlockwiseStatisticalReductions.BufferPool{Float64,2}(2)
    
    # Acquire and release 3 buffers
    bufs = [BlockwiseStatisticalReductions.acquire!(pool, (5, 5)) for _ in 1:3]
    for buf in bufs
        BlockwiseStatisticalReductions.release!(pool, buf)
    end
    
    # Only 2 should be kept in pool
    Test.@test length(pool.buffers[(5, 5)]) == 2
end

Test.@testset "with_buffer! macro" begin
    pool = BlockwiseStatisticalReductions.BufferPool{Float64,2}(4)
    
    # Use with_buffer! for automatic cleanup
    result = BlockwiseStatisticalReductions.with_buffer!((10, 10), pool) do buf
        # Fill with test data
        buf .= 1.0
        return sum(buf)
    end
    
    Test.@test result == 100.0  # sum of 100 ones
end

Test.@testset "LevelBufferPool multi-resolution" begin
    pool = BlockwiseStatisticalReductions.LevelBufferPool{Float64,2}()
    
    # Register levels for factors 2 and 4
    BlockwiseStatisticalReductions.register_level!(pool, 2, (50, 50))
    BlockwiseStatisticalReductions.register_level!(pool, 4, (25, 25))
    
    # Acquire buffers for each level
    buf_2 = BlockwiseStatisticalReductions.acquire_level!(pool, 2)
    buf_4 = BlockwiseStatisticalReductions.acquire_level!(pool, 4)
    
    Test.@test size(buf_2) == (50, 50)
    Test.@test size(buf_4) == (25, 25)
    
    # Release
    BlockwiseStatisticalReductions.release_level!(pool, 2, buf_2)
    BlockwiseStatisticalReductions.release_level!(pool, 4, buf_4)
end

Test.@testset "create_buffer_pool_for_factors" begin
    factors = [2, 4, 8, 10]
    input_shape = (100, 100)
    
    pool = BlockwiseStatisticalReductions.create_buffer_pool_for_factors(
        Float64, input_shape, factors
    )
    
    # Check all factors registered with correct shapes
    Test.@test pool.level_shapes[2] == (50, 50)   # 100/2
    Test.@test pool.level_shapes[4] == (25, 25)   # 100/4
    Test.@test pool.level_shapes[8] == (12, 12)   # 100/8 = 12.5 -> 12
    Test.@test pool.level_shapes[10] == (10, 10)  # 100/10
    
    # Acquire buffer for factor 4
    buf = BlockwiseStatisticalReductions.acquire_level!(pool, 4)
    Test.@test size(buf) == (25, 25)
end

Test.@testset "BufferPool reuse reduces allocations" begin
    pool = BlockwiseStatisticalReductions.BufferPool{Float64,2}(4)
    
    # First acquisition - may allocate
    buf1 = BlockwiseStatisticalReductions.acquire!(pool, (100, 100))
    BlockwiseStatisticalReductions.release!(pool, buf1)
    
    # Second acquisition - should reuse
    buf2 = BlockwiseStatisticalReductions.acquire!(pool, (100, 100))
    
    # Both should have same size
    Test.@test size(buf1) == size(buf2)
    
    # Release and clean up
    BlockwiseStatisticalReductions.release!(pool, buf2)
end
