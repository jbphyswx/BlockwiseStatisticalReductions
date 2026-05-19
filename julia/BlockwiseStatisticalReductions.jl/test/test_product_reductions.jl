"""
Tests for product coarsening - computing <x*y> without intermediate allocation.
"""

using Test: Test
using BlockwiseStatisticalReductions: BlockwiseStatisticalReductions
using Statistics: Statistics

Test.@testset "product_mean basic" begin
    # Simple 1D case
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]  # y = 2x
    
    win = BlockwiseStatisticalReductions.WindowConfig((2,), (2,), :valid)
    result = BlockwiseStatisticalReductions.product_mean(x, y, win)
    
    # Expected: [ (1*2 + 2*4)/2, (3*6 + 4*8)/2 ] = [5.0, 25.0]
    Test.@test result[1] ≈ (1*2 + 2*4) / 2  # = 5.0
    Test.@test result[2] ≈ (3*6 + 4*8) / 2  # = 25.0
end

Test.@testset "product_mean 2D" begin
    # 4x4 grid, 2x2 blocks
    x = reshape(1.0:16.0, 4, 4)
    y = 2 .* x  # y = 2x
    
    win = BlockwiseStatisticalReductions.WindowConfig((2, 2), (2, 2), :valid)
    result = BlockwiseStatisticalReductions.product_mean(x, y, win)
    
    # Should be 2x2 output
    Test.@test size(result) == (2, 2)
    
    # Each element is mean of products in 2x2 block
    # Block (1,1): indices (1,1), (2,1), (1,2), (2,2) -> values 1,2,5,6 for x
    # x values: 1, 2, 5, 6
    # y values: 2, 4, 10, 12
    # products: 2, 8, 50, 72 -> mean = 132/4 = 33
    expected_11 = (1*2 + 2*4 + 5*10 + 6*12) / 4
    Test.@test result[1, 1] ≈ expected_11
end

Test.@testset "product_mean matches explicit computation" begin
    for n in [10, 20, 50]
        x = rand(n, n)
        y = rand(n, n)
        
        win = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
        result = BlockwiseStatisticalReductions.product_mean(x, y, win)
        
        # Compare to explicit computation
        explicit = zeros(size(result))
        for i in 1:size(result, 1)
            for j in 1:size(result, 2)
                # Extract block
                x_block = x[(i-1)*5+1:i*5, (j-1)*5+1:j*5]
                y_block = y[(i-1)*5+1:i*5, (j-1)*5+1:j*5]
                explicit[i, j] = Statistics.mean(x_block .* y_block)
            end
        end
        
        Test.@test result ≈ explicit
    end
end

Test.@testset "product_moments basic" begin
    # Simple 1D case
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]  # y = 2x
    
    win = BlockwiseStatisticalReductions.WindowConfig((2,), (2,), :valid)
    moments = BlockwiseStatisticalReductions.product_moments(x, y, win; corrected=false)
    
    # First block: x=[1,2], y=[2,4]
    Test.@test moments.mean_x[1] ≈ 1.5
    Test.@test moments.mean_y[1] ≈ 3.0
    Test.@test moments.cov_xy[1] ≈ Statistics.cov([1.0, 2.0], [2.0, 4.0]; corrected=false)
    
    # Second block: x=[3,4], y=[6,8]
    Test.@test moments.mean_x[2] ≈ 3.5
    Test.@test moments.mean_y[2] ≈ 7.0
end

Test.@testset "product_moments matches individual computation" begin
    for _ in 1:10
        n = 20
        x = randn(n, n)
        y = randn(n, n)
        
        win = BlockwiseStatisticalReductions.WindowConfig((4, 4), (4, 4), :valid)
        moments = BlockwiseStatisticalReductions.product_moments(x, y, win)
        
        # Check a few random blocks
        for _ in 1:5
            i, j = rand(1:size(moments.mean_x, 1)), rand(1:size(moments.mean_x, 2))
            
            # Extract block
            x_block = x[(i-1)*4+1:i*4, (j-1)*4+1:j*4]
            y_block = y[(i-1)*4+1:i*4, (j-1)*4+1:j*4]
            
            Test.@test moments.mean_x[i, j] ≈ Statistics.mean(x_block)
            Test.@test moments.mean_y[i, j] ≈ Statistics.mean(y_block)
            Test.@test moments.var_x[i, j] ≈ Statistics.var(x_block)
            Test.@test moments.var_y[i, j] ≈ Statistics.var(y_block)
            Test.@test moments.cov_xy[i, j] ≈ Statistics.cov(vec(x_block), vec(y_block))
        end
    end
end

Test.@testset "covariance_from_moments" begin
    # Test the identity: Cov(x,y) = <xy> - <x><y>
    mean_x = [1.0, 2.0, 3.0]
    mean_y = [2.0, 4.0, 6.0]
    mean_xy = [3.0, 9.0, 19.0]  # Should give Cov = [1, 1, 1]
    
    cov = BlockwiseStatisticalReductions.covariance_from_moments(mean_x, mean_y, mean_xy)
    
    Test.@test cov[1] ≈ 3.0 - 1.0*2.0  # = 1.0
    Test.@test cov[2] ≈ 9.0 - 2.0*4.0  # = 1.0
    Test.@test cov[3] ≈ 19.0 - 3.0*6.0  # = 1.0
end

Test.@testset "variance_from_moments" begin
    # Test the identity: Var(x) = <x²> - <x>²
    mean = [1.0, 2.0, 3.0]
    mean_sq = [2.0, 5.0, 10.0]
    
    var = BlockwiseStatisticalReductions.variance_from_moments(mean, mean_sq)
    
    Test.@test var[1] ≈ 2.0 - 1.0^2  # = 1.0
    Test.@test var[2] ≈ 5.0 - 2.0^2  # = 1.0
    Test.@test var[3] ≈ 10.0 - 3.0^2  # = 1.0
end

Test.@testset "product_variance" begin
    # Var(x*y) computed via raw moments
    n = 20
    x = randn(n, n)
    y = randn(n, n)
    
    win = BlockwiseStatisticalReductions.WindowConfig((5, 5), (5, 5), :valid)
    prod_var = BlockwiseStatisticalReductions.product_variance(x, y, win; corrected=true)
    
    # Compare to explicit computation
    for i in 1:size(prod_var, 1)
        for j in 1:size(prod_var, 2)
            x_block = x[(i-1)*5+1:i*5, (j-1)*5+1:j*5]
            y_block = y[(i-1)*5+1:i*5, (j-1)*5+1:j*5]
            xy = x_block .* y_block
            expected = Statistics.var(vec(xy))  # Sample variance
            Test.@test prod_var[i, j] ≈ expected rtol=1e-10
        end
    end
end

Test.@testset "convenience wrappers" begin
    x = rand(10, 10)
    y = rand(10, 10)
    
    # Test keyword argument wrappers
    result = BlockwiseStatisticalReductions.blockwise_product_mean(x, y; window_sizes=(5, 5))
    Test.@test size(result) == (2, 2)
    
    moments = BlockwiseStatisticalReductions.blockwise_product_moments(x, y; window_sizes=(5, 5))
    Test.@test size(moments.mean_x) == (2, 2)
end

Test.@testset "product coarsening memory efficiency" begin
    # Verify no intermediate allocations in product_mean
    # This is more of a smoke test - real memory testing would need profiling
    n = 100
    x = rand(n, n, n)
    y = rand(n, n, n)
    
    win = BlockwiseStatisticalReductions.WindowConfig((10, 10, 10), (10, 10, 10), :valid)
    
    # Should complete without memory issues
    result = BlockwiseStatisticalReductions.product_mean(x, y, win)
    Test.@test size(result) == (10, 10, 10)
    
    # Product moments should also work on 3D
    moments = BlockwiseStatisticalReductions.product_moments(x, y, win)
    Test.@test size(moments.mean_x) == (10, 10, 10)
end
