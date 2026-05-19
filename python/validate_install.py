"""
Validation script to check that the package is installed correctly.
Run this after installation to verify everything works.
"""

import sys
import numpy as np

def check_imports():
    """Check that all required imports work."""
    print("Checking imports...")

    try:
        import blockwise_statistical_reductions as bsr
        print(f"  ✓ blockwise_statistical_reductions v{bsr.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False

    # Check core
    try:
        from bsr import WindowConfig, ReductionPlan
        print("  ✓ Core types")
    except ImportError as e:
        print(f"  ✗ Core types: {e}")
        return False

    # Check statistics
    try:
        from bsr import rolling_stats, blockwise_stats
        print("  ✓ Statistics functions")
    except ImportError as e:
        print(f"  ✗ Statistics: {e}")
        return False

    # Check backends
    try:
        from bsr import CPUBackend, DaskBackend
        print("  ✓ Backends")
    except ImportError as e:
        print(f"  ✗ Backends: {e}")
        return False

    # Check Dask (optional)
    try:
        from bsr import dask_rolling_stats, dask_blockwise_stats
        print("  ✓ Dask integration")
    except ImportError:
        print("  ○ Dask integration (optional, not installed)")

    # Check Xarray (optional)
    try:
        from bsr import xr_rolling_stats, xr_blockwise_stats
        print("  ✓ Xarray integration")
    except ImportError:
        print("  ○ Xarray integration (optional, not installed)")

    return True


def check_basic_functionality():
    """Check that basic operations work."""
    print("\nChecking basic functionality...")

    try:
        from blockwise_statistical_reductions import WindowConfig, blockwise_stats

        # Create test data
        data = np.random.randn(100, 100)

        # Compute blockwise mean
        result = blockwise_stats(data, (10, 10), "mean", strict=True)

        if result.values.shape == (10, 10):
            print("  ✓ Blockwise mean works")
        else:
            print(f"  ✗ Wrong shape: {result.values.shape}")
            return False

        # Test validation
        try:
            blockwise_stats(data, (11, 11), "mean", strict=True)
            print("  ✗ Strict validation not working")
            return False
        except ValueError:
            print("  ✓ Strict validation catches errors")

        return True

    except Exception as e:
        print(f"  ✗ Basic functionality failed: {e}")
        return False


def check_numba():
    """Check that Numba is working."""
    print("\nChecking Numba optimization...")

    try:
        from blockwise_statistical_reductions import rolling_stats, WindowConfig

        data = np.random.randn(100, 100)
        config = WindowConfig((10, 10))

        # Test numba path
        result_numba = rolling_stats(data, config, "mean", use_numba=True, strict=True)
        result_python = rolling_stats(data, config, "mean", use_numba=False, strict=True)

        if np.allclose(result_numba.values, result_python.values):
            print("  ✓ Numba and Python paths match")
            return True
        else:
            print("  ✗ Numba and Python paths differ")
            return False

    except Exception as e:
        print(f"  ✗ Numba check failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("BlockwiseStatisticalReductions Validation")
    print("=" * 60)

    results = []

    results.append(("Imports", check_imports()))
    results.append(("Basic functionality", check_basic_functionality()))
    results.append(("Numba optimization", check_numba()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("All checks passed! Package is working correctly.")
        return 0
    else:
        print("Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
