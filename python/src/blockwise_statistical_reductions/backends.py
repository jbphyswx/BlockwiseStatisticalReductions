"""
Backend abstractions for different execution environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np


class ExecutionBackend(ABC):
    """
    Abstract base class for execution backends.

    Backends control how plans are executed: single-threaded, multi-threaded,
    distributed via Dask, etc.
    """

    @abstractmethod
    def execute(self, plan: Any, data: np.ndarray, **kwargs) -> Any:
        """Execute a plan on data."""
        pass

    @abstractmethod
    def map_reduce(
        self,
        data: list,
        map_fn: Callable,
        reduce_fn: Callable,
    ) -> Any:
        """Map then reduce over partitioned data."""
        pass


class CPUBackend(ExecutionBackend):
    """Single/multi-threaded CPU execution."""

    def __init__(self, n_workers: int = 1):
        self.n_workers = n_workers

    def execute(self, plan: Any, data: np.ndarray, **kwargs) -> Any:
        """Execute on CPU."""
        # For now, direct execution
        # Could use multiprocessing for n_workers > 1
        return plan.execute(data)

    def map_reduce(
        self,
        data: list,
        map_fn: Callable,
        reduce_fn: Callable,
    ) -> Any:
        """Sequential map-reduce."""
        mapped = [map_fn(chunk) for chunk in data]
        return reduce_fn(mapped)


class DaskBackend(ExecutionBackend):
    """Dask-based parallel execution (local scheduler)."""

    def __init__(self, n_workers: int | None = None, threads_per_worker: int = 1):
        import dask
        self.dask = dask
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker

    def execute(self, plan: Any, data: np.ndarray, **kwargs) -> Any:
        """Execute using Dask."""
        from .dask_integration import execute_dask_plan
        return execute_dask_plan(plan, data, scheduler="threads")

    def map_reduce(
        self,
        data: list,
        map_fn: Callable,
        reduce_fn: Callable,
    ) -> Any:
        """Dask map-reduce."""
        import dask.bag as db

        bag = db.from_sequence(data, partition_size=1)
        mapped = bag.map(map_fn)
        # Tree reduction happens automatically in dask
        results = mapped.compute()
        return reduce_fn(list(results))


class DaskDistributedBackend(ExecutionBackend):
    """Dask Distributed execution (cluster)."""

    def __init__(self, client=None, address: str | None = None):
        from distributed import Client

        if client is not None:
            self.client = client
        elif address is not None:
            self.client = Client(address)
        else:
            self.client = Client()  # Start local cluster

    def execute(self, plan: Any, data: np.ndarray, **kwargs) -> Any:
        """Execute on distributed cluster."""
        from .dask_integration import execute_dask_plan

        # Scatter large data to workers
        future = self.client.scatter(data, broadcast=True)
        return execute_dask_plan(plan, future, scheduler=self.client)

    def map_reduce(
        self,
        data: list,
        map_fn: Callable,
        reduce_fn: Callable,
    ) -> Any:
        """Distributed map-reduce."""
        # Scatter data
        futures = self.client.scatter(data)

        # Map
        mapped_futures = [self.client.submit(map_fn, f) for f in futures]

        # Tree reduce
        while len(mapped_futures) > 1:
            new_futures = []
            for i in range(0, len(mapped_futures), 2):
                if i + 1 < len(mapped_futures):
                    # Submit reduction of pair
                    f = self.client.submit(reduce_fn, [mapped_futures[i], mapped_futures[i+1]])
                    new_futures.append(f)
                else:
                    new_futures.append(mapped_futures[i])
            mapped_futures = new_futures

        return mapped_futures[0].result()

    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()


class JAXBackend(ExecutionBackend):
    """JAX-based GPU/TPU execution."""

    def __init__(self, device: str = "gpu"):
        try:
            import jax
            self.jax = jax
            self.device = device
        except ImportError:
            raise ImportError("JAX not installed. Install with: pip install jax jaxlib")

    def execute(self, plan: Any, data: np.ndarray, **kwargs) -> Any:
        """Execute using JAX."""
        # Convert to JAX array
        jdata = self.jax.numpy.array(data)

        # JIT compile plan execution
        @self.jax.jit
        def run(x):
            return plan.execute_jax(x)

        return np.array(run(jdata))

    def map_reduce(
        self,
        data: list,
        map_fn: Callable,
        reduce_fn: Callable,
    ) -> Any:
        """JAX pmap-based map-reduce."""
        # pmap for multi-device, vmap for single-device batching
        import jax

        jdata = [self.jax.numpy.array(d) for d in data]

        # Vectorized map
        vmapped_fn = jax.vmap(lambda x: map_fn(np.array(x)))
        mapped = vmapped_fn(jdata)

        # P-tree reduce
        while len(mapped) > 1:
            new_mapped = []
            for i in range(0, len(mapped), 2):
                if i + 1 < len(mapped):
                    new_mapped.append(reduce_fn([mapped[i], mapped[i+1]]))
                else:
                    new_mapped.append(mapped[i])
            mapped = new_mapped

        return np.array(mapped[0])


def get_backend(name: str, **kwargs) -> ExecutionBackend:
    """
    Factory function to get a backend by name.

    Parameters
    ----------
    name : str
        "cpu", "dask", "distributed", "jax"
    **kwargs
        Passed to backend constructor

    Returns
    -------
    ExecutionBackend
    """
    backends = {
        "cpu": CPUBackend,
        "dask": DaskBackend,
        "distributed": DaskDistributedBackend,
        "jax": JAXBackend,
    }

    if name not in backends:
        raise ValueError(f"Unknown backend: {name}. Choose from {list(backends.keys())}")

    return backends[name](**kwargs)
