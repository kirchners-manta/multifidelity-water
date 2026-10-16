"""Correlated dummy forward model for tests and dry runs of ``mfmc-run``.

The model mimics the structure that MFMC exploits: the output of every fidelity
level is a deterministic function of the parameters (shared by all levels) plus
noise that is independent between levels and shrinks with the number of molecules::

    D(N, theta, seeds) = a(N) + b(N) * g(theta) + noise(N, seeds)

``a`` and ``b`` increase with ``N``, so that the correlation with the largest system
decreases with decreasing ``N``. The cost is a deterministic power law of ``N``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..argparser import constants
from ..md.forward_model import derive_seeds, evaluation_key
from ..md.protocol import MDProtocol, get_protocol
from ..md.system import calc_cpus


def _g(theta: np.ndarray) -> float:
    """Return the parameter dependence shared by all levels (O(1) under sampling).

    Parameters
    ----------
    theta : np.ndarray
        Parameters ``(epsilon_OO, sigma_OO, q_O)``.

    Returns
    -------
    float
        Sum of the deviations from the OPC3 values in units of the sampling widths.
    """
    eps = (theta[0] / constants.OPC3_EPSILON_OO - 1.0) / constants.NOISE_LJPARAMS
    sig = (theta[1] / constants.OPC3_SIGMA_OO - 1.0) / constants.NOISE_LJPARAMS
    q = (theta[2] / constants.OPC3_CHARGE_O - 1.0) / constants.NOISE_CHARGES
    return float(eps + sig + q)


class CorrelatedDummyForwardModel:
    """Cheap forward model with level-correlated outputs and a deterministic cost.

    Picklable, so that it can be used with :func:`mfwater.md.pool.run_pool`.

    Parameters
    ----------
    protocol : str | MDProtocol, optional
        Protocol; only part of the cache key.
    orthoboxy : bool, optional
        Only part of the cache key.
    ncpu : dict[int, int] or None, optional
        Table ``N -> ranks``; other ``N`` fall back to ``calc_cpus``.
    sleep : float, optional
        Seconds to sleep per evaluation (to test concurrency).
    incomplete_every : int, optional
        If > 0, evaluations whose packmol seed is divisible by this number report
        ``cost_complete = False`` (to test the exclusion of incomplete costs).
    require_cost : bool, optional
        Accepted for interface compatibility with ``MDForwardModel``; ignored.

    Attributes
    ----------
    last_cost : dict | None
        Cost record of the last evaluation (written to ``result.json``).
    """

    def __init__(
        self,
        protocol: str | MDProtocol = "paper1",
        orthoboxy: bool = False,
        ncpu: Any = None,
        sleep: float = 0.0,
        incomplete_every: int = 0,
        require_cost: bool = True,
    ) -> None:
        self.protocol = get_protocol(protocol)
        self.orthoboxy = orthoboxy
        self.ncpu = dict(ncpu) if ncpu else {}
        self.sleep = sleep
        self.incomplete_every = incomplete_every
        self.require_cost = require_cost
        self.last_cost: dict[str, Any] | None = None

    def n_mpi(self, n_molecules: int) -> int:
        """Return the number of MPI ranks of an evaluation of ``n_molecules``.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.

        Returns
        -------
        int
            Override from ``ncpu``, else ``calc_cpus``.
        """
        return int(self.ncpu.get(n_molecules, calc_cpus(n_molecules)))

    @staticmethod
    def cost(n_molecules: int, n_mpi: int) -> float:
        """Return the deterministic cost of one evaluation in core-seconds.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        n_mpi : int
            Ranks.

        Returns
        -------
        float
            ``n_mpi * 0.01 * N**1.1`` (superlinear in ``N``, like MD).
        """
        return n_mpi * 0.01 * n_molecules**1.1

    def __call__(
        self,
        n_molecules: int,
        theta: np.ndarray,
        seeds: tuple[int, int] | None = None,
    ) -> float:
        """Evaluate the model.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        theta : np.ndarray
            Parameters ``(epsilon_OO, sigma_OO, q_O)``.
        seeds : tuple[int, int] | None, optional
            ``(packmol_seed, velocity_seed)``; derived from the key if None.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.
        """
        theta = np.asarray(theta, dtype=float)
        if seeds is None:
            seeds = derive_seeds(evaluation_key(n_molecules, theta))
        if self.sleep:
            time.sleep(self.sleep)
        a = 1.5 + 0.1 * np.log2(n_molecules)
        b = 0.3 * n_molecules / (n_molecules + 16.0)
        sd = 0.5 / np.sqrt(n_molecules)
        # noise independent between levels: N is part of the generator seed
        noise = np.random.default_rng([seeds[0], seeds[1], n_molecules]).normal(0.0, sd)
        n_mpi = self.n_mpi(n_molecules)
        cpu = self.cost(n_molecules, n_mpi)
        complete = not (self.incomplete_every and seeds[0] % self.incomplete_every == 0)
        self.last_cost = {
            "cpu_time_s": cpu if complete else 0.5 * cpu,
            "n_mpi": n_mpi,
            "n_mpi_requested": n_mpi,
            "cost_complete": complete,
        }
        return float(a + b * _g(theta) + noise)
