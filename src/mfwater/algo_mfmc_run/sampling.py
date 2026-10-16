"""Samples (parameters and MD seeds) of the calibration and performance phases.

The distributions are those of the legacy ``chemmodel-prep``
(:func:`~mfwater.algo_chemical_model.chemical_model.sampl_lj_params`,
:func:`~mfwater.algo_chemical_model.chemical_model.sampl_partial_charges`, widths
``NOISE_LJPARAMS = 1/300`` and ``NOISE_CHARGES = 1/50``), drawn in the same order
(LJ parameters, seeds, charges). Values are stored as float32 like in paper I, so
that a stored sample is reproduced exactly by :func:`read_samples`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from ..algo_chemical_model.chemical_model import sampl_lj_params, sampl_partial_charges
from ..argparser import constants

#: Paper-I seeds are drawn from ``[1, 100000)``.
SEED_HIGH = 100000


@dataclass
class Samples:
    """Parameter samples and MD seeds shared by all models of a phase.

    Parameters
    ----------
    lj_params : np.ndarray
        float32 ``(2, n)``: epsilon (kcal/mol) and sigma (Angstrom).
    charges : np.ndarray
        float32 ``(2, n)``: q_O and q_H (e).
    seeds : np.ndarray
        int32 ``(2, n)``: packmol seed (row 0) and velocity seed (row 1).
    """

    lj_params: npt.NDArray[np.float32]
    charges: npt.NDArray[np.float32]
    seeds: npt.NDArray[np.int32]

    @property
    def n(self) -> int:
        """Return the number of samples."""
        return int(self.seeds.shape[1])

    def theta(self, j: int) -> npt.NDArray[np.float64]:
        """Return the parameters ``(epsilon, sigma, q_O)`` of sample ``j``.

        Parameters
        ----------
        j : int
            Sample index.

        Returns
        -------
        np.ndarray
            float64 array of shape ``(3,)`` (the float32 values, exactly).
        """
        return np.array(
            [self.lj_params[0, j], self.lj_params[1, j], self.charges[0, j]],
            dtype=np.float64,
        )

    def seed_pair(self, j: int) -> tuple[int, int]:
        """Return ``(packmol_seed, velocity_seed)`` of sample ``j``.

        Parameters
        ----------
        j : int
            Sample index.

        Returns
        -------
        tuple[int, int]
            The two seeds as Python integers.
        """
        return int(self.seeds[0, j]), int(self.seeds[1, j])

    def head(self, n: int) -> Samples:
        """Return the first ``n`` samples.

        Parameters
        ----------
        n : int
            Number of samples to keep.

        Returns
        -------
        Samples
            A view of the first ``n`` columns.

        Raises
        ------
        ValueError
            If fewer than ``n`` samples exist.
        """
        if n > self.n:
            raise ValueError(f"Only {self.n} samples available, {n} requested.")
        return Samples(self.lj_params[:, :n], self.charges[:, :n], self.seeds[:, :n])


def draw_samples(n: int, params: str, rng: np.random.Generator) -> Samples:
    """Draw ``n`` samples like ``chemmodel-prep``.

    Parameters
    ----------
    n : int
        Number of samples.
    params : str
        ``lj``, ``q`` or ``lj-q``: which parameters are perturbed (the others keep
        their OPC3 value, with zero noise).
    rng : np.random.Generator
        Source of all random numbers.

    Returns
    -------
    Samples
        The samples.
    """
    noise_lj = constants.NOISE_LJPARAMS if params in ("lj", "lj-q") else 0.0
    noise_q = constants.NOISE_CHARGES if params in ("q", "lj-q") else 0.0
    lj = sampl_lj_params(np.zeros((2, n), dtype=np.float32), noise_lj, rng)
    seeds = rng.integers(1, SEED_HIGH, size=(2, n), dtype=np.int32)
    charges = sampl_partial_charges(np.zeros((2, n), dtype=np.float32), noise_q, rng)
    return Samples(lj, charges, seeds)


def read_samples(path: Path) -> Samples:
    """Read the stored samples of a paper-I-format file.

    Parameters
    ----------
    path : Path
        HDF5 file with ``/models/lj_params``, ``/models/seeds`` and (optionally)
        ``/models/charges``. Without ``charges`` (old format) the OPC3 charges are used
        (the ``q`` parameter was not perturbed then).

    Returns
    -------
    Samples
        The stored samples (float32 / int32, as stored).

    Raises
    ------
    ValueError
        If ``lj_params`` or ``seeds`` are missing or the shapes disagree.
    """
    with h5py.File(path, "r") as f:
        models = f["models"]
        for name in ("lj_params", "seeds"):
            if name not in models:
                raise ValueError(f"{path} has no /models/{name} to replay.")
        lj = np.asarray(models["lj_params"], dtype=np.float32)
        seeds = np.asarray(models["seeds"], dtype=np.int32)
        if "charges" in models:
            charges = np.asarray(models["charges"], dtype=np.float32)
        else:
            q_o = np.float32(constants.OPC3_CHARGE_O)
            charges = np.empty((2, lj.shape[1]), dtype=np.float32)
            charges[0, :] = q_o
            charges[1, :] = -0.5 * q_o
    if not lj.shape == charges.shape == seeds.shape or lj.shape[0] != 2:
        raise ValueError(
            f"{path}: lj_params {lj.shape}, charges {charges.shape} and seeds "
            f"{seeds.shape} must all have shape (2, n)."
        )
    return Samples(lj, charges, seeds)
