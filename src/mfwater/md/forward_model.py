"""Forward models for the multifidelity MCMC algorithm.

A forward model maps a number of molecules ``N`` and a parameter vector
``theta = (epsilon_OO, sigma_OO, q_O)`` (kcal/mol, Angstrom, e) to the
diffusion coefficient ``D`` in units of 1e-9 m^2/s.

The real forward model is stochastic (MD). To make evaluations reproducible
and cacheable, every evaluation is identified by a key built from ``N`` and the
exact float64 values of ``theta``. Random seeds of the forward model are derived
from a hash of this key, so they never touch the random number generator of the
sampling algorithm.

Key compatibility: for the default protocol ``"mfda"`` without explicit seeds
the key is the 4-tuple ``(N, hex(eps), hex(sigma), hex(q_O))`` of the first MFDA
version. Hash, derived seeds and cache directory names of existing MFDA runs
depend on it and must never change (``test/test_md/test_mfda_regression.py``).
Every other combination uses the extended key
``(N, hex..., protocol, orthoboxy, seeds | None)``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np

from ..argparser import constants

#: Largest seed accepted by LAMMPS/packmol (signed 32 bit integer).
MAX_SEED = 2**31 - 1

#: Name of the protocol whose keys keep the original 4-tuple form.
LEGACY_PROTOCOL = "mfda"

#: Identifier of one evaluation: 4-tuple (legacy) or extended 7-tuple.
EvaluationKey = tuple[Any, ...]


class ForwardModel(Protocol):
    """Callable interface of all forward models.

    The algorithm only relies on this protocol. Caching is done by a wrapper
    (see :class:`mfwater.md.cache.CachedForwardModel`).
    """

    # positional-only, so that plain functions with other parameter names also match
    def __call__(self, n_molecules: int, theta: np.ndarray, /) -> float:
        """Evaluate the forward model.

        Parameters
        ----------
        n_molecules : int
            Number of molecules in the system.
        theta : np.ndarray
            Parameters ``(epsilon_OO, sigma_OO, q_O)``.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.
        """
        ...  # pragma: no cover


def _check_theta(theta: np.ndarray) -> np.ndarray:
    """Validate and normalise a parameter vector.

    Parameters
    ----------
    theta : np.ndarray
        Parameters ``(epsilon_OO, sigma_OO, q_O)``.

    Returns
    -------
    np.ndarray
        The parameters as a float64 array of shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``theta`` does not have exactly three finite entries.
    """
    arr = np.asarray(theta, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(
            f"theta must have shape (3,) = (epsilon, sigma, q_O), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"theta must be finite, got {arr}.")
    return arr


def _check_seeds(seeds: tuple[int, int] | None) -> tuple[int, int] | None:
    """Validate explicit MD seeds.

    Parameters
    ----------
    seeds : tuple[int, int] | None
        ``(packmol_seed, velocity_seed)`` or None.

    Returns
    -------
    tuple[int, int] | None
        The seeds as a tuple of Python ints, or None.

    Raises
    ------
    ValueError
        If a seed is not an integer in ``[1, MAX_SEED]`` (0 means "random" to
        some codes).
    """
    if seeds is None:
        return None
    if len(seeds) != 2:
        raise ValueError(f"seeds must be (packmol, velocity), got {seeds!r}.")
    out = []
    for s in seeds:
        if int(s) != s or not 1 <= int(s) <= MAX_SEED:
            raise ValueError(f"seed {s!r} is not an integer in [1, {MAX_SEED}].")
        out.append(int(s))
    return out[0], out[1]


def evaluation_key(
    n_molecules: int,
    theta: np.ndarray,
    protocol: str = LEGACY_PROTOCOL,
    orthoboxy: bool = False,
    seeds: tuple[int, int] | None = None,
) -> EvaluationKey:
    """Build the exact identifier of one forward evaluation.

    ``float.hex()`` is used because it is an exact float64 round trip, unlike
    decimal formatting.

    For ``protocol == "mfda"`` without explicit ``seeds`` the key is the
    original 4-tuple; ``orthoboxy`` is then *not* part of it (the box shape of
    MFDA results is checked against ``result.json`` instead). Otherwise the key
    is ``(n, hex(epsilon), hex(sigma), hex(q_O), protocol, orthoboxy, seeds)``.

    Parameters
    ----------
    n_molecules : int
        Number of molecules.
    theta : np.ndarray
        Parameters ``(epsilon_OO, sigma_OO, q_O)``.
    protocol : str, optional
        Name of the MD protocol.
    orthoboxy : bool, optional
        Whether the box is an OrthoBoXY box (tetragonal).
    seeds : tuple[int, int] | None, optional
        Explicit ``(packmol_seed, velocity_seed)``; None derives them from the
        key (:func:`derive_seeds`).

    Returns
    -------
    tuple
        The evaluation key.

    Raises
    ------
    ValueError
        If ``theta`` or ``seeds`` are invalid.
    """
    eps, sig, q = (float(x) for x in _check_theta(theta))
    checked = _check_seeds(seeds)
    base = (int(n_molecules), eps.hex(), sig.hex(), q.hex())
    if protocol == LEGACY_PROTOCOL and checked is None:
        return base
    return (*base, protocol, bool(orthoboxy), checked)


def _full_digest(key: EvaluationKey) -> str:
    """Return the full SHA-256 hex digest of ``repr(key)``.

    Parameters
    ----------
    key : tuple
        Key from :func:`evaluation_key`.

    Returns
    -------
    str
        64 hex characters.
    """
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def key_hash(key: EvaluationKey) -> str:
    """Return the short, stable hash of an evaluation key.

    Parameters
    ----------
    key : tuple
        Key from :func:`evaluation_key`.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 digest of ``repr(key)``.
    """
    return _full_digest(key)[:16]


def derive_seeds(key: EvaluationKey) -> tuple[int, int]:
    """Derive the MD seeds from the full hash of an evaluation key.

    Explicit seeds stored in an extended key are returned unchanged.

    Parameters
    ----------
    key : tuple
        Key from :func:`evaluation_key`.

    Returns
    -------
    tuple[int, int]
        ``(packmol_seed, velocity_seed)``, both in ``[1, 2**31 - 1]``.
    """
    if len(key) == 7 and key[6] is not None:
        return key[6]
    digest = _full_digest(key)
    # Two independent 64-bit slices of the digest; the modulo maps to [0, MAX-1]
    # and the +1 excludes 0, which some codes treat as "random seed".
    packmol = int(digest[16:32], 16) % MAX_SEED + 1
    velocity = int(digest[32:48], 16) % MAX_SEED + 1
    return packmol, velocity


class DummyForwardModel:
    """Cheap stand-in for the MD forward model, deterministic in ``(N, theta)``.

    The output is drawn from the distribution of the experimental measurements,
    with a mean that depends on ``N`` like the dummy of the first implementation.
    The generator is seeded from the hash of the evaluation key, so the same
    ``(N, theta)`` always gives the same value without caching.
    """

    def __call__(self, n_molecules: int, theta: np.ndarray) -> float:
        """Evaluate the dummy model.

        Parameters
        ----------
        n_molecules : int
            Number of molecules in the system.
        theta : np.ndarray
            Parameters ``(epsilon_OO, sigma_OO, q_O)``.

        Returns
        -------
        float
            Dummy diffusion coefficient in 1e-9 m^2/s.
        """
        key = evaluation_key(n_molecules, theta)
        rng = np.random.default_rng(derive_seeds(key)[0])
        return float(
            rng.normal(
                loc=constants.TARGET_DIFFUSION_COEFFICIENT.mean()
                * (np.log2(n_molecules) / 10),
                scale=constants.TARGET_DIFFUSION_COEFFICIENT.std(),
            )
        )


def forward_model_dummy(n_molecules: int, parameters: np.ndarray) -> float:
    """Dummy forward model (thin wrapper around :class:`DummyForwardModel`).

    Kept for backward compatibility with code that imports it.

    Parameters
    ----------
    n_molecules : int
        Number of molecules in the system for the given fidelity level.
    parameters : np.ndarray
        Parameter sample ``(epsilon_OO, sigma_OO, q_O)``.

    Returns
    -------
    float
        Dummy diffusion coefficient in 1e-9 m^2/s, deterministic in the input.
    """
    return DummyForwardModel()(n_molecules, parameters)
