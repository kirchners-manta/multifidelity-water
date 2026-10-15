"""Persistent on-disk cache for forward model evaluations.

Layout::

    <cache_dir>/N{n}/{key_hash}/result.json

A result exists if and only if ``result.json`` exists. It is written atomically
(temporary file + ``os.replace``), so an interrupted run never leaves a partial
result. Directories of MD attempts next to it (``attempt_k/``) are never touched.

The directory name is the hash of the evaluation key. For the ``mfda`` protocol
without explicit seeds the key is the one of the first MFDA version, so existing
caches stay valid; see :func:`mfwater.md.forward_model.evaluation_key`.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .forward_model import (
    LEGACY_PROTOCOL,
    EvaluationKey,
    ForwardModel,
    _check_theta,
    derive_seeds,
    evaluation_key,
    key_hash,
)

#: Unit of the cached diffusion coefficients.
D_UNIT = "1e-9 m^2/s"


class CachedForwardModel:
    """Forward model wrapper with an in-memory memo and a persistent cache.

    Parameters
    ----------
    model : ForwardModel
        The wrapped (expensive) forward model.
    cache_dir : str | Path
        Root directory of the cache. Created if it does not exist.

    Attributes
    ----------
    n_hits : int
        Number of evaluations answered from memo or disk.
    n_computed : int
        Number of evaluations computed by the wrapped model.
    """

    def __init__(self, model: ForwardModel, cache_dir: str | Path) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memo: dict[EvaluationKey, float] = {}
        self.n_hits = 0
        self.n_computed = 0

    def make_key(
        self,
        n_molecules: int,
        theta: np.ndarray,
        seeds: tuple[int, int] | None = None,
    ) -> EvaluationKey:
        """Build the evaluation key including protocol and box shape of the model.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        theta : np.ndarray
            Parameters ``(epsilon_OO, sigma_OO, q_O)``.
        seeds : tuple[int, int] | None, optional
            Explicit ``(packmol_seed, velocity_seed)``.

        Returns
        -------
        tuple
            Key from :func:`mfwater.md.forward_model.evaluation_key`. Models
            without ``protocol``/``orthoboxy`` attributes count as ``mfda`` with
            a cubic box.
        """
        protocol = getattr(self.model, "protocol", None)
        return evaluation_key(
            n_molecules,
            theta,
            protocol=LEGACY_PROTOCOL if protocol is None else protocol.name,
            orthoboxy=bool(getattr(self.model, "orthoboxy", False)),
            seeds=seeds,
        )

    def entry_dir(
        self,
        n_molecules: int,
        theta: np.ndarray,
        seeds: tuple[int, int] | None = None,
    ) -> Path:
        """Return the cache directory of one evaluation.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        theta : np.ndarray
            Parameters ``(epsilon_OO, sigma_OO, q_O)``.
        seeds : tuple[int, int] | None, optional
            Explicit ``(packmol_seed, velocity_seed)``.

        Returns
        -------
        Path
            ``<cache_dir>/N{n}/{key_hash}``.
        """
        key = self.make_key(n_molecules, theta, seeds)
        return self.cache_dir / f"N{key[0]}" / key_hash(key)

    def __len__(self) -> int:
        """Return the number of results stored on disk."""
        return sum(1 for _ in self.cache_dir.glob("N*/*/result.json"))

    def __call__(
        self,
        n_molecules: int,
        theta: np.ndarray,
        seeds: tuple[int, int] | None = None,
    ) -> float:
        """Return D for ``(n_molecules, theta)``, computing it only once.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        theta : np.ndarray
            Parameters ``(epsilon_OO, sigma_OO, q_O)``.
        seeds : tuple[int, int] | None, optional
            Explicit ``(packmol_seed, velocity_seed)`` instead of seeds derived
            from the key. They are part of the key. The wrapped model must
            accept a ``seeds`` keyword argument if they are given.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.

        Raises
        ------
        ValueError
            If ``theta`` or ``seeds`` are invalid or a stored result is
            inconsistent.
        RuntimeError
            If the wrapped model returns a non-finite value.
        """
        theta = _check_theta(theta)
        key = self.make_key(n_molecules, theta, seeds)
        if key in self._memo:
            self.n_hits += 1
            return self._memo[key]

        entry = self.cache_dir / f"N{key[0]}" / key_hash(key)
        result_file = entry / "result.json"
        if result_file.exists():
            d = self._read_result(
                result_file, key, getattr(self.model, "orthoboxy", None)
            )
            self._memo[key] = d
            self.n_hits += 1
            return d

        entry.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if seeds is None:
            d = float(self.model(n_molecules, theta))
        else:
            d = float(self.model(n_molecules, theta, seeds=seeds))  # type: ignore[call-arg]
        wall = time.perf_counter() - start
        if not math.isfinite(d):
            raise RuntimeError(
                f"Forward model returned non-finite D={d} for key {key}."
            )
        self._write_result(entry, key, theta, d, wall)
        self._memo[key] = d
        self.n_computed += 1
        return d

    @staticmethod
    def _read_result(
        path: Path,
        key: EvaluationKey,
        orthoboxy: bool | None = None,
    ) -> float:
        """Read and validate a stored result.

        Parameters
        ----------
        path : Path
            Path of ``result.json``.
        key : tuple
            Expected evaluation key.
        orthoboxy : bool | None, optional
            Box-shape flag of the wrapped model (``None`` if it has none). The
            key does not contain the box shape, so a differing stored flag means
            the cache directory is mixed between cubic and OrthoBoXY runs.

        Returns
        -------
        float
            The stored diffusion coefficient.

        Raises
        ------
        ValueError
            If the file does not belong to ``key`` (hash collision or manual
            edit) or its stored ``orthoboxy`` flag differs from ``orthoboxy``.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        stored = (
            data["N"],
            data["theta_hex"]["epsilon"],
            data["theta_hex"]["sigma"],
            data["theta_hex"]["q_O"],
        )
        if len(key) == 4:
            # legacy key of the mfda protocol; results of the first MFDA version
            # have no "protocol" entry, later ones store "mfda"
            ok = tuple(stored) == key and data.get("protocol", LEGACY_PROTOCOL) == (
                LEGACY_PROTOCOL
            )
            ok = ok and data.get("key") in (None, list(key))
        else:
            # extended key: stored as a JSON list (tuples become lists)
            ok = data.get("key") == json.loads(json.dumps(key))
        if not ok:
            raise ValueError(f"Cache entry {path} does not match key {key}.")
        stored_ob = data.get("orthoboxy")  # absent in results of older versions
        if orthoboxy is not None and stored_ob is not None and stored_ob != orthoboxy:
            raise ValueError(
                f"Cache entry {path} was computed with orthoboxy={stored_ob}, but "
                f"the forward model uses orthoboxy={orthoboxy}. Use a separate "
                "cache directory per box shape."
            )
        return float(data["D"])

    def _write_result(
        self,
        entry: Path,
        key: EvaluationKey,
        theta: np.ndarray,
        d: float,
        wall_time: float,
    ) -> None:
        """Write ``result.json`` atomically.

        Parameters
        ----------
        entry : Path
            Directory of the evaluation.
        key : tuple
            Evaluation key.
        theta : np.ndarray
            Parameters ``(epsilon_OO, sigma_OO, q_O)``.
        d : float
            Diffusion coefficient in 1e-9 m^2/s.
        wall_time : float
            Wall time of the evaluation in seconds.
        """
        packmol_seed, velocity_seed = derive_seeds(key)
        attempt = getattr(self.model, "last_attempt_dir", None)
        # all attempts that contributed (several after a resume from restart files)
        attempts = getattr(self.model, "last_attempt_dirs", None)
        payload: dict[str, Any] = {
            "D": d,
            "unit": D_UNIT,
            "N": key[0],
            "theta": {
                "epsilon": float(theta[0]),
                "sigma": float(theta[1]),
                "q_O": float(theta[2]),
            },
            "theta_hex": {"epsilon": key[1], "sigma": key[2], "q_O": key[3]},
            "key_hash": key_hash(key),
            "key": list(key),
            "protocol": getattr(
                getattr(self.model, "protocol", None), "name", LEGACY_PROTOCOL
            ),
            "seeds": {"packmol": packmol_seed, "velocity": velocity_seed},
            "orthoboxy": getattr(self.model, "orthoboxy", None),
            "wall_time_s": wall_time,
            # core-seconds of LAMMPS (wall time x MPI ranks, summed over the
            # attempts of a resumed evaluation); None for models without a log
            "cpu_time_s": None,
            "n_mpi": None,
            "n_mpi_requested": None,
            "attempt_dir": None if attempt is None else str(attempt),
            "attempt_dirs": None if attempts is None else [str(a) for a in attempts],
        }
        cost = getattr(self.model, "last_cost", None)
        if cost is not None:
            payload.update(cost)
        # The temporary file lives in the same directory so that os.replace is
        # an atomic rename on the same file system.
        fd, tmp = tempfile.mkstemp(dir=entry, prefix=".result_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, entry / "result.json")
        except BaseException:
            # Only the temporary file is removed, never simulation output.
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
