"""Evaluation results, calibration data and the HDF5 files of ``mfmc-run``.

The HDF5 files use the group and attribute names of paper I, so that the legacy
analysis notebooks work on them. Differences (all documented in the README):

* ``diffusion_coeff`` is stored in float64 (paper I: float32).
* Models are ordered by ``n_molecules`` (calibration) or by decreasing ``rho**2``
  (performance); the group names ``model_k`` follow that order. Always sort by the
  attribute ``n_molecules``, never by the h5py name order.

Files are written under a temporary name and linked into place
(:func:`write_hdf5_once`), so that a finished file is never replaced and several jobs
finishing a phase at the same time cannot corrupt it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import h5py
import numpy as np
import numpy.typing as npt

from .. import mfmc_core
from ..algo_input import model_groups
from ..md.cache import CachedForwardModel
from .sampling import Samples, read_samples

#: The MD forward model returns D in 1e-9 m^2/s, paper I stores 1e-12 m^2/s.
D_SCALE = 1000.0
D_UNIT = "1e-12 m^2/s"
COST_UNIT = "core-s"

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Evaluation:
    """One finished MD evaluation read from ``result.json``.

    Parameters
    ----------
    d : float
        Diffusion coefficient in 1e-12 m^2/s (paper-I unit).
    cost : float | None
        Core-seconds (LAMMPS wall time x ranks, summed over attempts); None if unknown.
    cost_complete : bool
        False if an attempt was killed without a ``Total wall time`` (the cost is then
        a lower bound and not used for ``w``).
    """

    d: float
    cost: float | None
    cost_complete: bool


def read_evaluation(
    forward: CachedForwardModel,
    n_molecules: int,
    theta: np.ndarray,
    seeds: tuple[int, int],
) -> Evaluation:
    """Read the stored result of one evaluation (never computes).

    Parameters
    ----------
    forward : CachedForwardModel
        Cached forward model (defines the cache directory and the key).
    n_molecules : int
        Number of molecules.
    theta : np.ndarray
        Parameters ``(epsilon_OO, sigma_OO, q_O)``.
    seeds : tuple[int, int]
        ``(packmol_seed, velocity_seed)``.

    Returns
    -------
    Evaluation
        The stored D and cost.

    Raises
    ------
    FileNotFoundError
        If the evaluation has no ``result.json``.
    ValueError
        If the stored result does not belong to the key.
    """
    path = forward.entry_dir(n_molecules, theta, seeds) / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"No result for N={n_molecules}, seeds={seeds}: {path}")
    key = forward.make_key(n_molecules, theta, seeds)
    d = CachedForwardModel._read_result(
        path, key, getattr(forward.model, "orthoboxy", None)
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cost = data.get("cpu_time_s")
    complete = bool(data.get("cost_complete", cost is not None)) and cost is not None
    return Evaluation(d * D_SCALE, None if cost is None else float(cost), complete)


def mean_cost(evals: Sequence[Evaluation], label: str) -> tuple[float, int]:
    """Return the mean cost over the evaluations with complete cost.

    Parameters
    ----------
    evals : Sequence[Evaluation]
        Evaluations of one model.
    label : str
        Model description for the error message.

    Returns
    -------
    w : float
        Mean cost in core-seconds.
    n_excluded : int
        Number of evaluations without complete cost (excluded from ``w``).

    Raises
    ------
    ValueError
        If no evaluation has a complete cost.
    """
    ok = [e.cost for e in evals if e.cost_complete and e.cost is not None]
    if not ok:
        raise ValueError(
            f"{label}: all {len(evals)} evaluations have no complete cost, cannot "
            "determine w."
        )
    return float(np.mean(ok)), len(evals) - len(ok)


def report_exclusions(
    phase: str,
    molecules: Sequence[int],
    n_excluded: Sequence[int],
    n_evals: Sequence[int],
) -> None:
    """Print the number of evaluations excluded from ``w`` per model.

    Parameters
    ----------
    phase : str
        Phase name for the heading.
    molecules : Sequence[int]
        Molecules of the models.
    n_excluded : Sequence[int]
        Excluded evaluations per model.
    n_evals : Sequence[int]
        Evaluations per model.
    """
    print(f"Evaluations without complete cost ({phase}):")
    for k, (n, ex, m) in enumerate(zip(molecules, n_excluded, n_evals), start=1):
        print(
            f"  model_{k} (N={n}): {ex} of {m} evaluations without complete cost, "
            "excluded from w"
        )


@dataclass
class Calibration:
    """Result of the calibration phase (or a paper-I calibration file).

    Parameters
    ----------
    molecules : tuple[int, ...]
        Molecules per model, descending (index 0 = high fidelity).
    D : np.ndarray
        ``(eta, m)`` diffusion coefficients in 1e-12 m^2/s, sample ``j`` shared by all.
    w : np.ndarray
        Cost per evaluation in core-seconds.
    n_cost_excluded : np.ndarray
        Evaluations excluded from ``w`` per model.
    mean, std, rho : np.ndarray
        Statistics per model (ddof = 0); ``rho[0] = 1``.
    samples : Samples | None
        The samples (None if the source file has none).
    source : str
        Description of where the data come from.
    cost_all : np.ndarray | None
        ``(eta, m)`` cost per evaluation (NaN if unknown), only for own results.
    cost_complete : np.ndarray | None
        ``(eta, m)`` bool, only for own results.
    """

    molecules: tuple[int, ...]
    D: FloatArray
    w: FloatArray
    n_cost_excluded: npt.NDArray[np.int64]
    mean: FloatArray
    std: FloatArray
    rho: FloatArray
    samples: Samples | None
    source: str
    cost_all: FloatArray | None = None
    cost_complete: npt.NDArray[np.bool_] | None = None

    @property
    def n_evals(self) -> int:
        """Return the number of calibration evaluations per model."""
        return int(self.D.shape[1])


def build_calibration(
    molecules: Sequence[int],
    evaluations: Sequence[Sequence[Evaluation]],
    samples: Samples,
    source: str = "computed",
) -> Calibration:
    """Compute the calibration statistics from the evaluations.

    Parameters
    ----------
    molecules : Sequence[int]
        Molecules per model, descending.
    evaluations : Sequence[Sequence[Evaluation]]
        Per model the ``m`` evaluations in sample order.
    samples : Samples
        The samples.
    source : str, optional
        Description for the output.

    Returns
    -------
    Calibration
        Statistics (:func:`mfwater.mfmc_core.model_statistics`) and costs
        (:func:`mean_cost`).
    """
    d = np.array([[e.d for e in row] for row in evaluations], dtype=np.float64)
    mean, std, rho = mfmc_core.model_statistics(d)
    w = np.empty(len(molecules))
    excl = np.empty(len(molecules), dtype=np.int64)
    for i, row in enumerate(evaluations):
        w[i], excl[i] = mean_cost(row, f"model_{i + 1} (N={molecules[i]})")
    cost_all = np.array(
        [[np.nan if e.cost is None else e.cost for e in row] for row in evaluations]
    )
    complete = np.array([[e.cost_complete for e in row] for row in evaluations])
    report_exclusions(
        "calibration", molecules, excl.tolist(), [len(r) for r in evaluations]
    )
    return Calibration(
        tuple(int(n) for n in molecules),
        d,
        w,
        excl,
        mean,
        std,
        rho,
        samples,
        source,
        cost_all,
        complete,
    )


def read_calibration(path: Path) -> Calibration:
    """Read a calibration file (own or paper-I format).

    Models are ordered by decreasing ``n_molecules``. ``mean``, ``std``,
    ``correlation`` and ``computation_time`` are taken from the attributes (as
    stored); missing statistics are computed from ``diffusion_coeff``.

    Parameters
    ----------
    path : Path
        HDF5 file.

    Returns
    -------
    Calibration
        The calibration data; ``samples`` is None if the file has no
        ``lj_params``/``seeds``.

    Raises
    ------
    RuntimeError
        If the file has no models, a model has no ``computation_time`` or the
        models have different numbers of evaluations.
    """
    with h5py.File(path, "r") as f:
        items = model_groups(f["models"])
        d = [np.asarray(g["diffusion_coeff"], dtype=np.float64) for _, g in items]
        if len({x.shape for x in d}) != 1:
            raise RuntimeError(f"{path}: models have different numbers of evaluations.")
        molecules = tuple(int(g.attrs["n_molecules"]) for _, g in items)
        for name, g in items:
            if "computation_time" not in g.attrs:
                raise RuntimeError(f"{path}: {name} has no 'computation_time'.")
        w = np.array([float(g.attrs["computation_time"]) for _, g in items])
        excl = np.array(
            [int(g.attrs.get("n_cost_excluded", 0)) for _, g in items], dtype=np.int64
        )
        stats = [
            all(a in g.attrs for _, g in items) for a in ("mean", "std", "correlation")
        ]
        d_arr = np.array(d)
        if all(stats):
            mean = np.array([float(g.attrs["mean"]) for _, g in items])
            std = np.array([float(g.attrs["std"]) for _, g in items])
            rho = np.array([float(g.attrs["correlation"]) for _, g in items])
        else:
            mean, std, rho = mfmc_core.model_statistics(d_arr)
        has_samples = "lj_params" in f["models"] and "seeds" in f["models"]
    samples = read_samples(path) if has_samples else None
    return Calibration(
        molecules, d_arr, w, excl, mean, std, rho, samples, f"file {path}"
    )


def write_calibration(f: h5py.File, cal: Calibration) -> None:
    """Write a calibration in paper-I format to an open new file.

    Parameters
    ----------
    f : h5py.File
        File opened for writing.
    cal : Calibration
        Calibration data (``samples`` must not be None).
    """
    assert cal.samples is not None
    models = f.create_group("models")
    models.attrs["n_models"] = len(cal.molecules)
    models.attrs["diffusion_unit"] = D_UNIT
    models.attrs["cost_unit"] = COST_UNIT
    models.create_dataset("lj_params", data=cal.samples.lj_params)
    models.create_dataset("charges", data=cal.samples.charges)
    models.create_dataset("seeds", data=cal.samples.seeds)
    for i, n in enumerate(cal.molecules):
        g = models.create_group(f"model_{i + 1}")
        g.attrs["n_molecules"] = n
        g.attrs["n_evals"] = cal.n_evals
        g.attrs["mean"] = cal.mean[i]
        g.attrs["std"] = cal.std[i]
        g.attrs["correlation"] = cal.rho[i]
        g.attrs["computation_time"] = cal.w[i]
        g.attrs["n_cost_excluded"] = int(cal.n_cost_excluded[i])
        g.create_dataset("diffusion_coeff", data=cal.D[i])
        if cal.cost_all is not None and cal.cost_complete is not None:
            g.create_dataset("computation_time_all", data=cal.cost_all[i])
            g.create_dataset("cost_complete", data=cal.cost_complete[i])
    f.attrs["last_algo"] = "mfmc-run"


@dataclass
class PerformanceResult:
    """Result of the performance phase for one budget.

    Parameters
    ----------
    budget : float
        Budget in core-seconds.
    selected : tuple[int, ...]
        Indices into ``Calibration.molecules`` of the models used, ordered by
        decreasing ``rho**2``.
    m_star : np.ndarray
        Evaluations per selected model (non-decreasing).
    unspent : float
        Budget left by rounding down (allocation, calibration costs).
    allocation_source : str
        ``computed`` or ``oms`` (taken from a replayed performance file).
    samples : Samples
        Samples of the performance phase (at least ``m_star[-1]``).
    D : list[np.ndarray]
        Per selected model its ``m_i`` diffusion coefficients (1e-12 m^2/s).
    w_performance : np.ndarray
        Mean costs measured in this phase (complete costs only).
    n_cost_excluded : np.ndarray
        Evaluations excluded from ``w_performance`` per selected model.
    alpha : np.ndarray
        Control-variate coefficients.
    estimator : float
        MFMC estimator (Eq. 19).
    errors : dict[str, float]
        RMSE (Eq. 20) per cost basis.
    gammas : dict[str, float]
        gamma (Eq. 22) per cost basis.
    rmse_cost : str
        Chosen basis (``mfmc_error`` and ``gamma``).
    """

    budget: float
    selected: tuple[int, ...]
    m_star: npt.NDArray[np.int64]
    unspent: float
    allocation_source: str
    samples: Samples
    D: list[FloatArray]
    w_performance: FloatArray
    n_cost_excluded: npt.NDArray[np.int64]
    alpha: FloatArray
    estimator: float
    errors: dict[str, float]
    gammas: dict[str, float]
    rmse_cost: str


def compute_performance(
    cal: Calibration,
    budget: float,
    selected: tuple[int, ...],
    m_star: npt.NDArray[np.int64],
    unspent: float,
    allocation_source: str,
    samples: Samples,
    evaluations: Sequence[Sequence[Evaluation]],
    rmse_cost: str,
) -> PerformanceResult:
    """Compute estimator, RMSE and gamma of one budget from its evaluations.

    The control variates use sigma and rho of the calibration in the order of
    ``selected`` (the same order for both, which fixes the legacy pairing bug).

    Parameters
    ----------
    cal : Calibration
        Calibration data.
    budget : float
        Budget in core-seconds.
    selected : tuple[int, ...]
        Indices into the calibration models, ordered by decreasing ``rho**2``.
    m_star : np.ndarray
        Evaluations per selected model.
    unspent : float
        Unspent budget of the allocation.
    allocation_source : str
        ``computed`` or ``oms``.
    samples : Samples
        Samples of this budget.
    evaluations : Sequence[Sequence[Evaluation]]
        Per selected model its ``m_i`` evaluations in sample order.
    rmse_cost : str
        ``calibration`` or ``performance``.

    Returns
    -------
    PerformanceResult
        All derived quantities.
    """
    idx = list(selected)
    rho = cal.rho[idx]
    sigma = cal.std[idx]
    alpha = mfmc_core.control_variates(rho, sigma)
    d = [np.array([e.d for e in row], dtype=np.float64) for row in evaluations]
    estimator = mfmc_core.mfmc_estimate(d, m_star, alpha)

    w_perf = np.empty(len(idx))
    excl = np.empty(len(idx), dtype=np.int64)
    for k, row in enumerate(evaluations):
        w_perf[k], excl[k] = mean_cost(
            row, f"model_{k + 1} (N={cal.molecules[idx[k]]}), budget {budget:g}"
        )
    report_exclusions(
        f"performance, budget {budget:g}",
        [cal.molecules[i] for i in idx],
        excl.tolist(),
        [len(r) for r in evaluations],
    )
    sigma1 = float(cal.std[0])
    errors: dict[str, float] = {}
    gammas: dict[str, float] = {}
    for basis, w in (("calibration", cal.w[idx]), ("performance", w_perf)):
        errors[basis] = mfmc_core.mfmc_rmse(sigma1, rho, w, budget)
        gammas[basis] = mfmc_core.gamma(rho, w)
    return PerformanceResult(
        budget,
        selected,
        m_star,
        unspent,
        allocation_source,
        samples,
        d,
        w_perf,
        excl,
        alpha,
        float(estimator),
        errors,
        gammas,
        rmse_cost,
    )


def write_performance(
    f: h5py.File, cal: Calibration, res: PerformanceResult, run_settings: dict[str, Any]
) -> None:
    """Write the result of one budget in paper-I format to an open new file.

    Parameters
    ----------
    f : h5py.File
        File opened for writing.
    cal : Calibration
        Calibration data (the ``*_initial`` entries).
    res : PerformanceResult
        Result of the budget.
    run_settings : dict
        Deterministic run settings, stored as JSON.
    """
    models = f.create_group("models")
    a = models.attrs
    a["budget"] = res.budget
    a["mfmc_estimator"] = res.estimator
    a["mfmc_error"] = res.errors[res.rmse_cost]
    a["gamma"] = res.gammas[res.rmse_cost]
    a["rmse_cost"] = res.rmse_cost
    for basis in ("calibration", "performance"):
        a[f"mfmc_error_{basis}_cost"] = res.errors[basis]
        a[f"gamma_{basis}_cost"] = res.gammas[basis]
    a["n_models"] = len(res.selected)
    # 1-based numbers of the calibration groups model_k (not indices into a name sort)
    a["selected_models"] = np.array([i + 1 for i in res.selected], dtype=np.int64)
    a["selected_molecules"] = np.array(
        [cal.molecules[i] for i in res.selected], dtype=np.int64
    )
    a["unspent_budget"] = res.unspent
    a["allocation_source"] = res.allocation_source
    a["diffusion_unit"] = D_UNIT
    a["cost_unit"] = COST_UNIT
    a["run_settings"] = json.dumps(run_settings, sort_keys=True)

    models.create_dataset("lj_params", data=res.samples.lj_params)
    models.create_dataset("charges", data=res.samples.charges)
    models.create_dataset("seeds", data=res.samples.seeds)
    if cal.samples is not None:
        models.create_dataset("lj_params_initial", data=cal.samples.lj_params)
        models.create_dataset("charges_initial", data=cal.samples.charges)
        models.create_dataset("seeds_initial", data=cal.samples.seeds)

    for k, i in enumerate(res.selected):
        g = models.create_group(f"model_{k + 1}")
        g.attrs["n_molecules"] = cal.molecules[i]
        g.attrs["calibration_model"] = i + 1
        g.attrs["n_evals"] = int(res.m_star[k])
        g.attrs["n_evals_initial"] = cal.n_evals
        g.attrs["alpha"] = res.alpha[k]
        g.attrs["correlation"] = cal.rho[i]
        g.attrs["mean"] = float(np.mean(res.D[k]))
        g.attrs["mean_initial"] = cal.mean[i]
        g.attrs["std_initial"] = cal.std[i]
        g.attrs["computation_time"] = res.w_performance[k]
        g.attrs["computation_time_initial"] = cal.w[i]
        g.attrs["n_cost_excluded"] = int(res.n_cost_excluded[k])
        g.attrs["n_cost_excluded_initial"] = int(cal.n_cost_excluded[i])
        g.create_dataset("diffusion_coeff", data=res.D[k])
        g.create_dataset("diffusion_coeff_initial", data=cal.D[i])
    f.attrs["last_algo"] = "mfmc-run"


def write_summary(
    f: h5py.File,
    budgets: Sequence[float],
    results: Sequence[PerformanceResult],
    labels: Sequence[str],
    run_settings: dict[str, Any],
) -> None:
    """Write the summary over all budgets to an open new file.

    Parameters
    ----------
    f : h5py.File
        File opened for writing.
    budgets : Sequence[float]
        Budgets in the order of the configuration.
    results : Sequence[PerformanceResult]
        Result per budget.
    labels : Sequence[str]
        File name labels of the budgets (``performance_<label>.hdf5``).
    run_settings : dict
        Deterministic run settings, stored as JSON.
    """
    f.create_dataset("budget", data=np.array(budgets))
    f.create_dataset("mfmc_estimator", data=np.array([r.estimator for r in results]))
    rmse = results[0].rmse_cost
    f.create_dataset("mfmc_error", data=np.array([r.errors[rmse] for r in results]))
    f.create_dataset("gamma", data=np.array([r.gammas[rmse] for r in results]))
    for basis in ("calibration", "performance"):
        f.create_dataset(
            f"mfmc_error_{basis}_cost",
            data=np.array([r.errors[basis] for r in results]),
        )
        f.create_dataset(
            f"gamma_{basis}_cost", data=np.array([r.gammas[basis] for r in results])
        )
    grp = f.create_group("allocation")
    for label, r in zip(labels, results):
        g = grp.create_group(f"budget_{label}")
        g.create_dataset("selected_models", data=np.array([i + 1 for i in r.selected]))
        g.create_dataset("n_evals", data=r.m_star)
    f.attrs["rmse_cost"] = rmse
    f.attrs["diffusion_unit"] = D_UNIT
    f.attrs["files"] = [f"performance_{label}.hdf5" for label in labels]
    f.attrs["run_settings"] = json.dumps(run_settings, sort_keys=True)
    f.attrs["last_algo"] = "mfmc-run"


# ---------------------------------------------------------------------------
# write once
# ---------------------------------------------------------------------------


def _differences(a: h5py.Group, b: h5py.Group, where: str = "/") -> list[str]:
    """Return the differences between two HDF5 groups (attributes and datasets).

    Parameters
    ----------
    a, b : h5py.Group
        Groups to compare.
    where : str, optional
        Path prefix for the messages.

    Returns
    -------
    list[str]
        One message per difference; empty if identical.
    """
    diffs: list[str] = []
    for key in sorted(set(a.attrs) | set(b.attrs)):
        if key not in a.attrs or key not in b.attrs:
            diffs.append(f"{where}@{key}: present in only one file")
        elif not np.array_equal(a.attrs[key], b.attrs[key], equal_nan=True):
            diffs.append(f"{where}@{key}: {a.attrs[key]!r} != {b.attrs[key]!r}")
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            diffs.append(f"{where}{key}: present in only one file")
            continue
        ia, ib = a[key], b[key]
        if isinstance(ia, h5py.Group) and isinstance(ib, h5py.Group):
            diffs += _differences(ia, ib, f"{where}{key}/")
        elif isinstance(ia, h5py.Dataset) and isinstance(ib, h5py.Dataset):
            if not np.array_equal(ia[()], ib[()], equal_nan=True):
                diffs.append(f"{where}{key}: data differ")
        else:
            diffs.append(f"{where}{key}: group/dataset mismatch")
    return diffs


def write_hdf5_once(path: Path, writer: Callable[[h5py.File], None]) -> bool:
    """Create an HDF5 file atomically; verify an identical existing one.

    The file is written under a temporary name and hard-linked to ``path``
    (``os.link`` fails if ``path`` exists), so a complete file appears at once and
    an existing file is never replaced. If another job created the file first, its
    content must equal ours; otherwise the run is inconsistent.

    Parameters
    ----------
    path : Path
        Target file.
    writer : Callable[[h5py.File], None]
        Fills a new file; must be deterministic.

    Returns
    -------
    bool
        True if this call created ``path``, False if an identical file existed.

    Raises
    ------
    RuntimeError
        If ``path`` exists with different content.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with h5py.File(tmp, "x") as f:
        writer(f)
    try:
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            with h5py.File(tmp, "r") as a, h5py.File(path, "r") as b:
                diffs = _differences(a, b)
            if diffs:
                raise RuntimeError(
                    f"{path} already exists with different content (written by another "
                    "job or an earlier run with different settings):\n  "
                    + "\n  ".join(diffs[:10])
                ) from None
            return False
    finally:
        tmp.unlink()  # only our own temporary file


def write_json_once(path: Path, data: dict[str, Any]) -> bool:
    """Create a JSON file atomically; verify an identical existing one.

    Parameters
    ----------
    path : Path
        Target file.
    data : dict
        JSON-serialisable content.

    Returns
    -------
    bool
        True if this call created the file, False if an identical one existed.

    Raises
    ------
    RuntimeError
        If ``path`` exists with different content.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            if existing != json.loads(json.dumps(data)):
                raise RuntimeError(
                    f"{path} already exists with different content: {existing} vs {data}."
                ) from None
            return False
    finally:
        tmp.unlink()
