"""The ``mfmc-run`` algorithm: the whole paper-I forward workflow in resumable jobs.

Run directory (``--workdir``)::

    manifest.json                 settings, seed, algorithm and package version, git hash
    config.toml, config_1.toml    copies of the ``-c`` file (changed content is kept)
    cache/N{n}/{key_hash}/        evaluation cache and MD output (see ``mfwater.md``)
    calibration.hdf5              phase 1: paper-I format calibration file
    selection.json                phase 1: selected models
    performance_<budget>.hdf5     phase 2: one paper-I format file per budget
    summary.hdf5                  all budgets

Phases (each idempotent; the work is done by the evaluation pool of
:mod:`mfwater.md.pool`, which several SLURM jobs may run on the same directory):

1. calibration: ``eta`` models x ``m`` evaluations at shared samples, then
   sigma, rho, w (SI Alg. 2) and the model selection (SI Alg. 3);
2. performance: for every budget the allocation (SI Alg. 4), fresh samples, the
   evaluations of all budgets in one pool, then estimator, RMSE and gamma.

A restart (resubmitting the job) re-enters the phases: evaluations with a
``result.json`` are never repeated, finished files are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .. import mfmc_core
from ..__version__ import __version__
from ..algo_mfda.run_manager import (
    _git_dirty,
    _git_hash,
    archive_config,
    write_manifest,
)
from ..md.cache import CachedForwardModel
from ..md.pipeline import MDForwardModel
from ..md.pool import PoolStatus, Task, run_pool
from .config import MFMCRunConfig, budget_label, build_config
from .dummy import CorrelatedDummyForwardModel
from .results import (
    Calibration,
    Evaluation,
    PerformanceResult,
    build_calibration,
    compute_performance,
    read_calibration,
    read_evaluation,
    write_calibration,
    write_hdf5_once,
    write_json_once,
    write_performance,
    write_summary,
)
from .sampling import Samples, draw_samples, read_samples

MANIFEST_NAME = "manifest.json"

#: Bump when a code change alters the results of a restart with identical settings.
ALGORITHM_VERSION = 1

#: Manifest entries that must be identical on a restart.
_STRICT_KEYS = (
    "algorithm",
    "algorithm_version",
    "forward_model",
    "orthoboxy",
    "protocol",
    "molecules",
    "evals",
    "budgets",
    "params",
    "seed",
    "rmse_cost",
    "calibration_from_sha256",
    "replay_from_sha256",
    "replay_performance_from_sha256",
    "replay_use_oms_allocation",
)
# Stored but never compared: total_cpus, ncpu_override, lammps_cmd, print_level,
# claim_timeout, stop_after (and the paths of replayed files, which are compared by
# their hash). ``ncpu_override`` changes the costs, not the physics.


def _msdiff_version() -> str | None:
    """Return the output of ``msdiff --version`` (None if unavailable).

    The msdiff version changes D (0.3.2 differs from the version of paper I), so it
    is recorded in the manifest.

    Returns
    -------
    str | None
        Stripped output, or None if msdiff is missing or fails.
    """
    try:
        out = subprocess.run(
            ["msdiff", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or out.stderr).strip() or None


def _sha256(path: Path | None) -> str | None:
    """Return the SHA-256 of a file (None for None).

    Parameters
    ----------
    path : Path | None
        File to hash.

    Returns
    -------
    str | None
        Hex digest.
    """
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_settings(cfg: MFMCRunConfig, seed: int) -> dict[str, Any]:
    """Collect the manifest settings of a run.

    Parameters
    ----------
    cfg : MFMCRunConfig
        Validated settings.
    seed : int
        Resolved seed.

    Returns
    -------
    dict
        JSON-serialisable manifest content.
    """
    return {
        "algorithm": "mfmc-run",
        "algorithm_version": ALGORITHM_VERSION,
        "forward_model": cfg.forward_model,
        "orthoboxy": cfg.orthoboxy,
        "protocol": cfg.protocol,
        "molecules": list(cfg.molecules),
        "evals": cfg.evals,
        "budgets": list(cfg.budgets),
        "params": cfg.params,
        "seed": int(seed),
        "rmse_cost": cfg.rmse_cost,
        "calibration_from_sha256": _sha256(cfg.calibration_from),
        "replay_from_sha256": _sha256(cfg.replay_from),
        "replay_performance_from_sha256": {
            budget_label(b): _sha256(p)
            for b, p in sorted(cfg.replay_performance_from.items())
        },
        "replay_use_oms_allocation": cfg.replay_use_oms_allocation,
        # stored only
        "calibration_from": (
            None if cfg.calibration_from is None else str(cfg.calibration_from)
        ),
        "replay_from": None if cfg.replay_from is None else str(cfg.replay_from),
        "replay_performance_from": {
            budget_label(b): str(p)
            for b, p in sorted(cfg.replay_performance_from.items())
        },
        "total_cpus": cfg.total_cpus,
        "ncpu_override": {str(k): v for k, v in sorted(cfg.ncpu_override.items())},
        "lammps_cmd": cfg.lammps_cmd,
        "print_level": cfg.print_level,
        "claim_timeout": cfg.claim_timeout,
        "stop_after": cfg.stop_after,
        "version": __version__,
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        # stored only; a change on restart prints a warning (like git_hash)
        "msdiff_version": _msdiff_version(),
    }


def run_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic subset of the settings stored in result files.

    Parameters
    ----------
    settings : dict
        Manifest settings.

    Returns
    -------
    dict
        The strict entries only (no git hash, so that jobs agree byte for byte).
    """
    return {k: settings[k] for k in _STRICT_KEYS}


def check_manifest(stored: dict[str, Any], settings: dict[str, Any]) -> None:
    """Compare the stored manifest with the current settings.

    Parameters
    ----------
    stored : dict
        Content of the existing manifest.
    settings : dict
        Settings of this invocation.

    Raises
    ------
    ValueError
        If the manifest is not one of ``mfmc-run`` or a strict entry differs.
    """
    if stored.get("algorithm") != "mfmc-run":
        raise ValueError(
            f"The manifest in this run directory is not from mfmc-run "
            f"(algorithm={stored.get('algorithm')!r}); use a new --workdir."
        )
    diffs = [
        f"  {k}: manifest has {stored.get(k)!r}, now {settings.get(k)!r}"
        for k in _STRICT_KEYS
        if stored.get(k) != settings.get(k)
    ]
    if diffs:
        hint = ""
        if stored.get("algorithm_version") != settings["algorithm_version"]:
            hint = " The algorithm changed since this run was started."
        raise ValueError(
            "Settings differ from the manifest of this run directory, refusing to "
            "restart (use a new --workdir for different settings)."
            + hint
            + "\n"
            + "\n".join(diffs)
        )
    for key in ("git_hash", "version", "msdiff_version"):
        old, new = stored.get(key), settings.get(key)
        if old is not None and new is not None and old != new:
            print(
                f"Warning: {key} differs from the manifest (manifest has {old!r}, now "
                f"{new!r}); the restart is only valid if the change does not affect "
                "the algorithm."
            )
    if settings.get("git_dirty"):
        print(
            "Warning: the package source has uncommitted changes; the restart may not "
            "replay the original run."
        )


def prepare_run(
    cfg: MFMCRunConfig, config_file: str | Path | None
) -> tuple[dict[str, Any], bool]:
    """Create the run directory and manifest, or check an existing manifest.

    Several jobs may start at the same time: the first to link the manifest wins,
    the others read it and compare.

    Parameters
    ----------
    cfg : MFMCRunConfig
        Validated settings.
    config_file : str | Path | None
        The ``-c`` file to archive.

    Returns
    -------
    settings : dict
        Settings of the run (with the seed of the manifest on a restart).
    restart : bool
        True if a matching manifest existed.

    Raises
    ------
    ValueError
        If the settings differ from an existing manifest.
    """
    workdir = cfg.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = workdir / MANIFEST_NAME
    for _ in range(2):
        if manifest.exists():
            with open(manifest, encoding="utf-8") as f:
                stored = json.load(f)
            seed = cfg.seed if cfg.seed is not None else stored.get("seed")
            settings = build_settings(cfg, int(seed))
            check_manifest(stored, settings)
            archive_config(workdir, config_file)
            return settings, True
        seed = (
            cfg.seed
            if cfg.seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )
        settings = build_settings(cfg, int(seed))
        try:
            write_manifest(manifest, settings)
        except FileExistsError:
            continue  # another job was faster: read its manifest
        archive_config(workdir, config_file)
        return settings, False
    raise RuntimeError(f"Cannot create or read {manifest}.")  # pragma: no cover


def make_forward(cfg: MFMCRunConfig, cache_dir: Path) -> CachedForwardModel:
    """Create the cached forward model of the run.

    Parameters
    ----------
    cfg : MFMCRunConfig
        Settings.
    cache_dir : Path
        Cache and MD output directory.

    Returns
    -------
    CachedForwardModel
        MD forward model with ``require_cost=True`` (the cost is needed for ``w``)
        and the ``ncpu`` override, or the correlated dummy.
    """
    model: Any
    if cfg.forward_model == "md":
        model = MDForwardModel(
            cache_dir,
            orthoboxy=cfg.orthoboxy,
            lammps_cmd=cfg.lammps_cmd,
            protocol=cfg.protocol,
            require_cost=True,
            ncpu=dict(cfg.ncpu_override) or None,
        )
    else:
        model = CorrelatedDummyForwardModel(
            protocol=cfg.protocol, orthoboxy=cfg.orthoboxy, ncpu=cfg.ncpu_override
        )
    return CachedForwardModel(model, cache_dir)


def _run_pool(
    tasks: list[Task], forward: CachedForwardModel, cfg: MFMCRunConfig, label: str
) -> bool:
    """Run a pool and report the outcome; return whether all tasks are done.

    Parameters
    ----------
    tasks : list[Task]
        Evaluations of the phase.
    forward : CachedForwardModel
        Cached forward model.
    cfg : MFMCRunConfig
        Settings.
    label : str
        Phase name for messages.

    Returns
    -------
    bool
        True if every task has a result; False if tasks are claimed by other jobs.

    Raises
    ------
    RuntimeError
        If an evaluation failed (finished results and the cache are kept).
    """
    res = run_pool(tasks, forward, cfg.total_cpus, stale_timeout=cfg.claim_timeout)
    if res.status is PoolStatus.FAILED:
        raise RuntimeError(
            f"{label}: {res.n_failed} evaluation(s) failed. Finished evaluations are "
            "kept; fix the cause (tracebacks in failure_*.json next to the claim files "
            "in the evaluation directories) and resubmit the same job to continue.\n"
            + "\n".join(res.failures)
        )
    if res.status is PoolStatus.WAITING_ON_OTHERS:
        print(
            f"\n{label}: {res.n_waiting} evaluation(s) are still being computed by "
            "other job(s) (live claims in "
            f"{forward.cache_dir}). This job has nothing left to claim and exits "
            "with code 0.\n"
            "  - The job that finishes the last evaluation of this phase continues "
            "with the next phase.\n"
            "  - If no other job is running any more (check `squeue`), resubmit this "
            f"script: claims without heartbeat for {cfg.claim_timeout:g} s are stale and "
            "get taken over (interrupted MD runs resume from their restart files)."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# phase 1
# ---------------------------------------------------------------------------


def calibration_samples(cfg: MFMCRunConfig, seed: int) -> Samples:
    """Return the shared samples of the calibration.

    Parameters
    ----------
    cfg : MFMCRunConfig
        Settings.
    seed : int
        Run seed.

    Returns
    -------
    Samples
        Replayed from ``replay_from`` or drawn from ``default_rng([seed, 0])``.

    Raises
    ------
    ValueError
        If the replay file has fewer samples than ``evals``.
    """
    if cfg.replay_from is not None:
        samples = read_samples(cfg.replay_from)
        print(f"Calibration samples replayed from {cfg.replay_from} ({samples.n}).")
        return samples.head(cfg.evals)
    return draw_samples(cfg.evals, cfg.params, np.random.default_rng([seed, 0]))


def run_calibration(
    cfg: MFMCRunConfig, settings: dict[str, Any], forward: CachedForwardModel
) -> Calibration | None:
    """Phase 1: obtain the calibration (from a file, a finished run or by computing it).

    Parameters
    ----------
    cfg : MFMCRunConfig
        Settings.
    settings : dict
        Manifest settings (seed).
    forward : CachedForwardModel
        Cached forward model.

    Returns
    -------
    Calibration | None
        The calibration, or None if evaluations are still computed by other jobs.
    """
    if cfg.calibration_from is not None:
        print(f"Phase 1 skipped: calibration read from {cfg.calibration_from}.")
        return read_calibration(cfg.calibration_from)
    path = cfg.workdir / "calibration.hdf5"
    if path.exists():
        print(f"Phase 1: {path} exists, calibration read from it.")
        return read_calibration(path)

    samples = calibration_samples(cfg, settings["seed"])
    tasks = [
        Task(n, samples.theta(j), samples.seed_pair(j))
        for n in cfg.molecules
        for j in range(cfg.evals)
    ]
    print(
        f"Phase 1: calibration, {len(cfg.molecules)} models x {cfg.evals} evaluations "
        f"= {len(tasks)} MD runs on {cfg.total_cpus} cores."
    )
    if not _run_pool(tasks, forward, cfg, "Calibration"):
        return None
    evaluations = [
        [
            read_evaluation(forward, n, samples.theta(j), samples.seed_pair(j))
            for j in range(cfg.evals)
        ]
        for n in cfg.molecules
    ]
    cal = build_calibration(cfg.molecules, evaluations, samples)
    created = write_hdf5_once(path, lambda f: write_calibration(f, cal))
    print(f"{'Wrote' if created else 'Verified existing'} {path}.")
    return cal


def select(cal: Calibration, cfg: MFMCRunConfig) -> tuple[tuple[int, ...], float]:
    """Select the models and write ``selection.json`` (once).

    Parameters
    ----------
    cal : Calibration
        Calibration data.
    cfg : MFMCRunConfig
        Settings (run directory).

    Returns
    -------
    selected : tuple[int, ...]
        Indices into the calibration models, ordered by decreasing ``rho**2``.
    v_star : float
        MSE of the selected set at a budget of one high-fidelity evaluation.
    """
    selected, v_star = mfmc_core.select_models(cal.std[0], cal.rho, cal.w)
    write_json_once(
        cfg.workdir / "selection.json",
        {
            "selected_index": list(selected),
            "selected_models": [i + 1 for i in selected],
            "selected_molecules": [cal.molecules[i] for i in selected],
            "molecules": list(cal.molecules),
            "rho": cal.rho.tolist(),
            "std": cal.std.tolist(),
            "w": cal.w.tolist(),
            "v_star": v_star,
        },
    )
    return selected, v_star


# ---------------------------------------------------------------------------
# phase 2
# ---------------------------------------------------------------------------


@dataclass
class BudgetPlan:
    """Allocation and samples of one budget.

    Parameters
    ----------
    index : int
        Position of the budget in the configuration (part of the sampling seed).
    budget : float
        Budget in core-seconds.
    selected : tuple[int, ...]
        Indices into the calibration models, ordered by decreasing ``rho**2``.
    m_star : np.ndarray
        Evaluations per selected model.
    unspent : float
        Budget left by rounding down.
    source : str
        ``computed`` or ``oms``.
    samples : Samples
        Samples of this budget (at least ``m_star[-1]``).
    """

    index: int
    budget: float
    selected: tuple[int, ...]
    m_star: np.ndarray
    unspent: float
    source: str
    samples: Samples


def _read_oms(
    path: Path, cal: Calibration
) -> tuple[tuple[int, ...], np.ndarray, Samples]:
    """Read models, ``n_evals`` and samples of a paper-I performance file.

    Parameters
    ----------
    path : Path
        Performance (``oms``) file.
    cal : Calibration
        Calibration, to map the models by ``n_molecules``.

    Returns
    -------
    selected : tuple[int, ...]
        Indices into the calibration models, in the file's model order.
    m_star : np.ndarray
        ``n_evals`` per model.
    samples : Samples
        Stored samples.

    Raises
    ------
    ValueError
        If a model of the file is not in the calibration.
    """
    with h5py.File(path, "r") as f:
        # model_k order of the file = decreasing rho**2, not name or size order
        items = sorted(
            (g for _, g in f["models"].items() if isinstance(g, h5py.Group)),
            key=lambda g: int(str(g.name).rsplit("_", 1)[1]),
        )
        sel = []
        m = []
        for g in items:
            n = int(g.attrs["n_molecules"])
            if n not in cal.molecules:
                raise ValueError(f"{path}: N={n} is not a calibration model.")
            sel.append(cal.molecules.index(n))
            m.append(int(g.attrs["n_evals"]))
    return tuple(sel), np.array(m, dtype=np.int64), read_samples(path)


def plan_budget(
    cfg: MFMCRunConfig,
    cal: Calibration,
    selected: tuple[int, ...],
    index: int,
    seed: int,
) -> BudgetPlan:
    """Allocate one budget and prepare its samples.

    Parameters
    ----------
    cfg : MFMCRunConfig
        Settings.
    cal : Calibration
        Calibration data.
    selected : tuple[int, ...]
        Selected models (indices, rho order).
    index : int
        Position of the budget in ``cfg.budgets``.
    seed : int
        Run seed; the samples come from ``default_rng([seed, 1, index])``.

    Returns
    -------
    BudgetPlan
        Allocation and samples.

    Raises
    ------
    ValueError
        If the budget is too small, or a replayed performance file disagrees with
        the computed allocation without ``replay_use_oms_allocation``.
    """
    budget = cfg.budgets[index]
    idx = list(selected)
    computed: tuple[np.ndarray, float] | None
    try:
        computed = mfmc_core.allocate(cal.rho[idx], cal.w[idx], budget)
    except ValueError as exc:
        computed = None
        alloc_error = str(exc)

    oms_path = cfg.replay_performance_from.get(budget)
    if oms_path is not None:
        o_sel, o_m, o_samples = _read_oms(oms_path, cal)
        same = (
            computed is not None
            and o_sel == selected
            and np.array_equal(o_m, computed[0])
        )
        if same:
            m_star, unspent, source = computed[0], computed[1], "computed"  # type: ignore[index]
        else:
            shown = (
                f"computed: models N={[cal.molecules[i] for i in selected]}, "
                f"n_evals={computed[0].tolist() if computed else alloc_error}"
            )
            replayed = (
                f"{oms_path}: models N={[cal.molecules[i] for i in o_sel]}, "
                f"n_evals={o_m.tolist()}"
            )
            if not cfg.replay_use_oms_allocation:
                raise ValueError(
                    f"Budget {budget:g}: the replayed performance file differs from the "
                    f"allocation computed from this calibration.\n  {shown}\n  "
                    f"{replayed}\nTo evaluate the samples of the file with its own models "
                    "and n_evals, set replay_use_oms_allocation = true "
                    "(--replay-use-oms-allocation)."
                )
            print(
                f"Budget {budget:g}: using models and n_evals of {oms_path} instead of "
                f"the computed allocation.\n  {shown}\n  {replayed}"
            )
            m_star = o_m
            unspent = budget - float(np.sum(o_m * cal.w[list(o_sel)]))
            selected, source = o_sel, "oms"
        return BudgetPlan(
            index,
            budget,
            selected,
            m_star,
            unspent,
            source,
            o_samples.head(int(m_star[-1])),
        )

    if computed is None:
        raise ValueError(f"Budget {budget:g}: {alloc_error}")
    m_star, unspent = computed
    samples = draw_samples(
        int(m_star[-1]), cfg.params, np.random.default_rng([seed, 1, index])
    )
    return BudgetPlan(index, budget, selected, m_star, unspent, "computed", samples)


def print_calibration_table(
    cal: Calibration,
    selected: tuple[int, ...],
    v_star: float,
    plans: list[BudgetPlan],
    errors: dict[float, str],
) -> None:
    """Print models, statistics, selection, gamma and the allocation per budget.

    Parameters
    ----------
    cal : Calibration
        Calibration data.
    selected : tuple[int, ...]
        Selected models.
    v_star : float
        MSE of the selection at ``p = w_1``.
    plans : list[BudgetPlan]
        Allocations that could be made.
    errors : dict[float, str]
        Per budget the reason why no allocation could be made.
    """
    print(f"\nCalibration ({cal.source}): D in 1e-12 m^2/s, w in core-s")
    print(
        f"{'Model':<8}{'N':>7}{'rho':>11}{'sigma':>12}{'mean':>12}{'w':>12}{'excl.':>7}  selected"
    )
    print("-" * 78)
    for i, n in enumerate(cal.molecules):
        mark = f"yes (#{selected.index(i) + 1} by rho^2)" if i in selected else "no"
        print(
            f"model_{i + 1:<2}{n:7d}{cal.rho[i]:11.6f}{cal.std[i]:12.4f}"
            f"{cal.mean[i]:12.4f}{cal.w[i]:12.5g}{int(cal.n_cost_excluded[i]):7d}  {mark}"
        )
    idx = list(selected)
    gam = mfmc_core.gamma(cal.rho[idx], cal.w[idx])
    print(
        f"\nSelected models: {[i + 1 for i in selected]} "
        f"(N = {[cal.molecules[i] for i in selected]}); "
        f"gamma (calibration costs) = {gam:.4f}; v* = {v_star:.4g}"
    )
    for plan in plans:
        rmse = mfmc_core.mfmc_rmse(cal.std[0], cal.rho[idx], cal.w[idx], plan.budget)
        print(
            f"  budget {plan.budget:g}: m* = {plan.m_star.tolist()} "
            f"({plan.source}), unspent {plan.unspent:.1f}, expected RMSE {rmse:.4f}"
        )
    for b, msg in errors.items():
        print(f"  budget {b:g}: NO ALLOCATION - {msg}")


def run_performance(
    cfg: MFMCRunConfig,
    settings: dict[str, Any],
    forward: CachedForwardModel,
    cal: Calibration,
    plans: list[BudgetPlan],
) -> bool:
    """Phase 2: evaluate all budgets in one pool and write the result files.

    Parameters
    ----------
    cfg : MFMCRunConfig
        Settings.
    settings : dict
        Manifest settings.
    forward : CachedForwardModel
        Cached forward model.
    cal : Calibration
        Calibration data.
    plans : list[BudgetPlan]
        One plan per budget.

    Returns
    -------
    bool
        True if the phase finished (all files written), False if waiting on others.
    """
    tasks = [
        Task(cal.molecules[i], plan.samples.theta(j), plan.samples.seed_pair(j))
        for plan in plans
        for k, i in enumerate(plan.selected)
        for j in range(int(plan.m_star[k]))
    ]
    print(
        f"Phase 2: performance, {len(plans)} budget(s), {len(tasks)} MD runs on "
        f"{cfg.total_cpus} cores."
    )
    if not _run_pool(tasks, forward, cfg, "Performance"):
        return False

    rs = run_settings(settings)
    results: list[PerformanceResult] = []
    labels: list[str] = []
    for plan in plans:
        evaluations: list[list[Evaluation]] = [
            [
                read_evaluation(
                    forward,
                    cal.molecules[i],
                    plan.samples.theta(j),
                    plan.samples.seed_pair(j),
                )
                for j in range(int(plan.m_star[k]))
            ]
            for k, i in enumerate(plan.selected)
        ]
        res = compute_performance(
            cal,
            plan.budget,
            plan.selected,
            plan.m_star,
            plan.unspent,
            plan.source,
            plan.samples,
            evaluations,
            cfg.rmse_cost,
        )
        label = budget_label(plan.budget)
        path = cfg.workdir / f"performance_{label}.hdf5"
        created = write_hdf5_once(
            path, partial(write_performance, cal=cal, res=res, run_settings=rs)
        )
        print(f"{'Wrote' if created else 'Verified existing'} {path}.")
        results.append(res)
        labels.append(label)

    summary = cfg.workdir / "summary.hdf5"
    created = write_hdf5_once(
        summary,
        lambda f: write_summary(f, [r.budget for r in results], results, labels, rs),
    )
    print(f"{'Wrote' if created else 'Verified existing'} {summary}.")
    _print_results(cal, results, cfg.rmse_cost)
    return True


def _print_results(
    cal: Calibration, results: list[PerformanceResult], rmse_cost: str
) -> None:
    """Print the estimator, RMSE and gamma of every budget.

    Parameters
    ----------
    cal : Calibration
        Calibration data.
    results : list[PerformanceResult]
        Result per budget.
    rmse_cost : str
        Chosen cost basis.
    """
    print(f"\nMFMC results (D in 1e-12 m^2/s; RMSE and gamma with {rmse_cost} costs):")
    print(
        f"{'Budget':>12}  {'Estimator':>12}  {'RMSE':>10}  {'gamma':>8}  models / n_evals"
    )
    for r in results:
        models = ", ".join(
            f"N={cal.molecules[i]}:{int(m)}" for i, m in zip(r.selected, r.m_star)
        )
        print(
            f"{r.budget:12g}  {r.estimator:12.4f}  {r.errors[rmse_cost]:10.4f}  "
            f"{r.gammas[rmse_cost]:8.4f}  {models}"
        )


def mfmc_run(args: argparse.Namespace) -> int:
    """Run (or continue) the MFMC forward workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (see ``mfwater -a mfmc-run --help``).

    Returns
    -------
    int
        ``0`` on success, also when the job leaves the remaining work to other jobs
        or stops after the calibration (``stop_after``).

    Raises
    ------
    ValueError
        If the settings are invalid, differ from the manifest, or a budget is too
        small.
    RuntimeError
        If an evaluation failed.
    """
    cfg = build_config(args)
    workdir = cfg.workdir
    summary = workdir / "summary.hdf5"
    settings, restart = prepare_run(cfg, args.config)
    print(f"Random seed: {settings['seed']}")
    print(
        f"{'Restarting' if restart else 'Starting'} mfmc-run in {workdir}: "
        f"forward model {cfg.forward_model}, protocol {cfg.protocol}, "
        f"{cfg.total_cpus} cores, ranks per model "
        f"{ {n: cfg.ranks(n) for n in cfg.molecules} }."
    )
    if cfg.forward_model == "dummy":
        print(
            "Note: forward model 'dummy' (no MD); use forward_model = \"md\" for real runs."
        )

    complete = summary.exists() and all(
        (workdir / f"performance_{budget_label(b)}.hdf5").exists() for b in cfg.budgets
    )
    if complete:
        print(f"Run complete: {summary} exists. Nothing to do.")
        return 0

    forward = make_forward(cfg, workdir / "cache")
    cal = run_calibration(cfg, settings, forward)
    if cal is None:
        return 0
    selected, v_star = select(cal, cfg)

    plans: list[BudgetPlan] = []
    errors: dict[float, str] = {}
    for b_index, budget in enumerate(cfg.budgets):
        try:
            plans.append(plan_budget(cfg, cal, selected, b_index, settings["seed"]))
        except ValueError as exc:
            if cfg.stop_after is None:
                raise
            errors[budget] = str(exc)
    # selection may change by "oms" allocations only per budget; the table shows the computed one
    print_calibration_table(cal, selected, v_star, plans, errors)

    if cfg.stop_after == "calibration":
        print(
            f"\nStopped after the calibration (stop_after = 'calibration'). Files: "
            f"{workdir / 'calibration.hdf5'}, {workdir / 'selection.json'}.\n"
            "To continue with the performance phase, resubmit the same job with "
            "stop_after removed (it is not compared with the manifest). Budgets "
            "listed as NO ALLOCATION must be raised first (budgets are part of the "
            "manifest: use a new --workdir, the calibration can be reused with "
            "calibration_from = '<workdir>/calibration.hdf5')."
        )
        return 0

    run_performance(cfg, settings, forward, cal, plans)
    return 0
