"""Run management for MFDA chains: manifest, parallel chains, restart, smoke test.

Run directory layout (``--workdir``)::

    manifest.json                 arguments, seed, algorithm and package version, git hash
    cache/chain_{c:03d}/          evaluation cache of chain c (and MD output)
        N{n}/{key_hash}/result.json
        N{n}/{key_hash}/attempt_{k}/{siminp,simout,msd}/
    chain_{c:03d}.hdf5            result of chain c
    <-o file>                     combined summary (default ``default.hdf5``)

Restart works by deterministic replay: a chain is rerun from the beginning with
the same seed, and every evaluation already in the cache returns instantly. The
manifest guards against replaying with different settings. Nothing in the run
directory is ever overwritten or deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..__version__ import __version__
from ..algo_chemical_model.chemical_model import calc_cpus
from ..argparser import constants
from .cache import CachedForwardModel
from .forward_model import DummyForwardModel, ForwardModel
from .markov_chain import (
    as_int_list,
    build_config,
    make_config,
    resolve_subchain_lengths,
    run_chain,
    write_results,
)
from .md_pipeline import MDForwardModel
from .multifidelity_mcmc import MFDAConfig

MANIFEST_NAME = "manifest.json"

#: Version of the algorithm as far as results are concerned. Bump it whenever a
#: change alters the results of a replay with identical settings (a restart
#: would then silently mix old cached evaluations with new chain logic).
#: 1 = initial scheme, 2 = burn-in added to the chain length.
ALGORITHM_VERSION = 2

#: Manifest entries that must be identical for a restart (error otherwise).
_STRICT_KEYS = (
    "algorithm_version",
    "forward_model",
    "orthoboxy",
    "molecules",
    "n_models",
    "subchain_lengths",
    "chain_length",
    "burnin",
    "params",
    "n_chains",
    "seed",
)

#: Entries whose change only prints a warning on a restart (see :func:`check_manifest`).
_WARN_KEYS = ("git_hash", "git_dirty", "version")

#: Stored for the record but never compared: ``lammps_cmd`` may change between
#: restarts (e.g. ``mpirun`` -> ``srun``), ``print_level`` does not affect results.
_STORED_ONLY_KEYS = ("lammps_cmd", "print_level")


def make_inner_model(settings: dict[str, Any], cache_dir: Path) -> ForwardModel:
    """Create the (uncached) forward model selected on the command line.

    Parameters
    ----------
    settings : dict
        Manifest settings (see :func:`build_settings`).
    cache_dir : Path
        Cache directory of the chain; MD attempts are written below it.

    Returns
    -------
    ForwardModel
        The forward model.

    Raises
    ------
    ValueError
        If the forward model name is unknown.
    """
    if settings["forward_model"] == "dummy":
        return DummyForwardModel()
    if settings["forward_model"] == "md":
        return MDForwardModel(
            cache_dir,
            orthoboxy=settings["orthoboxy"],
            lammps_cmd=settings["lammps_cmd"],
        )
    raise ValueError(f"Unknown forward model '{settings['forward_model']}'.")


def _git_hash() -> str | None:
    """Return the git hash of the package source, or ``None`` if unavailable.

    Returns
    -------
    str or None
        Output of ``git rev-parse HEAD`` in the package directory.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # no git or not a checkout (e.g. pip install): nothing to compare
        return None
    return out.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Return whether the package source has uncommitted changes.

    Returns
    -------
    bool or None
        ``True`` if ``git status --porcelain`` in the package directory is
        non-empty, ``None`` if git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def build_settings(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Collect all settings of a run.

    ``lammps_cmd`` and ``print_level`` are stored but not compared on a restart
    (see ``_STORED_ONLY_KEYS``). The number of levels is
    ``len(--molecules)``; ``--models`` is ignored.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    seed : int
        Resolved seed entropy.

    Returns
    -------
    dict
        JSON-serialisable settings, the content of the manifest.
    """
    molecules = as_int_list(args.n_molecules)
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "forward_model": args.forward_model,
        "orthoboxy": bool(args.orthoboxy),
        "lammps_cmd": args.lammps_cmd,
        "molecules": molecules,
        "n_models": len(molecules),
        "subchain_lengths": resolve_subchain_lengths(
            args.n_mc_subchain_lengths, len(molecules)
        ),
        "chain_length": int(args.n_mc_chain_length),
        "burnin": int(args.n_mc_burnin),
        "params": args.params,
        "n_chains": int(args.n_chains),
        "seed": int(seed),
        "version": __version__,
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "print_level": int(getattr(args, "print_level", 0)),
    }


def config_from_settings(settings: dict[str, Any]) -> MFDAConfig:
    """Build the algorithm settings from the manifest settings.

    Parameters
    ----------
    settings : dict
        Manifest settings (see :func:`build_settings`).

    Returns
    -------
    MFDAConfig
        Algorithm settings.
    """
    return make_config(
        settings["molecules"],
        settings["subchain_lengths"],
        settings["params"],
        settings["chain_length"],
        settings["burnin"],
        print_level=settings["print_level"],
    )


def write_manifest(path: Path, settings: dict[str, Any]) -> None:
    """Write the manifest; an existing manifest is never overwritten.

    Parameters
    ----------
    path : Path
        Manifest file.
    settings : dict
        Settings to store.

    Raises
    ------
    FileExistsError
        If ``path`` exists.
    """
    # Hard link of a temporary file: atomic, and fails if the target exists.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        # only our own temporary file is removed
        tmp.unlink()


def check_manifest(stored: dict[str, Any], settings: dict[str, Any]) -> None:
    """Compare the stored manifest with the current settings.

    Entries in ``_STRICT_KEYS`` must be identical. A manifest without
    ``algorithm_version`` counts as version 1. Differences in ``_WARN_KEYS``
    (git hash, uncommitted changes, package version) only print a warning
    prefixed with ``Warning:``; ``lammps_cmd`` and ``print_level`` are not
    compared.

    Parameters
    ----------
    stored : dict
        Content of the existing manifest.
    settings : dict
        Settings of the current invocation.

    Raises
    ------
    ValueError
        If any strict entry differs.
    """
    # manifests written before the key existed are version 1
    stored = {"algorithm_version": 1, **stored}
    diffs = []
    for key in _STRICT_KEYS:
        old, new = stored.get(key), settings.get(key)
        if old != new:
            diffs.append(f"  {key}: manifest has {old!r}, now {new!r}")
    if diffs:
        hint = ""
        if stored["algorithm_version"] != settings["algorithm_version"]:
            hint = (
                " The algorithm changed since this run was started; its cached "
                "evaluations cannot be replayed with the current code."
            )
        raise ValueError(
            "Settings differ from the manifest of this run directory, refusing to "
            "restart (use a new --workdir for different settings)."
            + hint
            + "\n"
            + "\n".join(diffs)
        )

    # warn-only entries: a None (no git checkout) on either side cannot be compared
    for key in ("git_hash", "version"):
        old, new = stored.get(key), settings.get(key)
        if old is not None and new is not None and old != new:
            print(
                f"Warning: {key} differs from the manifest (manifest has {old!r}, "
                f"now {new!r}); the restart is only valid if the change does "
                "not affect the algorithm."
            )
    if settings.get("git_dirty"):
        print(
            "Warning: the package source has uncommitted changes; the restart "
            "may not replay the original run."
        )


def archive_config(workdir: Path, config: str | Path | None) -> None:
    """Keep a copy of the TOML configuration file in the run directory.

    The configuration file is the record of what was submitted. The copy
    survives edits of the original (e.g. between restarts). Nothing is ever
    overwritten: a file with new content is stored under the next free name
    (``config.toml``, ``config_1.toml``, ...), identical content is not stored
    twice.

    Parameters
    ----------
    workdir : Path
        Run directory (must exist).
    config : str | Path | None
        Path of the configuration file, ``None`` if none was given.
    """
    if config is None:
        return
    content = Path(config).read_bytes()
    k = 0
    while True:
        target = workdir / ("config.toml" if k == 0 else f"config_{k}.toml")
        if not target.exists():
            target.write_bytes(content)
            return
        if target.read_bytes() == content:
            return
        k += 1


def prepare_run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], bool]:
    """Validate arguments, create the run directory and the manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    workdir : Path
        Run directory.
    settings : dict
        Settings of the run (with the seed of the manifest on a restart).
    restart : bool
        ``True`` if an existing manifest was found and matched.

    Raises
    ------
    ValueError
        If arguments are invalid or differ from an existing manifest.
    FileExistsError
        If the summary file already exists.
    """
    build_config(args)  # validation only, raises ValueError
    if args.n_mc_burnin == 0:
        # printed here, not per chain, so that it appears once per run
        print("Warning: --mcburnin 0, no burn-in is discarded.")
    workdir = Path(args.workdir)
    manifest = workdir / MANIFEST_NAME
    summary = _summary_path(workdir, args.output)
    if summary.exists():
        raise FileExistsError(
            f"Summary file {summary} exists: this run is complete, refusing to "
            "overwrite. Use a new --workdir or -o."
        )

    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            stored = json.load(f)
        # a restart without --seed continues with the seed of the first start
        seed = args.seed if args.seed is not None else stored["seed"]
        settings = build_settings(args, seed)
        check_manifest(stored, settings)
        archive_config(workdir, args.config)
        return workdir, settings, True

    seed = (
        int(args.seed)
        if args.seed is not None
        else int(np.random.SeedSequence().entropy)
    )
    settings = build_settings(args, seed)
    workdir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, settings)
    archive_config(workdir, args.config)
    return workdir, settings, False


def _summary_path(workdir: Path, output: str) -> Path:
    """Return the path of the combined summary file.

    Parameters
    ----------
    workdir : Path
        Run directory.
    output : str
        ``-o`` argument; relative paths are placed in the run directory.

    Returns
    -------
    Path
        Summary file path.
    """
    return workdir / output  # absolute ``output`` replaces the workdir


def _run_one_chain(
    chain: int, settings: dict[str, Any], workdir_str: str
) -> dict[str, Any]:
    """Run one chain and write its HDF5 file (top-level for pickling).

    Parameters
    ----------
    chain : int
        Chain index ``c``.
    settings : dict
        Manifest settings.
    workdir_str : str
        Run directory.

    Returns
    -------
    dict
        ``chain``, ``estimator`` (list), ``n_hits``, ``n_computed``.
    """
    workdir = Path(workdir_str)
    config = config_from_settings(settings)
    cache_dir = workdir / "cache" / f"chain_{chain:03d}"
    # each chain has its own cache so that processes never write to the same files
    forward = CachedForwardModel(make_inner_model(settings, cache_dir), cache_dir)
    rng = np.random.default_rng([settings["seed"], chain])

    if config.print_level == 1:
        # with several chains the debug output of the processes interleaves
        print(f"Chain {chain:03d}: start.")
    result = run_chain(config, forward, rng, settings["seed"])

    final = workdir / f"chain_{chain:03d}.hdf5"
    tmp = workdir / f".chain_{chain:03d}.{os.getpid()}.tmp.hdf5"
    write_results(tmp, result, config, {**settings, "chain": chain})
    try:
        # os.link fails if the final file exists: results are never overwritten
        os.link(tmp, final)
    finally:
        tmp.unlink()
    return {
        "chain": chain,
        "estimator": result.estimator.tolist(),
        "n_hits": forward.n_hits,
        "n_computed": forward.n_computed,
    }


def _load_finished_chain(path: Path, chain: int) -> dict[str, Any]:
    """Read the estimator of a chain finished in an earlier invocation.

    Parameters
    ----------
    path : Path
        ``chain_{c:03d}.hdf5``.
    chain : int
        Chain index.

    Returns
    -------
    dict
        Same keys as :func:`_run_one_chain`; evaluation counters are zero.
    """
    with h5py.File(path, "r") as f:
        est = np.asarray(f["estimator"]).tolist()
    return {"chain": chain, "estimator": est, "n_hits": 0, "n_computed": 0}


def run_chains(
    workdir: Path, settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Run all unfinished chains concurrently.

    Chains whose HDF5 file already exists are not rerun. A failing chain does
    not stop the others; failures are reported after all chains ended.

    Parameters
    ----------
    workdir : Path
        Run directory.
    settings : dict
        Manifest settings.

    Returns
    -------
    results : list[dict]
        Per finished chain (ordered by chain index) the dictionary of
        :func:`_run_one_chain`.
    failed : dict[int, str]
        Chain index and error message of every chain that raised.
    """
    n_chains = settings["n_chains"]
    done: dict[int, dict[str, Any]] = {}
    todo = []
    for c in range(n_chains):
        path = workdir / f"chain_{c:03d}.hdf5"
        if path.exists():
            print(f"Chain {c:03d}: finished in an earlier run, skipping.")
            done[c] = _load_finished_chain(path, c)
        else:
            todo.append(c)

    failed: dict[int, str] = {}
    if len(todo) == 1:
        # no process pool for a single chain: simpler errors, easier debugging
        try:
            res = _run_one_chain(todo[0], settings, str(workdir))
            done[todo[0]] = res
        except Exception as exc:  # noqa: BLE001 - reported below, not swallowed
            print(f"Chain {todo[0]:03d} failed: {exc!r}")
            failed[todo[0]] = repr(exc)
    elif todo:
        with ProcessPoolExecutor(max_workers=len(todo)) as pool:
            futures = {
                pool.submit(_run_one_chain, c, settings, str(workdir)): c for c in todo
            }
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    done[c] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"Chain {c:03d} failed: {exc!r}")
                    failed[c] = repr(exc)

    for c in sorted(done):
        r = done[c]
        print(
            f"Chain {c:03d}: {r['n_hits']} evaluations from cache, "
            f"{r['n_computed']} computed."
        )
    return [done[c] for c in sorted(done)], failed


def write_summary(
    path: Path, results: Sequence[dict[str, Any]], settings: dict[str, Any]
) -> np.ndarray:
    """Write the combined summary file and return the combined estimator.

    Parameters
    ----------
    path : Path
        Summary file; must not exist.
    results : Sequence[dict]
        Chain results from :func:`run_chains`.
    settings : dict
        Manifest settings.

    Returns
    -------
    np.ndarray
        Combined estimator, the mean over chains.

    Raises
    ------
    FileExistsError
        If ``path`` exists.
    """
    if path.exists():
        raise FileExistsError(f"Summary file {path} exists, refusing to overwrite.")
    estimators = np.array([r["estimator"] for r in results])
    combined = estimators.mean(axis=0)
    # between-chain standard deviation needs at least two chains
    spread = (
        estimators.std(axis=0, ddof=1)
        if len(estimators) > 1
        else np.full(estimators.shape[1], np.nan)
    )
    with h5py.File(path, "x") as f:
        f.create_dataset("estimators", data=estimators)
        f.create_dataset("estimator", data=combined)
        f.create_dataset("estimator_std_between_chains", data=spread)
        f.attrs["settings"] = json.dumps(settings)
        f.attrs["seed"] = str(settings["seed"])
        f.attrs["chain_files"] = [f"chain_{r['chain']:03d}.hdf5" for r in results]
    return combined


def markov_chain_eval(args: argparse.Namespace) -> int:
    """Run (or restart) the MFDA-MCMC algorithm with one or several chains.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    int
        Exit code, ``0`` for success.

    Raises
    ------
    ValueError
        If the arguments are invalid or differ from the manifest of the run
        directory.
    FileExistsError
        If the summary file already exists.
    RuntimeError
        If at least one chain failed. Finished chains and the cache are kept;
        resubmitting the same command continues the run.
    """
    workdir, settings, restart = prepare_run(args)
    print(f"Random seed: {settings['seed']}")
    print(
        f"{'Restarting' if restart else 'Starting'} run in {workdir} "
        f"with {settings['n_chains']} chain(s)."
    )

    results, failed = run_chains(workdir, settings)
    if failed:
        msgs = "; ".join(f"chain {c:03d}: {m}" for c, m in sorted(failed.items()))
        raise RuntimeError(
            f"{len(failed)} chain(s) failed ({msgs}). Finished chains and cached "
            "evaluations are kept; rerun the same command to continue."
        )

    config = config_from_settings(settings)
    combined = write_summary(_summary_path(workdir, args.output), results, settings)
    ests = np.array([r["estimator"] for r in results])
    for r in results:
        e = r["estimator"]
        print(
            f"Chain {r['chain']:03d} estimator (eps, sig, q): {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}"
        )
    print(
        f"\nMFDA-MCMC estimator (mean of {len(ests)} chain(s)): "
        f"{combined[0]:.6f} {combined[1]:.6f} {combined[2]:.6f}"
    )
    if len(ests) > 1:
        sd = ests.std(axis=0, ddof=1)
        print(
            f"Between-chain standard deviation:        "
            f"{sd[0]:.6f} {sd[1]:.6f} {sd[2]:.6f}"
        )
    print(f"Levels: {config.n_levels}; results in {workdir}")
    return 0


def required_cpus(args: argparse.Namespace) -> int:
    """Return the number of CPUs needed for ``--chains`` concurrent MD runs.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments (``n_molecules``, ``n_chains``).

    Returns
    -------
    int
        ``chains * calc_cpus(N_1)`` with ``N_1`` the largest molecule number.

    Raises
    ------
    ValueError
        If ``--molecules`` is missing.
    """
    molecules = as_int_list(args.n_molecules)
    if not molecules:
        raise ValueError("--molecules is required.")
    return int(args.n_chains) * calc_cpus(max(molecules))


def mfda_ncpu(args: argparse.Namespace) -> int:
    """Print the number of CPUs to request in the job script.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    int
        Exit code ``0``.
    """
    # only the number goes to stdout so that it can be used in shell scripts
    print(required_cpus(args))
    return 0


def mfda_smoke(args: argparse.Namespace) -> int:
    """Run exactly one forward evaluation at the OPC3 parameters (no MCMC).

    Meant to test fftool, packmol, LAMMPS, TRAVIS and msdiff before a long run.
    The evaluation is stored in ``<workdir>/smoke/`` like any cached evaluation.
    An evaluation that is already cached is not recomputed.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments; ``--molecules`` must be a single value.

    Returns
    -------
    int
        Exit code ``0``.

    Raises
    ------
    ValueError
        If ``--molecules`` is not exactly one value.
    """
    molecules = as_int_list(args.n_molecules)
    if len(molecules) != 1:
        raise ValueError("mfda-smoke needs exactly one value for --molecules.")
    n = molecules[0]
    if args.forward_model != "md":
        print("Warning: --forward-model is not 'md', this tests nothing external.")

    settings = {
        "forward_model": args.forward_model,
        "orthoboxy": bool(args.orthoboxy),
        "lammps_cmd": args.lammps_cmd,
    }
    cache_dir = Path(args.workdir) / "smoke"
    forward = CachedForwardModel(make_inner_model(settings, cache_dir), cache_dir)
    theta = np.array(
        [constants.OPC3_EPSILON_OO, constants.OPC3_SIGMA_OO, constants.OPC3_CHARGE_O]
    )
    start = time.perf_counter()
    d = forward(n, theta)
    wall = time.perf_counter() - start
    source = "cache" if forward.n_hits else "computed"
    print(f"N = {n}, theta = {theta.tolist()} (OPC3)")
    print(f"D = {d:.6f} 1e-9 m^2/s ({source}, {wall:.1f} s)")
    print(f"Evaluation directory: {forward.entry_dir(n, theta)}")
    return 0
