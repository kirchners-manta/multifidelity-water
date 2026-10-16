"""Job script generation for ``mfmc-run`` (shares the machinery of ``mfda-jobscript``)."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from ..algo_mfda.jobscript import load_profile, render_jobscript
from .config import build_config


def mfmc_jobscript(args: argparse.Namespace) -> int:
    """Write a ready-to-submit batch script for an ``mfmc-run`` described by a run TOML.

    The settings are validated with :func:`build_config`. ``--ntasks`` is the
    ``total_cpus`` of the run TOML, which must be given explicitly and fit on one node.
    Submitting the script several times runs several jobs on the same run directory
    (they share the work, see the README).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (``config`` is the run TOML, ``cluster`` or ``cluster_file``,
        optional ``jobscript``).

    Returns
    -------
    int
        Exit code ``0``.

    Raises
    ------
    ValueError
        If ``-c`` or ``total_cpus`` is missing, the cluster profile or the run settings
        are invalid, or ``total_cpus`` does not fit on one node.
    FileExistsError
        If the output file exists (it is never overwritten).
    """
    if args.config is None:
        raise ValueError("mfmc-jobscript needs the run settings: -c run.toml.")
    if args.total_cpus is None:
        raise ValueError(
            "mfmc-jobscript needs total_cpus (cores of the allocation) in the run TOML "
            "or as --total-cpus."
        )
    profile = load_profile(args.cluster, args.cluster_file)
    cfg = build_config(args)  # raises on invalid algorithm settings
    ntasks = cfg.total_cpus
    if ntasks > profile.cores_per_node:
        raise ValueError(
            f"total_cpus = {ntasks} does not fit on one node of '{profile.name}' "
            f"({profile.cores_per_node} cores)."
        )

    run_toml = Path(args.config).resolve()
    with open(run_toml, "rb") as f:
        run_keys = {k.replace("-", "_") for k in tomllib.load(f)}
    job_name = cfg.workdir.resolve().name
    text = render_jobscript(
        profile,
        ntasks,
        job_name,
        run_toml,
        "lammps_cmd" not in run_keys,
        template=f"jobscript_{profile.scheduler}_mfmc.sh.template",
    )
    out = Path(args.jobscript or f"run-mfmc-{profile.name}.sh")
    # mode "x" fails if the file exists: never overwrite a job script
    with open(out, "x", encoding="utf-8") as f:
        f.write(text)
    print(
        f"Wrote {out} ({ntasks} tasks, job name '{job_name}'). Submit it with sbatch; "
        "submit it k times to run k jobs on the same run directory."
    )
    return 0
