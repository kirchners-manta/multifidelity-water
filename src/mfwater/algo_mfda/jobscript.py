"""Cluster-independent job scripts for MFDA runs.

A cluster is described by a small TOML profile (``data/clusters/<name>.toml``)
holding the scheduler header values, the environment setup and the LAMMPS launch
command. The job script itself comes from one template per scheduler
(``data/jobscript_<scheduler>.sh.template``). Adding a cluster therefore means
adding one TOML file (or passing ``--cluster-file``).
"""

from __future__ import annotations

import argparse
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .markov_chain import build_config
from .run_manager import required_cpus

_DATA_DIR = Path(__file__).parent / "data"

#: Schedulers with a job script template.
SUPPORTED_SCHEDULERS = ("slurm",)


@dataclass(frozen=True)
class ClusterProfile:
    """Description of a cluster, read from a TOML profile.

    Parameters
    ----------
    name : str
        Profile name (used in the default output file name).
    scheduler : str
        Batch scheduler, one of `SUPPORTED_SCHEDULERS`.
    cores_per_node : int
        Cores of one node; all chains of a run share one node.
    header : dict[str, Any]
        Scheduler specific values (the ``[slurm]`` table for Slurm).
    setup : str
        Shell snippet that prepares the environment (modules, conda, PATH).
    lammps_cmd : str
        LAMMPS launch command with the placeholders ``{ncpu}`` and ``{input}``.
    """

    name: str
    scheduler: str
    cores_per_node: int
    header: dict[str, Any] = field(default_factory=dict)
    setup: str = ""
    lammps_cmd: str = ""


def load_profile(
    cluster: str | None = None, cluster_file: str | Path | None = None
) -> ClusterProfile:
    """Load a built-in cluster profile or a profile file.

    Parameters
    ----------
    cluster : str, optional
        Name of a built-in profile in ``data/clusters/``.
    cluster_file : str or Path, optional
        Path of a profile file (alternative to ``cluster``).

    Returns
    -------
    ClusterProfile
        The validated profile.

    Raises
    ------
    ValueError
        If neither or both of ``cluster`` and ``cluster_file`` are given, the
        built-in profile does not exist, or the profile is invalid (unknown
        scheduler, missing keys, LAMMPS command without placeholders).
    FileNotFoundError
        If ``cluster_file`` does not exist.
    """
    if (cluster is None) == (cluster_file is None):
        raise ValueError("Give exactly one of --cluster and --cluster-file.")
    if cluster_file is not None:
        path = Path(cluster_file)
        name = path.stem
    else:
        name = str(cluster)
        path = _DATA_DIR / "clusters" / f"{name}.toml"
        if not path.is_file():
            known = sorted(p.stem for p in (_DATA_DIR / "clusters").glob("*.toml"))
            raise ValueError(f"Unknown cluster '{name}', built-in clusters: {known}.")
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Cannot parse cluster profile {path}: {exc}") from exc

    scheduler = data.get("scheduler")
    if scheduler not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler {scheduler!r} in {path}, "
            f"supported: {list(SUPPORTED_SCHEDULERS)}."
        )
    cores = data.get("cores_per_node")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ValueError(f"'cores_per_node' in {path} must be a positive integer.")
    header = data.get(scheduler, {})
    for key in ("partition", "time"):
        if key not in header:
            raise ValueError(f"Missing '{key}' in table [{scheduler}] of {path}.")
    cmd = data.get("lammps", {}).get("cmd", "")
    if "{ncpu}" not in cmd or "{input}" not in cmd:
        raise ValueError(
            f"[lammps] cmd in {path} must contain the placeholders {{ncpu}} and {{input}}."
        )
    return ClusterProfile(
        name=name,
        scheduler=scheduler,
        cores_per_node=cores,
        header=dict(header),
        setup=str(data.get("environment", {}).get("setup", "")).strip("\n"),
        lammps_cmd=cmd,
    )


def _slurm_header(profile: ClusterProfile, ntasks: int, job_name: str) -> str:
    """Return the ``#SBATCH`` lines of a Slurm job script.

    Parameters
    ----------
    profile : ClusterProfile
        Cluster profile (``[slurm]`` table in ``header``).
    ntasks : int
        Number of MPI tasks.
    job_name : str
        Slurm job name.

    Returns
    -------
    str
        The header lines joined by newlines.
    """
    h = profile.header
    lines = [f"#SBATCH --partition={h['partition']}"]
    if "account" in h:
        lines.append(f"#SBATCH --account={h['account']}")
    # one node: every chain starts its own mpirun from the same allocation
    lines += [
        f"#SBATCH --ntasks={ntasks}",
        "#SBATCH --nodes=1",
        f"#SBATCH --time={h['time']}",
        f"#SBATCH --job-name={job_name}",
    ]
    lines += [f"#SBATCH {opt.removeprefix('#SBATCH ')}" for opt in h.get("extra", [])]
    return "\n".join(lines)


def render_jobscript(
    profile: ClusterProfile,
    ntasks: int,
    job_name: str,
    run_toml: Path,
    set_lammps_cmd: bool,
    template: str | None = None,
) -> str:
    """Fill the job script template of the scheduler of ``profile``.

    Parameters
    ----------
    profile : ClusterProfile
        Cluster profile.
    ntasks : int
        Number of MPI tasks.
    job_name : str
        Job name.
    run_toml : Path
        Absolute path of the run TOML passed to ``markov-chain``.
    set_lammps_cmd : bool
        Whether to add ``--lammps-cmd`` from the profile (``False`` if the run
        TOML sets ``lammps_cmd`` itself).
    template : str, optional
        File name of the template in ``data/``. Default: the MFDA template of the
        scheduler (``jobscript_<scheduler>.sh.template``). The MFMC job script uses
        ``jobscript_<scheduler>_mfmc.sh.template``.

    Returns
    -------
    str
        The job script.
    """
    template_name = template or f"jobscript_{profile.scheduler}.sh.template"
    text = (_DATA_DIR / template_name).read_text()
    option = ""
    if set_lammps_cmd:
        option = f" \\\n    --lammps-cmd {shlex.quote(profile.lammps_cmd)}"
    # plain replace (not str.format): the setup snippet contains shell braces
    return (
        text.replace("@SBATCH_HEADER@", _slurm_header(profile, ntasks, job_name))
        .replace("@ENVIRONMENT_SETUP@", profile.setup)
        .replace("@RUN_TOML@", shlex.quote(str(run_toml)))
        .replace("@LAMMPS_OPTION@", option)
    )


def mfda_jobscript(args: argparse.Namespace) -> int:
    """Write a batch job script for an MFDA run described by a run TOML.

    The run settings are validated with :func:`build_config`; the number of
    tasks is ``chains * calc_cpus(N_1)`` and must fit on one node.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (``config`` is the run TOML, ``cluster`` or
        ``cluster_file``, optional ``jobscript``).

    Returns
    -------
    int
        Exit code ``0``.

    Raises
    ------
    ValueError
        If ``-c`` is missing, the cluster profile or the run settings are
        invalid, or the tasks do not fit on one node.
    FileExistsError
        If the output file exists (it is never overwritten).
    """
    if args.config is None:
        raise ValueError("mfda-jobscript needs the run settings: -c run.toml.")
    profile = load_profile(args.cluster, args.cluster_file)
    build_config(args)  # raises on invalid algorithm settings
    ntasks = required_cpus(args)
    if ntasks > profile.cores_per_node:
        raise ValueError(
            f"{ntasks} tasks (chains x CPUs of the largest system) do not fit on one "
            f"node of '{profile.name}' ({profile.cores_per_node} cores); "
            "reduce --chains."
        )

    run_toml = Path(args.config).resolve()
    with open(run_toml, "rb") as f:
        run_keys = {k.replace("-", "_") for k in tomllib.load(f)}
    job_name = Path(args.workdir).resolve().name
    text = render_jobscript(
        profile, ntasks, job_name, run_toml, "lammps_cmd" not in run_keys
    )

    out = Path(args.jobscript or f"run-mfda-{profile.name}.sh")
    # mode "x" fails if the file exists: never overwrite a job script
    with open(out, "x", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {out} ({ntasks} tasks, job name '{job_name}').")
    return 0
