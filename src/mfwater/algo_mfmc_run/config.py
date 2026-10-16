"""Settings of an ``mfmc-run``: one validated, frozen description of the run."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py

from ..algo_input import model_groups
from ..algo_mfda.markov_chain import as_int_list
from ..md.protocol import PROTOCOLS
from ..md.system import calc_cpus

_PARAMS = ("lj", "q", "lj-q")
_RMSE_COSTS = ("calibration", "performance")
_FORWARD_MODELS = ("dummy", "md")
_STOP_AFTER = ("calibration",)


def budget_label(budget: float) -> str:
    """Return the file name label of a budget.

    Parameters
    ----------
    budget : float
        Budget in core-seconds.

    Returns
    -------
    str
        ``"100000"`` for ``1e5``; ``repr`` for non-integer budgets.
    """
    return str(int(budget)) if float(budget).is_integer() else repr(float(budget))


def parse_pairs(
    items: Sequence[str] | Mapping[str, object] | None, name: str
) -> dict[str, str]:
    """Parse ``KEY=VALUE`` strings (or a mapping) into a dictionary of strings.

    Parameters
    ----------
    items : Sequence[str] | Mapping | None
        ``["1024=6", ...]`` as given on the command line or a mapping (TOML table).
    name : str
        Option name, for error messages.

    Returns
    -------
    dict[str, str]
        Key and value as strings.

    Raises
    ------
    ValueError
        If an entry is not of the form ``KEY=VALUE``.
    """
    if not items:
        return {}
    if isinstance(items, Mapping):
        return {str(k): str(v) for k, v in items.items()}
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = str(item).partition("=")
        if not sep or not key or not value:
            raise ValueError(f"--{name}: expected KEY=VALUE, got {item!r}.")
        out[key.strip()] = value.strip()
    return out


@dataclass(frozen=True)
class MFMCRunConfig:
    """All settings of an ``mfmc-run``.

    Parameters
    ----------
    molecules : tuple[int, ...]
        Molecules per model, strictly descending (model 1 = high fidelity).
    evals : int
        Calibration evaluations ``m`` per model (the same for all models).
    budgets : tuple[float, ...]
        Budgets in core-seconds (the unit of the cost ``w``), distinct, positive.
    params : str
        ``lj``, ``q`` or ``lj-q``: which parameters are perturbed.
    orthoboxy : bool
        Tetragonal OrthoBoXY boxes.
    protocol : str
        Registered MD protocol (default ``paper1``).
    seed : int | None
        Seed of all sampling; None draws a random one (printed and stored).
    total_cpus : int
        Cores of the allocation that the evaluation pool may use.
    ncpu_override : dict[int, int]
        Table ``N -> MPI ranks`` replacing ``calc_cpus`` (stored, not compared on
        restart).
    stop_after : str | None
        ``"calibration"`` stops after the model selection.
    calibration_from : Path | None
        Paper-I-format calibration file that replaces phase 1 (stats and costs).
    replay_from : Path | None
        Paper-I-format calibration file whose stored samples and seeds are used
        (and evaluated again) instead of drawing new ones.
    replay_performance_from : dict[float, Path]
        Per budget a paper-I-format performance (``oms``) file whose samples and
        seeds are re-evaluated.
    replay_use_oms_allocation : bool
        Use model set and ``n_evals`` of the replayed ``oms`` files instead of the
        computed allocation (error if they differ and this is False).
    workdir : Path
        Run directory.
    forward_model : str
        ``md`` or ``dummy`` (correlated dummy, for dry runs).
    rmse_cost : str
        Cost basis of ``mfmc_error`` and ``gamma``: ``calibration`` or ``performance``.
    lammps_cmd : str
        LAMMPS command template (``{ncpu}``, ``{input}``).
    msdiff_cmd : str
        msdiff executable (name or path); stored (with its version), not compared.
    print_level : int
        Stored only.
    claim_timeout : float
        Seconds without heartbeat after which a claim of another job is stale.
    """

    molecules: tuple[int, ...]
    evals: int
    budgets: tuple[float, ...]
    params: str = "lj"
    orthoboxy: bool = False
    protocol: str = "paper1"
    seed: int | None = None
    total_cpus: int = 1
    ncpu_override: dict[int, int] = field(default_factory=dict)
    stop_after: str | None = None
    calibration_from: Path | None = None
    replay_from: Path | None = None
    replay_performance_from: dict[float, Path] = field(default_factory=dict)
    replay_use_oms_allocation: bool = False
    workdir: Path = Path("./mfmc_run")
    forward_model: str = "dummy"
    rmse_cost: str = "calibration"
    lammps_cmd: str = "mpirun -np {ncpu} lmp -i {input}"
    msdiff_cmd: str = "msdiff"
    print_level: int = 0
    claim_timeout: float = 1800.0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If any setting is invalid or inconsistent.
        """
        if len(self.molecules) < 2:
            raise ValueError("At least two models (--molecules) are needed.")
        if any(n < 1 for n in self.molecules) or any(
            a <= b for a, b in zip(self.molecules, self.molecules[1:])
        ):
            raise ValueError(
                f"--molecules must be positive and strictly descending, got "
                f"{list(self.molecules)}."
            )
        if self.evals < 2:
            raise ValueError(
                f"--evals must be at least 2 (the correlation needs two samples), "
                f"got {self.evals}."
            )
        if not self.budgets:
            raise ValueError("--budgets is required (at least one budget).")
        if any(b <= 0 for b in self.budgets) or len(set(self.budgets)) != len(
            self.budgets
        ):
            raise ValueError(
                f"--budgets must be positive and distinct, got {list(self.budgets)}."
            )
        if self.params not in _PARAMS:
            raise ValueError(f"Unknown params {self.params!r}, choose from {_PARAMS}.")
        if self.protocol not in PROTOCOLS:
            raise ValueError(
                f"Unknown protocol {self.protocol!r}, choose from {sorted(PROTOCOLS)}."
            )
        if self.total_cpus < 1:
            raise ValueError(f"--total-cpus must be >= 1, got {self.total_cpus}.")
        for n, r in self.ncpu_override.items():
            if r < 1:
                raise ValueError(f"--ncpu-override: ranks for N={n} must be >= 1.")
            if n not in self.molecules:
                raise ValueError(
                    f"--ncpu-override: N={n} is not in --molecules {list(self.molecules)}."
                )
        if self.stop_after not in (None, *_STOP_AFTER):
            raise ValueError(
                f"--stop-after must be one of {_STOP_AFTER}, got {self.stop_after!r}."
            )
        if self.rmse_cost not in _RMSE_COSTS:
            raise ValueError(f"--rmse-cost must be one of {_RMSE_COSTS}.")
        if self.forward_model not in _FORWARD_MODELS:
            raise ValueError(f"--forward-model must be one of {_FORWARD_MODELS}.")
        if self.calibration_from is not None and self.replay_from is not None:
            raise ValueError(
                "--calibration-from (skip the calibration) and --replay-from "
                "(repeat it with stored samples) exclude each other."
            )
        for b in self.replay_performance_from:
            if b not in self.budgets:
                raise ValueError(
                    f"--replay-performance-from: budget {b:g} is not in --budgets "
                    f"{list(self.budgets)}."
                )
        if self.replay_use_oms_allocation and not self.replay_performance_from:
            raise ValueError(
                "--replay-use-oms-allocation needs --replay-performance-from."
            )
        if self.claim_timeout <= 0:
            raise ValueError("--claim-timeout must be positive.")
        ranks = max(self.ranks(n) for n in self.molecules)
        if ranks > self.total_cpus:
            raise ValueError(
                f"The largest system needs {ranks} MPI ranks but --total-cpus is "
                f"{self.total_cpus}."
            )

    def ranks(self, n_molecules: int) -> int:
        """Return the MPI ranks of an evaluation with ``n_molecules``.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.

        Returns
        -------
        int
            Override from ``ncpu_override``, else ``calc_cpus``.
        """
        return self.ncpu_override.get(n_molecules, calc_cpus(n_molecules))


def _read_calibration_header(path: Path) -> tuple[list[int], int]:
    """Read molecules and evaluations of a paper-I-format calibration file.

    Parameters
    ----------
    path : Path
        Calibration file.

    Returns
    -------
    molecules : list[int]
        Molecules per model, descending.
    evals : int
        Evaluations per model.
    """
    with h5py.File(path, "r") as f:
        items = model_groups(f["models"])
        molecules = [int(g.attrs["n_molecules"]) for _, g in items]
        evals = int(items[0][1].attrs["n_evals"])
    return molecules, evals


def build_config(args: argparse.Namespace) -> MFMCRunConfig:
    """Build and validate the settings from parsed arguments.

    ``--molecules`` and ``--evals`` may be omitted if ``--calibration-from`` or
    ``--replay-from`` is given; they are then read from that file (and must match
    if both are given).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line (TOML file already merged).

    Returns
    -------
    MFMCRunConfig
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is missing or invalid.
    """
    molecules = as_int_list(args.n_molecules)
    evals_list = as_int_list(args.n_evals)
    if len(set(evals_list)) > 1:
        raise ValueError(
            "--evals must be a single value for mfmc-run: the calibration uses the "
            f"same number of evaluations for all models, got {evals_list}."
        )
    evals = evals_list[0] if evals_list else None

    source = args.calibration_from or args.replay_from
    if source is not None:
        file_mol, file_evals = _read_calibration_header(Path(source))
        if molecules and molecules != file_mol:
            raise ValueError(
                f"--molecules {molecules} differ from the models of {source}: {file_mol}."
            )
        if evals is not None and evals != file_evals:
            raise ValueError(
                f"--evals {evals} differs from the evaluations of {source}: {file_evals}."
            )
        molecules, evals = file_mol, file_evals
    if not molecules:
        raise ValueError("--molecules is required.")
    if evals is None:
        raise ValueError("--evals is required (calibration evaluations per model).")

    override = {
        int(k): int(v)
        for k, v in parse_pairs(args.ncpu_override, "ncpu-override").items()
    }
    replay_perf = {
        float(k): Path(v)
        for k, v in parse_pairs(
            args.replay_performance_from, "replay-performance-from"
        ).items()
    }
    for path in (
        *replay_perf.values(),
        *([Path(args.replay_from)] if args.replay_from else []),
    ):
        if not path.is_file():
            raise ValueError(f"Replay file {path} does not exist.")

    ranks = {
        n: override.get(n, calc_cpus(n)) for n in molecules
    }  # same rule as MFMCRunConfig.ranks
    if args.total_cpus is not None:
        total = int(args.total_cpus)
    elif os.environ.get("SLURM_NTASKS"):
        total = int(os.environ["SLURM_NTASKS"])
    else:
        # one evaluation at a time: enough for the largest system
        total = max(ranks.values())

    return MFMCRunConfig(
        molecules=tuple(molecules),
        evals=int(evals),
        budgets=tuple(float(b) for b in (args.budgets or ())),
        params=args.params,
        orthoboxy=bool(args.orthoboxy),
        protocol=args.protocol,
        seed=args.seed,
        total_cpus=total,
        ncpu_override=override,
        stop_after=args.stop_after,
        calibration_from=Path(args.calibration_from) if args.calibration_from else None,
        replay_from=Path(args.replay_from) if args.replay_from else None,
        replay_performance_from=replay_perf,
        replay_use_oms_allocation=bool(args.replay_use_oms_allocation),
        workdir=Path(args.workdir),
        forward_model=args.forward_model,
        rmse_cost=args.rmse_cost or "calibration",
        lammps_cmd=args.lammps_cmd,
        msdiff_cmd=args.msdiff_cmd,
        print_level=int(args.print_level),
        claim_timeout=float(args.claim_timeout),
    )
