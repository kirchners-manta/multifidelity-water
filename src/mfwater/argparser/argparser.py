"""
Command line parser utilities for MFWater.
"""

#############################################

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .. import __version__


def is_file(path: str | Path) -> str | Path:
    """Check whether a path points to an existing file.

    Parameters
    ----------
    path : str | Path
        Path to validate.

    Returns
    -------
    str | Path
        The original path if it exists and is not a directory.

    Raises
    ------
    argparse.ArgumentTypeError
        If the path does not exist or is a directory.
    """
    p = Path(path)

    if p.is_dir():
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': Is a directory.",
        )

    if p.is_file() is False:
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': No such file.",
        )

    return path


def is_dir(path: str | Path) -> str | Path:
    """Check whether a path points to an existing directory.

    Parameters
    ----------
    path : str | Path
        Path to validate.

    Returns
    -------
    str | Path
        The original path if it exists and is a directory.

    Raises
    ------
    argparse.ArgumentTypeError
        If the path does not exist or is a file.
    """
    p = Path(path)

    if p.is_file():
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': Is not a directory.",
        )

    if p.is_dir() is False:
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': No such file or directory.",
        )

    return path


def action_not_less_than(min_value: float = 0.0) -> type[argparse.Action]:
    """Create an argparse action that rejects values below a minimum.

    Parameters
    ----------
    min_value : float, optional
        Minimum accepted value, by default 0.0.

    Returns
    -------
    type[argparse.Action]
        Custom action class that enforces the lower bound.
    """

    def _normalize_values(
        values: Any,
    ) -> list[Any]:
        """Normalize argparse action values to a list."""
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if isinstance(values, Real):
            return [values]
        return list(values)

    class CustomActionLessThan(argparse.Action):
        """
        Custom action for limiting possible input values. Raise error if value is smaller than min_value.
        """

        def __call__(
            self,
            p: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: Any,
            option_string: str | None = None,
        ) -> None:

            normalized_values = _normalize_values(values)

            if any(value < min_value for value in normalized_values):
                p.error(
                    f"Option '{option_string}' takes only values larger than {min_value}. {normalized_values} is not accepted."
                )

            if len(normalized_values) == 1:
                value: float | int | list[float | int] = normalized_values[0]
            else:
                value = normalized_values

            setattr(args, self.dest, value)

    return CustomActionLessThan


def action_not_more_than(max_value: float = 0.0) -> type[argparse.Action]:
    """Create an argparse action that rejects values above a maximum.

    Parameters
    ----------
    max_value : float, optional
        Maximum accepted value, by default 0.0.

    Returns
    -------
    type[argparse.Action]
        Custom action class that enforces the upper bound.
    """

    def _normalize_values(
        values: Any,
    ) -> list[Any]:
        """Normalize argparse action values to a list."""
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if isinstance(values, Real):
            return [values]
        return list(values)

    class CustomActionMoreThan(argparse.Action):
        """
        Custom action for limiting possible input values. Raise error if value is larger than max_value.
        """

        def __call__(
            self,
            p: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: Any,
            option_string: str | None = None,
        ) -> None:
            normalized_values = _normalize_values(values)

            if any(value > max_value for value in normalized_values):
                p.error(
                    f"Option '{option_string}' takes only values smaller than {max_value}. {normalized_values} is not accepted."
                )

            if len(normalized_values) == 1:
                value: float | int | list[float | int] = normalized_values[0]
            else:
                value = normalized_values

            setattr(args, self.dest, value)

    return CustomActionMoreThan


def action_in_range(
    min_value: float = 0.0, max_value: float = 1.0
) -> type[argparse.Action]:
    """Create an argparse action that rejects values outside a range.

    Parameters
    ----------
    min_value : float, optional
        Minimum accepted value, by default 0.0.
    max_value : float, optional
        Maximum accepted value, by default 1.0.

    Returns
    -------
    type[argparse.Action]
        Custom action class that enforces the value range.
    """

    def _normalize_values(
        values: Any,
    ) -> list[Any]:
        """Normalize argparse action values to a list."""
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if isinstance(values, Real):
            return [values]
        return list(values)

    class CustomActionInRange(argparse.Action):
        """
        Custom action for limiting possible input values in a range. Raise error if value is not in range [min_value, max_value].
        """

        def __call__(
            self,
            p: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: Any,
            option_string: str | None = None,
        ) -> None:
            normalized_values = _normalize_values(values)

            if any(
                value < min_value or value > max_value for value in normalized_values
            ):
                p.error(
                    f"Option '{option_string}' takes only values between {min_value} and {max_value}. {normalized_values} is not accepted."
                )

            if len(normalized_values) == 1:
                value: float | int | list[float | int] = normalized_values[0]
            else:
                value = normalized_values

            setattr(args, self.dest, value)

    return CustomActionInRange


# custom formatter
class Formatter(argparse.HelpFormatter):  # pragma: no cover
    """Custom help formatter that preserves raw text blocks."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """Append default value information to the help string.

        Parameters
        ----------
        action : argparse.Action
            Command line option definition.

        Returns
        -------
        str | None
            Help string with the default value appended when appropriate.
        """
        helper = action.help
        if helper is not None and "%(default)" not in helper:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]

                if action.option_strings or action.nargs in defaulting_nargs:
                    helper += "\n - default: %(default)s"
                # uncomment if type is needed to be shown in help message
                # if action.type:
                #     helper += "\n - type: %(type)s"

        return helper

    def _split_lines(self, text: str, width: int) -> list[str]:
        """Split help text while preserving raw-text blocks.

        Parameters
        ----------
        text : str
            Help message text.
        width : int
            Available line width.

        Returns
        -------
        list[str]
            Wrapped lines.
        """
        if text.startswith("R|"):
            return text[2:].splitlines()

        # pylint: disable=protected-access
        return argparse.HelpFormatter._split_lines(self, text, width)


# custom parser
def parser(name: str = "mfwater", **kwargs: Any) -> argparse.ArgumentParser:
    """Create the MFWater command line parser.

    Parameters
    ----------
    name : str, optional
        Program name reported in help and version output, by default "mfwater".
    **kwargs : Any
        Additional keyword arguments forwarded to `argparse.ArgumentParser`.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser instance.
    """

    p = argparse.ArgumentParser(
        prog=name,
        description="Program to prepare and execute multifidelity water simulations.",
        epilog="Written by Tom Frömbgen, Allan Kuhn, Jürgen Dölz and Barbara Kirchner (University of Bonn, Germany).",
        formatter_class=lambda prog: Formatter(prog, max_help_position=60),
        add_help=False,
        **kwargs,
    )
    p.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="R|Show this help message and exit.",
    )
    p.add_argument(
        "-a",
        type=str,
        choices=[
            "build",
            "chemmodel-prep",
            "chemmodel-post",
            "mfmc-prep",
            "model-select",
            "eval-estimator",
            "mfmc",
            "markov-chain",
            "mfda-smoke",
            "mfda-ncpu",
            "mfda-jobscript",
            "mfmc-run",
            "mfmc-jobscript",
        ],
        dest="algorithm",
        help="R|Which algorithm to execute.",
        default=None,
    )
    p.add_argument(
        "-c",
        "--config",
        type=is_file,
        dest="config",
        metavar="CONFIG_FILE",
        default=None,
        help="R|Configuration file in TOML format. Its keys are the option names (e.g. 'molecules = [1000, 500, 100]', 'orthoboxy = true'). Options in the file override those on the command line.",
    )
    p.add_argument(
        "-i",
        type=is_file,
        dest="input",
        metavar="INPUT_FILE",
        default=None,
        help="R|Input file in HDF5 format.",
    )
    p.add_argument(
        "-o",
        type=str,
        dest="output",
        metavar="OUTPUT_FILE",
        default="default.hdf5",
        help="R|Output file in HDF5 format.",
    )
    p.add_argument(
        "-p",
        "--params",
        type=str,
        dest="params",
        choices=["lj", "q", "lj-q"],
        default="lj",
        help="R|Which parameters to be perturbed in the simulations.\n'lj' for Lennard-Jones parameters, 'q' for partial charges.",
    )
    p.add_argument(
        "-z",
        "--orthoboxy",
        default=False,
        help="R|Whether to use tetragonal boxes (in OrthoBoXY shape) for the models.",
        action="store_true",
        dest="orthoboxy",
    )
    p.add_argument(
        "--models",
        type=int,
        dest="n_models",
        default=6,
        action=action_not_less_than(1),
        help="R|Number of models to be used. Ignored by markov-chain; the number of levels follows --molecules.",
    )
    p.add_argument(
        "--molecules",
        type=int,
        dest="n_molecules",
        default=None,
        action=action_not_less_than(1),
        help="R|Number of molecules per model. If more than 1 model, supply several values (space separated).",
        nargs="+",
    )
    p.add_argument(
        "--evals",
        type=int,
        dest="n_evals",
        default=None,
        action=action_not_less_than(1),
        help="R|Number of evaluations per model. If more than 1 model, supply several values (space separated).",
        nargs="+",
    )
    p.add_argument(
        "--budget",
        type=float,
        dest="budget",
        default=0,
        action=action_not_less_than(0),
        help="R|Computational budget required for the estimator.",
    )
    p.add_argument(
        "--mcchainlength",
        type=int,
        dest="n_mc_chain_length",
        default=1000,
        action=action_not_less_than(1),
        help="R|Number of steps M_1 of the Markov chain on the finest level (level 1) that are kept after the burn-in. The chain runs B + M_1 steps in total (B = --mcburnin).",
    )
    p.add_argument(
        "--mcsubchainlength",
        type=int,
        dest="n_mc_subchain_lengths",
        default=[10],
        action=action_not_less_than(1),
        nargs="+",
        help="R|Fixed subchain lengths on the coarser levels 2..eta (space separated, len(--molecules) - 1 values).\nEvery step on level i runs a subchain of this fixed length on level i+1.\nA single value is used for all coarse levels (default: 10).",
    )
    p.add_argument(
        "--mcburnin",
        type=int,
        dest="n_mc_burnin",
        default=100,
        action=action_not_less_than(0),
        help="R|Number of burn-in steps B on the finest level, run in addition to --mcchainlength (M_1), so B + M_1 steps in total. All entries on all levels produced during the first B steps (including subchains) are discarded. B = 0 is allowed (warning).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        dest="overwrite",
        default=False,
        help="R|Only for 'build': replace an existing output file without asking (needed in non-interactive sessions). Result files of other algorithms are never overwritten.",
    )
    p.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=None,
        help="R|Seed of the random number generator of the Markov chain ('markov-chain') or of the random parameters and seeds of 'chemmodel-prep' (stored in the HDF5 file). If omitted, a random seed is drawn and printed.",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        dest="models_dir",
        default="models",
        help="R|Directory holding the MD evaluations (<models-dir>/model_k/eval_j) of 'chemmodel-prep' and 'chemmodel-post'. Use a new directory for every phase, results of an earlier phase are never overwritten.",
    )
    p.add_argument(
        "--rmse-cost",
        type=str,
        dest="rmse_cost",
        choices=["calibration", "performance"],
        default=None,
        help="R|Costs w_i used for the RMSE and gamma of the MFMC estimator: 'calibration' (the costs the allocation was based on) or 'performance' (costs measured in the evaluation run). If omitted, the algorithm's default is used: 'performance' for 'mfmc' (reproduces paper I).",
    )
    p.add_argument(
        "--forward-model",
        type=str,
        dest="forward_model",
        choices=["dummy", "md"],
        default="dummy",
        help="R|Forward model of the Markov chain: 'dummy' (cheap stand-in) or 'md' (LAMMPS + TRAVIS + msdiff).",
    )
    p.add_argument(
        "--workdir",
        type=str,
        dest="workdir",
        default="./mfda_run",
        help="R|Run directory holding the manifest, the evaluation cache (and MD output) and the results.",
    )
    p.add_argument(
        "--chains",
        type=int,
        dest="n_chains",
        default=1,
        action=action_not_less_than(1),
        help="R|Number of independent chains, run concurrently as separate processes.",
    )
    p.add_argument(
        "--lammps-cmd",
        type=str,
        dest="lammps_cmd",
        default="mpirun -np {ncpu} lmp -i {input}",
        help="R|Command template to run LAMMPS (placeholders {ncpu} and {input}), only used with '--forward-model md'.",
    )
    p.add_argument(
        "--print-level",
        type=int,
        dest="print_level",
        choices=[0, 1],
        default=0,
        help="R|0 = silent, 1 = debug output of the MFDA chains",
    )
    p.add_argument(
        "--cluster",
        type=str,
        dest="cluster",
        default=None,
        help="R|Built-in cluster profile (e.g. 'marvin') for 'mfda-jobscript' and 'mfmc-jobscript'.",
    )
    p.add_argument(
        "--cluster-file",
        type=is_file,
        dest="cluster_file",
        metavar="CLUSTER_FILE",
        default=None,
        help="R|Cluster profile file (TOML) for 'mfda-jobscript' and 'mfmc-jobscript', alternative to --cluster.",
    )
    p.add_argument(
        "--jobscript",
        type=str,
        dest="jobscript",
        default=None,
        help="R|Output file of 'mfda-jobscript' and 'mfmc-jobscript' (default: run-mfda-<cluster>.sh or run-mfmc-<cluster>.sh). An existing file is never overwritten.",
    )
    p.add_argument(
        "--budgets",
        type=float,
        dest="budgets",
        nargs="+",
        default=None,
        help="R|Budgets of 'mfmc-run' in core-seconds (space separated); the performance phase is run for each.",
    )
    p.add_argument(
        "--protocol",
        type=str,
        dest="protocol",
        choices=["paper1", "mfda"],
        default="paper1",
        help="R|MD protocol of 'mfmc-run': 'paper1' (1 ns equilibration + 2 ns production, as paper I) or 'mfda' (shortened).",
    )
    p.add_argument(
        "--total-cpus",
        type=int,
        dest="total_cpus",
        default=None,
        action=action_not_less_than(1),
        help="R|Cores that 'mfmc-run' may use for concurrent MD evaluations (default: $SLURM_NTASKS, else the ranks of the largest system). Required by 'mfmc-jobscript'.",
    )
    p.add_argument(
        "--ncpu-override",
        type=str,
        dest="ncpu_override",
        nargs="+",
        default=None,
        metavar="N=RANKS",
        help="R|MPI ranks per number of molecules for 'mfmc-run', e.g. '1024=6 512=4' (TOML: ncpu_override = {1024 = 6, 512 = 4}). Others use calc_cpus. Not part of the cache key.",
    )
    p.add_argument(
        "--stop-after",
        type=str,
        dest="stop_after",
        choices=["calibration"],
        default=None,
        help="R|'mfmc-run': stop after the calibration and model selection and print the table (models, rho, sigma, w, selection, gamma, m* per budget).",
    )
    p.add_argument(
        "--calibration-from",
        type=is_file,
        dest="calibration_from",
        metavar="HDF5_FILE",
        default=None,
        help="R|'mfmc-run': skip the calibration and take sigma, rho and w from this paper-I-format calibration file.",
    )
    p.add_argument(
        "--replay-from",
        type=is_file,
        dest="replay_from",
        metavar="HDF5_FILE",
        default=None,
        help="R|'mfmc-run': use the stored parameters and seeds (/models/lj_params, charges, seeds) of this paper-I-format calibration file instead of drawing samples; the MD runs are done again.",
    )
    p.add_argument(
        "--replay-performance-from",
        type=str,
        dest="replay_performance_from",
        nargs="+",
        default=None,
        metavar="BUDGET=OMS_FILE",
        help="R|'mfmc-run': per budget a paper-I-format performance file whose stored parameters and seeds are evaluated again, e.g. '1e5=/path/oms_1e5/x_oms.hdf5' (TOML: replay_performance_from = {\"1e5\" = \"/path\"}).",
    )
    p.add_argument(
        "--replay-use-oms-allocation",
        action="store_true",
        dest="replay_use_oms_allocation",
        default=False,
        help="R|'mfmc-run': use the models and n_evals of the replayed performance files instead of the computed allocation. Without this flag a difference is an error.",
    )
    p.add_argument(
        "--claim-timeout",
        type=float,
        dest="claim_timeout",
        default=1800.0,
        help="R|'mfmc-run': seconds without heartbeat after which the claim of another (probably dead) job on an evaluation may be taken over.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"{name} {__version__}",
        help="R|Show version and exit.",
    )
    return p


def config_to_argv(p: argparse.ArgumentParser, options: dict[str, Any]) -> list[str]:
    """Translate the contents of a configuration file into command line tokens.

    Going through the parser (instead of updating the parsed namespace) keeps
    all type conversions, ``choices`` and value checks of the command line.

    Parameters
    ----------
    p : argparse.ArgumentParser
        Parser defining the valid options.
    options : dict[str, Any]
        Mapping from option name to value, as read from the TOML file. A key is
        either the destination name (e.g. ``n_molecules``) or the name of a
        long option (e.g. ``molecules``); ``-`` and ``_`` are interchangeable.

    Returns
    -------
    list[str]
        Command line tokens equivalent to the configuration file.

    Raises
    ------
    SystemExit
        If a key is unknown or a value has the wrong form (reported through
        ``p.error``, like an invalid command line).
    """
    # map every accepted spelling of a key to its action
    lookup: dict[str, argparse.Action] = {}
    # pylint: disable=protected-access
    for action in p._actions:
        if action.dest in ("help", "version", "config"):
            continue
        lookup[action.dest.replace("-", "_")] = action
        for opt in action.option_strings:
            lookup[opt.lstrip("-").replace("-", "_")] = action

    tokens: list[str] = []
    for key, value in options.items():
        match = lookup.get(key.replace("-", "_"))
        if match is None:
            p.error(f"unknown option '{key}' in configuration file")
        action = match

        # longest option string is the most descriptive one
        flag = max(action.option_strings, key=len)
        if action.nargs == 0:  # store_true flags take no value
            if not isinstance(value, bool):
                p.error(f"option '{key}' in configuration file must be true or false")
            if value:
                tokens.append(flag)
            continue
        if isinstance(value, dict) and action.nargs in ("+", "*"):
            # a TOML table becomes KEY=VALUE tokens (e.g. ncpu_override = {1024 = 6})
            value = [f"{k}={v}" for k, v in value.items()]
        values = value if isinstance(value, list) else [value]
        if isinstance(value, bool) or any(isinstance(v, (bool, dict)) for v in values):
            p.error(f"invalid value for option '{key}' in configuration file")
        tokens += [flag, *(str(v) for v in values)]

    return tokens


def parse_args(
    p: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    """Parse command line arguments, including an optional TOML configuration file.

    The configuration file is read after the first pass over the command line.
    Its options are appended to the command line and parsed again, so they
    override the options given on the command line (as in the AIMD setup tool).

    Parameters
    ----------
    p : argparse.ArgumentParser
        Parser created by `parser`.
    argv : Sequence[str], optional
        Command line arguments, by default ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    Raises
    ------
    SystemExit
        If the command line or the configuration file is invalid.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(args_list)
    if args.config is None:
        return args

    try:
        with open(args.config, "rb") as f:
            options = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        p.error(f"cannot parse configuration file '{args.config}': {exc}")

    return p.parse_args(args_list + config_to_argv(p, options))
