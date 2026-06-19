"""
Command line parser utilities for MFWater.
"""

#############################################

from __future__ import annotations

import argparse
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
        ],
        dest="algorithm",
        help="R|Which algorithm to execute.",
        default=None,
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
        help="R|Number of models to be used.",
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
        default=10,
        action=action_not_less_than(1),
        help="R|Maximum number of MC steps (length of the Markov chain) on each fidelity level.\nA random integer between 1 and (including) the specified value will be drawn.",
    )
    p.add_argument(
        "--mcburnin",
        type=int,
        dest="n_mc_burnin",
        default=100,
        action=action_not_less_than(0),
        help="R|Number of burn-in samples to be discarded from the Markov Chain.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"{name} {__version__}",
        help="R|Show version and exit.",
    )
    return p
