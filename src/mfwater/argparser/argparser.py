# Part of the AIMD setup tool

"""
Parser for command line options.
"""

#############################################

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .. import __version__


# file and directory checks
def is_file(path: str | Path) -> str | Path:
    """Function to check if a file exists and is not a directory.

    Parameters
    ----------
    path : str | Path
        Path to the file.

    Returns
    -------
    str | Path
        Path to the file.

    Raises
    ------
    argparse.ArgumentTypeError
        If the file does not exist or is a directory.
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
    """Function to check if a directory exists and is not a file.

    Parameters
    ----------
    path : str | Path
        Path to the directory.

    Returns
    -------
    str | Path
        Path to the directory.

    Raises
    ------
    argparse.ArgumentTypeError
        If the directory does not exist or is a file.
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


# custom actions
def action_not_less_than(min_value: float = 0.0) -> type[argparse.Action]:
    """Function to create a custom action for argparse that limits the possible input values.

    Parameters
    ----------
    min_value : float, optional
        Minimum input value, by default 0.0

    Returns
    -------
    type[argparse.Action]
        Custom action for argparse.
    """

    class CustomActionLessThan(argparse.Action):
        """
        Custom action for limiting possible input values. Raise error if value is smaller than min_value.
        """

        def __call__(
            self,
            p: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: list[float | int] | float | int,  # type: ignore
            option_string: str | None = None,
        ) -> None:

            if isinstance(values, (int, float)):
                values = [values]

            if any(value < min_value for value in values):
                p.error(
                    f"Option '{option_string}' takes only values larger than {min_value}. {values} is not accepted."
                )

            if len(values) == 1:
                values = values[0]

            setattr(args, self.dest, values)

    return CustomActionLessThan


def action_not_more_than(max_value: float = 0.0) -> type[argparse.Action]:
    """Function to create a custom action for argparse that limits the possible input values.

    Parameters
    ----------
    max_value : float, optional
        Maximum input value, by default 0.0

    Returns
    -------
    type[argparse.Action]
        Custom action for argparse.
    """

    class CustomActionMoreThan(argparse.Action):
        """
        Custom action for limiting possible input values. Raise error if value is larger than max_value.
        """

        def __call__(
            self,
            p: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: list[float | int] | float | int,  # type: ignore
            option_string: str | None = None,
        ) -> None:
            if isinstance(values, (int, float)):
                values = [values]

            if any(value > max_value for value in values):
                p.error(
                    f"Option '{option_string}' takes only values smaller than {max_value}. {values} is not accepted."
                )

            if len(values) == 1:
                values = values[0]

            setattr(args, self.dest, values)

    return CustomActionMoreThan


def action_in_range(
    min_value: float = 0.0, max_value: float = 1.0
) -> type[argparse.Action]:
    """Function to create a custom action for argparse that limits the possible input values.

    Parameters
    ----------
    min_value : float, optional
        Minimum input value, by default 0.0
    max_value : float, optional
        maximum, by default 1.0

    Returns
    -------
    type[argparse.Action]
        Custom action for argparse.
    """

    class CustomActionInRange(argparse.Action):
        """
        Custom action for limiting possible input values in a range. Raise error if value is not in range [min_value, max_value].
        """

        def __call__(
            self,
            p: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: list[float | int] | float | int,  # type: ignore
            option_string: str | None = None,
        ) -> None:
            if isinstance(values, (int, float)):
                values = [values]

            if any(value < min_value or value > max_value for value in values):
                p.error(
                    f"Option '{option_string}' takes only values between {min_value} and {max_value}. {values} is not accepted."
                )

            if len(values) == 1:
                values = values[0]

            setattr(args, self.dest, values)

    return CustomActionInRange


# custom formatter
class Formatter(argparse.HelpFormatter):
    """
    Custom format for help message.
    """

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """
        Append default value and type of action to help string.

        Parameters
        ----------
        action : argparse.Action
            Command line option.

        Returns
        -------
        str | None
            Help string.
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
        """
        Re-implementation of `RawTextHelpFormatter._split_lines` that includes
        line breaks for strings starting with 'R|'.

        Parameters
        ----------
        text : str
            Help message.
        width : int
            Text width.

        Returns
        -------
        list[str]
            Split text.
        """
        if text.startswith("R|"):
            return text[2:].splitlines()

        # pylint: disable=protected-access
        return argparse.HelpFormatter._split_lines(self, text, width)


# custom parser
def parser(name: str = "mfwater", **kwargs: Any) -> argparse.ArgumentParser:
    """
    Parses the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        Container for command line arguments.
    """

    p = argparse.ArgumentParser(
        prog="mfwater",
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
        choices=["chemmodel-prep", "chemmodel-post"],
        dest="algorithm",
        help="R|Which algorithm to execute.",
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
        "--version",
        action="version",
        version=f"{name} {__version__}",
        help="R|Show version and exit.",
    )
    return p
