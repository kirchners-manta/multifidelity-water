"""
Entrypoint for command line interface.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..argparser import parser


def console_entry_point(argv: Sequence[str] | None = None) -> int:
    """Get the command line arguments and parse them.
    This function is the entry point for the command line interface.
    It is called by the `mfwater` command.

    Parameters
    ----------
    argv : Sequence[str]
        The command line arguments, by default None.

    Returns
    -------
    int
        The exit code of the program, by default 0.
    """
    args = parser().parse_args(argv)

    print(args)

    return 0
