"""
Entrypoint for command line interface.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..algo_chemical_model import chemical_model_post, chemical_model_prep
from ..argparser import parser
from ..algo_input import build_default_input
from ..algo_mfmc_preparation import multifidelity_preparation


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

    if args.algorithm == "build":
        return build_default_input(args)
    elif args.algorithm == "chemmodel-prep":
        return chemical_model_prep(args)
    elif args.algorithm == "chemmodel-post":
        return chemical_model_post(args)
    elif args.algorithm == "mfmcprep":
        return multifidelity_preparation(args)

    return 0
