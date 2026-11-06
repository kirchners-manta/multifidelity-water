"""Finite-size correction of the self-diffusion coefficient (Yeh and Hummer).

The diffusion coefficient from a periodic box of side ``L`` underestimates the
infinite-system value by ``C = xi k_B T / (6 pi eta L)`` (Yeh and Hummer,
J. Phys. Chem. B 108, 15873 (2004)). The MFDA likelihood adds ``C_k`` to the
simulated ``D`` of level ``k`` (see :func:`~.multifidelity_mcmc.log_likelihood`).

Temperature and viscosity are the constant values of the msdiff call of paper I
(:data:`mfwater.md.pipeline.PAPER_I_MSDIFF_ARGS`), so ``C_k`` depends on the
molecule number but not on the force field parameters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..argparser import constants
from ..md.pipeline import PAPER_I_MSDIFF_ARGS
from ..md.system import calc_box_size

#: Accepted values of ``--fs-correction``.
FS_CORRECTIONS = ("yeh-hummer", "none")


def yeh_hummer_correction(n_molecules: int, orthoboxy: bool) -> float:
    """Finite-size correction ``C = xi k_B T / (6 pi eta L)`` of one level.

    Parameters
    ----------
    n_molecules : int
        Number of molecules in the box.
    orthoboxy : bool
        ``True`` for the tetragonal OrthoBoXY box. OrthoBoXY corrects the finite
        size effect itself, so the correction is zero by construction.

    Returns
    -------
    float
        Correction in 1e-9 m^2/s (the unit of ``D`` in the likelihood).
    """
    if orthoboxy:
        return 0.0
    # same box as the MD code builds, so L cannot drift from the simulation
    length = calc_box_size(n_molecules, orthoboxy_shape=False)[0] * 1e-10  # m
    # arguments of the msdiff call: ("--hummer", T in K, eta in Pa s, uncertainty)
    temperature = float(PAPER_I_MSDIFF_ARGS[1])
    viscosity = float(PAPER_I_MSDIFF_ARGS[2])
    corr = (
        constants.YEH_HUMMER_XI
        * constants.K_B
        * temperature
        / (6.0 * math.pi * viscosity * length)
    )
    return corr * 1e9  # m^2/s -> 1e-9 m^2/s


def resolve_fs_corrections(
    method: str, molecules: Sequence[int], orthoboxy: bool
) -> list[float]:
    """Corrections ``C_k`` of all levels for the chosen method.

    Parameters
    ----------
    method : str
        ``"yeh-hummer"`` or ``"none"``.
    molecules : Sequence[int]
        Molecule number per level.
    orthoboxy : bool
        Whether the boxes are tetragonal OrthoBoXY boxes.

    Returns
    -------
    list[float]
        One correction per level in 1e-9 m^2/s (all zero for ``"none"``).

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    if method == "none":
        return [0.0] * len(molecules)
    if method == "yeh-hummer":
        return [yeh_hummer_correction(int(n), orthoboxy) for n in molecules]
    raise ValueError(
        f"Unknown --fs-correction '{method}', choose from {list(FS_CORRECTIONS)}."
    )
