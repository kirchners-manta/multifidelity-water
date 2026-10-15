"""System geometry and parallel resources of one MD evaluation.

These helpers are shared by the paper-I workflow (``algo_chemical_model``), the
MFDA algorithm and the MFMC forward pipeline. They are re-exported from
``mfwater.algo_chemical_model.chemical_model`` for backward compatibility.
"""

from __future__ import annotations

from ..argparser import constants


def calc_box_size(
    n: int,
    orthoboxy_shape: bool,
    rho: float = constants.WATER_DENSITY_298,
    m: float = constants.WATER_MASS,
) -> list[float]:
    """
    Calculate the box size of an MD simulation box of molecules.

    Parameters
    ----------
    n : int
        Number of molecules in the box.
    orthoboxy_shape : bool
        If True, the box is tetragonal with lx = ly /= lz. Useful for OrthoBoXY simulations.
        If False, the box is cubic.
    rho : float, optional
        Density of the box, by default 0.997 g/cm^3 (for water at 298.15 K).
    m: float, optional
        Mass of the molecules, by default 18.01528 g/mol (for water).


    Returns
    -------
    list[float]
        The box size in x, y, and z direction in Angstrom.
    """

    # calculate the volume of the box in cm^3
    v = n * m / (rho * constants.NA)

    # calculate the box size in Angstrom
    if orthoboxy_shape:
        lx = ly = (v / constants.ORTHOBOXY_RATIO * 1e24) ** (1 / 3)
        lz = lx * constants.ORTHOBOXY_RATIO
    else:
        lx = ly = lz = (v * 1e24) ** (1 / 3)

    return [lx, ly, lz]


def calc_cpus(n: int) -> int:
    """
    Calculate the number of CPUs to use for the simulation.

    The thresholds come from a scaling benchmark on Marvin (Intel Sapphire Rapids
    nodes, shared, LAMMPS 23Jun2022, OPC3 water, 100 000 timed steps, 5 replicas):
    the largest core count whose parallel efficiency t(1) / (p * t(p)) stays at
    about 0.7 or above. N = 32 gains at most a factor 1.7 from any number of cores.
    The measured sizes were N = 32, 64, 128, 256, 512 and 1024; N >= 2000 was not
    measured (the value 8 is kept from the previous version).

    Parameters
    ----------
    n : int
        Number of molecules in the box.

    Returns
    -------
    int
        The number of CPUs to use.
    """
    if n >= 512:
        cpus = 8
    elif n >= 256:
        cpus = 4
    elif n >= 64:
        cpus = 2
    else:
        cpus = 1

    return cpus
