"""Project-wide constants for MFWater."""

from __future__ import annotations

from typing import Final

# natural constants
KJ2KCAL: Final[float] = 0.239006  # convert kJ to kcal
NA: Final[float] = 6.022e23  # atoms/mol

# OPC3 constants
WATER_MASS: Final[float] = 18.01528  # g/mol
WATER_DENSITY_298: Final[float] = 0.997  # g/cm^3 at 298 K and 1 atm
OPC3_CHARGE_O: Final[float] = -0.895200  # in e
OPC3_SIGMA_OO: Final[float] = 3.17427  # in Angstrom
OPC3_EPSILON_OO: Final[float] = 0.68369 * KJ2KCAL  # in kcal/mol

# Multifidelity settings
NOISE_LJPARAMS: Final[float] = 1 / 300  # factor with which the mean value of the corresponding Gaussian to model the noise in LJ parameters is multiplied. 1/300 means that 99.7% of the samples are within +-1.0% of the values of the standard LJ parameters
NOISE_CHARGES: Final[float] = 1 / 50  # factor with which the mean value of the corresponding Gaussian to model the noise in the charges is multiplied.

# other
ORTHOBOXY_RATIO: Final[float] = 2.7933596497  # OrthoBoXY ratio of lz/lx = lz/ly
