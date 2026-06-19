"""Forward model for the multifidelity MCMC algorithm."""

from __future__ import annotations

import numpy as np

from ..argparser import constants


def forward_model_dummy(n_molecules: int, parameters: np.ndarray) -> float:
    """Dummy forward model for the chemical Metropolis-Hastings algorithm.

    Parameters
    ----------
    n_molecules : int
        Number of molecules in the system for the given fidelity level.
    parameters : np.ndarray
        Parameter sample for which to compute the forward model output.

    Returns
    -------
    float
        Dummy output of the forward model.
    """

    # dummy output, in a real implementation this would be the result of a simulation with the given parameters and number of molecules

    # generate dummy output from the distribution of measurements
    output = np.random.normal(
        loc=constants.TARGET_DIFFUSION_COEFFICIENT.mean() * (np.log2(n_molecules) / 10),
        scale=constants.TARGET_DIFFUSION_COEFFICIENT.std(),
    )

    return output
