"""Set up a Markov chain for MFWater."""

from __future__ import annotations

import argparse
import numpy as np

from ..algo_input import check_input_file
from ..argparser import constants

def markov_chain_eval(args: argparse.Namespace) -> int:
    """Set up the Markov chain state for MFWater.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing the path to the input file.

    Returns
    -------
    int
        Exit code, ``0`` for success.
    """

    check_input_file(args.input, args.algorithm)

    # initial parameter distribution
    # for our specific case, the initial distribution is a Gaussian distribution centered around the standard values of the LJ parameters and charges of the OPC3 water model with a standard deviation of 1/300 of the corresponding mean value for the LJ parameters and 1/50 of the corresponding mean value for the charges. 
    # This is therefore equivalent to the proposal Kernel with the same parameters. The initial guess is then a random sample from this distribution.
    initial_sample = proposal_kernel(
        np.array([constants.OPC3_EPSILON_OO, constants.OPC3_SIGMA_OO, constants.OPC3_CHARGE_O])
    )

    return 0


def proposal_kernel(
    means: np.ndarray,
    noise: np.ndarray = np.array(
        [
            constants.OPC3_EPSILON_OO * constants.NOISE_LJPARAMS,
            constants.OPC3_SIGMA_OO * constants.NOISE_LJPARAMS,
            abs(constants.OPC3_CHARGE_O) * constants.NOISE_CHARGES,
        ]
    ),
) -> np.ndarray:
    """Generate a proposal kernel for the Markov chain.

    Parameters
    ----------
    means : np.ndarray
        The mean values for Gaussians from which the new sample is proposed.
    noise : np.ndarray
        The standard deviations for the Gaussians from which the new sample is proposed. By default, it is set to 1/300 of the corresponding mean value for the LJ parameters and 1/50 of the corresponding mean value for the charges.

    Returns
    -------
    np.ndarray
        The array with random samples.

    Raises
    ------
    ValueError
        If the input arrays `means` and `noise` do not have the same shape.
    """

    # check whether means and noise have the same dimensions
    if means.shape != noise.shape:
        raise ValueError("The input arrays means and noise must have the same shape.")

    # create a random number generator
    rng = np.random.default_rng()

    # empty array to store the proposed parameters
    params = np.empty_like(means)

    # set up the Gaussians
    for i in range(len(means)):
        params[i] = rng.normal(loc=means[i], scale=noise[i])

    return params
