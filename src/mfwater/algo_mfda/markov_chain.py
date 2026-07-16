"""Set up a Markov chain for a Multifidelity Delayed Acceptance MCMC."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from ..algo_input import check_input_file
from ..argparser import constants
from .multifidelity_mcmc import multifidelity_markov_chain, proposal_kernel


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

    # check_input_file(args.input, args.algorithm)

    # initialize Markov chain and proposed samples with NaN
    # both are of shape n_models x max_subchain_length ** n_models x n_parameters
    # for the proposed samples, the low-fidelity model (first dimension, last entry) will be left empty.
    markov_chain = np.full(
        (args.n_models, args.n_mc_chain_length**args.n_models, 3), #n_mc_chain_length müsste zu einer liste werden
        np.nan,
        dtype=np.float64,
    )
    proposed_samples = np.full(
        (args.n_models, args.n_mc_chain_length**args.n_models, 3),
        np.nan,
        dtype=np.float64,
    )

    # initial parameter distribution
    # for our specific case, the initial distribution is a Gaussian distribution centered around the standard values of the LJ parameters and charges of the OPC3 water model with a standard deviation of 1/300 of the corresponding mean value for the LJ parameters and 1/50 of the corresponding mean value for the charges.
    # This is therefore equivalent to the proposal Kernel with the same parameters. The initial guess is then a random sample from this distribution.
    # Only put noise on the parameters that are specified by the user
    # initial parameter distribution
    means = np.array(
        [constants.OPC3_EPSILON_OO, constants.OPC3_SIGMA_OO, constants.OPC3_CHARGE_O]
    )
    param_map = {
        "eps": (
            constants.OPC3_EPSILON_OO * constants.NOISE_LJPARAMS,
            args.params in ["lj", "lj-q"],
        ),
        "sig": (
            constants.OPC3_SIGMA_OO * constants.NOISE_LJPARAMS,
            args.params in ["lj", "lj-q"],
        ),
        "q": (
            abs(constants.OPC3_CHARGE_O) * constants.NOISE_CHARGES,
            args.params in ["q", "lj-q"],
        ),
    }
    noise_scales = np.array(
        [scale if apply else 0.0 for scale, apply in param_map.values()]
    )
    initial_sample = proposal_kernel(means, noise_scales)

    # initialize multiindex that keeps track of how often functions that generate Markov chains are called
    fcount = [0] * args.n_models

    # initialize empty cache to store computed diffusion coefficient and save compute time
    diff_computed: dict[Any, float] = {}

    # start Markov chain at the high-fidelity model
    fidelity = 1

    # random number for (sub)chain length
    l = np.random.randint(1, args.n_mc_chain_length + 1) # Etwas fixes, großes. 100 oder 1000 samples z.B.

    # generate Markov chain
    _, markov_chain, proposed_samples, fcount, diff_computed = (
        multifidelity_markov_chain(
            initial_sample,
            markov_chain,
            proposed_samples,
            diff_computed,
            args.n_molecules,
            fidelity,
            l,
            args.n_mc_chain_length,
            fcount,
            kernel_noise=noise_scales,
        )
    )

    # how many samples in the Markov chain in have been written on each fidelity level
    n_samples = np.sum(~np.any(np.isnan(markov_chain), axis=2), axis=1)

    # warn if any fidelity level has no samples
    for i, n in enumerate(n_samples):
        if n == 0:
            print(f"WARNING: No samples found for fidelity level {i+1}")

    # compute the MCMC estimator
    # for the highest fidelity level, use only the markov chain (proposed_samples is NaN)
    # for lower fidelity levels, use markov_chain - proposed_samples
    mcmc_estim = np.zeros(3)
    for i in range(args.n_models):
        if i == args.n_models - 1:
            # highest fidelity: use only markov_chain
            contrib = np.nansum(markov_chain[i] / n_samples[i], axis=0)
        else:
            # lower fidelity: use difference
            contrib = np.nansum(
                (markov_chain[i] - proposed_samples[i]) / n_samples[i], axis=0
            )
        mcmc_estim += contrib
        print(
            f"Model {i+1}, contribution to eps, sig, q: {contrib[0]:10.6f} {contrib[1]:10.6f} {contrib[2]:10.6f}"
        )

    print(
        f"\nMFDA-MCMC estimator:                  {mcmc_estim[0]:10.6f} {mcmc_estim[1]:10.6f} {mcmc_estim[2]:10.6f}"
    )

    # debug
    print(f"Evaluated total of {len(diff_computed)} samples.")
    # print(diff_computed)

    return 0
