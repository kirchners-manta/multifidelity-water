"""
Implementation of the Multifidelity Monte Carlo (MFMC) algorithm.
"""

import argparse

import h5py
import numpy as np

from ..algo_input import check_input_file


def multifidelity_monte_carlo(args: argparse.Namespace) -> int:
    """
    Compute the estimator after everything has run.

    Parameters:
    ----------
    args: argparse.Namespace
        The command line arguments, including the input file and algorithm type.

    Returns:
    -------
    int
        Exit code, 0 for success.
    """

    check_input_file(args.input, args.algorithm)

    with h5py.File(args.input, "r+") as f:

        # get the models and their names
        model_items = [
            (name, mod)
            for name, mod in f["models"].items()
            if isinstance(mod, h5py.Group)
        ]

        # write out the means and alphas for the MFMC estimator
        means = np.array([mod.attrs["mean"] for _, mod in model_items])
        alphas = np.array([mod.attrs["alpha"] for _, mod in model_items])

        # compute the MFMC estimator
        mfmc_estim = means[0] + np.sum(alphas[1:] * (means[1:] - means[:-1]))
        f["models"].attrs["mfmc_estimator"] = mfmc_estim

    # print output to user
    print(f"MFMC Estimator: {mfmc_estim}")

    return 0
