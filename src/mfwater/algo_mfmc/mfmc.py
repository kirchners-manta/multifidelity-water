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
        # models have to be sorted by the correlation coefficient.
        ordered_models = sorted(
            model_items, key=lambda x: x[1].attrs["correlation"] ** 2, reverse=True
        )
        # write out the means and alphas for the MFMC estimator
        # mean is to be computed. We dont want to run MFMC prep again
        diffs = [np.array(mod["diffusion_coeff"][:]) for _, mod in ordered_models]
        means = np.array([np.mean(diff) for diff in diffs])
        evals = [mod.attrs["n_evals"] for _, mod in ordered_models]

        # compute the mean for the first n_eval evaluations of the previous model.
        means_lower = np.array(
            [np.mean(diffs[i][: evals[i - 1]]) for i in range(1, len(ordered_models))]
        )
        alphas = np.array([mod.attrs["alpha"] for _, mod in ordered_models])

        # compute the MFMC estimator
        # the second mean must be the MC estimator for the same model, but diff number of samples!
        mfmc_estim = means[0] + np.sum(alphas[1:] * (means[1:] - means_lower))

        f["models"].attrs["mfmc_estimator"] = mfmc_estim

    # print output to user
    print(f"MFMC Estimator: {mfmc_estim}")

    return 0
