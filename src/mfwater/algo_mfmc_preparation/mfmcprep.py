"""
Multifidelity Monte Carlo (MFMC) preparation module.
"""

import argparse

import h5py
import numpy as np

from ..algo_input import check_input_file


def multifidelity_preparation(args: argparse.Namespace) -> int:
    """
    Prepares the execution of the multifidelity Monte Carlo (MFMC) algorithm.

    Parameters:
    ----------
    args : argparse.Namespace
        The command line arguments

    Returns:
    -------
    int
        The exit code of the program.
    """

    # check input file
    check_input_file(args.input, args.algorithm)

    # open input file and process information from chemical model
    with h5py.File(args.input, "r+") as f:

        for k, mod in enumerate(f["models"].keys()):

            diffs = np.array(f["models"][mod]["diffusion_coeff"][:])

            # MC estimator is just the mean of the diffusion coefficients
            mc_estim = np.mean(diffs)

            # compute the standard deviation of the diffusion coefficients
            mc_std = np.sqrt(np.mean((diffs - mc_estim) ** 2))

            # add attributes to the model
            f["models"][mod].attrs["mean"] = mc_estim
            f["models"][mod].attrs["std"] = mc_std

            # compute the correlation coefficient w.r.t. the high-fidelity model
            if k == 0:
                corr = 1.0
            else:
                corr = np.mean(
                    (diffs - mc_estim)
                    * (
                        f["models"]["model_1"]["diffusion_coeff"][:]
                        - f["models"]["model_1"].attrs["mean"]
                    )
                ) / (mc_std * f["models"]["model_1"].attrs["std"])

            # add attributes to the model
            f["models"][mod].attrs["correlation"] = corr

            # debug
            # print(
            #     f"Model {mod}: mean = {mc_estim}, std = {mc_std}, correlation = {corr}"
            # )

    return 0
