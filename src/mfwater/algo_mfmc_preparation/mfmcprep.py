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

        # find model groups
        model_items = [
            (name, mod)
            for name, mod in f["models"].items()
            if isinstance(mod, h5py.Group)
        ]

        # iterate over the models
        for k, (name, mod) in enumerate(model_items):

            diffs = np.array(mod["diffusion_coeff"][:])

            # MC estimator is just the mean of the diffusion coefficients
            mc_estim = np.mean(diffs)

            # compute the standard deviation of the diffusion coefficients
            mc_std = np.sqrt(np.mean((diffs - mc_estim) ** 2))

            # add attributes to the model
            mod.attrs["mean"] = mc_estim
            mod.attrs["std"] = mc_std

            # compute the correlation coefficient w.r.t. the high-fidelity model
            if k == 0:
                corr = 1.0
            else:
                corr = np.mean(
                    (diffs - mc_estim)
                    * (
                        model_items[0][1]["diffusion_coeff"][:]
                        - model_items[0][1].attrs["mean"]
                    )
                ) / (mc_std * model_items[0][1].attrs["std"])

            # add attributes to the model
            mod.attrs["correlation"] = corr

        # print output to user
        print("MFMC preparation:")
        print(
            f"{'Model':<8}  {'Mols':>7}  {'Evals':>12}  {'Mean':>12}  {'Std':>12}  {'Correlation':>12}"
        )
        print("-" * 73)
        for _, (name, mod) in enumerate(model_items):
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals']:12d}  {mod.attrs['mean']:12.6f}  {mod.attrs['std']:12.6f}  {mod.attrs['correlation']:12.6f}"
            )

    return 0
