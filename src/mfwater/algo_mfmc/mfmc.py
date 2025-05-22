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

        # to compute the error of the MFMC estimator, we need the original correlations and weights again
        correlations = np.array(
            [mod.attrs["correlation"] for _, mod in ordered_models] + [0]
        )
        weights = np.array([mod.attrs["computation_time"] for _, mod in ordered_models])
        # debug
        # print(correlations)
        # print(weights)
        if correlations[0] != 1.0:
            raise ValueError("The first model must have a correlation of 1.0.")

        # compute differences in squared correlations
        # models have to be ordered for this
        differences = np.array(
            [
                (correlations[q] ** 2 - correlations[q + 1] ** 2)
                for q in range(len(ordered_models))
            ]
        )
        # debug
        # print(differences)

        # mfmc error
        mfmc_error = np.sqrt(
            ordered_models[0][1].attrs["std_initial"] ** 2
            / f["models"].attrs["budget"]
            * np.sum(np.sqrt(weights * differences)) ** 2
        )

        # set attributes for the models
        f["models"].attrs["mfmc_estimator"] = mfmc_estim
        f["models"].attrs["mfmc_error"] = mfmc_error
        for i, (_, mod) in enumerate(ordered_models):
            mod.attrs["mean"] = means[i]

        # print output to user
        print("Calculation of the MFMC estimator:")
        print(
            f"{'Model':<8}  {'Mols':>7}  {'Evals(init.)':>12}  {'Mean(init.)':>12}  {'Std(init.)':>12}  {'Evals(opt.)':>12}  {'Mean(opt.)':>12}"
        )
        print("-" * 87)
        for _, (name, mod) in enumerate(ordered_models):
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals_initial']:12d}  {mod.attrs['mean_initial']:12.6f}  {mod.attrs['std_initial']:12.6f}  {mod.attrs['n_evals']:12d}  {mod.attrs['mean']:12.6f}"
            )
        print("-" * 87)
        print(f"{'MFMC Estimator'} {' '*17} {mfmc_estim:12.6f}  {mfmc_error:12.6f}")

        # add last executed algorithm to the file
        f.attrs["last_algo"] = args.algorithm

    return 0
