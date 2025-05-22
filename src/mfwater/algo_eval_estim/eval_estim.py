"""
Compute the number of evaluations given budget to pass to the chemical model again.
"""

import argparse

import h5py
import numpy as np

from ..algo_input import check_input_file


def evaluate_estimator(args: argparse.Namespace) -> int:
    """Given budget, model correlation and weights, this function computes the number of evaluations per model.

    Parameters:
    ----------
    args: argparse.Namespace
        Command line arguments. Budget is required for the computation.

    Raises
    ------
    KeyError
        If the budget is not provided in the arguments.
    ValueError
        If the first model does not have a correlation or alpha of 1.0.
    ValueError
        If the budget is too low to evaluate the high-fidelity model at least once.

    Returns
    -------
    int
        0 if the function runs successfully.
    """

    if args.budget == None:
        raise KeyError("Please add a computational budget.")

    check_input_file(args.input, args.algorithm)

    with h5py.File(args.input, "r+") as f:

        models = f["models"]
        models.attrs["budget"] = args.budget

        # get all the models that are groups
        model_items = [
            (name, mod) for name, mod in models.items() if isinstance(mod, h5py.Group)
        ]
        # sort the models by correlation, else the computation of r is wrong.
        ordered_models = sorted(
            model_items, key=lambda x: x[1].attrs["correlation"] ** 2, reverse=True
        )

        # get weights and correlations
        # add an additional 0 to the correlations for a technically non-existing model
        weights = np.array([mod.attrs["computation_time"] for _, mod in ordered_models])
        correlations = np.array(
            [mod.attrs["correlation"] for _, mod in ordered_models] + [0]
        )
        # debug
        # print(correlations)
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

        # compute the ratio vector
        ratio = differences / weights
        r = np.sqrt(ratio / ratio[0])
        # debug
        # print(r)

        # compute the optimal number of evaluations
        # round down to the nearest integer
        high_fidelity_eval = args.budget / np.dot(weights, r)
        evaluations = np.floor(
            np.concatenate(([high_fidelity_eval], r[1:] * high_fidelity_eval))
        ).astype(int)
        # debug
        # print(evaluations)
        if evaluations[0] < 1:
            raise ValueError(
                "Increase the budget. The high-fidelity model has to be evaluated at least once."
            )
        if not all(
            evaluations[i] <= evaluations[i + 1] for i in range(len(evaluations) - 1)
        ):
            raise ValueError("The evaluations are not sorted in ascending order.")
        # initialize alpha vector for mf estimator
        stds = np.array([mod.attrs["std"] for _, mod in model_items])
        # debug
        # print(stds)
        correlations = correlations[:-1]
        alpha = (correlations / stds) * stds[0]
        # debug
        # print(alpha[0])
        if np.round(alpha[0], 6) != 1.0:
            raise ValueError("The first model must have an alpha of 1.0.")

        # save old attributes and datasets, remove deprecated ones
        models["lj_params_initial"] = models["lj_params"]
        models["seeds_initial"] = models["seeds"]
        del models["lj_params"]
        del models["seeds"]

        for k, (_, mod) in enumerate(ordered_models):

            # save the old mean, std and diffusion_coeff to be able to compare them later
            mod.attrs["mean_initial"] = mod.attrs["mean"]
            mod.attrs["std_initial"] = mod.attrs["std"]
            mod.attrs["n_evals_initial"] = mod.attrs["n_evals"]
            mod["diffusion_coeff_initial"] = mod["diffusion_coeff"]

            # and remove deprecated attributes / data sets
            del mod.attrs["mean"]
            del mod.attrs["std"]
            del mod["diffusion_coeff"]

            # update the attributes in the models
            mod.attrs["n_evals"] = evaluations[k]
            mod.attrs["alpha"] = alpha[k]

        # print output to user
        print("Estimated optimal number of evaluations:")
        print(
            f"{'Model':<8}  {'Mols':>7}  {'Evals(init.)':>12}  {'Mean(init.)':>12}  {'Std(init.)':>12}  {'Evals(opt.)':>12}"
        )
        print("-" * 73)
        for _, (name, mod) in enumerate(ordered_models):
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals_initial']:12d}  {mod.attrs['mean_initial']:12.6f}  {mod.attrs['std_initial']:12.6f}  {mod.attrs['n_evals']:12d}"
            )

        # add last executed algorithm to the file
        f.attrs["last_algo"] = args.algorithm

    return 0
