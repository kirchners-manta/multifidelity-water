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
        n_models = models.attrs["n_models"]

        # get all the models that are groups
        model_items = [
            (name, mod) for name, mod in models.items() if isinstance(mod, h5py.Group)
        ]

        # get weights and correlations
        # add an additional 0 to the correlations for a technically non-existing model
        weights = np.array([mod.attrs["computation_time"] for _, mod in model_items])
        correlations = np.array(
            [mod.attrs["correlation"] for _, mod in model_items] + [0]
        )
        # debug
        # print(correlations)
        if correlations[0] != 1.0:
            raise ValueError("The first model must have a correlation of 1.0.")

        # compute differences in squared correlations
        differences = np.array(
            [(correlations[q] ** 2 - correlations[q + 1] ** 2) for q in range(n_models)]
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

        # initialize alpha vector for mf estimator
        stds = np.array([mod.attrs["std"] for _, mod in model_items])
        correlations.pop()  # remove the last element which is 0
        alpha = correlations / stds * stds[0]
        # debug
        # print(alpha)
        if alpha[0] != 1.0:
            raise ValueError("The first model must have an alpha of 1.0.")

        # update the attributes in the models
        for k, (_, mod) in enumerate(model_items):
            mod.attrs["n_eval"] = evaluations[k]
            mod.attrs["alpha"] = alpha[k]

            # save the old mean, std and diffusion_coeff to be able to compare them later
            mod.attrs["mean_initial"] = mod.attrs["mean"]
            mod.attrs["std_initial"] = mod.attrs["std"]
            mod.attrs["velocity_seed_initial"] = mod.attrs["velocity_seed"]
            mod.attrs["packmol_seed_initial"] = mod.attrs["packmol_seed"]
            mod["diffusion_coeff_initial"] = mod["diffusion_coeff"]
            mod["lj_params_initial"] = mod["lj_params"]

            # and remove deprecated attributes / data sets
            del mod.attrs["mean"]
            del mod.attrs["std"]
            del mod.attrs["velocity_seed"]
            del mod.attrs["packmol_seed"]
            del mod["diffusion_coeff"]
            del mod["lj_params"]

        # print information for the user
        print("Optimal number of evaluations have been estimated:")
        for name, mod in model_items:
            print(
                f"{name}: {mod.attrs['n_molecules']} molecules, {mod.attrs['n_eval']} evaluations"
            )

    return 0
