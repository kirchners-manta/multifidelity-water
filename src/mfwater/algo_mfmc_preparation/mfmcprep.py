import argparse
import math
from pathlib import Path

import h5py
import numpy as np

from ..algo_chemical_model import chemical_model_post, chemical_model_prep
from ..algo_input import build_default_input


def setup_params(args: argparse.Namespace) -> str | Path:
    """
    Prepares input files and runs chemical model preparation/postprocessing steps.

    Args:
        args (argparse.Namespace): Parsed command-line arguments with keys like
            'output', 'n_models', 'n_molecules', and model evaluations 'm'.

    Returns:
        str | Path: Path to the generated input file.
    """
    args_forhdf5 = argparse.Namespace(
        output=args.output,
        n_models=args.n_models,
        n_molecules=args.n_molecules,
        n_evals=args.m,
    )
    build_default_input(args=args_forhdf5)
    args_forCM = argparse.Namespace(input=args_forhdf5.output)
    chemical_model_prep(args_forCM)
    chemical_model_post(args_forCM)

    return args_forCM.input


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
    if args.input is None:
        raise RuntimeError("No input file given.")
    elif Path(args.input).exists() is False:
        raise FileNotFoundError(f"Input file {args.input} does not exist.")

    # open input file and process information from chemical model
    with h5py.File(args.input, "r+") as f:

        for k, mod in enumerate(f["models"].keys()):

            diffs = np.array(f["models"][mod]["diffusion_coeff"][:])

            # MC estimator is just the mean of the diffusion coefficients
            mc_estim = np.mean(diffs)

            # compute the variance of the diffusion coefficients
            mc_var = np.mean((diffs - mc_estim) ** 2)

            # add attributes to the model
            f["models"][mod].attrs["mean"] = mc_estim
            f["models"][mod].attrs[
                "variance"
            ] = mc_var  # !!! check if var or std is needed

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
                ) / np.sqrt(
                    mc_var * f["models"]["model_1"].attrs["variance"]
                )  # !!! check if var or std is needed

            # add attributes to the model
            f["models"][mod].attrs["correlation"] = corr

            # debug
            print(
                f"Model {mod}: mean = {mc_estim}, variance = {mc_var}, correlation = {corr}"
            )

    return 0


def OptimalModel_select(path: str | Path) -> int:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' does not exist.")

    with h5py.File(path, "r") as f:
        models = f["models"]
        to_order = models.attrs["n_models"] - 1
        model_items = [
            (name, group)
            for name, group in models.items()
            if isinstance(group, h5py.Group)
        ]
        for _, group in model_items:
            if (
                "correlation" not in group.attrs
                or "computation_time" not in group.attrs
                or "variance" not in group.attrs
            ):
                raise KeyError(
                    f"Missing required attributes ('correlation', 'computation_time', or 'variance') in model {name}"
                )

        sorted_models = sorted(
            model_items, key=lambda x: x[1].attrs["correlation"], reverse=True
        )
        permutation = [name for name, _ in sorted_models]
        if permutation[0] != model_items[0][0]:
            raise ValueError(
                "\rho_{1,1} is always the largest. Something is wrong with the correlation coefficients!"
            )
        highfidelity_model = models[model_items[0][0]]
        v_star = highfidelity_model.attrs["variance"] ** 2
        M_star = [model_items[0][0]]  # List of optimal model names
        for z in range(to_order + 1):
            # Go over all subsets of size 'z'
            order_index = [
                to_order - z + j for j in range(1, z + 1)
            ]  # Start with largest possible index
            c = [0 for k in range(z)]
            for j in range(
                math.comb(to_order, z)
            ):  # Iterate over the set ${(i_1,...,i_z)\in {1,...,to_order}^z: i_1 <i_2<...<i_2}$
                for k in range(z):
                    if c[k] >= math.comb(order_index[k] - 1, k):
                        if order_index[k] > k + 1:
                            order_index[k] -= 1
                            order_index[:k] = [order_index[k] - k + j for j in range(k)]
                            c[: k + 1] = [0 for j in range(k + 1)]
                c = [a + 1 for a in c]
                cur_models = [model_items[0][0]] + [
                    permutation[order_index[k]] for k in range(z)
                ]  # current models that satisfy the condition on the correlations
                weights = np.array(
                    [
                        models[cur_models[i]].attrs["computation_time"]
                        for i in range(z + 1)
                    ]
                )
                correlation = [
                    models[cur_models[i]].attrs["correlation"] for i in range(z + 1)
                ] + [0]
                differences = np.array(
                    [
                        (correlation[q] ** 2 - correlation[q + 1] ** 2)
                        for q in range(z + 1)
                    ]
                )
                skip_model = False
                for i in range(1, z + 1):
                    if (
                        weights[i - 1] * differences[i]
                        <= weights[i] * differences[i - 1]
                    ):
                        skip_model = True  # This model does not satisfy the assumptions of the Theorem. Skip this selection
                        break
                if skip_model:
                    continue
                current_mse = (
                    highfidelity_model.attrs["variance"] ** 2 / weights[0]
                ) * np.dot(
                    np.sqrt(weights), np.sqrt(differences)
                ) ** 2  # Compute MSE given optimal coefficients
                if current_mse < v_star:
                    M_star = cur_models
                    v_star = current_mse
        with h5py.File("optimal_models.hdf5", "a") as g:
            if "models" not in g:
                g.create_group("models")
            g["models"].attrs["n_models"] = models.attrs["n_models"]
            for k, (name, mod) in enumerate(models.items()):
                if name not in M_star:
                    continue
                if name in g["models"]:
                    del g["models"][name]
                f.copy(mod, g["models"], name=name)
    return 0


def Estimate_eval(path: str | Path, budget: float) -> int:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' does not exist.")

    with h5py.File(path, "r+") as f:
        models = f["models"]
        n_models = models.attrs["n_models"]
        model_items = list(models.items())
        weights = np.array([mod.attrs["computation_time"] for _, mod in model_items])
        correlation = [mod.attrs["correlation"] for _, mod in model_items] + [0]
        if correlation[0] != 1:
            raise ValueError("Correlation of the first model with itself must be 1")
        differences = np.array(
            [(correlation[q] ** 2 - correlation[q + 1] ** 2) for q in range(n_models)]
        )
        ratio = differences / weights
        r = np.sqrt(ratio / ratio[0])
        high_fidelity_eval = budget / np.dot(weights, r)
        evaluations = np.floor(
            np.concatenate(([high_fidelity_eval], r[1:] * high_fidelity_eval))
        ).astype(int)
        if evaluations[0] < 1:
            raise ValueError(
                "Increase the budget. The high-fidelity model has to be evaluated at least once."
            )
        for k, (_, mod) in enumerate(model_items):
            mod.attrs["n_eval"] = evaluations[k]
    return 0


def MultifidelityMonteCarlo(path: str | Path) -> float:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file '{path}' does not exist.")

    with h5py.File(path, "r") as f:
        models = f["models"]
        MC_estim = [0 for _ in range(models.attrs["n_models"])]
        MC_estim_pre = [0 for _ in range(models.attrs["n_models"])]
        highfidelity_model = list(models.items())[0][0]
        alpha = [
            models[highfidelity_model].attrs["variance"]
            * (models[mod].attrs["correlation"] / models[mod].attrs["variance"])
            for _, mod in models.items()
        ]
        prev_mod = None
        for k, (name, mod) in enumerate(models.items()):
            diffs = np.array(mod["diffusion_coeff"][:])
            if mod.attrs["n_eval"] != len(diffs):
                raise ValueError("Something went wrong with the assignment of 'n_eval'")
            MC_estim[k] = np.mean(diffs)
            if k > 0:
                MC_estim_pre[k] = np.mean(diffs[: models[prev_mod].attrs["n_eval"]])
            prev_mod = name
        s = (
            sum(
                alpha[l] * (MC_estim[l] - MC_estim_pre[l])
                for l in range(1, models.attrs["n_models"])
            )
            + MC_estim[0]
        )

    return s
