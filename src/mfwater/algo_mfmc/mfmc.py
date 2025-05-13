"""
Implementation of the Multifidelity Monte Carlo (MFMC) algorithm.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from ..algo_input import check_input_file

def multifidelity_monte_carlo(args: argparse.Namespace) -> int:
    """Compute the estimator after everything has run.

    Args:
        args (argparse.Namespace): The input. 

    Raises:
        ValueError: The attribute 'n_eval' is not actually equal to the number of evaluations

    Returns:
        int: The exit code
    """
    check_input_file(args.input, args.algorithm)

    with h5py.File(args.input, "r+") as f:
        models = f["models"]
        model_items = [
            (name, mod)
            for name, mod in models.items()
            if isinstance(mod, h5py.Group)
        ]
        MC_estim = [0 for _ in range(models.attrs["n_models"])]
            #Compute the MC estimator for each model.
        MC_estim_pre = [0 for _ in range(models.attrs["n_models"]-1)]
            #Compute the MC estimator for each model, but with fewer evaluations.
        highfidelity_model = model_items[0][1]
        alpha = [
            highfidelity_model.attrs["std"]
            * (mod.attrs["correlation"] / mod.attrs["std"])
            for _, mod in model_items
        ]
        prev_mod = None
        for k, (name, mod) in enumerate(model_items):
            diffs = np.array(mod["diffusion_coeff"][:])
            if mod.attrs["n_eval"] != len(diffs):
                raise ValueError("Something went wrong with the assignment of 'n_eval'")
            MC_estim[k] = np.mean(diffs)
            if k > 0:
                MC_estim_pre[k-1] = np.mean(diffs[: models[prev_mod].attrs["n_eval"]])
            prev_mod = name
        s = (
            sum(
                alpha[l] * (MC_estim[l] - MC_estim_pre[l-1])
                for l in range(1, models.attrs["n_models"])
            )
            + MC_estim[0]
        )
        f["models"].attrs["Estimator"] = s
    return 0