"""
Compute the number of evaluations given budget to pass to the chemical model again.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from ..algo_input import check_input_file


def evaluate_estimator(args: argparse.Namespace) -> int:
    """given budget, model correlation and weights, this function computes the number of evaluations per model.

    Args:
        args (argparse.Namespace): the input. Make sure that --budget is set

    Raises:
        KeyError: no budget set. 
        ValueError: Something went wrong with the computation of the correlation
        ValueError: Increase the budget. The high-fidelity model has to be evaluated atleast once

    Returns:
        int: the exit code
    """


    check_input_file(args.input, args.algorithm)
    
    if args.budget == None:
        raise KeyError("please add a budget for your computation time.")
    
    with h5py.File(args.input, "r+") as f:
        models = f["models"]
        n_models = models.attrs["n_models"]
        model_items = [
            (name, mod) for name, mod in models.items() if isinstance(mod, h5py.Group)
        ]
        weights = np.array([mod.attrs["computation_time"] for _, mod in model_items])
        
        correlation = [
            mod.attrs["correlation"] for _, mod in model_items
        ] + [0]
        
        if correlation[0] != 1.0:
            raise ValueError("Correlation of the first model with itself must be 1")
        differences = np.array(
            [
                (correlation[q] ** 2 - correlation[q + 1] ** 2) for q in range(n_models)
            ]
        )
        ratio = differences / weights
        r = np.sqrt(ratio / ratio[0]) 
        high_fidelity_eval = args.budget / np.dot(weights, r)
        evaluations = np.floor(
            np.concatenate(([high_fidelity_eval], r[1:] * high_fidelity_eval))
        ).astype(int)
        if evaluations[0] < 1:
            raise ValueError(
                "Increase the budget. The high-fidelity model has to be evaluated at least once."
            )
        for k, (_, mod) in enumerate(model_items):
            mod.attrs["n_eval"] = evaluations[k]
            mod.attrs["budget"] = args.budget
    return 0