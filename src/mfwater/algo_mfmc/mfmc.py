"""
Implementation of the Multifidelity Monte Carlo (MFMC) algorithm.
"""

from pathlib import Path

import h5py
import numpy as np

# @CodingAllan use this to check the input file
# @CodingAllan these functions should probably take the entire argparse.Namespace as input, compare my implementations
# from ..algo_input import check_input_file


def evaluate_estimator(path: str | Path, budget: float) -> int:
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


def multifidelity_monte_carlo(path: str | Path) -> float:
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
