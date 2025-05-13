"""
Select models for the multifidelity Monte Carlo (MFMC) algorithm.
"""
import argparse
import math

import h5py
import numpy as np

# @CodingAllan use this to check the input file
# @CodingAllan these functions should probably take the entire argparse.Namespace as input, compare my implementations
from ..algo_input import check_input_file

#Function also takes args.output. Where we want to save the new h5py file.
def select_optimal_models(args: argparse.Namespace) -> int:
    """
    Given a collection of models, we extract the optimal model subset that accelerates the computation of the expectation.

    Parameters:
    ----------
        args: argparse.Namespace:
            The command line arguments

    Raises:
        ValueError: Error when sorting the correlation coefficients

    Returns:
        int: the exit code
    """
    
    check_input_file(args.input, args.algorithm)

    with h5py.File(args.input, "r") as f:
        models = f["models"]
        to_order = models.attrs["n_models"] - 1
        model_items = [
            (name, group)
            for name, group in models.items()
            if isinstance(group, h5py.Group)
        ]

        permutation = [
        name for name, _ in sorted(
            model_items, key=lambda x: x[1].attrs["correlation"], reverse=True
            )
        ]

        if permutation[0] != model_items[0][0]:
            raise ValueError(
                "\rho_{1,1} is always the largest. Something is wrong with the correlation coefficients!"
            )
        highfidelity_model = model_items[0][1]
        v_star = highfidelity_model.attrs["std"] ** 2
        M_star = [model_items[0][0]]  # List of optimal model names
        for z in range(to_order + 1):
            # Go over all subsets of size 'z'
            order_index = [
                to_order - z + j for j in range(1, z + 1)
            ]  # Start with largest possible index in the set ${(i_1,...,i_z)\in {1,...,to_order}^z: i_1 <i_2<...<i_2}$
            c = [0 for k in range(z)]
            for j in range(
                math.comb(to_order, z)
            ):  # Iterate over the set ${(i_1,...,i_z)\in {1,...,to_order}^z: i_1 <i_2<...<i_2}.$
                # This guarantees that we iterate over all subsets of size z, such that \rho_{0,permutation[i_1]}>\rho_{0,permutation[i_2]}>...>\rho_{0,permutaion[i_z]}.
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
                    highfidelity_model.attrs["std"] ** 2 / weights[0]
                ) * np.dot(
                    np.sqrt(weights), np.sqrt(differences)
                ) ** 2  # Compute MSE given optimal coefficients
                if current_mse < v_star:
                    M_star = cur_models
                    v_star = current_mse
        with h5py.File(args.output, "a") as g:
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
