"""
Select models for the multifidelity Monte Carlo (MFMC) algorithm.
"""

import argparse
import itertools
import math
import shutil
import subprocess

import h5py
import numpy as np

from ..algo_input import check_input_file

# def select_optimal_models(args: argparse.Namespace) -> int:
#     """
#     Given a collection of models, we extract the optimal model subset that accelerates the computation of the expectation.

#     Parameters:
#     ----------
#         args: argparse.Namespace:
#             The command line arguments

#     Raises:
#         ValueError: Error when sorting the correlation coefficients

#     Returns:
#         int: the exit code
#     """

#     check_input_file(args.input, args.algorithm)

#     with h5py.File(args.input, "r") as f:

#         models = f["models"]
#         to_order = models.attrs["n_models"] - 1
#         model_items = [
#             (name, group)
#             for name, group in models.items()
#             if isinstance(group, h5py.Group)
#         ]

#         permutation = [
#             name
#             for name, _ in sorted(
#                 model_items, key=lambda x: x[1].attrs["correlation"], reverse=True
#             )
#         ]

#         if permutation[0] != model_items[0][0]:
#             raise ValueError(
#                 "\rho_{1,1} is always the largest. Something is wrong with the correlation coefficients!"
#             )
#         highfidelity_model = model_items[0][1]
#         v_star = highfidelity_model.attrs["std"] ** 2
#         m_star = [model_items[0][0]]  # List of optimal model names

#         for z in range(to_order + 1):
#             # Go over all subsets of size 'z'
#             order_index = [
#                 to_order - z + j for j in range(1, z + 1)
#             ]  # Start with largest possible index in the set ${(i_1,...,i_z)\in {1,...,to_order}^z: i_1 <i_2<...<i_z}$
#             c = [0] * z
#             for _ in range(
#                 math.comb(to_order, z)
#             ):  # Iterate over the set ${(i_1,...,i_z)\in {1,...,to_order}^z: i_1 <i_2<...<i_z}.$
#                 # This guarantees that we iterate over all subsets of size z, such that \rho_{0,permutation[i_1]}>\rho_{0,permutation[i_2]}>...>\rho_{0,permutaion[i_z]}.
#                 for k in range(z):
#                     if c[k] >= math.comb(order_index[k] - 1, k):
#                         if order_index[k] > k + 1:
#                             order_index[k] -= 1
#                             order_index[:k] = [order_index[k] - k + j for j in range(k)]
#                             c[: k + 1] = [0 for j in range(k + 1)]
#                 c = [a + 1 for a in c]
#                 cur_models = [model_items[0][0]] + [
#                     permutation[order_index[k]] for k in range(z)
#                 ]  # current models that satisfy the condition on the correlations

#                 # get weights and correlations
#                 weights = np.array(
#                     [
#                         models[cur_models[i]].attrs["computation_time"]
#                         for i in range(z + 1)
#                     ]
#                 )
#                 # add an additional 0 here for a technically non-existing model
#                 correlations = np.array(
#                     [models[cur_models[i]].attrs["correlation"] for i in range(z + 1)]
#                     + [0]
#                 )
#                 # differences in squared correlations
#                 differences = np.array(
#                     [
#                         (correlations[q] ** 2 - correlations[q + 1] ** 2)
#                         for q in range(z + 1)
#                     ]
#                 )

#                 # check if model satisfies the cost / correlation condition, skip if not
#                 skip_model = False
#                 for i in range(1, z + 1):
#                     if (
#                         weights[i - 1] * differences[i]
#                         <= weights[i] * differences[i - 1]
#                     ):
#                         skip_model = True
#                         break
#                 if skip_model:
#                     continue

#                 # compute the MSE with optimal coefficients
#                 current_mse = (
#                     highfidelity_model.attrs["std"] ** 2 / weights[0]
#                 ) * np.dot(np.sqrt(weights), np.sqrt(differences)) ** 2
#                 # if the MSE is smaller than the current best, update the best models
#                 if current_mse < v_star:
#                     m_star = cur_models
#                     v_star = current_mse

#         # write optimal models to file
#         with h5py.File(args.output, "w") as g:
#             g.create_group("models")
#             g["models"].attrs["n_models"] = f["models"].attrs["n_models"]
#             for name, mod in f["models"].items():
#                 if name in m_star:
#                     f.copy(mod, g["models"], name=name)

#     print(
#         f"Optimal models selected and saved to {args.output} with the following settings:\n"
#     )
#     # if h5dump is available, print the content of the file
#     if shutil.which("h5dump") is not None:
#         subprocess.run(f"h5dump -n 1 {args.output}", shell=True)
#     else:
#         print("h5dump not available. Cannot print the content of the file.")

#     return 0


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

        # order the models by correlation coefficient
        models = f["models"]
        ordered_models = sorted(
            models.items(), key=lambda x: x[1].attrs["correlation"], reverse=True
        )
        # debug
        # print(ordered_models)

        # check if the first model is the one with the highest correlation
        if ordered_models[0][1].attrs["correlation"] != 1.0:
            raise ValueError(
                "The first model must have a correlation of 1.0 with itself."
            )

        # get group of the high-fidelity model
        highfidelity_model = ordered_models[0][1]
        # initialize the list of optimal models with the name of the high-fidelity model
        m_star = [ordered_models[0][0]]
        # initialize the error as the variance of the high-fidelity model
        v_star = highfidelity_model.attrs["std"] ** 2

        # get combinations of indices for model selection
        combinations = get_ordered_index_combinations(len(ordered_models))
        # debug
        # print(combinations)

        # select current models
        for combo in combinations:
            current_models = [ordered_models[0][0]] + [
                ordered_models[i][0] for i in combo
            ]
            # debug
            # print([i for i in combo])
            # print(current_models)

            # get weights and correlations
            weights = np.array(
                [
                    models[current_models[i]].attrs["computation_time"]
                    for i in range(len(current_models))
                ]
            )
            # debug
            # print(len(weights), weights)

            # add an additional 0 here for a technically non-existing model
            correlations = np.array(
                [
                    models[current_models[i]].attrs["correlation"]
                    for i in range(len(current_models))
                ]
                + [0]
            )
            # debug
            # print(len(correlations), correlations)

            # differences in squared correlations
            differences = np.array(
                [
                    (correlations[i] ** 2 - correlations[i + 1] ** 2)
                    for i in range(len(current_models))
                ]
            )
            # debug
            # print(len(differences), differences)

            # check if model satisfies the cost / correlation condition, skip if not
            skip_model = False
            for i in range(1, len(current_models)):
                if weights[i - 1] * differences[i] <= weights[i] * differences[i - 1]:
                    skip_model = True
                    break
            if skip_model:
                continue

            # compute the MSE with optimal coefficients
            current_mse = (highfidelity_model.attrs["std"] ** 2 / weights[0]) * np.dot(
                np.sqrt(weights), np.sqrt(differences)
            ) ** 2
            # if the MSE is smaller than the current best, update the best models
            if current_mse < v_star:
                m_star = current_models
                v_star = current_mse

        # write optimal models to file
        with h5py.File(args.output, "w") as g:
            g.create_group("models")
            g["models"].attrs["n_models"] = f["models"].attrs["n_models"]
            for name, mod in f["models"].items():
                if name in m_star:
                    f.copy(mod, g["models"], name=name)

    print(
        f"Optimal models selected and saved to {args.output} with the following settings:\n"
    )
    # if h5dump is available, print the content of the file
    if shutil.which("h5dump") is not None:
        subprocess.run(f"h5dump -n 1 {args.output}", shell=True)
    else:
        print("h5dump not available. Cannot print the content of the file.")

    return 0


def get_ordered_index_combinations(n: int) -> list[tuple[int, ...]]:
    """Generate all combinations of indices from 1 to n. Index 0 is excluded (useful for model selection).

    Parameters
    ----------
    n : int
        The upper limit of the range.

    Returns
    -------
    list[tuple[int, ...]]
        A list of tuples, each representing a combination of indices.
    """
    if n < 2:
        raise ValueError("N must be at least 2")

    indices = list(range(1, n))  # start from index 1
    all_combinations = []

    for r in range(1, n):  # lengths from 1 to n-1
        comb = list(itertools.combinations(indices, r))
        all_combinations.extend(comb)

    return all_combinations
