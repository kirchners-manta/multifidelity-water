"""Prepare input data for the Multifidelity Monte Carlo (MFMC) algorithm."""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from .. import mfmc_core
from ..algo_input import check_input_file, model_groups


def multifidelity_preparation(args: argparse.Namespace) -> int:
    """
    Compute mean, standard deviation and correlation of every model.

    The models are ordered by decreasing number of molecules (never by the h5py name
    order), so that the first model is the high-fidelity model. The statistics are
    computed by :func:`mfwater.mfmc_core.model_statistics` (SI Algorithm 2) and stored
    as the attributes ``mean``, ``std`` and ``correlation`` of the model groups.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    int
        Exit code, ``0`` for success.

    Raises
    ------
    ValueError
        If the models do not all have the same number of evaluations.
    """
    check_input_file(args.input, args.algorithm)

    with h5py.File(args.input, "r+") as f:
        model_items = model_groups(f["models"])

        # The correlation needs sample j of every model to belong to the same
        # parameters and seeds, so every model needs the same number of samples.
        # Uneven numbers (n_i >= n_hf) could be supported in the future.
        n_eval = model_items[0][1].attrs["n_evals"]
        for _, mod in model_items:
            if mod.attrs["n_evals"] != n_eval:
                raise ValueError(
                    "To compute the correlation we need the same number of samples from each model!"
                )

        diffs = np.array([mod["diffusion_coeff"][:] for _, mod in model_items])
        mean, std, rho = mfmc_core.model_statistics(diffs)

        # compute everything first, write afterwards: a failure leaves the file intact
        for k, (_, mod) in enumerate(model_items):
            mod.attrs["mean"] = mean[k]
            mod.attrs["std"] = std[k]
            mod.attrs["correlation"] = rho[k]

        print("MFMC preparation:")
        print(
            f"{'Model':<8}  {'Mols':>7}  {'Evals':>12}  {'Mean':>12}  {'Std':>12}  {'Correlation':>12}"
        )
        print("-" * 73)
        for name, mod in model_items:
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals']:12d}  {mod.attrs['mean']:12.6f}  {mod.attrs['std']:12.6f}  {mod.attrs['correlation']:12.6f}"
            )

        f.attrs["last_algo"] = args.algorithm

    return 0
