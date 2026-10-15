"""Estimate optimal model evaluations from a given budget."""

from __future__ import annotations

import argparse

import h5py

from .. import mfmc_core
from ..algo_input import attr_array, check_input_file, models_by_rho


def evaluate_estimator(args: argparse.Namespace) -> int:
    """
    Compute the optimal number of evaluations per model.

    The models are ordered by decreasing squared correlation. The allocation is
    :func:`mfwater.mfmc_core.allocate` (SI Algorithm 4) and the control-variate
    coefficients are :func:`mfwater.mfmc_core.control_variates`, both with standard
    deviations and correlations in the **same** order. Everything is computed before the
    file is changed, so a failing run leaves it untouched.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments. ``args.budget`` is the computational budget in the unit
        of the model costs (core-seconds).

    Returns
    -------
    int
        Exit code, ``0`` for success.

    Raises
    ------
    ValueError
        If the first model does not have a correlation of 1, the budget is not positive
        or too low to evaluate the high-fidelity model at least once, or the models
        violate the cost/correlation condition (evaluations not ascending).
    """
    check_input_file(args.input, args.algorithm)

    with h5py.File(args.input, "r+") as f:
        models = f["models"]

        # order by rho: the allocation and alpha are only correct in this order
        ordered_models = models_by_rho(models)
        weights = attr_array(ordered_models, "computation_time")
        rho = attr_array(ordered_models, "correlation")
        std = attr_array(ordered_models, "std")

        evaluations, _ = mfmc_core.allocate(rho, weights, float(args.budget))
        alpha = mfmc_core.control_variates(rho, std)

        models.attrs["budget"] = args.budget

        # save old attributes and datasets, remove deprecated ones
        if "lj_params" in models:
            models["lj_params_initial"] = models["lj_params"]
            del models["lj_params"]
        if "seeds" in models:
            models["seeds_initial"] = models["seeds"]
            del models["seeds"]
        if "charges" in models:
            models["charges_initial"] = models["charges"]
            del models["charges"]

        for k, (_, mod) in enumerate(ordered_models):
            # keep the calibration mean, std, diffusion_coeff and cost to compare later
            mod.attrs["mean_initial"] = mod.attrs["mean"]
            mod.attrs["std_initial"] = mod.attrs["std"]
            mod.attrs["n_evals_initial"] = mod.attrs["n_evals"]
            mod.attrs["computation_time_initial"] = mod.attrs["computation_time"]
            mod["diffusion_coeff_initial"] = mod["diffusion_coeff"]

            # and remove deprecated attributes / data sets
            del mod.attrs["mean"]
            del mod.attrs["std"]
            del mod["diffusion_coeff"]

            mod.attrs["n_evals"] = int(evaluations[k])
            mod.attrs["alpha"] = alpha[k]

        print("Estimated optimal number of evaluations:")
        print(
            f"{'Model':<8}  {'Mols':>7}  {'Evals(init.)':>12}  {'Mean(init.)':>12}  {'Std(init.)':>12}  {'Evals(opt.)':>12}"
        )
        print("-" * 73)
        for name, mod in ordered_models:
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals_initial']:12d}  {mod.attrs['mean_initial']:12.6f}  {mod.attrs['std_initial']:12.6f}  {mod.attrs['n_evals']:12d}"
            )

        f.attrs["last_algo"] = args.algorithm

    return 0
