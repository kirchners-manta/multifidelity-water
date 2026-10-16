"""Implementation of the Multifidelity Monte Carlo (MFMC) algorithm."""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from .. import mfmc_core
from ..algo_input import attr_array, check_input_file, models_by_rho

#: Default cost basis of the RMSE of the legacy ``mfmc`` algorithm. The stored paper-I
#: values were computed with the costs measured in the performance run.
DEFAULT_RMSE_COST = "performance"


def multifidelity_monte_carlo(args: argparse.Namespace) -> int:
    """
    Compute the MFMC estimator, its RMSE and gamma after the evaluation runs.

    The models are ordered by decreasing squared correlation. The estimator is
    :func:`mfwater.mfmc_core.mfmc_estimate` (Eq. 19) with the control-variate
    coefficients ``alpha`` stored by ``eval-estimator``. The RMSE is
    :func:`mfwater.mfmc_core.mfmc_rmse` (Eq. 20) and ``gamma`` is
    :func:`mfwater.mfmc_core.gamma` (Eq. 22), both computed with the cost basis chosen by
    ``args.rmse_cost``:

    * ``performance`` (default): the costs ``computation_time`` measured in the
      evaluation (performance) run. This reproduces the paper-I ``mfmc_error``.
    * ``calibration``: the costs ``computation_time_initial`` of the calibration run, i.e.
      the ones the allocation was based on.

    The result attributes ``mfmc_estimator``, ``mfmc_error`` and ``gamma`` are written to
    the ``models`` group, as well as ``mfmc_error_calibration_cost`` and
    ``mfmc_error_performance_cost`` for every basis whose costs are available; likewise
    ``gamma_calibration_cost`` and ``gamma_performance_cost`` next to ``gamma`` (the
    value of the chosen basis).

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments, including the input file and algorithm type.
        ``args.rmse_cost`` is ``None`` (use the default ``performance``),
        ``"performance"`` or ``"calibration"``.

    Returns
    -------
    int
        Exit code, ``0`` for success.

    Raises
    ------
    ValueError
        If the first model does not have a correlation of 1, ``rmse_cost`` is unknown, or
        the data are inconsistent.
    RuntimeError
        If the costs of the chosen basis are not stored in the file.
    """
    check_input_file(args.input, args.algorithm)

    rmse_cost = getattr(args, "rmse_cost", None) or DEFAULT_RMSE_COST
    if rmse_cost not in ("calibration", "performance"):
        raise ValueError(
            f"rmse_cost must be 'calibration' or 'performance', got {rmse_cost!r}."
        )

    with h5py.File(args.input, "r+") as f:
        models = f["models"]
        # models have to be sorted by the correlation coefficient
        ordered_models = models_by_rho(models)

        diffs = [np.array(mod["diffusion_coeff"][:]) for _, mod in ordered_models]
        means = np.array([np.mean(diff) for diff in diffs])
        evals = np.array(
            [int(mod.attrs["n_evals"]) for _, mod in ordered_models], dtype=np.int64
        )
        alphas = attr_array(ordered_models, "alpha")
        rho = attr_array(ordered_models, "correlation")
        sigma1 = float(ordered_models[0][1].attrs["std_initial"])
        budget = float(models.attrs["budget"])

        mfmc_estim = mfmc_core.mfmc_estimate(diffs, evals, alphas)

        # costs of both bases, where available: the performance-run costs overwrite
        # `computation_time` in chemmod-post, the calibration costs were saved by
        # eval-estimator as `computation_time_initial` (absent in old paper-I files)
        errors: dict[str, float] = {}
        gammas: dict[str, float] = {}
        for basis, attr in (
            ("performance", "computation_time"),
            ("calibration", "computation_time_initial"),
        ):
            if not all(attr in mod.attrs for _, mod in ordered_models):
                continue
            w = attr_array(ordered_models, attr)
            errors[basis] = mfmc_core.mfmc_rmse(sigma1, rho, w, budget)
            gammas[basis] = mfmc_core.gamma(rho, w)
        if rmse_cost not in errors:
            raise RuntimeError(
                f"The costs for --rmse-cost {rmse_cost} are not stored in '{args.input}' "
                "(missing attribute "
                f"'{'computation_time_initial' if rmse_cost == 'calibration' else 'computation_time'}'"
                ")."
            )
        mfmc_error = errors[rmse_cost]

        models.attrs["mfmc_estimator"] = mfmc_estim
        models.attrs["mfmc_error"] = mfmc_error
        models.attrs["gamma"] = gammas[rmse_cost]
        models.attrs["rmse_cost"] = rmse_cost
        for basis, err in errors.items():
            models.attrs[f"mfmc_error_{basis}_cost"] = err
            models.attrs[f"gamma_{basis}_cost"] = gammas[basis]
        for i, (_, mod) in enumerate(ordered_models):
            mod.attrs["mean"] = means[i]

        print("Calculation of the MFMC estimator:")
        print(
            f"{'Model':<8}  {'Mols':>7}  {'Evals(init.)':>12}  {'Mean(init.)':>12}  {'Std(init.)':>12}  {'Evals(opt.)':>12}  {'Mean(opt.)':>12}"
        )
        print("-" * 87)
        for name, mod in ordered_models:
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals_initial']:12d}  {mod.attrs['mean_initial']:12.6f}  {mod.attrs['std_initial']:12.6f}  {mod.attrs['n_evals']:12d}  {mod.attrs['mean']:12.6f}"
            )
        print("-" * 87)
        print(f"{'MFMC Estimator'} {' '*17} {mfmc_estim:12.6f}  {mfmc_error:12.6f}")
        print(f"RMSE cost basis: {rmse_cost}; gamma = {gammas[rmse_cost]:.6f}")

        f.attrs["last_algo"] = args.algorithm

    return 0
