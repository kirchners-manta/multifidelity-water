"""Select the optimal model subset for the MFMC algorithm."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import h5py

from .. import mfmc_core
from ..algo_input import attr_array, check_input_file, model_groups


def select_optimal_models(args: argparse.Namespace) -> int:
    """
    Select the optimal model subset for estimating the expectation.

    The selection itself is :func:`mfwater.mfmc_core.select_models` (SI Algorithm 3).
    The selected models are written to a **new** file ``args.output`` and renamed
    ``model_1``, ``model_2``, ... in the order of decreasing squared correlation, which is
    the order the later algorithms (``eval-estimator``, ``mfmc``) require.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments. ``args.input`` is the prepared file and
        ``args.output`` the file to create.

    Returns
    -------
    int
        Exit code, ``0`` for success.

    Raises
    ------
    FileExistsError
        If the output file already exists (existing files are never overwritten).
    ValueError
        If the first model does not have a correlation of 1 or the data are invalid.
    """
    # refuse before any work is done, as the MFDA algorithm does
    if Path(args.output).exists():
        raise FileExistsError(
            f"Output file '{args.output}' exists, refusing to overwrite it. "
            "Choose another file with -o."
        )

    check_input_file(args.input, args.algorithm)

    with h5py.File(args.input, "r") as f:
        # models by decreasing size: index 0 is the high-fidelity model
        model_items = model_groups(f["models"])
        std = attr_array(model_items, "std")
        rho = attr_array(model_items, "correlation")
        weights = attr_array(model_items, "computation_time")

        # indices into model_items, in the order of decreasing rho**2
        selected, _ = mfmc_core.select_models(std[0], rho, weights)

        # Copy only the selected groups (plus the datasets of the `models` group) into
        # the new file. Copying them one by one under their new names avoids the name
        # collisions that renaming inside a copy of all models could cause.
        with h5py.File(args.output, "x") as g:
            gm = g.create_group("models")
            for key, value in f["models"].attrs.items():
                gm.attrs[key] = value
            for name, item in f["models"].items():
                if isinstance(item, h5py.Dataset):
                    f["models"].copy(item, gm, name=name)
            for count, idx in enumerate(selected, start=1):
                f["models"].copy(model_items[idx][1], gm, name=f"model_{count}")
            gm.attrs["n_models"] = len(selected)
            g.attrs["last_algo"] = args.algorithm

    print(f"Optimal models selected and saved to '{args.output}':")
    print(f"{'Model':<8}  {'Mols':>7}  {'Evals':>12}  {'Mean':>12}  {'Std':>12}")
    print("-" * 59)
    with h5py.File(args.output, "r") as g:
        # the new names are model_1 .. model_n in selection order, so this is not the
        # alphabetical order of h5py but the selection order
        for count in range(1, len(selected) + 1):
            name = f"model_{count}"
            mod = g["models"][name]
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals']:12d}  {mod.attrs['mean']:12.6f}  {mod.attrs['std']:12.6f}"
            )

    return 0


def get_ordered_index_combinations(n: int) -> list[tuple[int, ...]]:
    """
    Generate all combinations of indices from 1 to ``n - 1``.

    Parameters
    ----------
    n : int
        The upper limit of the range.

    Returns
    -------
    list of tuple of int
        A list of tuples, each representing a combination of indices.

    Raises
    ------
    ValueError
        If n is less than 2.
    """
    if n < 2:
        raise ValueError("N must be at least 2")

    indices = list(range(1, n))  # start from index 1
    all_combinations = []

    for r in range(1, n):  # lengths from 1 to n-1
        comb = list(itertools.combinations(indices, r))
        all_combinations.extend(comb)

    return all_combinations
