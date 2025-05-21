"""
Module to build an .hdf5 file with default input for MF-Water.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import h5py


def build_default_input(args: argparse.Namespace) -> int:
    """Create an input file for the MF-Water program with default values.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing the path to the input file.

    Returns
    -------
    int
        The exit code of the function, by default 0.
    """

    # if no output file is specified, use the default name
    if args.output is None:
        args.output = "default.hdf5"

    # check if specified output file already exists
    if Path(args.output).is_file():
        print(
            f"File '{args.output}' already exists. Do you want to overwrite it? (y/n)"
        )
        overwrite = input().strip().lower()
        if overwrite != "y":
            print("Exiting.")
            return 1

    # default settings for molecules and evaluations
    if args.n_molecules is None:
        args.n_molecules = [
            2 ** (args.n_models + 4 - i) for i in range(1, args.n_models + 1)
        ]
    if args.n_evals is None:
        args.n_evals = [100 for _ in range(args.n_models)]

    # check if the input model specifications are valid
    # number of models / molecules
    if type(args.n_molecules) is int:
        args.n_molecules = [args.n_molecules]
    if len(args.n_molecules) != args.n_models:
        raise ValueError(
            f"Number of molecules ({len(args.n_molecules)}) does not match number of models ({args.n_models})."
        )
    # also make sure that the numbers of molecules are ordered n_1 > n_2 > ... > n_n
    if args.n_molecules != sorted(args.n_molecules, reverse=True) or len(
        set(args.n_molecules)
    ) != len(args.n_molecules):
        raise ValueError(
            f"Number of molecules ({args.n_molecules}) is not ordered. Please provide a list of integers in descending order."
        )

    # evaluations
    # the number of evaluations has to be 0 < m_1 <= m_2 <= ... <= m_n
    if type(args.n_evals) is int:
        args.n_evals = [args.n_evals]
    if len(args.n_evals) != args.n_models:
        raise ValueError(
            f"Number of evaluations ({len(args.n_evals)}) does not match number of models ({args.n_models})."
        )
    if args.n_evals != sorted(args.n_evals):
        raise ValueError(
            f"Number of evaluations ({args.n_evals}) is not ordered. Please provide a list of integers in ascending or equal order. The last model must have the highest number of evaluations."
        )

    with h5py.File(Path(args.output), "w") as f:

        # create a group for the models with an attribute being the number of models
        models = f.create_group("models")
        models.attrs["n_models"] = args.n_models
        # create a subroup for each model
        # add the number of model evaluations and the number of molecules
        for i in range(1, models.attrs["n_models"] + 1):
            model = models.create_group(f"model_{i}")
            model.attrs["n_evals"] = args.n_evals[i - 1]
            model.attrs["n_molecules"] = args.n_molecules[i - 1]

        model_items = [
            (name, mod) for name, mod in models.items() if isinstance(mod, h5py.Group)
        ]

        # print output to user
        print(f"Input file '{args.output}' created with the following settings:")
        print(f"{'Model':<8}  {'Mols':>7}  {'Evals':>12}")
        print("-" * 31)
        for _, (name, mod) in enumerate(model_items):
            print(
                f"{name:<8}  {mod.attrs['n_molecules']:7d}  {mod.attrs['n_evals']:12d}"
            )

        # add last executed algorithm to the file
        f.attrs["last_algo"] = args.algorithm

    return 0


def check_input_file(input: str | Path, algo: str) -> None:
    """Check if the input file exists and is valid.

    Parameters
    ----------
    input : str | Path
        The path to the input file.
    algo : str
        The name of the algorithm.

    Returns
    -------
    None
    None
    """

    # check input file
    if input is None:
        raise RuntimeError("No input file given.")
    elif Path(input).exists() is False:
        raise FileNotFoundError(f"Input file {input} does not exist.")

    if algo not in ["chemmodel-prep", "eval-estimator"]:
        # check for the model directory
        if not (Path.cwd() / "models").exists():
            raise RuntimeError(
                "No models directory found. Run chemical_model_prep first."
            )

    class ModelInfo(TypedDict):
        """Helper class to define the model information."""

        attrs: list[str]
        datasets: list[str]

    class AlgoInput(TypedDict):
        """Helper class to define the algorithm input."""

        group: list[str]
        attrs: list[str]
        datasets: list[str]
        last_algo: list[str]
        models: ModelInfo

    # dictionary to store which attributes and datasets are required for each algorithm
    req_attrs: dict[str, AlgoInput] = {
        "chemmodel-prep": {
            "group": ["models"],
            "attrs": ["n_models"],
            "datasets": ["lj_params", "seeds"],
            "last_algo": ["build", "eval-estimator"],
            "models": {
                "attrs": ["n_evals", "n_molecules"],
                "datasets": [],
            },
        },
        "chemmodel-post": {
            "group": ["models"],
            "attrs": ["n_models"],
            "datasets": ["lj_params", "seeds"],
            "last_algo": ["chemmodel-prep", "chemmodel-post", "mfmc-prep"],
            "models": {"attrs": ["n_evals", "n_molecules"], "datasets": []},
        },
        "mfmc-prep": {
            "group": ["models"],
            "attrs": ["n_models"],
            "datasets": ["lj_params", "seeds"],
            "last_algo": ["chemmodel-post", "mfmc-prep"],
            "models": {
                "attrs": [
                    "n_evals",
                    "n_molecules",
                    "computation_time",
                    "mean",
                    "std",
                ],
                "datasets": [
                    "diffusion_coeff",
                ],
            },
        },
        "model-select": {
            "group": ["models"],
            "attrs": ["n_models"],
            "datasets": ["lj_params", "seeds"],
            "last_algo": ["mfmc-prep"],
            "models": {
                "attrs": [
                    "n_evals",
                    "n_molecules",
                    "correlation",
                    "mean",
                    "std",
                    "computation_time",
                ],
                "datasets": [
                    "diffusion_coeff",
                ],
            },
        },
        "eval-estimator": {
            "group": ["models"],
            "attrs": ["n_models"],
            "datasets": ["lj_params", "seeds"],
            "last_algo": ["model-select"],
            "models": {
                "attrs": [
                    "n_evals",
                    "n_molecules",
                    "correlation",
                    "mean",
                    "std",
                    "computation_time",
                ],
                "datasets": [
                    "diffusion_coeff",
                ],
            },
        },
        "mfmc": {
            "group": ["models"],
            "attrs": ["n_models", "budget"],
            "datasets": ["lj_params", "lj_params_initial", "seeds", "seeds_initial"],
            "last_algo": ["chemmodel-post", "mfmc"],
            "models": {
                "attrs": [
                    "alpha",
                    "n_evals",
                    "n_evals_initial",
                    "n_molecules",
                    "correlation",
                    "mean",
                    "mean_initial",
                    "std",
                    "std_initial",
                    "computation_time",
                    "computation_time_initial",
                ],
                "datasets": [
                    "diffusion_coeff",
                    "diffusion_coeff_initial",
                ],
            },
        },
    }

    # check if the right attributes are present in the input file
    with h5py.File(input, "r") as f:
        # check if the lastly executed algorithm is accepted
        if "last_algo" in f.attrs.keys():
            if f.attrs["last_algo"] not in req_attrs[algo]["last_algo"]:
                raise RuntimeError(
                    f"Last executed algorithm '{f.attrs['last_algo']}' is not compatible with the current selected algorithm '{algo}'.\nPossible last executed algorithms are: {req_attrs[algo]['last_algo']}"
                )
        # check if the group is present
        for group in req_attrs[algo]["group"]:
            if group not in f.keys():
                raise RuntimeError(f"Group {group} not found in input file.")
        # check if the attributes are present
        for attr in req_attrs[algo]["attrs"]:
            if attr not in f["models"].attrs.keys():
                raise RuntimeError(f"Attribute {attr} not found in input file.")
        # check if the models are present
        for model in f["models"].keys():
            # check if the model is valid
            if isinstance(model, h5py.Group):
                for attr in req_attrs[algo]["models"]["attrs"]:
                    if attr not in f["models"][model].attrs.keys():
                        raise RuntimeError(
                            f"Attribute {attr} not found in input file for model {model}."
                        )
                # check if the datasets are present
                for dataset in req_attrs[algo]["models"]["datasets"]:
                    if dataset not in f["models"][model].keys():
                        raise RuntimeError(
                            f"Dataset {dataset} not found in input file for model {model}."
                        )
        # we require constant n_evals for mfmc-prep. In the future, one could look into having uneven number of samples
        # as long as n_evals of every low fidelity is larger or equal to n_eval of highfidelity model.
        if algo == "mfmc-prep":
            model_items = [
                (name, mod)
                for name, mod in f["models"].items()
                if isinstance(mod, h5py.Group)
            ]
            n_eval = model_items[0][1].attrs["n_evals"]
            for _, mod in model_items:
                if mod.attrs["n_evals"] != n_eval:
                    raise ValueError(
                        "To compute the correlation we need the same number of samples from each model!"
                    )
        # we require that the high-fidelity model is at least evaluated once and a a decreasing ordering on the number of evaluations for "mfmc".
        if algo == "mfmc":
            model_items = [
                (name, mod)
                for name, mod in f["models"].items()
                if isinstance(mod, h5py.Group)
            ]
            ordered_models = sorted(
                model_items, key=lambda x: x[1].attrs["correlation"] ** 2, reverse=True
            )
            evals = [mod.attrs["n_evals"] for _, mod in ordered_models]
            if evals[0] < 1:
                raise ValueError(
                    "Increase the budget. The high-fidelity model has to be evaluated at least once."
                )
            if not all(evals[i] <= evals[i + 1] for i in range(len(evals) - 1)):
                raise ValueError(
                    "Something went wrong with the model selection. The condition on the ratio between cost and weights is not fulfilled."
                )
