"""
Module to build an .hdf5 file with default input for MF-Water.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    if type(args.n_molecules) is int:
        args.n_molecules = [args.n_molecules]
    if len(args.n_molecules) != args.n_models:
        print(
            f"Number of molecules ({len(args.n_molecules)}) does not match number of models ({args.n_models})."
        )
        return 1
    if type(args.n_evals) is int:
        args.n_evals = [args.n_evals]
    if len(args.n_evals) != args.n_models:
        print(
            f"Number of evaluations ({len(args.n_evals)}) does not match number of models ({args.n_models})."
        )
        return 1

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

    # print output for user
    print(f"Input file '{args.output}' created with the following settings:")
    print(f"Number of models: {args.n_models}")
    print(f"Number of molecules: {args.n_molecules}")
    print(f"Number of evaluations: {args.n_evals}")

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
    int
        The exit code of the function, by default 0.
    """

    # check input file
    if input is None:
        raise RuntimeError("No input file given.")
    elif Path(input).exists() is False:
        raise FileNotFoundError(f"Input file {input} does not exist.")

    if algo != "chemmodel-prep":
        # check for the model directory
        if not (Path.cwd() / "models").exists():
            raise RuntimeError(
                "No models directory found. Run chemical_model_prep first."
            )

    # check if the right attributes are present in the input file
    if algo not in ["chemmodel-prep", "chemmodel-post", "mfmc-prep"]:
        with h5py.File(input, "r") as f:
            for name, group in f["models"].items():

                # debug
                # print(f"name: {name}, group: {group}")

                if isinstance(group, h5py.Group):
                    for attr in ["correlation", "computation_time", "std"]:
                        if attr not in group.attrs:
                            raise RuntimeError(
                                f"Missing required attribute '{attr}' in model {name}. Please run algorithm 'mfmc-prep'."
                            )
                else:
                    raise RuntimeError(
                        f"Model {name} is not an hdf5 group. Please run algorithm 'mfmc-prep'."
                    )
