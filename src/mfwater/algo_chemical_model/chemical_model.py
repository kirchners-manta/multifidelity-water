"""
Implementation of the chemical model for the MFWater project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


def chemical_model_prep(args: argparse.Namespace) -> int:
    """
    Prepare the chemical model for the MFWater project.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    int
        The exit code of the program, by default 0.
    """

    # if no input file is given, create a default one
    if args.input is None:

        input_file = Path.cwd() / "default_input.hdf5"

        with h5py.File(input_file, "w") as f:
            # number of models
            f.create_dataset("n_models", data=np.int32(8))
            # number of model evaluations: a vector of size n_models with entries 100
            f.create_dataset(
                "n_evals", data=np.array([100] * f["n_models"][()], dtype=np.int32)
            )
            # number of molecules: a vector of size n_models with entries 2**11 to 2**4
            f.create_dataset(
                "n_molecules",
                data=np.array(
                    [
                        2 ** (f["n_models"][()] + 3 - i)
                        for i in range(f["n_models"][()])
                    ],
                    dtype=np.int32,
                ),
            )

        args.input = input_file

    # work with the given or created input file
    with h5py.File(args.input, "r+") as f:

        # check if the input file is valid
        if "n_models" not in f:
            raise ValueError("Input file does not contain n_models dataset.")
        if "n_evals" not in f:
            raise ValueError("Input file does not contain n_evals dataset.")
        if "n_molecules" not in f:
            raise ValueError("Input file does not contain n_molecules dataset.")

        # fill the sampl_lj_params dataset with random samples
        for i in range(f["n_models"][()]):

            f.create_dataset(
                f"sampl_lj_params_{i}",
                data=sampl_lj_params(np.zeros((2, f["n_evals"][i]), dtype=np.float32)),
            )

    return 0


def chemical_model_post(args: argparse.Namespace) -> int:
    """
    Post-process the chemical model for the MFWater project.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    int
        The exit code of the program, by default 0.
    """

    return 0


def sampl_lj_params(ar: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Generate random samples for the Lennard-Jones parameters from a Gaussian distribution.

    Parameters
    ----------
    ar : np.ndarray
        The array to fill with random samples.

    Returns
    -------
    np.ndarray
        The array with random samples.
    """
    # define the mean and standard deviation of the Gaussian distribution
    # from the values of the standard LJ parameters of the OPC3 water model
    sig_opc3 = 3.17427  # Angstrom
    eps_opc3 = 0.68369  # kJ/mol

    # create an array of size n_models x n_evals with random samples
    # from a Gaussian distribution with mean sigma/epsilon and standard deviation 1/6 * sigma/epsilon
    # this ensures that 99.7% of the samples are within +-1/2 of the values of the standard LJ parameter

    # the first row will contain the random samples for the sigma parameter
    ar[0, :] = np.random.normal(loc=sig_opc3, scale=sig_opc3 / 6, size=ar[0, :].shape)
    # the second row will contain the random samples for the epsilon parameter
    ar[1, :] = np.random.normal(loc=eps_opc3, scale=eps_opc3 / 6, size=ar[1, :].shape)

    return ar
