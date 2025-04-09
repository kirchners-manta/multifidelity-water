"""
Implementation of the chemical model for the MFWater project.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from shutil import which

import h5py
import numpy as np
from numpy.typing import NDArray

# constants
KJ2KCAL = 0.239006
NA = 6.022e23  # atoms/mol


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

            # create a group for the models with an attribute being the number of models
            models = f.create_group("models")
            models.attrs["n_models"] = 8

            # create a subroup for each model
            # add the number of model evaluations and the number of molecules
            for i in range(1, models.attrs["n_models"] + 1):
                model = models.create_group(f"model_{i}")
                model.attrs["n_evals"] = 100
                model.attrs["n_molecules"] = 2 ** (models.attrs["n_models"] + 4 - i)

        args.input = input_file

    # work with the given or created input file
    with h5py.File(args.input, "r+") as f:

        # iterate over the models
        for i, mod in enumerate(f["models"].keys()):

            # add datasets of the LJ parameters including Gaussian noise
            params = f["models"][mod].create_dataset(
                "lj_params",
                data=sampl_lj_params(
                    np.zeros((2, f["models"][mod].attrs["n_evals"]), dtype=np.float32)
                ),
            )

        # debug
        # f.visit(inspect_hdf5)

    # now, as the input file is prepared, the LAMMPS input files can be created
    setup_lammps_input(args.input)

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
    eps_opc3 = 0.68369 * KJ2KCAL  # kcal/mol

    # create an array of size n_models x n_evals with random samples
    # from a Gaussian distribution with mean sigma/epsilon and standard deviation 1/6 * sigma/epsilon
    # this ensures that 99.7% of the samples are within +-1/2 of the values of the standard LJ parameter

    # the first row will contain the random samples for the sigma parameter
    ar[0, :] = np.random.normal(loc=sig_opc3, scale=sig_opc3 / 6, size=ar[0, :].shape)
    # the second row will contain the random samples for the epsilon parameter
    ar[1, :] = np.random.normal(loc=eps_opc3, scale=eps_opc3 / 6, size=ar[1, :].shape)

    return ar


def setup_lammps_input(input: str | Path) -> None:
    """
    Setup the LAMMPS input files for the MD simulation.

    Parameters
    ----------
    input : str | Path
        The path to the hdf5 input file.

    Raises
    -------
    RuntimeError
        If the external software is not installed or not in PATH.
    """

    # check whether the external software is installed
    software = ["fftool", "packmol"]
    for s in software:
        if which(s) is None:
            raise RuntimeError(f"{s} is not installed or not in PATH.")

    # get directory of the data files
    data_dir = Path(__file__).parent / "data"

    # create a head directory for the calculations
    head_dir = Path.cwd() / "models"
    head_dir.mkdir(parents=False, exist_ok=True)

    with h5py.File(input, "r") as f:

        # iterate over the models
        for i, mod in enumerate(f["models"].keys()):

            # get the number of molecules and corresponding box size
            n = f["models"][mod].attrs["n_molecules"]
            l = calc_box_size(n)

            # create a directory for the model
            model_dir = head_dir / mod
            model_dir.mkdir(parents=False, exist_ok=True)

            # get the number of evaluations
            n_evals = f["models"][mod].attrs["n_evals"]

            # create a subdirectory for each evaluation
            for j in range(1, n_evals + 1):
                #  for j in range(1, 4):

                eval_dir = model_dir / f"eval_{j}"
                eval_dir.mkdir(parents=False, exist_ok=True)

                # create a subdirectory for the simulation input files
                # and remove it if it existed before

                sim_dir = eval_dir / "siminp"
                if sim_dir.exists():
                    subprocess.run(f"rm -r {sim_dir}", shell=True, check=True)
                sim_dir.mkdir(parents=False, exist_ok=False)

                # copy the data files to the model directory
                # and call fftool
                subprocess.run(f"cp {data_dir}/opc3* {sim_dir}", shell=True, check=True)
                subprocess.run(
                    f"cd {sim_dir} && fftool {n} opc3.zmat -b {l:.3f}",
                    shell=True,
                    check=True,
                    capture_output=True,
                )

                # insert random seed command in the packmol input file that was generated above
                with open(sim_dir / "pack.inp", encoding="utf-8") as p:
                    packinp = p.readlines()
                    for k, line in enumerate(packinp):
                        if "inside box" in line:
                            packinp[k] = (
                                f"inside box 0.500 0.500 0.500 {l-0.5:3f} {l-0.5:3f} {l-0.5:3f}\n"
                            )
                            break
                    packinp.insert(len(packinp), f"seed -1\n")

                with open(sim_dir / "pack.inp", "w", encoding="utf-8") as p:
                    p.writelines(packinp)

                # call packmol to construct the box
                subprocess.run(
                    f"cd {sim_dir} && packmol < pack.inp",
                    shell=True,
                    check=True,
                    capture_output=True,
                )

                # call fftool again to generate the data files
                subprocess.run(
                    f"cd {sim_dir} && fftool {n} opc3.zmat -b {l:.3f} -l",
                    shell=True,
                    check=True,
                    capture_output=True,
                )

                # remove unnecessary files
                subprocess.run(
                    f"cd {sim_dir} && rm opc3* pack.inp simbox.xyz in.lmp",
                    shell=True,
                    check=True,
                    capture_output=True,
                )

                # copy the custom LAMMPS input file and adjust the LJ parameters
                subprocess.run(
                    f"cp {data_dir}/input.lmp {sim_dir}",
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                with open(sim_dir / "input.lmp", encoding="utf-8") as lmp:
                    lmpinp = lmp.readlines()
                    for k, line in enumerate(lmpinp):
                        if "VAR_EPS" in line:
                            lmpinp[k] = (
                                f"pair_coeff    2    2     {f['models'][mod]['lj_params'][1][j]:.6f}     {f['models'][mod]['lj_params'][0][j]:.6f}  # Ow-Ow\n"
                            )
                        if "velocity all create" in line:
                            lmpinp[k] = (
                                f"velocity all create ${{vTK}} {np.random.randint(1, 99999)}\n"
                            )
                            break
                    with open(sim_dir / "input.lmp", "w", encoding="utf-8") as lmp:
                        lmp.writelines(lmpinp)


def calc_box_size(n: int, rho: float = 0.997, m: float = 18.01528) -> float:
    """
    Calculate the box size of an MD simulation box of molecules.

    Parameters
    ----------
    n : int
        Number of molecules in the box.
    rho : float, optional
        Density of the box, by default 0.997 g/cm^3 (for water at 298.15 K).
    m: float, optional
        Mass of the molecules, by default 18.01528 g/mol (for water).

    Returns
    -------
    float
        The box size in Angstrom.
    """
    # calculate the volume of the box in cm^3
    v = n * m / (rho * NA)

    # calculate the box size in Angstrom
    box_size = (v * 1e24) ** (1 / 3)

    return box_size


def inspect_hdf5(filename: str) -> None:
    """
    Inspect the HDF5 file and print its contents using the visit method.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file that is already opened in a context manager.
    """

    print(filename)
