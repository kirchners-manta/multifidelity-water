"""
Implementation of the chemical model for the MFWater project.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from ..algo_input import check_input_file

# constants
KJ2KCAL = 0.239006
NA = 6.022e23  # atoms/mol


def chemical_model_prep(args: argparse.Namespace) -> int:
    """Prepare the chemical model for the MFWater project.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    int
        The exit code of the function, by default 0.
    """

    # check input file
    check_input_file(args.input, args.algorithm)

    # work with the given or created input file
    with h5py.File(args.input, "r+") as f:

        # get the number of models
        # get the number of evaluations of the last model because it is the maximum. 
        #@Tom: This would be true if the models are ordered by correlation. However, a-priori we dont know that. 
        
        n_evals = max([f["models"][mod].attrs["n_evals"] for mod in f["models"].keys()])
        # add datasets of the LJ parameters including Gaussian noise
        f["models"].create_dataset(
            "lj_params",
            data=sampl_lj_params(np.zeros((2, n_evals), dtype=np.float32)),
        )
        # add packmol and velocity seeds
        # first row is for packmol, second for velocity
        f["models"].create_dataset(
            "seeds",
            data=np.random.randint(1, 100000, size=(2, n_evals), dtype=np.int32),
        )

    # now, as the input file is prepared, the LAMMPS input files can be created
    setup_lammps_input(args.input, args.orthoboxy)

    print("Chemical model preparation done.")

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

    # check input file
    check_input_file(args.input, args.algorithm)

    # read information on models from the input file
    with h5py.File(args.input, "r+") as f:

        # find model groups
        model_items = [
            (name, mod)
            for name, mod in f["models"].items()
            if isinstance(mod, h5py.Group)
        ]

        # iterate over the models
        for name, mod in model_items:

            comptimes = []
            diffusion_coeffs = []
            for j in range(1, mod.attrs["n_evals"] + 1):
                # read msdiff output to get the diffusion coefficient
                msdpath = (
                    Path.cwd()
                    / "models"
                    / name
                    / f"eval_{j}"
                    / "msd"
                    / "msdiff_out.csv"
                )
                with open(msdpath, encoding="utf-8") as msd:
                    msdlog = msd.readlines()
                    diff = float(msdlog[-1].split(",")[0].strip())
                    diffusion_coeffs.append(diff)

                # read LAMMPS output
                logpath = (
                    Path.cwd() / "models" / name / f"eval_{j}" / "simout" / "log.lammps"
                )
                with open(logpath, encoding="utf-8") as lmp:
                    lmplog = lmp.readlines()

                    # check if the LJ parameters actually used are the same as the ones in the input file
                    for line in lmplog:
                        if "pair_coeff    2    2" in line:
                            eps = float(line.split()[3])
                            sig = float(line.split()[4])
                            if not (
                                np.allclose(
                                    eps,
                                    f["models"]["lj_params"][0][j - 1],
                                    rtol=1e-5,
                                )
                                and np.allclose(
                                    sig,
                                    f["models"]["lj_params"][1][j - 1],
                                    rtol=1e-5,
                                )
                            ):

                                raise ValueError(
                                    f"Error in model {name} evaluation {j}: the LJ parameters used in the simulation are not the same as the ones in the input file.\n Expected {f['models']['lj_params'][0][j-1]} {f['models']['lj_params'][1][j-1]} but got {eps} {sig}"
                                )
                            break

                    # to get the cost (time) of the simulation, read lines reversely
                    for line in lmplog[::-1]:
                        if "Total wall time:" in line:
                            # get the time of the simulation in seconds
                            time = sum(
                                x * int(t)
                                for x, t in zip(
                                    [3600, 60, 1], line.split()[-1].split(":")
                                )
                            )
                        if "Loop time of" in line:
                            # get the number of CPUs used for the simulation
                            n_cpus = int(line.split()[5])
                            break

                comptimes.append(time * n_cpus)

            # add the computation time to the model
            mod.attrs["computation_time"] = np.mean(comptimes)
            # add the diffusion coefficient to the model if it is not already present, otherwise update it
            if "diffusion_coeff" not in mod.keys():
                mod.create_dataset(
                    "diffusion_coeff", data=np.array(diffusion_coeffs, dtype=np.float32)
                )
            else:
                mod["diffusion_coeff"][:] = diffusion_coeffs

    print("Chemical model post-processing done.")

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
    # from a Gaussian distribution with mean epsilon/sigma and standard deviation of 1/120 * epsilon  and 1/120 * sigma
    # this ensures that 99.7% of the samples are within +-2.5 of the values of the standard LJ parameters

    # the first row will contain the random samples for the epsilon parameter
    ar[0, :] = np.random.normal(loc=eps_opc3, scale=eps_opc3 / 120, size=ar[0, :].shape)
    # the second row will contain the random samples for the sigma parameter
    ar[1, :] = np.random.normal(loc=sig_opc3, scale=sig_opc3 / 120, size=ar[1, :].shape)

    return ar


def setup_lammps_input(input: str | Path, orthoboxy: bool) -> None:
    """
    Setup the LAMMPS input files for the MD simulation.

    Parameters
    ----------
    input : str | Path
        The path to the hdf5 input file.
    orthoboxy : bool
        Whether to use OrthoBoXY-shaped boxes for the models.

    Raises
    -------
    RuntimeError
        If the external software is not installed or not in PATH.
    """

    # check whether the external software is installed
    software = ["fftool", "packmol"]
    for s in software:
        if shutil.which(s) is None:
            raise RuntimeError(f"{s} is not installed or not in PATH.")

    # get directory of the data files
    data_dir = Path(__file__).parent / "data"

    # create a head directory for the calculations
    head_dir = Path.cwd() / "models"
    head_dir.mkdir(parents=False, exist_ok=True)

    with h5py.File(input, "r+") as f:

        # find model groups
        model_items = [
            (name, mod)
            for name, mod in f["models"].items()
            if isinstance(mod, h5py.Group)
        ]

        # output to user
        print(f"Setting up LAMMPS input files.")

        # iterate over the models
        for name, mod in model_items:

            # output to user
            print(f"Model {name}:")

            # get the number of molecules and corresponding box size
            n = mod.attrs["n_molecules"]
            lx, ly, lz = calc_box_size(n, orthoboxy_shape=orthoboxy)

            # create a directory for the model
            model_dir = head_dir / name
            model_dir.mkdir(parents=False, exist_ok=True)

            # get the number of evaluations
            n_evals = mod.attrs["n_evals"]

            # create a subdirectory for each evaluation
            for j in range(1, n_evals + 1):
                # debug
                # for j in range(1, 4):

                eval_dir = model_dir / f"eval_{j}"
                eval_dir.mkdir(parents=False, exist_ok=True)

                # create a subdirectory for the simulation input files
                # and remove it if it existed before

                sim_dir = eval_dir / "siminp"
                if sim_dir.exists():
                    shutil.rmtree(sim_dir)
                sim_dir.mkdir(parents=False, exist_ok=False)

                # copy the data files to the model directory
                # and call fftool
                shutil.copy(data_dir / "opc3.zmat", sim_dir)
                shutil.copy(data_dir / "opc3.ff", sim_dir)

                subprocess.run(
                    f"cd {sim_dir} && fftool {n} opc3.zmat -b {lx:.6f},{ly:.6f},{lz:.6f}",
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
                                f"inside box 0.500000 0.500000 0.500000 {lx-0.5:.6f} {ly-0.5:.6f} {lz-0.5:.6f}\n"
                            )
                            break
                    packinp.insert(
                        len(packinp), f"seed {f['models']['seeds'][0][j-1]}\n"
                    )

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
                    f"cd {sim_dir} && fftool {n} opc3.zmat -b {lx:.6f},{ly:.6f},{lz:.6f} -l",
                    shell=True,
                    check=True,
                    capture_output=True,
                )

                # remove unnecessary files
                (sim_dir / "opc3.zmat").unlink()
                (sim_dir / "opc3.ff").unlink()
                (sim_dir / "opc3_pack.xyz").unlink()
                (sim_dir / "pack.inp").unlink()
                (sim_dir / "simbox.xyz").unlink()
                (sim_dir / "in.lmp").unlink()

                # copy the custom LAMMPS input file and adjust the LJ parameters
                shutil.copy(data_dir / "input.lmp", sim_dir)

                # add LJ parameters including Gaussian noise to the LAMMPS input file
                with open(sim_dir / "input.lmp", encoding="utf-8") as lmp:
                    lmpinp = lmp.readlines()
                    for k, line in enumerate(lmpinp):
                        if "VAR_EPS" in line:
                            lmpinp[k] = (
                                f"pair_coeff    2    2     {f['models']['lj_params'][0][j-1]:.6f}     {f['models']['lj_params'][1][j-1]:.6f}  # Ow-Ow\n"
                            )
                        if "velocity all create" in line:
                            lmpinp[k] = (
                                f"velocity all create ${{vTK}} {f['models']['seeds'][1][j-1]}\n"
                            )
                            break
                    with open(sim_dir / "input.lmp", "w", encoding="utf-8") as lmp:
                        lmp.writelines(lmpinp)

                # copy the runscript to the model directory and adjust the simulation parameters
                shutil.copy(data_dir / "run-lammps-marvin.sh", sim_dir)

                with open(sim_dir / "run-lammps-marvin.sh", encoding="utf-8") as rsh:
                    rshinp = rsh.readlines()
                    for k, line in enumerate(rshinp):
                        if "N_CPU" in line:
                            rshinp[k] = f"#SBATCH --ntasks={calc_cpus(n)}\n"
                        if "JOB_NAME" in line:
                            rshinp[k] = f"#SBATCH --job-name={name}_{n}_{j}\n"
                            break
                    with open(
                        sim_dir / "run-lammps-marvin.sh", "w", encoding="utf-8"
                    ) as rsh:
                        rsh.writelines(rshinp)


def calc_box_size(
    n: int,
    orthoboxy_shape: bool,
    rho: float = 0.997,
    m: float = 18.01528,
) -> list[float]:
    """
    Calculate the box size of an MD simulation box of molecules.

    Parameters
    ----------
    n : int
        Number of molecules in the box.
    orthoboxy_shape : bool
        If True, the box is tetragonal with lx = ly /= lz. Useful for OrthoBoXY simulations.
        If False, the box is cubic.
    rho : float, optional
        Density of the box, by default 0.997 g/cm^3 (for water at 298.15 K).
    m: float, optional
        Mass of the molecules, by default 18.01528 g/mol (for water).


    Returns
    -------
    list[float]
        The box size in x, y, and z direction in Angstrom.
    """

    # OrthoBoXY ratio of lz/lx = lz/ly
    RATIO = 2.7933596497

    # calculate the volume of the box in cm^3
    v = n * m / (rho * NA)

    # calculate the box size in Angstrom
    if orthoboxy_shape:
        lx = ly = (v / RATIO * 1e24) ** (1 / 3)
        lz = lx * RATIO
    else:
        lx = ly = lz = (v * 1e24) ** (1 / 3)

    return [lx, ly, lz]


def calc_cpus(n: int) -> int:
    """
    Calculate the number of CPUs to use for the simulation.

    Parameters
    ----------
    n : int
        Number of molecules in the box.

    Returns
    -------
    int
        The number of CPUs to use.
    """
    # get the number of CPUs to use for the simulation

    if n >= 2000:
        cpus = 8
    elif n >= 1000:
        cpus = 6
    elif n >= 500:
        cpus = 4
    elif n >= 100:
        cpus = 2
    else:
        cpus = 1

    return cpus
