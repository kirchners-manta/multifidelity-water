"""
Implementation of the chemical model for the MFWater project.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import warnings
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from ..algo_input import check_input_file, model_groups
from ..argparser import constants
from ..md.lammps_log import parse_lammps_log
from ..md.system import calc_box_size, calc_cpus  # noqa: F401  (re-export)
from ..msdiff_io import parse_msdiff_d_raw

#: Datasets written by ``chemmodel-prep`` into the ``models`` group.
_PREP_DATASETS = ("lj_params", "seeds", "charges")

#: Result directories of one evaluation. ``siminp`` is included because ``chemmodel-prep``
#: used to delete and recreate it, which would destroy the inputs of an earlier phase.
_EVAL_SUBDIRS = ("siminp", "simout", "msd")

_PAIR_OW = re.compile(r"^\s*pair_coeff\s+2\s+2\s+(\S+)\s+(\S+)")
_CHARGE_H = re.compile(r"^\s*set\s+type\s+1\s+charge\s+(\S+)")
_CHARGE_O = re.compile(r"^\s*set\s+type\s+2\s+charge\s+(\S+)")


def chemical_model_prep(args: argparse.Namespace) -> int:
    """
    Prepare the chemical model for the MFWater project.

    Draws the perturbed parameters (LJ parameters, partial charges) and the packmol and
    velocity seeds for the maximum number of evaluations of all models, stores them in
    the input file together with the random seed, and creates the LAMMPS input of every
    evaluation below ``args.models_dir``.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments. ``args.seed`` seeds the random number generator (a
        random seed is drawn and printed if ``None``), ``args.models_dir`` is the
        directory of the evaluations (default ``models``).

    Returns
    -------
    int
        The exit code of the function, by default 0.

    Raises
    ------
    RuntimeError
        If ``fftool`` or ``packmol`` is missing, if results or inputs of an earlier phase
        exist in the evaluation directories (nothing is deleted or overwritten), or if the
        input file already holds the sampled datasets.
    """
    check_input_file(args.input, args.algorithm)
    models_dir = Path(getattr(args, "models_dir", "models"))

    # Everything that can refuse runs before the first write, so a refused run leaves
    # the input file and the directories as they were.
    _check_external_software()
    with h5py.File(args.input, "r") as f:
        model_items = model_groups(f["models"])
        evals = {name: int(mod.attrs["n_evals"]) for name, mod in model_items}
        existing = [d for d in _PREP_DATASETS if d in f["models"]]
    if existing:
        raise RuntimeError(
            f"The input file '{args.input}' already contains {existing}. "
            "Run eval-estimator (which keeps them with an '_initial' suffix) or use a "
            "fresh input file."
        )
    check_no_stale_results(models_dir, evals)

    # The seed is part of the result: with it, the sampled parameters and seeds can be
    # reproduced. A drawn seed is printed and stored for the same reason.
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(seed)
    print(f"Random seed: {seed}")

    # check which parameters to perturb and determine noise on the parameters
    noise_lj = constants.NOISE_LJPARAMS if args.params in ["lj", "lj-q"] else 0.0
    noise_q = constants.NOISE_CHARGES if args.params in ["q", "lj-q"] else 0.0

    # the maximum is the number of evaluations of the last model
    n_evals = max(evals.values())

    with h5py.File(args.input, "r+") as f:
        models = f["models"]
        models.attrs["seed"] = seed
        # LJ parameters including Gaussian noise
        models.create_dataset(
            "lj_params",
            data=sampl_lj_params(
                np.zeros((2, n_evals), dtype=np.float32), noise_lj, rng
            ),
        )
        # packmol (row 0) and velocity (row 1) seeds
        models.create_dataset(
            "seeds",
            data=rng.integers(1, 100000, size=(2, n_evals), dtype=np.int32),
        )
        # partial charges including Gaussian noise
        models.create_dataset(
            "charges",
            data=sampl_partial_charges(
                np.zeros((2, n_evals), dtype=np.float32), noise_q, rng
            ),
        )

    # The input file is closed here: setup_lammps_input opens it itself.
    setup_lammps_input(args.input, args.orthoboxy, models_dir=models_dir)

    print("Chemical model preparation done.")
    with h5py.File(args.input, "r+") as f:
        f.attrs["last_algo"] = args.algorithm

    return 0


def check_lammps_parameters(
    log_path: Path,
    lj: tuple[float, float] | None,
    charges: tuple[float, float] | None,
    rtol: float = 1e-5,
) -> None:
    """
    Check that the parameters a LAMMPS run used are the ones stored in the input file.

    The whole log is scanned (no early exit), every echo of a parameter line must agree,
    and a parameter that is checked but missing from the log is an error. Silently
    skipping a missing line would make the check meaningless.

    Parameters
    ----------
    log_path : Path
        ``log.lammps`` of the evaluation.
    lj : tuple of float, optional
        Expected ``(epsilon, sigma)`` of the O-O pair; ``None`` to skip the check.
    charges : tuple of float, optional
        Expected ``(q_O, q_H)``; ``None`` to skip the check (files of the old format have
        no partial charges).
    rtol : float, optional
        Relative tolerance, by default ``1e-5`` (the log has six printed digits).

    Raises
    ------
    ValueError
        If a parameter in the log differs from the expected one, or an expected parameter
        is not found in the log.
    """
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    found: dict[str, list[tuple[float, ...]]] = {"lj": [], "q_H": [], "q_O": []}
    for line in lines:
        if m := _PAIR_OW.match(line):
            found["lj"].append((float(m.group(1)), float(m.group(2))))
        elif m := _CHARGE_H.match(line):
            found["q_H"].append((float(m.group(1)),))
        elif m := _CHARGE_O.match(line):
            found["q_O"].append((float(m.group(1)),))

    checks: list[tuple[str, tuple[float, ...]]] = []
    if lj is not None:
        checks.append(("lj", lj))
    if charges is not None:
        checks.append(("q_O", (charges[0],)))
        checks.append(("q_H", (charges[1],)))
    labels = {
        "lj": "Lennard-Jones parameters (epsilon, sigma)",
        "q_O": "partial charge of the oxygen atoms",
        "q_H": "partial charge of the hydrogen atoms",
    }
    for key, expected in checks:
        if not found[key]:
            raise ValueError(
                f"The {labels[key]} were not found in {log_path}, so they cannot be "
                "compared with the input file."
            )
        for got in found[key]:
            if not np.allclose(got, expected, rtol=rtol):
                raise ValueError(
                    f"The {labels[key]} used in the simulation are not the same as the "
                    f"ones in the input file ({log_path}).\n"
                    f"Expected {expected} but got {got}."
                )


def chemical_model_post(args: argparse.Namespace) -> int:
    """
    Post-process the chemical model for the MFWater project.

    For every evaluation the diffusion coefficient is read from ``msd/msdiff_out.csv`` and
    the cost (wall time times MPI ranks, core-seconds) from ``simout/log.lammps``. The
    parameters recorded in the log must equal the stored ones.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments. ``args.models_dir`` is the directory of the
        evaluations (default ``models``).

    Returns
    -------
    int
        The exit code of the program, by default 0.

    Raises
    ------
    RuntimeError
        If a LAMMPS log is incomplete (see
        :func:`mfwater.md.lammps_log.parse_lammps_log`) or an existing ``diffusion_coeff``
        dataset has a different length.
    ValueError
        If the parameters used in a simulation differ from the stored ones.
    """
    models_dir = Path(getattr(args, "models_dir", "models"))
    check_input_file(args.input, args.algorithm, models_dir=models_dir)

    with h5py.File(args.input, "r+") as f:
        models = f["models"]
        model_items = model_groups(models)

        lj_all = models["lj_params"][:]
        charges_all = models["charges"][:] if "charges" in models else None
        if charges_all is None:
            warnings.warn(
                "The input file has no partial charges (old format): the charges in the "
                "LAMMPS logs are not checked.",
                stacklevel=2,
            )

        for name, mod in model_items:
            comptimes = []
            diffusion_coeffs = []
            for j in range(1, int(mod.attrs["n_evals"]) + 1):
                eval_dir = models_dir / name / f"eval_{j}"
                # raw msdiff unit (1e-12 m^2/s), as stored in paper I; the shared
                # parser finds the column by header (old and new msdiff layout)
                diffusion_coeffs.append(
                    parse_msdiff_d_raw(eval_dir / "msd" / "msdiff_out.csv")
                )

                logpath = eval_dir / "simout" / "log.lammps"
                try:
                    check_lammps_parameters(
                        logpath,
                        (float(lj_all[0][j - 1]), float(lj_all[1][j - 1])),
                        (
                            None
                            if charges_all is None
                            else (
                                float(charges_all[0][j - 1]),
                                float(charges_all[1][j - 1]),
                            )
                        ),
                    )
                except ValueError as exc:
                    raise ValueError(f"Model {name} evaluation {j}: {exc}") from exc

                # the parser reads only this log: nothing is carried over between
                # evaluations, and an unfinished run raises instead of reusing old values
                comptimes.append(parse_lammps_log(logpath).cpu_time_s)

            mod.attrs["computation_time"] = np.mean(comptimes)
            # add the diffusion coefficient to the model if it is not already present,
            # otherwise update it
            if "diffusion_coeff" not in mod.keys():
                mod.create_dataset(
                    "diffusion_coeff", data=np.array(diffusion_coeffs, dtype=np.float32)
                )
            else:
                if mod["diffusion_coeff"].shape != (len(diffusion_coeffs),):
                    raise RuntimeError(
                        f"Model {name}: the stored diffusion_coeff has shape "
                        f"{mod['diffusion_coeff'].shape}, but {len(diffusion_coeffs)} "
                        "evaluations were read."
                    )
                mod["diffusion_coeff"][:] = diffusion_coeffs

        print("Chemical model post-processing done.")

        f.attrs["last_algo"] = args.algorithm

    return 0


def sampl_lj_params(
    ar: npt.NDArray[np.float32],
    noise: float,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float32]:
    """
    Generate random samples for the Lennard-Jones parameters from a Gaussian distribution.

    Parameters
    ----------
    ar : numpy.ndarray
        The array to fill with random samples, shape ``(2, n)``.
    noise : float
        The amount of noise to add to the samples in multiples of the mean value.
    rng : numpy.random.Generator, optional
        Random number generator; a new unseeded one is created if ``None``.

    Returns
    -------
    numpy.ndarray
        The array with random samples: row 0 is epsilon, row 1 is sigma.
    """
    if rng is None:
        rng = np.random.default_rng()

    # the first row will contain the random samples for the epsilon parameter
    ar[0, :] = rng.normal(
        loc=constants.OPC3_EPSILON_OO,
        scale=constants.OPC3_EPSILON_OO * noise,
        size=ar[0, :].shape,
    )
    # the second row will contain the random samples for the sigma parameter
    ar[1, :] = rng.normal(
        loc=constants.OPC3_SIGMA_OO,
        scale=constants.OPC3_SIGMA_OO * noise,
        size=ar[1, :].shape,
    )

    return ar


def sampl_partial_charges(
    ar: npt.NDArray[np.float32],
    noise: float,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float32]:
    """
    Generate random samples for the partial charges from a Gaussian distribution.

    Only the charge of the oxygen atoms is perturbed; the hydrogen charge is derived from
    it (``q_H = -q_O / 2``) so that the molecule stays neutral.

    Parameters
    ----------
    ar : numpy.ndarray
        The array to fill with random samples, shape ``(2, n)``.
    noise : float
        The amount of noise to add to the samples in multiples of the mean value.
    rng : numpy.random.Generator, optional
        Random number generator; a new unseeded one is created if ``None``.

    Returns
    -------
    numpy.ndarray
        The array with random samples: row 0 is q_O, row 1 is q_H.
    """
    if rng is None:
        rng = np.random.default_rng()

    ar[0, :] = rng.normal(
        loc=constants.OPC3_CHARGE_O,
        scale=abs(constants.OPC3_CHARGE_O) * noise,
        size=ar[0, :].shape,
    )
    ar[1, :] = -0.5 * ar[0, :]

    return ar


def _check_external_software() -> None:
    """
    Check that the external programs of the setup are installed.

    Raises
    ------
    RuntimeError
        If ``fftool`` or ``packmol`` is not in ``PATH``.
    """
    for s in ("fftool", "packmol"):
        if shutil.which(s) is None:
            raise RuntimeError(f"{s} is not installed or not in PATH.")


def check_no_stale_results(models_dir: Path, evals: dict[str, int]) -> None:
    """
    Refuse to set up evaluations whose directories already hold results or inputs.

    Evaluation directories of an earlier phase (calibration) contain ``siminp``,
    ``simout`` and ``msd``. Setting up the next phase in the same directories would
    overwrite the inputs and make ``chemmodel-post`` read the old results as if they were
    new. Nothing is deleted; the user has to choose another ``--models-dir``.

    Parameters
    ----------
    models_dir : pathlib.Path
        Directory of the evaluations.
    evals : dict of str to int
        Number of evaluations per model name.

    Raises
    ------
    RuntimeError
        If any ``siminp``, ``simout`` or ``msd`` directory exists for the given
        evaluations.
    """
    stale = [
        models_dir / name / f"eval_{j}" / sub
        for name, n in evals.items()
        for j in range(1, n + 1)
        for sub in _EVAL_SUBDIRS
        if (models_dir / name / f"eval_{j}" / sub).exists()
    ]
    if stale:
        shown = ", ".join(str(p) for p in stale[:3])
        more = f" (and {len(stale) - 3} more)" if len(stale) > 3 else ""
        raise RuntimeError(
            f"{len(stale)} directories of an earlier phase exist: {shown}{more}. "
            "Refusing to overwrite them and to mix old results into the new phase. "
            f"Use a new directory with --models-dir (current: '{models_dir}'); "
            "nothing was changed."
        )


def _run_external(
    cmd: Sequence[str], cwd: Path, stdin_file: Path | None = None
) -> None:
    """
    Run an external program without a shell and report failures with its stderr.

    Parameters
    ----------
    cmd : sequence of str
        Program and arguments.
    cwd : pathlib.Path
        Working directory.
    stdin_file : pathlib.Path, optional
        File connected to the standard input (``packmol < pack.inp``).

    Raises
    ------
    RuntimeError
        If the program fails; the message contains the end of its error output.
    """
    stdin = open(stdin_file, encoding="utf-8") if stdin_file else None
    try:
        subprocess.run(
            list(cmd),
            cwd=cwd,
            check=True,
            capture_output=True,
            stdin=stdin if stdin else subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command {' '.join(cmd)!r} failed in {cwd} with exit code "
            f"{exc.returncode}:\n{(exc.stderr or exc.stdout or '')[-500:]}"
        ) from exc
    finally:
        if stdin:
            stdin.close()


def setup_lammps_input(
    input: str | Path, orthoboxy: bool, models_dir: str | Path = "models"
) -> None:
    """
    Setup the LAMMPS input files for the MD simulation.

    Parameters
    ----------
    input : str | Path
        The path to the hdf5 input file.
    orthoboxy : bool
        Whether to use OrthoBoXY-shaped boxes for the models.
    models_dir : str | Path, optional
        Directory in which ``<model>/eval_<j>`` are created, by default ``models``
        (relative to the current directory).

    Raises
    ------
    RuntimeError
        If the external software is not installed or not in PATH, an external program
        fails, or directories of an earlier phase exist.
    """
    _check_external_software()

    # get directory of the data files
    data_dir = Path(__file__).parent / "data"

    head_dir = Path(models_dir)

    with h5py.File(input, "r") as f:
        model_items = model_groups(f["models"])
        specs = [
            (name, int(mod.attrs["n_molecules"]), int(mod.attrs["n_evals"]))
            for name, mod in model_items
        ]
        lj = f["models"]["lj_params"][:]
        charges = f["models"]["charges"][:]
        seeds = f["models"]["seeds"][:]

    # nothing may exist yet: the directories below are created, never replaced
    check_no_stale_results(head_dir, {name: n_evals for name, _, n_evals in specs})
    head_dir.mkdir(parents=False, exist_ok=True)

    print("Setting up LAMMPS input files.")

    for name, n, n_evals in specs:
        print(f"{name}")

        # TODO: reuse the shared box builder of mfwater.md.pipeline (MDForwardModel.
        # _build_box) once it is available as a function; it is a private method of the
        # MD pipeline and the package md/ is not changed from here.
        lx, ly, lz = calc_box_size(n, orthoboxy_shape=orthoboxy)
        box = f"{lx:.6f},{ly:.6f},{lz:.6f}"

        model_dir = head_dir / name
        model_dir.mkdir(parents=False, exist_ok=True)

        for j in range(1, n_evals + 1):
            eval_dir = model_dir / f"eval_{j}"
            eval_dir.mkdir(parents=False, exist_ok=True)

            # check_no_stale_results guarantees that siminp does not exist
            sim_dir = eval_dir / "siminp"
            sim_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(data_dir / "opc3.zmat", sim_dir)
            shutil.copy(data_dir / "opc3.ff", sim_dir)

            _run_external(["fftool", str(n), "opc3.zmat", "--box", box], sim_dir)

            # insert the box margin and the random seed into the packmol input file
            pack = sim_dir / "pack.inp"
            packinp = pack.read_text(encoding="utf-8").splitlines(keepends=True)
            for k, line in enumerate(packinp):
                if "inside box" in line:
                    packinp[k] = (
                        f"inside box 0.500000 0.500000 0.500000 {lx-0.5:.6f} {ly-0.5:.6f} {lz-0.5:.6f}\n"
                    )
                    break
            packinp.append(f"seed {seeds[0][j-1]}\n")
            pack.write_text("".join(packinp), encoding="utf-8")

            # packmol reads its input from stdin
            _run_external(["packmol"], sim_dir, stdin_file=pack)
            _run_external(
                ["fftool", str(n), "opc3.zmat", "--box", box, "--lmp"], sim_dir
            )

            # remove the intermediate files that were created in this fresh directory
            for tmp in (
                "opc3.zmat",
                "opc3.ff",
                "opc3_pack.xyz",
                "pack.inp",
                "simbox.xyz",
                "in.lmp",
            ):
                (sim_dir / tmp).unlink()

            # copy the custom LAMMPS input file and set the sampled parameters
            shutil.copy(data_dir / "input.lmp", sim_dir)
            lmp_path = sim_dir / "input.lmp"
            lmpinp = lmp_path.read_text(encoding="utf-8").splitlines(keepends=True)
            for k, line in enumerate(lmpinp):
                if "VAR_EPS" in line:
                    lmpinp[k] = (
                        f"pair_coeff    2    2     {lj[0][j-1]:.6f}     {lj[1][j-1]:.6f}  # Ow-Ow\n"
                    )
                if "VAR_q_H" in line:
                    lmpinp[k] = f"set type 1 charge {charges[1][j-1]:11.8f}  # Hw\n"
                if "VAR_q_O" in line:
                    lmpinp[k] = f"set type 2 charge {charges[0][j-1]:11.8f}  # Ow\n"
                if "velocity all create" in line:
                    lmpinp[k] = f"velocity all create ${{vTK}} {seeds[1][j-1]}\n"
                    break
            lmp_path.write_text("".join(lmpinp), encoding="utf-8")

            # copy the runscript and adjust the simulation parameters
            shutil.copy(data_dir / "run-lammps-marvin.sh", sim_dir)
            rsh_path = sim_dir / "run-lammps-marvin.sh"
            rshinp = rsh_path.read_text(encoding="utf-8").splitlines(keepends=True)
            for k, line in enumerate(rshinp):
                if "N_CPU" in line:
                    rshinp[k] = f"#SBATCH --ntasks={calc_cpus(n)}\n"
                if "JOB_NAME" in line:
                    rshinp[k] = f"#SBATCH --job-name={name}_{n}_{j}\n"
                    break
            rsh_path.write_text("".join(rshinp), encoding="utf-8")
