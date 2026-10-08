"""MD forward model: fftool/packmol, LAMMPS, TRAVIS and msdiff for one evaluation.

External programs ``fftool``, ``packmol``, ``mpirun``/``lmp``, ``travis`` and
``msdiff`` must be in ``PATH``. On Marvin, load LAMMPS first with
``module load LAMMPS/23Jun2022-foss-2022a-kokkos``.

Directory layout of one evaluation (``<root>`` is ``work_dir_root``)::

    <root>/N{n}/{key_hash}/attempt_{k}/
        siminp/   fftool/packmol logs and files, data.lmp, input.lmp
        simout/   LAMMPS working directory: log.lammps, *.lammpstrj, ...
        msd/      TRAVIS and msdiff input/output, msdiff_out.csv

Every call creates a new ``attempt_{k}`` directory; old attempts (e.g. from an
interrupted run) are never reused or deleted.
"""

from __future__ import annotations

import math
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..algo_chemical_model.chemical_model import calc_box_size, calc_cpus
from .forward_model import (
    _check_theta,
    derive_seeds,
    evaluation_key,
    key_hash,
)

#: Directory with the paper-I templates (LAMMPS input, force field, TRAVIS input).
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "algo_chemical_model" / "data"

#: Default LAMMPS command. ``{ncpu}`` and ``{input}`` are filled in; LAMMPS runs in
#: ``simout/`` and reads ``../siminp/input.lmp``.
DEFAULT_LAMMPS_CMD = "mpirun -np {ncpu} lmp -i {input}"

#: MSD file written by TRAVIS for water with the template (atom #2, molecule H2O).
TRAVIS_MSD_FILE = "msd_H2O_#2.csv"

#: msdiff reports D in 1e-12 m^2/s; the likelihood uses 1e-9 m^2/s.
MSDIFF_TO_NANO = 1.0e-3


def _run(
    args: Sequence[str],
    cwd: Path,
    log_name: str,
    stdin_file: Path | None = None,
    stdout_name: str | None = None,
) -> None:
    """Run an external program, logging stdout/stderr to files.

    Parameters
    ----------
    args : Sequence[str]
        Command and arguments (no shell).
    cwd : Path
        Working directory.
    log_name : str
        Base name of the log; ``<log_name>.out`` and ``<log_name>.err`` are
        created in ``cwd``.
    stdin_file : Path | None, optional
        File to feed to stdin. If None, stdin is closed so that interactive
        programs fail instead of hanging.
    stdout_name : str | None, optional
        File name for captured stdout instead of ``<log_name>.out`` (e.g.
        ``travis.log``).

    Raises
    ------
    RuntimeError
        If the program fails.
    """
    out_path = cwd / (stdout_name or f"{log_name}.out")
    err_path = cwd / f"{log_name}.err"
    stdin_handle = open(stdin_file, encoding="utf-8") if stdin_file else None
    try:
        with open(out_path, "w", encoding="utf-8") as out, open(
            err_path, "w", encoding="utf-8"
        ) as err:
            subprocess.run(
                list(args),
                cwd=cwd,
                check=True,
                stdin=stdin_handle if stdin_handle else subprocess.DEVNULL,
                stdout=out,
                stderr=err,
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command {' '.join(args)!r} failed with exit code {exc.returncode}. "
            f"See {out_path} and {err_path}."
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command {args[0]!r} not found (cwd {cwd}).") from exc
    finally:
        if stdin_handle:
            stdin_handle.close()


def write_lammps_input(
    template: Path,
    target: Path,
    theta: np.ndarray,
    velocity_seed: int,
) -> None:
    """Write ``input.lmp`` from the template for one parameter set.

    The formatting follows ``setup_lammps_input`` of the paper-I workflow.
    ``read_data`` is redirected to ``../siminp/data.lmp`` because LAMMPS runs in
    ``simout/``.

    Parameters
    ----------
    template : Path
        Template ``input.lmp`` with ``VAR_EPS``, ``VAR_SIG``, ``VAR_q_H``, ``VAR_q_O``.
    target : Path
        Output file.
    theta : np.ndarray
        ``(epsilon_OO, sigma_OO, q_O)`` in kcal/mol, Angstrom, e.
    velocity_seed : int
        Seed for the initial velocities.

    Raises
    ------
    RuntimeError
        If a placeholder or required line is missing from the template.
    """
    eps, sig, q_o = (float(x) for x in theta)
    q_h = -0.5 * q_o  # neutral molecule
    done = {"pair": False, "qh": False, "qo": False, "vel": False, "data": False}
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    for k, line in enumerate(lines):
        if "VAR_EPS" in line:
            lines[k] = f"pair_coeff    2    2     {eps:.6f}     {sig:.6f}  # Ow-Ow\n"
            done["pair"] = True
        elif "VAR_q_H" in line:
            lines[k] = f"set type 1 charge {q_h:11.8f}  # Hw\n"
            done["qh"] = True
        elif "VAR_q_O" in line:
            lines[k] = f"set type 2 charge {q_o:11.8f}  # Ow\n"
            done["qo"] = True
        elif "velocity all create" in line:
            lines[k] = f"velocity all create ${{vTK}} {velocity_seed}\n"
            done["vel"] = True
        elif line.strip().startswith("read_data"):
            lines[k] = "read_data ../siminp/data.lmp\n"
            done["data"] = True
    missing = [name for name, ok in done.items() if not ok]
    if missing:
        raise RuntimeError(f"LAMMPS template {template} lacks entries: {missing}.")
    target.write_text("".join(lines), encoding="utf-8")


def parse_msdiff_output(path: Path) -> float:
    """Read the diffusion coefficient from ``msdiff_out.csv``.

    Two formats exist. Older msdiff (paper I) writes
    ``D / 10^-12 m^2/s, delta_D / ..., K / ...`` with D in column 0 and no species
    column; current msdiff writes ``Species, D_0 / 10^-12 m^2/s, ...``. The column
    is located by header (``D_0 /`` or ``D /``; not ``D_z`` or ``delta_D``), the
    unit is checked, and the value of the last row is converted to 1e-9 m^2/s.
    The Hummer term K is never added.

    Parameters
    ----------
    path : Path
        ``msdiff_out.csv``.

    Returns
    -------
    float
        Diffusion coefficient D_0 in 1e-9 m^2/s.

    Raises
    ------
    RuntimeError
        If the file is malformed, has an unexpected unit, or D is not finite.
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"{path} contains no data rows.")
    header = [h.strip() for h in lines[0].split(",")]
    cols = [i for i, h in enumerate(header) if h.startswith(("D_0 /", "D /"))]
    if len(cols) != 1 or "10^-12 m^2/s" not in header[cols[0]]:
        raise RuntimeError(
            f"Unexpected msdiff header in {path}: {header}. Expected a column "
            "'D_0 / 10^-12 m^2/s' or 'D / 10^-12 m^2/s'."
        )
    value = float(lines[-1].split(",")[cols[0]].strip()) * MSDIFF_TO_NANO
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite diffusion coefficient in {path}.")
    return value


class MDForwardModel:
    """Forward model running one MD simulation per evaluation.

    fftool, packmol, LAMMPS (via ``lammps_cmd``), TRAVIS and msdiff must be in
    ``PATH`` (on Marvin: ``module load LAMMPS/23Jun2022-foss-2022a-kokkos``).
    The TRAVIS input is the user's template
    ``algo_chemical_model/data/travis_input_msd.txt`` (fixed cell, 1000 fs
    between frames, 2000 frames, correlation depth 600, atom ``#2`` = molecular
    centre of mass, confirmed by the user). Run length is taken from the LAMMPS
    template (1 ns equilibration + 2 ns production).

    Parameters
    ----------
    work_dir_root : str | Path
        Root directory; evaluations live in ``<root>/N{n}/{key_hash}/attempt_{k}``.
        Use the same directory as the cache of
        :class:`mfwater.algo_mfda.cache.CachedForwardModel`.
    orthoboxy : bool, optional
        Tetragonal OrthoBoXY boxes. Not supported yet (TRAVIS/msdiff need a
        different analysis), so True raises ``NotImplementedError``.
    lammps_cmd : str, optional
        Command template with ``{ncpu}`` and ``{input}``.
    travis_template : str | Path | None, optional
        TRAVIS input file; default is the template in the package.
    msdiff_args : Sequence[str], optional
        Extra msdiff arguments, e.g. ``("--hummer", "298.15", "0.00089")``.
        The box length (``-l``) is added automatically for cubic boxes.

    Attributes
    ----------
    last_attempt_dir : Path | None
        Attempt directory of the most recent call.

    Raises
    ------
    NotImplementedError
        If ``orthoboxy`` is True.
    """

    def __init__(
        self,
        work_dir_root: str | Path,
        orthoboxy: bool = False,
        lammps_cmd: str = DEFAULT_LAMMPS_CMD,
        travis_template: str | Path | None = None,
        msdiff_args: Sequence[str] = (),
    ) -> None:
        if orthoboxy:
            raise NotImplementedError(
                "OrthoBoXY boxes are not supported by the MD forward model yet."
            )
        self.work_dir_root = Path(work_dir_root)
        self.orthoboxy = orthoboxy
        self.lammps_cmd = lammps_cmd
        self.travis_template = (
            Path(travis_template)
            if travis_template is not None
            else TEMPLATE_DIR / "travis_input_msd.txt"
        )
        self.msdiff_args = tuple(msdiff_args)
        self.last_attempt_dir: Path | None = None

    def _required_programs(self) -> list[str]:
        """Return the executables needed for one evaluation."""
        lammps_exe = shlex.split(self.lammps_cmd)[0]
        return ["fftool", "packmol", lammps_exe, "travis", "msdiff"]

    def _new_attempt_dir(self, n_molecules: int, key_h: str) -> Path:
        """Create the next free ``attempt_{k}`` directory with its subfolders.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        key_h : str
            Short key hash.

        Returns
        -------
        Path
            The new attempt directory.
        """
        base = self.work_dir_root / f"N{n_molecules}" / key_h
        base.mkdir(parents=True, exist_ok=True)
        existing = [
            int(m.group(1))
            for p in base.iterdir()
            if (m := re.fullmatch(r"attempt_(\d+)", p.name))
        ]
        k = max(existing, default=0) + 1
        while True:
            attempt = base / f"attempt_{k}"
            try:
                attempt.mkdir()  # fails if it exists: never reuse an attempt
                break
            except FileExistsError:
                k += 1
        for sub in ("siminp", "simout", "msd"):
            (attempt / sub).mkdir()
        return attempt

    def __call__(self, n_molecules: int, theta: np.ndarray) -> float:
        """Run the MD pipeline and return the diffusion coefficient.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        theta : np.ndarray
            ``(epsilon_OO, sigma_OO, q_O)`` in kcal/mol, Angstrom, e.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.

        Raises
        ------
        RuntimeError
            If a program is missing or a step fails.
        """
        theta = _check_theta(theta)
        for prog in self._required_programs():
            if shutil.which(prog) is None:
                raise RuntimeError(f"{prog} is not installed or not in PATH.")

        key = evaluation_key(n_molecules, theta)
        packmol_seed, velocity_seed = derive_seeds(key)
        attempt = self._new_attempt_dir(n_molecules, key_hash(key))
        self.last_attempt_dir = attempt
        siminp, simout, msd = attempt / "siminp", attempt / "simout", attempt / "msd"

        self._build_box(n_molecules, siminp, packmol_seed)
        write_lammps_input(
            TEMPLATE_DIR / "input.lmp", siminp / "input.lmp", theta, velocity_seed
        )
        self._run_lammps(n_molecules, simout)
        lx = calc_box_size(n_molecules, orthoboxy_shape=self.orthoboxy)[0]
        return self._analyse(msd, lx)

    def _build_box(self, n: int, siminp: Path, packmol_seed: int) -> None:
        """Create the LAMMPS data file with fftool and packmol.

        Same steps as ``setup_lammps_input`` (``algo_chemical_model``); kept
        separate for now, to be deduplicated later. Unlike the original, no
        intermediate files are removed.

        Parameters
        ----------
        n : int
            Number of molecules.
        siminp : Path
            Simulation input directory.
        packmol_seed : int
            Seed for packmol.
        """
        lx, ly, lz = calc_box_size(n, orthoboxy_shape=self.orthoboxy)
        box = f"{lx:.6f},{ly:.6f},{lz:.6f}"
        shutil.copy(TEMPLATE_DIR / "opc3.zmat", siminp)
        shutil.copy(TEMPLATE_DIR / "opc3.ff", siminp)

        _run(["fftool", str(n), "opc3.zmat", "--box", box], siminp, "fftool_pack")

        pack = siminp / "pack.inp"
        packinp = pack.read_text(encoding="utf-8").splitlines(keepends=True)
        for k, line in enumerate(packinp):
            if "inside box" in line:
                # keep molecules 0.5 A away from the box walls (as in paper I)
                packinp[k] = (
                    f"inside box 0.500000 0.500000 0.500000 "
                    f"{lx - 0.5:.6f} {ly - 0.5:.6f} {lz - 0.5:.6f}\n"
                )
                break
        packinp.append(f"seed {packmol_seed}\n")
        pack.write_text("".join(packinp), encoding="utf-8")

        _run(["packmol"], siminp, "packmol", stdin_file=pack)
        _run(
            ["fftool", str(n), "opc3.zmat", "--box", box, "--lmp"], siminp, "fftool_lmp"
        )
        if not (siminp / "data.lmp").exists():
            raise RuntimeError(f"fftool did not create {siminp / 'data.lmp'}.")

    def _run_lammps(self, n: int, simout: Path) -> None:
        """Run LAMMPS in ``simout`` with input from ``../siminp``.

        Parameters
        ----------
        n : int
            Number of molecules (sets the number of MPI ranks).
        simout : Path
            Simulation output directory (working directory of LAMMPS).
        """
        cmd = self.lammps_cmd.format(ncpu=calc_cpus(n), input="../siminp/input.lmp")
        _run(shlex.split(cmd), simout, "lammps_stdout")
        if not (simout / "prod.lammpstrj").exists():
            raise RuntimeError(f"LAMMPS did not write {simout / 'prod.lammpstrj'}.")

    def _analyse(self, msd: Path, lx_angstrom: float) -> float:
        """Run TRAVIS and msdiff and parse D.

        Parameters
        ----------
        msd : Path
            Analysis directory.
        lx_angstrom : float
            Cubic box edge length in Angstrom.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.

        Raises
        ------
        RuntimeError
            If TRAVIS produces no unique ``msd_*.csv`` or msdiff fails.
        """
        shutil.copy(self.travis_template, msd / "travis_input_msd.txt")
        # TRAVIS is serial: call it directly, never through mpirun.
        _run(
            [
                "travis",
                "-i",
                "travis_input_msd.txt",
                "-p",
                "../simout/prod.lammpstrj",
            ],
            msd,
            "travis",
            stdout_name="travis.log",  # same name as in the user's own runs
        )
        msd_file = find_travis_msd_file(msd)
        # msdiff expects the box length in pm; no Hummer correction unless the
        # user passes --hummer via msdiff_args.
        _run(
            [
                "msdiff",
                "-f",
                msd_file.name,
                "-l",
                f"{lx_angstrom * 100.0:.4f}",
                *self.msdiff_args,
            ],
            msd,
            "msdiff",
        )
        return parse_msdiff_output(msd / "msdiff_out.csv")


def find_travis_msd_file(msd: Path) -> Path:
    """Locate the MSD table written by TRAVIS.

    TRAVIS writes ``msd_H2O_#2.csv`` (the MSD) and ``msd_H2O_#2_fit.csv``
    (regression curve) into its working directory; only the former is used and
    the latter is left untouched.

    Parameters
    ----------
    msd : Path
        Analysis directory.

    Returns
    -------
    Path
        The MSD file.

    Raises
    ------
    RuntimeError
        If ``TRAVIS_MSD_FILE`` does not exist in ``msd``.
    """
    path = msd / TRAVIS_MSD_FILE
    if not path.is_file():
        raise RuntimeError(
            f"TRAVIS did not write {path}; found {sorted(p.name for p in msd.iterdir())}."
        )
    return path
