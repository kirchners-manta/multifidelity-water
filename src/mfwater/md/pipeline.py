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
interrupted run) are never modified or deleted. LAMMPS writes a restart file
every 100 ps (``simout/restart.<step>``). If the newest attempt of an evaluation
was interrupted after writing restart files, the next call creates
``attempt_{k+1}`` that resumes from the newest restart file (recorded in
``attempt_{k+1}/resume_from.txt``) instead of starting over. The production
trajectory is then spread over several attempts and merged into a new file
``msd/prod_combined.lammpstrj`` for TRAVIS.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..msdiff_io import parse_msdiff_d_raw
from .forward_model import (
    _check_theta,
    derive_seeds,
    evaluation_key,
    key_hash,
)
from .lammps_log import parse_lammps_log
from .protocol import MFDA_PROTOCOL, MDProtocol, get_protocol
from .system import calc_box_size, calc_cpus

#: Directory with the paper-I templates (LAMMPS input, force field, TRAVIS input).
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "algo_chemical_model" / "data"

#: Prompts of the TRAVIS template whose answer (next line) is set by MFDA.
TRAVIS_FRAMES_PROMPT = "How many trajectory frames to read"
TRAVIS_DEPTH_PROMPT = "Enter the correlation depth"

#: Name of the merged production trajectory (in ``msd/``) of a resumed evaluation.
COMBINED_TRAJECTORY = "prod_combined.lammpstrj"

#: Default LAMMPS command. ``{ncpu}`` and ``{input}`` are filled in; LAMMPS runs in
#: ``simout/`` and reads ``../siminp/input.lmp``.
DEFAULT_LAMMPS_CMD = "mpirun -np {ncpu} lmp -i {input}"

#: MSD file written by TRAVIS for water with the template (atom #2, molecule H2O).
TRAVIS_MSD_FILE = "msd_H2O_#2.csv"

#: MSD files written by TRAVIS with the OrthoBoXY template (x-y plane and z only);
#: names as in the paper-I scripts (``scripts/pp_subdir_o.sh``).
TRAVIS_MSD_FILE_XY = "msd_H2O_#2_XY.csv"
TRAVIS_MSD_FILE_Z = "msd_H2O_#2_Z.csv"

#: msdiff options of paper I: Hummer correction for 298.15 K and a viscosity of
#: 0.89e-3 kg/(m s) with zero uncertainty. Needed to be comparable to paper-I data.
PAPER_I_MSDIFF_ARGS = ("--hummer", "298.15", "0.89e-3", "0.0")

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
    protocol: MDProtocol | None = None,
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
    protocol : MDProtocol | None, optional
        If given, the template is first checked against the protocol
        (:meth:`MDProtocol.validate_template`).

    Raises
    ------
    RuntimeError
        If a placeholder or required line is missing from the template or the
        template contradicts ``protocol``.
    """
    if protocol is not None:
        protocol.validate_template()
    eps, sig, q_o = (float(x) for x in theta)
    q_h = -0.5 * q_o  # neutral molecule
    # Rounding: LAMMPS receives epsilon and sigma with 6 decimals and q with 8,
    # while the cache key (evaluation_key) uses the exact float64 values. The
    # rounding (1e-6) is far below the kernel widths of the sampler
    # (s_sigma ~ 1e-2 Angstrom, s_eps ~ 5e-4 kcal/mol), so it is irrelevant.
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


RESUME_HEADER = """\
# Resume input generated by mfwater (md_pipeline.write_resume_input).
# Continues the run of {template} from a LAMMPS restart file.
#
# Stored in the restart file (not repeated here): units, atom_style, box,
# timestep size and number, per-atom data (positions, image flags, velocities,
# charges, topology), pair_style, bond_style, angle_style, special_bonds,
# pair_modify and, for the lj/cut styles, the pair coefficients.
# Not stored (re-specified below): kspace_style, neighbor, fixes, variables,
# thermo and dump settings, restart output.
# Not issued (would be wrong or would discard restart state): minimize,
# velocity create, reset_timestep, set charge.
# The fix nvt keeps its ID (fTHERMO) and style so that LAMMPS restores the
# thermostat state from the restart file.
"""


def write_resume_input(
    target: Path,
    restart_file: Path,
    restart_step: int,
    protocol: MDProtocol = MFDA_PROTOCOL,
) -> None:
    """Write ``input_resume.lmp`` that continues a run from a restart file.

    The settings that a restart file does not store are repeated exactly as in
    the LAMMPS template of the protocol; ``run ... upto`` stops at the same
    absolute timestep as in an uninterrupted run. The stage follows from
    ``restart_step``: below ``protocol.eq_end_step`` the equilibration is
    finished first, otherwise only the production run is performed.

    Parameters
    ----------
    target : Path
        Output file.
    restart_file : Path
        LAMMPS restart file (binary) to read; its absolute path is written.
    restart_step : int
        Timestep of ``restart_file``.
    protocol : MDProtocol, optional
        Protocol of the interrupted run (default: MFDA).

    Raises
    ------
    ValueError
        If ``restart_step`` is not in ``[0, protocol.prod_end_step)`` (``run
        upto`` with a target in the past is an error in LAMMPS).
    """
    eq_end, prod_end = protocol.eq_end_step, protocol.prod_end_step
    if not 0 <= restart_step < prod_end:
        raise ValueError(
            f"Cannot resume from step {restart_step}: must be in [0, {prod_end})."
        )
    lines = [
        RESUME_HEADER.replace("{template}", protocol.lammps_template.name),
        f'read_restart "{restart_file.resolve()}"\n',
        "# kspace is not stored in restart files",
        "kspace_style pppm 1.0e-5",
        "neighbor 2.0 bin",
        "timestep 1.0",
        "variable vTK equal 298.15",
        "variable vPBAR equal 1.0",
        "",
        f"# same order and IDs as in {protocol.lammps_template.name}",
        "fix fSHAKE all shake 0.0001 20 0 b 1 a 1",
        "fix fMOM all momentum 100 linear 1 1 1",
        "fix fTHERMO all nvt temp ${vTK} ${vTK} 100",
        f"restart {protocol.restart_every} restart.*",
        "",
        "thermo 1000",
        "thermo_style custom step etotal ke pe temp press vol density",
        "",
    ]
    dump = protocol.dump_every
    if restart_step < eq_end:
        lines += [
            f"dump dEQUI all custom {dump} equi.lammpstrj id element xu yu zu",
            "dump_modify dEQUI element H O sort id",
            f"run {eq_end} upto",
            "write_data data.after_eq.lmp",
            "undump dEQUI",
            "",
        ]
    elif restart_step == eq_end:
        # the interrupted run ended its equilibration but may not have written this
        lines += ["write_data data.after_eq.lmp", ""]
    lines += [
        f"dump dPROD all custom {dump} prod.lammpstrj id element xu yu zu",
        "dump_modify dPROD element H O sort id",
        f"run {prod_end} upto",
        "write_data data.after_prod.lmp",
        "undump dPROD",
        "unfix fTHERMO",
        "unfix fMOM",
        "unfix fSHAKE",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def find_restart_files(
    simout: Path, include_empty: bool = False
) -> list[tuple[int, Path]]:
    """List the non-empty LAMMPS restart files of a ``simout`` directory.

    Parameters
    ----------
    simout : Path
        LAMMPS working directory.
    include_empty : bool, optional
        Also list empty files (default False).

    Returns
    -------
    list[tuple[int, Path]]
        ``(timestep, path)`` for files ``restart.<step>``, sorted by timestep.
        Empty files (interrupted write) are ignored unless ``include_empty``.
    """
    found = []
    if simout.is_dir():
        for p in simout.iterdir():
            m = re.fullmatch(r"restart\.(\d+)", p.name)
            if m and p.is_file() and (include_empty or p.stat().st_size > 0):
                found.append((int(m.group(1)), p))
    return sorted(found)


def select_restart_file(
    candidates: list[tuple[int, Path]], other_sizes: list[int]
) -> tuple[tuple[int, Path] | None, list[Path]]:
    """Pick the newest restart file that does not look truncated.

    All restart files of one evaluation (fixed number of molecules) are
    expected to have the same size: LAMMPS writes a binary file whose size is
    set by the number of atoms and the stored state. A file cut short, e.g. by
    a job killed while writing it, is therefore smaller than its predecessor.
    This is a heuristic, not a validity check of the content.

    UNVERIFIED: that the sizes are constant between the equilibration and the
    production restart files (e.g. extra fix state in the NPT stage). If a real
    run shows otherwise, adapt only this function (e.g. compare within a
    stage, or allow a relative tolerance).

    Rules: the newest file is accepted only if its size equals that of the
    next-older candidate, otherwise it is skipped and the next-older one is
    checked in the same way. The oldest candidate has no predecessor; it is
    accepted if it is non-empty and equals every size in ``other_sizes``
    (restart files of the evaluation that are not candidates, e.g. beyond the
    final step). Nothing is deleted or modified.

    Parameters
    ----------
    candidates : list[tuple[int, Path]]
        ``(timestep, path)`` of the non-empty candidates, sorted by timestep.
    other_sizes : list[int]
        Sizes in bytes of the other non-empty restart files of the evaluation.

    Returns
    -------
    tuple[tuple[int, Path] | None, list[Path]]
        The accepted ``(timestep, path)`` or None if none is valid, and the
        skipped files (newest first).
    """
    skipped: list[Path] = []
    for i in range(len(candidates) - 1, -1, -1):
        step, path = candidates[i]
        size = path.stat().st_size
        if i > 0:
            ok = size == candidates[i - 1][1].stat().st_size
        else:
            ok = size > 0 and all(size == other for other in other_sizes)
        if ok:
            return (step, path), skipped
        skipped.append(path)
    return None, skipped


def iter_frames(path: Path) -> Iterator[tuple[int, list[str]]]:
    """Iterate over the complete frames of a LAMMPS custom text trajectory.

    Parameters
    ----------
    path : Path
        ``.lammpstrj`` file (``ITEM: TIMESTEP`` ... ``ITEM: ATOMS`` layout).

    Yields
    ------
    tuple[int, list[str]]
        Timestep and the lines of the frame (with line endings).

    Raises
    ------
    RuntimeError
        If the file is malformed or ends inside a frame (after all complete
        frames have been yielded), e.g. because LAMMPS was killed while writing.
    """
    with open(path, encoding="utf-8") as f:
        while True:
            first = f.readline()
            if not first:
                return
            if not first.startswith("ITEM: TIMESTEP"):
                raise RuntimeError(f"{path}: expected 'ITEM: TIMESTEP', got {first!r}.")
            head = [first]
            try:
                for _ in range(3):  # timestep, ITEM: NUMBER OF ATOMS, count
                    head.append(f.readline())
                step, n_atoms = int(head[1]), int(head[3])
                body = [f.readline() for _ in range(5 + n_atoms)]  # box + ATOMS
            except ValueError as exc:
                raise RuntimeError(f"{path}: truncated or malformed frame.") from exc
            # a killed LAMMPS may leave an empty or half-written last line
            if any(not line.endswith("\n") for line in head + body):
                raise RuntimeError(f"{path}: truncated frame at step {step}.")
            yield step, head + body


def _set_answers(lines: list[str], prompt: str, answer: str) -> int:
    """Replace the answer line below every prompt containing ``prompt``.

    Parameters
    ----------
    lines : list[str]
        Template lines (modified in place).
    prompt : str
        Text identifying the TRAVIS prompt line (starting with ``!``).
    answer : str
        New answer.

    Returns
    -------
    int
        Number of replaced answers.
    """
    n = 0
    for i, line in enumerate(lines):
        if line.startswith("!") and prompt in line:
            if i + 1 >= len(lines):
                raise ValueError(f"TRAVIS template ends right after prompt {prompt!r}.")
            lines[i + 1] = answer
            n += 1
    return n


def patch_travis_input(
    template: str,
    frames: int | None = MFDA_PROTOCOL.travis_frames,
    depth: int | None = MFDA_PROTOCOL.travis_depth,
) -> str:
    """Set the number of frames and the correlation depth of a TRAVIS input.

    The shared paper-I templates hard-code 2000 frames and depth 600 (or read
    all frames); MFDA uses a shorter production run, so the answers are
    replaced in a copy. Every depth prompt is patched (the OrthoBoXY template
    has one per MSD observation). The start-frame answer is not touched. The
    defaults are the values of the MFDA protocol.

    Parameters
    ----------
    template : str
        Content of the TRAVIS input template.
    frames : int | None, optional
        Trajectory frames to read; None keeps the answer of the template.
    depth : int | None, optional
        MSD correlation depth in frames; None keeps the answer of the template.

    Returns
    -------
    str
        Patched input.

    Raises
    ------
    ValueError
        If a value is to be set but the template lacks the frames prompt or any
        depth prompt.
    """
    lines = template.split("\n")
    if (
        frames is not None
        and _set_answers(lines, TRAVIS_FRAMES_PROMPT, str(frames)) != 1
    ):
        raise ValueError(
            f"TRAVIS template needs exactly one {TRAVIS_FRAMES_PROMPT!r} prompt."
        )
    if depth is not None and _set_answers(lines, TRAVIS_DEPTH_PROMPT, str(depth)) < 1:
        raise ValueError(f"TRAVIS template lacks a {TRAVIS_DEPTH_PROMPT!r} prompt.")
    return "\n".join(lines)


def merge_trajectories(
    segments: Sequence[tuple[Path, int | None]],
    target: Path,
    first_step: int = MFDA_PROTOCOL.eq_end_step,
    last_step: int = MFDA_PROTOCOL.prod_end_step,
    spacing: int = MFDA_PROTOCOL.dump_every,
) -> int:
    """Merge production trajectory pieces of several attempts into a new file.

    Each piece contributes its frames with timestep below its ``stop`` (the
    restart step of the next attempt; ``None`` = all frames). A resumed run
    writes a frame at its restart step again, so this cut removes duplicates;
    a frame repeated within the result is skipped. The result must have
    constant spacing and run from ``first_step`` to ``last_step``.

    Parameters
    ----------
    segments : Sequence[tuple[Path, int | None]]
        ``(trajectory, stop_step)`` in chronological order. Missing files are
        skipped for pieces with a ``stop_step`` (an attempt that never reached
        production) but are an error for the last piece.
    target : Path
        New output file; must not exist. Written as ``<target>.part`` and
        renamed once complete and validated.
    first_step, last_step : int, optional
        Expected first and last timestep (default: production of the MFDA
        protocol; pass ``protocol.eq_end_step`` and ``protocol.prod_end_step``).
    spacing : int, optional
        Expected distance of consecutive frames in steps.

    Returns
    -------
    int
        Number of frames written.

    Raises
    ------
    RuntimeError
        If ``target`` exists, a file is missing/truncated, frames are not
        ordered, the spacing is not constant or the range is incomplete.
    """
    if target.exists():
        raise RuntimeError(f"{target} already exists; not overwriting.")
    part = target.with_name(target.name + ".part")
    n_frames = 0
    prev: int | None = None
    with open(part, "x", encoding="utf-8") as out:
        for k, (path, stop) in enumerate(segments):
            if not path.exists():
                if stop is None or k == len(segments) - 1:
                    raise RuntimeError(f"Trajectory {path} is missing.")
                continue
            for step, lines in iter_frames(path):
                if stop is not None and step >= stop:
                    break  # later frames belong to the resumed attempt
                if prev is not None and step == prev:
                    continue
                if prev is None:
                    if step != first_step:
                        raise RuntimeError(
                            f"{path}: first frame is step {step}, expected {first_step}."
                        )
                elif step != prev + spacing:
                    raise RuntimeError(
                        f"{path}: step {step} follows step {prev}, expected a "
                        f"constant spacing of {spacing} (gap or disorder)."
                    )
                out.writelines(lines)
                prev, n_frames = step, n_frames + 1
    if prev != last_step:
        raise RuntimeError(
            f"Combined trajectory ends at step {prev}, expected {last_step}; "
            f"partial result left in {part}."
        )
    os.rename(part, target)
    return n_frames


def parse_msdiff_output(path: Path) -> float:
    """Read the diffusion coefficient from ``msdiff_out.csv`` in 1e-9 m^2/s.

    Thin wrapper around :func:`mfwater.msdiff_io.parse_msdiff_d_raw` (which
    handles both msdiff layouts and returns 1e-12 m^2/s) that converts to the
    unit of the likelihood.

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
    return parse_msdiff_d_raw(path) * MSDIFF_TO_NANO


class MDForwardModel:
    """Forward model running one MD simulation per evaluation.

    fftool, packmol, LAMMPS (via ``lammps_cmd``), TRAVIS and msdiff must be in
    ``PATH`` (on Marvin: ``module load LAMMPS/23Jun2022-foss-2022a-kokkos``).
    The TRAVIS input is the user's template
    ``algo_chemical_model/data/travis_input_msd.txt`` (fixed cell, 1000 fs
    between frames, atom ``#2`` = molecular centre of mass, confirmed by the
    user). MFDA writes a patched copy into ``msd/`` that reads
    the frames and correlation depth of the protocol; the shared template is not
    modified. Run lengths, dump interval and the TRAVIS settings are those of
    the :class:`~mfwater.md.protocol.MDProtocol` (default ``"mfda"``: 250 ps
    equilibration + 1 ns production, 1000 frames, depth 300 = 30 %). The
    TRAVIS inputs are the shared paper-I templates in both cases.

    Parameters
    ----------
    work_dir_root : str | Path
        Root directory; evaluations live in ``<root>/N{n}/{key_hash}/attempt_{k}``.
        Use the same directory as the cache of
        :class:`mfwater.md.cache.CachedForwardModel`.
    orthoboxy : bool, optional
        Tetragonal OrthoBoXY boxes (``calc_box_size(n, orthoboxy_shape=True)``).
        TRAVIS then uses ``travis_input_msd_orthoboxy.txt`` (two MSD
        observations, x-y and z) and msdiff is called with ``--orthoboxy``.
    lammps_cmd : str, optional
        Command template with ``{ncpu}`` and ``{input}``.
    travis_template : str | Path | None, optional
        TRAVIS input file; default is the cubic or OrthoBoXY template in the
        package, depending on ``orthoboxy``.
    msdiff_args : Sequence[str] | None, optional
        msdiff arguments besides ``-f``, ``--from-travis`` and ``--orthoboxy``.
        Default is the paper-I call, :data:`PAPER_I_MSDIFF_ARGS`. The box
        length is read by msdiff from ``travis.log`` (``--from-travis``).
    protocol : str | MDProtocol, optional
        Registered protocol name (``"mfda"`` or ``"paper1"``) or a protocol
        object. Default ``"mfda"``, which reproduces the MFDA evaluations
        exactly.
    require_cost : bool, optional
        Raise if the cost cannot be read from the LAMMPS log of the (last)
        attempt. By default a failure is only recorded (``cost_error``): the
        cost is a by-product and must not kill a long MFDA evaluation. MFMC
        needs the cost, so its callers set this to True.

    Raises
    ------
    ValueError
        If ``protocol`` is unknown.

    Attributes
    ----------
    protocol : MDProtocol
        The resolved protocol.
    ----------
    last_attempt_dir : Path | None
        Attempt directory of the most recent call (the last one used).
    last_attempt_dirs : list[Path]
        All attempt directories that contributed to the most recent result, in
        chronological order (several if the evaluation was resumed).
    last_cost : dict | None
        Cost of the most recent evaluation, written into ``result.json`` by
        :class:`~mfwater.md.cache.CachedForwardModel`; see :meth:`_record_cost`.
    """

    def __init__(
        self,
        work_dir_root: str | Path,
        orthoboxy: bool = False,
        lammps_cmd: str = DEFAULT_LAMMPS_CMD,
        travis_template: str | Path | None = None,
        msdiff_args: Sequence[str] | None = None,
        protocol: str | MDProtocol = "mfda",
        require_cost: bool = False,
    ) -> None:
        self.work_dir_root = Path(work_dir_root)
        self.protocol = get_protocol(protocol)
        self.orthoboxy = orthoboxy
        self.lammps_cmd = lammps_cmd
        self.travis_template = (
            Path(travis_template)
            if travis_template is not None
            else TEMPLATE_DIR
            / (
                "travis_input_msd_orthoboxy.txt"
                if orthoboxy
                else "travis_input_msd.txt"
            )
        )
        self.msdiff_args = tuple(
            PAPER_I_MSDIFF_ARGS if msdiff_args is None else msdiff_args
        )
        self.last_attempt_dir: Path | None = None
        self.last_attempt_dirs: list[Path] = []
        self.require_cost = require_cost
        self.last_cost: dict[str, Any] | None = None

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

    @staticmethod
    def _newest_attempt(base: Path) -> Path | None:
        """Return the attempt directory with the highest number, if any.

        Parameters
        ----------
        base : Path
            ``<root>/N{n}/{key_hash}``.

        Returns
        -------
        Path | None
            Newest ``attempt_{k}`` or None.
        """
        if not base.is_dir():
            return None
        found = [
            (int(m.group(1)), p)
            for p in base.iterdir()
            if (m := re.fullmatch(r"attempt_(\d+)", p.name)) and p.is_dir()
        ]
        return max(found)[1] if found else None

    @staticmethod
    def _read_resume_from(attempt: Path) -> Path | None:
        """Return the restart file an attempt was resumed from (absolute path).

        Parameters
        ----------
        attempt : Path
            Attempt directory.

        Returns
        -------
        Path | None
            Restart file, or None if the attempt started fresh.
        """
        marker = attempt / "resume_from.txt"
        if not marker.is_file():
            return None
        # stored relative to the evaluation directory (<root>/N{n}/{hash}) so
        # that the whole work directory can be moved
        return attempt.parent / marker.read_text(encoding="utf-8").strip()

    def _resume_point(self, attempt: Path) -> tuple[int, Path] | None:
        """Find the restart file from which an interrupted attempt can go on.

        Parameters
        ----------
        attempt : Path
            Newest attempt of an evaluation.

        Returns
        -------
        tuple[int, Path] | None
            ``(timestep, restart file)`` or None if the attempt is complete
            (analysis finished) or has no usable restart file. Restart files at
            or after ``protocol.prod_end_step`` are ignored (``run upto`` could not
            continue from them). The files of all attempts of the chain are
            considered (several interruptions); truncated-looking ones are
            skipped, see :func:`select_restart_file`.
        """
        if (attempt / "msd" / "msdiff_out.csv").exists():
            return None
        # Candidates: restart files of the whole attempt chain. An older attempt
        # is valid only up to the step its successor resumed from; the same step
        # may exist in two attempts, the later attempt then wins.
        by_step: dict[int, Path] = {}
        used: set[Path] = set()
        for att, stop in self._attempt_chain(attempt):
            for step, path in find_restart_files(att / "simout"):
                if step < self.protocol.prod_end_step and (
                    stop is None or step <= stop
                ):
                    by_step[step] = path
                    used.add(path)
        candidates = sorted(by_step.items())
        # Everything else (files beyond the final step or beyond a stop step,
        # empty files excluded) serves as size reference for the oldest candidate.
        every = [
            p
            for att in attempt.parent.glob("attempt_*")
            for _, p in find_restart_files(att / "simout", include_empty=True)
        ]
        others = [p.stat().st_size for p in every if p not in used and p.stat().st_size]
        point, skipped = select_restart_file(candidates, others)
        for path in skipped:
            print(
                f"Warning: skipping restart file {path} "
                "(size differs from the older restart files, probably truncated)."
            )
        if point is None and every:
            print(
                f"Warning: no valid restart file in {attempt.parent}; "
                "starting from the beginning."
            )
        return point

    def _attempt_chain(self, attempt: Path) -> list[tuple[Path, int | None]]:
        """Follow ``resume_from.txt`` back to the fresh attempt.

        Parameters
        ----------
        attempt : Path
            Last attempt of the evaluation.

        Returns
        -------
        list[tuple[Path, int | None]]
            ``(attempt, stop_step)`` from the oldest to the newest attempt;
            ``stop_step`` is the restart step the next attempt resumed from
            (``None`` for the last one).
        """
        chain: list[tuple[Path, int | None]] = [(attempt, None)]
        current = attempt
        while (src := self._read_resume_from(current)) is not None:
            m = re.fullmatch(r"restart\.(\d+)", src.name)
            if m is None or src.parent.name != "simout":
                raise RuntimeError(f"Malformed resume_from.txt in {current}.")
            current = src.parent.parent
            # the older attempt is only valid up to the step its successor
            # restarted from (that step is repeated by the successor)
            chain.insert(0, (current, int(m.group(1))))
        return chain

    def __call__(
        self,
        n_molecules: int,
        theta: np.ndarray,
        seeds: tuple[int, int] | None = None,
    ) -> float:
        """Run the MD pipeline and return the diffusion coefficient.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        theta : np.ndarray
            ``(epsilon_OO, sigma_OO, q_O)`` in kcal/mol, Angstrom, e.
        seeds : tuple[int, int] | None, optional
            Explicit ``(packmol_seed, velocity_seed)``, e.g. shared by all
            models of an MFMC calibration sample. Default: derived from the
            evaluation key. Explicit seeds are part of the key, hence of the
            work directory name.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.

        Raises
        ------
        RuntimeError
            If a program is missing or a step fails.
        ValueError
            If ``theta`` or ``seeds`` are invalid.
        """
        theta = _check_theta(theta)
        self.last_cost = None
        for prog in self._required_programs():
            if shutil.which(prog) is None:
                raise RuntimeError(f"{prog} is not installed or not in PATH.")

        key = evaluation_key(
            n_molecules,
            theta,
            protocol=self.protocol.name,
            orthoboxy=self.orthoboxy,
            seeds=seeds,
        )
        packmol_seed, velocity_seed = derive_seeds(key)
        base = self.work_dir_root / f"N{n_molecules}" / key_hash(key)
        # Resume only if the newest attempt was interrupted after a restart file;
        # decided before the new attempt directory exists.
        newest = self._newest_attempt(base)
        point = None if newest is None else self._resume_point(newest)

        attempt = self._new_attempt_dir(n_molecules, key_hash(key))
        self.last_attempt_dir = attempt
        siminp, simout, msd = attempt / "siminp", attempt / "simout", attempt / "msd"

        if point is None:
            self._build_box(n_molecules, siminp, packmol_seed)
            write_lammps_input(
                self.protocol.lammps_template,
                siminp / "input.lmp",
                theta,
                velocity_seed,
                protocol=self.protocol,
            )
            self.last_attempt_dirs = [attempt]
            self._run_lammps(n_molecules, simout, "input.lmp")
            self.last_cost = self._record_cost([attempt])
            return self._analyse(msd, "../simout/prod.lammpstrj")

        step, restart_file = point
        # Record the source first: previous attempts are only read, never changed.
        rel = restart_file.relative_to(base)
        (attempt / "resume_from.txt").write_text(f"{rel}\n", encoding="utf-8")
        write_resume_input(
            siminp / "input_resume.lmp", restart_file, step, protocol=self.protocol
        )
        print(f"MD N={n_molecules}: resuming {attempt.name} from {rel} (step {step}).")
        chain = self._attempt_chain(attempt)
        self.last_attempt_dirs = [a for a, _ in chain]
        self._run_lammps(n_molecules, simout, "input_resume.lmp")
        self.last_cost = self._record_cost(self.last_attempt_dirs)
        merge_trajectories(
            [(a / "simout" / "prod.lammpstrj", stop) for a, stop in chain],
            msd / COMBINED_TRAJECTORY,
            first_step=self.protocol.eq_end_step,
            last_step=self.protocol.prod_end_step,
            spacing=self.protocol.dump_every,
        )
        return self._analyse(msd, COMBINED_TRAJECTORY)

    def _record_cost(self, attempts: Sequence[Path]) -> dict[str, Any]:
        """Read the cost of an evaluation from the LAMMPS logs of its attempts.

        Each log is parsed on its own (:func:`~mfwater.md.lammps_log.parse_lammps_log`).
        The cost of a resumed evaluation is the **sum over its attempts**, so
        the work that was repeated after the last restart file counts. An
        attempt that was killed has no ``Total wall time`` line; its cost is
        unknown, so the sum is then a lower bound (``cost_complete`` False).

        Parameters
        ----------
        attempts : Sequence[Path]
            Attempt directories in chronological order; the last one finished.

        Returns
        -------
        dict
            ``cpu_time_s`` (core-seconds, None if the last log is unreadable),
            ``n_mpi`` (ranks of the last attempt), ``cost_attempts`` (per
            attempt wall time, ranks and core-seconds or the error),
            ``cost_complete`` and, on failure, ``cost_error``.

        Raises
        ------
        RuntimeError
            If the log of the last attempt cannot be parsed and
            ``require_cost`` is True.
        """
        entries: list[dict[str, Any]] = []
        total = 0.0
        complete = True
        last_ranks: int | None = None
        for k, att in enumerate(attempts):
            try:
                cost = parse_lammps_log(att / "simout" / "log.lammps")
            except RuntimeError as exc:
                entries.append({"attempt": att.name, "error": str(exc)})
                if k == len(attempts) - 1:
                    if self.require_cost:
                        raise
                    return {
                        "cpu_time_s": None,
                        "n_mpi": None,
                        "cost_attempts": entries,
                        "cost_complete": False,
                        "cost_error": str(exc),
                    }
                complete = False
                continue
            entries.append(
                {
                    "attempt": att.name,
                    "wall_time_s": cost.wall_time_s,
                    "n_mpi": cost.n_mpi,
                    "cpu_time_s": cost.cpu_time_s,
                }
            )
            total += cost.cpu_time_s
            last_ranks = cost.n_mpi
        return {
            "cpu_time_s": total,
            "n_mpi": last_ranks,
            "cost_attempts": entries,
            "cost_complete": complete,
        }

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

    def _run_lammps(self, n: int, simout: Path, input_name: str) -> None:
        """Run LAMMPS in ``simout`` with input from ``../siminp``.

        Parameters
        ----------
        n : int
            Number of molecules (sets the number of MPI ranks).
        simout : Path
            Simulation output directory (working directory of LAMMPS).
        input_name : str
            Input file in ``siminp`` (``input.lmp`` or ``input_resume.lmp``).

        Raises
        ------
        RuntimeError
            If LAMMPS fails or does not write ``prod.lammpstrj``.
        """
        cmd = self.lammps_cmd.format(ncpu=calc_cpus(n), input=f"../siminp/{input_name}")
        _run(shlex.split(cmd), simout, "lammps_stdout")
        if not (simout / "prod.lammpstrj").exists():
            raise RuntimeError(f"LAMMPS did not write {simout / 'prod.lammpstrj'}.")

    def _analyse(self, msd: Path, trajectory: str) -> float:
        """Run TRAVIS and msdiff (as in paper I) and parse D.

        msdiff reads the box lengths from ``travis.log`` in ``msd`` (the stdout
        of TRAVIS), so no box length is passed.

        Parameters
        ----------
        msd : Path
            Analysis directory.
        trajectory : str
            Production trajectory as seen from ``msd``: ``../simout/prod.lammpstrj``
            or, after a resume, the merged ``prod_combined.lammpstrj``.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.

        Raises
        ------
        RuntimeError
            If TRAVIS does not write the expected MSD file(s) or msdiff fails.
        """
        (msd / "travis_input_msd.txt").write_text(
            patch_travis_input(
                self.travis_template.read_text(),
                frames=self.protocol.travis_frames,
                depth=self.protocol.travis_depth,
            )
        )
        # TRAVIS is serial: call it directly, never through mpirun.
        _run(
            [
                "travis",
                "-i",
                "travis_input_msd.txt",
                "-p",
                trajectory,
            ],
            msd,
            "travis",
            stdout_name="travis.log",  # msdiff --from-travis reads this file
        )
        if self.orthoboxy:
            xy_file = find_travis_msd_file(msd, TRAVIS_MSD_FILE_XY)
            z_file = find_travis_msd_file(msd, TRAVIS_MSD_FILE_Z)
            msdiff_cmd = [
                "msdiff",
                "-f",
                xy_file.name,
                "--from-travis",
                *self.msdiff_args,
                "--orthoboxy",
                z_file.name,
            ]
        else:
            msd_file = find_travis_msd_file(msd, TRAVIS_MSD_FILE)
            msdiff_cmd = [
                "msdiff",
                "-f",
                msd_file.name,
                "--from-travis",
                *self.msdiff_args,
            ]
        _run(msdiff_cmd, msd, "msdiff")
        return parse_msdiff_output(msd / "msdiff_out.csv")


def find_travis_msd_file(msd: Path, name: str = TRAVIS_MSD_FILE) -> Path:
    """Locate an MSD table written by TRAVIS.

    TRAVIS also writes ``*_fit.csv`` (regression curves) into its working
    directory; those are left untouched and never used.

    Parameters
    ----------
    msd : Path
        Analysis directory.
    name : str, optional
        File name, default ``TRAVIS_MSD_FILE`` (cubic box).

    Returns
    -------
    Path
        The MSD file.

    Raises
    ------
    RuntimeError
        If ``name`` does not exist in ``msd``.
    """
    path = msd / name
    if not path.is_file():
        raise RuntimeError(
            f"TRAVIS did not write {path}; found {sorted(p.name for p in msd.iterdir())}."
        )
    return path
