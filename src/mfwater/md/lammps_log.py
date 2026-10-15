"""Parser for the computational cost recorded in a LAMMPS log file.

Paper I defines the cost of one evaluation as the LAMMPS ``Total wall time``
multiplied by the number of MPI ranks (core-seconds), including minimisation
and equilibration. This is the unit of the MFMC budgets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: ``Total wall time: 1:23:45`` (hours may exceed two digits).
_WALL = re.compile(r"^Total wall time:\s*(\d+):(\d{2}):(\d{2})\s*$")

#: ``Loop time of 12.3 on 6 procs for 1000 steps with 3072 atoms``.
_LOOP = re.compile(r"^Loop time of\s+\S+\s+on\s+(\d+)\s+procs\s+for\b")


@dataclass(frozen=True)
class LammpsCost:
    """Cost of one LAMMPS run read from its log.

    Parameters
    ----------
    wall_time_s : float
        ``Total wall time`` in seconds (LAMMPS prints whole seconds).
    n_mpi : int
        Number of MPI ranks the run used.
    """

    wall_time_s: float
    n_mpi: int

    @property
    def cpu_time_s(self) -> float:
        """Cost in core-seconds: wall time times MPI ranks."""
        return self.wall_time_s * self.n_mpi


def parse_lammps_log(path: Path) -> LammpsCost:
    """Read wall time and number of MPI ranks from one LAMMPS log file.

    Only the given file is used; nothing is carried over from other files. The
    last ``Total wall time`` line is taken (the one LAMMPS prints at the very
    end of a finished run). The number of ranks comes from the ``Loop time of
    ... on N procs`` lines, which must all agree.

    Parameters
    ----------
    path : Path
        ``log.lammps`` of a finished run.

    Returns
    -------
    LammpsCost
        Wall time and ranks.

    Raises
    ------
    RuntimeError
        If the file cannot be read, lacks the ``Total wall time`` line (run
        interrupted) or any ``Loop time`` line, or the ranks of the loops
        differ.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Cannot read LAMMPS log {path}: {exc}.") from exc
    wall: float | None = None
    procs: set[int] = set()
    for line in lines:
        if m := _WALL.match(line):
            h, mi, s = (int(x) for x in m.groups())
            wall = float(3600 * h + 60 * mi + s)
        elif m := _LOOP.match(line):
            procs.add(int(m.group(1)))
    if wall is None:
        raise RuntimeError(
            f"LAMMPS log {path} has no 'Total wall time:' line (run unfinished?)."
        )
    if not procs:
        raise RuntimeError(
            f"LAMMPS log {path} has no 'Loop time of ... on N procs' line."
        )
    if len(procs) > 1:
        raise RuntimeError(f"LAMMPS log {path} reports different rank counts {procs}.")
    return LammpsCost(wall_time_s=wall, n_mpi=procs.pop())
