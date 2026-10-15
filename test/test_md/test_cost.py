"""Tests for the LAMMPS log parser and the recorded evaluation cost."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.md import pipeline
from mfwater.md.cache import CachedForwardModel
from mfwater.md.lammps_log import LammpsCost, parse_lammps_log
from mfwater.md.pipeline import MDForwardModel
from mfwater.md.protocol import MFDA_PROTOCOL

from ..test_mfda.test_forward_model import interrupted_lammps, make_fake_run

THETA = np.array([0.1634, 3.17427, -0.8952])
EQ, PROD_END = MFDA_PROTOCOL.eq_end_step, MFDA_PROTOCOL.prod_end_step


def log_text(wall: str | None = "0:12:34", procs: tuple[int, ...] = (6, 6)) -> str:
    """Excerpt of a LAMMPS log (minimisation + two runs, final timing lines)."""
    out = ["LAMMPS (23 Jun 2022)\n", "Per MPI rank memory allocation (min/avg/max)\n"]
    for k, p in enumerate(procs):
        out.append(
            f"Loop time of {10.5 * (k + 1)} on {p} procs for 1000 steps with 3072 atoms\n"
        )
        out.append("\nPerformance: 1.0 ns/day\n\n")
    out.append("Total # of neighbors = 123\n")
    if wall is not None:
        out.append(f"Total wall time: {wall}\n")
    return "".join(out)


# ------------------------------------------------------------------ the parser
def test_parse_cost(tmp_path: Path) -> None:
    f = tmp_path / "log.lammps"
    f.write_text(log_text("0:12:34"))
    cost = parse_lammps_log(f)
    assert cost == LammpsCost(wall_time_s=754.0, n_mpi=6)
    assert cost.cpu_time_s == 754.0 * 6  # core-seconds as in paper I


def test_parse_long_run_hours_above_99(tmp_path: Path) -> None:
    f = tmp_path / "log.lammps"
    f.write_text(log_text("123:00:01", (2, 2)))
    assert parse_lammps_log(f).wall_time_s == 123 * 3600 + 1


def test_parse_takes_last_wall_time(tmp_path: Path) -> None:
    f = tmp_path / "log.lammps"
    f.write_text(log_text("0:00:05") + log_text("0:00:09"))
    assert parse_lammps_log(f).wall_time_s == 9.0


def test_parse_errors_are_explicit(tmp_path: Path) -> None:
    f = tmp_path / "log.lammps"
    f.write_text(log_text(None))
    with pytest.raises(RuntimeError, match="no 'Total wall time:'"):
        parse_lammps_log(f)
    f.write_text("Total wall time: 0:00:10\n")
    with pytest.raises(RuntimeError, match="no 'Loop time"):
        parse_lammps_log(f)
    f.write_text(log_text("0:00:10", (6, 8)))
    with pytest.raises(RuntimeError, match="different rank counts"):
        parse_lammps_log(f)
    f.write_text("Total wall time: soon\nLoop time of 1 on 2 procs for 3 steps\n")
    with pytest.raises(RuntimeError, match="no 'Total wall time:'"):
        parse_lammps_log(f)
    with pytest.raises(RuntimeError, match="Cannot read"):
        parse_lammps_log(tmp_path / "missing.lammps")


def test_parse_never_reuses_values_of_other_files(tmp_path: Path) -> None:
    good, bad = tmp_path / "a.lammps", tmp_path / "b.lammps"
    good.write_text(log_text("0:00:10", (4, 4)))
    bad.write_text(log_text(None, (4, 4)))
    parse_lammps_log(good)
    with pytest.raises(RuntimeError):  # no stale wall time from the first call
        parse_lammps_log(bad)


# ------------------------------------------------------ cost in the MD pipeline
def fake_lammps_with_log(
    wall: str, procs: tuple[int, ...] = (3, 3)
) -> Callable[[list[str], Path], None]:
    """LAMMPS fake that writes a trajectory and a complete log."""

    def hook(args: list[str], cwd: Path) -> None:
        (cwd / "prod.lammpstrj").write_text("traj")
        (cwd / "log.lammps").write_text(log_text(wall, procs))

    return hook


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[Any]]:
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")

    def install(hook: Callable[[list[str], Path], None] | None = None) -> list[Any]:
        calls: list[Any] = []
        monkeypatch.setattr(
            pipeline.subprocess, "run", make_fake_run(calls, lammps_hook=hook)
        )
        return calls

    return install


def test_result_json_records_cost(
    tmp_path: Path, patched: Callable[..., list[Any]]
) -> None:
    patched(fake_lammps_with_log("0:00:10"))
    cached = CachedForwardModel(MDForwardModel(tmp_path), tmp_path)
    cached(100, THETA)
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    assert data["cpu_time_s"] == 30.0 and data["n_mpi"] == 3
    assert data["protocol"] == "mfda"
    assert data["cost_complete"] is True
    assert [a["cpu_time_s"] for a in data["cost_attempts"]] == [30.0]


def test_unreadable_log_is_recorded_not_fatal_by_default(
    tmp_path: Path, patched: Callable[..., list[Any]]
) -> None:
    patched()  # default fake writes a log without timing lines
    cached = CachedForwardModel(MDForwardModel(tmp_path), tmp_path)
    assert cached(100, THETA) == pytest.approx(2.5)
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    assert data["cpu_time_s"] is None and data["n_mpi"] is None
    assert "Total wall time" in data["cost_error"]
    assert data["cost_complete"] is False


def test_require_cost_raises_before_the_analysis(
    tmp_path: Path, patched: Callable[..., list[Any]]
) -> None:
    calls = patched()
    model = MDForwardModel(tmp_path, require_cost=True)
    with pytest.raises(RuntimeError, match="Total wall time"):
        model(100, THETA)
    assert "travis" not in [c[0][0] for c in calls]


def test_non_md_model_has_null_cost(tmp_path: Path) -> None:
    cached = CachedForwardModel(lambda n, th: 1.0, tmp_path)
    cached(100, THETA)
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    assert data["cpu_time_s"] is None and data["n_mpi"] is None


def test_cost_of_resumed_evaluation_sums_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    calls: list[Any] = []
    model = MDForwardModel(tmp_path, require_cost=True)
    cached = CachedForwardModel(model, tmp_path)

    def install(inner: Callable[[list[str], Path], None], log: str) -> None:
        def hook(args: list[str], cwd: Path) -> None:
            (cwd / "log.lammps").write_text(log)
            inner(args, cwd)

        monkeypatch.setattr(
            pipeline.subprocess, "run", make_fake_run(calls, lammps_hook=hook)
        )

    # attempt 1 is killed: its log has loop times but no 'Total wall time'
    install(
        interrupted_lammps(range(EQ, 450_001, 1000), (200_000, 300_000, 400_000)),
        log_text(None, (4,)),
    )
    with pytest.raises(RuntimeError, match="mpirun"):
        cached(100, THETA)
    # attempt 2 resumes from restart.400000 and finishes: 20 s on 4 ranks
    install(
        interrupted_lammps(range(400_000, PROD_END + 1, 1000), (), finish=True),
        log_text("0:00:20", (4,)),
    )
    cached(100, THETA)
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    # killed attempt: cost unknown, so the sum is a documented lower bound
    assert data["cpu_time_s"] == 80.0 and data["n_mpi"] == 4
    assert data["cost_complete"] is False
    assert [a["attempt"] for a in data["cost_attempts"]] == ["attempt_1", "attempt_2"]
    assert "error" in data["cost_attempts"][0]
    assert data["cost_attempts"][1]["cpu_time_s"] == 80.0


def test_cost_sums_complete_logs_of_all_attempts(
    tmp_path: Path, patched: Callable[..., list[Any]]
) -> None:
    patched(fake_lammps_with_log("0:00:10", (2,)))
    model = MDForwardModel(tmp_path)
    entries = []
    for _ in range(2):
        model(100, THETA)
        assert model.last_attempt_dir is not None
        entries.append(model.last_attempt_dir)
    assert model._record_cost(entries) == {
        "cpu_time_s": 40.0,
        "n_mpi": 2,
        "n_mpi_requested": 2,  # calc_cpus(100)
        "cost_attempts": [
            {"attempt": a.name, "wall_time_s": 10.0, "n_mpi": 2, "cpu_time_s": 20.0}
            for a in entries
        ],
        "cost_complete": True,
    }
