"""Tests for the MPI-rank override of the MD forward model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.md import pipeline
from mfwater.md.cache import CachedForwardModel
from mfwater.md.forward_model import evaluation_key, key_hash
from mfwater.md.pipeline import MDForwardModel
from mfwater.md.system import calc_cpus

from ..test_mfda.test_forward_model import make_fake_run

THETA = np.array([0.1634, 3.17427, -0.8952])


def lammps_ranks(calls: list[Any]) -> int:
    """Value after ``-np`` of the (only) mpirun call."""
    args = next(c[0] for c in calls if c[0][0] == "mpirun")
    return int(args[args.index("-np") + 1])


def run_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int, **kwargs: Any
) -> tuple[MDForwardModel, list[Any]]:
    calls: list[Any] = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")

    def hook(args: list[str], cwd: Path) -> None:
        (cwd / "prod.lammpstrj").write_text("traj")
        ranks = args[args.index("-np") + 1]
        (cwd / "log.lammps").write_text(
            f"Loop time of 1.0 on {ranks} procs for 10 steps with 3 atoms\n"
            "Total wall time: 0:00:10\n"
        )

    monkeypatch.setattr(
        pipeline.subprocess, "run", make_fake_run(calls, lammps_hook=hook)
    )
    model = MDForwardModel(tmp_path, **kwargs)
    model(n, THETA)
    return model, calls


def test_default_is_calc_cpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = run_with(tmp_path, monkeypatch, 1024)
    assert lammps_ranks(calls) == calc_cpus(1024) == 8


def test_mapping_override_with_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = run_with(tmp_path / "a", monkeypatch, 1024, ncpu={1024: 6})
    assert lammps_ranks(calls) == 6  # paper I
    _, calls = run_with(tmp_path / "b", monkeypatch, 256, ncpu={1024: 6})
    assert lammps_ranks(calls) == calc_cpus(256)


def test_callable_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = run_with(tmp_path, monkeypatch, 100, ncpu=lambda n: 3)
    assert lammps_ranks(calls) == 3


@pytest.mark.parametrize("bad", [0, -2, 1.5])
def test_invalid_override_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    monkeypatch.setattr(pipeline.subprocess, "run", make_fake_run([]))
    model = MDForwardModel(tmp_path, ncpu=lambda n: bad)
    with pytest.raises(ValueError, match="MPI ranks"):
        model(100, THETA)


def test_ncpu_is_not_part_of_the_key_but_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = run_with(tmp_path, monkeypatch, 100, ncpu=lambda n: 3)
    cached = CachedForwardModel(model, tmp_path)
    # directory is the plain key hash, independent of the rank count
    assert cached.entry_dir(100, THETA).name == key_hash(evaluation_key(100, THETA))
    model2 = MDForwardModel(tmp_path / "other", ncpu=lambda n: 5)
    assert CachedForwardModel(model2, tmp_path).entry_dir(100, THETA) == (
        cached.entry_dir(100, THETA)
    )
    cached(100, THETA)  # runs again (cache was empty), now through the wrapper
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    assert data["n_mpi"] == 3 and data["n_mpi_requested"] == 3
    assert data["cpu_time_s"] == 30.0


def test_recorded_ranks_follow_the_log_not_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command that ignores {ncpu} (e.g. srun) is caught by the log."""
    calls: list[Any] = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")

    def hook(args: list[str], cwd: Path) -> None:
        (cwd / "prod.lammpstrj").write_text("traj")
        (cwd / "log.lammps").write_text(
            "Loop time of 1.0 on 4 procs for 10 steps with 3 atoms\n"
            "Total wall time: 0:00:10\n"
        )

    monkeypatch.setattr(
        pipeline.subprocess, "run", make_fake_run(calls, lammps_hook=hook)
    )
    model = MDForwardModel(tmp_path, ncpu=lambda n: 6)
    CachedForwardModel(model, tmp_path)(100, THETA)
    cost = model.last_cost
    assert isinstance(cost, dict)
    assert (cost["n_mpi"], cost["n_mpi_requested"], cost["cpu_time_s"]) == (4, 6, 40.0)
