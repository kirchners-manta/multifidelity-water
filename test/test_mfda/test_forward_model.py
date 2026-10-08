"""Tests for the MFDA forward models and the persistent cache (no real MD)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mfwater.algo_mfda import cache as cache_mod
from mfwater.algo_mfda import md_pipeline
from mfwater.algo_mfda.cache import CachedForwardModel
from mfwater.algo_mfda.forward_model import (
    MAX_SEED,
    DummyForwardModel,
    derive_seeds,
    evaluation_key,
    key_hash,
)
from mfwater.algo_mfda.md_pipeline import MDForwardModel, parse_msdiff_output

THETA = np.array([0.1634, 3.17427, -0.8952])


# ---------------------------------------------------------------- key and seeds
def test_key_roundtrip_exact() -> None:
    key = evaluation_key(100, THETA)
    back = np.array([float.fromhex(h) for h in key[1:]])
    assert np.array_equal(back, THETA)
    assert key[0] == 100


def test_key_hash_stable() -> None:
    # fixed value: guards against accidental changes that would invalidate caches
    key = evaluation_key(100, THETA)
    assert key_hash(key) == key_hash(evaluation_key(100, THETA.copy()))
    assert len(key_hash(key)) == 16
    assert key_hash(key) != key_hash(evaluation_key(101, THETA))
    assert key_hash(key) != key_hash(evaluation_key(100, THETA + 1e-15))


def test_seed_range_and_determinism() -> None:
    for n in (10, 100, 1000):
        for shift in (0.0, 1e-3, 0.5):
            seeds = derive_seeds(evaluation_key(n, THETA + shift))
            assert all(1 <= s <= MAX_SEED for s in seeds)
            assert seeds == derive_seeds(evaluation_key(n, THETA + shift))


def test_invalid_theta() -> None:
    with pytest.raises(ValueError):
        evaluation_key(10, np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        evaluation_key(10, np.array([1.0, np.nan, 3.0]))


def test_dummy_deterministic() -> None:
    model = DummyForwardModel()
    assert model(100, THETA) == model(100, THETA)
    assert model(100, THETA) != model(100, THETA + 1e-3)
    assert model(100, THETA) != model(200, THETA)


# ------------------------------------------------------------------------ cache
class Counting:
    """Forward model counting its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, n_molecules: int, theta: np.ndarray) -> float:
        self.calls += 1
        return 1.5 + 0.001 * n_molecules


def test_cache_hit_memo_and_disk(tmp_path: Path) -> None:
    model = Counting()
    cached = CachedForwardModel(model, tmp_path / "cache")
    d1 = cached(100, THETA)
    d2 = cached(100, THETA)
    assert d1 == d2 and model.calls == 1
    assert (cached.n_hits, cached.n_computed) == (1, 1)

    # new wrapper, same directory: result comes from disk
    model2 = Counting()
    cached2 = CachedForwardModel(model2, tmp_path / "cache")
    assert cached2(100, THETA) == d1
    assert model2.calls == 0
    assert len(cached2) == 1


def test_result_json_content(tmp_path: Path) -> None:
    cached = CachedForwardModel(Counting(), tmp_path)
    cached(100, THETA)
    entry = cached.entry_dir(100, THETA)
    assert entry.parent.name == "N100"
    data = json.loads((entry / "result.json").read_text())
    assert data["D"] == pytest.approx(1.6)
    assert data["unit"] == "1e-9 m^2/s"
    assert data["N"] == 100
    assert data["theta"]["sigma"] == THETA[1]
    assert float.fromhex(data["theta_hex"]["q_O"]) == THETA[2]
    assert data["key_hash"] == entry.name
    assert 1 <= data["seeds"]["packmol"] <= MAX_SEED
    assert data["wall_time_s"] >= 0
    assert not list(entry.glob(".result_*"))  # temp file removed


def test_atomic_write_failure_leaves_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = CachedForwardModel(Counting(), tmp_path)

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cache_mod.os, "replace", boom)
    with pytest.raises(OSError):
        cached(100, THETA)
    entry = cached.entry_dir(100, THETA)
    assert not (entry / "result.json").exists()
    assert not list(entry.glob(".result_*"))


def test_nonfinite_result_not_cached(tmp_path: Path) -> None:
    cached = CachedForwardModel(lambda n, th: float("nan"), tmp_path)
    with pytest.raises(RuntimeError):
        cached(100, THETA)
    assert len(cached) == 0


# ----------------------------------------------------------- fake MD toolchain
MSDIFF_CSV = (
    " Species, D_0 / 10^-12 m^2/s,        delta_D, K / 10^-12 m^2/s\n"
    "       1,        2500.000000,       0.000100,         0.000000\n"
)


def make_fake_run(
    calls: list[Any],
    fail_on: str | None = None,
    travis_files: tuple[str, ...] = ("msd_H2O_#2.csv", "msd_H2O_#2_fit.csv"),
) -> Any:
    """Return a replacement for ``subprocess.run`` that fakes all programs."""

    def fake_run(
        args: list[str],
        cwd: Any = None,
        check: bool = False,
        stdin: Any = None,
        stdout: Any = None,
        stderr: Any = None,
    ) -> None:
        cwd = Path(cwd)
        prog = args[0]
        calls.append((list(args), cwd))
        if prog == fail_on:
            raise subprocess.CalledProcessError(3, args)
        if prog == "fftool":
            if "--lmp" in args:
                (cwd / "data.lmp").write_text("data")
            else:
                (cwd / "pack.inp").write_text(
                    "tolerance 2.0\ninside box 0.0 0.0 0.0 10.0 10.0 10.0\n"
                )
        elif prog == "mpirun":
            (cwd / "prod.lammpstrj").write_text("traj")
            (cwd / "log.lammps").write_text("log")
        elif prog == "travis":
            for name in travis_files:
                (cwd / name).write_text("msd")
        elif prog == "msdiff":
            (cwd / "msdiff_out.csv").write_text(MSDIFF_CSV)

    return fake_run


@pytest.fixture
def fake_md(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    monkeypatch.setattr(md_pipeline.subprocess, "run", make_fake_run(calls))
    return calls


def test_md_pipeline_layout_and_units(tmp_path: Path, fake_md: list[Any]) -> None:
    model = MDForwardModel(tmp_path)
    d = model(100, THETA)
    assert d == pytest.approx(2.5)  # 2500e-12 -> 2.5e-9 m^2/s

    attempt = model.last_attempt_dir
    assert attempt is not None and attempt.name == "attempt_1"
    assert attempt.parent == tmp_path / "N100" / key_hash(evaluation_key(100, THETA))
    for sub in ("siminp", "simout", "msd"):
        assert (attempt / sub).is_dir()
    assert (attempt / "msd" / "travis_input_msd.txt").exists()
    assert (attempt / "msd" / "msdiff_out.csv").exists()

    progs = [c[0][0] for c in fake_md]
    assert progs == ["fftool", "packmol", "fftool", "mpirun", "travis", "msdiff"]
    lammps_args, lammps_cwd = fake_md[3]
    assert lammps_cwd == attempt / "simout"
    assert lammps_args[:3] == ["mpirun", "-np", "2"]
    assert lammps_args[-1] == "../siminp/input.lmp"
    msdiff_args, _ = fake_md[5]
    # cubic box of 100 molecules: edge ~14.4 A = ~1440 pm
    assert msdiff_args[:3] == ["msdiff", "-f", "msd_H2O_#2.csv"]
    assert 1400 < float(msdiff_args[4]) < 1500
    assert "seed" in (attempt / "siminp" / "pack.inp").read_text()


def test_input_lmp_parameters(tmp_path: Path, fake_md: list[Any]) -> None:
    model = MDForwardModel(tmp_path)
    model(100, THETA)
    assert model.last_attempt_dir is not None
    text = (model.last_attempt_dir / "siminp" / "input.lmp").read_text()
    assert f"pair_coeff    2    2     {THETA[0]:.6f}     {THETA[1]:.6f}" in text
    assert f"set type 2 charge {THETA[2]:11.8f}" in text
    assert f"set type 1 charge {-0.5 * THETA[2]:11.8f}" in text
    assert (
        f"velocity all create ${{vTK}} {derive_seeds(evaluation_key(100, THETA))[1]}"
        in text
    )
    assert "VAR_" not in text
    assert "read_data ../siminp/data.lmp" in text


def test_interrupted_attempt_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    calls: list[Any] = []
    model = MDForwardModel(tmp_path)

    # first attempt: LAMMPS fails, leaving files behind
    monkeypatch.setattr(
        md_pipeline.subprocess, "run", make_fake_run(calls, fail_on="mpirun")
    )
    with pytest.raises(RuntimeError, match="mpirun"):
        model(100, THETA)
    first = model.last_attempt_dir
    assert first is not None
    marker = first / "siminp" / "marker.txt"
    marker.write_text("keep me")
    snapshot = sorted(p.name for p in (first / "siminp").iterdir())

    # second attempt works and uses a new directory
    monkeypatch.setattr(md_pipeline.subprocess, "run", make_fake_run(calls))
    d = model(100, THETA)
    assert d == pytest.approx(2.5)
    assert model.last_attempt_dir == first.parent / "attempt_2"
    assert marker.read_text() == "keep me"
    assert sorted(p.name for p in (first / "siminp").iterdir()) == snapshot
    assert not (first / "msd" / "msdiff_out.csv").exists()


def test_cache_with_md_records_attempt(tmp_path: Path, fake_md: list[Any]) -> None:
    cached = CachedForwardModel(MDForwardModel(tmp_path), tmp_path)
    d = cached(100, THETA)
    assert cached(100, THETA) == d
    assert sum(c[0][0] == "mpirun" for c in fake_md) == 1
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    assert data["attempt_dir"].endswith("attempt_1")


@pytest.mark.parametrize("missing", ["fftool", "packmol", "mpirun", "travis", "msdiff"])
def test_missing_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(
        md_pipeline.shutil, "which", lambda p: None if p == missing else f"/bin/{p}"
    )
    with pytest.raises(RuntimeError, match=missing):
        MDForwardModel(tmp_path)(100, THETA)
    assert not any(tmp_path.iterdir())  # nothing created before the check


DATA_DIR = Path(__file__).parent / "data"


def test_parse_msdiff_old_format() -> None:
    # real output of the older msdiff used in paper I; K (889) must not be added
    d = parse_msdiff_output(DATA_DIR / "msdiff_out_old.csv")
    assert d == pytest.approx(1.20928281695)


def test_parse_msdiff_new_format(tmp_path: Path) -> None:
    f = tmp_path / "msdiff_out.csv"
    f.write_text(MSDIFF_CSV)
    assert parse_msdiff_output(f) == pytest.approx(2.5)


def test_stray_fit_file_ignored(tmp_path: Path) -> None:
    (tmp_path / "msd_H2O_#2.csv").write_text("msd")
    (tmp_path / "msd_H2O_#2_fit.csv").write_text("fit")
    assert md_pipeline.find_travis_msd_file(tmp_path).name == "msd_H2O_#2.csv"
    assert (tmp_path / "msd_H2O_#2_fit.csv").read_text() == "fit"


def test_travis_serial_and_log(tmp_path: Path, fake_md: list[Any]) -> None:
    MDForwardModel(tmp_path)(100, THETA)
    travis = next(c for c in fake_md if c[0][0] == "travis")
    assert travis[0][0] == "travis"  # no mpirun wrapper
    assert (travis[1] / "travis.log").exists()


def test_missing_travis_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/bin/{p}")
    monkeypatch.setattr(
        md_pipeline.subprocess, "run", make_fake_run([], travis_files=())
    )
    with pytest.raises(RuntimeError, match="msd_H2O"):
        MDForwardModel(tmp_path)(100, THETA)


def test_custom_lammps_cmd_and_msdiff_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/bin/{p}")
    monkeypatch.setattr(md_pipeline.subprocess, "run", make_fake_run(calls))
    model = MDForwardModel(
        tmp_path,
        lammps_cmd="mpirun --bind-to none -np {ncpu} lmp -i {input}",
        msdiff_args=["--hummer", "298.15", "0.00089"],
    )
    model(1000, THETA)
    lammps = next(c[0] for c in calls if c[0][0] == "mpirun")
    assert lammps[:5] == ["mpirun", "--bind-to", "none", "-np", "6"]
    msdiff = next(c[0] for c in calls if c[0][0] == "msdiff")
    assert msdiff[-3:] == ["--hummer", "298.15", "0.00089"]


def test_orthoboxy_not_supported(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        MDForwardModel(tmp_path, orthoboxy=True)


def test_parse_msdiff_rejects_unknown_unit(tmp_path: Path) -> None:
    f = tmp_path / "msdiff_out.csv"
    f.write_text("Species, D_0 / 10^-9 m^2/s\n1, 2.5\n")
    with pytest.raises(RuntimeError, match="header"):
        parse_msdiff_output(f)
    f.write_text("Species, D_0 / 10^-12 m^2/s\n")
    with pytest.raises(RuntimeError, match="no data"):
        parse_msdiff_output(f)
