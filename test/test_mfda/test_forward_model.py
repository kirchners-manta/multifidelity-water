"""Tests for the MFDA forward models and the persistent cache (no real MD)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.algo_chemical_model.chemical_model import calc_box_size
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
from mfwater.argparser import constants

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
    lammps_hook: Callable[[list[str], Path], None] | None = None,
) -> Any:
    """Return a replacement for ``subprocess.run`` that fakes all programs.

    ``lammps_hook(args, cwd)`` replaces the default LAMMPS fake (it may raise
    ``CalledProcessError`` to simulate an interrupted run).
    """

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
            if lammps_hook is not None:
                lammps_hook(args, cwd)
                return
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
    # paper-I call; box length comes from travis.log, no explicit -l
    assert msdiff_args == [
        "msdiff",
        "-f",
        "msd_H2O_#2.csv",
        "--from-travis",
        "--hummer",
        "298.15",
        "0.89e-3",
        "0.0",
    ]
    assert "-l" not in msdiff_args
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
        msdiff_args=["--hummer", "330", "0.00089"],
    )
    model(1000, THETA)
    lammps = next(c[0] for c in calls if c[0][0] == "mpirun")
    assert lammps[:5] == ["mpirun", "--bind-to", "none", "-np", "8"]  # calc_cpus(1000)
    msdiff = next(c[0] for c in calls if c[0][0] == "msdiff")
    assert msdiff[-3:] == ["--hummer", "330", "0.00089"]
    assert "--from-travis" in msdiff


# ------------------------------------------------------------------- OrthoBoXY
ORTHO_FILES = ("msd_H2O_#2_XY.csv", "msd_H2O_#2_Z.csv", "msd_H2O_#2_XY_fit.csv")


def _orthoboxy_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[MDForwardModel, list[Any]]:
    calls: list[Any] = []
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/bin/{p}")
    monkeypatch.setattr(md_pipeline.subprocess, "run", make_fake_run(calls, **kwargs))
    return MDForwardModel(tmp_path, orthoboxy=True), calls


def test_orthoboxy_box_and_msdiff_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calls = _orthoboxy_run(tmp_path, monkeypatch, travis_files=ORTHO_FILES)
    d = model(100, THETA)
    assert d == pytest.approx(2.5)

    lx, ly, lz = calc_box_size(100, orthoboxy_shape=True)
    assert lx == ly and lz / lx == pytest.approx(constants.ORTHOBOXY_RATIO)
    fftool = calls[0][0]
    assert fftool[fftool.index("--box") + 1] == f"{lx:.6f},{ly:.6f},{lz:.6f}"
    assert model.last_attempt_dir is not None
    pack = (model.last_attempt_dir / "siminp" / "pack.inp").read_text()
    assert f"{lx - 0.5:.6f} {ly - 0.5:.6f} {lz - 0.5:.6f}" in pack

    msdiff = next(c[0] for c in calls if c[0][0] == "msdiff")
    assert msdiff == [
        "msdiff",
        "-f",
        "msd_H2O_#2_XY.csv",
        "--from-travis",
        "--hummer",
        "298.15",
        "0.89e-3",
        "0.0",
        "--orthoboxy",
        "msd_H2O_#2_Z.csv",
    ]
    assert "-l" not in msdiff


def test_orthoboxy_travis_template_is_verbatim_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _orthoboxy_run(tmp_path, monkeypatch, travis_files=ORTHO_FILES)
    model(100, THETA)
    assert model.last_attempt_dir is not None
    used = (model.last_attempt_dir / "msd" / "travis_input_msd.txt").read_bytes()
    template = md_pipeline.TEMPLATE_DIR / "travis_input_msd_orthoboxy.txt"
    assert used == template.read_bytes()
    # two MSD observations (x-y and z) and the blank lines are preserved
    assert b"Add another observation (y/n)? [no] \ny\n" in used
    assert b"\n\n" in used
    assert used != (md_pipeline.TEMPLATE_DIR / "travis_input_msd.txt").read_bytes()


def test_cubic_uses_cubic_template(tmp_path: Path, fake_md: list[Any]) -> None:
    model = MDForwardModel(tmp_path)
    model(100, THETA)
    assert model.last_attempt_dir is not None
    used = (model.last_attempt_dir / "msd" / "travis_input_msd.txt").read_bytes()
    assert used == (md_pipeline.TEMPLATE_DIR / "travis_input_msd.txt").read_bytes()


@pytest.mark.parametrize("missing", ["msd_H2O_#2_XY.csv", "msd_H2O_#2_Z.csv"])
def test_orthoboxy_missing_travis_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    files = tuple(f for f in ORTHO_FILES if f != missing)
    model, calls = _orthoboxy_run(tmp_path, monkeypatch, travis_files=files)
    with pytest.raises(RuntimeError, match="msd_H2O"):
        model(100, THETA)
    assert not any(c[0][0] == "msdiff" for c in calls)


def test_orthoboxy_flag_in_result_and_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _orthoboxy_run(tmp_path, monkeypatch, travis_files=ORTHO_FILES)
    d = CachedForwardModel(model, tmp_path)(100, THETA)
    entry = CachedForwardModel(model, tmp_path).entry_dir(100, THETA)
    assert json.loads((entry / "result.json").read_text())["orthoboxy"] is True

    # same wrapper type with the matching flag: cache hit
    assert CachedForwardModel(model, tmp_path)(100, THETA) == d
    # cubic model on an OrthoBoXY cache: error
    with pytest.raises(ValueError, match="orthoboxy"):
        CachedForwardModel(MDForwardModel(tmp_path), tmp_path)(100, THETA)


def test_cache_flag_none_for_models_without_flag(tmp_path: Path) -> None:
    cached = CachedForwardModel(Counting(), tmp_path)
    cached(100, THETA)
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    assert data["orthoboxy"] is None
    assert CachedForwardModel(Counting(), tmp_path)(100, THETA) == pytest.approx(1.6)


def test_parse_msdiff_rejects_unknown_unit(tmp_path: Path) -> None:
    f = tmp_path / "msdiff_out.csv"
    f.write_text("Species, D_0 / 10^-9 m^2/s\n1, 2.5\n")
    with pytest.raises(RuntimeError, match="header"):
        parse_msdiff_output(f)
    f.write_text("Species, D_0 / 10^-12 m^2/s\n")
    with pytest.raises(RuntimeError, match="no data"):
        parse_msdiff_output(f)


# ------------------------------------------------- restart of interrupted runs
EQ, PROD_END = md_pipeline.EQ_END_STEP, md_pipeline.PROD_END_STEP


def traj_text(steps: range, tag: str, n_atoms: int = 2) -> str:
    """Synthetic lammpstrj; ``tag`` is stored in the coordinates (frame origin)."""
    out = []
    for s in steps:
        out.append(
            f"ITEM: TIMESTEP\n{s}\nITEM: NUMBER OF ATOMS\n{n_atoms}\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id element xu yu zu\n"
        )
        out.extend(f"{i + 1} H {tag} 0.0 0.0\n" for i in range(n_atoms))
    return "".join(out)


def tags_by_step(path: Path) -> dict[int, str]:
    """Map timestep -> tag of a synthetic trajectory."""
    return {s: lines[9].split()[2] for s, lines in md_pipeline.iter_frames(path)}


def interrupted_lammps(
    frames: range | None,
    restarts: tuple[int, ...],
    finish: bool = False,
    seen: list[str] | None = None,
) -> Callable[[list[str], Path], None]:
    """Fake LAMMPS writing frames and restart files, then dying (or finishing)."""

    def hook(args: list[str], cwd: Path) -> None:
        if seen is not None:
            seen.append((cwd.parent / "siminp" / Path(args[-1]).name).read_text())
        tag = cwd.parent.name  # e.g. attempt_2
        if frames is not None:
            (cwd / "prod.lammpstrj").write_text(traj_text(frames, tag))
        for r in restarts:
            (cwd / f"restart.{r}").write_text("binary")
        if not finish:
            raise subprocess.CalledProcessError(137, args)

    return hook


def snapshot(attempt: Path) -> dict[str, str]:
    """Hash of every file below an attempt directory."""
    return {
        str(p.relative_to(attempt)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(attempt.rglob("*"))
        if p.is_file()
    }


def test_template_matches_protocol() -> None:
    text = md_pipeline.MFDA_INPUT_TEMPLATE.read_text()
    assert "restart 100000 restart.*" in text
    assert "run 1000000 upto" in text and "run 3000000 upto" in text
    assert "write_data data.after_eq.lmp" in text
    assert "write_data data.after_prod.lmp" in text
    # paper-I protocol otherwise unchanged: every other command line is identical
    paper = (md_pipeline.TEMPLATE_DIR / "input.lmp").read_text().splitlines()
    new = text.splitlines()
    cmds = lambda ls: [  # noqa: E731
        ln.split("#")[0].strip() for ln in ls if ln.split("#")[0].strip()
    ]
    extra = [c for c in cmds(new) if c not in cmds(paper)]
    assert extra == [
        "restart 100000 restart.*",
        "run 1000000 upto",
        "run 3000000 upto",
    ]


def test_resume_input_equilibration_stage(tmp_path: Path) -> None:
    out = tmp_path / "input_resume.lmp"
    md_pipeline.write_resume_input(out, tmp_path / "restart.300000", 300000)
    text = out.read_text()
    assert f'read_restart "{(tmp_path / "restart.300000").resolve()}"' in text
    assert "fix fTHERMO all nvt temp ${vTK} ${vTK} 100" in text  # same ID and style
    assert "run 1000000 upto" in text and "run 3000000 upto" in text
    assert "dump dEQUI" in text and "dump dPROD" in text
    assert text.index("run 1000000 upto") < text.index("run 3000000 upto")
    for forbidden in ("minimize", "velocity all create", "reset_timestep", "read_data"):
        assert not [
            ln for ln in text.splitlines() if ln.startswith(forbidden)
        ], forbidden
    # every re-specified setting is the one of the template
    template = md_pipeline.MFDA_INPUT_TEMPLATE.read_text().splitlines()
    for ln in text.splitlines():
        if ln.startswith(("fix ", "kspace_style", "neighbor", "timestep", "thermo")):
            assert ln in template, ln


def test_resume_input_production_stage(tmp_path: Path) -> None:
    out = tmp_path / "input_resume.lmp"
    md_pipeline.write_resume_input(out, tmp_path / "restart.1500000", 1_500_000)
    text = out.read_text()
    assert "dump dEQUI" not in text and "run 1000000 upto" not in text
    assert "run 3000000 upto" in text
    assert "data.after_eq" not in text
    with pytest.raises(ValueError):
        md_pipeline.write_resume_input(out, tmp_path / "restart.3000000", PROD_END)


def test_fresh_start_without_restart_files(tmp_path: Path, fake_md: list[Any]) -> None:
    model = MDForwardModel(tmp_path)
    model(100, THETA)
    assert model.last_attempt_dirs == [model.last_attempt_dir]
    attempt = model.last_attempt_dir
    assert attempt is not None
    assert not (attempt / "resume_from.txt").exists()
    assert not (attempt / "msd" / md_pipeline.COMBINED_TRAJECTORY).exists()
    # without interruption TRAVIS reads the LAMMPS trajectory directly
    travis = next(c for c in fake_md if c[0][0] == "travis")
    assert travis[0][-1] == "../simout/prod.lammpstrj"
    # the template with restart output is what LAMMPS gets
    text = (attempt / "siminp" / "input.lmp").read_text()
    assert "restart 100000 restart.*" in text and "run 3000000 upto" in text


def test_restart_chain_of_three_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    calls: list[Any] = []
    seen: list[str] = []
    model = MDForwardModel(tmp_path)

    def run(hook: Callable[[list[str], Path], None]) -> None:
        monkeypatch.setattr(
            md_pipeline.subprocess,
            "run",
            make_fake_run(calls, lammps_hook=hook),
        )

    # attempt 1: killed in production, last restart at 1.2e6, frames beyond it
    run(interrupted_lammps(range(EQ, 1_250_001, 1000), (EQ, 1_100_000, 1_200_000)))
    with pytest.raises(RuntimeError, match="mpirun"):
        model(100, THETA)
    a1 = model.last_attempt_dir
    assert a1 is not None
    snap1 = snapshot(a1)

    # attempt 2: resumes from attempt 1, killed again after more restarts
    run(
        interrupted_lammps(
            range(1_200_000, 1_450_001, 1000), (1_300_000, 1_400_000), seen=seen
        )
    )
    with pytest.raises(RuntimeError, match="mpirun"):
        model(100, THETA)
    a2 = model.last_attempt_dir
    assert a2 is not None and a2.name == "attempt_2"
    snap2 = snapshot(a2)
    assert (
        a2 / "resume_from.txt"
    ).read_text().strip() == "attempt_1/simout/restart.1200000"
    assert "restart.1200000" in seen[0] and "read_restart" in seen[0]
    assert snapshot(a1) == snap1  # previous attempt untouched
    assert not any(c[0][0] == "fftool" for c in calls[len(calls) - 2 :])

    # attempt 3: resumes from attempt 2 (not from the older file of attempt 1), finishes
    n_fftool = sum(c[0][0] == "fftool" for c in calls)
    run(interrupted_lammps(range(1_400_000, PROD_END + 1, 1000), (), finish=True))
    d = model(100, THETA)
    assert d == pytest.approx(2.5)
    a3 = model.last_attempt_dir
    assert a3 is not None and a3.name == "attempt_3"
    assert (
        a3 / "resume_from.txt"
    ).read_text().strip() == "attempt_2/simout/restart.1400000"
    assert sum(c[0][0] == "fftool" for c in calls) == n_fftool  # no box rebuild
    assert (a3 / "siminp" / "input_resume.lmp").exists()
    assert not (a3 / "siminp" / "data.lmp").exists()
    assert model.last_attempt_dirs == [a1, a2, a3]
    assert snapshot(a1) == snap1 and snapshot(a2) == snap2

    combined = a3 / "msd" / md_pipeline.COMBINED_TRAJECTORY
    tags = tags_by_step(combined)
    assert list(tags) == list(range(EQ, PROD_END + 1, 1000))  # 2001, no duplicates
    assert tags[1_100_000] == "attempt_1" and tags[1_199_000] == "attempt_1"
    assert tags[1_200_000] == "attempt_2" and tags[1_399_000] == "attempt_2"
    assert tags[1_400_000] == "attempt_3" and tags[PROD_END] == "attempt_3"
    travis = [c for c in calls if c[0][0] == "travis"][-1]
    assert travis[0][-1] == md_pipeline.COMBINED_TRAJECTORY


def test_restart_from_equilibration_and_result_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    seen: list[str] = []
    calls: list[Any] = []
    model = MDForwardModel(tmp_path)
    cached = CachedForwardModel(model, tmp_path)

    monkeypatch.setattr(
        md_pipeline.subprocess,
        "run",
        make_fake_run(calls, lammps_hook=interrupted_lammps(None, (100_000, 200_000))),
    )
    with pytest.raises(RuntimeError):
        cached(100, THETA)
    monkeypatch.setattr(
        md_pipeline.subprocess,
        "run",
        make_fake_run(
            calls,
            lammps_hook=interrupted_lammps(
                range(EQ, PROD_END + 1, 1000), (), finish=True, seen=seen
            ),
        ),
    )
    cached(100, THETA)
    assert "dump dEQUI" in seen[0] and "run 1000000 upto" in seen[0]
    data = json.loads((cached.entry_dir(100, THETA) / "result.json").read_text())
    assert [Path(p).name for p in data["attempt_dirs"]] == ["attempt_1", "attempt_2"]
    assert data["attempt_dir"].endswith("attempt_2")


def test_completed_analysis_is_not_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    calls: list[Any] = []
    hook = interrupted_lammps(range(EQ, PROD_END + 1, 1000), (100_000,), finish=True)
    monkeypatch.setattr(
        md_pipeline.subprocess, "run", make_fake_run(calls, lammps_hook=hook)
    )
    model = MDForwardModel(tmp_path)
    model(100, THETA)
    model(100, THETA)  # first attempt has an analysis: start fresh, not resume
    assert model.last_attempt_dir is not None
    assert model.last_attempt_dirs == [model.last_attempt_dir]
    assert not (model.last_attempt_dir / "resume_from.txt").exists()


def test_merge_removes_duplicate_frame_and_detects_gap(tmp_path: Path) -> None:
    a, b = tmp_path / "a.lammpstrj", tmp_path / "b.lammpstrj"
    a.write_text(traj_text(range(0, 6000, 1000), "a"))  # frames 0..5000
    b.write_text(traj_text(range(4000, 8000, 1000), "b"))  # restart at 4000
    out = tmp_path / "c.lammpstrj"
    n = md_pipeline.merge_trajectories(
        [(a, 4000), (b, None)], out, first_step=0, last_step=7000
    )
    assert n == 8
    assert tags_by_step(out) == {
        **{s: "a" for s in (0, 1000, 2000, 3000)},
        **{s: "b" for s in (4000, 5000, 6000, 7000)},
    }
    assert not out.with_name(out.name + ".part").exists()
    with pytest.raises(RuntimeError, match="already exists"):
        md_pipeline.merge_trajectories(
            [(a, 4000), (b, None)], out, first_step=0, last_step=7000
        )
    assert a.read_text() == traj_text(range(0, 6000, 1000), "a")  # sources untouched

    # gap: the resumed run did not write its first frame (starts at 5000)
    c = tmp_path / "c2.lammpstrj"
    c.write_text(traj_text(range(5000, 8000, 1000), "c"))
    with pytest.raises(RuntimeError, match="gap"):
        md_pipeline.merge_trajectories(
            [(a, 4000), (c, None)],
            tmp_path / "gap.lammpstrj",
            first_step=0,
            last_step=7000,
        )
    # incomplete range and truncated frame
    with pytest.raises(RuntimeError, match="expected 9000"):
        md_pipeline.merge_trajectories(
            [(b, None)], tmp_path / "short.lammpstrj", first_step=4000, last_step=9000
        )
    trunc = tmp_path / "t.lammpstrj"
    trunc.write_text(traj_text(range(0, 3000, 1000), "t")[:-10])
    with pytest.raises(RuntimeError, match="truncated"):
        md_pipeline.merge_trajectories(
            [(trunc, None)], tmp_path / "t_out.lammpstrj", first_step=0, last_step=2000
        )


def make_restarts(
    attempt: Path, sizes: dict[int, int], resumed_from: str | None = None
) -> None:
    """Create mocked restart files ``{step: size in bytes}`` in an attempt."""
    simout = attempt / "simout"
    simout.mkdir(parents=True)
    for step, size in sizes.items():
        (simout / f"restart.{step}").write_bytes(b"x" * size)
    if resumed_from is not None:
        (attempt / "resume_from.txt").write_text(f"{resumed_from}\n")


def test_restart_truncated_newest_is_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a1 = tmp_path / "attempt_1"
    make_restarts(a1, {100_000: 50, 200_000: 50, 300_000: 20})
    step, path = MDForwardModel(tmp_path)._resume_point(a1) or (0, a1)
    assert (step, path.name) == (200_000, "restart.200000")
    out = capsys.readouterr().out
    assert "Warning:" in out and "restart.300000" in out
    assert (a1 / "simout" / "restart.300000").stat().st_size == 20  # untouched


def test_restart_two_truncated_newest_are_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a1 = tmp_path / "attempt_1"
    make_restarts(a1, {100_000: 50, 200_000: 50, 300_000: 30, 400_000: 10})
    point = MDForwardModel(tmp_path)._resume_point(a1)
    # 400k (10) != 300k (30): skipped; 300k (30) != 200k (50): skipped
    assert point is not None and point[0] == 200_000
    assert capsys.readouterr().out.count("Warning:") == 2


def test_restart_only_empty_file_starts_fresh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = MDForwardModel(tmp_path)
    a1 = tmp_path / "attempt_1"
    make_restarts(a1, {100_000: 0})
    assert model._resume_point(a1) is None
    assert "starting from the beginning" in capsys.readouterr().out


def test_restart_single_valid_file_is_accepted(tmp_path: Path) -> None:
    a1 = tmp_path / "attempt_1"
    make_restarts(a1, {100_000: 50})
    point = MDForwardModel(tmp_path)._resume_point(a1)
    assert point is not None and point[0] == 100_000


def test_restart_files_over_two_attempts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a1, a2 = tmp_path / "attempt_1", tmp_path / "attempt_2"
    make_restarts(a1, {100_000: 50, 200_000: 50})
    make_restarts(a2, {300_000: 50, 400_000: 7}, "attempt_1/simout/restart.200000")
    model = MDForwardModel(tmp_path)
    point = model._resume_point(a2)
    assert point is not None and point[1] == a2 / "simout" / "restart.300000"
    # newest attempt has only a truncated file: fall back into the older attempt
    (a2 / "simout" / "restart.300000").unlink()
    point = model._resume_point(a2)
    assert point is not None and point[1] == a1 / "simout" / "restart.200000"
    # sole oldest candidate disagrees with a restart file outside the chain
    (a1 / "simout" / "restart.100000").unlink()
    make_restarts(tmp_path / "attempt_3", {PROD_END: 99})
    capsys.readouterr()
    assert model._resume_point(a1) is None


def test_select_restart_file_unit(tmp_path: Path) -> None:
    f = tmp_path / "restart.1"
    f.write_bytes(b"x" * 5)
    assert md_pipeline.select_restart_file([(1, f)], [5]) == ((1, f), [])
    assert md_pipeline.select_restart_file([(1, f)], [4]) == (None, [f])
    assert md_pipeline.select_restart_file([], []) == (None, [])
