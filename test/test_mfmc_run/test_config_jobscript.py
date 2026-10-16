"""Tests of the configuration, sampling, result files and job scripts of ``mfmc-run``."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pytest

from mfwater.algo_mfda.jobscript import load_profile, render_jobscript
from mfwater.algo_mfmc_run.config import budget_label, build_config
from mfwater.algo_mfmc_run.results import write_hdf5_once, write_json_once
from mfwater.algo_mfmc_run.sampling import draw_samples, read_samples
from mfwater.argparser import constants, parse_args, parser
from mfwater.cli import console_entry_point

DATA = Path(__file__).parents[1] / "data"
CAL = DATA / "paper1" / "m6_e10_c_lj" / "m6_e10_c_lj.hdf5"

RUN_TOML = """\
forward_model = "md"
workdir = "./myrun"
molecules = [1024, 512, 256, 128, 64, 32]
evals = 10
budgets = [1e5, 1e6]
params = "lj"
total_cpus = 16
seed = 5
ncpu_override = {1024 = 6, 512 = 4, 256 = 2, 128 = 2, 64 = 1, 32 = 1}
replay_performance_from = {"1e5" = "%s"}
"""


def parse(*argv: str) -> argparse.Namespace:
    return parse_args(parser(), ["-a", "mfmc-run", *argv])


def write_toml(tmp: Path, text: str = RUN_TOML) -> Path:
    path = tmp / "run.toml"
    path.write_text(text % CAL if "%s" in text else text)
    return path


def test_toml_tables_and_config(tmp_path: Path) -> None:
    run = write_toml(tmp_path)
    cfg = build_config(parse("-c", str(run)))
    assert cfg.molecules == (1024, 512, 256, 128, 64, 32)
    assert cfg.ncpu_override == {1024: 6, 512: 4, 256: 2, 128: 2, 64: 1, 32: 1}
    assert cfg.budgets == (1e5, 1e6)
    assert cfg.replay_performance_from == {1e5: CAL}
    assert cfg.protocol == "paper1" and cfg.rmse_cost == "calibration"
    assert cfg.total_cpus == 16 and cfg.seed == 5


@pytest.mark.parametrize(
    "extra, match",
    [
        (["--molecules", "32", "64"], "descending"),
        (["--molecules", "64"], "two models"),
        (["--evals", "1"], "at least 2"),
        (["--budgets", "5", "5"], "distinct"),
        (["--budgets", "-5"], "positive"),
        (["--ncpu-override", "99=2"], "not in --molecules"),
        (["--ncpu-override", "bad"], "KEY=VALUE"),
        (["--total-cpus", "1", "--ncpu-override", "64=4"], "MPI ranks"),
        (["--replay-performance-from", f"7={CAL}"], "not in --budgets"),
        (["--replay-use-oms-allocation"], "needs"),
        (["--evals", "3", "4"], "single value"),
    ],
)
def test_invalid_settings(extra: list[str], match: str) -> None:
    base = [
        "--molecules",
        "64",
        "32",
        "--evals",
        "4",
        "--budgets",
        "10",
        "--total-cpus",
        "8",
    ]
    with pytest.raises(ValueError, match=match):
        build_config(parse(*base, *extra))


def test_missing_required() -> None:
    with pytest.raises(ValueError, match="--molecules"):
        build_config(parse("--budgets", "1"))
    with pytest.raises(ValueError, match="--budgets"):
        build_config(parse("--molecules", "64", "32", "--evals", "4"))


def test_molecules_and_evals_from_calibration_file() -> None:
    cfg = build_config(
        parse("--calibration-from", str(CAL), "--budgets", "1e5", "--total-cpus", "8")
    )
    assert cfg.molecules == (1024, 512, 256, 128, 64, 32) and cfg.evals == 10
    with pytest.raises(ValueError, match="differ"):
        build_config(
            parse(
                "--calibration-from",
                str(CAL),
                "--molecules",
                "64",
                "32",
                "--budgets",
                "1",
            )
        )


def test_budget_label() -> None:
    assert budget_label(1e5) == "100000" and budget_label(2.5) == "2.5"


def test_sampling_matches_chemmodel_prep_distribution() -> None:
    s = draw_samples(2000, "lj-q", np.random.default_rng([1, 0]))
    assert s.lj_params.dtype == np.float32 and s.seeds.dtype == np.int32
    assert s.seeds.min() >= 1 and s.seeds.max() < 100000
    eps0 = constants.OPC3_EPSILON_OO
    assert s.lj_params[0].mean() == pytest.approx(eps0, rel=1e-3)
    assert s.lj_params[0].std() / eps0 == pytest.approx(1 / 300, rel=0.1)
    assert s.charges[0].std() / abs(constants.OPC3_CHARGE_O) == pytest.approx(
        1 / 50, rel=0.1
    )
    assert np.allclose(s.charges[1], -0.5 * s.charges[0])
    again = draw_samples(2000, "lj-q", np.random.default_rng([1, 0]))
    assert np.array_equal(s.seeds, again.seeds)
    other = draw_samples(2000, "lj-q", np.random.default_rng([1, 1, 0]))
    assert not np.array_equal(s.seeds, other.seeds)
    # q-only: LJ stays at OPC3
    q = draw_samples(5, "q", np.random.default_rng(1))
    assert np.all(q.lj_params[0] == np.float32(eps0))
    theta = s.theta(3)
    assert theta.dtype == np.float64 and theta[2] == float(s.charges[0, 3])
    with pytest.raises(ValueError, match="available"):
        s.head(3000)


def test_read_samples_paper_and_old_format(tmp_path: Path) -> None:
    s = read_samples(CAL)
    with h5py.File(CAL) as f:
        assert np.array_equal(s.seeds, f["models/seeds"][:])
        # LAMMPS: model_1/eval_1 of m6_e10_c_lj used epsilon 0.162877, velocity seed 90203
        assert s.lj_params[0, 0] == pytest.approx(0.162877, abs=1e-6)
        assert s.seeds[1, 0] == 90203
    # old format without charges: OPC3 charges
    old = tmp_path / "old.hdf5"
    with h5py.File(old, "w") as f:
        g = f.create_group("models")
        g["lj_params"] = np.ones((2, 3), dtype=np.float32)
        g["seeds"] = np.ones((2, 3), dtype=np.int32)
    q = read_samples(old)
    assert np.all(q.charges[0] == np.float32(constants.OPC3_CHARGE_O))
    with h5py.File(tmp_path / "bad.hdf5", "w") as f:
        f.create_group("models")["seeds"] = np.ones((2, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="lj_params"):
        read_samples(tmp_path / "bad.hdf5")


def test_write_once(tmp_path: Path) -> None:
    path = tmp_path / "x.hdf5"

    def writer(value: float) -> object:
        def w(f: h5py.File) -> None:
            f.attrs["a"] = value
            f.create_dataset("d", data=np.array([1.0, np.nan]))

        return w

    assert write_hdf5_once(path, writer(1.0))  # type: ignore[arg-type]
    assert not write_hdf5_once(path, writer(1.0))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="different content"):
        write_hdf5_once(path, writer(2.0))  # type: ignore[arg-type]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.hdf5"]  # no temp files
    j = tmp_path / "s.json"
    assert write_json_once(j, {"a": [1, 2]})
    assert not write_json_once(j, {"a": [1, 2]})
    with pytest.raises(RuntimeError, match="different content"):
        write_json_once(j, {"a": [1, 3]})


# ----------------------------------------------------------------- job scripts


def test_mfda_jobscript_byte_identical() -> None:
    """The generalised generator reproduces the MFDA script of commit 063aada exactly."""
    profile = load_profile("marvin")
    for set_cmd, name in (
        (True, "jobscript_mfda_marvin.sh"),
        (False, "jobscript_mfda_marvin_no_lammps.sh"),
    ):
        text = render_jobscript(profile, 16, "myrun", Path("/abs/run.toml"), set_cmd)
        assert text == (DATA / "jobscript" / name).read_text()


def test_mfmc_jobscript_marvin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = write_toml(tmp_path)
    out = tmp_path / "run.sh"
    argv = [
        "-a",
        "mfmc-jobscript",
        "-c",
        str(run),
        "--cluster",
        "marvin",
        "--jobscript",
        str(out),
    ]
    assert console_entry_point(argv) == 0
    text = out.read_text()
    for line in (
        "#SBATCH --partition=intelsr_long",
        "#SBATCH --account=ag_mctc_kirchner",
        "#SBATCH --ntasks=16",
        "#SBATCH --nodes=1",
        "#SBATCH --job-name=myrun",
        "module load LAMMPS",
        f"-a mfmc-run -c {run.resolve()}",
        "--lammps-cmd 'mpirun --bind-to none -np {ncpu} lmp -i {input}'",
        "submit this script k times",
    ):
        assert line in text
    assert "@" not in text.replace("${", "")
    with pytest.raises(FileExistsError):
        console_entry_point(argv)


def test_mfmc_jobscript_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = write_toml(tmp_path)
    no_cpus = tmp_path / "nocpus.toml"
    no_cpus.write_text(
        "\n".join(l for l in run.read_text().splitlines() if "total_cpus" not in l)
    )
    with pytest.raises(ValueError, match="total_cpus"):
        console_entry_point(
            ["-a", "mfmc-jobscript", "-c", str(no_cpus), "--cluster", "marvin"]
        )
    big = tmp_path / "big.toml"  # the run TOML overrides the command line
    big.write_text(run.read_text().replace("total_cpus = 16", "total_cpus = 200"))
    with pytest.raises(ValueError, match="does not fit"):
        console_entry_point(
            ["-a", "mfmc-jobscript", "-c", str(big), "--cluster", "marvin"]
        )
    with pytest.raises(ValueError, match="-c"):
        console_entry_point(["-a", "mfmc-jobscript", "--cluster", "marvin"])
    assert not list(tmp_path.glob("run-mfmc-*.sh"))
