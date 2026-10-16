"""End-to-end tests of ``mfmc-run`` with the correlated dummy forward model."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from mfwater import mfmc_core
from mfwater.algo_mfmc_run import run as run_mod
from mfwater.algo_mfmc_run.dummy import CorrelatedDummyForwardModel
from mfwater.cli import console_entry_point
from mfwater.md.pool import PoolResult, PoolStatus

FIXTURES = Path(__file__).parents[1] / "data" / "paper1" / "m6_e10_c_lj"
CAL = FIXTURES / "m6_e10_c_lj.hdf5"

#: one MPI rank per evaluation keeps the dummy runs fast
ONE_RANK = ["64=1", "32=1", "16=1", "8=1"]


def argv(
    workdir: Path, *extra: str, budgets: tuple[str, ...] = ("6", "20")
) -> list[str]:
    """Return the command line of a small dummy run in ``workdir``."""
    return [
        "-a",
        "mfmc-run",
        "--workdir",
        str(workdir),
        "--molecules",
        "64",
        "32",
        "16",
        "8",
        "--evals",
        "8",
        "--budgets",
        *budgets,
        "--total-cpus",
        "16",
        "--ncpu-override",
        *ONE_RANK,
        "--seed",
        "3",
        "--params",
        "lj-q",
        *extra,
    ]


def test_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wd = tmp_path / "run"
    assert console_entry_point(argv(wd)) == 0
    out = capsys.readouterr().out
    assert "0 of 8 evaluations without complete cost, excluded from w" in out
    for name in ("manifest.json", "calibration.hdf5", "selection.json", "summary.hdf5"):
        assert (wd / name).exists()
    assert (wd / "performance_6.hdf5").exists()
    assert (wd / "performance_20.hdf5").exists()

    manifest = json.loads((wd / "manifest.json").read_text())
    assert manifest["seed"] == 3 and manifest["algorithm"] == "mfmc-run"
    assert "msdiff_version" in manifest

    with h5py.File(wd / "calibration.hdf5") as f:
        m = f["models"]
        assert m.attrs["n_models"] == 4
        assert m["lj_params"].shape == (2, 8) and m["seeds"].shape == (2, 8)
        assert [m[f"model_{k}"].attrs["n_molecules"] for k in (1, 2, 3, 4)] == [
            64,
            32,
            16,
            8,
        ]
        d = np.array([m[f"model_{k}/diffusion_coeff"][:] for k in (1, 2, 3, 4)])
        mean, std, rho = mfmc_core.model_statistics(d)
        assert m["model_3"].attrs["correlation"] == pytest.approx(rho[2])
        assert m["model_1"].attrs["std"] == pytest.approx(std[0])
        cost = m["model_1"].attrs["computation_time"]
        assert cost == pytest.approx(0.01 * 64**1.1)
        assert d.mean() > 1000  # 1e-12 m^2/s, the paper-I unit

    sel = json.loads((wd / "selection.json").read_text())
    with h5py.File(wd / "performance_20.hdf5") as f:
        m = f["models"]
        n = int(m.attrs["n_models"])
        assert n == len(sel["selected_index"])
        assert list(m.attrs["selected_models"]) == sel["selected_models"]
        ds = [m[f"model_{k}/diffusion_coeff"][:] for k in range(1, n + 1)]
        evals = [int(m[f"model_{k}"].attrs["n_evals"]) for k in range(1, n + 1)]
        alpha = [float(m[f"model_{k}"].attrs["alpha"]) for k in range(1, n + 1)]
        assert [len(x) for x in ds] == evals
        assert m.attrs["mfmc_estimator"] == pytest.approx(
            mfmc_core.mfmc_estimate(ds, evals, alpha)
        )
        # default basis: calibration costs
        assert m.attrs["rmse_cost"] == "calibration"
        assert m.attrs["mfmc_error"] == m.attrs["mfmc_error_calibration_cost"]
        assert m.attrs["gamma"] == m.attrs["gamma_calibration_cost"]
        assert "mfmc_error_performance_cost" in m.attrs
        assert "gamma_performance_cost" in m.attrs
        assert m["seeds"].shape == (2, max(evals))
        assert "diffusion_coeff_initial" in m["model_1"]
        w = [float(m[f"model_{k}"].attrs["computation_time"]) for k in range(1, n + 1)]
        assert m["model_1"].attrs["n_cost_excluded"] == 0
        assert w[0] > 0
    with h5py.File(wd / "summary.hdf5") as f:
        assert list(f["budget"][:]) == [6.0, 20.0]
        assert f["mfmc_estimator"].shape == (2,)

    # a finished run is recognised and nothing is touched
    before = (wd / "summary.hdf5").stat().st_mtime_ns
    assert console_entry_point(argv(wd)) == 0
    assert "Run complete" in capsys.readouterr().out
    assert (wd / "summary.hdf5").stat().st_mtime_ns == before


def test_deterministic_across_runs(tmp_path: Path) -> None:
    """Same seed, same results (the dummy is deterministic)."""
    for name in ("a", "b"):
        console_entry_point(argv(tmp_path / name, budgets=("6",)))
    with h5py.File(tmp_path / "a" / "performance_6.hdf5") as fa, h5py.File(
        tmp_path / "b" / "performance_6.hdf5"
    ) as fb:
        assert (
            fa["models"].attrs["mfmc_estimator"] == fb["models"].attrs["mfmc_estimator"]
        )
        assert np.array_equal(fa["models/seeds"][:], fb["models/seeds"][:])


def test_rmse_cost_performance(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    console_entry_point(argv(wd, "--rmse-cost", "performance", budgets=("6",)))
    with h5py.File(wd / "performance_6.hdf5") as f:
        a = f["models"].attrs
        assert a["rmse_cost"] == "performance"
        assert a["mfmc_error"] == a["mfmc_error_performance_cost"]
        assert a["gamma"] == a["gamma_performance_cost"]


def test_stop_after_calibration_then_continue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wd = tmp_path / "run"
    assert console_entry_point(argv(wd, "--stop-after", "calibration")) == 0
    out = capsys.readouterr().out
    assert "Selected models" in out and "m* =" in out and "gamma" in out
    assert "Stopped after the calibration" in out
    assert (wd / "calibration.hdf5").exists() and (wd / "selection.json").exists()
    assert not (wd / "summary.hdf5").exists()
    sel_time = (wd / "selection.json").stat().st_mtime_ns
    cal_time = (wd / "calibration.hdf5").stat().st_mtime_ns

    # stop_after is not compared with the manifest: dropping it continues the run
    assert console_entry_point(argv(wd)) == 0
    out = capsys.readouterr().out
    assert "calibration read from" in out
    assert (wd / "summary.hdf5").exists()
    assert (wd / "selection.json").stat().st_mtime_ns == sel_time
    assert (wd / "calibration.hdf5").stat().st_mtime_ns == cal_time


def test_stop_after_reports_too_small_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wd = tmp_path / "run"
    assert (
        console_entry_point(argv(wd, "--stop-after", "calibration", budgets=("0.5",)))
        == 0
    )
    assert "NO ALLOCATION" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Budget 0.5"):
        console_entry_point(argv(tmp_path / "run2", budgets=("0.5",)))


def test_restart_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wd = tmp_path / "run"

    def killed(*args: Any, **kwargs: Any) -> bool:
        raise KeyboardInterrupt("job killed between the phases")

    with monkeypatch.context() as mp:
        mp.setattr(run_mod, "run_performance", killed)
        with pytest.raises(KeyboardInterrupt):
            console_entry_point(argv(wd))
    assert (wd / "calibration.hdf5").exists() and not (wd / "summary.hdf5").exists()
    n_results = len(list((wd / "cache").glob("N*/*/result.json")))
    cal_time = (wd / "calibration.hdf5").stat().st_mtime_ns

    assert console_entry_point(argv(wd)) == 0
    assert (wd / "summary.hdf5").exists()
    assert (wd / "calibration.hdf5").stat().st_mtime_ns == cal_time
    assert len(list((wd / "cache").glob("N*/*/result.json"))) > n_results
    # a second complete run from scratch gives the same estimator
    console_entry_point(argv(tmp_path / "fresh"))
    with h5py.File(wd / "performance_20.hdf5") as fa, h5py.File(
        tmp_path / "fresh" / "performance_20.hdf5"
    ) as fb:
        assert (
            fa["models"].attrs["mfmc_estimator"] == fb["models"].attrs["mfmc_estimator"]
        )


def test_restart_in_the_middle_of_a_phase(tmp_path: Path) -> None:
    """Part of the evaluations are cached; the restart computes only the rest."""
    wd = tmp_path / "run"
    console_entry_point(argv(wd, "--stop-after", "calibration"))
    # remove nothing: simulate a partial calibration by running into a new dir where
    # only half of the results exist (copy of the cache without calibration.hdf5)
    wd2 = tmp_path / "half"
    (wd2 / "cache").mkdir(parents=True)
    results = sorted((wd / "cache").glob("N*/*/result.json"))
    for r in results[: len(results) // 2]:
        dest = wd2 / "cache" / r.parent.parent.name / r.parent.name
        dest.mkdir(parents=True)
        (dest / "result.json").write_bytes(r.read_bytes())
    assert console_entry_point(argv(wd2)) == 0
    with h5py.File(wd / "calibration.hdf5") as fa, h5py.File(
        wd2 / "calibration.hdf5"
    ) as fb:
        assert np.array_equal(
            fa["models/model_1/diffusion_coeff"][:],
            fb["models/model_1/diffusion_coeff"][:],
        )


def test_manifest_strict_and_stored_only(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    console_entry_point(argv(wd, "--stop-after", "calibration"))
    # stored-only: total cores, lammps command, claim timeout
    stored_only = [
        "--lammps-cmd",
        "srun -n {ncpu} lmp -i {input}",
        "--claim-timeout",
        "99",
    ]
    console_entry_point(argv(wd, "--stop-after", "calibration", *stored_only))
    bad = argv(wd, "--stop-after", "calibration")
    bad[bad.index("--evals") + 1] = "9"
    with pytest.raises(ValueError, match="evals"):
        console_entry_point(bad)
    with pytest.raises(ValueError, match="protocol"):
        console_entry_point(argv(wd, "--protocol", "mfda"))
    # a seed omitted on restart continues with the stored seed
    no_seed = argv(wd, "--stop-after", "calibration")
    i = no_seed.index("--seed")
    del no_seed[i : i + 2]
    assert console_entry_point(no_seed) == 0


def test_foreign_manifest_refused(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    wd.mkdir()
    (wd / "manifest.json").write_text(json.dumps({"seed": 1, "n_chains": 2}))
    with pytest.raises(ValueError, match="not from mfmc-run"):
        console_entry_point(argv(wd))


def test_msdiff_version_change_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wd = tmp_path / "run"
    monkeypatch.setattr(run_mod, "_msdiff_version", lambda: "msdiff 0.3.2")
    console_entry_point(argv(wd, "--stop-after", "calibration"))
    capsys.readouterr()
    monkeypatch.setattr(run_mod, "_msdiff_version", lambda: "msdiff 0.1.0")
    console_entry_point(argv(wd, "--stop-after", "calibration"))
    assert "Warning: msdiff_version differs" in capsys.readouterr().out


def _patched_main(extra_code: str) -> str:
    """Return a script that patches ``make_forward`` and runs the CLI."""
    return textwrap.dedent(
        f"""
        import sys
        from mfwater.algo_mfmc_run import run
        from mfwater.algo_mfmc_run.dummy import CorrelatedDummyForwardModel
        from mfwater.md.cache import CachedForwardModel
        from mfwater.cli import console_entry_point

        def make_forward(cfg, cache_dir):
            {extra_code}
        run.make_forward = make_forward
        if __name__ == "__main__":  # spawned pool workers re-import this file
            sys.exit(console_entry_point(sys.argv[1:]))
        """
    )


def test_two_concurrent_jobs(tmp_path: Path) -> None:
    """Two jobs share a run directory: no evaluation is claimed twice, one result set."""
    wd = tmp_path / "run"
    script = tmp_path / "job.py"
    script.write_text(
        _patched_main(
            "return CachedForwardModel(CorrelatedDummyForwardModel("
            "cfg.protocol, cfg.orthoboxy, cfg.ncpu_override, sleep=0.3), cache_dir)"
        )
    )
    cmd = [sys.executable, str(script), *argv(wd)]
    procs = [
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for _ in range(2)
    ]
    outs = [p.communicate(timeout=300)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    assert (wd / "summary.hdf5").exists()
    entries = [r.parent for r in (wd / "cache").glob("N*/*/result.json")]
    assert entries
    # one claim per evaluation, no takeover: nobody computed a task twice
    for e in entries:
        assert (e / "claim.json").exists()
        assert not list(e.glob("claim_takeover_*"))
    # both jobs did something: together they computed every evaluation once
    computed = [o.count("computed=True") for o in outs]
    assert sum(computed) == len(entries)


def test_waiting_on_others_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_pool(tasks: Any, forward: Any, total: int, **kw: Any) -> PoolResult:
        return PoolResult(PoolStatus.WAITING_ON_OTHERS, n_waiting=3)

    monkeypatch.setattr(run_mod, "run_pool", fake_pool)
    wd = tmp_path / "run"
    assert console_entry_point(argv(wd)) == 0
    out = capsys.readouterr().out
    assert "3 evaluation(s) are still being computed by other job(s)" in out
    assert "resubmit" in out and "squeue" in out
    assert not (wd / "calibration.hdf5").exists()


def test_failed_pool_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pool(tasks: Any, forward: Any, total: int, **kw: Any) -> PoolResult:
        return PoolResult(PoolStatus.FAILED, n_failed=1, failures=["N=8: boom"])

    monkeypatch.setattr(run_mod, "run_pool", fake_pool)
    with pytest.raises(RuntimeError, match="resubmit"):
        console_entry_point(argv(tmp_path / "run"))


def test_incomplete_costs_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Evaluations without complete cost count for D but not for w."""
    from mfwater.md.cache import CachedForwardModel

    def make_forward(cfg: Any, cache_dir: Path) -> CachedForwardModel:
        model = CorrelatedDummyForwardModel(
            cfg.protocol, cfg.orthoboxy, cfg.ncpu_override, incomplete_every=3
        )
        return CachedForwardModel(model, cache_dir)

    monkeypatch.setattr(run_mod, "make_forward", make_forward)
    wd = tmp_path / "run"
    console_entry_point(argv(wd, "--stop-after", "calibration"))
    out = capsys.readouterr().out
    with h5py.File(wd / "calibration.hdf5") as f:
        for k in (1, 2, 3, 4):
            g = f[f"models/model_{k}"]
            n_ex = int(g.attrs["n_cost_excluded"])
            complete = g["cost_complete"][:]
            assert n_ex == int((~complete).sum())
            assert g["diffusion_coeff"].shape == (8,)  # D of excluded ones still used
            n = int(g.attrs["n_molecules"])
            assert g.attrs["computation_time"] == pytest.approx(0.01 * n**1.1)
            assert (
                f"model_{k} (N={n}): {n_ex} of 8 evaluations without complete cost"
                in out
            )
        assert (
            sum(
                int(f[f"models/model_{k}"].attrs["n_cost_excluded"])
                for k in (1, 2, 3, 4)
            )
            > 0
        )


def test_all_costs_incomplete_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mfwater.md.cache import CachedForwardModel

    def make_forward(cfg: Any, cache_dir: Path) -> CachedForwardModel:
        model = CorrelatedDummyForwardModel(
            cfg.protocol, cfg.orthoboxy, cfg.ncpu_override, incomplete_every=1
        )
        return CachedForwardModel(model, cache_dir)

    monkeypatch.setattr(run_mod, "make_forward", make_forward)
    with pytest.raises(ValueError, match="no complete cost"):
        console_entry_point(argv(tmp_path / "run"))


# ------------------------------------------------------- paper-I fixtures


def paper_argv(workdir: Path, *extra: str) -> list[str]:
    """Return the command line of a run on the paper-I calibration fixture."""
    return [
        "-a",
        "mfmc-run",
        "--workdir",
        str(workdir),
        "--total-cpus",
        "8",
        "--ncpu-override",
        "1024=6",
        "512=4",
        "256=2",
        "128=2",
        "64=1",
        "32=1",
        "--seed",
        "1",
        *extra,
    ]


def test_calibration_from_paper_fixture(tmp_path: Path) -> None:
    """Stats and costs come from the file: the allocation equals the paper's."""
    oms = FIXTURES / "oms_1e5" / "m6_e10_c_lj_oms.hdf5"
    wd = tmp_path / "run"
    code = console_entry_point(
        paper_argv(wd, "--calibration-from", str(CAL), "--budgets", "1e5")
    )
    assert code == 0
    assert not (wd / "calibration.hdf5").exists()
    with h5py.File(wd / "performance_100000.hdf5") as f, h5py.File(oms) as ref:
        m, r = f["models"], ref["models"]
        assert list(m.attrs["selected_molecules"]) == [1024, 32]
        for k in (1, 2):
            assert m[f"model_{k}"].attrs["n_evals"] == r[f"model_{k}"].attrs["n_evals"]
            assert m[f"model_{k}"].attrs["alpha"] == pytest.approx(
                r[f"model_{k}"].attrs["alpha"], rel=1e-5
            )
        assert m["model_1"].attrs["computation_time_initial"] == pytest.approx(61107.0)


def test_replay_performance_and_oms_allocation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oms5 = FIXTURES / "oms_1e5" / "m6_e10_c_lj_oms.hdf5"
    oms6 = FIXTURES / "oms_1e6" / "m6_e10_c_lj_oms.hdf5"
    base = ("--calibration-from", str(CAL), "--budgets", "1e5")

    wd = tmp_path / "same"
    console_entry_point(
        paper_argv(wd, *base, "--replay-performance-from", f"1e5={oms5}")
    )
    with h5py.File(wd / "performance_100000.hdf5") as f, h5py.File(oms5) as ref:
        assert np.array_equal(f["models/seeds"][:], ref["models/seeds"][:])
        assert np.array_equal(f["models/lj_params"][:], ref["models/lj_params"][:])
        assert f["models"].attrs["allocation_source"] == "computed"

    # a file with another allocation is an error unless asked for explicitly
    with pytest.raises(ValueError, match="replay_use_oms_allocation|replay-use-oms"):
        console_entry_point(
            paper_argv(
                tmp_path / "bad", *base, "--replay-performance-from", f"1e5={oms6}"
            )
        )
    capsys.readouterr()
    wd = tmp_path / "oms"
    console_entry_point(
        paper_argv(
            wd,
            *base,
            "--replay-performance-from",
            f"1e5={oms6}",
            "--replay-use-oms-allocation",
        )
    )
    out = capsys.readouterr().out
    assert "using models and n_evals of" in out
    with h5py.File(wd / "performance_100000.hdf5") as f, h5py.File(oms6) as ref:
        assert f["models"].attrs["allocation_source"] == "oms"
        for k in range(1, int(ref["models"].attrs["n_models"]) + 1):
            assert (
                f["models"][f"model_{k}"].attrs["n_evals"]
                == ref["models"][f"model_{k}"].attrs["n_evals"]
            )


def test_replay_from_calibration(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    code = console_entry_point(
        paper_argv(
            wd,
            "--replay-from",
            str(CAL),
            "--budgets",
            "1e5",
            "--stop-after",
            "calibration",
        )
    )
    assert code == 0
    with h5py.File(wd / "calibration.hdf5") as f, h5py.File(CAL) as ref:
        for name in ("lj_params", "charges", "seeds"):
            assert np.array_equal(f["models"][name][:], ref["models"][name][:])
        assert f["models/model_1"].attrs["n_molecules"] == 1024
        assert f["models/model_1/diffusion_coeff"].shape == (10,)
