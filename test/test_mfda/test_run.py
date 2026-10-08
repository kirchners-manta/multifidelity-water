"""End-to-end tests of the MFDA run management with the dummy forward model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from mfwater.algo_mfda import run_manager
from mfwater.algo_mfda.forward_model import DummyForwardModel
from mfwater.algo_mfda.markov_chain import build_config
from mfwater.algo_mfda.run_manager import markov_chain_eval, mfda_smoke
from mfwater.argparser import parser

BASE = [
    "-a",
    "markov-chain",
    "--models",
    "3",
    "--molecules",
    "64",
    "32",
    "16",
    "--mcchainlength",
    "8",
    "--mcsubchainlength",
    "2",
    "3",
    "--mcburnin",
    "2",
    "--params",
    "lj-q",
    "--seed",
    "7",
]


def make_args(workdir: Path, *extra: str) -> argparse.Namespace:
    """Parse arguments for a run in ``workdir`` (later options override earlier)."""
    return parser().parse_args([*BASE, "--workdir", str(workdir), *extra])


def read_chain(workdir: Path, c: int = 0) -> dict[str, np.ndarray]:
    """Read all datasets of a chain file."""
    with h5py.File(workdir / f"chain_{c:03d}.hdf5", "r") as f:
        return {k: np.asarray(f[k]) for k in f}


def test_complete_run(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    assert markov_chain_eval(make_args(wd)) == 0
    assert (wd / "manifest.json").exists()
    assert (wd / "chain_000.hdf5").exists()
    assert (wd / "default.hdf5").exists()
    assert list((wd / "cache" / "chain_000").glob("N*/*/result.json"))
    chain = read_chain(wd)
    assert chain["samples_level1"].shape == (6, 3)
    assert chain["samples_level3"].shape == (6 * 2 * 3, 3)


class Interrupted:
    """Dummy forward model that fails after ``limit`` new evaluations."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self.inner = DummyForwardModel()

    def __call__(self, n_molecules: int, theta: np.ndarray) -> float:
        if self.count >= self.limit:
            raise RuntimeError("interrupted")
        self.count += 1
        return self.inner(n_molecules, theta)


def test_restart_reproduces_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = tmp_path / "ref"
    markov_chain_eval(make_args(ref))

    wd = tmp_path / "interrupted"
    original = run_manager.make_inner_model
    monkeypatch.setattr(
        run_manager, "make_inner_model", lambda s, c: Interrupted(limit=5)
    )
    with pytest.raises(RuntimeError, match="failed"):
        markov_chain_eval(make_args(wd))
    assert not (wd / "chain_000.hdf5").exists()
    assert not (wd / "default.hdf5").exists()
    cached = list((wd / "cache" / "chain_000").glob("N*/*/result.json"))
    assert len(cached) == 5

    # restart: same command, working model; cached evaluations are reused
    monkeypatch.setattr(run_manager, "make_inner_model", original)
    markov_chain_eval(make_args(wd))
    a, b = read_chain(ref), read_chain(wd)
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_restart_skips_finished_chains_and_refuses_complete_run(
    tmp_path: Path,
) -> None:
    wd = tmp_path / "run"
    markov_chain_eval(make_args(wd))
    before = (wd / "chain_000.hdf5").stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="complete"):
        markov_chain_eval(make_args(wd))
    assert (wd / "chain_000.hdf5").stat().st_mtime_ns == before
    # new summary name, chain file exists: chain is skipped, not overwritten
    markov_chain_eval(make_args(wd, "-o", "second.hdf5"))
    assert (wd / "chain_000.hdf5").stat().st_mtime_ns == before
    assert (wd / "second.hdf5").exists()


@pytest.mark.parametrize(
    "extra",
    [["--seed", "8"], ["--mcburnin", "3"], ["--params", "lj"], ["--chains", "2"]],
)
def test_manifest_mismatch(tmp_path: Path, extra: list[str]) -> None:
    wd = tmp_path / "run"
    markov_chain_eval(make_args(wd))
    manifest = (wd / "manifest.json").read_text()
    with pytest.raises(ValueError, match="differ from the manifest"):
        markov_chain_eval(make_args(wd, "-o", "other.hdf5", *extra))
    assert (wd / "manifest.json").read_text() == manifest


def test_manifest_git_hash_mismatch(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    markov_chain_eval(make_args(wd))
    path = wd / "manifest.json"
    data = json.loads(path.read_text())
    data["git_hash"] = "0" * 40
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="git_hash"):
        run_manager.prepare_run(make_args(wd, "-o", "other.hdf5"))


def test_restart_without_seed_uses_manifest_seed(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    argv = [a for a in BASE if a not in ("--seed", "7")]
    args = parser().parse_args([*argv, "--workdir", str(wd)])
    assert args.seed is None
    run_manager.prepare_run(args)
    seed = json.loads((wd / "manifest.json").read_text())["seed"]
    _, settings, restart = run_manager.prepare_run(args)
    assert restart and settings["seed"] == seed


def test_two_chains(tmp_path: Path) -> None:
    wd = tmp_path / "run"
    markov_chain_eval(make_args(wd, "--chains", "2"))
    c0, c1 = read_chain(wd, 0), read_chain(wd, 1)
    assert not np.array_equal(c0["samples_level1"], c1["samples_level1"])
    assert (wd / "cache" / "chain_000").is_dir()
    assert (wd / "cache" / "chain_001").is_dir()
    with h5py.File(wd / "default.hdf5", "r") as f:
        np.testing.assert_allclose(
            f["estimator"], (c0["estimator"] + c1["estimator"]) / 2
        )
        np.testing.assert_allclose(
            f["estimator_std_between_chains"],
            np.std([c0["estimator"], c1["estimator"]], axis=0, ddof=1),
        )


def test_single_subchain_value_is_broadcast() -> None:
    args = parser().parse_args(
        ["--models", "3", "--molecules", "64", "32", "16", "--mcsubchainlength", "4"]
    )
    assert build_config(args).subchain_lengths == (4, 4)
    args = parser().parse_args(["--models", "3", "--molecules", "64", "32", "16"])
    assert build_config(args).subchain_lengths == (10, 10)
    assert args.n_mc_chain_length == 1000 and args.n_mc_burnin == 100


def test_ncpu(capsys: pytest.CaptureFixture[str]) -> None:
    args = parser().parse_args(
        ["-a", "mfda-ncpu", "--molecules", "1000", "500", "--chains", "3"]
    )
    assert run_manager.mfda_ncpu(args) == 0
    assert capsys.readouterr().out.strip() == "18"  # 3 * calc_cpus(1000) = 3 * 6


def test_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = parser().parse_args(
        ["-a", "mfda-smoke", "--molecules", "32", "--workdir", str(tmp_path)]
    )
    assert mfda_smoke(args) == 0
    assert "D = " in capsys.readouterr().out
    assert list((tmp_path / "smoke").glob("N32/*/result.json"))
    args.n_molecules = [32, 16]
    with pytest.raises(ValueError, match="exactly one"):
        mfda_smoke(args)
