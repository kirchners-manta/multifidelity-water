"""Tests of the cluster-independent job script generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from mfwater.algo_mfda.jobscript import load_profile, mfda_jobscript
from mfwater.argparser import parse_args, parser
from mfwater.cli import console_entry_point

RUN_TOML = """\
forward_model = "md"
workdir = "{workdir}"
chains = 2
molecules = [64, 32]
mcchainlength = 10
mcsubchainlength = 2
mcburnin = 2
"""

PROFILE = """\
scheduler = "slurm"
cores_per_node = {cores}
[slurm]
partition = "testpart"
account = "acc"
time = "01:00:00"
extra = ["--mem-per-cpu=2G"]
[environment]
setup = "module load fake\\nexport FOO=${{BAR}}"
[lammps]
cmd = "srun -n {{ncpu}} lmp -i {{input}}"
"""


@pytest.fixture(autouse=True)
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in an empty directory so that default output files are isolated."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_inputs(tmp: Path, cores: int = 96, extra_run: str = "") -> tuple[Path, Path]:
    """Write a run TOML and a cluster profile, return their paths."""
    run = tmp / "run.toml"
    run.write_text(RUN_TOML.format(workdir="./myrun") + extra_run)
    cluster = tmp / "testcluster.toml"
    cluster.write_text(PROFILE.format(cores=cores))
    return run, cluster


def parse(run: Path, cluster: Path, *extra: str) -> argparse.Namespace:
    """Parse the mfda-jobscript command line."""
    return parse_args(
        parser(),
        [
            "-a",
            "mfda-jobscript",
            "-c",
            str(run),
            "--cluster-file",
            str(cluster),
            *extra,
        ],
    )


def test_generated_script(in_tmp: Path) -> None:
    run, cluster = write_inputs(in_tmp)
    assert mfda_jobscript(parse(run, cluster)) == 0
    text = (in_tmp / "run-mfda-testcluster.sh").read_text()
    assert text.startswith("#!/bin/bash\n")
    for line in (
        "#SBATCH --partition=testpart",
        "#SBATCH --account=acc",
        "#SBATCH --ntasks=",
        "#SBATCH --nodes=1",
        "#SBATCH --time=01:00:00",
        "#SBATCH --job-name=myrun",
        "#SBATCH --mem-per-cpu=2G",
        "module load fake",
        "export FOO=${BAR}",
        f"-a markov-chain -c {run.resolve()}",
        "--lammps-cmd 'srun -n {ncpu} lmp -i {input}'",
    ):
        assert line in text
    # 2 chains x calc_cpus(64) tasks
    ntasks = int(next(l for l in text.splitlines() if "--ntasks=" in l).split("=")[1])
    assert ntasks % 2 == 0 and ntasks >= 2
    assert "@" not in text.replace("${", "")  # no unfilled placeholder


def test_builtin_marvin_profile() -> None:
    p = load_profile("marvin")
    assert p.scheduler == "slurm"
    assert p.cores_per_node == 96
    assert p.header["partition"] == "intelsr_long"
    assert "module load LAMMPS" in p.setup


def test_jobscript_option_and_cli(in_tmp: Path) -> None:
    run, cluster = write_inputs(in_tmp)
    out = in_tmp / "custom.sh"
    argv = [
        "-a",
        "mfda-jobscript",
        "-c",
        str(run),
        "--cluster-file",
        str(cluster),
        "--jobscript",
        str(out),
    ]
    assert console_entry_point(argv) == 0
    assert out.exists()


def test_refuses_overwrite(in_tmp: Path) -> None:
    run, cluster = write_inputs(in_tmp)
    out = in_tmp / "existing.sh"
    out.write_text("keep me")
    with pytest.raises(FileExistsError):
        mfda_jobscript(parse(run, cluster, "--jobscript", str(out)))
    assert out.read_text() == "keep me"


def test_unknown_scheduler(in_tmp: Path) -> None:
    run, cluster = write_inputs(in_tmp)
    cluster.write_text(PROFILE.format(cores=96).replace("slurm", "pbs"))
    with pytest.raises(ValueError, match="slurm"):
        mfda_jobscript(parse(run, cluster))
    assert not (in_tmp / "run-mfda-testcluster.sh").exists()


def test_unknown_builtin_cluster() -> None:
    with pytest.raises(ValueError, match="marvin"):
        load_profile("nonexistent")


def test_ntasks_exceeds_node(in_tmp: Path) -> None:
    run, cluster = write_inputs(in_tmp, cores=1)
    with pytest.raises(ValueError, match="do not fit"):
        mfda_jobscript(parse(run, cluster))
    assert not (in_tmp / "run-mfda-testcluster.sh").exists()


def test_invalid_run_settings(in_tmp: Path) -> None:
    run, cluster = write_inputs(in_tmp)
    run.write_text(RUN_TOML.format(workdir="./w").replace("[64, 32]", "[32, 64]"))
    with pytest.raises(ValueError):
        mfda_jobscript(parse(run, cluster))


def test_run_toml_lammps_cmd_wins(in_tmp: Path) -> None:
    run, cluster = write_inputs(
        in_tmp, extra_run='lammps_cmd = "mine {ncpu} {input}"\n'
    )
    mfda_jobscript(parse(run, cluster))
    text = (in_tmp / "run-mfda-testcluster.sh").read_text()
    assert "    --lammps-cmd" not in text  # the option line, not the comment


def test_requires_run_toml(in_tmp: Path) -> None:
    _, cluster = write_inputs(in_tmp)
    args = parse_args(
        parser(), ["-a", "mfda-jobscript", "--cluster-file", str(cluster)]
    )
    with pytest.raises(ValueError, match="-c"):
        mfda_jobscript(args)


def test_cluster_options_exclusive(in_tmp: Path) -> None:
    run, cluster = write_inputs(in_tmp)
    with pytest.raises(ValueError, match="exactly one"):
        mfda_jobscript(parse(run, cluster, "--cluster", "marvin"))
    assert os.listdir(in_tmp).count("run-mfda-marvin.sh") == 0
