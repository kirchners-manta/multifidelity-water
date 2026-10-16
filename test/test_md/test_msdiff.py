"""Tests for the configurable msdiff executable and its recorded version."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.md import pipeline
from mfwater.md.cache import CachedForwardModel
from mfwater.md.pipeline import (
    MDForwardModel,
    parse_msdiff_output,
    query_msdiff_version,
)

from ..test_mfda.test_forward_model import make_fake_run

THETA = np.array([0.1634, 3.17427, -0.8952])

OLD_HEADER = (
    " D_0 / 10^-12 m^2/s,        delta_D, K / 10^-12 m^2/s,    D_z / 10^-12 m^2/s\n"
)
NEW_HEADER = " Species, D_0 / 10^-12 m^2/s,        delta_D, K / 10^-12 m^2/s,    D_z / 10^-12 m^2/s\n"


@pytest.mark.parametrize(
    "header, row",
    [
        (
            OLD_HEADER,
            "   2500.000000,       0.000100,       889.000000,         7.000000\n",
        ),
        (
            NEW_HEADER,
            "       1,        2500.000000,       0.000100,       889.000000,         7.000000\n",
        ),
    ],
    ids=["old", "new"],
)
def test_parse_msdiff_both_headers(tmp_path: Path, header: str, row: str) -> None:
    """D_0 is found by header in both layouts; K and D_z are never used."""
    f = tmp_path / "msdiff_out.csv"
    f.write_text(header + row)
    assert parse_msdiff_output(f) == pytest.approx(2.5)  # 1e-9 m^2/s


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_query_msdiff_version(tmp_path: Path) -> None:
    ok = _script(tmp_path / "msdiff_new", 'echo "msdiff 0.3.2"\n')
    assert query_msdiff_version(ok) == "msdiff 0.3.2"
    # an old build rejects --version and prints its usage with exit code 2
    old = _script(tmp_path / "msdiff_old", 'echo "usage: msdiff [-h] -f F"\nexit 2\n')
    assert query_msdiff_version(old) == f"unknown ({old})"
    assert query_msdiff_version(str(tmp_path / "missing")) == (
        f"unknown ({tmp_path / 'missing'})"
    )


def test_msdiff_cmd_used_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    exe = _script(tmp_path / "my_msdiff", 'echo "msdiff 9.9"\n')
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    fake = make_fake_run([])

    def run(args: list[str], cwd: Any = None, **kw: Any) -> None:
        calls.append((list(args), cwd))
        # the fake knows the program by name: present the custom msdiff as "msdiff"
        fake(["msdiff" if args[0] == exe else args[0], *args[1:]], cwd=cwd, **kw)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    model = MDForwardModel(tmp_path / "w", protocol="paper1", msdiff_cmd=exe)
    assert exe in model._required_programs()
    forward = CachedForwardModel(model, tmp_path / "w")
    forward(100, THETA, seeds=(5, 6))
    msdiff_calls = [c for c, _ in calls if c[0] == exe]
    assert len(msdiff_calls) == 1 and "--from-travis" in msdiff_calls[0]
    assert not any(c[0] == "msdiff" for c, _ in calls)
    result = next((tmp_path / "w").glob("N100/*/result.json"))
    data = json.loads(result.read_text())
    assert data["msdiff_cmd"] == exe and data["msdiff_version"] == "msdiff 9.9"


def test_default_msdiff_cmd_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    monkeypatch.setattr(pipeline.subprocess, "run", make_fake_run(calls))
    MDForwardModel(tmp_path)(100, THETA)
    assert [c[0] for c, _ in calls][-1] == "msdiff"
