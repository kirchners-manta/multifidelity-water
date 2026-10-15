"""Regression tests: the MFDA protocol must stay byte-identical.

Production MFDA runs on Marvin depend on the cache key hash (cache directory
names), the derived seeds and the generated LAMMPS/TRAVIS inputs. The reference
values in ``data/mfda_reference/`` were recorded from the code *before* the MD
layer was moved to :mod:`mfwater.md` and extended with protocols. Do not
regenerate them to make a failing test pass: a failure means that existing
caches or running chains would be invalidated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.algo_mfda import md_pipeline
from mfwater.algo_mfda.cache import CachedForwardModel
from mfwater.algo_mfda.forward_model import derive_seeds, evaluation_key, key_hash
from mfwater.md.pipeline import MDForwardModel

from ..test_mfda.test_forward_model import make_fake_run

REF = Path(__file__).parent / "data" / "mfda_reference"
KEY_CASES = json.loads((REF / "keys.json").read_text())
THETA = np.array([0.1634, 3.17427, -0.8952])
ORTHO_FILES = ("msd_H2O_#2.csv", "msd_H2O_#2_XY.csv", "msd_H2O_#2_Z.csv")


@pytest.mark.parametrize("case", KEY_CASES, ids=lambda c: f"N{c['n']}")
def test_key_hash_and_seeds_match_recorded_values(case: dict[str, Any]) -> None:
    key = evaluation_key(case["n"], np.array(case["theta"]))
    assert repr(key) == case["key"]
    assert key_hash(key) == case["hash"]
    assert list(derive_seeds(key)) == case["seeds"]


@pytest.mark.parametrize("orthoboxy", [False, True], ids=["cubic", "orthoboxy"])
@pytest.mark.parametrize("case", KEY_CASES, ids=lambda c: f"N{c['n']}")
def test_cache_entry_dir_matches_recorded_hash(
    tmp_path: Path, case: dict[str, Any], orthoboxy: bool
) -> None:
    # the key never contained the box shape for MFDA; the shape is checked in
    # result.json instead, so cubic and orthoboxy share the directory name
    model = MDForwardModel(tmp_path, orthoboxy=orthoboxy)
    cached = CachedForwardModel(model, tmp_path)
    entry = cached.entry_dir(case["n"], np.array(case["theta"]))
    assert entry == tmp_path / f"N{case['n']}" / case["hash"]


def test_generated_lammps_inputs_match_recorded_files(tmp_path: Path) -> None:
    case = KEY_CASES[0]
    theta = np.array(case["theta"])
    out = tmp_path / "input.lmp"
    md_pipeline.write_lammps_input(
        md_pipeline.MFDA_INPUT_TEMPLATE, out, theta, case["seeds"][1]
    )
    assert out.read_bytes() == (REF / "input_mfda_N16.lmp").read_bytes()
    for step, name in [
        (100_000, "input_resume_eq.lmp"),
        (500_000, "input_resume_prod.lmp"),
    ]:
        out = tmp_path / name
        md_pipeline.write_resume_input(out, Path(f"/RST/simout/restart.{step}"), step)
        assert out.read_bytes() == (REF / name).read_bytes()


def test_patched_travis_inputs_match_recorded_files() -> None:
    for tmpl, name in [
        ("travis_input_msd.txt", "travis_cubic.txt"),
        ("travis_input_msd_orthoboxy.txt", "travis_orthoboxy.txt"),
    ]:
        text = (md_pipeline.TEMPLATE_DIR / tmpl).read_text()
        assert md_pipeline.patch_travis_input(text) == (REF / name).read_text()


@pytest.mark.parametrize("orthoboxy", [False, True], ids=["cubic", "orthoboxy"])
def test_full_pipeline_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orthoboxy: bool
) -> None:
    """External-program calls and generated input files of one evaluation."""
    calls: list[Any] = []
    monkeypatch.setattr(md_pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    monkeypatch.setattr(
        md_pipeline.subprocess, "run", make_fake_run(calls, travis_files=ORTHO_FILES)
    )
    model = MDForwardModel(tmp_path, orthoboxy=orthoboxy)
    CachedForwardModel(model, tmp_path)(100, THETA)
    snap = json.loads(
        (REF / f"pipeline_{'ortho' if orthoboxy else 'cubic'}.json").read_text()
    )
    attempt = model.last_attempt_dir
    assert attempt is not None
    assert [[a, str(c.relative_to(tmp_path))] for a, c in calls] == snap["calls"]
    assert str(attempt.parent.relative_to(tmp_path)) == snap["entry"]
    for rel, text in snap["files"].items():
        assert (attempt / rel).read_text() == text, rel
