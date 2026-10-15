"""Tests for the extended evaluation key (protocol, box shape, explicit seeds)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.md import pipeline
from mfwater.md.cache import CachedForwardModel
from mfwater.md.forward_model import (
    MAX_SEED,
    derive_seeds,
    evaluation_key,
    key_hash,
)
from mfwater.md.pipeline import MDForwardModel
from mfwater.md.protocol import PAPER1_PROTOCOL

from ..test_mfda.test_forward_model import make_fake_run

THETA = np.array([0.1634, 3.17427, -0.8952])
SEEDS = (12345, 67890)


# ------------------------------------------------------------------- the key
def test_legacy_key_unchanged_for_mfda_without_seeds() -> None:
    key = evaluation_key(100, THETA)
    assert len(key) == 4
    # box shape is not part of the legacy key (checked in result.json instead)
    assert evaluation_key(100, THETA, protocol="mfda", orthoboxy=True) == key
    assert evaluation_key(100, THETA, "mfda", False, None) == key
    assert key_hash(key) == "2e31a4941f0a1a2e"  # N100/2e31a4941f0a1a2e (recorded)


def test_extended_key_layout() -> None:
    base = evaluation_key(100, THETA)
    key = evaluation_key(100, THETA, "paper1", True, SEEDS)
    assert key == (*base, "paper1", True, SEEDS)
    assert evaluation_key(100, THETA, "paper1") == (*base, "paper1", False, None)
    assert evaluation_key(100, THETA, "mfda", False, SEEDS) == (
        *base,
        "mfda",
        False,
        SEEDS,
    )


def test_all_key_ingredients_change_the_hash() -> None:
    variants = [
        evaluation_key(100, THETA),
        evaluation_key(100, THETA, "paper1"),
        evaluation_key(100, THETA, "paper1", True),
        evaluation_key(100, THETA, "paper1", False, SEEDS),
        evaluation_key(100, THETA, "paper1", False, (SEEDS[0], SEEDS[1] + 1)),
        evaluation_key(100, THETA, "mfda", False, SEEDS),
        evaluation_key(100, THETA, "mfda", True, SEEDS),
    ]
    assert len({key_hash(k) for k in variants}) == len(variants)


def test_seeds_derived_or_explicit() -> None:
    derived = derive_seeds(evaluation_key(100, THETA, "paper1"))
    assert all(1 <= s <= MAX_SEED for s in derived)
    # other protocols get their own, independent derived seeds
    assert derived != derive_seeds(evaluation_key(100, THETA))
    # explicit seeds are returned unchanged
    assert derive_seeds(evaluation_key(100, THETA, "paper1", False, SEEDS)) == SEEDS
    assert derive_seeds(evaluation_key(100, THETA, "mfda", False, SEEDS)) == SEEDS


@pytest.mark.parametrize("bad", [(0, 1), (1, MAX_SEED + 1), (1.5, 2), (1, 2, 3)])
def test_invalid_seeds(bad: Any) -> None:
    with pytest.raises(ValueError, match="seed"):
        evaluation_key(100, THETA, "paper1", False, bad)


# ------------------------------------------------------------------- the cache
class Counting:
    """Forward model recording how it was called."""

    def __init__(self, protocol: Any = None, orthoboxy: bool = False) -> None:
        self.calls: list[Any] = []
        if protocol is not None:
            self.protocol = protocol
        self.orthoboxy = orthoboxy

    def __call__(
        self, n: int, theta: np.ndarray, seeds: tuple[int, int] | None = None
    ) -> float:
        self.calls.append(seeds)
        return 1.5


def test_cache_without_protocol_attribute_uses_legacy_key(tmp_path: Path) -> None:
    cached = CachedForwardModel(Counting(), tmp_path)
    cached(100, THETA)
    entry = cached.entry_dir(100, THETA)
    assert entry.name == "2e31a4941f0a1a2e"
    data = json.loads((entry / "result.json").read_text())
    assert data["protocol"] == "mfda" and data["key"] == list(
        evaluation_key(100, THETA)
    )


def test_legacy_result_json_without_new_fields_is_still_read(tmp_path: Path) -> None:
    """Results written by the first MFDA version have no 'key'/'protocol'."""
    model = Counting()
    cached = CachedForwardModel(model, tmp_path)
    key = evaluation_key(100, THETA)
    entry = tmp_path / "N100" / key_hash(key)
    entry.mkdir(parents=True)
    legacy = {
        "D": 2.0,
        "N": 100,
        "theta_hex": {"epsilon": key[1], "sigma": key[2], "q_O": key[3]},
        "orthoboxy": None,
    }
    (entry / "result.json").write_text(json.dumps(legacy))
    assert cached(100, THETA) == 2.0
    assert model.calls == []


def test_protocol_seeds_and_box_shape_select_distinct_entries(tmp_path: Path) -> None:
    paper = PAPER1_PROTOCOL
    dirs = set()
    for model in (
        Counting(),
        Counting(paper),
        Counting(paper, orthoboxy=True),
    ):
        cached = CachedForwardModel(model, tmp_path)
        dirs.add(cached.entry_dir(100, THETA))
        dirs.add(cached.entry_dir(100, THETA, SEEDS))
    assert len(dirs) == 6
    assert all(d.parent == tmp_path / "N100" for d in dirs)


def test_cache_passes_explicit_seeds_and_stores_them(tmp_path: Path) -> None:
    model = Counting()
    cached = CachedForwardModel(model, tmp_path)
    assert cached(100, THETA, seeds=SEEDS) == 1.5
    cached(100, THETA, seeds=SEEDS)  # memo
    cached(100, THETA)  # other key: other entry, called without seeds
    assert model.calls == [SEEDS, None]
    data = json.loads((cached.entry_dir(100, THETA, SEEDS) / "result.json").read_text())
    assert data["seeds"] == {"packmol": SEEDS[0], "velocity": SEEDS[1]}
    # a fresh wrapper finds the result on disk
    model2 = Counting()
    assert CachedForwardModel(model2, tmp_path)(100, THETA, seeds=SEEDS) == 1.5
    assert model2.calls == []


def test_result_with_wrong_key_is_rejected(tmp_path: Path) -> None:
    cached = CachedForwardModel(Counting(PAPER1_PROTOCOL), tmp_path)
    cached(100, THETA)
    entry = cached.entry_dir(100, THETA)
    result = entry / "result.json"
    data = json.loads(result.read_text())
    data["key"][4] = "other"
    result.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        CachedForwardModel(Counting(PAPER1_PROTOCOL), tmp_path)(100, THETA)


def test_orthoboxy_mismatch_still_detected_for_mfda(tmp_path: Path) -> None:
    CachedForwardModel(Counting(orthoboxy=False), tmp_path)(100, THETA)
    with pytest.raises(ValueError, match="orthoboxy"):
        CachedForwardModel(Counting(orthoboxy=True), tmp_path)(100, THETA)


# ------------------------------------------------------------- the MD pipeline
def test_md_forward_model_uses_explicit_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    monkeypatch.setattr(pipeline.subprocess, "run", make_fake_run(calls))
    model = MDForwardModel(tmp_path, protocol="paper1")
    cached = CachedForwardModel(model, tmp_path)
    cached(100, THETA, seeds=SEEDS)
    attempt = model.last_attempt_dir
    assert attempt is not None
    # same directory as the cache entry
    assert attempt.parent == cached.entry_dir(100, THETA, SEEDS)
    assert f"seed {SEEDS[0]}" in (attempt / "siminp" / "pack.inp").read_text()
    assert (
        f"velocity all create ${{vTK}} {SEEDS[1]}\n"
        in (attempt / "siminp" / "input.lmp").read_text()
    )
    data = json.loads((attempt.parent / "result.json").read_text())
    assert data["protocol"] == "paper1"
    assert data["seeds"] == {"packmol": SEEDS[0], "velocity": SEEDS[1]}


def test_md_forward_model_derives_seeds_per_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    monkeypatch.setattr(pipeline.subprocess, "run", make_fake_run(calls))
    model = MDForwardModel(tmp_path, protocol="paper1")
    CachedForwardModel(model, tmp_path)(100, THETA)
    attempt = model.last_attempt_dir
    assert attempt is not None
    velocity = derive_seeds(evaluation_key(100, THETA, "paper1"))[1]
    assert (
        f"velocity all create ${{vTK}} {velocity}\n"
        in (attempt / "siminp" / "input.lmp").read_text()
    )
