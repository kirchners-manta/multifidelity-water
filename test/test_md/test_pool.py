"""Tests for the evaluation pool (:mod:`mfwater.md.pool`)."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.md.cache import CachedForwardModel
from mfwater.md.pool import PoolStatus, Task, run_pool, try_claim

THETA = np.array([0.15, 3.16, -0.8])


class FakeModel:
    """Picklable fake forward model that logs start/end times of every call.

    Parameters
    ----------
    log_dir : str
        Directory receiving one JSON file per call.
    sleep : float
        Duration of one evaluation in seconds.
    cores : dict[int, int]
        Cores per ``N``.
    fail_times : dict[int, int]
        Number of times an evaluation of ``N`` raises before it succeeds.
    """

    def __init__(
        self,
        log_dir: str | Path,
        sleep: float = 0.3,
        cores: dict[int, int] | None = None,
        fail_times: dict[int, int] | None = None,
    ) -> None:
        self.log_dir = str(log_dir)
        self.sleep = sleep
        self.cores = cores or {}
        self.fail_times = fail_times or {}

    def n_mpi(self, n: int) -> int:
        return self.cores.get(n, 1)

    def __call__(
        self, n: int, theta: np.ndarray, seeds: tuple[int, int] | None = None
    ) -> float:
        tag = f"{n}_{os.getpid()}_{time.time_ns()}"
        start = time.time()
        log = Path(self.log_dir)
        n_prev = len(list(log.glob(f"fail_{n}_*")))
        if n_prev < self.fail_times.get(n, 0):
            (log / f"fail_{n}_{tag}").write_text("")
            raise RuntimeError(f"boom N={n}")
        time.sleep(self.sleep)
        rec = {"n": n, "cores": self.n_mpi(n), "start": start, "end": time.time()}
        (log / f"call_{tag}.json").write_text(json.dumps(rec))
        return float(n) / 100.0


def calls(log_dir: Path) -> list[dict[str, Any]]:
    """Return the records of all completed fake evaluations."""
    return [json.loads(p.read_text()) for p in log_dir.glob("call_*.json")]


def make(tmp_path: Path, **kw: Any) -> tuple[CachedForwardModel, Path]:
    """Build a cached fake model and its log directory."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(exist_ok=True)
    return CachedForwardModel(FakeModel(log_dir, **kw), tmp_path / "cache"), log_dir


def tasks_for(ns: list[int], seed0: int = 1) -> list[Task]:
    """Tasks with distinct explicit seeds, one per entry of ``ns``."""
    return [Task(n, THETA, seeds=(seed0 + i, 7)) for i, n in enumerate(ns)]


def quiet(_msg: str) -> None:
    """Discard log lines."""


def test_packing_and_order(tmp_path: Path) -> None:
    fwd, log_dir = make(tmp_path, cores={1024: 4, 512: 2, 256: 1})
    ns = [256, 256, 512, 1024, 512, 256]
    res = run_pool(tasks_for(ns), fwd, 4, tick=0.05, log=quiet)
    assert res.status is PoolStatus.COMPLETE
    assert res.n_computed == len(ns)
    cs = sorted(calls(log_dir), key=lambda c: c["start"])
    assert len(cs) == 6
    # largest N started first
    assert cs[0]["n"] == 1024
    # never more cores in use than available (check at every start time)
    for c in cs:
        used = sum(o["cores"] for o in cs if o["start"] <= c["start"] + 1e-6 < o["end"])
        assert used <= 4
    # smaller tasks did run concurrently
    assert any(
        a["end"] > b["start"] for a in cs for b in cs if a is not b and a["n"] != 1024
    )


def test_second_call_uses_cache(tmp_path: Path) -> None:
    fwd, log_dir = make(tmp_path)
    t = tasks_for([256, 256, 512])
    run_pool(t, fwd, 2, tick=0.05, log=quiet)
    res = run_pool(t, fwd, 2, tick=0.05, log=quiet)
    assert (res.n_cached, res.n_computed) == (3, 0)
    assert len(calls(log_dir)) == 3


def test_too_many_cores_raises(tmp_path: Path) -> None:
    fwd, _ = make(tmp_path, cores={512: 8})
    with pytest.raises(ValueError, match="needs 8 cores"):
        run_pool(tasks_for([512]), fwd, 4, log=quiet)


def _proc(cache_dir: str, log_dir: str, out: Any) -> None:
    """Run one pool in a separate process on the shared directory."""
    fwd = CachedForwardModel(FakeModel(log_dir, sleep=0.2), cache_dir)
    res = run_pool(
        tasks_for([256] * 12 + [512] * 4), fwd, 4, tick=0.05, poll_timeout=0, log=quiet
    )
    out.put((res.n_computed, res.status.value))


def test_two_processes_never_compute_same_task(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    out = ctx.Queue()
    ps = [
        ctx.Process(target=_proc, args=(str(tmp_path / "cache"), str(log_dir), out))
        for _ in range(2)
    ]
    for p in ps:
        p.start()
    results = [out.get(timeout=120) for _ in ps]
    for p in ps:
        p.join(timeout=60)
    assert len(calls(log_dir)) == 16  # every task exactly once
    assert sum(r[0] for r in results) == 16


def test_live_claim_gives_waiting(tmp_path: Path) -> None:
    fwd, _ = make(tmp_path)
    entry = fwd.entry_dir(256, THETA, (1, 7))
    assert try_claim(entry, 60.0) is not None
    res = run_pool(tasks_for([256, 512]), fwd, 2, tick=0.05, log=quiet)
    assert res.status is PoolStatus.WAITING_ON_OTHERS
    assert (res.n_computed, res.n_waiting) == (1, 1)
    assert res.exit_code == 0


def test_poll_picks_up_result_of_others(tmp_path: Path) -> None:
    fwd, _ = make(tmp_path)
    entry = fwd.entry_dir(256, THETA, (1, 7))
    assert try_claim(entry, 60.0) is not None
    # "other job" finishes shortly after we start polling
    threading.Timer(0.3, lambda: (entry / "result.json").write_text("{}")).start()
    res = run_pool(
        tasks_for([256]), fwd, 1, poll_timeout=5, poll_interval=0.1, log=quiet
    )
    assert res.status is PoolStatus.COMPLETE
    assert res.n_cached == 1


def test_stale_claim_takeover(tmp_path: Path) -> None:
    fwd, _ = make(tmp_path)
    entry = fwd.entry_dir(256, THETA, (1, 7))
    assert try_claim(entry, 60.0) is not None
    old = time.time() - 3600
    for p in entry.glob("claim*"):
        os.utime(p, (old, old))
    before = (entry / "claim.json").read_text()
    res = run_pool(tasks_for([256]), fwd, 1, stale_timeout=60, tick=0.05, log=quiet)
    assert res.status is PoolStatus.COMPLETE and res.n_computed == 1
    assert (entry / "claim_takeover_1.json").exists()
    assert (entry / "claim.json").read_text() == before  # untouched
    assert (entry / "claim.json").stat().st_mtime == pytest.approx(old, abs=1)


def test_takeover_race_has_one_winner(tmp_path: Path) -> None:
    entry = tmp_path / "e"
    assert try_claim(entry, 60.0) is not None
    old = time.time() - 3600
    for p in entry.glob("claim*"):
        os.utime(p, (old, old))
    wins = [try_claim(entry, 60.0) for _ in range(3)]
    assert sum(w is not None for w in wins) == 1  # later ones see a live takeover


def test_heartbeat_keeps_claim_alive(tmp_path: Path) -> None:
    fwd, _ = make(tmp_path, sleep=1.0)
    entry = fwd.entry_dir(256, THETA, (1, 7))
    run_pool(tasks_for([256]), fwd, 1, heartbeat_interval=0.1, tick=0.05, log=quiet)
    hb = entry / "claim.heartbeat"
    assert hb.exists()
    assert hb.stat().st_mtime > (entry / "claim.json").stat().st_mtime + 0.3


def test_failure_then_retry_succeeds(tmp_path: Path) -> None:
    fwd, _ = make(tmp_path, fail_times={512: 1})
    res = run_pool(tasks_for([256, 512]), fwd, 2, tick=0.05, log=quiet)
    assert res.status is PoolStatus.COMPLETE
    assert res.n_computed == 2 and res.n_failed == 0
    entry = fwd.entry_dir(512, THETA, (2, 7))
    assert len(list(entry.glob("failure_*.json"))) == 1


def test_persistent_failure_continues_and_reports(tmp_path: Path) -> None:
    fwd, _ = make(tmp_path, fail_times={512: 99})
    lines: list[str] = []
    res = run_pool(tasks_for([256, 512, 256]), fwd, 2, tick=0.05, log=lines.append)
    assert res.status is PoolStatus.FAILED
    assert res.exit_code == 1
    assert res.n_failed == 1 and res.n_computed == 2
    assert "N=512" in res.failures[0]
    # exactly one retry: two failure records
    entry = fwd.entry_dir(512, THETA, (2, 7))
    assert len(list(entry.glob("failure_*.json"))) == 2
    assert any("pool summary" in ln and "failed=1" in ln for ln in lines)
