"""Evaluation pool: run many MD evaluations concurrently, across SLURM jobs.

:func:`run_pool` packs evaluations (tasks) onto the cores of one allocation and
lets several jobs work on the same cache directory without computing a task
twice.

Claiming protocol (per evaluation directory ``<cache>/N{n}/{key_hash}/``)::

    result.json                    task done (written by CachedForwardModel)
    claim.json                     first owner, created with O_CREAT | O_EXCL
    claim.heartbeat                touched every ``heartbeat_interval`` seconds
    claim_takeover_<k>.json        owner k after a stale claim (O_EXCL as well)
    claim_takeover_<k>.heartbeat   its heartbeat
    failure_<time>_<pid>.json      traceback of a failed attempt

* The current owner is the claim file with the highest ``k``. Its liveness is
  the mtime of its heartbeat file (or of the claim file before the first
  heartbeat). A claim is *stale* if that is older than ``stale_timeout``.
* A stale claim is taken over by creating ``claim_takeover_<k+1>.json`` with
  ``O_EXCL``; of several competing jobs exactly one succeeds. Old claims and
  attempt directories are never modified or deleted, so the MD pipeline resumes
  from restart files of earlier attempts.
* Claims are never released. A finished task is recognised by ``result.json``;
  a task that failed in this call keeps its claim, which becomes stale after
  ``stale_timeout`` once the heartbeat stops.
* A job that is resubmitted after a crash has to wait ``stale_timeout`` for its
  own old claims, unless the timeout is lowered.

Assumptions that are NOT verified:

* ``O_CREAT | O_EXCL`` is atomic between processes on *different nodes* of the
  shared (Lustre) file system. This is believed to hold on modern Lustre but has
  not been tested here; test it on the cluster before relying on it.
* Clocks of all nodes and the file system agree to well within
  ``stale_timeout`` (mtimes are compared with ``time.time()`` of the checking
  node).
* A "stale" owner that is in fact alive (e.g. a hung node) is not stopped, so
  both may compute the same task. The result file is written atomically, so
  the cache stays consistent.

Workers are separate processes (``spawn`` start method, one executor per task so
that a killed worker cannot break the others). They rely on ``result.json`` on
disk; the in-memory memo of :class:`~mfwater.md.cache.CachedForwardModel` is not
shared. ``forward`` and the tasks must therefore be picklable.
"""

from __future__ import annotations

import enum
import json
import multiprocessing
import os
import socket
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cache import CachedForwardModel

#: Default time without heartbeat after which a claim counts as stale (seconds).
DEFAULT_STALE_TIMEOUT = 1800.0
#: Default interval between heartbeat updates (seconds).
DEFAULT_HEARTBEAT_INTERVAL = 120.0


class PoolStatus(enum.Enum):
    """Outcome of :func:`run_pool`."""

    #: All tasks have a result.
    COMPLETE = "complete"
    #: Remaining tasks are claimed by live claims of other jobs.
    WAITING_ON_OTHERS = "waiting_on_others"
    #: At least one task failed (also after the retry).
    FAILED = "failed"


@dataclass(eq=False)
class Task:
    """One MD evaluation.

    The protocol and box shape are properties of the forward model passed to
    :func:`run_pool`, not of the task.

    Parameters
    ----------
    n_molecules : int
        Number of molecules (fidelity level).
    theta : np.ndarray
        Parameters ``(epsilon_OO, sigma_OO, q_O)``.
    seeds : tuple[int, int] | None, optional
        Explicit ``(packmol_seed, velocity_seed)``.
    """

    n_molecules: int
    theta: np.ndarray
    seeds: tuple[int, int] | None = None


@dataclass
class PoolResult:
    """Summary returned by :func:`run_pool`.

    Attributes
    ----------
    status : PoolStatus
        Overall outcome.
    n_cached : int
        Tasks that already had a result.
    n_computed : int
        Tasks computed by this call.
    n_waiting : int
        Tasks left that are claimed by live claims of other jobs.
    n_failed : int
        Tasks that failed in this call.
    failures : list[str]
        One message per failed task.
    """

    status: PoolStatus
    n_cached: int = 0
    n_computed: int = 0
    n_waiting: int = 0
    n_failed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """Return 1 if any task failed, else 0 (waiting is not an error)."""
        return 1 if self.status is PoolStatus.FAILED else 0


# ---------------------------------------------------------------------------
# claiming
# ---------------------------------------------------------------------------


def _create_exclusive(path: Path) -> bool:
    """Create a claim file atomically and write the owner information.

    Parameters
    ----------
    path : Path
        File to create; it must not exist.

    Returns
    -------
    bool
        True if this call created the file, False if it already existed.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    info = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pid": os.getpid(),
        "start_time": time.time(),
    }
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return True


def _owner_files(entry: Path) -> tuple[Path, int] | None:
    """Return the file of the current owner and its index ``k``.

    Parameters
    ----------
    entry : Path
        Evaluation directory.

    Returns
    -------
    tuple[Path, int] | None
        ``(claim file, k)`` with ``k = 0`` for ``claim.json``; None if there
        is no claim.
    """
    best = -1
    for p in entry.glob("claim_takeover_*.json"):
        try:
            best = max(best, int(p.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    if best >= 1:
        return entry / f"claim_takeover_{best}.json", best
    if (entry / "claim.json").exists():
        return entry / "claim.json", 0
    return None


def _heartbeat_path(claim_file: Path) -> Path:
    """Return the heartbeat file belonging to a claim file."""
    return claim_file.with_suffix(".heartbeat")


def _claim_age(claim_file: Path) -> float:
    """Return seconds since the last sign of life of a claim.

    Parameters
    ----------
    claim_file : Path
        Claim file of the owner.

    Returns
    -------
    float
        Age of the heartbeat file, or of the claim file if there is no
        heartbeat yet. 0 if the files vanished (treated as live).
    """
    for p in (_heartbeat_path(claim_file), claim_file):
        try:
            return time.time() - p.stat().st_mtime
        except FileNotFoundError:
            continue
    return 0.0


def try_claim(entry: Path, stale_timeout: float) -> Path | None:
    """Try to become the owner of an evaluation directory.

    Parameters
    ----------
    entry : Path
        Evaluation directory (created if missing).
    stale_timeout : float
        Seconds without heartbeat after which a claim may be taken over.

    Returns
    -------
    Path | None
        The claim file of the new owner (``claim.json`` or
        ``claim_takeover_<k>.json``), or None if a live claim exists or another
        job won the race.
    """
    entry.mkdir(parents=True, exist_ok=True)
    owner = _owner_files(entry)
    if owner is None:
        claim = entry / "claim.json"
        if _create_exclusive(claim):
            _heartbeat_path(claim).touch()
            return claim
        owner = _owner_files(entry)
        if owner is None:  # pragma: no cover - file removed by hand
            return None
    claim_file, k = owner
    if _claim_age(claim_file) <= stale_timeout:
        return None
    # Stale: exactly one job can create takeover number k+1.
    new = entry / f"claim_takeover_{k + 1}.json"
    if _create_exclusive(new):
        _heartbeat_path(new).touch()
        return new
    return None


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def _worker(
    forward: CachedForwardModel,
    n_molecules: int,
    theta: np.ndarray,
    seeds: tuple[int, int] | None,
) -> tuple[float, bool]:
    """Evaluate one task in a worker process.

    Parameters
    ----------
    forward : CachedForwardModel
        Cached forward model (a pickled copy; it reads ``result.json``).
    n_molecules : int
        Number of molecules.
    theta : np.ndarray
        Parameters ``(epsilon_OO, sigma_OO, q_O)``.
    seeds : tuple[int, int] | None
        Explicit seeds.

    Returns
    -------
    tuple[float, bool]
        ``(D, computed)``; ``computed`` is False if the result appeared on
        disk in the meantime.
    """
    d = forward(n_molecules, theta, seeds=seeds)
    return d, forward.n_computed > 0


@dataclass(eq=False)
class _State:
    """Bookkeeping of one task inside :func:`run_pool`."""

    task: Task
    entry: Path
    ncpu: int
    label: str
    claim: Path | None = None
    attempts: int = 0
    start: float = 0.0
    executor: ProcessPoolExecutor | None = None


def _record_failure(entry: Path, label: str, exc: BaseException) -> None:
    """Write the traceback of a failed attempt to a new file in ``entry``.

    Parameters
    ----------
    entry : Path
        Evaluation directory.
    label : str
        Task description.
    exc : BaseException
        The exception of the worker.
    """
    stem = f"failure_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    path = entry / f"{stem}.json"
    n = 0
    while path.exists():  # never overwrite an earlier record
        n += 1
        path = entry / f"{stem}_{n}.json"
    text = "".join(traceback.format_exception(exc))
    path.write_text(json.dumps({"task": label, "traceback": text}, indent=2))


def run_pool(
    tasks: Sequence[Task],
    forward: CachedForwardModel,
    total_cpus: int,
    *,
    ncpu_of: Callable[[int], int] | None = None,
    stale_timeout: float = DEFAULT_STALE_TIMEOUT,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    poll_timeout: float = 0.0,
    poll_interval: float = 10.0,
    tick: float = 1.0,
    max_retries: int = 1,
    log: Callable[[str], None] | None = None,
) -> PoolResult:
    """Run tasks concurrently on ``total_cpus`` cores, sharing work across jobs.

    Tasks are started largest ``N`` first whenever ``ncpu(N)`` cores are free
    (greedy packing; smaller tasks may fill gaps). Cores are never
    oversubscribed. Before a task starts it is claimed (see the module
    docstring); tasks with ``result.json`` are skipped, tasks with a live claim
    of another job are left alone.

    ``O_EXCL`` atomicity across Lustre nodes is assumed, not verified (see the
    module docstring).

    Parameters
    ----------
    tasks : Sequence[Task]
        Evaluations to make sure exist in the cache. Duplicates are merged.
    forward : CachedForwardModel
        Cached forward model; must be picklable. Its wrapped model should have
        ``n_mpi(n)`` (as :class:`~mfwater.md.pipeline.MDForwardModel`, which
        respects the ``ncpu`` override) unless ``ncpu_of`` is given. Use
        ``require_cost=True`` on the MD model if the cost is needed.
    total_cpus : int
        Cores available to this call.
    ncpu_of : Callable[[int], int] | None, optional
        Cores per task as a function of ``N``; overrides ``forward.model.n_mpi``.
        Without both, one core per task.
    stale_timeout : float, optional
        Seconds without heartbeat after which a claim may be taken over.
    heartbeat_interval : float, optional
        Seconds between heartbeat updates of the claims held by this call.
    poll_timeout : float, optional
        If only tasks with live claims of others remain, keep checking for this
        many seconds before returning ``WAITING_ON_OTHERS``.
    poll_interval : float, optional
        Seconds between checks while waiting on others.
    tick : float, optional
        Maximum seconds between scheduler iterations while tasks run.
    max_retries : int, optional
        Retries of a failed task within this call (default one).
    log : Callable[[str], None] | None, optional
        Receives one line per event; defaults to ``print`` with flushing.

    Returns
    -------
    PoolResult
        Status and counts. ``status`` is ``FAILED`` if any task failed,
        else ``WAITING_ON_OTHERS`` if tasks remain claimed by others, else
        ``COMPLETE``.

    Raises
    ------
    ValueError
        If a task needs more cores than ``total_cpus``.
    """
    if log is None:

        def log(msg: str) -> None:
            print(msg, flush=True)

    def ncpu_for(n: int) -> int:
        if ncpu_of is not None:
            return int(ncpu_of(n))
        n_mpi = getattr(forward.model, "n_mpi", None)
        return int(n_mpi(n)) if n_mpi is not None else 1

    # merge duplicates, largest N first (stable for equal N)
    states: list[_State] = []
    seen: set[Path] = set()
    for t in tasks:
        entry = forward.entry_dir(t.n_molecules, t.theta, t.seeds)
        if entry in seen:
            continue
        seen.add(entry)
        ncpu = ncpu_for(t.n_molecules)
        if ncpu > total_cpus:
            raise ValueError(
                f"Task N={t.n_molecules} needs {ncpu} cores, but only "
                f"{total_cpus} are available."
            )
        label = f"N={t.n_molecules} key={entry.name} ncpu={ncpu}"
        states.append(_State(t, entry, ncpu, label))
    states.sort(key=lambda s: -s.task.n_molecules)

    result = PoolResult(PoolStatus.COMPLETE)
    pending = list(states)
    running: dict[Future[tuple[float, bool]], _State] = {}
    free = total_cpus
    ctx = multiprocessing.get_context("spawn")
    last_beat = time.time()
    waiting_since: float | None = None

    def start(st: _State) -> None:
        nonlocal free
        free -= st.ncpu
        st.attempts += 1
        st.start = time.time()
        st.executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        fut = st.executor.submit(
            _worker, forward, st.task.n_molecules, st.task.theta, st.task.seeds
        )
        running[fut] = st
        log(f"start  {st.label} (attempt {st.attempts}, free cores {free})")

    def mark_cached(st: _State) -> None:
        pending.remove(st)
        result.n_cached += 1
        log(f"cached {st.label}")

    try:
        while pending or running:
            now = time.time()
            if now - last_beat >= heartbeat_interval:
                for st in running.values():
                    if st.claim is not None:
                        _heartbeat_path(st.claim).touch()
                last_beat = now

            # schedule: largest first, claim only if the task can start now
            for st in list(pending):
                if (st.entry / "result.json").exists():
                    mark_cached(st)
                    continue
                if st.ncpu > free:
                    continue
                if st.claim is None:
                    st.claim = try_claim(st.entry, stale_timeout)
                    if st.claim is None:
                        continue
                    if (st.entry / "result.json").exists():
                        # finished between the check and the claim
                        mark_cached(st)
                        continue
                    if st.claim.name != "claim.json":
                        log(f"takeover {st.label} via {st.claim.name}")
                pending.remove(st)
                start(st)

            if running:
                waiting_since = None
                done, _ = wait(
                    list(running), timeout=tick, return_when="FIRST_COMPLETED"
                )
                for fut in done:
                    st = running.pop(fut)
                    free += st.ncpu
                    assert st.executor is not None
                    st.executor.shutdown(wait=True)
                    wall = time.time() - st.start
                    exc = fut.exception()
                    if exc is None:
                        _, computed = fut.result()
                        if computed:
                            result.n_computed += 1
                        else:
                            result.n_cached += 1
                        log(f"finish {st.label} wall={wall:.1f}s computed={computed}")
                    else:
                        _record_failure(st.entry, st.label, exc)
                        log(f"FAILED {st.label} wall={wall:.1f}s: {exc!r}")
                        if st.attempts <= max_retries:
                            pending.insert(0, st)  # keeps its claim
                        else:
                            result.n_failed += 1
                            result.failures.append(f"{st.label}: {exc!r}")
                continue

            if not pending:
                break
            # Nothing running, so every remaining task fits (ncpu <= total_cpus)
            # and was refused only because of a live claim of another job.
            if waiting_since is None:
                waiting_since = time.time()
            if time.time() - waiting_since >= poll_timeout:
                result.n_waiting = len(pending)
                break
            time.sleep(min(poll_interval, max(poll_timeout, 0.01)))
    finally:
        for fut, st in running.items():  # only reached on error/interrupt
            fut.cancel()
            if st.executor is not None:
                st.executor.shutdown(wait=False, cancel_futures=True)

    if result.n_failed:
        result.status = PoolStatus.FAILED
    elif result.n_waiting:
        result.status = PoolStatus.WAITING_ON_OTHERS
    log(
        f"pool summary: cached={result.n_cached} computed={result.n_computed} "
        f"waiting={result.n_waiting} failed={result.n_failed} "
        f"status={result.status.value}"
    )
    for msg in result.failures:
        log(f"  failure: {msg}")
    return result


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_STALE_TIMEOUT",
    "PoolResult",
    "PoolStatus",
    "Task",
    "run_pool",
    "try_claim",
]
