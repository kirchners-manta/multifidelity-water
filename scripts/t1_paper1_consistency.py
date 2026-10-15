#!/usr/bin/env python3
"""
T1: exact consistency check of the paper-I HDF5 files with ``mfwater.mfmc_core``.

The script is a standalone tool and not part of the package. It walks a directory tree
of paper-I files (default ``/Users/froembge/phd/mfmc_water``) and only ever opens HDF5
files in read-only mode. No molecular dynamics is involved.

For every calibration file (``m<eta>_e<m>_<c|o>_<lj|q>.hdf5``) it recomputes mean, std
and correlation from ``diffusion_coeff`` and compares them with the stored attributes,
``data/corr_*.csv`` (4 digits), ``data/oms_*.csv`` (selected models) and, for the
OrthoBoXY runs, SI Table S1. For every ``oms_*/..._oms.hdf5`` it maps the models back to
the calibration via ``n_molecules`` and recomputes the allocation ``m*``, the control
variates ``alpha``, the MFMC estimator and its RMSE from the ``*_initial`` data and the
new ``diffusion_coeff``.

It also emulates the legacy ``eval-estimator`` pairing (standard deviations in model-name
order, correlations in rho order; ``algo_eval_estim/eval_estim.py`` line 97) to decide
whether that bug changed any stored ``alpha`` or ``mfmc_estimator``.

Usage
-----
``python scripts/t1_paper1_consistency.py [ROOT]``

Exit status is 1 if any hard mismatch is found, else 0.

Notes
-----
The paper-I files store ``mean``, ``std`` and ``correlation`` as float32 and the legacy
code did its arithmetic partly in float32. The estimator is therefore reproduced in two
ways: bitwise with a float32 emulation of the legacy arithmetic (compared at
``rtol = 1e-10``) and with the float64 ``mfmc_core`` (compared at ``rtol = 1e-5``, the
float32 resolution of the data).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from mfwater import mfmc_core as core

DEFAULT_ROOT = Path("/Users/froembge/phd/mfmc_water")

# rtol for quantities that are exactly reproducible (float64 arithmetic on stored values)
RTOL_EXACT = 1e-10
# rtol for float64 recomputation compared with float32-stored attributes / estimators
RTOL_F32 = 1e-5

# SI Table S1 (OrthoBoXY runs): correlation coefficients, 4 digits, models 1..6.
SI_TABLE_S1: dict[tuple[str, int], list[float]] = {
    ("lj", 10): [1.0, 0.9476, 0.9408, 0.7946, 0.8833, 0.8460],
    ("lj", 100): [1.0, 0.8763, 0.8600, 0.8210, 0.8284, 0.7902],
    ("lj", 1000): [1.0, 0.8940, 0.8753, 0.8478, 0.8271, 0.7227],
    ("q", 10): [1.0, 0.9921, 0.9590, 0.9650, 0.9647, 0.8922],
    ("q", 100): [1.0, 0.9831, 0.9876, 0.9768, 0.9744, 0.9408],
    ("q", 1000): [1.0, 0.9824, 0.9795, 0.9732, 0.9644, 0.9420],
}

CAL_RE = re.compile(r"^m(\d+)_e(\d+)_([co])_(lj|q)\.hdf5$")


@dataclass
class Check:
    """
    One comparison between a stored and a recomputed quantity.

    Parameters
    ----------
    file : str
        File the quantity belongs to (relative to the root).
    what : str
        Description of the quantity.
    deviation : float
        Maximum relative (or, for integers, absolute) deviation.
    tol : float
        Tolerance for ``deviation``.
    note : str
        Optional remark.
    """

    file: str
    what: str
    deviation: float
    tol: float
    note: str = ""

    @property
    def ok(self) -> bool:
        """
        Whether the deviation is within the tolerance.

        Returns
        -------
        bool
            True if ``deviation <= tol``.
        """
        return self.deviation <= self.tol


@dataclass
class Models:
    """
    Models of one HDF5 file in model order (index 0 = highest ``n_molecules``).

    Parameters
    ----------
    names : list of str
        Group names in model order.
    n_mol : numpy.ndarray
        Molecules per model.
    attrs : dict
        Attributes as float64 arrays by name (``mean``, ``std``, ``correlation``,
        ``computation_time``, ``alpha``, ``n_evals``, ..., where present).
    datasets : dict
        Raw (native dtype) data sets as lists per model by name.
    root_attrs : dict
        Attributes of the ``models`` group.
    name_order_ok : bool
        True if the h5py group names sorted numerically are in descending
        ``n_molecules`` order.
    """

    names: list[str]
    n_mol: npt.NDArray[np.int64]
    attrs: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    datasets: dict[str, list[npt.NDArray[Any]]] = field(default_factory=dict)
    root_attrs: dict[str, Any] = field(default_factory=dict)
    name_order_ok: bool = True


def load_models(path: Path) -> Models:
    """
    Read the model groups of a paper-I file (read-only).

    Models are sorted by the numeric suffix of the group name (not h5py name order, which
    would put ``model_10`` before ``model_2``).

    Parameters
    ----------
    path : pathlib.Path
        HDF5 file.

    Returns
    -------
    Models
        The models in name order, with ``name_order_ok`` telling whether this is also the
        descending ``n_molecules`` order.
    """
    with h5py.File(path, "r") as f:
        grp = f["models"]
        names = sorted(
            (k for k, v in grp.items() if isinstance(v, h5py.Group)),
            key=lambda s: int(s.split("_")[1]),
        )
        n_mol = np.array([int(grp[n].attrs["n_molecules"]) for n in names])
        out = Models(names=names, n_mol=n_mol, root_attrs=dict(grp.attrs))
        out.name_order_ok = bool(np.all(np.diff(n_mol) < 0))
        keys = set().union(*(grp[n].attrs.keys() for n in names)) - {"n_molecules"}
        for k in sorted(keys):
            out.attrs[k] = np.array(
                [float(grp[n].attrs[k]) for n in names], dtype=np.float64
            )
        for dname in ("diffusion_coeff", "diffusion_coeff_initial"):
            if dname in grp[names[0]]:
                out.datasets[dname] = [np.array(grp[n][dname][:]) for n in names]
        for top in ("lj_params", "charges", "seeds", "lj_params_initial"):
            if top in grp:
                out.datasets[top] = [np.array(grp[top][:])]
            if f"{top}" not in grp and top.endswith("_initial"):
                continue
        for top in ("charges_initial", "seeds_initial"):
            if top in grp:
                out.datasets[top] = [np.array(grp[top][:])]
    return out


def rel_dev(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """
    Compute the maximum elementwise relative deviation ``|a - b| / |b|``.

    Parameters
    ----------
    a : array_like
        Recomputed values.
    b : array_like
        Reference (stored) values.

    Returns
    -------
    float
        The maximum relative deviation (absolute deviation where ``b == 0``).
    """
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    denom = np.where(b_ == 0.0, 1.0, np.abs(b_))
    return float(np.max(np.abs(a_ - b_) / denom))


def read_csv_matrix(path: Path) -> dict[int, list[float]]:
    """
    Read a ``model_idx,10,100,1000`` csv from ``data/`` into columns by evaluation count.

    Parameters
    ----------
    path : pathlib.Path
        csv file.

    Returns
    -------
    dict
        Maps the number of evaluations (10, 100, 1000) to the column over models 1..6.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [int(x) for x in rows[0][1:]]
    return {e: [float(r[1 + i]) for r in rows[1:]] for i, e in enumerate(header)}


def check_calibration(
    root: Path, path: Path, cal: Models
) -> tuple[list[Check], dict[str, Any]]:
    """
    Compare a calibration file with recomputed statistics and published tables.

    Parameters
    ----------
    root : pathlib.Path
        Root of the paper-I tree (for ``data/*.csv``).
    path : pathlib.Path
        The calibration file.
    cal : Models
        Its models.

    Returns
    -------
    checks : list of Check
        All comparisons.
    info : dict
        Selection, gamma and summary values for the table.
    """
    rel = str(path.relative_to(root))
    mt = CAL_RE.match(path.name)
    assert mt is not None
    _, e_s, box, par = mt.groups()
    e = int(e_s)
    checks: list[Check] = []

    if not cal.name_order_ok:
        checks.append(Check(rel, "model names in descending n_molecules order", 1, 0))

    D32 = np.array(cal.datasets["diffusion_coeff"])
    mean, std, rho = core.model_statistics(D32)
    checks.append(
        Check(rel, "mean vs attrs", rel_dev(mean, cal.attrs["mean"]), RTOL_F32)
    )
    checks.append(Check(rel, "std vs attrs", rel_dev(std, cal.attrs["std"]), RTOL_F32))
    checks.append(
        Check(rel, "rho vs attrs", rel_dev(rho, cal.attrs["correlation"]), RTOL_F32)
    )

    # data/corr_*.csv holds 4-digit values
    csv_corr = root / "data" / f"corr_{box}_{par}.csv"
    if csv_corr.exists():
        col = np.array(read_csv_matrix(csv_corr)[e])
        dev = float(np.max(np.abs(np.round(cal.attrs["correlation"], 4) - col)))
        checks.append(Check(rel, f"rho vs {csv_corr.name} (abs, 4 digits)", dev, 1e-4))
    if box == "o":
        ref = np.array(SI_TABLE_S1[(par, e)])
        dev = float(np.max(np.abs(np.round(cal.attrs["correlation"], 4) - ref)))
        checks.append(Check(rel, "rho vs SI Table S1 (abs, 4 digits)", dev, 1e-4))

    # model selection from stored attrs (what the legacy code saw)
    sigma_a = cal.attrs["std"]
    rho_a = cal.attrs["correlation"]
    w = cal.attrs["computation_time"]
    sel, v_star = core.select_models(sigma_a[0], rho_a, w)
    gam = core.gamma(rho_a[list(sel)], w[list(sel)])
    sel_re, _ = core.select_models(std[0], rho, w)
    if sel_re != sel:
        checks.append(
            Check(rel, f"selection from recomputed stats {sel_re} != attrs {sel}", 1, 0)
        )
    csv_oms = root / "data" / f"oms_{box}_{par}.csv"
    if csv_oms.exists():
        col_o = np.array(read_csv_matrix(csv_oms)[e])
        mask = np.zeros(len(col_o))
        mask[list(sel)] = 1
        checks.append(
            Check(
                rel,
                f"selection vs {csv_oms.name}",
                float(np.max(np.abs(mask - col_o))),
                0,
            )
        )
    non_monotone = bool(np.any(np.diff(rho_a**2) > 0))
    info = {
        "eta": len(cal.names),
        "m": D32.shape[1],
        "sel": sel,
        "gamma": gam,
        "non_monotone_rho": non_monotone,
        "rho": rho_a,
    }
    return checks, info


def legacy_alpha(rho_rho_order: np.ndarray, std_name_order: np.ndarray) -> np.ndarray:
    """
    Emulate the alpha computation of ``eval_estim.py`` line 97.

    Parameters
    ----------
    rho_rho_order : numpy.ndarray
        Correlations ordered by decreasing ``rho**2`` (``ordered_models``).
    std_name_order : numpy.ndarray
        Standard deviations in model-name order (``model_items``).

    Returns
    -------
    numpy.ndarray
        The (possibly mispaired) alpha as the legacy code computed it.
    """
    return (rho_rho_order / std_name_order) * std_name_order[0]


def legacy_estimator_f32(
    diffs: list[np.ndarray],
    evals: np.ndarray,
    alphas: np.ndarray,
    means_stored: np.ndarray,
) -> float:
    """
    Emulate the estimator arithmetic of ``algo_mfmc/mfmc.py`` incl. float32 means.

    The legacy code wrote the float32 ``np.mean`` of each model as attribute ``mean``.
    The full-array mean is taken from that attribute: ``np.mean`` of a float32 array with
    thousands of entries depends on the NumPy version (summation order), so recomputing it
    today differs from the stored value at the float32 rounding level. The lower means
    (first ``n_evals[i-1]`` samples) are recomputed.

    Parameters
    ----------
    diffs : list of numpy.ndarray
        Stored (float32) ``diffusion_coeff`` per model in rho order.
    evals : numpy.ndarray
        ``n_evals`` per model in rho order.
    alphas : numpy.ndarray
        Stored ``alpha`` in rho order.
    means_stored : numpy.ndarray
        Stored ``mean`` attributes in rho order (float32 precision).

    Returns
    -------
    float
        The estimator as the legacy code computed it.
    """
    means = means_stored.astype(np.float32)
    means_lower = np.array(
        [np.mean(diffs[i][: int(evals[i - 1])]) for i in range(1, len(diffs))]
    )
    return float(means[0] + np.sum(alphas[1:] * (means[1:] - means_lower)))


def legacy_rmse(std1: float, rho: np.ndarray, w: np.ndarray, budget: float) -> float:
    """
    Emulate the error formula of ``algo_mfmc/mfmc.py`` incl. float32 ``std_initial**2``.

    Parameters
    ----------
    std1 : float
        Stored (float32) ``std_initial`` of the high-fidelity model.
    rho : numpy.ndarray
        Correlations in rho order, ``rho[0] = 1``.
    w : numpy.ndarray
        Costs in rho order.
    budget : float
        Budget.

    Returns
    -------
    float
        The stored-style RMSE; the float32 square of ``std1`` is the only difference from
        :func:`mfwater.mfmc_core.mfmc_rmse`.
    """
    r2 = np.concatenate((rho**2, [0.0]))
    diff = r2[:-1] - r2[1:]
    return float(
        np.sqrt(
            np.float32(std1) ** 2 / np.float64(budget) * np.sum(np.sqrt(w * diff)) ** 2
        )
    )


def check_performance(
    root: Path, path: Path, cal: Models, cal_info: dict[str, Any], perf: Models
) -> tuple[list[Check], dict[str, Any]]:
    """
    Recompute m*, alpha, estimator and RMSE of an ``oms`` file and compare.

    Parameters
    ----------
    root : pathlib.Path
        Root of the tree.
    path : pathlib.Path
        The performance (``oms``) file.
    cal : Models
        Models of the parent calibration file.
    cal_info : dict
        Output info of :func:`check_calibration` for the parent.
    perf : Models
        Models of the performance file.

    Returns
    -------
    checks : list of Check
        All comparisons.
    info : dict
        Values for the table, including the alpha-bug analysis.
    """
    rel = str(path.relative_to(root))
    checks: list[Check] = []
    budget = float(perf.root_attrs["budget"])

    # map performance models to calibration models via n_molecules
    idx_in_cal = []
    for n in perf.n_mol:
        hits = np.flatnonzero(cal.n_mol == n)
        if hits.size != 1:
            checks.append(
                Check(rel, f"n_molecules {n} not unique in calibration", 1, 0)
            )
            return checks, {}
        idx_in_cal.append(int(hits[0]))
    cidx = np.array(idx_in_cal)

    # consistency of the *_initial copies with the calibration file
    for k, name in enumerate(perf.names):
        c = int(cidx[k])
        d0 = perf.datasets["diffusion_coeff_initial"][k]
        d1 = cal.datasets["diffusion_coeff"][c]
        checks.append(
            Check(
                rel,
                f"{name}: diffusion_coeff_initial == calibration",
                rel_dev(d0, d1),
                0,
            )
        )
    for attr_p, attr_c in (
        ("std_initial", "std"),
        ("mean_initial", "mean"),
        ("correlation", "correlation"),
    ):
        checks.append(
            Check(
                rel,
                f"{attr_p} == calibration {attr_c}",
                rel_dev(perf.attrs[attr_p], cal.attrs[attr_c][cidx]),
                0,
            )
        )
    for top in ("lj_params", "seeds", "charges"):
        if f"{top}_initial" in perf.datasets and top in cal.datasets:
            checks.append(
                Check(
                    rel,
                    f"{top}_initial == calibration {top}",
                    rel_dev(perf.datasets[f"{top}_initial"][0], cal.datasets[top][0]),
                    0,
                )
            )

    # selection reproduces the models in the file
    sel = cal_info["sel"]
    if set(cal.n_mol[list(sel)]) != set(perf.n_mol):
        checks.append(
            Check(
                rel,
                f"selected models {cal.n_mol[list(sel)].tolist()} != file "
                f"{perf.n_mol.tolist()}",
                1,
                0,
            )
        )

    # --- rho order of the selected models in this file
    rho_p = perf.attrs["correlation"]  # name order (== descending N)
    std_p = perf.attrs["std_initial"]
    order = sorted(range(len(rho_p)), key=lambda i: (-(rho_p[i] ** 2), i))
    order_arr = np.array(order)
    reordered = order != sorted(order)

    # inputs of the legacy eval-estimator: attrs of the file at that time. Costs for m*
    # were the calibration costs (the performance run later overwrote computation_time).
    w_cal = cal.attrs["computation_time"][cidx]
    m_expected, unspent = core.allocate(rho_p[order_arr], w_cal[order_arr], budget)
    m_stored = perf.attrs["n_evals"][order_arr].astype(np.int64)
    checks.append(
        Check(
            rel,
            f"m* (attrs) {m_expected.tolist()} vs stored {m_stored.tolist()}",
            float(np.max(np.abs(m_expected - m_stored))),
            0,
        )
    )
    # the same from statistics recomputed from the *_initial data
    D_sel = np.array(
        [perf.datasets["diffusion_coeff_initial"][i] for i in range(len(rho_p))]
    )
    _, std_r, rho_r = core.model_statistics(D_sel)
    try:
        m_re, _ = core.allocate(rho_r[order_arr], w_cal[order_arr], budget)
        dev_m = float(np.max(np.abs(m_re - m_stored)))
    except ValueError as exc:  # pragma: no cover - only on pathological data
        dev_m = 1.0
        print(f"  warning: {rel}: allocate from recomputed stats failed: {exc}")
    checks.append(Check(rel, "m* (recomputed from D_initial) vs stored", dev_m, 0))

    # --- alpha: correct pairing versus legacy pairing versus stored
    alpha_correct = core.control_variates(rho_p[order_arr], std_p[order_arr])
    alpha_legacy = legacy_alpha(rho_p[order_arr], std_p)  # std in name order (bug)
    alpha_stored = perf.attrs["alpha"][order_arr]
    dev_correct = rel_dev(alpha_stored, alpha_correct)
    dev_legacy = rel_dev(alpha_stored, alpha_legacy)
    checks.append(
        Check(
            rel, "alpha stored vs legacy-paired recomputation", dev_legacy, RTOL_EXACT
        )
    )
    bug_acted = rel_dev(alpha_legacy, alpha_correct) > RTOL_EXACT
    if not bug_acted:
        checks.append(
            Check(rel, "alpha stored vs correctly paired", dev_correct, RTOL_EXACT)
        )

    # --- estimator
    diffs = [perf.datasets["diffusion_coeff"][i] for i in order]
    evals = perf.attrs["n_evals"][order_arr]
    if any(len(d) != int(m) for d, m in zip(diffs, evals)):
        checks.append(Check(rel, "len(diffusion_coeff) == n_evals", 1, 0))
    est_stored = float(perf.root_attrs["mfmc_estimator"])
    est_f32 = legacy_estimator_f32(
        diffs, evals, alpha_stored, perf.attrs["mean"][order_arr]
    )
    mean_re = np.array([np.mean(d) for d in diffs])
    checks.append(
        Check(
            rel,
            "mean attr vs np.mean(float32) (summation order)",
            rel_dev(mean_re, perf.attrs["mean"][order_arr]),
            RTOL_F32,
        )
    )
    checks.append(
        Check(
            rel,
            "estimator (legacy f32 emulation, stored alpha)",
            rel_dev(est_f32, est_stored),
            RTOL_EXACT,
        )
    )
    est_core = core.mfmc_estimate(diffs, m_stored, alpha_stored)
    checks.append(
        Check(
            rel,
            "estimator (mfmc_core f64, stored alpha)",
            rel_dev(est_core, est_stored),
            RTOL_F32,
        )
    )
    est_corrected = core.mfmc_estimate(diffs, m_stored, alpha_correct)
    dev_est_bug = rel_dev(est_corrected, est_stored)

    # --- RMSE: legacy used the computation_time stored in the oms file at mfmc time
    # (performance-run costs), whereas the allocation used calibration costs.
    err_stored = float(perf.root_attrs["mfmc_error"])
    rho_o = rho_p[order_arr]
    w_perf = perf.attrs["computation_time"][order_arr]
    err_perf = legacy_rmse(float(std_p[0]), rho_o, w_perf, budget)
    err_cal = legacy_rmse(float(std_p[0]), rho_o, w_cal[order_arr], budget)
    err_core = core.mfmc_rmse(float(std_p[0]), rho_o, w_perf, budget)
    checks.append(
        Check(
            rel,
            "RMSE (legacy f32 emulation, performance-run costs) vs stored",
            rel_dev(err_perf, err_stored),
            RTOL_EXACT,
        )
    )
    checks.append(
        Check(
            rel,
            "RMSE (mfmc_core f64, performance-run costs) vs stored",
            rel_dev(err_core, err_stored),
            RTOL_F32,
        )
    )
    info = {
        "budget": budget,
        "n_models": len(rho_p),
        "reordered": reordered,
        "m": m_expected.tolist(),
        "unspent": unspent,
        "bug_acted": bug_acted,
        "alpha_dev_correct": dev_correct,
        "alpha_dev_legacy": dev_legacy,
        "est_dev_if_corrected": dev_est_bug,
        "est_stored": est_stored,
        "est_corrected": est_corrected,
        "err_dev_cal_costs": rel_dev(err_cal, err_stored),
        "err_dev_perf_costs": rel_dev(err_perf, err_stored),
        "alpha_stored": alpha_stored,
        "alpha_correct": alpha_correct,
        "rho_order_names": [perf.names[i] for i in order],
    }
    return checks, info


def main(argv: list[str] | None = None) -> int:
    """
    Run all T1 checks under a root directory and print table and mismatches.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        1 if any hard mismatch was found, else 0.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    root: Path = args.root.resolve()

    all_checks: list[Check] = []
    cal_rows: list[str] = []
    perf_rows: list[str] = []
    bug_rows: list[str] = []
    n_cal = n_perf = 0

    cal_files = sorted(p for p in root.rglob("*.hdf5") if CAL_RE.match(p.name))
    for cpath in cal_files:
        cal = load_models(cpath)
        checks, cinfo = check_calibration(root, cpath, cal)
        all_checks += checks
        n_cal += 1
        cal_rows.append(
            f"{cpath.stem:<14} eta={cinfo['eta']} m={cinfo['m']:<5d} "
            f"selected(N)={cal.n_mol[list(cinfo['sel'])].tolist()} "
            f"gamma={cinfo['gamma']:.4f} "
            f"non-monotone rho={'yes' if cinfo['non_monotone_rho'] else 'no'} "
            f"max dev: {max((c.deviation for c in checks[:3]), default=0):.1e}"
        )
        for ppath in sorted(cpath.parent.glob("oms_*/*_oms.hdf5")):
            perf = load_models(ppath)
            pchecks, pinfo = check_performance(root, ppath, cal, cinfo, perf)
            all_checks += pchecks
            n_perf += 1
            if not pinfo:
                continue
            bad = [c for c in pchecks if not c.ok]
            perf_rows.append(
                f"{str(ppath.parent.relative_to(root)):<26} p={pinfo['budget']:.0e} "
                f"models={pinfo['n_models']} m*={pinfo['m']} "
                f"rho-order!=name-order={'yes' if pinfo['reordered'] else 'no':<3} "
                f"alpha dev(correct)={pinfo['alpha_dev_correct']:.1e} "
                f"alpha dev(legacy)={pinfo['alpha_dev_legacy']:.1e} "
                f"err dev(perf w)={pinfo['err_dev_perf_costs']:.1e} "
                f"err dev(cal w)={pinfo['err_dev_cal_costs']:.1e} "
                f"{'MISMATCH' if bad else 'ok'}"
            )
            if pinfo["reordered"] or pinfo["bug_acted"]:
                bug_rows.append(
                    f"{ppath.relative_to(root)}: rho order {pinfo['rho_order_names']}; "
                    f"bug changes alpha: {pinfo['bug_acted']}; "
                    f"stored alpha {np.round(pinfo['alpha_stored'], 6).tolist()} "
                    f"correct {np.round(pinfo['alpha_correct'], 6).tolist()}; "
                    f"estimator stored {pinfo['est_stored']:.6f} "
                    f"corrected {pinfo['est_corrected']:.6f} "
                    f"(rel {pinfo['est_dev_if_corrected']:.2e})"
                )

    print(f"Root: {root}\n{n_cal} calibration files, {n_perf} performance files\n")
    print("== Calibration files ==")
    print("\n".join(cal_rows))
    print("\n== Performance (oms) files ==")
    print("\n".join(perf_rows))

    print("\n== Alpha pairing bug (eval_estim.py:97) ==")
    if not bug_rows:
        print(
            "In no performance file does the rho order of the selected models differ "
            "from the model-name order: the bug did not act; no stored alpha or "
            "mfmc_estimator is affected."
        )
    else:
        print("\n".join(bug_rows))
    changed = [r for r in bug_rows if "bug changes alpha: True" in r]
    print(f"Files in which the bug changes alpha: {len(changed)} of {n_perf}.")

    mismatches = [c for c in all_checks if not c.ok]
    print(f"\n== Mismatches ({len(mismatches)} of {len(all_checks)} checks) ==")
    for c in mismatches:
        print(
            f"{c.file}: {c.what}: deviation {c.deviation:.3e} > tol {c.tol:.1e} {c.note}"
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
