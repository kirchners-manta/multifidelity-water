"""
T1: regression of ``mfwater.mfmc_core`` against the stored paper-I results.

The fixtures in ``test/data/paper1`` are small, read-only copies of paper-I HDF5 files
(same directory layout as the original tree, so that the checks of
``scripts/t1_paper1_consistency.py`` run on them unchanged). The stored values are the
expected values. The legacy alpha-pairing bug (``eval_estim.py:97``) was found not to have
acted in *any* paper-I file (the selected models are in the same order by name and by
rho in every ``oms`` file), so no stored value needed to be replaced by a corrected one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

from mfwater import mfmc_core as core

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "paper1"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "t1_paper1_consistency.py"


@pytest.fixture(scope="module")
def t1() -> ModuleType:
    """
    Import the T1 script as a module.

    Returns
    -------
    types.ModuleType
        The loaded ``t1_paper1_consistency`` module.
    """
    spec = importlib.util.spec_from_file_location("t1_paper1_consistency", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses needs the module in sys.modules
    spec.loader.exec_module(mod)
    return mod


def cal_files() -> list[Path]:
    """
    List the calibration fixtures.

    Returns
    -------
    list of pathlib.Path
        Calibration files (``m6_e*_*.hdf5`` directly below a run directory).
    """
    return sorted(p for p in FIXTURES.glob("m6_*/m6_*.hdf5"))


def perf_files() -> list[Path]:
    """
    List the performance (``oms``) fixtures.

    Returns
    -------
    list of pathlib.Path
        The ``*_oms.hdf5`` files.
    """
    return sorted(FIXTURES.glob("m6_*/oms_*/*_oms.hdf5"))


def test_fixtures_present_and_small() -> None:
    """The fixtures exist, cover c/o and lj/q, and each file is below 100 kB."""
    files = cal_files() + perf_files()
    assert len(cal_files()) == 3
    assert len(perf_files()) == 6
    assert all(p.stat().st_size < 100_000 for p in files)


def test_script_reports_no_mismatch_on_fixtures(
    t1: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full T1 script runs on the fixture tree and finds zero mismatches."""
    assert t1.main([str(FIXTURES)]) == 0
    out = capsys.readouterr().out
    assert "3 calibration files, 6 performance files" in out
    assert "Mismatches (0 of" in out
    assert "the bug did not act" in out


@pytest.mark.parametrize(
    "path", perf_files(), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_performance_file_reproduced(t1: ModuleType, path: Path) -> None:
    """Recompute m*, alpha, estimator and RMSE of an ``oms`` file with ``mfmc_core``."""
    cal_path = path.parents[1] / f"{path.parents[1].name}.hdf5"
    cal = t1.load_models(cal_path)
    perf = t1.load_models(path)

    # models of the performance file in model order, mapped back via n_molecules
    cidx = np.array([int(np.flatnonzero(cal.n_mol == n)[0]) for n in perf.n_mol])
    budget = float(perf.root_attrs["budget"])
    rho = perf.attrs["correlation"]
    sigma = perf.attrs["std_initial"]
    # In all paper-I files the selected models are already ordered by decreasing rho,
    # i.e. name order == rho order (the precondition under which the legacy pairing of
    # std and rho was correct).
    assert np.all(np.diff(rho**2) < 0)

    # selection: the models in the file are the ones select_models picks
    sel, _ = core.select_models(
        cal.attrs["std"][0], cal.attrs["correlation"], cal.attrs["computation_time"]
    )
    assert set(cal.n_mol[list(sel)]) == set(perf.n_mol)

    # allocation with the calibration costs reproduces the stored n_evals exactly
    m, _ = core.allocate(rho, cal.attrs["computation_time"][cidx], budget)
    assert m.tolist() == perf.attrs["n_evals"].astype(int).tolist()

    # alpha: stored == correctly paired (== legacy paired, since the orders coincide)
    alpha = core.control_variates(rho, sigma)
    np.testing.assert_allclose(alpha, perf.attrs["alpha"], rtol=1e-10, atol=0)

    # estimator: float32-resolution agreement with float64 mfmc_core
    est = core.mfmc_estimate(perf.datasets["diffusion_coeff"], m, alpha)
    assert est == pytest.approx(float(perf.root_attrs["mfmc_estimator"]), rel=1e-5)

    # RMSE uses the performance-run costs stored in the file (legacy behaviour)
    err = core.mfmc_rmse(sigma[0], rho, perf.attrs["computation_time"], budget)
    assert err == pytest.approx(float(perf.root_attrs["mfmc_error"]), rel=1e-6)


@pytest.mark.parametrize("path", cal_files(), ids=lambda p: p.stem)
def test_calibration_file_reproduced(t1: ModuleType, path: Path) -> None:
    """Recompute mean/std/rho and the selection of a calibration file."""
    cal = t1.load_models(path)
    D = np.array(cal.datasets["diffusion_coeff"])
    mean, std, rho = core.model_statistics(D)
    np.testing.assert_allclose(mean, cal.attrs["mean"], rtol=1e-5)
    np.testing.assert_allclose(std, cal.attrs["std"], rtol=1e-5)
    np.testing.assert_allclose(rho, cal.attrs["correlation"], rtol=1e-5)
    # SI Table S1 (OrthoBoXY runs) and the published selection tables are checked by the
    # script; here we only require the paper-I selection of the fixture.
    sel, _ = core.select_models(
        cal.attrs["std"][0], cal.attrs["correlation"], cal.attrs["computation_time"]
    )
    expected = {
        "m6_e10_c_lj": [1024, 32],
        "m6_e1000_c_q": [1024, 64, 32],
        "m6_e10_o_q": [1024, 64],
    }[path.stem]
    assert cal.n_mol[list(sel)].tolist() == expected


def test_alpha_bug_would_matter_if_orders_differed(t1: ModuleType) -> None:
    """
    Document the bug: with differing orders the legacy pairing gives wrong alpha.

    ``m6_e10_c_lj`` has non-monotone rho over all six models (rho_3 > rho_2). If the
    selection had kept models 2 and 3, the legacy code would have paired the standard
    deviations in name order with correlations in rho order. This is *not* what happened
    in the stored files (their selected sets are ordered), which the tests above verify.
    """
    cal = t1.load_models(FIXTURES / "m6_e10_c_lj" / "m6_e10_c_lj.hdf5")
    rho, sigma = cal.attrs["correlation"], cal.attrs["std"]
    assert rho[2] > rho[1]
    idx = [0, 1, 2]
    order = [0, 2, 1]  # decreasing rho**2
    correct = core.control_variates(rho[order], sigma[order])
    legacy = t1.legacy_alpha(rho[order], sigma[idx])
    assert not np.allclose(correct, legacy, rtol=1e-6)


def test_lj_params_are_in_kcal_per_mol() -> None:
    """
    ``lj_params`` are stored in kcal/mol, the unit of LAMMPS ``units real``.

    The noise-free epsilon is 0.68369 kJ/mol * KJ2KCAL = 0.163406 kcal/mol and must not be
    converted again. The value of ``m6_e10_c_lj`` model_1/eval_1 written to a Marvin
    ``input.lmp`` as ``pair_coeff ... 0.162877 3.170409`` is ``lj_params[:, 0]``.
    """
    with h5py.File(FIXTURES / "m6_e1000_c_q" / "m6_e1000_c_q.hdf5", "r") as f:
        lj_const = f["models/lj_params"][:]
    # charge-noise runs keep the LJ parameters at their OPC3 values
    np.testing.assert_allclose(lj_const[0], 0.68369 * 0.239006, rtol=1e-6)
    assert abs(lj_const[0, 0] - 0.68369) > 0.5  # i.e. not kJ/mol

    with h5py.File(FIXTURES / "m6_e10_c_lj" / "m6_e10_c_lj.hdf5", "r") as f:
        lj = f["models/lj_params"][:]
    assert f"{lj[0, 0]:.6f} {lj[1, 0]:.6f}" == "0.162877 3.170409"
