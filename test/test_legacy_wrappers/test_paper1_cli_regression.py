"""
Regression of the legacy command line algorithms against the stored paper-I results.

The paper-I fixtures of ``test/data/paper1`` (read-only copies) are copied to a temporary
directory and run through ``mfmc-prep``, ``model-select``, ``eval-estimator`` and ``mfmc``
(the MD phases are replaced by copying the stored performance data into the file). The
values must agree with the stored ones to float32 resolution of the data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from mfwater import console_entry_point
from mfwater import mfmc_core as core

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "paper1"


def perf_files() -> list[Path]:
    """
    List the performance (``oms``) fixtures.

    Returns
    -------
    list of pathlib.Path
        The ``*_oms.hdf5`` files.
    """
    return sorted(FIXTURES.glob("m6_*/oms_*/*_oms.hdf5"))


def _models(f: h5py.File) -> dict[int, h5py.Group]:
    """
    Map the number of molecules to the model group.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file.

    Returns
    -------
    dict of int to h5py.Group
        Model groups by ``n_molecules``.
    """
    return {
        int(g.attrs["n_molecules"]): g
        for g in f["models"].values()
        if isinstance(g, h5py.Group)
    }


@pytest.mark.parametrize(
    "perf", perf_files(), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_full_legacy_chain_reproduces_paper1(tmp_path: Path, perf: Path) -> None:
    """prep -> select -> estimator -> (stored performance data) -> mfmc vs. stored values."""
    cal_src = perf.parents[1] / f"{perf.parents[1].name}.hdf5"
    cal = tmp_path / "cal.hdf5"
    sel = tmp_path / "sel.hdf5"
    shutil.copy(cal_src, cal)
    with h5py.File(perf) as p:
        budget = float(p["models"].attrs["budget"])
        # plain values: the groups are not usable after the file is closed
        stored = {
            n: {
                "n_evals": int(g.attrs["n_evals"]),
                "alpha": float(g.attrs["alpha"]),
                "cost": float(g.attrs["computation_time"]),
                "D": g["diffusion_coeff"][:],
            }
            for n, g in _models(p).items()
        }
        stored_err = float(p["models"].attrs["mfmc_error"])
        stored_est = float(p["models"].attrs["mfmc_estimator"])

    assert console_entry_point(["-a", "mfmc-prep", "-i", str(cal)]) == 0
    assert (
        console_entry_point(["-a", "model-select", "-i", str(cal), "-o", str(sel)]) == 0
    )
    assert (
        console_entry_point(
            ["-a", "eval-estimator", "-i", str(sel), "--budget", str(budget)]
        )
        == 0
    )

    with h5py.File(sel) as f:
        mods = [
            f["models"][f"model_{k}"]
            for k in range(1, int(f["models"].attrs["n_models"]) + 1)
        ]
        assert [int(m.attrs["n_molecules"]) for m in mods] == list(stored)
        for m in mods:
            ref = stored[int(m.attrs["n_molecules"])]
            assert int(m.attrs["n_evals"]) == ref["n_evals"]
            assert float(m.attrs["alpha"]) == pytest.approx(ref["alpha"], rel=1e-5)

    # the performance runs: stored diffusion coefficients and costs, as chemmod-post
    # would write them
    with h5py.File(sel, "r+") as f:
        for n_mol, grp in _models(f).items():
            grp.create_dataset("diffusion_coeff", data=stored[n_mol]["D"])
            grp.attrs["computation_time"] = stored[n_mol]["cost"]
        f.attrs["last_algo"] = "chemmodel-post"

    assert console_entry_point(["-a", "mfmc", "-i", str(sel)]) == 0
    with h5py.File(sel) as f:
        attrs = f["models"].attrs
        # default basis is 'performance': the stored paper-I RMSE
        assert attrs["rmse_cost"] == "performance"
        assert float(attrs["mfmc_error"]) == pytest.approx(stored_err, rel=1e-6)
        assert float(attrs["mfmc_estimator"]) == pytest.approx(stored_est, rel=1e-5)
        assert float(attrs["mfmc_error_performance_cost"]) == float(attrs["mfmc_error"])
        mods = [
            f["models"][f"model_{k}"]
            for k in range(1, int(f["models"].attrs["n_models"]) + 1)
        ]
        rho = np.array([m.attrs["correlation"] for m in mods], dtype=float)
        w_perf = np.array([m.attrs["computation_time"] for m in mods], dtype=float)
        w_cal = np.array(
            [m.attrs["computation_time_initial"] for m in mods], dtype=float
        )
        assert float(attrs["gamma"]) == pytest.approx(
            core.gamma(rho, w_perf), rel=1e-12
        )
        # gamma is stored for both bases; `gamma` is the chosen one
        assert float(attrs["gamma_performance_cost"]) == float(attrs["gamma"])
        assert float(attrs["gamma_calibration_cost"]) == pytest.approx(
            core.gamma(rho, w_cal), rel=1e-12
        )
        sigma1 = float(mods[0].attrs["std_initial"])
        assert float(attrs["mfmc_error_calibration_cost"]) == pytest.approx(
            core.mfmc_rmse(sigma1, rho, w_cal, budget), rel=1e-12
        )


@pytest.mark.parametrize(
    "perf", perf_files(), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_mfmc_on_stored_performance_file(tmp_path: Path, perf: Path) -> None:
    """``mfmc`` on a copy of the stored performance file reproduces its estimator and RMSE."""
    work = tmp_path / perf.name
    shutil.copy(perf, work)
    with h5py.File(perf) as p:
        stored_err = float(p["models"].attrs["mfmc_error"])
        stored_est = float(p["models"].attrs["mfmc_estimator"])

    assert console_entry_point(["-a", "mfmc", "-i", str(work)]) == 0
    with h5py.File(work) as f:
        attrs = f["models"].attrs
        # float64 on float32 data: about 1e-7 relative agreement, not bitwise
        assert float(attrs["mfmc_error"]) == pytest.approx(stored_err, rel=1e-6)
        assert float(attrs["mfmc_estimator"]) == pytest.approx(stored_est, rel=1e-5)
        assert 0.0 < float(attrs["gamma"]) < 1.0
        # old files carry no calibration costs: only the performance basis is stored
        assert "mfmc_error_calibration_cost" not in attrs
        assert "gamma_calibration_cost" not in attrs
        assert float(attrs["gamma_performance_cost"]) == float(attrs["gamma"])


def test_rmse_cost_calibration_needs_stored_costs(tmp_path: Path) -> None:
    """The calibration basis is refused with a clear message if the costs are missing."""
    work = tmp_path / "perf.hdf5"
    shutil.copy(perf_files()[0], work)
    with pytest.raises(RuntimeError, match="computation_time_initial"):
        console_entry_point(
            ["-a", "mfmc", "-i", str(work), "--rmse-cost", "calibration"]
        )
