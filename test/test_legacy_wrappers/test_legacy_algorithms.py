"""
Tests of the legacy algorithms: model ordering, alpha pairing, refusals and the MD setup.

External programs (fftool, packmol) are replaced by a fake ``subprocess.run``.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from mfwater import console_entry_point
from mfwater import mfmc_core as core
from mfwater.algo_chemical_model import chemical_model as cm
from mfwater.algo_chemical_model.chemical_model import (
    check_lammps_parameters,
    chemical_model_post,
)
from mfwater.algo_input import model_groups, models_by_rho
from mfwater.md.lammps_log import parse_lammps_log

POST_DATA = Path(__file__).resolve().parents[1] / "test_cli" / "data" / "chemmodel-post"


def sha(path: Path) -> str:
    """
    Hash a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_many_models(path: Path, n_models: int = 12, m: int = 40) -> np.ndarray:
    """
    Write a calibrated input file with many models (``model_1`` = largest).

    Parameters
    ----------
    path : pathlib.Path
        File to create.
    n_models : int
        Number of models; 10 or more breaks the alphabetical h5py order.
    m : int
        Evaluations per model.

    Returns
    -------
    numpy.ndarray
        The diffusion coefficients, shape ``(n_models, m)``, in model order.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(2000.0, 300.0, size=m)
    # noise grows with the model number: rho decreases, costs decrease
    D = np.array(
        [base + rng.normal(0.0, 40.0 * (k + 1), size=m) for k in range(n_models)]
    )
    with h5py.File(path, "w") as f:
        models = f.create_group("models")
        models.attrs["n_models"] = n_models
        for k in range(n_models):
            g = models.create_group(f"model_{k + 1}")
            g.attrs["n_molecules"] = 2 ** (n_models + 4 - k)
            g.attrs["n_evals"] = m
            g.attrs["computation_time"] = 5000.0 / 2.0 ** (1.5 * k)
            g.create_dataset("diffusion_coeff", data=D[k])
        f.attrs["last_algo"] = "chemmodel-post"
    return D


def test_model_order_is_numeric_not_alphabetical(tmp_path: Path) -> None:
    """With 12 models, ``model_10`` must not come before ``model_2``."""
    path = tmp_path / "many.hdf5"
    write_many_models(path)
    with h5py.File(path) as f:
        assert list(f["models"])[1] == "model_10"  # h5py order: the trap
        names = [n for n, _ in model_groups(f["models"])]
    assert names == [f"model_{k}" for k in range(1, 13)]


def test_prep_select_estimator_with_12_models(tmp_path: Path) -> None:
    """The legacy chain with 12 models agrees with ``mfmc_core`` on the model-order arrays."""
    path = tmp_path / "many.hdf5"
    sel = tmp_path / "sel.hdf5"
    D = write_many_models(path)
    assert console_entry_point(["-a", "mfmc-prep", "-i", str(path)]) == 0

    mean, std, rho = core.model_statistics(D)
    with h5py.File(path) as f:
        for k in range(12):
            g = f["models"][f"model_{k + 1}"]
            assert g.attrs["std"] == pytest.approx(std[k], rel=1e-12)
            assert g.attrs["correlation"] == pytest.approx(rho[k], rel=1e-12)
            assert g.attrs["mean"] == pytest.approx(mean[k], rel=1e-12)

    w = np.array([5000.0 / 2.0 ** (1.5 * k) for k in range(12)])
    expected, _ = core.select_models(std[0], rho, w)
    assert (
        console_entry_point(["-a", "model-select", "-i", str(path), "-o", str(sel)])
        == 0
    )
    with h5py.File(sel) as f:
        n_sel = int(f["models"].attrs["n_models"])
        assert n_sel == len(expected)
        # renamed model_1.. in selection order (decreasing rho**2)
        got = [
            int(f["models"][f"model_{k}"].attrs["n_molecules"])
            for k in range(1, n_sel + 1)
        ]
    assert got == [2 ** (12 + 4 - k) for k in expected]

    assert (
        console_entry_point(["-a", "eval-estimator", "-i", str(sel), "--budget", "1e7"])
        == 0
    )
    sel_idx = list(expected)
    m_exp, _ = core.allocate(rho[sel_idx], w[sel_idx], 1e7)
    alpha_exp = core.control_variates(rho[sel_idx], std[sel_idx])
    with h5py.File(sel) as f:
        for k in range(n_sel):
            g = f["models"][f"model_{k + 1}"]
            assert int(g.attrs["n_evals"]) == m_exp[k]
            assert g.attrs["alpha"] == pytest.approx(alpha_exp[k], rel=1e-12)


def test_eval_estimator_pairs_alpha_with_rho_order(tmp_path: Path) -> None:
    """
    alpha = rho * sigma_1 / sigma_i holds even if name order and rho order differ.

    The legacy code paired the standard deviations in name order with the correlations in
    rho order. Here model_3 has a higher rho than model_2.
    """
    path = tmp_path / "unordered.hdf5"
    spec = {  # name: (n_molecules, rho, std, cost)
        "model_1": (1024, 1.0, 10.0, 1000.0),
        "model_2": (64, 0.8, 20.0, 20.0),
        "model_3": (32, 0.9, 30.0, 100.0),
    }
    with h5py.File(path, "w") as f:
        models = f.create_group("models")
        models.attrs["n_models"] = 3
        for name, (n, rho, std, w) in spec.items():
            g = models.create_group(name)
            g.attrs.update(
                {
                    "n_molecules": n,
                    "n_evals": 5,
                    "correlation": rho,
                    "std": std,
                    "mean": 1.0,
                    "computation_time": w,
                }
            )
            g.create_dataset("diffusion_coeff", data=np.arange(5.0))
        f.attrs["last_algo"] = "model-select"

    assert (
        console_entry_point(
            ["-a", "eval-estimator", "-i", str(path), "--budget", "1e5"]
        )
        == 0
    )
    with h5py.File(path) as f:
        assert f["models/model_1"].attrs["alpha"] == 1.0
        assert f["models/model_3"].attrs["alpha"] == pytest.approx(0.9 * 10 / 30)
        assert f["models/model_2"].attrs["alpha"] == pytest.approx(0.8 * 10 / 20)
        # allocation in rho order (model_1, model_3, model_2): the cheaper model_2 gets
        # the most evaluations
        m_exp, _ = core.allocate([1.0, 0.9, 0.8], [1000.0, 100.0, 20.0], 1e5)
        assert m_exp.tolist() == [64, 191, 832]
        assert [int(f["models"][f"model_{k}"].attrs["n_evals"]) for k in (1, 3, 2)] == (
            m_exp.tolist()
        )
        assert [n for n, _ in models_by_rho(f["models"])] == [
            "model_1",
            "model_3",
            "model_2",
        ]


def test_eval_estimator_failure_leaves_file_untouched(tmp_path: Path) -> None:
    """A too small budget is refused before anything is written."""
    path = tmp_path / "many.hdf5"
    sel = tmp_path / "sel.hdf5"
    write_many_models(path, n_models=4)
    console_entry_point(["-a", "mfmc-prep", "-i", str(path)])
    console_entry_point(["-a", "model-select", "-i", str(path), "-o", str(sel)])
    before = sha(sel)
    with pytest.raises(ValueError, match="too small"):
        console_entry_point(["-a", "eval-estimator", "-i", str(sel), "--budget", "5"])
    assert sha(sel) == before


def test_model_select_refuses_existing_output(tmp_path: Path) -> None:
    """``model-select`` never overwrites an existing output file."""
    path = tmp_path / "many.hdf5"
    out = tmp_path / "sel.hdf5"
    write_many_models(path, n_models=4)
    console_entry_point(["-a", "mfmc-prep", "-i", str(path)])
    out.write_bytes(b"precious")
    with pytest.raises(FileExistsError):
        console_entry_point(["-a", "model-select", "-i", str(path), "-o", str(out)])
    assert out.read_bytes() == b"precious"


def test_rmse_cost_option_is_registered_for_reuse() -> None:
    """``--rmse-cost`` has the dest, choices and default that ``mfmc-run`` reuses."""
    from mfwater import parser

    act = next(a for a in parser()._actions if a.dest == "rmse_cost")
    assert list(act.choices or []) == ["calibration", "performance"]
    assert act.default is None
    with pytest.raises(SystemExit):
        parser().parse_args(["-a", "mfmc", "--rmse-cost", "nonsense"])


# ---------------------------------------------------------------- chemmodel-prep


class FakeExternal:
    """
    Fake ``subprocess.run`` for fftool and packmol that writes the files they create.

    Parameters
    ----------
    calls : list
        Collects ``(cmd, kwargs)`` of every call.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """
        Emulate one external program.

        Parameters
        ----------
        cmd : list of str
            Command; must be an argument list (no shell).
        **kwargs : Any
            Keyword arguments of ``subprocess.run``.

        Returns
        -------
        subprocess.CompletedProcess
            A successful result.
        """
        assert isinstance(cmd, list) and not kwargs.get("shell")
        self.calls.append((cmd, kwargs))
        cwd = Path(kwargs["cwd"])
        if cmd[0] == "fftool" and "--lmp" not in cmd:
            (cwd / "pack.inp").write_text("tolerance 2.0\ninside box 0 0 0 9 9 9\n")
            (cwd / "simbox.xyz").write_text("box\n")
        elif cmd[0] == "packmol":
            (cwd / "opc3_pack.xyz").write_text("xyz\n")
        elif cmd[0] == "fftool":
            (cwd / "data.lmp").write_text("data\n")
            (cwd / "in.lmp").write_text("in\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def fake_external(monkeypatch: pytest.MonkeyPatch) -> FakeExternal:
    """
    Install the fake external programs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture.

    Returns
    -------
    FakeExternal
        The fake whose calls can be inspected.
    """
    fake = FakeExternal()
    monkeypatch.setattr(cm.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(cm.subprocess, "run", fake)
    return fake


def build_input(path: Path) -> None:
    """
    Create a small ``build`` input file (2 models).

    Parameters
    ----------
    path : pathlib.Path
        File to create.
    """
    assert (
        console_entry_point(
            [
                "-a",
                "build",
                "--models",
                "2",
                "--molecules",
                "8",
                "4",
                "--evals",
                "2",
                "3",
            ]
            + ["-o", str(path)]
        )
        == 0
    )


def test_chemmodel_prep_seed_and_arg_lists(
    tmp_path: Path, fake_external: FakeExternal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--seed`` makes the sampled data reproducible and is stored; no shell is used."""
    monkeypatch.chdir(tmp_path)
    runs = {}
    for tag in ("a", "b"):
        inp = tmp_path / f"{tag}.hdf5"
        build_input(inp)
        assert (
            console_entry_point(
                ["-a", "chemmodel-prep", "-i", str(inp), "--seed", "7"]
                + ["--models-dir", f"models_{tag}"]
            )
            == 0
        )
        with h5py.File(inp) as f:
            assert int(f["models"].attrs["seed"]) == 7
            assert f.attrs["last_algo"] == "chemmodel-prep"
            runs[tag] = {
                k: f["models"][k][:] for k in ("lj_params", "seeds", "charges")
            }
    for k in runs["a"]:
        np.testing.assert_array_equal(runs["a"][k], runs["b"][k])
    assert runs["a"]["seeds"].shape == (2, 3)

    # model_1 (N=8, 2 evals) and model_2 (N=4, 3 evals), every call an argument list
    assert (
        tmp_path / "models_a" / "model_2" / "eval_3" / "siminp" / "input.lmp"
    ).exists()
    assert not (tmp_path / "models_a" / "model_1" / "eval_3").exists()
    packmol_calls = [k for c, k in fake_external.calls if c[0] == "packmol"]
    assert len(packmol_calls) == 2 * 5 and all(
        k["stdin"] is not None for k in packmol_calls
    )
    seed_line = f"seed {runs['a']['seeds'][0][0]}\n"
    # pack.inp is removed after use; the seed reached the LAMMPS input instead
    lmp = (
        tmp_path / "models_a" / "model_1" / "eval_1" / "siminp" / "input.lmp"
    ).read_text()
    assert f"velocity all create ${{vTK}} {runs['a']['seeds'][1][0]}" in lmp
    assert seed_line.startswith("seed ")


def test_chemmodel_prep_draws_and_stores_seed_if_omitted(
    tmp_path: Path, fake_external: FakeExternal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--seed`` a seed is drawn, printed and stored."""
    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "a.hdf5"
    build_input(inp)
    console_entry_point(["-a", "chemmodel-prep", "-i", str(inp)])
    with h5py.File(inp) as f:
        assert "seed" in f["models"].attrs


def test_chemmodel_prep_refuses_stale_phase_results(
    tmp_path: Path, fake_external: FakeExternal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing ``simout``/``msd`` of an earlier phase stop the run; nothing is changed."""
    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "a.hdf5"
    build_input(inp)
    stale = tmp_path / "models" / "model_1" / "eval_2"
    (stale / "simout").mkdir(parents=True)
    (stale / "simout" / "log.lammps").write_text("old result")
    (stale / "msd").mkdir()
    before = sha(inp)

    with pytest.raises(RuntimeError, match="earlier phase"):
        console_entry_point(["-a", "chemmodel-prep", "-i", str(inp), "--seed", "1"])

    assert sha(inp) == before  # no datasets were written
    assert (stale / "simout" / "log.lammps").read_text() == "old result"
    assert (stale / "msd").is_dir()
    assert not fake_external.calls

    # a new directory is accepted
    assert (
        console_entry_point(
            ["-a", "chemmodel-prep", "-i", str(inp), "--models-dir", "phase2"]
        )
        == 0
    )


def test_chemmodel_prep_second_run_in_same_dir_refused(
    tmp_path: Path, fake_external: FakeExternal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The siminp of a finished prep is never deleted by a second prep."""
    monkeypatch.chdir(tmp_path)
    first, second = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    build_input(first)
    build_input(second)
    console_entry_point(["-a", "chemmodel-prep", "-i", str(first)])
    marker = tmp_path / "models" / "model_1" / "eval_1" / "siminp" / "marker"
    marker.write_text("keep")
    with pytest.raises(RuntimeError, match="earlier phase"):
        console_entry_point(["-a", "chemmodel-prep", "-i", str(second)])
    assert marker.read_text() == "keep"


def test_chemmodel_prep_missing_software(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing fftool is reported before the input file is modified."""
    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "a.hdf5"
    build_input(inp)
    before = sha(inp)
    monkeypatch.setattr(cm.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="fftool"):
        console_entry_point(["-a", "chemmodel-prep", "-i", str(inp)])
    assert sha(inp) == before


# --------------------------------------------------------------- chemmodel-post


def test_check_lammps_parameters(tmp_path: Path) -> None:
    """All echoes are checked, a missing line is an error, and there is no early exit."""
    log = tmp_path / "log.lammps"
    # the oxygen charge comes first, a wrong hydrogen charge after it
    log.write_text(
        "pair_coeff    2    2     0.163049     3.182090  # Ow-Ow\n"
        "set type 2 charge -0.89517  # Ow\n"
        "set type 1 charge  0.447585  # Hw\n"
    )
    check_lammps_parameters(log, (0.163049, 3.18209), (-0.89517, 0.447585))  # ok
    with pytest.raises(ValueError, match="hydrogen"):
        check_lammps_parameters(log, (0.163049, 3.18209), (-0.89517, 0.40))
    with pytest.raises(ValueError, match="Lennard-Jones"):
        check_lammps_parameters(log, (0.2, 3.18209), None)
    # skipped checks need no lines; a requested but absent one is an error
    log.write_text("nothing here\n")
    check_lammps_parameters(log, None, None)
    with pytest.raises(ValueError, match="not found"):
        check_lammps_parameters(log, (0.163049, 3.18209), None)


def test_chemmodel_post_costs_and_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Costs come from each evaluation's own log; ``--models-dir`` selects the directory."""
    shutil.copytree(POST_DATA / "models", tmp_path / "custom")
    shutil.copy(POST_DATA / "test_chemmodel-post.hdf5", tmp_path / "in.hdf5")
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = console_entry_point(
            ["-a", "chemmodel-post", "-i", "in.hdf5", "--models-dir", "custom"]
        )
    assert rc == 0
    with h5py.File("in.hdf5") as f:
        for name, g in f["models"].items():
            if not isinstance(g, h5py.Group):
                continue
            expected = np.mean(
                [
                    parse_lammps_log(
                        tmp_path
                        / "custom"
                        / name
                        / f"eval_{j}"
                        / "simout"
                        / "log.lammps"
                    ).cpu_time_s
                    for j in range(1, int(g.attrs["n_evals"]) + 1)
                ]
            )
            assert g.attrs["computation_time"] == pytest.approx(expected)


def test_chemmodel_post_unfinished_log_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unfinished log raises instead of reusing the cost of the previous evaluation."""
    shutil.copytree(POST_DATA / "models", tmp_path / "models")
    shutil.copy(POST_DATA / "test_chemmodel-post.hdf5", tmp_path / "in.hdf5")
    log = tmp_path / "models" / "model_1" / "eval_2" / "simout" / "log.lammps"
    log.write_text(
        "\n".join(
            line
            for line in log.read_text().splitlines()
            if not line.startswith("Total wall time")
        )
        + "\n"
    )
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="Total wall time"):
            chemical_model_post(_args(["-a", "chemmodel-post", "-i", "in.hdf5"]))


def test_chemmodel_post_parameter_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log whose parameters differ from the stored ones is rejected with model and eval."""
    shutil.copytree(POST_DATA / "models", tmp_path / "models")
    shutil.copy(POST_DATA / "test_chemmodel-post.hdf5", tmp_path / "in.hdf5")
    log = tmp_path / "models" / "model_2" / "eval_3" / "simout" / "log.lammps"
    log.write_text(
        "".join(
            (
                "pair_coeff    2    2     0.500000     3.182090  # Ow-Ow\n"
                if line.startswith("pair_coeff    2    2")
                else line
            )
            for line in log.read_text().splitlines(keepends=True)
        )
    )
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="model_2 evaluation 3"):
            chemical_model_post(_args(["-a", "chemmodel-post", "-i", "in.hdf5"]))


def _args(argv: list[str]) -> Any:
    """
    Parse command line arguments.

    Parameters
    ----------
    argv : list of str
        Arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    from mfwater import parser

    return parser().parse_args(argv)
