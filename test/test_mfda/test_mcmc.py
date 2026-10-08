"""Tests for the core MFDA-MCMC algorithm."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from mfwater.algo_mfda.markov_chain import (
    ChainResult,
    MemoizedForward,
    build_config,
    make_config,
    markov_chain_eval,
    run_chain,
)
from mfwater.algo_mfda.multifidelity_mcmc import (
    MFDAConfig,
    log_prior,
    mfda_estimator,
    proposal_kernel,
)
from mfwater.argparser import constants


def toy_forward(n: int, theta: np.ndarray) -> float:
    """Deterministic toy forward model with a level dependent bias."""
    return float(2.5 + 5.0 * (theta[0] / 0.16 - 1.0) + 0.01 * n)


def make_args(**kw: object) -> argparse.Namespace:
    """Namespace with valid defaults."""
    d: dict[str, object] = dict(
        n_models=3,
        n_molecules=[64, 32, 16],
        n_mc_subchain_lengths=[2, 3],
        n_mc_chain_length=6,
        n_mc_burnin=0,
        params="lj",
        seed=1,
        output="out.hdf5",
    )
    d.update(kw)
    return argparse.Namespace(**d)


def run(
    seed: int = 0,
    forward: Callable[[int, np.ndarray], float] = toy_forward,
    mols: tuple[int, ...] = (64, 32, 16),
    sub: tuple[int, ...] = (2, 3),
    m1: int = 6,
    burn: int = 0,
    params: str = "lj-q",
) -> tuple[ChainResult, MFDAConfig]:
    """Run a small chain and return (result, config)."""
    cfg = make_config(mols, sub, params)
    res = run_chain(cfg, forward, np.random.default_rng(seed), m1, burn, seed)
    return res, cfg


@pytest.mark.parametrize(
    "mols,sub,m1",
    [((64, 32), (3,), 10), ((64, 32, 16), (2, 3), 6), ((64, 32, 16), (1, 1), 4)],
)
def test_list_lengths(mols: tuple[int, ...], sub: tuple[int, ...], m1: int) -> None:
    res, _ = run(mols=mols, sub=sub, m1=m1)
    k = m1
    for i in range(len(mols)):
        assert len(res.samples[i]) == k
        if i < len(mols) - 1:
            assert len(res.proposals[i]) == k
            k *= sub[i]
        else:
            assert len(res.proposals[i]) == 0  # no X'_eta


def test_proposals_are_subchain_elements() -> None:
    res, _ = run(m1=6, sub=(2, 3))
    for i, m_sub in enumerate((2, 3)):
        for j, prop in enumerate(res.proposals[i]):
            sub = res.samples[i + 1][j * m_sub : (j + 1) * m_sub]
            assert any(np.array_equal(prop, s) for s in sub)


def test_burnin_removes_first_b_steps() -> None:
    full, _ = run(burn=0, m1=8)
    cut, _ = run(burn=3, m1=8)
    # level i has M_1 * prod(M~) entries per fine step: 1, 2, 6
    per_step = [1, 2, 6]
    for i, n in enumerate(per_step):
        assert len(cut.samples[i]) == (8 - 3) * n
        np.testing.assert_array_equal(cut.samples[i], full.samples[i][3 * n :])
    for i in range(2):
        np.testing.assert_array_equal(
            cut.proposals[i], full.proposals[i][3 * per_step[i] :]
        )


def test_seed_reproducibility() -> None:
    a, _ = run(seed=5)
    b, _ = run(seed=5)
    c, _ = run(seed=6)
    for x, y in zip(a.samples, b.samples):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.estimator, b.estimator)
    assert not np.array_equal(a.samples[0], c.samples[0])


def test_rng_independent_of_caching() -> None:
    cfg = make_config((64, 32, 16), (2, 3), "lj-q")
    rng1 = np.random.default_rng(3)
    memo = MemoizedForward(toy_forward)
    r1 = run_chain(cfg, memo, rng1, 6, 0)
    rng2 = np.random.default_rng(3)
    r2 = run_chain(cfg, toy_forward, rng2, 6, 0)
    for x, y in zip(r1.samples, r2.samples):
        np.testing.assert_array_equal(x, y)
    # identical RNG state afterwards: same number of draws
    assert rng1.random() == rng2.random()
    assert memo.n_cached > 0


def test_fixed_parameters_stay_at_mean() -> None:
    res, cfg = run(params="lj")
    for level in res.samples + [p for p in res.proposals if len(p)]:
        assert np.all(level[:, 2] == constants.OPC3_CHARGE_O)
    assert np.any(res.samples[0][:, 0] != constants.OPC3_EPSILON_OO)
    res, cfg = run(params="q")
    assert np.all(res.samples[0][:, :2] == cfg.means[:2])


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="At least 2"):
        build_config(make_args(n_models=1, n_molecules=[8], n_mc_subchain_lengths=[]))
    with pytest.raises(ValueError, match="--molecules must list"):
        build_config(make_args(n_molecules=[64, 32]))
    with pytest.raises(ValueError, match="descending"):
        build_config(make_args(n_molecules=[64, 64, 16]))
    with pytest.raises(ValueError, match="--mcsubchainlength"):
        build_config(make_args(n_mc_subchain_lengths=[2]))
    with pytest.raises(ValueError, match="Subchain lengths"):
        build_config(make_args(n_mc_subchain_lengths=[2, 0]))
    with pytest.raises(ValueError, match="--mcburnin"):
        build_config(make_args(n_mc_burnin=6))
    with pytest.raises(ValueError, match="No free"):
        MFDAConfig(
            (8, 4),
            (1,),
            np.zeros(3),
            np.ones(3),
            np.zeros(3, dtype=bool),
        )
    with pytest.raises(ValueError, match="No free"):
        log_prior(np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="Burn-in"):
        run(burn=6, m1=6)
    with pytest.raises(ValueError, match="same shape"):
        proposal_kernel(np.zeros(3), np.ones(2), np.random.default_rng(0))


def test_estimator_formula() -> None:
    x1 = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])]
    p1 = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])]
    x2 = [
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array([3.0, 3.0, 3.0]),
    ]
    p2 = [np.array([1.0, 0.0, 1.0])] * 3
    x3 = [np.array([10.0, 10.0, 10.0]), np.array([20.0, 20.0, 20.0])]
    est, contribs = mfda_estimator([x1, x2, x3], [p1, p2, []])
    # hand-computed: means of x1/p1, x2/p2 and x3
    np.testing.assert_allclose(contribs[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(contribs[1], [1.0, 2.0, 1.0])
    np.testing.assert_allclose(contribs[2], [15.0, 15.0, 15.0])
    np.testing.assert_allclose(est, [17.0, 18.0, 17.0])


def test_estimator_errors() -> None:
    with pytest.raises(ValueError, match="No samples"):
        mfda_estimator([[], [np.zeros(3)]], [[], []])
    with pytest.raises(ValueError, match="proposals"):
        mfda_estimator([[np.zeros(3)], [np.zeros(3)]], [[], []])


def test_markov_chain_eval_writes_hdf5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mfwater.algo_mfda.markov_chain.forward_model_dummy", toy_forward
    )
    out = tmp_path / "chain.hdf5"
    args = make_args(output=str(out), n_mc_burnin=2)
    assert markov_chain_eval(args) == 0
    with h5py.File(out, "r") as f:
        assert f["samples_level1"].shape == (4, 3)
        assert f["samples_level3"].shape == (4 * 6, 3)
        assert "proposals_level3" not in f
        assert f["estimator"].shape == (3,)
        assert f.attrs["seed"] == "1"
        assert f.attrs["n_burnin"] == 2
    with pytest.raises(FileExistsError):
        markov_chain_eval(args)
