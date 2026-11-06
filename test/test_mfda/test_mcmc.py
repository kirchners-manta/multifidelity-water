"""Tests for the core MFDA-MCMC algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from mfwater.algo_mfda.cache import CachedForwardModel
from mfwater.algo_mfda.forward_model import ForwardModel
from mfwater.algo_mfda.markov_chain import (
    ChainResult,
    build_config,
    make_config,
    run_chain,
)
from mfwater.algo_mfda.multifidelity_mcmc import (
    MFDAConfig,
    acceptance_probability,
    log_likelihood,
    mfda_estimator,
    proposal_kernel,
)
from mfwater.argparser import constants


def toy_forward(n: int, theta: np.ndarray) -> float:
    """Deterministic toy forward model with a level dependent bias."""
    return float(2.5 + 5.0 * (theta[0] / 0.16 - 1.0) + 0.01 * n)


class ToyModel:
    """Class version of :func:`toy_forward` (satisfies the ForwardModel protocol)."""

    def __call__(self, n_molecules: int, theta: np.ndarray) -> float:
        return toy_forward(n_molecules, theta)


def make_args(**kw: object) -> argparse.Namespace:
    """Namespace with valid defaults."""
    d: dict[str, object] = dict(
        n_models=3,
        n_molecules=[64, 32, 16],
        n_mc_subchain_lengths=[2, 3],
        n_mc_chain_length=6,
        n_mc_burnin=0,
        noise_stds=[0.0, 0.0, 0.0],
        fs_correction="none",
        orthoboxy=False,
        params="lj",
        seed=1,
        output="out.hdf5",
        print_level=0,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def run(
    seed: int = 0,
    forward: ForwardModel = toy_forward,
    mols: tuple[int, ...] = (64, 32, 16),
    sub: tuple[int, ...] = (2, 3),
    m1: int = 6,
    burn: int = 0,
    params: str = "lj-q",
) -> tuple[ChainResult, MFDAConfig]:
    """Run a small chain and return (result, config)."""
    zeros = (0.0,) * len(mols)
    cfg = make_config(
        mols, sub, params, m1, burn, noise_stds=zeros, fs_corrections=zeros
    )
    res = run_chain(cfg, forward, np.random.default_rng(seed), seed)
    return res, cfg


@pytest.mark.parametrize(
    "mols,sub,m1",
    [((64, 32), (3,), 10), ((64, 32, 16), (2, 3), 6), ((64, 32, 16), (1, 1), 4)],
)
@pytest.mark.parametrize("burn", [0, 3])
def test_list_lengths(
    mols: tuple[int, ...], sub: tuple[int, ...], m1: int, burn: int
) -> None:
    # K_1 = M_1 independent of the burn-in
    res, _ = run(mols=mols, sub=sub, m1=m1, burn=burn)
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


@pytest.mark.parametrize("burn", [1, 3])
def test_burnin_is_added_to_chain_length(burn: int) -> None:
    # the un-truncated run has B + M_1 fine steps; the kept part is its tail
    full, _ = run(burn=0, m1=8 + burn)
    cut, _ = run(burn=burn, m1=8)
    # level i has prod(M~) entries per fine step: 1, 2, 6
    per_step = [1, 2, 6]
    for i, n in enumerate(per_step):
        assert len(cut.samples[i]) == 8 * n
        np.testing.assert_array_equal(cut.samples[i], full.samples[i][burn * n :])
    for i in range(2):
        np.testing.assert_array_equal(
            cut.proposals[i], full.proposals[i][burn * per_step[i] :]
        )
    assert cut.n_burnin == burn


def test_seed_reproducibility() -> None:
    a, _ = run(seed=5)
    b, _ = run(seed=5)
    c, _ = run(seed=6)
    for x, y in zip(a.samples, b.samples):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.estimator, b.estimator)
    assert not np.array_equal(a.samples[0], c.samples[0])


def test_rng_independent_of_caching(tmp_path: Path) -> None:
    cfg = make_config(
        (64, 32, 16),
        (2, 3),
        "lj-q",
        6,
        0,
        noise_stds=(0.0, 0.0, 0.0),  # test values, not defaults
        fs_corrections=(0.0, 0.0, 0.0),
    )
    rng1 = np.random.default_rng(3)
    memo = CachedForwardModel(ToyModel(), tmp_path / "cache")
    r1 = run_chain(cfg, memo, rng1)
    rng2 = np.random.default_rng(3)
    r2 = run_chain(cfg, toy_forward, rng2)
    for x, y in zip(r1.samples, r2.samples):
        np.testing.assert_array_equal(x, y)
    # identical RNG state afterwards: same number of draws
    assert rng1.random() == rng2.random()
    assert memo.n_hits > 0


def test_fixed_parameters_stay_at_mean() -> None:
    res, cfg = run(params="lj")
    for level in res.samples + [p for p in res.proposals if len(p)]:
        assert np.all(level[:, 2] == constants.OPC3_CHARGE_O)
    assert np.any(res.samples[0][:, 0] != constants.OPC3_EPSILON_OO)
    res, cfg = run(params="q")
    assert np.all(res.samples[0][:, :2] == cfg.means[:2])


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="At least two levels"):
        build_config(
            make_args(n_molecules=[8], n_mc_subchain_lengths=[], noise_stds=[0.0])
        )
    with pytest.raises(ValueError, match="--molecules is required"):
        build_config(make_args(n_molecules=None))
    with pytest.raises(ValueError, match="descending"):
        build_config(make_args(n_molecules=[64, 64, 16]))
    with pytest.raises(ValueError, match="--mcsubchainlength"):
        build_config(make_args(n_mc_subchain_lengths=[2, 3, 4]))
    with pytest.raises(ValueError, match="Subchain lengths"):
        build_config(make_args(n_mc_subchain_lengths=[2, 0]))
    with pytest.raises(ValueError, match="--mcchainlength"):
        build_config(make_args(n_mc_chain_length=0))
    with pytest.raises(ValueError, match="--mcburnin"):
        build_config(make_args(n_mc_burnin=-1))
    with pytest.raises(ValueError, match="No free"):
        MFDAConfig(
            (8, 4),
            (1,),
            np.zeros(3),
            np.ones(3),
            np.zeros(3, dtype=bool),
            1,
            0,
            (0.0, 0.0),
            (0.0, 0.0),
        )
    with pytest.raises(ValueError, match="same shape"):
        proposal_kernel(np.zeros(3), np.ones(2), np.random.default_rng(0))


def test_burnin_larger_than_chain_length_is_allowed() -> None:
    res, _ = run(burn=10, m1=2)
    assert len(res.samples[0]) == 2


def test_models_option_is_ignored() -> None:
    # the number of levels follows --molecules, not --models
    assert build_config(make_args(n_models=7)).n_levels == 3


def test_config_arrays_are_read_only() -> None:
    means = np.zeros(3)
    cfg = MFDAConfig(
        (8, 4),
        (1,),
        means,
        np.ones(3),
        np.ones(3, dtype=bool),
        1,
        0,
        (0.0,) * 2,
        (0.0,) * 2,
    )
    for arr in (cfg.means, cfg.kernel_noise, cfg.free_mask):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1
    means[0] = 5.0  # the caller's array is untouched and still writable
    assert cfg.means[0] == 0.0


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


# --- likelihood with noise width s_k and finite-size correction C_k ------------


def _old_log_likelihood(d: float) -> float:
    """Former likelihood: -0.5 sum_h (D - D_h)^2 / sigma_exp^2."""
    ref = constants.TARGET_DIFFUSION_COEFFICIENT
    return float(-0.5 * np.sum((d - ref) ** 2) / ref.std() ** 2)


def _hand_log_likelihood(d: float, s: float, c: float) -> float:
    """Target formula evaluated with plain Python floats."""
    ref = [float(x) for x in constants.TARGET_DIFFUSION_COEFFICIENT]
    alpha = len(ref)
    mean = sum(ref) / alpha
    var0 = sum((x - mean) ** 2 for x in ref) / alpha  # ddof = 0
    return -((mean - (d + c)) ** 2) / (2.0 * (var0 / alpha + s**2))


def test_log_likelihood_hand_value() -> None:
    d, s, c = 2.1, 0.07, 0.45
    assert log_likelihood(d, s, c) == pytest.approx(
        _hand_log_likelihood(d, s, c), rel=1e-12
    )
    # the correction shifts D: L(D, C) = L(D + C, 0)
    assert log_likelihood(d, s, c) == pytest.approx(log_likelihood(d + c, s, 0.0))
    # a larger noise width flattens the likelihood
    assert log_likelihood(d, 0.5, c) > log_likelihood(d, s, c)


def test_zero_noise_zero_correction_matches_old_differences() -> None:
    # constants are dropped, so compare differences of log L, not absolute values
    ds = [1.5, 2.3, 2.6, 3.4]
    new = [log_likelihood(d, 0.0, 0.0) for d in ds]
    old = [_old_log_likelihood(d) for d in ds]
    np.testing.assert_allclose(np.diff(new), np.diff(old), rtol=1e-12)


def test_zero_settings_reproduce_old_acceptance_ratios() -> None:
    cfg = make_config(
        (64, 32, 16), (2, 3), "lj", 1, 0, noise_stds=(0, 0, 0), fs_corrections=(0, 0, 0)
    )
    cur, prop = np.array([0.16, 3.1, -0.9]), np.array([0.17, 3.2, -0.9])
    p = acceptance_probability(1, cfg, cur, prop, toy_forward)
    old = (
        _old_log_likelihood(toy_forward(64, prop))
        + _old_log_likelihood(toy_forward(32, cur))
        - _old_log_likelihood(toy_forward(64, cur))
        - _old_log_likelihood(toy_forward(32, prop))
    )
    assert p == pytest.approx(min(1.0, float(np.exp(old))), rel=1e-12)


def _level_forward(n: int, theta: np.ndarray) -> float:
    """Level dependent D with a theta dependence that makes ratios non-trivial."""
    base = {64: 2.3, 32: 2.4, 16: 2.5}[n]
    slope = {64: 3.0, 32: -3.0, 16: -10.0}[n]
    return base + slope * (theta[0] - 0.16)


def test_acceptance_uses_settings_of_the_correct_level() -> None:
    noise, corr = (0.05, 0.2, 0.4), (0.1, 0.5, 0.9)
    cfg = make_config(
        (64, 32, 16), (2, 3), "lj", 1, 0, noise_stds=noise, fs_corrections=corr
    )
    cur, prop = np.array([0.15, 3.17, -0.9]), np.array([0.16, 3.17, -0.9])

    def hand(i_fine: int) -> float:
        n = (64, 32, 16)
        f, c = n[i_fine], n[i_fine + 1]
        lr = (
            _hand_log_likelihood(_level_forward(f, prop), noise[i_fine], corr[i_fine])
            + _hand_log_likelihood(
                _level_forward(c, cur), noise[i_fine + 1], corr[i_fine + 1]
            )
            - _hand_log_likelihood(_level_forward(f, cur), noise[i_fine], corr[i_fine])
            - _hand_log_likelihood(
                _level_forward(c, prop), noise[i_fine + 1], corr[i_fine + 1]
            )
        )
        return min(1.0, float(np.exp(lr)))

    # non-saturated probabilities, otherwise the comparison below proves nothing
    assert 0.0 < hand(0) < 1.0 and 0.0 < hand(1) < 1.0
    for level in (1, 2):
        assert acceptance_probability(
            level, cfg, cur, prop, _level_forward
        ) == pytest.approx(hand(level - 1), rel=1e-12)

    # swapping the per-level settings must change the result (the test has teeth)
    swapped = make_config(
        (64, 32, 16),
        (2, 3),
        "lj",
        1,
        0,
        noise_stds=noise[::-1],
        fs_corrections=corr[::-1],
    )
    assert acceptance_probability(
        1, swapped, cur, prop, _level_forward
    ) != pytest.approx(hand(0), rel=1e-2)


def test_chemical_mh_uses_coarsest_level_settings() -> None:
    # the coarsest-level chain depends on (s_k, C_k) of level 3: changing them
    # changes the samples
    def chain(noise: tuple[float, ...], corr: tuple[float, ...]) -> np.ndarray:
        cfg = make_config(
            (64, 32, 16), (2, 3), "lj", 6, 0, noise_stds=noise, fs_corrections=corr
        )
        return run_chain(cfg, toy_forward, np.random.default_rng(4)).samples[2]

    base = chain((0.1, 0.1, 0.1), (0.0, 0.0, 0.0))
    last = chain((0.1, 0.1, 0.002), (0.0, 0.0, 0.0))
    assert not np.array_equal(base, last)


def test_likelihood_settings_validation() -> None:
    kw = dict(
        molecules=(8, 4),
        subchain_lengths=(1,),
        means=np.zeros(3),
        kernel_noise=np.ones(3),
        free_mask=np.ones(3, dtype=bool),
        chain_length=1,
        n_burnin=0,
    )
    MFDAConfig(**kw, noise_stds=(0.0, 0.0), fs_corrections=(0.0, 0.0))  # 0 is legal
    with pytest.raises(ValueError, match="Expected 2 values for noise_stds"):
        MFDAConfig(**kw, noise_stds=(0.0,), fs_corrections=(0.0, 0.0))
    with pytest.raises(ValueError, match="Expected 2 values for fs_corrections"):
        MFDAConfig(**kw, noise_stds=(0.0, 0.0), fs_corrections=(0.0,) * 3)
    with pytest.raises(ValueError, match=">= 0"):
        MFDAConfig(**kw, noise_stds=(0.1, -0.1), fs_corrections=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        MFDAConfig(**kw, noise_stds=(0.1, float("nan")), fs_corrections=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        MFDAConfig(**kw, noise_stds=(0.1, 0.1), fs_corrections=(0.0, float("inf")))
    with pytest.raises(ValueError, match="one value per level"):
        build_config(make_args(noise_stds=[0.1, 0.1]))
    with pytest.raises(ValueError, match="fs-correction"):
        build_config(make_args(fs_correction="bogus"))
