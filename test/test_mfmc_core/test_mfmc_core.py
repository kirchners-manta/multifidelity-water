"""
Unit tests for the pure MFMC mathematics in ``mfwater.mfmc_core``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfwater import mfmc_core as core

# ---------------------------------------------------------------- model_statistics


def test_model_statistics_matches_numpy_ddof0() -> None:
    """Mean, std (ddof = 0) and correlation agree with NumPy's definitions."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=50)
    D = np.stack([z, 0.5 * z + rng.normal(size=50), -z + 0.1 * rng.normal(size=50)])
    mean, std, rho = core.model_statistics(D)
    np.testing.assert_allclose(mean, D.mean(axis=1), rtol=1e-14)
    np.testing.assert_allclose(std, D.std(axis=1, ddof=0), rtol=1e-14)
    # np.corrcoef is independent of ddof, so it validates the 1/m convention in
    # numerator and denominator together.
    np.testing.assert_allclose(rho, np.asarray(np.corrcoef(D))[0], rtol=1e-12)
    assert rho[0] == 1.0


def test_model_statistics_hand_computed() -> None:
    """Two models with three samples, computed by hand."""
    D = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    mean, std, rho = core.model_statistics(D)
    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(std, [np.sqrt(2 / 3), 2 * np.sqrt(2 / 3)])
    np.testing.assert_allclose(rho, [1.0, 1.0])


@pytest.mark.parametrize(
    "D",
    [np.zeros(5), np.ones((2, 1)), np.array([[1.0, np.nan], [1.0, 2.0]])],
)
def test_model_statistics_rejects_bad_input(D: np.ndarray) -> None:
    """Wrong shape, a single sample and non-finite values raise ``ValueError``."""
    with pytest.raises(ValueError):
        core.model_statistics(D)


def test_model_statistics_rejects_zero_variance() -> None:
    """A constant model has an undefined correlation."""
    with pytest.raises(ValueError, match="zero variance"):
        core.model_statistics(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))


# ---------------------------------------------------------------- select_models


def test_select_single_model_and_two_models() -> None:
    """A lone model is returned unchanged; a cheap correlated model is added."""
    assert core.select_models(3.0, [1.0], [10.0]) == ((0,), 9.0)
    # Hand-computed: delta = [0.19, 0.81], sum sqrt(w delta) = sqrt(1.9) + 0.9,
    # v = sigma^2 / 10 * sum^2 with sigma = 2.
    sel, v = core.select_models(2.0, [1.0, 0.9], [10.0, 1.0])
    assert sel == (0, 1)
    assert v == pytest.approx(4.0 / 10.0 * (np.sqrt(1.9) + 0.9) ** 2)


def test_select_three_models_all_kept_and_reordered() -> None:
    """Indices refer to the input order; the result is in decreasing-rho**2 order."""
    # Input order is (hf, rho 0.8, rho 0.95); rho order is therefore (0, 2, 1).
    sel, v = core.select_models(2.0, [1.0, 0.8, 0.95], [100.0, 1.0, 10.0])
    assert sel == (0, 2, 1)
    # Hand-computed: delta = [0.0975, 0.2625, 0.64], w = [100, 10, 1].
    expected = 4.0 / 100.0 * (np.sqrt(9.75) + np.sqrt(2.625) + np.sqrt(0.64)) ** 2
    assert v == pytest.approx(expected)


def test_select_drops_intermediate_model() -> None:
    """Model 1 is expensive and barely better than model 2, so it must be dropped."""
    # Hand-check of the admissibility condition for {0,1,2}:
    # w0 * delta1 = 100 * 0.0179 = 1.79 <= w1 * delta0 = 50 * 0.19 = 9.5 -> rejected.
    sel, v = core.select_models(2.0, [1.0, 0.9, 0.89], [100.0, 50.0, 1.0])
    assert sel == (0, 2)
    expected = 4.0 / 100.0 * (np.sqrt(100 * (1 - 0.89**2)) + 0.89) ** 2
    assert v == pytest.approx(expected)
    # The same problem with permuted input must select the same physical models.
    sel_p, v_p = core.select_models(2.0, [1.0, 0.89, 0.9], [100.0, 1.0, 50.0])
    assert sel_p == (0, 1)
    assert v_p == pytest.approx(v)


def test_select_keeps_only_high_fidelity_when_lf_is_useless() -> None:
    """An expensive, weakly correlated model gives no speed-up."""
    sel, v = core.select_models(2.0, [1.0, 0.3], [10.0, 9.0])
    assert sel == (0,)
    assert v == 4.0


def test_select_tie_break_is_deterministic() -> None:
    """Two identical low-fidelity models: the lower input index is chosen."""
    # delta for either {0,1} or {0,2} is identical, hence equal MSE; the first
    # enumerated subset ({0,1}) must win because replacement requires strict "<".
    # The full set {0,1,2} is inadmissible (delta_2 = 0).
    sel, _ = core.select_models(1.0, [1.0, 0.9, 0.9], [10.0, 1.0, 1.0])
    assert sel == (0, 1)


def test_select_vstar_equals_gamma_times_variance() -> None:
    """``v*`` is ``sigma1**2 * gamma`` of the selected set."""
    rho = np.array([1.0, 0.95, 0.9, 0.7])
    w = np.array([100.0, 20.0, 5.0, 1.0])
    sel, v = core.select_models(1.5, rho, w)
    idx = list(sel)
    assert v == pytest.approx(1.5**2 * core.gamma(rho[idx], w[idx]))


def test_select_validation() -> None:
    """Shape, rho_1 = 1, sigma and cost are validated."""
    with pytest.raises(ValueError, match="same length"):
        core.select_models(1.0, [1.0, 0.9], [1.0])
    with pytest.raises(ValueError, match=r"rho\[0\] = 1"):
        core.select_models(1.0, [0.9, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="sigma1"):
        core.select_models(0.0, [1.0], [1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        core.select_models(1.0, [1.0, 0.5], [1.0, 0.0])


# ---------------------------------------------------------------- allocate


def test_allocate_two_models_hand_computed() -> None:
    """Hand-computed 2-model allocation."""
    # delta = [0.19, 0.81]; u = [1, sqrt(0.81 / 0.19)] = [1, 6.5293];
    # m1 = 1000 / (10 + 6.5293) = 60.5 -> 60, m2 = 60.5 * 6.5293 = 395.0 -> 395.
    m, unspent = core.allocate([1.0, 0.9], [10.0, 1.0], 1000.0)
    assert m.tolist() == [60, 395]
    assert m.dtype == np.int64
    assert unspent == pytest.approx(1000.0 - (60 * 10 + 395 * 1))


def test_allocate_three_models_hand_computed() -> None:
    """3-model allocation: u_i = sqrt(w1 delta_i / (w_i delta_1))."""
    rho = np.array([1.0, 0.9, 0.8])
    w = np.array([100.0, 10.0, 1.0])
    delta = np.array([1 - 0.81, 0.81 - 0.64, 0.64])
    u = np.sqrt(w[0] * delta / (w * delta[0]))
    m1 = 5000.0 / np.dot(w, u)
    expected = np.floor(m1 * u).astype(int)
    m, unspent = core.allocate(rho, w, 5000.0)
    assert m.tolist() == expected.tolist() == [33, 100, 618]
    assert unspent == pytest.approx(5000.0 - np.dot(m, w))
    assert 0.0 <= unspent < w.sum() * 10


def test_allocate_budget_too_small_names_required_budget() -> None:
    """floor(m1) = 0 raises and states the minimal budget."""
    rho = np.array([1.0, 0.9])
    w = np.array([10.0, 1.0])
    # u_2 = sqrt((0.81 / 1) / (0.19 / 10)); cost of one hf evaluation = 10 + u_2.
    need = 10.0 + np.sqrt(0.81 * 10.0 / 0.19)
    with pytest.raises(ValueError, match="at least 16.5293"):
        core.allocate(rho, w, need - 0.1)
    m, _ = core.allocate(rho, w, need * (1 + 1e-9))
    assert m[0] == 1


def test_allocate_validation() -> None:
    """Unordered rho, violated cost/correlation condition and bad budgets raise."""
    with pytest.raises(ValueError, match="decreasing"):
        core.allocate([1.0, 0.8, 0.9], [10.0, 2.0, 1.0], 1e4)
    with pytest.raises(ValueError, match="budget"):
        core.allocate([1.0, 0.9], [10.0, 1.0], -1.0)
    with pytest.raises(ValueError, match="same length"):
        core.allocate([1.0, 0.9], [10.0], 100.0)
    # Not a selected set (model 1 expensive, nearly redundant): m would decrease.
    with pytest.raises(ValueError, match="not non-decreasing"):
        core.allocate([1.0, 0.9, 0.89], [100.0, 50.0, 1.0], 1e6)


# ---------------------------------------------------------------- control_variates


def test_control_variates_hand_computed() -> None:
    """alpha_i = rho_i sigma_1 / sigma_i."""
    alpha = core.control_variates([1.0, 0.9, 0.5], [2.0, 4.0, 1.0])
    np.testing.assert_allclose(alpha, [1.0, 0.45, 1.0])


def test_control_variates_non_monotone_input_order() -> None:
    """Alpha follows the model, not the position, when the input order is non-monotone."""
    rho = np.array([1.0, 0.8, 0.9])
    sigma = np.array([2.0, 4.0, 3.0])
    alpha = core.control_variates(rho, sigma)
    np.testing.assert_allclose(alpha, [1.0, 0.8 * 2 / 4, 0.9 * 2 / 3])

    # Permuting both arrays consistently permutes alpha identically.
    perm = [0, 2, 1]
    alpha_perm = core.control_variates(rho[perm], sigma[perm])
    np.testing.assert_allclose(alpha_perm, alpha[perm])

    # The legacy pairing (rho in rho order, std in name order) gives a different,
    # wrong answer: this is the bug of eval_estim.py:97 that this function fixes.
    legacy = (rho[perm] / sigma) * sigma[0]
    assert not np.allclose(legacy, alpha[perm])


def test_control_variates_validation() -> None:
    """Mismatched lengths, rho_1 != 1 and non-positive sigma are rejected."""
    with pytest.raises(ValueError, match="same length"):
        core.control_variates([1.0, 0.5], [1.0])
    with pytest.raises(ValueError, match=r"rho\[0\] = 1"):
        core.control_variates([0.9, 0.5], [1.0, 1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        core.control_variates([1.0, 0.5], [1.0, 0.0])


# ---------------------------------------------------------------- mfmc_estimate


def test_mfmc_estimate_matches_explicit_loop() -> None:
    """Eq. 19 against a literal translation with loops and one-based sums."""
    rng = np.random.default_rng(3)
    m = np.array([3, 7, 12])
    alpha = np.array([1.0, 0.7, 1.3])
    # Provide more samples than needed to check that the surplus is ignored.
    samples = [rng.normal(size=n) for n in (5, 9, 12)]

    s = sum(samples[0][j] for j in range(m[0])) / m[0]
    for i in (1, 2):
        own = sum(samples[i][j] for j in range(m[i])) / m[i]
        shared = sum(samples[i][j] for j in range(m[i - 1])) / m[i - 1]
        s += alpha[i] * (own - shared)
    assert core.mfmc_estimate(samples, m, alpha) == pytest.approx(s, rel=1e-13)


def test_mfmc_estimate_single_model_is_plain_mean() -> None:
    """With one model the estimator is the Monte Carlo mean of the first m samples."""
    x = np.array([1.0, 2.0, 3.0, 10.0])
    assert core.mfmc_estimate([x], np.array([3]), np.array([1.0])) == 2.0


def test_mfmc_estimate_validation() -> None:
    """Too few samples, decreasing m and length mismatches raise."""
    x = np.arange(5.0)
    with pytest.raises(ValueError, match="needs 6 samples"):
        core.mfmc_estimate([x, x], np.array([2, 6]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="non-decreasing"):
        core.mfmc_estimate([x, x], np.array([4, 2]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="same length"):
        core.mfmc_estimate([x], np.array([2, 3]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="integer"):
        core.mfmc_estimate([x, x], np.array([2.0, 3.0]), np.array([1.0, 1.0]))


# ---------------------------------------------------------------- rmse and gamma


def test_gamma_is_one_for_high_fidelity_alone() -> None:
    """Without low-fidelity models there is no gain."""
    assert core.gamma([1.0], [123.0]) == pytest.approx(1.0)


def test_gamma_hand_computed() -> None:
    """gamma = (sqrt(w1 d1) + sqrt(w2 d2))^2 / w1 for two models."""
    expected = (np.sqrt(10 * 0.19) + np.sqrt(0.81)) ** 2 / 10
    assert core.gamma([1.0, 0.9], [10.0, 1.0]) == pytest.approx(expected)


def test_rmse_equals_mc_rmse_for_single_model() -> None:
    """eta* = 1: MFMC degenerates to MC (Eq. 20 = Eq. 21)."""
    assert core.mfmc_rmse(2.5, [1.0], [40.0], 1e4) == pytest.approx(
        core.mc_rmse(2.5, 40.0, 1e4)
    )
    assert core.mc_rmse(2.0, 100.0, 400.0) == pytest.approx(1.0)  # sigma sqrt(w/p)


def test_rmse_ratio_is_sqrt_gamma() -> None:
    """RMSE_MFMC / RMSE_MC = sqrt(gamma), independent of budget."""
    rho = np.array([1.0, 0.95, 0.85])
    w = np.array([100.0, 10.0, 1.0])
    for p in (1e3, 1e6):
        ratio = core.mfmc_rmse(3.0, rho, w, p) / core.mc_rmse(3.0, w[0], p)
        assert ratio == pytest.approx(np.sqrt(core.gamma(rho, w)))


def test_rmse_and_gamma_validation() -> None:
    """Unordered models, bad budgets and bad sigma are rejected."""
    with pytest.raises(ValueError, match="ordered"):
        core.gamma([1.0, 0.8, 0.9], [10.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="budget"):
        core.mfmc_rmse(1.0, [1.0], [1.0], 0.0)
    with pytest.raises(ValueError, match="sigma1"):
        core.mfmc_rmse(-1.0, [1.0], [1.0], 1.0)
    with pytest.raises(ValueError, match="sigma"):
        core.mc_rmse(0.0, 1.0, 1.0)


# ---------------------------------------------------------------- property test


def test_estimator_unbiased_and_variance_matches_eq20() -> None:
    """
    Synthetic correlated Gaussian models: unbiased estimator, variance of Eq. 20.

    Model i is ``mu_i + s_i * (rho_i Z + sqrt(1 - rho_i**2) eps_i)``. Sample ``j`` of all
    models shares ``Z_j`` (nested samples, as in the real workflow), so the correlation
    with the high-fidelity model is exactly ``rho_i``. Population ``rho``, ``sigma`` are
    used for the allocation and ``alpha`` so that Eq. 20 is exact up to the rounding of m*.
    """
    rng = np.random.default_rng(20261015)
    mu = np.array([5.0, 4.0, 3.0])
    s = np.array([2.0, 2.5, 1.5])
    rho = np.array([1.0, 0.95, 0.85])
    w = np.array([100.0, 10.0, 1.0])
    budget = 2e4

    m, _ = core.allocate(rho, w, budget)
    alpha = core.control_variates(rho, s)

    n_rep = 3000
    est = np.empty(n_rep)
    for r in range(n_rep):
        z = rng.standard_normal(m[-1])
        # For i = 0, rho = 1 and the noise term vanishes (sqrt(0) = 0).
        samples = [
            mu[i]
            + s[i]
            * (
                rho[i] * z[: m[i]]
                + np.sqrt(1 - rho[i] ** 2) * rng.standard_normal(m[i])
            )
            for i in range(3)
        ]
        est[r] = core.mfmc_estimate(samples, m, alpha)

    # Exact variance for the integer allocation m with optimal alpha:
    # sigma1^2 [1/m1 - sum_i (1/m_{i-1} - 1/m_i) rho_i^2].
    inv = 1.0 / m
    var_exact = s[0] ** 2 * (inv[0] - np.sum((inv[:-1] - inv[1:]) * rho[1:] ** 2))
    var_eq20 = core.mfmc_rmse(s[0], rho, w, budget) ** 2

    # Rounding down m* spends slightly less than the budget, so Eq. 20 is a lower bound
    # that is accurate to well below the Monte Carlo error with m1 ~ 100.
    assert var_exact == pytest.approx(var_eq20, rel=0.03)

    # Unbiased: mean within 4 standard errors of the high-fidelity mean.
    se = np.sqrt(var_exact / n_rep)
    assert abs(est.mean() - mu[0]) < 4 * se
    # Empirical variance within 10 % (relative standard error is sqrt(2/n_rep) = 2.6 %).
    assert est.var(ddof=1) == pytest.approx(var_exact, rel=0.10)
    assert est.var(ddof=1) == pytest.approx(var_eq20, rel=0.12)
