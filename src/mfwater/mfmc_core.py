"""
Pure multifidelity Monte Carlo (MFMC) mathematics.

This module contains the numerics of the MFMC framework of paper I (Froembgen et al.,
J. Chem. Phys. 164, 024504 (2026)) and its Supporting Information (SI), Algorithms 2-4.
It operates on plain NumPy arrays only: there is no HDF5 access and no other I/O, so it
can be used by the legacy command line algorithms as well as by automated workflows.

Conventions
-----------
* All per-model arrays are in *model order*: index 0 is the high-fidelity model, so
  ``sigma[0]``, ``rho[0]`` (which must be exactly 1) and ``w[0]`` belong to the same
  model. All per-model arrays passed to one function must use the **same** order. Mixing
  orders (e.g. standard deviations in name order, correlations in rho order) silently
  produces wrong control-variate coefficients, which is why :func:`control_variates`
  takes both arrays and checks their shapes.
* ``rho[i]`` is the correlation coefficient of model ``i`` with the high-fidelity model,
  ``sigma[i]`` the standard deviation (``ddof = 0``, as in SI Algorithm 2), and ``w[i]``
  the cost of one evaluation of model ``i``. Budgets are in the same unit as ``w``.
* The ordering required by the allocation, the RMSE and gamma is
  ``rho[0]**2 > rho[1]**2 > ... > rho[-1]**2``. :func:`select_models` accepts any
  input order and returns the selected indices in this ordering.
* With ``rho[eta] := 0`` for the (non-existing) model after the last one, the
  differences ``delta_i = rho[i]**2 - rho[i+1]**2`` appear in all formulas.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Explicit aliases so that mypy also accepts them when run without NumPy installed
# (the pre-commit mypy hook uses an isolated environment).
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

# Tolerance for the requirement rho[0] == 1. Legacy files store rho[0] as exactly 1.0, so
# this only absorbs harmless rounding of values that went through other code paths.
_RHO1_ATOL = 1e-12


def _as_vector(x: npt.ArrayLike, name: str) -> FloatArray:
    """
    Convert an input to a finite one-dimensional float64 array.

    Parameters
    ----------
    x : array_like
        Input values.
    name : str
        Name of the argument, used in error messages.

    Returns
    -------
    numpy.ndarray
        The values as a one-dimensional ``float64`` array.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, is empty, or contains non-finite values.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"{name} must be a non-empty 1-D array, got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _check_rho(rho: FloatArray, name: str = "rho") -> None:
    """
    Check that correlation coefficients are valid for a high-fidelity reference.

    Parameters
    ----------
    rho : numpy.ndarray
        Correlation coefficients with the high-fidelity model.
    name : str
        Name of the argument, used in error messages.

    Raises
    ------
    ValueError
        If ``rho[0]`` is not 1 or any ``|rho|`` exceeds 1.
    """
    if abs(rho[0] - 1.0) > _RHO1_ATOL:
        raise ValueError(
            f"The first model must be the high-fidelity model with {name}[0] = 1, "
            f"got {rho[0]!r}."
        )
    if np.any(np.abs(rho) > 1.0 + 1e-12):
        raise ValueError(f"{name} must satisfy |rho| <= 1, got {rho!r}.")


def _check_same_length(**arrays: FloatArray) -> None:
    """
    Check that all given arrays have the same length.

    Parameters
    ----------
    **arrays : numpy.ndarray
        Named arrays to compare.

    Raises
    ------
    ValueError
        If the lengths differ.
    """
    sizes = {name: a.size for name, a in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"Inputs must have the same length, got {sizes}.")


def _check_positive(w: FloatArray, name: str = "w") -> None:
    """
    Check that costs are strictly positive.

    Parameters
    ----------
    w : numpy.ndarray
        Costs per model evaluation.
    name : str
        Name of the argument, used in error messages.

    Raises
    ------
    ValueError
        If any value is not strictly positive.
    """
    if np.any(w <= 0.0):
        raise ValueError(f"{name} must be strictly positive, got {w!r}.")


def _rho_differences(rho: FloatArray, strict: bool) -> FloatArray:
    """
    Compute ``rho[i]**2 - rho[i+1]**2`` with ``rho[eta] = 0``.

    Parameters
    ----------
    rho : numpy.ndarray
        Correlation coefficients in model order.
    strict : bool
        If True the squared correlations must be strictly decreasing, otherwise
        non-increasing is accepted.

    Returns
    -------
    numpy.ndarray
        The differences ``delta_i`` (SI Algorithms 3 and 4).

    Raises
    ------
    ValueError
        If the models are not ordered by decreasing squared correlation.
    """
    r2 = np.concatenate((rho**2, [0.0]))
    delta = r2[:-1] - r2[1:]
    bad = delta <= 0.0 if strict else delta < 0.0
    if np.any(bad):
        kind = "strictly decreasing" if strict else "non-increasing"
        raise ValueError(
            f"Models must be ordered by {kind} rho**2 (got rho = {rho!r}). "
            "Use select_models to obtain the ordering."
        )
    return delta


def model_statistics(D: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Compute mean, standard deviation and correlation with the high-fidelity model.

    Implements SI Algorithm 2 (lines 6-10). Standard deviation and correlation use
    ``ddof = 0``, i.e. a division by ``m`` and not ``m - 1``.

    Parameters
    ----------
    D : array_like
        Model evaluations of shape ``(eta, m)``. Row 0 is the high-fidelity model. All
        models share the same ``m`` samples (sample ``j`` of each model was computed with
        the same parameters and seeds). Data are converted to ``float64``.

    Returns
    -------
    mean : numpy.ndarray
        Monte Carlo estimates per model, shape ``(eta,)``.
    std : numpy.ndarray
        Standard deviations (``ddof = 0``), shape ``(eta,)``.
    rho : numpy.ndarray
        Correlation coefficients with model 0, shape ``(eta,)``; ``rho[0]`` is set to
        exactly 1.

    Raises
    ------
    ValueError
        If ``D`` is not two-dimensional, has fewer than two samples, contains non-finite
        values, or a model has zero variance.
    """
    arr = np.asarray(D, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError(f"D must have shape (eta, m), got {arr.shape}.")
    if arr.shape[1] < 2:
        raise ValueError("At least two samples per model are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("D must contain only finite values.")

    mean = arr.mean(axis=1)
    centred = arr - mean[:, None]
    std = np.sqrt(np.mean(centred**2, axis=1))
    if np.any(std == 0.0):
        raise ValueError("A model has zero variance; the correlation is undefined.")
    rho = np.mean(centred * centred[0], axis=1) / (std * std[0])
    # The self-correlation is exactly 1 by definition. Setting it avoids a rounding
    # residual that would trip the rho[0] == 1 checks downstream.
    rho[0] = 1.0
    return mean, std, rho


def select_models(
    sigma1: float, rho: npt.ArrayLike, w: npt.ArrayLike
) -> tuple[tuple[int, ...], float]:
    """
    Select the optimal subset of models (SI Algorithm 3).

    All subsets that contain the high-fidelity model (index 0) are enumerated. A subset is
    admissible if, in the ordering by decreasing ``rho**2``,
    ``w[i-1] * delta[i] > w[i] * delta[i-1]`` holds for all its members ``i >= 2``
    (``delta_i = rho_i**2 - rho_{i+1}**2``, ``rho_{eta+1} = 0``). The admissible subset
    with the smallest MSE ``v = sigma1**2 / w1 * (sum_i sqrt(w_i delta_i))**2`` wins.

    Tie-breaking (deterministic): the high-fidelity model is always first; remaining
    models are sorted by decreasing ``rho**2`` and, for equal ``rho**2``, by increasing
    input index. Subsets are enumerated by increasing size and, within one size, in
    lexicographic order of their positions in that ordering. A subset replaces the
    current best only if its MSE is *strictly* smaller, so among equal-MSE subsets the
    first enumerated one (the smaller, then the lexicographically earlier one) is kept.
    The search starts from the high-fidelity model alone with ``v* = sigma1**2``.

    Parameters
    ----------
    sigma1 : float
        Standard deviation of the high-fidelity model.
    rho : array_like
        Correlation coefficients with the high-fidelity model, shape ``(eta,)``, in any
        order but with ``rho[0] = 1``.
    w : array_like
        Cost per evaluation of each model, shape ``(eta,)``, same order as ``rho``.

    Returns
    -------
    indices : tuple of int
        Indices into the *input* arrays of the selected models, in the ordering by
        decreasing ``rho**2`` (starts with 0).
    v_star : float
        MSE of the selected set at a budget equal to the cost of one high-fidelity
        evaluation, ``p = w[0]``. For a budget ``p`` multiply by ``w[0] / p``.

    Raises
    ------
    ValueError
        If the inputs have inconsistent shapes, ``rho[0] != 1``, ``sigma1 <= 0`` or a
        cost is not positive.
    """
    rho_a = _as_vector(rho, "rho")
    w_a = _as_vector(w, "w")
    _check_same_length(rho=rho_a, w=w_a)
    _check_rho(rho_a)
    _check_positive(w_a)
    if not sigma1 > 0.0:
        raise ValueError(f"sigma1 must be positive, got {sigma1!r}.")

    eta = rho_a.size
    # Stable sort on -rho**2: equal values keep increasing input index.
    rest = sorted(range(1, eta), key=lambda i: (-(rho_a[i] ** 2), i))
    order = [0] + rest

    best: tuple[int, ...] = (0,)
    v_star = float(sigma1**2)

    # Enumerate subsets of positions 1..eta-1 by increasing size (lexicographic within).
    for size in range(1, eta):
        for combo in itertools.combinations(range(1, eta), size):
            positions = (0,) + combo
            idx = [order[p] for p in positions]
            rr = rho_a[idx]
            ww = w_a[idx]
            r2 = np.concatenate((rr**2, [0.0]))
            delta = r2[:-1] - r2[1:]

            # Cost/correlation admissibility condition (SI Alg. 3 line 7).
            admissible = all(
                ww[i - 1] * delta[i] > ww[i] * delta[i - 1] for i in range(1, len(idx))
            )
            if not admissible:
                continue

            v_m = float(sigma1**2 / ww[0] * np.sum(np.sqrt(ww * delta)) ** 2)
            if v_m < v_star:
                best = tuple(idx)
                v_star = v_m

    return best, v_star


def allocate(
    rho: npt.ArrayLike, w: npt.ArrayLike, budget: float
) -> tuple[IntArray, float]:
    """
    Compute the optimal number of evaluations per model (SI Algorithm 4, lines 1-14).

    With ``u_i = sqrt(w_1 delta_i / (w_i delta_1))``, the high-fidelity model is
    evaluated ``m_1 = budget / (w . u)`` times and model ``i`` ``m_i = m_1 u_i`` times;
    all values are rounded down. ``m_i`` is the *total* number of evaluations of model
    ``i`` (the samples of model ``i-1`` are a subset of those of model ``i``).

    Parameters
    ----------
    rho : array_like
        Correlation coefficients of the *selected* models, shape ``(eta,)``, ordered by
        strictly decreasing ``rho**2`` with ``rho[0] = 1``.
    w : array_like
        Cost per evaluation of the selected models, same order.
    budget : float
        Computational budget in the unit of ``w``.

    Returns
    -------
    m_star : numpy.ndarray
        Integer evaluations per model, shape ``(eta,)``, non-decreasing, ``m_star[0] >= 1``.
    unspent : float
        Budget not used because of rounding down, ``budget - sum(m_star * w)``.

    Raises
    ------
    ValueError
        If the inputs are inconsistent, the models are not ordered by strictly decreasing
        ``rho**2`` (or violate the cost/correlation condition, so that ``m_star`` would
        not be monotone), or ``floor(m_1) = 0``. In the last case the message states the
        minimal budget required for one high-fidelity evaluation.
    """
    rho_a = _as_vector(rho, "rho")
    w_a = _as_vector(w, "w")
    _check_same_length(rho=rho_a, w=w_a)
    _check_rho(rho_a)
    _check_positive(w_a)
    if not (np.isfinite(budget) and budget > 0.0):
        raise ValueError(f"budget must be positive and finite, got {budget!r}.")

    delta = _rho_differences(rho_a, strict=True)

    # Same operation order as the legacy eval-estimator so that the floor() of a
    # value that is an exact integer in theory gives the same result.
    ratio = delta / w_a
    u = np.sqrt(ratio / ratio[0])
    cost_per_hf = float(np.dot(w_a, u))
    m1 = budget / cost_per_hf
    if np.floor(m1) < 1:
        raise ValueError(
            f"Budget {budget:g} is too small: the high-fidelity model must be evaluated "
            f"at least once. A budget of at least {cost_per_hf:.6g} is needed for "
            f"these models."
        )
    m_star = np.floor(np.concatenate(([m1], u[1:] * m1))).astype(np.int64)
    if np.any(np.diff(m_star) < 0):
        raise ValueError(
            f"The evaluations {m_star.tolist()} are not non-decreasing. The models "
            "violate the cost/correlation condition of SI Algorithm 3; select the "
            "models with select_models first."
        )
    unspent = float(budget - np.dot(m_star, w_a))
    return m_star, unspent


def control_variates(rho: npt.ArrayLike, sigma: npt.ArrayLike) -> FloatArray:
    """
    Compute the control-variate coefficients ``alpha_i = rho_i * sigma_1 / sigma_i``.

    This is the corrected form of SI Algorithm 4 line 17. Both inputs must be in the
    **same** model order (index 0 = high fidelity). The legacy eval-estimator paired
    standard deviations in name order with correlations in rho order, which is only
    correct if both orders coincide.

    Parameters
    ----------
    rho : array_like
        Correlation coefficients with the high-fidelity model, ``rho[0] = 1``.
    sigma : array_like
        Standard deviations, same order and length as ``rho``.

    Returns
    -------
    numpy.ndarray
        Coefficients ``alpha``; ``alpha[0] = 1``.

    Raises
    ------
    ValueError
        If the shapes differ, ``rho[0] != 1`` or a standard deviation is not positive.
    """
    rho_a = _as_vector(rho, "rho")
    sigma_a = _as_vector(sigma, "sigma")
    _check_same_length(rho=rho_a, sigma=sigma_a)
    _check_rho(rho_a)
    _check_positive(sigma_a, "sigma")
    # Same operation order as the legacy code, (rho / sigma) * sigma_1, for bitwise
    # reproducibility of stored values.
    return (rho_a / sigma_a) * sigma_a[0]


def mfmc_estimate(
    samples: Sequence[npt.ArrayLike], m_star: npt.ArrayLike, alpha: npt.ArrayLike
) -> float:
    """
    Compute the MFMC estimator (Eq. 19, SI Algorithm 4 line 20).

    ``s = mean(f_1[:m_1]) + sum_{i>=2} alpha_i * (mean(f_i[:m_i]) - mean(f_i[:m_{i-1}]))``.
    Model ``i`` uses its first ``m_i`` samples, the subtracted mean uses the first
    ``m_{i-1}`` samples of the same model, i.e. the samples it shares with model ``i-1``.
    Samples beyond ``m_i`` are ignored.

    Parameters
    ----------
    samples : sequence of array_like
        Evaluations per model in model order; ``samples[i]`` has at least ``m_star[i]``
        entries. Data are converted to ``float64``.
    m_star : array_like of int
        Evaluations per model, non-decreasing with ``m_star[0] >= 1``.
    alpha : array_like
        Control-variate coefficients, see :func:`control_variates`.

    Returns
    -------
    float
        The MFMC estimate.

    Raises
    ------
    ValueError
        If the lengths are inconsistent, ``m_star`` is not a non-decreasing positive
        integer vector, or a model has fewer samples than ``m_star`` requires.
    """
    m_arr = np.asarray(m_star)
    if m_arr.ndim != 1 or m_arr.size == 0 or not np.issubdtype(m_arr.dtype, np.integer):
        raise ValueError("m_star must be a non-empty 1-D integer array.")
    alpha_a = _as_vector(alpha, "alpha")
    if not (len(samples) == m_arr.size == alpha_a.size):
        raise ValueError(
            f"samples ({len(samples)}), m_star ({m_arr.size}) and alpha "
            f"({alpha_a.size}) must have the same length."
        )
    if m_arr[0] < 1 or np.any(np.diff(m_arr) < 0):
        raise ValueError(
            f"m_star must be non-decreasing with m_star[0] >= 1, got {m_arr.tolist()}."
        )

    data = []
    for i, (s, m) in enumerate(zip(samples, m_arr)):
        a = np.asarray(s, dtype=np.float64)
        if a.ndim != 1 or a.size < m:
            raise ValueError(
                f"Model {i} needs {int(m)} samples but provides shape {a.shape}."
            )
        data.append(a)

    estimate = float(np.mean(data[0][: m_arr[0]]))
    for i in range(1, m_arr.size):
        mean_own = np.mean(data[i][: m_arr[i]])
        # Same samples that model i-1 used: this is what makes the term a control variate.
        mean_shared = np.mean(data[i][: m_arr[i - 1]])
        estimate += float(alpha_a[i] * (mean_own - mean_shared))
    return estimate


def _sum_sqrt_wdelta(rho: FloatArray, w: FloatArray) -> float:
    """
    Compute ``sum_i sqrt(w_i * delta_i)`` for models ordered by decreasing ``rho**2``.

    Parameters
    ----------
    rho : numpy.ndarray
        Correlation coefficients, ``rho[0] = 1``, ordered by decreasing ``rho**2``.
    w : numpy.ndarray
        Costs per evaluation.

    Returns
    -------
    float
        The sum.

    Raises
    ------
    ValueError
        On inconsistent or unordered inputs.
    """
    _check_same_length(rho=rho, w=w)
    _check_rho(rho)
    _check_positive(w)
    delta = _rho_differences(rho, strict=False)
    return float(np.sum(np.sqrt(w * delta)))


def mfmc_rmse(
    sigma1: float, rho: npt.ArrayLike, w: npt.ArrayLike, budget: float
) -> float:
    """
    Compute the RMSE of the optimal MFMC estimator (Eq. 20).

    ``RMSE = sigma1 / sqrt(budget) * sum_i sqrt(w_i * (rho_i**2 - rho_{i+1}**2))``.
    This is the continuous (non-rounded) optimum, as in the legacy ``mfmc`` algorithm.

    Parameters
    ----------
    sigma1 : float
        Standard deviation of the high-fidelity model.
    rho : array_like
        Correlation coefficients of the selected models, ordered by decreasing
        ``rho**2``, ``rho[0] = 1``.
    w : array_like
        Costs per evaluation, same order.
    budget : float
        Computational budget in the unit of ``w``.

    Returns
    -------
    float
        Root mean squared error of the estimator.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or ``sigma1`` or ``budget`` is not positive.
    """
    if not sigma1 > 0.0:
        raise ValueError(f"sigma1 must be positive, got {sigma1!r}.")
    if not (np.isfinite(budget) and budget > 0.0):
        raise ValueError(f"budget must be positive and finite, got {budget!r}.")
    total = _sum_sqrt_wdelta(_as_vector(rho, "rho"), _as_vector(w, "w"))
    return float(np.sqrt(sigma1**2 / budget * total**2))


def mc_rmse(sigma: float, w: float, budget: float) -> float:
    """
    Compute the RMSE of the plain Monte Carlo estimator of one model (Eq. 21).

    With ``budget / w`` evaluations, ``RMSE = sigma * sqrt(w / budget)``.

    Parameters
    ----------
    sigma : float
        Standard deviation of the model (the high-fidelity model for the comparison).
    w : float
        Cost per evaluation.
    budget : float
        Computational budget in the unit of ``w``.

    Returns
    -------
    float
        Root mean squared error of the MC estimator.

    Raises
    ------
    ValueError
        If any argument is not positive and finite.
    """
    for name, val in (("sigma", sigma), ("w", w), ("budget", budget)):
        if not (np.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {val!r}.")
    return float(sigma * np.sqrt(w / budget))


def gamma(rho: npt.ArrayLike, w: npt.ArrayLike) -> float:
    """
    Compute the MSE ratio of the MFMC estimator to the high-fidelity MC estimator (Eq. 22).

    ``gamma = (sum_i sqrt(w_i * delta_i))**2 / w_1`` with
    ``delta_i = rho_i**2 - rho_{i+1}**2``. It is independent of the budget; ``gamma < 1``
    means the MFMC estimator is cheaper for equal accuracy. For the high-fidelity model
    alone ``gamma = 1``. This equals ``mfmc_rmse**2 / mc_rmse**2`` and ``v_star / sigma1**2``
    of :func:`select_models`.

    Parameters
    ----------
    rho : array_like
        Correlation coefficients of the models, ordered by decreasing ``rho**2``,
        ``rho[0] = 1``.
    w : array_like
        Costs per evaluation, same order.

    Returns
    -------
    float
        The ratio gamma.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or the models are not ordered.
    """
    w_a = _as_vector(w, "w")
    total = _sum_sqrt_wdelta(_as_vector(rho, "rho"), w_a)
    return float(total**2 / w_a[0])
