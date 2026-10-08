"""Multifidelity Delayed Acceptance MCMC (MFDA) for MFWater.

The algorithm follows the multilevel delayed acceptance scheme (MLDA) of
Lykkegaard et al., arXiv:2202.03876.

Levels are numbered ``1..eta`` with **level 1 the finest** (largest number of
molecules) and level ``eta`` the coarsest. Every step on a level ``i < eta``
runs a coarse subchain of fixed length on level ``i + 1``, picks one of its
states uniformly as proposal and corrects it with a delayed-acceptance step.
The coarsest level uses a plain Metropolis-Hastings (MH) step with a Gaussian
random-walk proposal.

Only the components of the parameter vector ``theta = (eps_OO, sigma_OO, q_O)``
selected via the free mask are calibrated; the others stay exactly at their
prior mean (reduced parameter space).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..argparser import constants

#: Signature of the forward model: ``forward(n_molecules, theta) -> D``.
ForwardCallable = Callable[[int, np.ndarray], float]


@dataclass(frozen=True, eq=False)
class MFDAConfig:
    """Immutable settings of the MFDA algorithm.

    Parameters
    ----------
    molecules : tuple[int, ...]
        Number of molecules ``N_1 > N_2 > ... > N_eta`` per level (level 1 = finest).
    subchain_lengths : tuple[int, ...]
        Fixed subchain lengths ``M~_2 ... M~_eta``; entry ``i - 2`` belongs to level ``i``.
    means : np.ndarray
        Prior means ``mu`` (also the value of all non-calibrated components), shape ``(3,)``.
    kernel_noise : np.ndarray
        Standard deviations ``s`` of prior and proposal kernel, shape ``(3,)``.
        Prior and kernel share the covariance ``Sigma = diag(s**2)``.
    free_mask : np.ndarray
        Boolean array of shape ``(3,)``, ``True`` for calibrated components.
    print_level : int
        ``0`` = silent, ``1`` = debug output.

    Raises
    ------
    ValueError
        If array shapes are inconsistent, no component is free, or lengths do not match.
    """

    molecules: tuple[int, ...]
    subchain_lengths: tuple[int, ...]
    means: np.ndarray
    kernel_noise: np.ndarray
    free_mask: np.ndarray
    print_level: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if len(self.molecules) < 2:
            raise ValueError("At least two levels (models) are required.")
        if len(self.subchain_lengths) != len(self.molecules) - 1:
            raise ValueError(
                f"Expected {len(self.molecules) - 1} subchain lengths, "
                f"got {len(self.subchain_lengths)}."
            )
        if not (self.means.shape == self.kernel_noise.shape == self.free_mask.shape):
            raise ValueError("means, kernel_noise and free_mask must share one shape.")
        if not np.any(self.free_mask):
            raise ValueError(
                "No free parameter: at least one component must be calibrated."
            )

    @property
    def n_levels(self) -> int:
        """Number of levels ``eta``."""
        return len(self.molecules)

    def subchain_length(self, level: int) -> int:
        """Return the fixed subchain length ``M~_level`` (``level >= 2``).

        Parameters
        ----------
        level : int
            Level (1-based) on which the subchain runs.

        Returns
        -------
        int
            Subchain length on that level.
        """
        return self.subchain_lengths[level - 2]


@dataclass
class ChainState:
    """Storage of all samples ``X_i`` and proposals ``X'_i`` per level.

    Index ``i - 1`` belongs to level ``i``. The proposal list of the coarsest
    level stays empty (no coarser level exists).

    Parameters
    ----------
    samples : list[list[np.ndarray]]
        Chain samples per level.
    proposals : list[list[np.ndarray]]
        Proposals per level (one per sample for levels ``i < eta``).
    """

    samples: list[list[np.ndarray]]
    proposals: list[list[np.ndarray]]

    @classmethod
    def empty(cls, n_levels: int) -> ChainState:
        """Create an empty state.

        Parameters
        ----------
        n_levels : int
            Number of levels.

        Returns
        -------
        ChainState
            State with empty lists for every level.
        """
        return cls([[] for _ in range(n_levels)], [[] for _ in range(n_levels)])

    def lengths(self) -> tuple[list[int], list[int]]:
        """Return the current list lengths.

        Returns
        -------
        tuple[list[int], list[int]]
            Lengths of the sample lists and of the proposal lists, per level.
        """
        return (
            [len(s) for s in self.samples],
            [len(p) for p in self.proposals],
        )

    def truncate_front(self, n_samples: list[int], n_proposals: list[int]) -> None:
        """Discard the first entries of every list (burn-in).

        Parameters
        ----------
        n_samples : list[int]
            Number of leading samples to drop per level.
        n_proposals : list[int]
            Number of leading proposals to drop per level.
        """
        for i in range(len(self.samples)):
            del self.samples[i][: n_samples[i]]
            del self.proposals[i][: n_proposals[i]]


def mfda_step(
    current: np.ndarray,
    level: int,
    config: MFDAConfig,
    state: ChainState,
    forward: ForwardCallable,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perform one delayed-acceptance step on a level ``i < eta``.

    Runs a coarse subchain of fixed length ``M~_{i+1}`` on level ``i + 1``
    (always to its full length, required for an unbiased multilevel estimator),
    draws ``k ~ U{1..M~_{i+1}}``, takes the ``k``-th subchain state as proposal
    and accepts or rejects it with the delayed-acceptance probability.

    Parameters
    ----------
    current : np.ndarray
        Current sample on this level.
    level : int
        Level (1-based) of the step, ``1 <= level < eta``.
    config : MFDAConfig
        Algorithm settings.
    state : ChainState
        Storage that is appended to.
    forward : ForwardCallable
        Forward model ``forward(n_molecules, theta) -> D``.
    rng : np.random.Generator
        The single random number generator of the algorithm.

    Returns
    -------
    np.ndarray
        New current sample (accepted proposal or ``current``).
    """
    n_sub = config.subchain_length(level + 1)

    # the coarse subchain starts from the current sample
    if level + 1 == config.n_levels:
        subchain = chemical_mh(current, level + 1, n_sub, config, state, forward, rng)
    else:
        subchain = multifidelity_markov_chain(
            current, level + 1, n_sub, config, state, forward, rng
        )

    # k is drawn after the full subchain is generated; the draw count is fixed
    k = int(rng.integers(1, n_sub + 1))
    proposal = subchain[k - 1]
    state.proposals[level - 1].append(proposal.copy())

    # u is drawn in every step, so the RNG stream does not depend on the branch taken
    u = rng.random()

    # identical to current: the coarse level rejected everything up to k, so reject
    if not np.array_equal(proposal, current):
        p = acceptance_probability(
            config.molecules[level - 1],
            config.molecules[level],
            current,
            proposal,
            forward,
        )
        if u <= p:
            current = proposal

    state.samples[level - 1].append(current.copy())
    return current


def multifidelity_markov_chain(
    initial_sample: np.ndarray,
    level: int,
    length: int,
    config: MFDAConfig,
    state: ChainState,
    forward: ForwardCallable,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate a chain of fixed length on a level ``i < eta``.

    Parameters
    ----------
    initial_sample : np.ndarray
        Sample from which the chain starts.
    level : int
        Level (1-based), ``1 <= level < eta``.
    length : int
        Number of steps on this level.
    config : MFDAConfig
        Algorithm settings.
    state : ChainState
        Storage that is appended to (samples of this and all coarser levels).
    forward : ForwardCallable
        Forward model ``forward(n_molecules, theta) -> D``.
    rng : np.random.Generator
        The single random number generator of the algorithm.

    Returns
    -------
    list[np.ndarray]
        The samples generated on this level (the subchain), in order.
    """
    current = initial_sample
    subchain: list[np.ndarray] = []

    if config.print_level == 1:
        print(f"MFDA:       level {level}, chain length {length}  start.")

    for _ in range(length):
        current = mfda_step(current, level, config, state, forward, rng)
        subchain.append(current)

    if config.print_level == 1:
        print(f"MFDA:       level {level}, chain length {length}  end.")

    return subchain


def chemical_mh(
    initial_sample: np.ndarray,
    level: int,
    length: int,
    config: MFDAConfig,
    state: ChainState,
    forward: ForwardCallable,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Metropolis-Hastings on the coarsest level with the chemical forward model.

    Parameters
    ----------
    initial_sample : np.ndarray
        Sample from which the chain starts.
    level : int
        The coarsest level ``eta``.
    length : int
        Number of MH steps.
    config : MFDAConfig
        Algorithm settings.
    state : ChainState
        Storage that is appended to.
    forward : ForwardCallable
        Forward model ``forward(n_molecules, theta) -> D``.
    rng : np.random.Generator
        The single random number generator of the algorithm.

    Returns
    -------
    list[np.ndarray]
        The generated samples, in order.
    """
    mols = config.molecules[level - 1]
    current = initial_sample
    subchain: list[np.ndarray] = []

    if config.print_level == 1:
        print(f"ChemicalMH: level {level}, chain length {length}  start.")

    for _ in range(length):
        proposal = proposal_kernel(current, config.kernel_noise, rng, config.free_mask)

        # D of (N, theta) is looked up / computed by the forward callable (cached there)
        d_prop = forward(mols, proposal)
        d_curr = forward(mols, current)

        # log space for numerical stability; the Gaussian kernel is symmetric and cancels
        log_r = (
            log_likelihood(d_prop)
            + log_prior(proposal, config.means, config.kernel_noise, config.free_mask)
            - log_likelihood(d_curr)
            - log_prior(current, config.means, config.kernel_noise, config.free_mask)
        )
        p = 1.0 if log_r >= 0 else float(np.exp(log_r))

        # u is drawn in every step so the RNG stream is independent of p
        u = rng.random()
        if u <= p:
            current = proposal

        state.samples[level - 1].append(current.copy())
        subchain.append(current)

    if config.print_level == 1:
        print(f"ChemicalMH: level {level}, chain length {length}  end.")

    return subchain


def acceptance_probability(
    n_fine: int,
    n_coarse: int,
    current_sample: np.ndarray,
    proposal: np.ndarray,
    forward: ForwardCallable,
) -> float:
    """Delayed-acceptance probability of a proposal from the coarser level.

    ``p = min(1, L_i(theta') L_{i+1}(theta) / (L_i(theta) L_{i+1}(theta')))``.
    The prior cancels and is not included.

    Parameters
    ----------
    n_fine : int
        Number of molecules on the finer level ``i``.
    n_coarse : int
        Number of molecules on the coarser level ``i + 1``.
    current_sample : np.ndarray
        Current sample ``theta``.
    proposal : np.ndarray
        Proposed sample ``theta'``.
    forward : ForwardCallable
        Forward model ``forward(n_molecules, theta) -> D``.

    Returns
    -------
    float
        Acceptance probability in ``[0, 1]``.
    """
    # four forward evaluations; repeated ones are served by the cache of `forward`
    log_fine_prop = log_likelihood(forward(n_fine, proposal))
    log_fine_curr = log_likelihood(forward(n_fine, current_sample))
    log_coarse_prop = log_likelihood(forward(n_coarse, proposal))
    log_coarse_curr = log_likelihood(forward(n_coarse, current_sample))

    log_r = log_fine_prop + log_coarse_curr - log_fine_curr - log_coarse_prop
    return 1.0 if log_r >= 0 else float(np.exp(log_r))


def proposal_kernel(
    kernel_means: np.ndarray,
    kernel_noise: np.ndarray,
    rng: np.random.Generator,
    free_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian random-walk kernel ``N(kernel_means, diag(kernel_noise**2))``.

    Parameters
    ----------
    kernel_means : np.ndarray
        Centre of the Gaussian (current sample, or ``mu`` for the initial sample).
    kernel_noise : np.ndarray
        Standard deviations, same shape as ``kernel_means``.
    rng : np.random.Generator
        The single random number generator of the algorithm.
    free_mask : np.ndarray, optional
        Boolean mask of components to perturb; the others are copied unchanged.
        By default all components are perturbed.

    Returns
    -------
    np.ndarray
        The new sample.

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
    """
    kernel_means = np.asarray(kernel_means, dtype=float)
    kernel_noise = np.asarray(kernel_noise, dtype=float)
    mask = (
        np.ones(kernel_means.shape, dtype=bool)
        if free_mask is None
        else np.asarray(free_mask, dtype=bool)
    )
    if not (kernel_means.shape == kernel_noise.shape == mask.shape):
        raise ValueError(
            "kernel_means, kernel_noise and free_mask must have the same shape."
        )

    # draw only for free components: fixed components stay exactly at their value
    proposal = kernel_means.copy()
    proposal[mask] = rng.normal(loc=kernel_means[mask], scale=kernel_noise[mask])
    return proposal


def log_likelihood(input_val: float) -> float:
    """Logarithmic likelihood of a diffusion coefficient given the reference data.

    ``log L(D) = -0.5 sum_h (D - D_h^exp)^2 / sigma_exp^2``, with ``sigma_exp``
    the standard deviation of the reference values.

    Parameters
    ----------
    input_val : float
        Diffusion coefficient in 1e-9 m^2/s.

    Returns
    -------
    float
        The logarithmic likelihood.
    """
    reference_values = constants.TARGET_DIFFUSION_COEFFICIENT

    return float(
        -0.5
        * np.sum((input_val - reference_values) ** 2)
        / (reference_values.std() ** 2)
    )


def log_prior(
    sample: np.ndarray,
    means: np.ndarray,
    kernel_noise: np.ndarray,
    free_mask: np.ndarray,
) -> float:
    """Logarithmic Gaussian prior ``N(mu, Sigma)`` over the free components.

    Parameters
    ----------
    sample : np.ndarray
        Parameter sample.
    means : np.ndarray
        Prior means ``mu``.
    kernel_noise : np.ndarray
        Prior standard deviations.
    free_mask : np.ndarray
        Boolean mask of calibrated components.

    Returns
    -------
    float
        The logarithmic prior density.

    Raises
    ------
    ValueError
        If the shapes differ or no component is free.
    """
    if not (sample.shape == means.shape == kernel_noise.shape == free_mask.shape):
        raise ValueError(
            f"sample {sample.shape}, means {means.shape}, kernel_noise "
            f"{kernel_noise.shape} and free_mask {free_mask.shape} must share one shape."
        )
    if not np.any(free_mask):
        raise ValueError("No free parameter: the prior is undefined.")

    # Reduced parameter space: non-calibrated components are not random variables
    # (they are fixed at mu by construction), so they carry no prior density and
    # are excluded here instead of being treated as a Dirac measure.
    s = kernel_noise[free_mask]
    z = (sample[free_mask] - means[free_mask]) / s
    return float(np.sum(-0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * z**2))


def mfda_estimator(
    samples: list[np.ndarray], proposals: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Multilevel estimator ``sum_{i<eta} [mean(X_i) - mean(X'_i)] + mean(X_eta)``.

    Parameters
    ----------
    samples : list[np.ndarray]
        Per level ``i`` (index ``i - 1``) the samples ``X_i``, shape ``(K_i, d)``.
    proposals : list[np.ndarray]
        Per level the proposals ``X'_i``, shape ``(K_i, d)``; the coarsest level
        entry is ignored (may be empty).

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        The estimator and the contribution of each level. Level 1 (finest) is
        the first entry; the last entry is the coarsest-level mean.

    Raises
    ------
    ValueError
        If a level has no samples or sample/proposal counts differ.
    """
    n_levels = len(samples)
    contributions: list[np.ndarray] = []
    for i in range(n_levels):
        if len(samples[i]) == 0:
            raise ValueError(f"No samples on level {i + 1}.")
        mean_x = np.mean(samples[i], axis=0)
        if i == n_levels - 1:
            contributions.append(mean_x)
        else:
            if len(proposals[i]) != len(samples[i]):
                raise ValueError(
                    f"Level {i + 1}: {len(samples[i])} samples but "
                    f"{len(proposals[i])} proposals."
                )
            contributions.append(mean_x - np.mean(proposals[i], axis=0))
    return np.sum(contributions, axis=0), contributions
