"""Multifidelity Markov Chain Monte Carlo (MCMC) for MFWater."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..argparser import constants
from .forward_model import forward_model_dummy


def multifidelity_markov_chain(
    initial_sample: np.ndarray,
    markov_chain: np.ndarray,
    proposed_samples: np.ndarray,
    diff_computed: dict[Any, float],
    molecules: list[int],
    fidelity: int,
    length: int,
    max_length: int,
    fcount: list[int],
    print_level: int = 0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[Any, float]]:
    """Generate a Markov chain for a given fidelity level.

    Parameters
    ----------
    initial_sample: np.ndarray
        The sample to start the Markov chain.
    markov_chain : np.ndarray
        The entire Markov chain object, which is initially empty and gets filled with the generated samples for fidelity level specified.
    proposed_samples : np.ndarray
        The object to store the proposed samples for the coarser fidelity level.
    diff_computed: dict
        Stores already computed diffusion coefficients along with their number of molecules and parameters.
    molecules: list[int]
        Molecule numbers for each model.
    fidelity : int
        The fidelity level for which the Markov chain is generated in the current function call.
    length: int
        Length of the Markov chain to be generated.
    max_length: int
        Maximum length of the next Markov chain (given to enable recursive function calls).
    fcount: list[int]
        Counter / multiindex that keeps track of how often each function has been carried out.
    print_level: int
        Optional input specifying the print level (default 0).
        0 = no print.
        1 = debug.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, list[int], dict
        The generated subchain, entire Markov chain and the proposed samples, as well as the function counter, and the stored diffusion coeffients.
    """

    # initialize variables
    n_models = len(molecules)
    current = initial_sample
    subchain = np.zeros((length, 3), dtype=np.float64)

    if print_level == 1:
        print(
            f"MFMCMC:     f^{fidelity},  fcount: {fcount},  Chain length: {length}   start."
        )

    # generate the Markov chain
    for j in range(length):

        # draw random number for the length of the next Markov chain
        l = np.random.randint(1, max_length + 1)

        # if we are at the lowest fidelity, use chemical Metropolis-Hastings
        if fidelity == n_models - 1:

            subchain_return, markov_chain, fcount, diff_computed = chemical_mh(
                current,
                markov_chain,
                diff_computed,
                molecules[n_models - 1],
                l,
                max_length,
                fcount,
                **kwargs,
            )

        # otherwise, use recursion
        else:

            subchain_return, markov_chain, proposed_samples, fcount, diff_computed = (
                multifidelity_markov_chain(
                    current,
                    markov_chain,
                    proposed_samples,
                    diff_computed,
                    molecules,
                    fidelity + 1,
                    l,
                    max_length,
                    fcount,
                    **kwargs,
                )
            )

        # draw random number
        num = np.random.randint(0, l)

        # update proposal
        read_idx = [num, slice(None)]
        proposal = _read_from_markov_chain(subchain_return, read_idx)

        # store proposed sample
        write_idx = [fidelity - 1, (fcount[fidelity] - 1) * max_length + j, slice(None)]
        proposed_samples = _write_to_markov_chain(proposed_samples, proposal, write_idx)

        # check if proposal was rejected by low-fidelity filter
        if not np.array_equal(proposal, current):

            # compute acceptance probability
            alpha, diff_computed = acceptance_probability(
                molecules[fidelity - 1],
                molecules[fidelity],
                current,
                proposal,
                diff_computed,
                **kwargs,
            )

            # draw random number from Uniform(0,1) and accept or reject proposal
            if np.random.rand() < alpha:
                current = proposal

        # sample got rejected already
        else:
            pass

        # store current sample in subchain
        subchain[j, :] = current

        # store the current sample in the Markov chain
        write_idx = [fidelity - 1, (fcount[fidelity] - 1) * max_length + j, slice(None)]
        markov_chain = _write_to_markov_chain(markov_chain, current, write_idx, 0)

    # increase counter
    fcount[fidelity - 1] += 1

    if print_level == 1:
        print(
            f"MFMCMC:     f^{fidelity},  fcount: {fcount},  Chain length: {length}     end."
        )

    return subchain, markov_chain, proposed_samples, fcount, diff_computed


def chemical_mh(
    initial_sample: np.ndarray,
    markov_chain: np.ndarray,
    diff_computed: dict[Any, float],
    mols: int,
    length: int,
    max_length: int,
    fcount: list[int],
    print_level: int = 0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[Any, float]]:
    """Chemical version of the Metropolis Hastings algorithm.

    Parameters
    ----------
    initial_sample: np.ndarray
        The sample to start the Markov chain.
    markov_chain : np.ndarray
        The entire Markov chain object, which is initially empty and gets filled with the generated samples for fidelity level specified.
    diff_computed: dict
        Stores already computed diffusion coefficients along with their number of molecules and parameters.
    mols: int
        Number of molecules to run MD simulations for.
    length: int
        Length of the Markov chain to be generated.
    max_length: int
        Maximum length of the next Markov chain (given to enable recursive function calls).
    fcount: list[int]
        Counter / multiindex that keeps track of how often each function has been carried out.
    print_level: int
        Optional input specifying the print level (default 0).
        0 = no print.
        1 = debug.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, list[int], dict
        The generated subchain, entire Markov chain and the proposed samples, as well as the function counter and the stored diffusion coefficients.
    """

    if print_level == 1:
        print(
            f"ChemicalMH: f^{len(fcount)},  fcount: {fcount},  Chain length: {length}   start."
        )

    # initial sample
    current = initial_sample

    # store subchain
    subchain = np.zeros((length, 3), dtype=np.float64)

    for j in range(length):

        # propose new sample
        proposal = proposal_kernel(current, **kwargs)

        # compute diffusion coefficient for both the current sample and the proposed sample
        coeffs: list[float] = []
        pair = [proposal, current]
        for sample in pair:
            key = (mols, sample[0], sample[1], sample[2])

            # check if sample was already computed and reuse if so
            if key not in diff_computed:
                d_sample = forward_model_dummy(mols, sample)
                diff_computed[key] = d_sample
            else:
                d_sample = diff_computed[key]

            coeffs.append(d_sample)

        # compute acceptance probability
        # use logarithmic formula for numerical stability
        log_r = (
            log_likelihood(coeffs[0])
            + log_prior(pair[0], **kwargs)
            - log_likelihood(coeffs[1])
            - log_prior(pair[1], **kwargs)
        )

        if log_r >= 0:
            alpha = 1.0
        else:
            alpha = np.exp(log_r)

        # debug
        # print(f"alpha_MH = {alpha}")

        # accept or reject the proposed sample
        if np.random.rand() < alpha:
            current = proposal

        # store the current sample in the Markov chain
        write_idx = [
            len(fcount) - 1,
            max_length * fcount[len(fcount) - 1] + j,
            slice(None),
        ]
        markov_chain = _write_to_markov_chain(markov_chain, current, write_idx)

        # append sample to subchain
        subchain[j, :] = current

    fcount[-1] += 1

    if print_level == 1:
        print(
            f"ChemicalMH: f^{len(fcount)},  fcount: {fcount},  Chain length: {length}     end."
        )

    return subchain, markov_chain, fcount, diff_computed


def acceptance_probability(
    n_molecules_1: int,
    n_molecules_2: int,
    current_sample: np.ndarray,
    proposal: np.ndarray,
    diff_computed: dict[Any, float],
    **kwargs: Any,
) -> tuple[float, dict[Any, float]]:
    """Compute the acceptance probability for a proposed sample of parameters based on the likelihoods at both fidelity levels.

    Parameters
    ----------
    n_molecules_1 : int
        Number of molecules in the system for the higher fidelity level.
    n_molecules_2 : int
        Number of molecules in the system for the lower fidelity level.
    current_sample : np.ndarray
        The current sample of parameters in the Markov chain.
    proposal : np.ndarray
        The proposed sample of parameters for which the acceptance probability is computed.

    Returns
    -------
    float, dict
        The acceptance probability for the proposed sample, and the stored diffusion coefficients.
    """

    # iterate over the two fidelity levels and compute the diffusion coefficient for both the current sample and the proposed sample at both fidelity levels, then compute the acceptance probability based on the likelihoods and priors at both fidelity levels
    mols = [n_molecules_1, n_molecules_2]
    params = [current_sample, proposal]

    # use logarithmic formula for numerical stability
    log_like = np.zeros((2, 2), dtype=float)

    for n in range(len(mols)):
        for a in range(len(params)):

            key = (mols[n], params[a][0], params[a][1], params[a][2])
            if key not in diff_computed:
                d = forward_model_dummy(mols[n], params[a])
                diff_computed[key] = d
            else:
                d = diff_computed[key]

            log_like[n, a] = log_likelihood(d)

    log_r = log_like[0, 1] + log_like[1, 0] - log_like[0, 0] - log_like[1, 1]

    if log_r >= 0:
        alpha = 1.0
    else:
        alpha = np.exp(log_r)

    # debug
    # print(f"alpha_AP = {alpha}")

    return alpha, diff_computed


def proposal_kernel(
    kernel_means: np.ndarray,
    kernel_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Proposal kernel for the Markov chain.

    Parameters
    ----------
    means : np.ndarray
        The mean values for Gaussians from which the new sample is proposed.
    noise : np.ndarray
        The standard deviations for the Gaussians from which the new sample is proposed. By default, it is set to 1/300 of the corresponding mean value for the LJ parameters and 1/50 of the corresponding mean value for the charges.

    Returns
    -------
    np.ndarray
        The array with random samples.

    Raises
    ------
    ValueError
        If the input arrays `means` and `noise` do not have the same shape.
    """

    kernel_means = np.asarray(kernel_means, dtype=float)
    if kernel_noise is None:
        kernel_noise = np.array(
            [
                constants.OPC3_EPSILON_OO * constants.NOISE_LJPARAMS,
                constants.OPC3_SIGMA_OO * constants.NOISE_LJPARAMS,
                abs(constants.OPC3_CHARGE_O) * constants.NOISE_CHARGES,
            ],
            dtype=float,
        )
    else:
        kernel_noise = np.asarray(kernel_noise, dtype=float)

    # check whether kernel_means and kernel_noise have the same dimensions
    if kernel_means.shape != kernel_noise.shape:
        raise ValueError(
            "The input arrays kernel_means and kernel_noise must have the same shape."
        )

    # create a random number generator
    rng = np.random.default_rng()

    # For fixed parameters (zero noise), return exactly the mean; for varying parameters, sample normally
    proposal = rng.normal(loc=kernel_means, scale=kernel_noise)

    # Ensure fixed parameters are exactly at their means (not just numerically close)
    fixed_mask = kernel_noise == 0
    if np.any(fixed_mask):
        proposal[fixed_mask] = kernel_means[fixed_mask]

    return proposal


def log_likelihood(input_val: float) -> float:
    """Calculate the logarithmic likelihood of a proposed sample of parameters based on the diffusion coefficient they yield compared to the experimental data.

    Parameters
    ----------
    input_val : float
        The diffusion coefficient obtained from a simulation with the proposed parameters.

    Returns
    -------
    float
        The logarithmic likelihood of the proposed sample.
    """

    # load experimental data
    reference_values = constants.TARGET_DIFFUSION_COEFFICIENT

    logl = (
        -0.5
        * np.sum((input_val - reference_values) ** 2)
        / (reference_values.std() ** 2)
    )

    return logl


def log_prior(
    sample: np.ndarray,
    means: np.ndarray | None = None,
    kernel_noise: np.ndarray | None = None,
    **kwargs: Any,
) -> float:
    """Calculate the logarithmic prior probability of a proposed sample of parameters based on the initial distribution.

    Parameters
    ----------
    sample : np.ndarray
        The proposed sample of parameters.
    means : np.ndarray
        The mean values for Gaussians from which the new sample is proposed. By default, it is set to the standard values of the LJ parameters and charges of the OPC3 water model.
    kernel_noise : np.ndarray
        The standard deviations for the Gaussians from which the new sample is proposed. By default, it is set to 1/300 of the corresponding mean value for the LJ parameters and 1/50 of the corresponding mean value for the charges.

    Returns
    -------
    float
        The logarithmic prior probability of the proposed sample.

    Raises
    ------
    ValueError
        If the input arrays `sample`, `means`, and `kernel_noise` do not have the same shape.
    """

    # Support both 'kernel_noise' and 'noise' for backwards compatibility
    if kernel_noise is None and "noise" in kwargs:
        kernel_noise = kwargs["noise"]

    if kernel_noise is None:
        kernel_noise = np.array(
            [
                constants.OPC3_EPSILON_OO * constants.NOISE_LJPARAMS,
                constants.OPC3_SIGMA_OO * constants.NOISE_LJPARAMS,
                abs(constants.OPC3_CHARGE_O) * constants.NOISE_CHARGES,
            ],
            dtype=float,
        )
    else:
        kernel_noise = np.asarray(kernel_noise, dtype=float)

    if means is None:
        means = np.array(
            [
                constants.OPC3_EPSILON_OO,
                constants.OPC3_SIGMA_OO,
                constants.OPC3_CHARGE_O,
            ],
            dtype=float,
        )
    else:
        means = np.asarray(means, dtype=float)

    # check whether sample and kernel_noise have the same dimensions
    if sample.shape != kernel_noise.shape or sample.shape != means.shape:
        raise ValueError(
            f"The input arrays sample {sample.shape}, kernel_noise {kernel_noise.shape}, and means {means.shape} must have the same shape."
        )

    # Handle fixed parameters (zero noise): they must equal their means
    # Otherwise return -inf (impossible state)
    fixed_mask = kernel_noise == 0
    if np.any(fixed_mask):
        if not np.allclose(sample[fixed_mask], means[fixed_mask]):
            return -np.inf
        # For fixed parameters, contribute 0 to log probability
        # For varying parameters, compute normal Gaussian prior
        varying_mask = ~fixed_mask
        if np.any(varying_mask):
            logp = np.sum(
                -0.5 * np.log(2 * np.pi)
                - np.log(kernel_noise[varying_mask])
                - 0.5
                * (
                    (sample[varying_mask] - means[varying_mask])
                    / kernel_noise[varying_mask]
                )
                ** 2
            )
        else:
            # All parameters are fixed
            logp = 0.0
    else:
        # All parameters vary, use standard Gaussian prior
        logp = np.sum(
            -0.5 * np.log(2 * np.pi)
            - np.log(kernel_noise)
            - 0.5 * ((sample - means) / kernel_noise) ** 2
        )

    return logp


def _format_indices(indices: list[object]) -> str:
    """Format indices for display, replacing slice(None) with :"""
    formatted = []
    for idx in indices:
        if isinstance(idx, slice) and idx == slice(None):
            formatted.append(":")
        else:
            formatted.append(str(idx))
    return "[" + ", ".join(formatted) + "]"


def _write_to_markov_chain(
    mc: np.ndarray, sample: np.ndarray, idx: list[int | object], print_level: int = 0
) -> np.ndarray:
    """Write a sample to the Markov chain at specified position.

    Parameters
    ----------
    mc : np.ndarray
        The Markov chain.
    sample:
        The sample to write to the chain.
    idx : list
        Which position (multiindex) to write to: [fidelity, iteration in the chain, samples].
    print_level: int
        Optional input specifying the print level (default 0).
        0 = no print.
        1 = debug.

    Returns
    -------
    np.ndarray
        The updated Markov chain.

    Raises
    ------
    ValueError
        If existing arguments in the chain would be overwritten.
    """

    if np.any(np.isnan(mc[tuple(idx)])):
        mc[tuple(idx)] = sample
        if print_level == 1:
            print(
                f"Wrote values {mc[tuple(idx)]} into {_format_indices(idx)} of the Markov chain."
            )
    else:
        raise ValueError(
            f"Overwriting values {mc[tuple(idx)]} in the Markov chain at position {_format_indices(idx)}"
        )

    return mc


def _read_from_markov_chain(
    mc: np.ndarray, idx: list[int | object], print_level: int = 0
) -> np.ndarray:
    """Read a sample from the Markov chain at specified position.

    Parameters
    ----------
    mc : np.ndarray
        The Markov chain.
    idx : list
        Which position (multiindex) to read from: [fidelity, iteration in the chain, samples].
    print_level: int
        Optional input specifying the print level (default 0).
        0 = no print.
        1 = debug.

    Returns
    -------
    np.ndarray
        The sample read from the chain.

    Raises
    ------
    ValueError
        If existing arguments in the chain would be overwritten.
    """

    if np.any(np.isnan(mc[tuple(idx)])):
        raise ValueError(
            f"Setting proposal from uninitialized values (NaN) at position {_format_indices(idx)} in subchain."
        )
    else:
        if print_level == 1:
            print(
                f"Read values  {mc[tuple(idx)]} from {_format_indices(idx)} in the subchain."
            )

        return mc[tuple(idx)]
