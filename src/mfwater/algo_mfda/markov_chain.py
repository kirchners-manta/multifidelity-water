"""Set up and run a Markov chain for a Multifidelity Delayed Acceptance MCMC."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..argparser import constants
from .multifidelity_mcmc import (
    ChainState,
    ForwardCallable,
    MFDAConfig,
    mfda_estimator,
    mfda_step,
    proposal_kernel,
)

#: Parameters ``(eps_OO, sigma_OO, q_O)`` that are calibrated for each ``--params`` choice.
_FREE_MASKS: dict[str, tuple[bool, bool, bool]] = {
    "lj": (True, True, False),
    "q": (False, False, True),
    "lj-q": (True, True, True),
}


@dataclass
class ChainResult:
    """Result of one MFDA chain (post-burn-in).

    Parameters
    ----------
    samples : list[np.ndarray]
        Per level ``i`` (index ``i - 1``) the samples ``X_i``, shape ``(K_i, 3)``.
    proposals : list[np.ndarray]
        Per level the proposals ``X'_i``, shape ``(K_i, 3)``; empty for the coarsest level.
    estimator : np.ndarray
        Multilevel estimator ``(eps, sigma, q)``.
    contributions : list[np.ndarray]
        Contribution of each level to the estimator (level 1 first).
    seed : int
        Entropy used to seed the random generator.
    n_burnin : int
        Number of discarded fine-level steps.
    """

    samples: list[np.ndarray]
    proposals: list[np.ndarray]
    estimator: np.ndarray
    contributions: list[np.ndarray]
    seed: int | None
    n_burnin: int


def run_chain(
    config: MFDAConfig,
    forward: ForwardCallable,
    rng: np.random.Generator,
    chain_length: int,
    n_burnin: int,
    seed: int | None = None,
) -> ChainResult:
    """Run one MFDA chain on the finest level.

    All entries (all levels, samples and proposals) produced during the first
    ``n_burnin`` fine-level steps, including their subchains, are discarded.

    Parameters
    ----------
    config : MFDAConfig
        Algorithm settings.
    forward : ForwardCallable
        Forward model ``forward(n_molecules, theta) -> D`` (cached by the caller).
    rng : np.random.Generator
        The single random number generator of the algorithm.
    chain_length : int
        Fixed number of fine-level steps ``M_1``.
    n_burnin : int
        Number of burn-in steps ``B`` on the fine level.
    seed : int, optional
        Seed entropy, only stored in the result.

    Returns
    -------
    ChainResult
        Post-burn-in samples, proposals and estimator.

    Raises
    ------
    ValueError
        If ``chain_length < 1`` or ``not 0 <= n_burnin < chain_length``.
    """
    if chain_length < 1:
        raise ValueError(f"Chain length must be >= 1, got {chain_length}.")
    if not 0 <= n_burnin < chain_length:
        raise ValueError(
            f"Burn-in must satisfy 0 <= B < chain length ({chain_length}), got {n_burnin}."
        )

    state = ChainState.empty(config.n_levels)

    # initial sample theta_0 ~ N(mu, Sigma) on the free components
    current = proposal_kernel(config.means, config.kernel_noise, rng, config.free_mask)

    # B = 0 keeps everything, i.e. cut at zero entries
    cut: tuple[list[int], list[int]] = ([0] * config.n_levels, [0] * config.n_levels)
    for step in range(1, chain_length + 1):
        current = mfda_step(current, 1, config, state, forward, rng)
        if step == n_burnin:
            # list lengths after step B mark the end of the burn-in on every level
            cut = state.lengths()

    state.truncate_front(*cut)

    samples = [np.asarray(s, dtype=float).reshape(-1, 3) for s in state.samples]
    proposals = [np.asarray(p, dtype=float).reshape(-1, 3) for p in state.proposals]
    estimator, contributions = mfda_estimator(samples, proposals)

    return ChainResult(samples, proposals, estimator, contributions, seed, n_burnin)


def as_int_list(value: int | Sequence[int] | None) -> list[int]:
    """Normalise a command line value to a list of ints.

    The ``action_not_less_than`` argparse action stores a single value as a
    scalar and several values as a list.

    Parameters
    ----------
    value : int, Sequence[int] or None
        Value as stored in the argument namespace.

    Returns
    -------
    list[int]
        The values as a list (empty for ``None``).
    """
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


def resolve_subchain_lengths(
    lengths: int | Sequence[int] | None, n_models: int
) -> list[int]:
    """Expand a single ``--mcsubchainlength`` value to all coarse levels.

    Parameters
    ----------
    lengths : Sequence[int] or None
        Subchain lengths as given on the command line.
    n_models : int
        Number of levels ``eta``.

    Returns
    -------
    list[int]
        The lengths unchanged, or one value repeated ``n_models - 1`` times if
        exactly one value was given. ``None`` is returned as an empty list so
        that the caller reports the wrong length.
    """
    values = as_int_list(lengths)
    if len(values) == 1:
        return values * max(n_models - 1, 0)
    return values


def build_config(args: argparse.Namespace) -> MFDAConfig:
    """Validate the command line arguments and build the algorithm settings.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    MFDAConfig
        Algorithm settings.

    Raises
    ------
    ValueError
        If the number of models, molecule numbers, subchain lengths or chain
        lengths are invalid, or no parameter is calibrated.
    """
    if args.n_models < 2:
        raise ValueError(f"At least 2 models are required, got {args.n_models}.")
    molecules = as_int_list(args.n_molecules)
    if len(molecules) != args.n_models:
        raise ValueError(
            f"--molecules must list {args.n_models} values (one per model), "
            f"got {molecules}."
        )
    if any(a <= b for a, b in zip(molecules[:-1], molecules[1:])):
        raise ValueError(
            f"--molecules must be strictly descending (level 1 = finest), got {molecules}."
        )
    sub = resolve_subchain_lengths(args.n_mc_subchain_lengths, args.n_models)
    if len(sub) != args.n_models - 1:
        raise ValueError(
            f"--mcsubchainlength must list {args.n_models - 1} values "
            f"(levels 2..{args.n_models}), got {sub}."
        )
    if any(m < 1 for m in sub):
        raise ValueError(f"Subchain lengths must be >= 1, got {sub}.")
    if args.n_mc_chain_length < 1:
        raise ValueError(f"--mcchainlength must be >= 1, got {args.n_mc_chain_length}.")
    if not 0 <= args.n_mc_burnin < args.n_mc_chain_length:
        raise ValueError(
            f"--mcburnin must satisfy 0 <= B < mcchainlength ({args.n_mc_chain_length}), "
            f"got {args.n_mc_burnin}."
        )

    return make_config(
        molecules, sub, args.params, print_level=getattr(args, "print_level", 0)
    )


def make_config(
    molecules: Sequence[int],
    subchain_lengths: Sequence[int],
    params: str,
    print_level: int = 0,
) -> MFDAConfig:
    """Build the :class:`MFDAConfig` with the OPC3 prior ``N(mu, diag(s**2))``.

    Parameters
    ----------
    molecules : Sequence[int]
        Molecule numbers per level (level 1 = finest).
    subchain_lengths : Sequence[int]
        Subchain lengths for levels ``2..eta``.
    params : str
        Which parameters are calibrated: ``"lj"``, ``"q"`` or ``"lj-q"``.
    print_level : int
        ``0`` = silent, ``1`` = debug.

    Returns
    -------
    MFDAConfig
        Algorithm settings.

    Raises
    ------
    ValueError
        If ``params`` is unknown or settings are inconsistent.
    """
    if params not in _FREE_MASKS:
        raise ValueError(f"Unknown params '{params}', choose from {list(_FREE_MASKS)}.")

    # prior mean = OPC3 values; the same Sigma is used for prior and proposal kernel
    means = np.array(
        [constants.OPC3_EPSILON_OO, constants.OPC3_SIGMA_OO, constants.OPC3_CHARGE_O]
    )
    noise = np.array(
        [
            constants.OPC3_EPSILON_OO * constants.NOISE_LJPARAMS,
            constants.OPC3_SIGMA_OO * constants.NOISE_LJPARAMS,
            abs(constants.OPC3_CHARGE_O) * constants.NOISE_CHARGES,
        ]
    )
    return MFDAConfig(
        molecules=tuple(int(n) for n in molecules),
        subchain_lengths=tuple(int(m) for m in subchain_lengths),
        means=means,
        kernel_noise=noise,
        free_mask=np.array(_FREE_MASKS[params], dtype=bool),
        print_level=print_level,
    )


def write_results(
    path: str | Path,
    result: ChainResult,
    config: MFDAConfig,
    args: dict[str, Any],
    chain_length: int,
) -> None:
    """Write a chain result to a new HDF5 file.

    Parameters
    ----------
    path : str or Path
        Output file; must not exist.
    result : ChainResult
        The chain result.
    config : MFDAConfig
        Algorithm settings (stored as attributes).
    args : dict
        Command line arguments (stored as JSON).
    chain_length : int
        Fine-level chain length ``M_1``.

    Raises
    ------
    FileExistsError
        If ``path`` already exists (existing output is never overwritten).
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output file {path} exists, refusing to overwrite.")

    # mode "x" fails if the file exists, as a second guard against overwriting
    with h5py.File(path, "x") as f:
        for i, s in enumerate(result.samples, start=1):
            f.create_dataset(f"samples_level{i}", data=s)
        for i, p in enumerate(result.proposals, start=1):
            if i < config.n_levels:
                f.create_dataset(f"proposals_level{i}", data=p)
        f.create_dataset("estimator", data=result.estimator)
        f.create_dataset("contributions", data=np.array(result.contributions))
        # the entropy of a SeedSequence exceeds 64 bit, so it is stored as text
        f.attrs["seed"] = str(result.seed)
        f.attrs["args"] = json.dumps(args, default=str)
        f.attrs["n_burnin"] = result.n_burnin
        f.attrs["chain_length"] = chain_length
        f.attrs["molecules"] = np.array(config.molecules)
        f.attrs["subchain_lengths"] = np.array(config.subchain_lengths)
        f.attrs["free_mask"] = config.free_mask
