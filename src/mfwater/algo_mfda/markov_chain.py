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
from .forward_model import forward_model_dummy
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


class MemoizedForward:
    """In-memory memo around a forward model.

    D of a given ``(N, theta)`` is computed once and reused, because the forward
    model is stochastic. This is a local stand-in until a persistent cache wraps
    the forward model. The memo never draws from the algorithm RNG.

    Parameters
    ----------
    forward : ForwardCallable
        Forward model ``forward(n_molecules, theta) -> D``.
    """

    def __init__(self, forward: ForwardCallable) -> None:
        self._forward = forward
        self._memo: dict[tuple[Any, ...], float] = {}
        self.n_computed = 0
        self.n_cached = 0

    def __call__(self, n_molecules: int, theta: np.ndarray) -> float:
        """Return D for ``(n_molecules, theta)``, computing it only once.

        Parameters
        ----------
        n_molecules : int
            Number of molecules.
        theta : np.ndarray
            Parameter sample.

        Returns
        -------
        float
            Diffusion coefficient in 1e-9 m^2/s.
        """
        # float.hex() is an exact key: equal floats <=> equal keys
        key = (int(n_molecules), *(float(t).hex() for t in theta))
        if key in self._memo:
            self.n_cached += 1
        else:
            self._memo[key] = float(self._forward(n_molecules, theta))
            self.n_computed += 1
        return self._memo[key]


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
    molecules = args.n_molecules
    if molecules is None or len(molecules) != args.n_models:
        raise ValueError(
            f"--molecules must list {args.n_models} values (one per model), "
            f"got {molecules}."
        )
    if any(a <= b for a, b in zip(molecules[:-1], molecules[1:])):
        raise ValueError(
            f"--molecules must be strictly descending (level 1 = finest), got {molecules}."
        )
    sub = args.n_mc_subchain_lengths
    if sub is None or len(sub) != args.n_models - 1:
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


def markov_chain_eval(args: argparse.Namespace) -> int:
    """Run the MFDA-MCMC algorithm for MFWater.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    int
        Exit code, ``0`` for success.

    Raises
    ------
    ValueError
        If the arguments are invalid (see :func:`build_config`).
    FileExistsError
        If the output file already exists.
    """
    config = build_config(args)

    # fail before any compute is spent if the output cannot be written
    if Path(args.output).exists():
        raise FileExistsError(
            f"Output file {args.output} exists, refusing to overwrite."
        )

    # one generator for the whole algorithm; print the seed so the run is reproducible
    seed = args.seed if args.seed is not None else np.random.SeedSequence().entropy
    print(f"Random seed: {seed}")
    rng = np.random.default_rng(seed)

    # NOTE: forward_model_dummy draws from the global np.random, so runs are not
    # reproducible with the dummy model until the deterministic forward model is wired in.
    forward = MemoizedForward(forward_model_dummy)

    result = run_chain(
        config, forward, rng, args.n_mc_chain_length, args.n_mc_burnin, seed
    )

    # level 1 is the finest, the last level the coarsest
    for i, contrib in enumerate(result.contributions, start=1):
        kind = "coarsest, mean(X)" if i == config.n_levels else "mean(X) - mean(X')"
        print(
            f"Level {i} ({kind}), contribution to eps, sig, q: "
            f"{contrib[0]:10.6f} {contrib[1]:10.6f} {contrib[2]:10.6f}"
        )
    est = result.estimator
    print(
        f"\nMFDA-MCMC estimator:                  "
        f"{est[0]:10.6f} {est[1]:10.6f} {est[2]:10.6f}"
    )
    print(
        f"Forward evaluations: {forward.n_computed} computed, {forward.n_cached} from cache."
    )

    write_results(args.output, result, config, vars(args), args.n_mc_chain_length)
    return 0
