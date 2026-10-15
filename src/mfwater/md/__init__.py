"""Shared MD layer: forward model, evaluation cache and MD pipeline.

Used by the MFDA algorithm (``algo_mfda``) and by the MFMC forward pipeline.
The modules ``mfwater.algo_mfda.{forward_model,cache,md_pipeline}`` re-export
the names defined here, so that older import paths keep working.
"""

from .cache import CachedForwardModel
from .forward_model import (
    DummyForwardModel,
    ForwardModel,
    derive_seeds,
    evaluation_key,
    key_hash,
)
from .lammps_log import LammpsCost, parse_lammps_log
from .pipeline import MDForwardModel
from .protocol import (
    MFDA_PROTOCOL,
    PAPER1_PROTOCOL,
    PROTOCOLS,
    MDProtocol,
    get_protocol,
)
from .system import calc_box_size, calc_cpus

__all__ = [
    "CachedForwardModel",
    "DummyForwardModel",
    "ForwardModel",
    "LammpsCost",
    "MDProtocol",
    "MFDA_PROTOCOL",
    "PAPER1_PROTOCOL",
    "PROTOCOLS",
    "MDForwardModel",
    "calc_box_size",
    "calc_cpus",
    "derive_seeds",
    "evaluation_key",
    "get_protocol",
    "key_hash",
    "parse_lammps_log",
]
