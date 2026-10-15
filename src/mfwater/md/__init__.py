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
from .pipeline import MDForwardModel
from .system import calc_box_size, calc_cpus

__all__ = [
    "CachedForwardModel",
    "DummyForwardModel",
    "ForwardModel",
    "MDForwardModel",
    "calc_box_size",
    "calc_cpus",
    "derive_seeds",
    "evaluation_key",
    "key_hash",
]
