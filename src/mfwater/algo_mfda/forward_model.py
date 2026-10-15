"""Backward-compatible import path of :mod:`mfwater.md.forward_model`.

The implementation moved to the shared MD layer; everything is re-exported here.
"""

from ..md.forward_model import (  # noqa: F401
    MAX_SEED,
    DummyForwardModel,
    ForwardModel,
    _check_theta,
    _full_digest,
    derive_seeds,
    evaluation_key,
    forward_model_dummy,
    key_hash,
)
