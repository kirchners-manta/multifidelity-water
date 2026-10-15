"""Backward-compatible import path of :mod:`mfwater.md.cache`.

The implementation moved to the shared MD layer; everything is re-exported here.
``os`` is imported because tests patch ``mfwater.algo_mfda.cache.os.replace``
(the same module object that :mod:`mfwater.md.cache` uses).
"""

import os  # noqa: F401

from ..md.cache import D_UNIT, CachedForwardModel  # noqa: F401
