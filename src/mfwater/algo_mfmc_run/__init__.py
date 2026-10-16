"""
Algorithm: MFMC run
===================

Automated paper-I forward workflow (calibration, model selection, allocation,
evaluation, MFMC estimator) in restartable, multi-job-capable SLURM jobs.
"""

from .config import MFMCRunConfig, build_config
from .dummy import CorrelatedDummyForwardModel
from .jobscript import mfmc_jobscript
from .run import mfmc_run

__all__ = [
    "CorrelatedDummyForwardModel",
    "MFMCRunConfig",
    "build_config",
    "mfmc_jobscript",
    "mfmc_run",
]
