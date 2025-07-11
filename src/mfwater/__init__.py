"""
MFWater
======

This program is developed by Tom Frömbgen, Allan Kuhn, Jürgen Dölz and Barbara Kirchner (University of Bonn, Germany).
It is published under the MIT license.
"""

from .__version__ import __version__
from .algo_chemical_model import chemical_model_post, chemical_model_prep
from .algo_eval_estim import evaluate_estimator
from .algo_input import build_default_input
from .algo_mfmc import multifidelity_monte_carlo
from .algo_mfmc_preparation import multifidelity_preparation
from .algo_model_selection import get_ordered_index_combinations, select_optimal_models
from .argparser import (
    action_in_range,
    action_not_less_than,
    action_not_more_than,
    is_dir,
    is_file,
    parser,
)
from .cli import console_entry_point
