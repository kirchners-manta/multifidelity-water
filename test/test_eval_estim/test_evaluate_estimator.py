"""
Test the evaluate_estimator function.
"""

from __future__ import annotations

import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import evaluate_estimator, parser

data_dir = Path(__file__).parent / "data"


def test_eval_estimator_fail_correlation(tmp_path: Path) -> None:
    """Test evaluate_estimator function with correlations that are > 1

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test
    """

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # copy the input file to the temporary path
        shutil.copy(data_dir / "test_eval-estimator_fail_correlation.hdf5", tmp_path)
        test_file = tmp_path / "test_eval-estimator_fail_correlation.hdf5"
        os.chdir(tmp_path)

        # run the evaluate_estimator function
        with pytest.raises(ValueError):
            evaluate_estimator(
                parser().parse_args(
                    [
                        "-a",
                        "eval-estimator",
                        "-i",
                        f"{test_file}",
                        "--budget",
                        "1000",
                    ]
                )
            )


def test_eval_estimator_fail_budget(tmp_path: Path) -> None:
    """Test evaluate_estimator function with a too small budget.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test
    """

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # copy the input file to the temporary path
        shutil.copy(data_dir / "test_eval-estimator_fail_budget.hdf5", tmp_path)
        test_file = tmp_path / "test_eval-estimator_fail_budget.hdf5"
        os.chdir(tmp_path)

        # run the evaluate_estimator function
        with pytest.raises(ValueError):
            evaluate_estimator(
                parser().parse_args(
                    [
                        "-a",
                        "eval-estimator",
                        "-i",
                        f"{test_file}",
                        "--budget",
                        "5",
                    ]
                )
            )
