"""
Test the select_optimal_models function.
"""

from __future__ import annotations

import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import parser, select_optimal_models

data_dir = Path(__file__).parent / "data"


def test_select_optimal_models_fail_correlation(tmp_path: Path) -> None:
    """Test select_optimal_models function with correlations that are > 1

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test
    """

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # copy the input file to the temporary path
        shutil.copy(data_dir / "test_model-select_fail_correlation.hdf5", tmp_path)
        test_file = tmp_path / "test_model-select_fail_correlation.hdf5"
        os.chdir(tmp_path)

        # run the select_optimal_models function
        with pytest.raises(ValueError):
            select_optimal_models(
                parser().parse_args(
                    [
                        "-a",
                        "model-select",
                        "-i",
                        f"{test_file}",
                        "--budget",
                        "1000",
                    ]
                )
            )
