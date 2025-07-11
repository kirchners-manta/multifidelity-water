"""
Test the multifidelity Monte Carlo preparation functionality.
"""

from __future__ import annotations

import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import multifidelity_preparation, parser

data_dir = Path(__file__).parent / "data"


def test_mfmc_prep_fail_evaluations(tmp_path: Path) -> None:
    """Test multifidelity Monte Carlo preparation with evaluations that are unequal across models.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test
    """

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # copy the input file to the temporary path
        shutil.copy(data_dir / "test_mfmc-prep_fail_nevals.hdf5", tmp_path)
        test_file = tmp_path / "test_mfmc-prep_fail_nevals.hdf5"
        os.chdir(tmp_path)

        # run the multifidelity Monte Carlo preparation
        with pytest.raises(ValueError):
            multifidelity_preparation(
                parser().parse_args(["-a", "mfmc-prep", "-i", f"{test_file}"])
            )
