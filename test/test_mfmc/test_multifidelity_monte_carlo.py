"""
Test the multifidelity Monte Carlo function.
"""

from __future__ import annotations

import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import multifidelity_monte_carlo, parser

data_dir = Path(__file__).parent / "data"


def test_mfmc_fail_correlation(tmp_path: Path) -> None:
    """Test multifidelity Monte Carlo algorithm with correlations that are > 1

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test
    """

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # copy the input file to the temporary path
        shutil.copy(data_dir / "test_mfmc_fail_correlation.hdf5", tmp_path)
        test_file = tmp_path / "test_mfmc_fail_correlation.hdf5"
        os.chdir(tmp_path)

        # run the multifidelity Monte Carlo algorithm
        with pytest.raises(ValueError):
            multifidelity_monte_carlo(
                parser().parse_args(["-a", "mfmc", "-i", f"{test_file}"])
            )
