"""
Test for the command line entry point function
"""

from __future__ import annotations

import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import console_entry_point


@pytest.mark.parametrize(
    ("algo", "args"),
    [
        ("build", "--models 3 --molecules 5 4 3 --evals 3 3 3"),
        ("chemmodel-prep", ""),
        ("chemmodel-post", ""),
        ("mfmc-prep", ""),
        ("model-select", ""),
        ("eval-estimator", "--budget 1e6"),
        ("mfmc", ""),
    ],
)
def test_entrypoint_parametrized(tmp_path: Path, algo: str, args: str) -> None:
    """Test command line entry point function

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test
    algo : str
        The algorithm to test, e.g. "build", "chemmodel-prep", etc.
    args : str
        Command line arguments for the algorithm
    """
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):

        # define input and output file paths
        data_dir = Path(__file__).parent / "data"
        output_file = tmp_path / f"test_{algo}_out.hdf5"
        input_file = data_dir / algo / f"test_{algo}.hdf5"

        # change to the temporary path
        os.chdir(tmp_path)

        if algo == "build":
            assert 0 == console_entry_point(
                f"-a {algo} {args} -o {output_file}".split()
            )
        else:
            shutil.copy(input_file, tmp_path / f"test_{algo}.hdf5")
            if (data_dir / algo / "models").exists():
                shutil.copytree(
                    data_dir / algo / "models",
                    tmp_path / "models",
                    dirs_exist_ok=False,
                )
            assert 0 == console_entry_point(
                f"-a {algo} -i {tmp_path / f'test_{algo}.hdf5'} -o {output_file} {args}".split()
            )
