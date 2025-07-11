"""
Test command line options / arguments
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import parser

p = parser()


def test_defaults() -> None:
    """Test default command line arguments"""

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # parse the default arguments
        args = p.parse_args([])

        assert args.algorithm == None
        assert args.input == None
        assert args.output == "default.hdf5"
        assert args.orthoboxy == False
        assert args.n_models == 6
        assert args.n_molecules == None
        assert args.n_evals == None
        assert args.budget == 0


def test_fail_type() -> None:
    """Test command line arguments with wrong type"""

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(SystemExit):
            p.parse_args(["-a", "7"])
        with pytest.raises(SystemExit):
            p.parse_args(["--orthoboxy", "a"])
        with pytest.raises(SystemExit):
            p.parse_args(["--models", "c"])
        with pytest.raises(SystemExit):
            p.parse_args(["--molecules", "adsfsdf"])
        with pytest.raises(SystemExit):
            p.parse_args(["--evals", "hallo"])
        with pytest.raises(SystemExit):
            p.parse_args(["--budget", "sada"])


def test_fail_value(tmp_path: Path) -> None:
    """Test command line arguments with wrong value


    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test

    """

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(SystemExit):
            p.parse_args(["-a", "Build"])
        with pytest.raises(SystemExit):
            p.parse_args(["-i", str(tmp_path / "nofile.hdf5")])
        with pytest.raises(SystemExit):
            p.parse_args(["--models", "1.23"])
        with pytest.raises(SystemExit):
            p.parse_args(["--models", "-5"])
        with pytest.raises(SystemExit):
            p.parse_args(["--molecules", "3.45567"])
        with pytest.raises(SystemExit):
            p.parse_args(["--molecules", "-1"])
        with pytest.raises(SystemExit):
            p.parse_args(["--evals", "9.9812"])
        with pytest.raises(SystemExit):
            p.parse_args(["--evals", "0"])
        with pytest.raises(SystemExit):
            p.parse_args(["--budget", "-2.5"])
