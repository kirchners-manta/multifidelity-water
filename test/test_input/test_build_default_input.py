"""
Test the build function for default input file creation.
"""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from mfwater import build_default_input, check_input_file, parser

p = parser()


def test_models_molecules_evals() -> None:
    """Test the input numbers for models, molecules, and evals."""

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(ValueError):
            build_default_input(
                p.parse_args(
                    [
                        "-a",
                        "build",
                        "--models",
                        "2",
                        "--molecules",
                        "3",
                    ]
                )
            )

        with pytest.raises(ValueError):
            build_default_input(
                p.parse_args(
                    [
                        "-a",
                        "build",
                        "--models",
                        "3",
                        "--molecules",
                        "3",
                        "2",
                        "4",
                    ]
                )
            )

        with pytest.raises(ValueError):
            build_default_input(
                p.parse_args(
                    [
                        "-a",
                        "build",
                        "--models",
                        "3",
                        "--molecules",
                        "3",
                        "2",
                        "1",
                        "--evals",
                        "10",
                    ]
                )
            )

        with pytest.raises(ValueError):
            build_default_input(
                p.parse_args(
                    [
                        "-a",
                        "build",
                        "--models",
                        "3",
                        "--molecules",
                        "3",
                        "2",
                        "1",
                        "--evals",
                        "10",
                        "10",
                        "5",
                    ]
                )
            )


def test_existing_file_no_overwrite(tmp_path: Path) -> None:
    """Test behavior when file exists and user declines overwrite."""
    file_path = tmp_path / "test_output.txt"
    file_path.write_text("existing content")  # Create the file

    args = p.parse_args(["-a", "build", "-o", str(file_path)])

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with patch("builtins.input", return_value="n"):
            with pytest.raises(RuntimeError):
                build_default_input(args)


def test_existing_file_overwrite(tmp_path: Path) -> None:
    """Test behavior when file exists and user accepts overwrite."""
    file_path = tmp_path / "test_output.txt"
    file_path.write_text("existing content")  # Create the file

    args = p.parse_args(["-a", "build", "-o", str(file_path)])

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with patch("builtins.input", return_value="y"):
            build_default_input(args)
            assert file_path.exists()  # Check if overwritten


def test_check_input_file_exists(tmp_path: Path) -> None:
    """Test that check_input_file raises an error if the file does not exist or if not input file was provided."""
    non_existent_file = tmp_path / "non_existent.hdf5"
    with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
        with pytest.raises(FileNotFoundError):
            check_input_file(non_existent_file, "mfmc")

        with pytest.raises(ValueError):
            check_input_file(None, "mfmc")


def test_check_no_models_directory(tmp_path: Path) -> None:
    """Test that check_input_file raises an error if no models directory is present."""
    with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
        with pytest.raises(FileNotFoundError):
            os.chdir(tmp_path)
            build_default_input(
                p.parse_args(["-a", "build", "-o", str(tmp_path / "input.hdf5")])
            )
            check_input_file(tmp_path / "input.hdf5", "chemmodel-post")
