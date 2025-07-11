"""
Test helper functions for command line arguments parsing
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import (
    action_in_range,
    action_not_less_than,
    action_not_more_than,
    is_dir,
    is_file,
)


def test_file_or_dir(tmp_path: Path) -> None:
    """Test file or directory argument parsing

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the test
    """

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):

        # Test file
        # create a temporary file
        (tmp_path / "test.txt").write_text("This is a test file.")
        assert is_file(tmp_path / "test.txt") == tmp_path / "test.txt"

        # Test directory
        assert is_dir(tmp_path) == tmp_path

        # Test invalid file
        with pytest.raises(argparse.ArgumentTypeError):
            is_file(tmp_path / "non_existent.txt")

        # Test invalid directory
        with pytest.raises(argparse.ArgumentTypeError):
            is_dir(tmp_path / "non_existent_dir")

        # test a directory given to is_file
        with pytest.raises(argparse.ArgumentTypeError):
            is_file(tmp_path)

        # test a file given to is_dir
        with pytest.raises(argparse.ArgumentTypeError):
            is_dir(tmp_path / "test.txt")


def test_custom_argparse_action_in_range() -> None:
    """Test custom argparse action for values in a range."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--value",
        action=action_in_range(1, 10),
        type=float,
        nargs="+",
        help="R|Value must be in the range [1, 10].",
    )

    # Test single valid value
    args = parser.parse_args(["--value", "5"])
    assert args.value == 5

    # Test list of valid values
    args = parser.parse_args(["--value", "5", "6"])
    assert args.value == [5, 6]

    # Test invalid value
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(SystemExit):
            parser.parse_args(["--value", "0"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--value", "11"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--value", "2", "5", "12"])


def test_custom_argparse_action_not_less_than() -> None:
    """Test custom argparse action for values not less than a given value."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--value",
        action=action_not_less_than(5),
        type=float,
        nargs="+",
        help="R|Value must be not less than 5.",
    )

    # Test valid value
    args = parser.parse_args(["--value", "5"])
    assert args.value == 5

    # Test list of valid values
    args = parser.parse_args(["--value", "5", "6"])
    assert args.value == [5, 6]

    # Test invalid value
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(SystemExit):
            parser.parse_args(["--value", "4"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--value", "3", "6"])


def test_custom_argparse_action_not_more_than() -> None:
    """Test custom argparse action for values not more than a given value."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--value",
        action=action_not_more_than(10),
        type=float,
        nargs="+",
        help="R|Value must be not more than 10.",
    )

    # Test valid value
    args = parser.parse_args(["--value", "10"])
    assert args.value == 10

    # Test list of valid values
    args = parser.parse_args(["--value", "5", "6"])
    assert args.value == [5, 6]

    # Test invalid value
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(SystemExit):
            parser.parse_args(["--value", "11"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--value", "3", "12"])
