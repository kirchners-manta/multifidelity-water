"""
Test the build function for default input file creation.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest

from mfwater import build_default_input, parser

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
