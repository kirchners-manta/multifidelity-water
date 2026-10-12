"""
Test the TOML configuration file option
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from mfwater import parser
from mfwater.argparser import config_to_argv, parse_args

p = parser()


def write_config(tmp_path: Path, text: str) -> str:
    """Write a configuration file and return its path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.
    text : str
        TOML content.

    Returns
    -------
    str
        Path of the configuration file.
    """
    f = tmp_path / "config.toml"
    f.write_text(text)
    return str(f)


def test_no_config() -> None:
    """Without a configuration file the parser behaves as before"""
    args = parse_args(p, ["--models", "3"])
    assert args.n_models == 3
    assert args.config is None


def test_config_values(tmp_path: Path) -> None:
    """Long option names, dest names, lists and flags are read from the file"""
    cfg = write_config(
        tmp_path,
        'algorithm = "markov-chain"\n'
        "models = 3\n"
        "molecules = [1000, 500, 100]\n"
        "n_chains = 4\n"
        "orthoboxy = true\n"
        'forward-model = "md"\n'
        "budget = 1e6\n",
    )
    args = parse_args(p, ["-c", cfg])
    assert args.algorithm == "markov-chain"
    assert args.n_models == 3
    assert args.n_molecules == [1000, 500, 100]
    assert args.n_chains == 4
    assert args.orthoboxy is True
    assert args.forward_model == "md"
    assert args.budget == 1e6
    # untouched options keep their defaults
    assert args.n_mc_burnin == 100


def test_config_overrides_command_line(tmp_path: Path) -> None:
    """Options of the file win over the command line"""
    cfg = write_config(tmp_path, "models = 4\nmolecules = [8, 4]\n")
    args = parse_args(p, ["--models", "2", "--molecules", "5", "4", "3", "-c", cfg])
    assert args.n_models == 4
    assert args.n_molecules == [8, 4]


def test_config_false_flag_is_ignored(tmp_path: Path) -> None:
    """A false flag does not switch off a flag given on the command line"""
    cfg = write_config(tmp_path, "orthoboxy = false\n")
    assert parse_args(p, ["-z", "-c", cfg]).orthoboxy is True
    assert parse_args(p, ["-c", cfg]).orthoboxy is False


@pytest.mark.parametrize(
    "text",
    [
        "no_such_option = 1\n",  # unknown key
        "config = 'other.toml'\n",  # the file cannot name another file
        "models = 'abc'\n",  # wrong type
        "models = 0\n",  # violates the value check of the option
        "forward-model = 'foo'\n",  # not in choices
        "orthoboxy = 1\n",  # flags must be boolean
        "models = true\n",  # booleans are not integers
        "models = [\n",  # invalid TOML
    ],
)
def test_config_errors(tmp_path: Path, text: str) -> None:
    """Invalid configuration files are reported like invalid command lines"""
    cfg = write_config(tmp_path, text)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(SystemExit):
            parse_args(p, ["-c", cfg])


def test_missing_config_file() -> None:
    """A missing configuration file is rejected"""
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        with pytest.raises(SystemExit):
            parse_args(p, ["-c", "does_not_exist.toml"])


def test_config_to_argv() -> None:
    """Keys map to the longest option string"""
    assert config_to_argv(p, {"n_molecules": [3, 2], "orthoboxy": True}) == [
        "--molecules",
        "3",
        "2",
        "--orthoboxy",
    ]
