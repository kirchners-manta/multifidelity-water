"""Tests for the MD protocols (``mfda`` and ``paper1``)."""

from __future__ import annotations

import dataclasses
import difflib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfwater.md.pipeline import (
    TEMPLATE_DIR,
    MDForwardModel,
    patch_travis_input,
    write_lammps_input,
    write_resume_input,
)
from mfwater.md.protocol import (
    ALL_FRAMES,
    MFDA_PROTOCOL,
    PAPER1_PROTOCOL,
    PROTOCOLS,
    MDProtocol,
    get_protocol,
)

from ..test_mfda.test_forward_model import make_fake_run

THETA = np.array([0.1634, 3.17427, -0.8952])
ORTHO_FILES = ("msd_H2O_#2.csv", "msd_H2O_#2_XY.csv", "msd_H2O_#2_Z.csv")

# Paper-I Marvin input (values given by the user): VAR_EPS, VAR_SIG, q_H, q_O, seed
MARVIN = {
    "eps": 0.162877,
    "sig": 3.170409,
    "q_h": 0.44760001,
    "q_o": -0.89520001,
    "seed": 90203,
}


def marvin_input() -> list[str]:
    """Reconstruct the input.lmp of a paper-I evaluation.

    Replays the substitutions of ``chemmod-prep`` (``setup`` in
    ``algo_chemical_model/chemical_model.py``) on the paper-I template. The
    charges are taken from the stored ``charges`` dataset, not recomputed.
    """
    lines = (TEMPLATE_DIR / "input.lmp").read_text().splitlines(keepends=True)
    for k, line in enumerate(lines):
        if "VAR_EPS" in line:
            lines[k] = (
                f"pair_coeff    2    2     {MARVIN['eps']:.6f}     "
                f"{MARVIN['sig']:.6f}  # Ow-Ow\n"
            )
        if "VAR_q_H" in line:
            lines[k] = f"set type 1 charge {MARVIN['q_h']:11.8f}  # Hw\n"
        if "VAR_q_O" in line:
            lines[k] = f"set type 2 charge {MARVIN['q_o']:11.8f}  # Ow\n"
        if "velocity all create" in line:
            lines[k] = f"velocity all create ${{vTK}} {MARVIN['seed']}\n"
            break
    return lines


# ---------------------------------------------------------------- the registry
def test_registry_and_lookup() -> None:
    assert set(PROTOCOLS) == {"mfda", "paper1"}
    assert get_protocol("mfda") is MFDA_PROTOCOL
    assert get_protocol(PAPER1_PROTOCOL) is PAPER1_PROTOCOL
    with pytest.raises(ValueError, match="Unknown MD protocol"):
        get_protocol("nope")
    # the name is part of the cache key: it must not alias different settings
    clash = dataclasses.replace(MFDA_PROTOCOL, eq_end_step=1)
    with pytest.raises(ValueError, match="different settings"):
        get_protocol(clash)
    custom = dataclasses.replace(MFDA_PROTOCOL, name="custom")
    assert get_protocol(custom) is custom


def test_protocol_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        MFDA_PROTOCOL.eq_end_step = 1  # type: ignore[misc]


def test_mfda_protocol_values() -> None:
    p = MFDA_PROTOCOL
    assert (p.eq_end_step, p.prod_end_step) == (250_000, 1_250_000)
    assert (p.dump_every, p.restart_every) == (1000, 100_000)
    assert (p.minimize_max_iter, p.minimize_max_eval) == (10_000, 100_000)
    assert (p.travis_frames, p.travis_depth) == (1000, 300)


def test_paper1_protocol_values() -> None:
    p = PAPER1_PROTOCOL
    # 1 ns equilibration, then 2 ns more: absolute step 3 000 000
    assert (p.eq_end_step, p.prod_end_step) == (1_000_000, 3_000_000)
    assert (p.minimize_max_iter, p.minimize_max_eval) == (100, 1000)
    assert p.travis_depth == 600 and p.travis_frames == ALL_FRAMES


@pytest.mark.parametrize("protocol", list(PROTOCOLS.values()), ids=lambda p: p.name)
def test_templates_agree_with_protocols(protocol: MDProtocol) -> None:
    protocol.validate_template()


def test_validate_template_detects_contradictions(tmp_path: Path) -> None:
    text = MFDA_PROTOCOL.lammps_template.read_text()
    for old, new in [
        ("run 250000 upto", "run 250000"),
        ("run 1250000 upto", "run 1300000 upto"),
        ("restart 100000", "restart 50000"),
        ("minimize 1.0e-4 1.0e-6 10000", "minimize 1.0e-4 1.0e-6 100"),
        ("dump dPROD all custom 1000", "dump dPROD all custom 500"),
    ]:
        assert old in text
        bad = tmp_path / "bad.lmp"
        bad.write_text(text.replace(old, new))
        proto = dataclasses.replace(MFDA_PROTOCOL, name="bad", lammps_template=bad)
        with pytest.raises(RuntimeError, match="contradicts protocol"):
            proto.validate_template()
        with pytest.raises(RuntimeError, match="contradicts protocol"):
            write_lammps_input(bad, tmp_path / "o.lmp", THETA, 1, protocol=proto)


# -------------------------------------------------------- paper1 vs paper-I run
def test_paper1_template_differs_from_paper_input_only_in_restart_and_upto() -> None:
    old = (TEMPLATE_DIR / "input.lmp").read_text().splitlines()
    new = PAPER1_PROTOCOL.lammps_template.read_text().splitlines()
    ops = [
        op
        for op in difflib.SequenceMatcher(None, old, new).get_opcodes()
        if op[0] != "equal"
    ]
    changes = [(tag, old[i1:i2], new[j1:j2]) for tag, i1, i2, j1, j2 in ops]
    assert [c[0] for c in changes] == ["insert", "replace", "replace"]
    inserted = [ln for ln in changes[0][2] if ln and not ln.startswith("#")]
    assert inserted == ["restart 100000 restart.*"]
    assert changes[1][1:] == (["run 1000000"], ["run 1000000 upto"])
    assert changes[2][1:] == (["run 2000000"], ["run 3000000 upto"])


def test_paper1_input_matches_paper_i_marvin_input(tmp_path: Path) -> None:
    """Generated paper1 input == Marvin input except restart, run upto, read_data."""
    out = tmp_path / "input.lmp"
    theta = np.array([MARVIN["eps"], MARVIN["sig"], MARVIN["q_o"]])
    write_lammps_input(
        PAPER1_PROTOCOL.lammps_template,
        out,
        theta,
        MARVIN["seed"],  # type: ignore[arg-type]
        protocol=PAPER1_PROTOCOL,
    )
    old = marvin_input()
    new = out.read_text().splitlines(keepends=True)
    changes = [
        (tag, "".join(old[i1:i2]), "".join(new[j1:j2]))
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, old, new, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]
    commands = [
        (tag, a.strip(), "".join(ln for ln in b.splitlines(True) if "#" not in ln[:1]))
        for tag, a, b in changes
    ]
    assert len(commands) == 4
    # (1) read_data is redirected because LAMMPS runs in simout/
    assert commands[0] == (
        "replace",
        "read_data data.lmp",
        "read_data ../siminp/data.lmp\n",
    )
    # (2) restart files every 100 ps (3 comment lines + 1 command)
    assert commands[1][0] == "insert"
    assert commands[1][2].split() == ["restart", "100000", "restart.*"]
    # (3), (4) absolute timesteps; the dynamics are those of `run 1e6`, `run 2e6`
    assert commands[2] == ("replace", "run 1000000", "run 1000000 upto\n")
    assert commands[3] == ("replace", "run 2000000", "run 3000000 upto\n")
    # the substituted physical parameters are identical to the Marvin input,
    # including the hydrogen charge (-q_O / 2 rounds to the stored value)
    for key in ("pair_coeff    2    2", "set type 1 charge", "set type 2 charge"):
        a = [ln for ln in old if ln.startswith(key)]
        b = [ln for ln in new if ln.startswith(key)]
        assert a == b and len(a) == 1, key
    assert "velocity all create ${vTK} 90203\n" in new


def _commands(text: str) -> list[str]:
    """Return the non-comment lines of a TRAVIS input (comments start with '!')."""
    # Whitespace at line ends and trailing blank lines are normalised: the
    # pre-commit hooks edit the stored copies of the paper-I inputs.
    lines = [ln.rstrip() for ln in text.split("\n") if not ln.startswith("!")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


@pytest.mark.parametrize(
    "template, reference",
    [
        ("travis_input_msd.txt", "input_normal.txt"),
        ("travis_input_msd_orthoboxy.txt", "input_orthoboxy.txt"),
    ],
)
@pytest.mark.parametrize("protocol", [PAPER1_PROTOCOL])
def test_paper1_travis_inputs_equal_paper_i_post_processing(
    template: str, reference: str, protocol: MDProtocol
) -> None:
    """The non-comment lines equal the inputs used for paper I (all frames, depth 600).

    The reference files are the paper-I post-processing inputs
    (``scripts/input_normal.txt`` and ``input_orthoboxy.txt`` of the paper-I data).
    The repository template differs for cubic boxes: it asks for 2000 frames.
    """
    text = (TEMPLATE_DIR / template).read_text()
    patched = patch_travis_input(text, protocol.travis_frames, protocol.travis_depth)
    ref = (Path(__file__).parents[1] / "data" / "travis" / reference).read_text()
    assert _commands(patched) == _commands(ref)


def test_all_frames_sentinel_is_not_none() -> None:
    text = (TEMPLATE_DIR / "travis_input_msd.txt").read_text()
    assert patch_travis_input(text, None, None) == text  # keeps the template's 2000
    assert "2000" in _commands(text)
    assert "2000" not in _commands(patch_travis_input(text, ALL_FRAMES, None))
    with pytest.raises(ValueError, match="travis_frames"):
        dataclasses.replace(PAPER1_PROTOCOL, travis_frames="some")
    with pytest.raises(ValueError, match="travis_frames"):
        dataclasses.replace(PAPER1_PROTOCOL, travis_frames=0)


def test_patch_travis_input_none_keeps_template() -> None:
    text = (TEMPLATE_DIR / "travis_input_msd.txt").read_text()
    assert patch_travis_input(text, None, None) == text
    # a missing prompt is only an error if a value is to be set
    stripped = text.replace("How many trajectory", "X")
    assert patch_travis_input(stripped, None, 600) == text.replace(
        "How many trajectory", "X"
    )
    with pytest.raises(ValueError, match="frames to read"):
        patch_travis_input(stripped, 10, None)


def test_paper1_resume_input_uses_protocol_steps(tmp_path: Path) -> None:
    out = tmp_path / "r.lmp"
    write_resume_input(
        out, tmp_path / "restart.500000", 500_000, protocol=PAPER1_PROTOCOL
    )
    text = out.read_text()
    assert "run 1000000 upto" in text and "run 3000000 upto" in text
    assert "run 250000" not in text
    assert "Continues the run of input_paper1.lmp" in text
    out2 = tmp_path / "r2.lmp"
    write_resume_input(out2, tmp_path / "restart.1500000", 1_500_000, PAPER1_PROTOCOL)
    text2 = out2.read_text()
    assert "run 1000000 upto" not in text2 and "run 3000000 upto" in text2
    assert "dEQUI" not in text2
    with pytest.raises(ValueError):
        write_resume_input(
            out2, tmp_path / "restart.3000000", 3_000_000, PAPER1_PROTOCOL
        )


def test_resume_input_of_mfda_unchanged_by_protocol_argument(tmp_path: Path) -> None:
    a, b = tmp_path / "a.lmp", tmp_path / "b.lmp"
    write_resume_input(a, tmp_path / "restart.100000", 100_000)
    write_resume_input(b, tmp_path / "restart.100000", 100_000, MFDA_PROTOCOL)
    assert a.read_bytes() == b.read_bytes()
    assert "input_mfda.lmp" in a.read_text()


# ------------------------------------------------------------ the forward model
@pytest.mark.parametrize("orthoboxy", [False, True], ids=["cubic", "orthoboxy"])
def test_md_forward_model_with_paper1_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orthoboxy: bool
) -> None:
    from mfwater.md import pipeline

    calls: list[Any] = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda p: f"/usr/bin/{p}")
    monkeypatch.setattr(
        pipeline.subprocess, "run", make_fake_run(calls, travis_files=ORTHO_FILES)
    )
    model = MDForwardModel(tmp_path, orthoboxy=orthoboxy, protocol="paper1")
    assert model.protocol is PAPER1_PROTOCOL
    assert model(100, THETA) == pytest.approx(2.5)
    attempt = model.last_attempt_dir
    assert attempt is not None
    text = (attempt / "siminp" / "input.lmp").read_text()
    assert "run 1000000 upto" in text and "run 3000000 upto" in text
    assert "minimize 1.0e-4 1.0e-6 100 1000" in text
    name = "input_orthoboxy.txt" if orthoboxy else "input_normal.txt"
    ref = (Path(__file__).parents[1] / "data" / "travis" / name).read_text()
    written = (attempt / "msd" / "travis_input_msd.txt").read_text()
    assert _commands(written) == _commands(ref)  # as the paper-I inputs


def test_unknown_protocol_in_forward_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown MD protocol"):
        MDForwardModel(tmp_path, protocol="nope")
