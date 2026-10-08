"""Tests for ``chemical_model_post``: reading D from both msdiff CSV layouts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import pytest
from mfwater import console_entry_point
from mfwater.msdiff_io import parse_msdiff_d_raw

DATA = Path(__file__).parent.parent / "test_cli" / "data" / "chemmodel-post"
OLD_CSV = Path(__file__).parent.parent / "test_mfda" / "data" / "msdiff_out_old.csv"
NEW_CSV = (
    " Species, D_0 / 10^-12 m^2/s,        delta_D, K / 10^-12 m^2/s\n"
    "       1,        2500.000000,       0.000100,       500.000000\n"
)


def test_raw_parser_both_layouts(tmp_path: Path) -> None:
    # raw unit is 1e-12 m^2/s and K must never be added
    assert parse_msdiff_d_raw(OLD_CSV) == pytest.approx(1209.28281695)
    f = tmp_path / "msdiff_out.csv"
    f.write_text(NEW_CSV)
    assert parse_msdiff_d_raw(f) == pytest.approx(2500.0)


@pytest.mark.parametrize("layout", ["old", "new"])
def test_chemmodel_post_msdiff_layouts(tmp_path: Path, layout: str) -> None:
    """All msdiff files are replaced by one layout; D stays in 1e-12 m^2/s."""
    shutil.copytree(DATA / "models", tmp_path / "models")
    shutil.copy(DATA / "test_chemmodel-post.hdf5", tmp_path / "in.hdf5")
    for csv in (tmp_path / "models").rglob("msdiff_out.csv"):
        if layout == "old":
            shutil.copy(OLD_CSV, csv)
        else:
            csv.write_text(NEW_CSV)
    expected = 1209.28281695 if layout == "old" else 2500.0

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        rc = console_entry_point(
            f"-a chemmodel-post -i {tmp_path / 'in.hdf5'} -o {tmp_path / 'out.hdf5'}".split()
        )
    finally:
        os.chdir(cwd)
    assert rc == 0
    with h5py.File(tmp_path / "in.hdf5", "r") as f:
        for name, mod in f["models"].items():
            if isinstance(mod, h5py.Group):
                assert mod["diffusion_coeff"][:] == pytest.approx(expected, rel=1e-6)
