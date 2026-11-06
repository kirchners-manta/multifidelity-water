"""Tests for the Yeh-Hummer finite-size correction."""

from __future__ import annotations

import pytest

from mfwater.algo_mfda.finite_size import (
    resolve_fs_corrections,
    yeh_hummer_correction,
)

# msdiff --hummer 298.15 0.89e-3 reports K = 444.555578e-12 m^2/s for N = 128 (cubic)
MSDIFF_K_128 = 444.555578e-3  # in 1e-9 m^2/s


def test_n128_matches_msdiff() -> None:
    assert yeh_hummer_correction(128, False) == pytest.approx(MSDIFF_K_128, rel=1e-3)


def test_scales_with_inverse_box_length() -> None:
    # L ~ N^(1/3): 8 times the molecules double L and halve C
    assert yeh_hummer_correction(64, False) == pytest.approx(
        2.0 * yeh_hummer_correction(512, False), rel=1e-12
    )


def test_orthoboxy_is_zero() -> None:
    assert yeh_hummer_correction(128, True) == 0.0
    assert resolve_fs_corrections("yeh-hummer", [128, 32], True) == [0.0, 0.0]


def test_resolve() -> None:
    assert resolve_fs_corrections("none", [128, 32], False) == [0.0, 0.0]
    c = resolve_fs_corrections("yeh-hummer", [512, 128, 32], False)
    assert c == sorted(c) and all(x > 0 for x in c)
    with pytest.raises(ValueError, match="fs-correction"):
        resolve_fs_corrections("hummer", [128], False)
