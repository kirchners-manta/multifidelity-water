"""Tests for the core-count rule ``calc_cpus``."""

import pytest

from mfwater.algo_chemical_model.chemical_model import calc_cpus


@pytest.mark.parametrize(
    "n, expected",
    [
        (32, 1),
        (63, 1),
        (64, 2),  # measured sizes of the Marvin benchmark: 32 .. 1024
        (128, 2),
        (255, 2),
        (256, 4),
        (511, 4),
        (512, 8),
        (1024, 8),
        (2000, 8),  # not measured, value kept from the previous version
    ],
)
def test_calc_cpus(n: int, expected: int) -> None:
    assert calc_cpus(n) == expected


def test_calc_cpus_is_monotonic() -> None:
    values = [calc_cpus(n) for n in range(1, 3000)]
    assert values == sorted(values)
