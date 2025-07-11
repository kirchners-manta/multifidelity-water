"""
Test the get_ordered_index_combinations function with a simple case.
"""

import pytest

from mfwater import get_ordered_index_combinations


def test_index_combinations_too_small() -> None:
    """Test get_ordered_index_combinations with a too small number of models."""
    with pytest.raises(ValueError):
        get_ordered_index_combinations(1)
