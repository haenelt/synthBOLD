"""Tests for configuration structures."""

import pytest
from pydantic import ValidationError

from synthbold.config import Range


def test_range_valid() -> None:
    """Test that a valid Range can be instantiated."""
    r = Range(min=1.5, max=5.0)
    assert r.min == 1.5
    assert r.max == 5.0


def test_range_min_equals_max() -> None:
    """Test that Range allows min to be exactly equal to max."""
    r = Range(min=2.0, max=2.0)
    assert r.min == 2.0
    assert r.max == 2.0


def test_range_invalid_max_less_than_min() -> None:
    """Test that Range raises a ValidationError if max is less than min."""
    with pytest.raises(ValidationError):
        Range(min=5.0, max=1.0)
