import pytest
from pydantic import ValidationError

from synthbold.config import Range

# --- Range ---


def test_range_valid() -> None:
    r = Range(min=1.5, max=5.0)
    assert r.min == 1.5
    assert r.max == 5.0


def test_range_min_equals_max() -> None:
    r = Range(min=2.0, max=2.0)
    assert r.min == r.max == 2.0


def test_range_invalid_max_less_than_min() -> None:
    with pytest.raises(ValidationError):
        Range(min=5.0, max=1.0)
