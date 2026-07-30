import pytest
from pydantic import ValidationError

from synthbold.config import GeometryParams, LogNormalRange, Range

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


# --- LogNormalRange ---


def test_log_normal_range_valid() -> None:
    r = LogNormalRange(mean=0.1, std=0.02)
    assert r.mean == 0.1
    assert r.std == 0.02


def test_geometry_params_default_spline_fields() -> None:
    geom = GeometryParams()
    assert geom.spline_nb_levels == 1
    assert geom.spline_radius == LogNormalRange(mean=0.1, std=0.02)
