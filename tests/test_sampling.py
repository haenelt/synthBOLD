import math

import pytest
import torch

from synthbold.sampling import (
    sample_1d,
    sample_log_uniform,
    sample_polar,
    sample_uniform,
)

DEVICE = torch.device("cpu")


def _generator(seed: int = 0) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


# --- sample_uniform ---


def test_sample_uniform_output_shape_int_size() -> None:
    out = sample_uniform(5, 0.0, 1.0, DEVICE, _generator())
    assert out.shape == torch.Size((5,))


def test_sample_uniform_output_shape_tuple_size() -> None:
    out = sample_uniform((2, 3), 0.0, 1.0, DEVICE, _generator())
    assert out.shape == torch.Size((2, 3))


def test_sample_uniform_output_dtype() -> None:
    out = sample_uniform(5, 0.0, 1.0, DEVICE, _generator())
    assert out.dtype == torch.float32


def test_sample_uniform_values_within_range() -> None:
    out = sample_uniform(1000, 2.0, 5.0, DEVICE, _generator())
    assert out.min().item() >= 2.0
    assert out.max().item() < 5.0


def test_sample_uniform_reproducible() -> None:
    out1 = sample_uniform(10, 0.0, 1.0, DEVICE, _generator(42))
    out2 = sample_uniform(10, 0.0, 1.0, DEVICE, _generator(42))
    assert torch.equal(out1, out2)


def test_sample_uniform_different_seeds_differ() -> None:
    out1 = sample_uniform(10, 0.0, 1.0, DEVICE, _generator(1))
    out2 = sample_uniform(10, 0.0, 1.0, DEVICE, _generator(2))
    assert not torch.equal(out1, out2)


def test_sample_uniform_degenerate_range_returns_constant() -> None:
    out = sample_uniform(5, 3.0, 3.0, DEVICE, _generator())
    assert torch.allclose(out, torch.full((5,), 3.0))


# --- sample_log_uniform ---


def test_sample_log_uniform_output_shape() -> None:
    out = sample_log_uniform(5, 0.1, 10.0, DEVICE, _generator())
    assert out.shape == torch.Size((5,))


def test_sample_log_uniform_values_within_range() -> None:
    out = sample_log_uniform(1000, 0.1, 10.0, DEVICE, _generator())
    assert out.min().item() >= 0.1
    assert out.max().item() < 10.0


def test_sample_log_uniform_reproducible() -> None:
    out1 = sample_log_uniform(10, 0.1, 10.0, DEVICE, _generator(42))
    out2 = sample_log_uniform(10, 0.1, 10.0, DEVICE, _generator(42))
    assert torch.equal(out1, out2)


def test_sample_log_uniform_zero_low_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        sample_log_uniform(5, 0.0, 10.0, DEVICE, _generator())


def test_sample_log_uniform_negative_high_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        sample_log_uniform(5, 0.1, -1.0, DEVICE, _generator())


def test_sample_log_uniform_matches_exp_of_uniform_log_space() -> None:
    # log_uniform(low, high) should be exp of a uniform draw in [log(low), log(high)).
    low, high = 0.5, 8.0
    out = sample_log_uniform(20, low, high, DEVICE, _generator(7))
    expected = torch.exp(
        sample_uniform(20, math.log(low), math.log(high), DEVICE, _generator(7))
    )
    assert torch.allclose(out, expected)


def test_sample_log_uniform_evenly_covers_orders_of_magnitude() -> None:
    # Plain uniform sampling over a multi-decade range concentrates draws near
    # `high`; log-uniform sampling should give roughly balanced coverage per decade.
    out = sample_log_uniform(5000, 0.01, 100.0, DEVICE, _generator(0))
    frac_below_1 = (out < 1.0).float().mean().item()
    # Uniform-in-log over 4 decades ([0.01,100)) puts ~2 decades below 1.0.
    assert frac_below_1 == pytest.approx(0.5, abs=0.05)


# --- sample_polar ---


def test_sample_polar_output_shape() -> None:
    out = sample_polar(5, 0.0, math.pi, DEVICE, _generator())
    assert out.shape == torch.Size((5,))


def test_sample_polar_values_within_range() -> None:
    out = sample_polar(1000, 0.0, math.pi, DEVICE, _generator())
    assert out.min().item() >= 0.0
    assert out.max().item() <= math.pi


def test_sample_polar_reproducible() -> None:
    out1 = sample_polar(10, 0.0, math.pi, DEVICE, _generator(42))
    out2 = sample_polar(10, 0.0, math.pi, DEVICE, _generator(42))
    assert torch.equal(out1, out2)


def test_sample_polar_different_seeds_differ() -> None:
    out1 = sample_polar(10, 0.0, math.pi, DEVICE, _generator(1))
    out2 = sample_polar(10, 0.0, math.pi, DEVICE, _generator(2))
    assert not torch.equal(out1, out2)


def test_sample_polar_full_range_is_isotropic_not_uniform_in_angle() -> None:
    # Isotropic sampling on the sphere concentrates more mass near the equator
    # (theta = pi/2) than near the poles, unlike uniform-in-angle sampling.
    out = sample_polar(20000, 0.0, math.pi, DEVICE, _generator(0))
    frac_near_equator = ((out - math.pi / 2).abs() < math.pi / 6).float().mean().item()
    frac_near_pole = (out < math.pi / 6).float().mean().item()
    assert frac_near_equator > frac_near_pole


def test_sample_polar_degenerate_range_returns_constant() -> None:
    out = sample_polar(5, math.pi / 4, math.pi / 4, DEVICE, _generator())
    assert torch.allclose(out, torch.full((5,), math.pi / 4), atol=1e-6)


# --- sample_1d ---


def test_sample_1d_returns_given_value_moved_to_device_and_dtype() -> None:
    value = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    out = sample_1d(value, "value", 3, 0.0, 1.0, DEVICE, _generator())
    assert torch.equal(out, value.to(DEVICE, dtype=torch.float32))
    assert out.dtype == torch.float32
    assert out.device == DEVICE


def test_sample_1d_samples_uniform_when_none() -> None:
    out = sample_1d(None, "value", 100, 2.0, 5.0, DEVICE, _generator())
    assert out.shape == torch.Size((100,))
    assert out.min().item() >= 2.0
    assert out.max().item() < 5.0


def test_sample_1d_polar_samples_within_range() -> None:
    out = sample_1d(None, "value", 100, 0.0, math.pi, DEVICE, _generator(), polar=True)
    assert out.shape == torch.Size((100,))
    assert out.min().item() >= 0.0
    assert out.max().item() <= math.pi


def test_sample_1d_raises_on_wrong_shape() -> None:
    value = torch.zeros(5)
    with pytest.raises(ValueError):
        sample_1d(value, "value", 3, 0.0, 1.0, DEVICE, _generator())
