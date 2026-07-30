from typing import Self

import torch

from synthbold.base import Transform
from synthbold.config import Config
from synthbold.transforms.functional import (
    Pipeline,
    apply_deformation,
    percentile_scale,
    psc,
    sample_fractal_noise,
    sample_lowres_noise,
    zscore,
)

SHAPE_CUBE = (6, 6, 6)
SHAPE_NONCUBE = (4, 6, 8)


class DummyTransform(Transform):
    """Dummy transform for testing."""

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.ones(shape) * 2.0

    @staticmethod
    def apply(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return x + transform

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(device="cpu")


# --- sample_lowres_noise ---


def test_sample_lowres_noise_output_shape() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    noise = sample_lowres_noise(
        3, (2, 3, 4), SHAPE_NONCUBE, torch.device("cpu"), generator
    )
    assert noise.shape == torch.Size((3, *SHAPE_NONCUBE))


def test_sample_lowres_noise_batch1_shape() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    noise = sample_lowres_noise(
        1, (2, 2, 2), SHAPE_CUBE, torch.device("cpu"), generator
    )
    assert noise.shape == torch.Size((1, *SHAPE_CUBE))


def test_sample_lowres_noise_reproducible() -> None:
    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(42)
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(42)

    n1 = sample_lowres_noise(2, (2, 2, 2), SHAPE_CUBE, torch.device("cpu"), gen1)
    n2 = sample_lowres_noise(2, (2, 2, 2), SHAPE_CUBE, torch.device("cpu"), gen2)
    assert torch.equal(n1, n2)


def test_sample_lowres_noise_different_seeds_differ() -> None:
    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(1)
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(2)

    n1 = sample_lowres_noise(2, (2, 2, 2), SHAPE_CUBE, torch.device("cpu"), gen1)
    n2 = sample_lowres_noise(2, (2, 2, 2), SHAPE_CUBE, torch.device("cpu"), gen2)
    assert not torch.equal(n1, n2)


def test_sample_lowres_noise_batch_elements_independent() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    noise = sample_lowres_noise(
        2, (2, 2, 2), SHAPE_CUBE, torch.device("cpu"), generator
    )
    assert not torch.equal(noise[0], noise[1])


def test_sample_lowres_noise_identity_when_shapes_match() -> None:
    # With lowres_shape == target_shape, interpolation is a no-op and the result
    # is just the raw Gaussian draw, regardless of align_corners.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    noise = sample_lowres_noise(
        1, SHAPE_CUBE, SHAPE_CUBE, torch.device("cpu"), generator
    )

    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(0)
    expected = torch.randn(1, *SHAPE_CUBE, generator=expected_generator)

    assert torch.allclose(noise, expected, atol=1e-6)


def test_sample_lowres_noise_align_corners_changes_output() -> None:
    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(0)
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(0)

    n1 = sample_lowres_noise(
        1, (2, 2, 2), (8, 8, 8), torch.device("cpu"), gen1, align_corners=False
    )
    n2 = sample_lowres_noise(
        1, (2, 2, 2), (8, 8, 8), torch.device("cpu"), gen2, align_corners=True
    )
    assert not torch.equal(n1, n2)


def test_sample_lowres_noise_is_smoother_than_full_resolution_noise() -> None:
    target_shape = (16, 16, 16)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    smooth = sample_lowres_noise(
        1, (2, 2, 2), target_shape, torch.device("cpu"), generator
    )

    raw_generator = torch.Generator(device="cpu")
    raw_generator.manual_seed(0)
    raw = torch.randn(1, *target_shape, generator=raw_generator)

    smooth_diff = (smooth[:, 1:] - smooth[:, :-1]).abs().mean()
    raw_diff = (raw[:, 1:] - raw[:, :-1]).abs().mean()
    assert smooth_diff < raw_diff


# --- sample_fractal_noise ---


def test_sample_fractal_noise_output_shape() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    noise = sample_fractal_noise(
        3, (2, 2, 2), SHAPE_NONCUBE, 4, 0.5, torch.device("cpu"), generator
    )
    assert noise.shape == torch.Size((3, *SHAPE_NONCUBE))


def test_sample_fractal_noise_reproducible() -> None:
    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(42)
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(42)

    device = torch.device("cpu")
    n1 = sample_fractal_noise(2, (2, 2, 2), SHAPE_CUBE, 3, 0.5, device, gen1)
    n2 = sample_fractal_noise(2, (2, 2, 2), SHAPE_CUBE, 3, 0.5, device, gen2)
    assert torch.equal(n1, n2)


def test_sample_fractal_noise_different_seeds_differ() -> None:
    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(1)
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(2)

    device = torch.device("cpu")
    n1 = sample_fractal_noise(2, (2, 2, 2), SHAPE_CUBE, 3, 0.5, device, gen1)
    n2 = sample_fractal_noise(2, (2, 2, 2), SHAPE_CUBE, 3, 0.5, device, gen2)
    assert not torch.equal(n1, n2)


def test_sample_fractal_noise_batch_elements_independent() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    noise = sample_fractal_noise(
        2, (2, 2, 2), SHAPE_CUBE, 3, 0.5, torch.device("cpu"), generator
    )
    assert not torch.equal(noise[0], noise[1])


def test_sample_fractal_noise_single_octave_matches_lowres_noise() -> None:
    # With a single octave, the only amplitude weight is 1.0, so the normalized
    # result reduces to a single call of sample_lowres_noise.
    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(0)
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(0)

    fractal = sample_fractal_noise(
        1, (2, 2, 2), SHAPE_CUBE, 1, 0.5, torch.device("cpu"), gen1
    )
    lowres = sample_lowres_noise(1, (2, 2, 2), SHAPE_CUBE, torch.device("cpu"), gen2)
    assert torch.allclose(fractal, lowres)


def test_sample_fractal_noise_more_octaves_add_higher_frequency_detail() -> None:
    target_shape = (16, 16, 16)

    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(0)
    one_octave = sample_fractal_noise(
        1, (2, 2, 2), target_shape, 1, 0.5, torch.device("cpu"), gen1
    )

    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(0)
    many_octaves = sample_fractal_noise(
        1, (2, 2, 2), target_shape, 4, 0.5, torch.device("cpu"), gen2
    )

    one_diff = (one_octave[:, 1:] - one_octave[:, :-1]).abs().mean()
    many_diff = (many_octaves[:, 1:] - many_octaves[:, :-1]).abs().mean()
    assert many_diff > one_diff


# --- apply_deformation ---


def test_apply_deformation_preserves_shape() -> None:
    data = torch.rand(2, *SHAPE_NONCUBE)
    disp = torch.rand(2, *SHAPE_NONCUBE, 3)
    out = apply_deformation(data, disp)
    assert out.shape == data.shape


def test_apply_deformation_zero_displacement_is_identity_cubic() -> None:
    data = torch.rand(1, *SHAPE_CUBE)
    out = apply_deformation(data, torch.zeros((1, *SHAPE_CUBE, 3)))
    assert torch.allclose(out, data, atol=1e-6)


def test_apply_deformation_zero_displacement_is_identity_noncubic() -> None:
    # Regression test: a previous axis-ordering bug swapped the X and Z axes, which
    # only became visible for non-cubic shapes.
    data = torch.rand(1, *SHAPE_NONCUBE)
    out = apply_deformation(data, torch.zeros((1, *SHAPE_NONCUBE, 3)))
    assert torch.allclose(out, data, atol=1e-6)


def test_apply_deformation_zero_displacement_nearest_is_exact() -> None:
    data = torch.rand(1, *SHAPE_NONCUBE)
    out = apply_deformation(data, torch.zeros((1, *SHAPE_NONCUBE, 3)), interp="nearest")
    assert torch.equal(out, data)


def test_apply_deformation_shift_along_x() -> None:
    data = torch.zeros(1, 10, 10, 10)
    data[0, 5, 5, 5] = 1.0
    disp = torch.zeros(1, 10, 10, 10, 3)
    disp[..., 0] = 2.0
    out = apply_deformation(data, disp, interp="nearest")
    assert (out > 0.5).nonzero().tolist() == [[0, 3, 5, 5]]


def test_apply_deformation_shift_along_y() -> None:
    data = torch.zeros(1, 10, 10, 10)
    data[0, 5, 5, 5] = 1.0
    disp = torch.zeros(1, 10, 10, 10, 3)
    disp[..., 1] = 2.0
    out = apply_deformation(data, disp, interp="nearest")
    assert (out > 0.5).nonzero().tolist() == [[0, 5, 3, 5]]


def test_apply_deformation_shift_along_z() -> None:
    data = torch.zeros(1, 10, 10, 10)
    data[0, 5, 5, 5] = 1.0
    disp = torch.zeros(1, 10, 10, 10, 3)
    disp[..., 2] = 2.0
    out = apply_deformation(data, disp, interp="nearest")
    assert (out > 0.5).nonzero().tolist() == [[0, 5, 5, 3]]


def test_apply_deformation_preserves_integer_dtype() -> None:
    data = torch.randint(0, 5, (1, *SHAPE_CUBE), dtype=torch.int64)
    disp = torch.zeros((1, *SHAPE_CUBE, 3))
    out = apply_deformation(data, disp, interp="nearest")
    assert out.dtype == data.dtype
    assert torch.equal(out, data)


def test_apply_deformation_batch_elements_independent() -> None:
    torch.manual_seed(0)
    data = torch.rand(2, *SHAPE_CUBE)
    disp = torch.zeros(2, *SHAPE_CUBE, 3)
    disp[0, ..., 0] = 1.0
    disp[1, ..., 0] = -1.0

    out = apply_deformation(data, disp, interp="nearest")
    out0 = apply_deformation(data[:1], disp[:1], interp="nearest")
    out1 = apply_deformation(data[1:], disp[1:], interp="nearest")

    assert torch.equal(out[:1], out0)
    assert torch.equal(out[1:], out1)
    assert not torch.equal(out[0], out[1])


# --- Pipeline ---


def test_pipeline() -> None:
    """Test the Pipeline execution."""
    t1 = DummyTransform(device="cpu")
    t2 = DummyTransform(device="cpu")

    pipeline = Pipeline([t1, t2], prob=1.0)

    input_data = torch.ones((2, 2, 2, 2))

    # prob=1.0 forces both transforms to apply to every batch element:
    # t1 adds 2.0 -> 3.0
    # t2 adds 2.0 -> 5.0
    output = pipeline(input_data)

    assert output.shape == (2, 2, 2, 2)
    assert torch.all(output == 5.0)
    assert len(pipeline.transforms) == 2


# --- zscore ---


def test_zscore_output_shape() -> None:
    data = torch.rand(3, 4, 5, 6)
    out = zscore(data)
    assert out.shape == data.shape


def test_zscore_zero_mean_unit_std() -> None:
    torch.manual_seed(0)
    data = torch.randn(4, 10, 10, 10) * 5 + 3
    out = zscore(data)
    dims = (1, 2, 3)
    assert torch.allclose(out.mean(dim=dims), torch.zeros(4), atol=1e-5)
    assert torch.allclose(out.std(dim=dims, unbiased=False), torch.ones(4), atol=1e-4)


def test_zscore_constant_input_is_near_zero() -> None:
    # std is 0 for a constant input; eps in the denominator should keep this finite
    # rather than producing NaN/inf.
    data = torch.full((2, 3, 3, 3), 7.0)
    out = zscore(data)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)


def test_zscore_invariant_to_affine_transform() -> None:
    # z-scoring is invariant to per-element scale and offset, so an affine transform
    # of one batch element should normalize identically to another.
    base = torch.arange(64).reshape(4, 4, 4).float()
    data = torch.stack([base, base * 10 + 100])
    out = zscore(data)
    assert torch.allclose(out[0], out[1], atol=1e-4)


# --- psc ---


def test_psc_output_shape() -> None:
    data = torch.rand(3, 4, 5, 6) + 1.0
    out = psc(data)
    assert out.shape == data.shape


def test_psc_known_values() -> None:
    data = torch.tensor([8.0, 10.0, 12.0]).view(1, 3, 1, 1)
    out = psc(data)
    expected = torch.tensor([-20.0, 0.0, 20.0]).view(1, 3, 1, 1)
    assert torch.allclose(out, expected, atol=1e-4)


def test_psc_invariant_to_scale() -> None:
    # percent signal change is invariant to a per-element multiplicative scale, so
    # scaling one batch element should give the same percent deviations as another.
    base = torch.tensor([8.0, 10.0, 12.0])
    data = torch.stack([base, base * 10]).view(2, 3, 1, 1)
    out = psc(data)
    assert torch.allclose(out[0], out[1], atol=1e-4)


# --- percentile_scale ---


def test_percentile_scale_output_shape() -> None:
    data = torch.rand(3, 4, 5, 6)
    out = percentile_scale(data)
    assert out.shape == data.shape


def test_percentile_scale_output_in_unit_range() -> None:
    torch.manual_seed(0)
    data = torch.randn(2, 10, 10, 10)
    out = percentile_scale(data)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_percentile_scale_known_values() -> None:
    data = torch.linspace(0, 100, 101).view(1, 101, 1, 1)
    out = percentile_scale(data, lower=0.1, upper=0.9).view(-1)
    assert torch.isclose(out[10], torch.tensor(0.0), atol=1e-3)
    assert torch.isclose(out[50], torch.tensor(0.5), atol=1e-3)
    assert torch.isclose(out[90], torch.tensor(1.0), atol=1e-3)


def test_percentile_scale_clamps_outliers() -> None:
    data = torch.linspace(0, 100, 101).view(1, 101, 1, 1)
    out = percentile_scale(data, lower=0.1, upper=0.9).view(-1)
    assert out[0] == 0.0
    assert out[-1] == 1.0


def test_percentile_scale_batch_elements_independent() -> None:
    # Same relative distribution shape (linear rescale), so per-element percentile
    # normalization should produce identical output regardless of the raw scale.
    base = torch.linspace(0, 100, 101)
    data = torch.stack([base, base * 10]).view(2, 101, 1, 1)
    out = percentile_scale(data)
    assert torch.allclose(out[0], out[1], atol=1e-4)
