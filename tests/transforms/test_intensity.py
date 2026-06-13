import pytest
import torch

from synthbold.config import Config
from synthbold.transforms.intensity import BiasField, GammaTransform

SMALL_SHAPE = (8, 8, 8)
LOWRES_SHAPE = (2, 2, 2)


# --- BiasField ---


def test_biasfield_init_stores_attrs() -> None:
    bf = BiasField(
        lowres_shape=LOWRES_SHAPE, amplitude_range=(0.1, 0.5), device="cpu", seed=0
    )
    assert bf.lowres_shape == LOWRES_SHAPE
    assert bf.amplitude_range == (0.1, 0.5)
    assert bf.device == torch.device("cpu")
    assert bf.seed == 0


def test_biasfield_from_config_returns_instance() -> None:
    config = Config()
    bf = BiasField.from_config(config)
    assert isinstance(bf, BiasField)
    assert bf.lowres_shape == config.transform.bias_lowres_shape
    assert bf.amplitude_range == (
        config.transform.bias_amplitude.min,
        config.transform.bias_amplitude.max,
    )


def test_biasfield_sample_shape_batched() -> None:
    bf = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    bias = bf.sample(shape)
    assert bias.shape == torch.Size(shape)


def test_biasfield_sample_shape_batch1() -> None:
    bf = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    bias = bf.sample(shape)
    assert bias.shape == torch.Size(shape)


def test_biasfield_sample_positive() -> None:
    bf = BiasField(
        lowres_shape=LOWRES_SHAPE, amplitude_range=(0.0, 2.0), device="cpu", seed=0
    )
    bias = bf.sample((4, *SMALL_SHAPE))
    assert (bias > 0).all()


def test_biasfield_sample_reproducible() -> None:
    bf1 = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=7)
    bf2 = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=7)
    b1 = bf1.sample((2, *SMALL_SHAPE))
    b2 = bf2.sample((2, *SMALL_SHAPE))
    assert torch.equal(b1, b2)


def test_biasfield_sample_different_seeds_differ() -> None:
    bf1 = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=1)
    bf2 = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=2)
    b1 = bf1.sample((2, *SMALL_SHAPE))
    b2 = bf2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(b1, b2)


def test_biasfield_zero_amplitude_produces_ones() -> None:
    # amplitude == 0 => exp(noise * 0) == 1 everywhere
    bf = BiasField(
        lowres_shape=LOWRES_SHAPE, amplitude_range=(0.0, 0.0), device="cpu", seed=0
    )
    bias = bf.sample((2, *SMALL_SHAPE))
    assert torch.allclose(bias, torch.ones_like(bias))


def test_biasfield_apply_multiplies() -> None:
    bf = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu")
    x = torch.ones(2, *SMALL_SHAPE)
    bias = torch.full((2, *SMALL_SHAPE), 3.0)
    out = bf.apply(x, bias)
    assert torch.allclose(out, bias)


def test_biasfield_apply_preserves_shape() -> None:
    bf = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu")
    x = torch.rand(3, *SMALL_SHAPE)
    bias = torch.rand(3, *SMALL_SHAPE) + 0.5
    out = bf.apply(x, bias)
    assert out.shape == x.shape


def test_biasfield_call_4d_shape() -> None:
    bf = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=0)
    x = torch.ones(3, *SMALL_SHAPE)
    out = bf(x)
    assert out.shape == x.shape


def test_biasfield_call_3d_shape() -> None:
    bf = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=0)
    x = torch.ones(*SMALL_SHAPE)
    out = bf(x)
    assert out.shape == x.shape


def test_biasfield_call_wrong_dim_raises() -> None:
    bf = BiasField(lowres_shape=LOWRES_SHAPE, device="cpu", seed=0)
    x = torch.ones(2, 2)
    with pytest.raises(ValueError):
        bf(x)


def test_biasfield_call_output_positive() -> None:
    bf = BiasField(
        lowres_shape=LOWRES_SHAPE, amplitude_range=(0.0, 1.0), device="cpu", seed=0
    )
    x = torch.ones(2, *SMALL_SHAPE)
    out = bf(x)
    assert (out > 0).all()


# --- GammaTransform ---


def test_gammatransform_init_stores_attrs() -> None:
    gt = GammaTransform(gamma_range=(0.5, 1.5), device="cpu", seed=0)
    assert gt.gamma_range == (0.5, 1.5)
    assert gt.device == torch.device("cpu")
    assert gt.seed == 0


def test_gammatransform_init_nonpositive_range_raises() -> None:
    with pytest.raises(ValueError):
        GammaTransform(gamma_range=(0.0, 1.5), device="cpu")


def test_gammatransform_from_config_returns_instance() -> None:
    config = Config()
    gt = GammaTransform.from_config(config)
    assert isinstance(gt, GammaTransform)
    assert gt.gamma_range == (
        config.transform.gamma_exponent.min,
        config.transform.gamma_exponent.max,
    )


def test_gammatransform_sample_shape_batched() -> None:
    gt = GammaTransform(device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    gamma = gt.sample(shape)
    assert gamma.shape == (3, 1, 1, 1)


def test_gammatransform_sample_within_range() -> None:
    gt = GammaTransform(gamma_range=(0.5, 1.5), device="cpu", seed=0)
    gamma = gt.sample((4, *SMALL_SHAPE))
    assert (gamma >= 0.5).all()
    assert (gamma <= 1.5).all()


def test_gammatransform_sample_reproducible() -> None:
    gt1 = GammaTransform(device="cpu", seed=7)
    gt2 = GammaTransform(device="cpu", seed=7)
    g1 = gt1.sample((2, *SMALL_SHAPE))
    g2 = gt2.sample((2, *SMALL_SHAPE))
    assert torch.equal(g1, g2)


def test_gammatransform_sample_different_seeds_differ() -> None:
    gt1 = GammaTransform(device="cpu", seed=1)
    gt2 = GammaTransform(device="cpu", seed=2)
    g1 = gt1.sample((2, *SMALL_SHAPE))
    g2 = gt2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(g1, g2)


def test_gammatransform_apply_identity_gamma_one() -> None:
    gt = GammaTransform(device="cpu")
    x = torch.rand(2, *SMALL_SHAPE)
    gamma = torch.ones(2, 1, 1, 1)
    out = gt.apply(x, gamma)
    assert torch.allclose(out, x, atol=1e-5)


def test_gammatransform_apply_preserves_shape() -> None:
    gt = GammaTransform(device="cpu")
    x = torch.rand(3, *SMALL_SHAPE)
    gamma = torch.full((3, 1, 1, 1), 2.0)
    out = gt.apply(x, gamma)
    assert out.shape == x.shape


def test_gammatransform_apply_preserves_min_max() -> None:
    gt = GammaTransform(device="cpu")
    x = torch.rand(2, *SMALL_SHAPE) * 10 - 3
    gamma = torch.full((2, 1, 1, 1), 0.5)
    out = gt.apply(x, gamma)
    x_min = x.amin(dim=(-3, -2, -1))
    x_max = x.amax(dim=(-3, -2, -1))
    out_min = out.amin(dim=(-3, -2, -1))
    out_max = out.amax(dim=(-3, -2, -1))
    assert torch.allclose(out_min, x_min, atol=1e-5)
    assert torch.allclose(out_max, x_max, atol=1e-5)


def test_gammatransform_apply_constant_volume_unchanged() -> None:
    gt = GammaTransform(device="cpu")
    x = torch.full((2, *SMALL_SHAPE), 5.0)
    gamma = torch.full((2, 1, 1, 1), 2.0)
    out = gt.apply(x, gamma)
    assert torch.equal(out, x)


def test_gammatransform_apply_gamma_below_one_brightens() -> None:
    gt = GammaTransform(device="cpu")
    x = torch.linspace(0, 1, 8).reshape(1, 2, 2, 2)
    gamma = torch.full((1, 1, 1, 1), 0.5)
    out = gt.apply(x, gamma)
    interior = (x > 0) & (x < 1)
    assert (out[interior] >= x[interior]).all()
