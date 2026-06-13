import pytest
import torch

from synthbold.config import Config
from synthbold.transforms.noise import (
    GaussianNoise,
    KSpaceSpikeNoise,
    MultiplicativeGammaNoise,
    NoncentralChiNoise,
    PerlinNoise,
    PoissonNoise,
    RicianNoise,
    SpeckleNoise,
)

SMALL_SHAPE = (8, 8, 8)


# --- GaussianNoise ---


def test_gaussiannoise_init_stores_attrs() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.1, 0.5), device="cpu", seed=0)
    assert gn.mu_range == (0.0, 1.0)
    assert gn.std_range == (0.1, 0.5)
    assert gn.device == torch.device("cpu")
    assert gn.seed == 0


def test_gaussiannoise_from_config_returns_instance() -> None:
    config = Config()
    gn = GaussianNoise.from_config(config)
    assert isinstance(gn, GaussianNoise)
    assert gn.mu_range == (
        config.transform.gnoise_mu.min,
        config.transform.gnoise_mu.max,
    )
    assert gn.std_range == (
        config.transform.gnoise_std.min,
        config.transform.gnoise_std.max,
    )


def test_gaussiannoise_sample_shape_batched() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    noise = gn.sample(shape)
    assert noise.shape == torch.Size(shape)


def test_gaussiannoise_sample_shape_batch1() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    noise = gn.sample(shape)
    assert noise.shape == torch.Size(shape)


def test_gaussiannoise_sample_reproducible() -> None:
    gn1 = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=7)
    gn2 = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=7)
    n1 = gn1.sample((2, *SMALL_SHAPE))
    n2 = gn2.sample((2, *SMALL_SHAPE))
    assert torch.equal(n1, n2)


def test_gaussiannoise_sample_different_seeds_differ() -> None:
    gn1 = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=1)
    gn2 = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=2)
    n1 = gn1.sample((2, *SMALL_SHAPE))
    n2 = gn2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(n1, n2)


def test_gaussiannoise_zero_std_is_constant_mu() -> None:
    # std == 0 => noise == mu everywhere
    mu = 5.0
    gn = GaussianNoise(mu_range=(mu, mu), std_range=(0.0, 0.0), device="cpu", seed=0)
    noise = gn.sample((2, *SMALL_SHAPE))
    assert torch.allclose(noise, torch.full_like(noise, mu))


def test_gaussiannoise_apply_adds() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu")
    x = torch.ones(2, *SMALL_SHAPE)
    noise = torch.full((2, *SMALL_SHAPE), 3.0)
    out = gn.apply(x, noise)
    assert torch.allclose(out, torch.full_like(out, 4.0))


def test_gaussiannoise_apply_preserves_shape() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu")
    x = torch.rand(3, *SMALL_SHAPE)
    noise = torch.rand(3, *SMALL_SHAPE)
    out = gn.apply(x, noise)
    assert out.shape == x.shape


def test_gaussiannoise_call_4d_shape() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.zeros(3, *SMALL_SHAPE)
    out = gn(x)
    assert out.shape == x.shape


def test_gaussiannoise_call_3d_shape() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.zeros(*SMALL_SHAPE)
    out = gn(x)
    assert out.shape == x.shape


def test_gaussiannoise_call_wrong_dim_raises() -> None:
    gn = GaussianNoise(mu_range=(0.0, 1.0), std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.ones(2, 2)
    with pytest.raises(ValueError):
        gn(x)


# --- KSpaceSpikeNoise ---


def test_kspacespikenoise_init_stores_attrs() -> None:
    ksn = KSpaceSpikeNoise(
        num_spikes_range=(2, 5), intensity_range=(0.5, 2.0), device="cpu", seed=0
    )
    assert ksn.num_spikes_range == (2, 5)
    assert ksn.intensity_range == (0.5, 2.0)
    assert ksn.device == torch.device("cpu")
    assert ksn.seed == 0


def test_kspacespikenoise_init_negative_min_raises() -> None:
    with pytest.raises(ValueError):
        KSpaceSpikeNoise(num_spikes_range=(-1, 3), device="cpu", seed=0)


def test_kspacespikenoise_init_max_below_min_raises() -> None:
    with pytest.raises(ValueError):
        KSpaceSpikeNoise(num_spikes_range=(3, 1), device="cpu", seed=0)


def test_kspacespikenoise_from_config_returns_instance() -> None:
    config = Config()
    ksn = KSpaceSpikeNoise.from_config(config)
    assert isinstance(ksn, KSpaceSpikeNoise)
    assert ksn.num_spikes_range == config.transform.spike_num
    assert ksn.intensity_range == (
        config.transform.spike_intensity.min,
        config.transform.spike_intensity.max,
    )


def test_kspacespikenoise_sample_shape() -> None:
    ksn = KSpaceSpikeNoise(num_spikes_range=(2, 2), device="cpu", seed=0)
    spikes = ksn.sample((3, *SMALL_SHAPE))
    assert spikes.shape == torch.Size((3, 2, 4))


def test_kspacespikenoise_sample_locations_within_bounds() -> None:
    ksn = KSpaceSpikeNoise(num_spikes_range=(1, 4), device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    spikes = ksn.sample(shape)
    locations = spikes[..., :3]
    for dim, size in enumerate(SMALL_SHAPE):
        assert torch.all(locations[..., dim] >= 0)
        assert torch.all(locations[..., dim] < size)


def test_kspacespikenoise_sample_intensity_within_range() -> None:
    ksn = KSpaceSpikeNoise(
        num_spikes_range=(1, 4), intensity_range=(0.5, 2.0), device="cpu", seed=0
    )
    spikes = ksn.sample((3, *SMALL_SHAPE))
    intensity = spikes[..., 3]
    assert torch.all(intensity >= 0.5)
    assert torch.all(intensity <= 2.0)


def test_kspacespikenoise_sample_reproducible() -> None:
    ksn1 = KSpaceSpikeNoise(num_spikes_range=(1, 4), device="cpu", seed=7)
    ksn2 = KSpaceSpikeNoise(num_spikes_range=(1, 4), device="cpu", seed=7)
    s1 = ksn1.sample((2, *SMALL_SHAPE))
    s2 = ksn2.sample((2, *SMALL_SHAPE))
    assert torch.equal(s1, s2)


def test_kspacespikenoise_sample_different_seeds_differ() -> None:
    ksn1 = KSpaceSpikeNoise(num_spikes_range=(1, 4), device="cpu", seed=1)
    ksn2 = KSpaceSpikeNoise(num_spikes_range=(1, 4), device="cpu", seed=2)
    s1 = ksn1.sample((2, *SMALL_SHAPE))
    s2 = ksn2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(s1, s2)


def test_kspacespikenoise_apply_preserves_shape() -> None:
    ksn = KSpaceSpikeNoise(num_spikes_range=(1, 3), device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE)
    spikes = ksn.sample(x.shape)
    out = ksn.apply(x, spikes)
    assert out.shape == x.shape


def test_kspacespikenoise_zero_spikes_is_noop() -> None:
    # num_spikes == 0 => k-space is unchanged, so the FFT round trip reconstructs the
    # original data.
    ksn = KSpaceSpikeNoise(num_spikes_range=(0, 0), device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE)
    spikes = ksn.sample(x.shape)
    assert spikes.shape == torch.Size((2, 0, 4))
    out = ksn.apply(x, spikes)
    assert torch.allclose(out, x, atol=1e-5)


def test_kspacespikenoise_apply_changes_spectrum() -> None:
    ksn = KSpaceSpikeNoise(
        num_spikes_range=(1, 1), intensity_range=(1.0, 1.0), device="cpu", seed=0
    )
    x = torch.rand(2, *SMALL_SHAPE)
    spikes = ksn.sample(x.shape)
    out = ksn.apply(x, spikes)
    assert not torch.allclose(out, x)


# --- MultiplicativeGammaNoise ---


def test_multiplicativegammanoise_init_stores_attrs() -> None:
    mgn = MultiplicativeGammaNoise(shape_range=(2.0, 50.0), device="cpu", seed=0)
    assert mgn.shape_range == (2.0, 50.0)
    assert mgn.device == torch.device("cpu")
    assert mgn.seed == 0


def test_multiplicativegammanoise_init_nonpositive_shape_raises() -> None:
    with pytest.raises(ValueError):
        MultiplicativeGammaNoise(shape_range=(0.0, 50.0), device="cpu", seed=0)


def test_multiplicativegammanoise_from_config_returns_instance() -> None:
    config = Config()
    mgn = MultiplicativeGammaNoise.from_config(config)
    assert isinstance(mgn, MultiplicativeGammaNoise)
    assert mgn.shape_range == (
        config.transform.mgamma_shape.min,
        config.transform.mgamma_shape.max,
    )


def test_multiplicativegammanoise_sample_shape() -> None:
    mgn = MultiplicativeGammaNoise(shape_range=(2.0, 50.0), device="cpu", seed=0)
    concentration = mgn.sample((3, *SMALL_SHAPE))
    assert concentration.shape == torch.Size((3, 1, 1, 1))


def test_multiplicativegammanoise_sample_within_range() -> None:
    mgn = MultiplicativeGammaNoise(shape_range=(2.0, 50.0), device="cpu", seed=0)
    concentration = mgn.sample((5, *SMALL_SHAPE))
    assert torch.all(concentration >= 2.0)
    assert torch.all(concentration <= 50.0)


def test_multiplicativegammanoise_sample_reproducible() -> None:
    mgn1 = MultiplicativeGammaNoise(shape_range=(2.0, 50.0), device="cpu", seed=7)
    mgn2 = MultiplicativeGammaNoise(shape_range=(2.0, 50.0), device="cpu", seed=7)
    c1 = mgn1.sample((4, *SMALL_SHAPE))
    c2 = mgn2.sample((4, *SMALL_SHAPE))
    assert torch.equal(c1, c2)


def test_multiplicativegammanoise_sample_different_seeds_differ() -> None:
    mgn1 = MultiplicativeGammaNoise(shape_range=(2.0, 50.0), device="cpu", seed=1)
    mgn2 = MultiplicativeGammaNoise(shape_range=(2.0, 50.0), device="cpu", seed=2)
    c1 = mgn1.sample((4, *SMALL_SHAPE))
    c2 = mgn2.sample((4, *SMALL_SHAPE))
    assert not torch.equal(c1, c2)


def test_multiplicativegammanoise_apply_preserves_shape() -> None:
    x = torch.rand(2, *SMALL_SHAPE)
    concentration = torch.full((2, 1, 1, 1), 10.0)
    out = MultiplicativeGammaNoise.apply(x, concentration)
    assert out.shape == x.shape


def test_multiplicativegammanoise_apply_zero_data_is_zero() -> None:
    x = torch.zeros(2, *SMALL_SHAPE)
    concentration = torch.full((2, 1, 1, 1), 10.0)
    out = MultiplicativeGammaNoise.apply(x, concentration)
    assert torch.allclose(out, torch.zeros_like(out))


def test_multiplicativegammanoise_apply_nonnegative_for_nonnegative_input() -> None:
    x = torch.rand(2, *SMALL_SHAPE) * 100.0
    concentration = torch.full((2, 1, 1, 1), 10.0)
    out = MultiplicativeGammaNoise.apply(x, concentration)
    assert torch.all(out >= 0.0)


def test_multiplicativegammanoise_apply_mean_close_to_data() -> None:
    torch.manual_seed(0)
    x = torch.full((1, 64, 64, 64), 10.0)
    concentration = torch.full((1, 1, 1, 1), 50.0)
    out = MultiplicativeGammaNoise.apply(x, concentration)
    assert torch.isclose(out.mean(), x.mean(), rtol=0.05)


def test_multiplicativegammanoise_apply_higher_shape_reduces_relative_noise() -> None:
    torch.manual_seed(0)
    x = torch.full((1, *SMALL_SHAPE), 10.0)

    low_shape = torch.full((1, 1, 1, 1), 1.0)
    high_shape = torch.full((1, 1, 1, 1), 100.0)

    out_low = MultiplicativeGammaNoise.apply(x, low_shape)
    out_high = MultiplicativeGammaNoise.apply(x, high_shape)

    std_low = (out_low - x).std()
    std_high = (out_high - x).std()
    assert std_low > std_high


# --- PerlinNoise ---


def test_perlinnoise_init_stores_attrs() -> None:
    pn = PerlinNoise(
        base_shape=(2, 2, 2),
        octaves=3,
        persistence=0.6,
        amplitude_range=(0.1, 0.5),
        device="cpu",
        seed=0,
    )
    assert pn.base_shape == (2, 2, 2)
    assert pn.octaves == 3
    assert pn.persistence == 0.6
    assert pn.amplitude_range == (0.1, 0.5)
    assert pn.device == torch.device("cpu")
    assert pn.seed == 0


def test_perlinnoise_init_zero_octaves_raises() -> None:
    with pytest.raises(ValueError):
        PerlinNoise(base_shape=(2, 2, 2), octaves=0, device="cpu", seed=0)


def test_perlinnoise_init_nonpositive_persistence_raises() -> None:
    with pytest.raises(ValueError):
        PerlinNoise(base_shape=(2, 2, 2), persistence=0.0, device="cpu", seed=0)


def test_perlinnoise_init_persistence_above_one_raises() -> None:
    with pytest.raises(ValueError):
        PerlinNoise(base_shape=(2, 2, 2), persistence=1.5, device="cpu", seed=0)


def test_perlinnoise_from_config_returns_instance() -> None:
    config = Config()
    pn = PerlinNoise.from_config(config)
    assert isinstance(pn, PerlinNoise)
    assert pn.base_shape == config.transform.perlin_base_shape
    assert pn.octaves == config.transform.perlin_octaves
    assert pn.persistence == config.transform.perlin_persistence
    assert pn.amplitude_range == (
        config.transform.perlin_amplitude.min,
        config.transform.perlin_amplitude.max,
    )


def test_perlinnoise_sample_shape_batched() -> None:
    pn = PerlinNoise(base_shape=(2, 2, 2), device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    noise = pn.sample(shape)
    assert noise.shape == torch.Size(shape)


def test_perlinnoise_sample_shape_batch1() -> None:
    pn = PerlinNoise(base_shape=(2, 2, 2), device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    noise = pn.sample(shape)
    assert noise.shape == torch.Size(shape)


def test_perlinnoise_sample_reproducible() -> None:
    pn1 = PerlinNoise(base_shape=(2, 2, 2), device="cpu", seed=7)
    pn2 = PerlinNoise(base_shape=(2, 2, 2), device="cpu", seed=7)
    n1 = pn1.sample((2, *SMALL_SHAPE))
    n2 = pn2.sample((2, *SMALL_SHAPE))
    assert torch.equal(n1, n2)


def test_perlinnoise_sample_different_seeds_differ() -> None:
    pn1 = PerlinNoise(base_shape=(2, 2, 2), device="cpu", seed=1)
    pn2 = PerlinNoise(base_shape=(2, 2, 2), device="cpu", seed=2)
    n1 = pn1.sample((2, *SMALL_SHAPE))
    n2 = pn2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(n1, n2)


def test_perlinnoise_zero_amplitude_is_zero() -> None:
    # amplitude == 0 => noise == 0 everywhere
    pn = PerlinNoise(
        base_shape=(2, 2, 2), amplitude_range=(0.0, 0.0), device="cpu", seed=0
    )
    noise = pn.sample((2, *SMALL_SHAPE))
    assert torch.allclose(noise, torch.zeros_like(noise))


def test_perlinnoise_apply_adds() -> None:
    pn = PerlinNoise(base_shape=(2, 2, 2), device="cpu")
    x = torch.ones(2, *SMALL_SHAPE)
    noise = torch.full((2, *SMALL_SHAPE), 3.0)
    out = pn.apply(x, noise)
    assert torch.allclose(out, torch.full_like(out, 4.0))


def test_perlinnoise_apply_preserves_shape() -> None:
    pn = PerlinNoise(base_shape=(2, 2, 2), device="cpu")
    x = torch.rand(3, *SMALL_SHAPE)
    noise = pn.sample(x.shape)
    out = pn.apply(x, noise)
    assert out.shape == x.shape


# --- PoissonNoise ---


def test_poissonnoise_init_stores_attrs() -> None:
    pn = PoissonNoise(peak_range=(1.0, 10.0), device="cpu", seed=0)
    assert pn.peak_range == (1.0, 10.0)
    assert pn.device == torch.device("cpu")
    assert pn.seed == 0


def test_poissonnoise_from_config_returns_instance() -> None:
    config = Config()
    pn = PoissonNoise.from_config(config)
    assert isinstance(pn, PoissonNoise)
    assert pn.peak_range == (
        config.transform.pnoise_peak.min,
        config.transform.pnoise_peak.max,
    )


def test_poissonnoise_sample_shape() -> None:
    pn = PoissonNoise(peak_range=(1.0, 10.0), device="cpu", seed=0)
    peak = pn.sample((3, *SMALL_SHAPE))
    assert peak.shape == torch.Size((3, 1, 1, 1))


def test_poissonnoise_sample_within_range() -> None:
    pn = PoissonNoise(peak_range=(1.0, 10.0), device="cpu", seed=0)
    peak = pn.sample((5, *SMALL_SHAPE))
    assert torch.all(peak >= 1.0)
    assert torch.all(peak <= 10.0)


def test_poissonnoise_sample_reproducible() -> None:
    pn1 = PoissonNoise(peak_range=(1.0, 10.0), device="cpu", seed=7)
    pn2 = PoissonNoise(peak_range=(1.0, 10.0), device="cpu", seed=7)
    p1 = pn1.sample((4, *SMALL_SHAPE))
    p2 = pn2.sample((4, *SMALL_SHAPE))
    assert torch.equal(p1, p2)


def test_poissonnoise_sample_different_seeds_differ() -> None:
    pn1 = PoissonNoise(peak_range=(1.0, 10.0), device="cpu", seed=1)
    pn2 = PoissonNoise(peak_range=(1.0, 10.0), device="cpu", seed=2)
    p1 = pn1.sample((4, *SMALL_SHAPE))
    p2 = pn2.sample((4, *SMALL_SHAPE))
    assert not torch.equal(p1, p2)


def test_poissonnoise_apply_preserves_shape() -> None:
    x = torch.rand(2, *SMALL_SHAPE) * 100.0
    peak = torch.full((2, 1, 1, 1), 10.0)
    out = PoissonNoise.apply(x, peak)
    assert out.shape == x.shape


def test_poissonnoise_apply_zero_data_is_zero() -> None:
    x = torch.zeros(2, *SMALL_SHAPE)
    peak = torch.full((2, 1, 1, 1), 10.0)
    out = PoissonNoise.apply(x, peak)
    assert torch.allclose(out, torch.zeros_like(out))


def test_poissonnoise_apply_nonnegative_for_nonnegative_input() -> None:
    x = torch.rand(2, *SMALL_SHAPE) * 100.0
    peak = torch.full((2, 1, 1, 1), 10.0)
    out = PoissonNoise.apply(x, peak)
    assert torch.all(out >= 0.0)


def test_poissonnoise_apply_higher_peak_reduces_relative_noise() -> None:
    torch.manual_seed(0)
    x = torch.full((1, *SMALL_SHAPE), 50.0)

    low_peak = torch.full((1, 1, 1, 1), 1.0)
    high_peak = torch.full((1, 1, 1, 1), 1000.0)

    out_low = PoissonNoise.apply(x, low_peak)
    out_high = PoissonNoise.apply(x, high_peak)

    std_low = (out_low - x).std()
    std_high = (out_high - x).std()
    assert std_low > std_high


# --- RicianNoise ---


def test_riciannoise_init_stores_attrs() -> None:
    rn = RicianNoise(std_range=(0.1, 0.5), device="cpu", seed=0)
    assert rn.std_range == (0.1, 0.5)
    assert rn.device == torch.device("cpu")
    assert rn.seed == 0


def test_riciannoise_from_config_returns_instance() -> None:
    config = Config()
    rn = RicianNoise.from_config(config)
    assert isinstance(rn, RicianNoise)
    assert rn.std_range == (
        config.transform.rnoise_std.min,
        config.transform.rnoise_std.max,
    )


def test_riciannoise_sample_shape_batched() -> None:
    rn = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    noise = rn.sample(shape)
    assert noise.shape == torch.Size((2, *shape))


def test_riciannoise_sample_shape_batch1() -> None:
    rn = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    noise = rn.sample(shape)
    assert noise.shape == torch.Size((2, *shape))


def test_riciannoise_sample_reproducible() -> None:
    rn1 = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=7)
    rn2 = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=7)
    n1 = rn1.sample((2, *SMALL_SHAPE))
    n2 = rn2.sample((2, *SMALL_SHAPE))
    assert torch.equal(n1, n2)


def test_riciannoise_sample_different_seeds_differ() -> None:
    rn1 = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=1)
    rn2 = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=2)
    n1 = rn1.sample((2, *SMALL_SHAPE))
    n2 = rn2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(n1, n2)


def test_riciannoise_zero_std_is_noop_for_nonnegative_data() -> None:
    # std == 0 => both noise channels are zero, so the magnitude equals |data|.
    rn = RicianNoise(std_range=(0.0, 0.0), device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE)
    noise = rn.sample(x.shape)
    out = rn.apply(x, noise)
    assert torch.allclose(out, x)


def test_riciannoise_apply_preserves_shape() -> None:
    rn = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.rand(3, *SMALL_SHAPE)
    noise = rn.sample(x.shape)
    out = rn.apply(x, noise)
    assert out.shape == x.shape


def test_riciannoise_apply_nonnegative() -> None:
    rn = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE) - 0.5  # include negative values
    noise = rn.sample(x.shape)
    out = rn.apply(x, noise)
    assert torch.all(out >= 0.0)


# --- NoncentralChiNoise ---


def test_noncentralchinoise_init_stores_attrs() -> None:
    ncn = NoncentralChiNoise(std_range=(0.1, 0.5), dof=4, device="cpu", seed=0)
    assert ncn.std_range == (0.1, 0.5)
    assert ncn.dof == 4
    assert ncn.device == torch.device("cpu")
    assert ncn.seed == 0


def test_noncentralchinoise_init_default_dof() -> None:
    ncn = NoncentralChiNoise(std_range=(0.1, 0.5), device="cpu", seed=0)
    assert ncn.dof == 4


def test_noncentralchinoise_init_dof_too_small_raises() -> None:
    with pytest.raises(ValueError):
        NoncentralChiNoise(std_range=(0.1, 0.5), dof=1, device="cpu", seed=0)


def test_noncentralchinoise_from_config_returns_instance() -> None:
    config = Config()
    ncn = NoncentralChiNoise.from_config(config)
    assert isinstance(ncn, NoncentralChiNoise)
    assert ncn.std_range == (
        config.transform.ncchi_std.min,
        config.transform.ncchi_std.max,
    )
    assert ncn.dof == config.transform.ncchi_dof


def test_noncentralchinoise_sample_shape_batched() -> None:
    ncn = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    noise = ncn.sample(shape)
    assert noise.shape == torch.Size((4, *shape))


def test_noncentralchinoise_sample_shape_batch1() -> None:
    ncn = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    noise = ncn.sample(shape)
    assert noise.shape == torch.Size((4, *shape))


def test_noncentralchinoise_sample_reproducible() -> None:
    ncn1 = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=7)
    ncn2 = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=7)
    n1 = ncn1.sample((2, *SMALL_SHAPE))
    n2 = ncn2.sample((2, *SMALL_SHAPE))
    assert torch.equal(n1, n2)


def test_noncentralchinoise_sample_different_seeds_differ() -> None:
    ncn1 = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=1)
    ncn2 = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=2)
    n1 = ncn1.sample((2, *SMALL_SHAPE))
    n2 = ncn2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(n1, n2)


def test_noncentralchinoise_zero_std_is_noop_for_nonnegative_data() -> None:
    # std == 0 => all noise channels are zero, so the magnitude equals |data|.
    ncn = NoncentralChiNoise(std_range=(0.0, 0.0), dof=4, device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE)
    noise = ncn.sample(x.shape)
    out = ncn.apply(x, noise)
    assert torch.allclose(out, x)


def test_noncentralchinoise_apply_preserves_shape() -> None:
    ncn = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=0)
    x = torch.rand(3, *SMALL_SHAPE)
    noise = ncn.sample(x.shape)
    out = ncn.apply(x, noise)
    assert out.shape == x.shape


def test_noncentralchinoise_apply_nonnegative() -> None:
    ncn = NoncentralChiNoise(std_range=(0.0, 1.0), dof=4, device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE) - 0.5  # include negative values
    noise = ncn.sample(x.shape)
    out = ncn.apply(x, noise)
    assert torch.all(out >= 0.0)


def test_noncentralchinoise_dof2_matches_riciannoise() -> None:
    # dof=2 reduces to the Rician case.
    ncn = NoncentralChiNoise(std_range=(0.0, 1.0), dof=2, device="cpu", seed=0)
    rn = RicianNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE)
    noise = ncn.sample(x.shape)
    out_ncn = ncn.apply(x, noise)
    out_rn = rn.apply(x, noise)
    assert torch.allclose(out_ncn, out_rn)


def test_noncentralchinoise_higher_dof_increases_mean_magnitude() -> None:
    torch.manual_seed(0)
    std = 1.0
    ncn4 = NoncentralChiNoise(std_range=(std, std), dof=8, device="cpu", seed=0)
    ncn2 = NoncentralChiNoise(std_range=(std, std), dof=2, device="cpu", seed=0)
    x = torch.zeros(4, *SMALL_SHAPE)

    noise4 = ncn4.sample(x.shape)
    noise2 = ncn2.sample(x.shape)
    out4 = ncn4.apply(x, noise4)
    out2 = ncn2.apply(x, noise2)

    assert out4.mean() > out2.mean()


# --- SpeckleNoise ---


def test_specklenoise_init_stores_attrs() -> None:
    sn = SpeckleNoise(std_range=(0.1, 0.5), device="cpu", seed=0)
    assert sn.std_range == (0.1, 0.5)
    assert sn.device == torch.device("cpu")
    assert sn.seed == 0


def test_specklenoise_from_config_returns_instance() -> None:
    config = Config()
    sn = SpeckleNoise.from_config(config)
    assert isinstance(sn, SpeckleNoise)
    assert sn.std_range == (
        config.transform.speckle_std.min,
        config.transform.speckle_std.max,
    )


def test_specklenoise_sample_shape_batched() -> None:
    sn = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    noise = sn.sample(shape)
    assert noise.shape == torch.Size(shape)


def test_specklenoise_sample_shape_batch1() -> None:
    sn = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    noise = sn.sample(shape)
    assert noise.shape == torch.Size(shape)


def test_specklenoise_sample_reproducible() -> None:
    sn1 = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=7)
    sn2 = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=7)
    n1 = sn1.sample((2, *SMALL_SHAPE))
    n2 = sn2.sample((2, *SMALL_SHAPE))
    assert torch.equal(n1, n2)


def test_specklenoise_sample_different_seeds_differ() -> None:
    sn1 = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=1)
    sn2 = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=2)
    n1 = sn1.sample((2, *SMALL_SHAPE))
    n2 = sn2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(n1, n2)


def test_specklenoise_zero_std_is_noop() -> None:
    # std == 0 => noise == 0 everywhere, so output equals input unchanged.
    sn = SpeckleNoise(std_range=(0.0, 0.0), device="cpu", seed=0)
    x = torch.rand(2, *SMALL_SHAPE)
    noise = sn.sample(x.shape)
    out = sn.apply(x, noise)
    assert torch.allclose(out, x)


def test_specklenoise_apply_scales_with_data() -> None:
    sn = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.zeros(2, *SMALL_SHAPE)
    noise = torch.full((2, *SMALL_SHAPE), 0.5)
    out = sn.apply(x, noise)
    # data == 0 => multiplicative noise has no effect.
    assert torch.allclose(out, x)


def test_specklenoise_apply_preserves_shape() -> None:
    sn = SpeckleNoise(std_range=(0.0, 1.0), device="cpu", seed=0)
    x = torch.rand(3, *SMALL_SHAPE)
    noise = sn.sample(x.shape)
    out = sn.apply(x, noise)
    assert out.shape == x.shape
