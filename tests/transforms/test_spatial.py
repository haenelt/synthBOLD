import pytest
import torch

from synthbold.config import Config
from synthbold.transforms.spatial import (
    DeformedSphericalMask,
    GaussianSharpening,
    GaussianSmoothing,
    GibbsRinging,
    RandomFlip,
    SphericalMask,
)

SHAPE_3D = (8, 10, 12)
SHAPE_4D = (3, 8, 10, 12)
SMALL_SHAPE = (8, 8, 8)


# --- RandomFlip ---


def test_randomflip_init_stores_attrs() -> None:
    rf = RandomFlip(flip_prob=0.7, device="cpu", seed=0)
    assert rf.flip_prob == 0.7
    assert rf.device == torch.device("cpu")
    assert rf.seed == 0


def test_randomflip_from_config_returns_instance() -> None:
    config = Config()
    rf = RandomFlip.from_config(config)
    assert isinstance(rf, RandomFlip)
    assert rf.flip_prob == config.transform.flip_prob


def test_randomflip_sample_shape_3d() -> None:
    rf = RandomFlip(device="cpu", seed=0)
    flip = rf.sample(SHAPE_3D)
    assert flip.shape == torch.Size([3])


def test_randomflip_sample_shape_4d() -> None:
    rf = RandomFlip(device="cpu", seed=0)
    flip = rf.sample(SHAPE_4D)
    assert flip.shape == torch.Size([3])


def test_randomflip_sample_prob1_all_true() -> None:
    rf = RandomFlip(flip_prob=1.0, device="cpu", seed=0)
    flip = rf.sample(SHAPE_3D)
    assert flip.all()


def test_randomflip_sample_prob0_all_false() -> None:
    rf = RandomFlip(flip_prob=0.0, device="cpu", seed=0)
    flip = rf.sample(SHAPE_3D)
    assert not flip.any()


def test_randomflip_sample_reproducible() -> None:
    rf1 = RandomFlip(device="cpu", seed=42)
    rf2 = RandomFlip(device="cpu", seed=42)
    assert torch.equal(rf1.sample(SHAPE_3D), rf2.sample(SHAPE_3D))


def test_randomflip_sample_different_seeds_may_differ() -> None:
    # With prob=0.5 and seed=1 vs seed=2, samples are very unlikely to match.
    rf1 = RandomFlip(flip_prob=0.5, device="cpu", seed=1)
    rf2 = RandomFlip(flip_prob=0.5, device="cpu", seed=2)
    s1 = rf1.sample(SHAPE_3D)
    s2 = rf2.sample(SHAPE_3D)
    assert not torch.equal(s1, s2)


def test_randomflip_apply_no_flip_unchanged() -> None:
    x = torch.arange(8 * 10 * 12, dtype=torch.float32).reshape(*SHAPE_3D)
    flip = torch.zeros(3, dtype=torch.bool)
    out = RandomFlip.apply(x, flip)
    assert torch.equal(out, x)


def test_randomflip_apply_all_flip_correct() -> None:
    x = torch.arange(8 * 10 * 12, dtype=torch.float32).reshape(*SHAPE_3D)
    flip = torch.ones(3, dtype=torch.bool)
    out = RandomFlip.apply(x, flip)
    assert torch.equal(out, torch.flip(x, dims=[0, 1, 2]))


def test_randomflip_apply_single_axis() -> None:
    x = torch.arange(8 * 10 * 12, dtype=torch.float32).reshape(*SHAPE_3D)
    flip = torch.tensor([True, False, False])
    out = RandomFlip.apply(x, flip)
    assert torch.equal(out, torch.flip(x, dims=[0]))


def test_randomflip_apply_preserves_shape_3d() -> None:
    x = torch.rand(*SHAPE_3D)
    flip = torch.ones(3, dtype=torch.bool)
    out = RandomFlip.apply(x, flip)
    assert out.shape == x.shape


def test_randomflip_apply_preserves_shape_4d() -> None:
    x = torch.rand(*SHAPE_4D)
    flip = torch.ones(3, dtype=torch.bool)
    out = RandomFlip.apply(x, flip)
    assert out.shape == x.shape


def test_randomflip_apply_4d_does_not_flip_batch_dim() -> None:
    x = torch.rand(*SHAPE_4D)
    flip = torch.ones(3, dtype=torch.bool)
    out = RandomFlip.apply(x, flip)
    assert torch.equal(out, torch.flip(x, dims=[1, 2, 3]))


def test_randomflip_call_3d_preserves_shape() -> None:
    rf = RandomFlip(device="cpu", seed=0)
    x = torch.rand(*SHAPE_3D)
    out = rf(x)
    assert out.shape == x.shape


def test_randomflip_call_4d_preserves_shape() -> None:
    rf = RandomFlip(device="cpu", seed=0)
    x = torch.rand(*SHAPE_4D)
    out = rf(x)
    assert out.shape == x.shape


def test_randomflip_call_wrong_dim_raises() -> None:
    rf = RandomFlip(device="cpu", seed=0)
    x = torch.rand(4, 4)
    with pytest.raises(ValueError):
        rf(x)


# --- GaussianSmoothing ---


def test_gaussiansmoothing_init_stores_attrs() -> None:
    gs = GaussianSmoothing(
        sigma_range=(0.5, 1.0), kernel_size=3, isotropic=True, device="cpu", seed=0
    )
    assert gs.sigma_range == (0.5, 1.0)
    assert gs.kernel_size == 3
    assert gs.isotropic is True
    assert gs.device == torch.device("cpu")
    assert gs.seed == 0


def test_gaussiansmoothing_init_even_kernel_raises() -> None:
    with pytest.raises(ValueError):
        GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=4, device="cpu")


def test_gaussiansmoothing_from_config_returns_instance() -> None:
    config = Config()
    gs = GaussianSmoothing.from_config(config)
    assert isinstance(gs, GaussianSmoothing)
    assert gs.sigma_range == (
        config.transform.gsmooth_sigma.min,
        config.transform.gsmooth_sigma.max,
    )
    assert gs.kernel_size == config.transform.gsmooth_kernel_size
    assert gs.isotropic == config.transform.gsmooth_isotropic


def test_gaussiansmoothing_sample_shape_batched() -> None:
    gs = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    kernel = gs.sample(shape)
    assert kernel.shape == torch.Size([3, 1, 3, 3, 3])


def test_gaussiansmoothing_sample_shape_batch1() -> None:
    gs = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=5, device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    kernel = gs.sample(shape)
    assert kernel.shape == torch.Size([1, 1, 5, 5, 5])


def test_gaussiansmoothing_sample_kernel_sums_to_one() -> None:
    gs = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=0)
    kernel = gs.sample((4, *SMALL_SHAPE))
    kernel_sums = kernel.sum(dim=(1, 2, 3, 4))
    assert torch.allclose(kernel_sums, torch.ones(4), atol=1e-5)


def test_gaussiansmoothing_sample_reproducible() -> None:
    gs1 = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=7)
    gs2 = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=7)
    k1 = gs1.sample((2, *SMALL_SHAPE))
    k2 = gs2.sample((2, *SMALL_SHAPE))
    assert torch.equal(k1, k2)


def test_gaussiansmoothing_sample_different_seeds_differ() -> None:
    gs1 = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=1)
    gs2 = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=2)
    k1 = gs1.sample((2, *SMALL_SHAPE))
    k2 = gs2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(k1, k2)


def test_gaussiansmoothing_sample_anisotropic_shape() -> None:
    gs = GaussianSmoothing(
        sigma_range=(0.5, 1.0), kernel_size=3, isotropic=False, device="cpu", seed=0
    )
    kernel = gs.sample((2, *SMALL_SHAPE))
    assert kernel.shape == torch.Size([2, 1, 3, 3, 3])


def test_gaussiansmoothing_choose_sigma_invalid_nonpositive_raises() -> None:
    gs = GaussianSmoothing(sigma_range=(0.0, 1.0), device="cpu")
    with pytest.raises(ValueError):
        gs.sample((2, *SMALL_SHAPE))


def test_gaussiansmoothing_choose_sigma_min_gt_max_raises() -> None:
    gs = GaussianSmoothing(sigma_range=(2.0, 1.0), device="cpu")
    with pytest.raises(ValueError):
        gs.sample((2, *SMALL_SHAPE))


def test_gaussiansmoothing_apply_preserves_shape_4d() -> None:
    gs = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=0)
    x = torch.rand(3, *SMALL_SHAPE)
    kernel = gs.sample(x.shape)
    out = GaussianSmoothing.apply(x, kernel)
    assert out.shape == x.shape


def test_gaussiansmoothing_apply_constant_volume_unchanged() -> None:
    # Smoothing a constant volume with a normalized kernel leaves values unchanged.
    gs = GaussianSmoothing(sigma_range=(0.5, 1.0), kernel_size=3, device="cpu", seed=0)
    x = torch.ones(2, *SMALL_SHAPE)
    kernel = gs.sample(x.shape)
    out = GaussianSmoothing.apply(x, kernel)
    # Interior voxels should stay at 1.0; boundary may differ due to padding.
    assert torch.allclose(out[:, 1:-1, 1:-1, 1:-1], torch.ones(2, 6, 6, 6), atol=1e-5)


# --- GaussianSharpening ---


def test_gaussiansharpening_init_stores_attrs() -> None:
    gs = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=0,
    )
    assert gs.sigma_range == (0.5, 1.0)
    assert gs.amount_range == (0.5, 2.0)
    assert gs.kernel_size == 3
    assert gs.device == torch.device("cpu")
    assert gs.seed == 0


def test_gaussiansharpening_init_even_kernel_raises() -> None:
    with pytest.raises(ValueError):
        GaussianSharpening(
            sigma_range=(0.5, 1.0), amount_range=(0.5, 2.0), kernel_size=4, device="cpu"
        )


def test_gaussiansharpening_from_config_returns_instance() -> None:
    config = Config()
    gs = GaussianSharpening.from_config(config)
    assert isinstance(gs, GaussianSharpening)
    assert gs.sigma_range == (
        config.transform.sharpen_sigma.min,
        config.transform.sharpen_sigma.max,
    )
    assert gs.amount_range == (
        config.transform.sharpen_amount.min,
        config.transform.sharpen_amount.max,
    )
    assert gs.kernel_size == config.transform.sharpen_kernel_size


def test_gaussiansharpening_sample_shape_batched() -> None:
    gs = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=0,
    )
    kernel = gs.sample((3, *SMALL_SHAPE))
    assert kernel.shape == torch.Size([3, 1, 3, 3, 3])


def test_gaussiansharpening_sample_reproducible() -> None:
    gs1 = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=7,
    )
    gs2 = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=7,
    )
    k1 = gs1.sample((2, *SMALL_SHAPE))
    k2 = gs2.sample((2, *SMALL_SHAPE))
    assert torch.equal(k1, k2)


def test_gaussiansharpening_sample_different_seeds_differ() -> None:
    gs1 = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=1,
    )
    gs2 = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=2,
    )
    k1 = gs1.sample((2, *SMALL_SHAPE))
    k2 = gs2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(k1, k2)


def test_gaussiansharpening_choose_sigma_invalid_nonpositive_raises() -> None:
    gs = GaussianSharpening(
        sigma_range=(0.0, 1.0), amount_range=(0.5, 2.0), device="cpu"
    )
    with pytest.raises(ValueError):
        gs.sample((2, *SMALL_SHAPE))


def test_gaussiansharpening_choose_sigma_min_gt_max_raises() -> None:
    gs = GaussianSharpening(
        sigma_range=(2.0, 1.0), amount_range=(0.5, 2.0), device="cpu"
    )
    with pytest.raises(ValueError):
        gs.sample((2, *SMALL_SHAPE))


def test_gaussiansharpening_choose_amount_negative_raises() -> None:
    gs = GaussianSharpening(
        sigma_range=(0.5, 1.0), amount_range=(-0.5, 1.0), device="cpu"
    )
    with pytest.raises(ValueError):
        gs.sample((2, *SMALL_SHAPE))


def test_gaussiansharpening_apply_preserves_shape_4d() -> None:
    gs = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=0,
    )
    x = torch.rand(3, *SMALL_SHAPE)
    kernel = gs.sample(x.shape)
    out = GaussianSharpening.apply(x, kernel)
    assert out.shape == x.shape


def test_gaussiansharpening_apply_constant_volume_unchanged() -> None:
    # Unsharp masking of a constant volume leaves values unchanged, since the blurred
    # version equals the input everywhere (away from boundary effects).
    gs = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=0,
    )
    x = torch.ones(2, *SMALL_SHAPE)
    kernel = gs.sample(x.shape)
    out = GaussianSharpening.apply(x, kernel)
    assert torch.allclose(out[:, 1:-1, 1:-1, 1:-1], torch.ones(2, 6, 6, 6), atol=1e-5)


def test_gaussiansharpening_apply_zero_amount_is_noop() -> None:
    # With amount=0 the unsharp-masking kernel reduces to an identity kernel, leaving
    # the input unchanged.
    gs = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.0, 0.0),
        kernel_size=3,
        device="cpu",
        seed=0,
    )
    x = torch.rand(2, *SMALL_SHAPE)
    kernel = gs.sample(x.shape)
    out = GaussianSharpening.apply(x, kernel)
    assert torch.allclose(out, x, atol=1e-5)


def test_gaussiansharpening_apply_sharpens_nonconstant_volume() -> None:
    gs = GaussianSharpening(
        sigma_range=(0.5, 1.0),
        amount_range=(0.5, 2.0),
        kernel_size=3,
        device="cpu",
        seed=0,
    )
    x = torch.rand(2, *SMALL_SHAPE)
    kernel = gs.sample(x.shape)
    out = GaussianSharpening.apply(x, kernel)
    assert not torch.allclose(out, x)


# --- GibbsRinging ---


def test_gibbsringing_init_stores_attrs() -> None:
    gr = GibbsRinging(cutoff_range=(0.5, 0.9), device="cpu", seed=0)
    assert gr.cutoff_range == (0.5, 0.9)
    assert gr.device == torch.device("cpu")
    assert gr.seed == 0


def test_gibbsringing_init_cutoff_too_low_raises() -> None:
    with pytest.raises(ValueError):
        GibbsRinging(cutoff_range=(0.0, 0.5), device="cpu")


def test_gibbsringing_init_cutoff_too_high_raises() -> None:
    with pytest.raises(ValueError):
        GibbsRinging(cutoff_range=(0.5, 1.5), device="cpu")


def test_gibbsringing_from_config_returns_instance() -> None:
    config = Config()
    gr = GibbsRinging.from_config(config)
    assert isinstance(gr, GibbsRinging)
    assert gr.cutoff_range == (
        config.transform.gibbs_cutoff.min,
        config.transform.gibbs_cutoff.max,
    )


def test_gibbsringing_sample_shape_batched() -> None:
    gr = GibbsRinging(cutoff_range=(0.5, 1.0), device="cpu", seed=0)
    cutoff = gr.sample((3, *SMALL_SHAPE))
    assert cutoff.shape == torch.Size((3, 1, 1, 1))


def test_gibbsringing_sample_within_range() -> None:
    gr = GibbsRinging(cutoff_range=(0.5, 0.9), device="cpu", seed=0)
    cutoff = gr.sample((5, *SMALL_SHAPE))
    assert torch.all(cutoff >= 0.5)
    assert torch.all(cutoff <= 0.9)


def test_gibbsringing_sample_reproducible() -> None:
    gr1 = GibbsRinging(cutoff_range=(0.5, 1.0), device="cpu", seed=7)
    gr2 = GibbsRinging(cutoff_range=(0.5, 1.0), device="cpu", seed=7)
    c1 = gr1.sample((4, *SMALL_SHAPE))
    c2 = gr2.sample((4, *SMALL_SHAPE))
    assert torch.equal(c1, c2)


def test_gibbsringing_sample_different_seeds_differ() -> None:
    gr1 = GibbsRinging(cutoff_range=(0.5, 1.0), device="cpu", seed=1)
    gr2 = GibbsRinging(cutoff_range=(0.5, 1.0), device="cpu", seed=2)
    c1 = gr1.sample((4, *SMALL_SHAPE))
    c2 = gr2.sample((4, *SMALL_SHAPE))
    assert not torch.equal(c1, c2)


def test_gibbsringing_apply_preserves_shape() -> None:
    gr = GibbsRinging(cutoff_range=(0.5, 1.0), device="cpu", seed=0)
    x = torch.rand(3, *SMALL_SHAPE)
    cutoff = gr.sample(x.shape)
    out = GibbsRinging.apply(x, cutoff)
    assert out.shape == x.shape


def test_gibbsringing_apply_cutoff_one_is_noop() -> None:
    # cutoff == 1.0 retains the full spectrum, so the volume is unchanged.
    x = torch.rand(2, *SMALL_SHAPE)
    cutoff = torch.ones(2, 1, 1, 1)
    out = GibbsRinging.apply(x, cutoff)
    assert torch.allclose(out, x, atol=1e-5)


def test_gibbsringing_apply_constant_volume_unchanged() -> None:
    # A constant volume only has a DC component, which any cutoff > 0 retains.
    x = torch.full((2, *SMALL_SHAPE), 3.0)
    cutoff = torch.full((2, 1, 1, 1), 0.3)
    out = GibbsRinging.apply(x, cutoff)
    assert torch.allclose(out, x, atol=1e-5)


def test_gibbsringing_apply_truncation_changes_volume() -> None:
    torch.manual_seed(0)
    x = torch.rand(1, *SMALL_SHAPE)
    cutoff = torch.full((1, 1, 1, 1), 0.3)
    out = GibbsRinging.apply(x, cutoff)
    assert not torch.allclose(out, x)


# --- SphericalMask ---


def test_sphericalmask_init_stores_attrs() -> None:
    sm = SphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=0)
    assert sm.radius_range == (2.0, 4.0)
    assert sm.sphere_prob == 1.0
    assert sm.device == torch.device("cpu")
    assert sm.seed == 0


def test_sphericalmask_init_stores_sphere_prob() -> None:
    sm = SphericalMask(radius_range=(2.0, 4.0), sphere_prob=0.5, device="cpu", seed=0)
    assert sm.sphere_prob == 0.5


def test_sphericalmask_from_config_returns_instance() -> None:
    config = Config()
    sm = SphericalMask.from_config(config)
    assert isinstance(sm, SphericalMask)
    assert sm.radius_range == (
        config.transform.sphere_radius.min,
        config.transform.sphere_radius.max,
    )
    assert sm.sphere_prob == config.transform.sphere_prob


def test_sphericalmask_sample_shape_batched() -> None:
    sm = SphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=0)
    mask = sm.sample((3, *SMALL_SHAPE))
    assert mask.shape == torch.Size([3, *SMALL_SHAPE])


def test_sphericalmask_sample_dtype_bool() -> None:
    sm = SphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=0)
    mask = sm.sample((2, *SMALL_SHAPE))
    assert mask.dtype == torch.bool


def test_sphericalmask_sample_reproducible() -> None:
    sm1 = SphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=7)
    sm2 = SphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=7)
    m1 = sm1.sample((2, *SMALL_SHAPE))
    m2 = sm2.sample((2, *SMALL_SHAPE))
    assert torch.equal(m1, m2)


def test_sphericalmask_sample_different_seeds_may_differ() -> None:
    sm1 = SphericalMask(radius_range=(1.0, 2.0), device="cpu", seed=1)
    sm2 = SphericalMask(radius_range=(1.0, 2.0), device="cpu", seed=2)
    m1 = sm1.sample((2, *SMALL_SHAPE))
    m2 = sm2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(m1, m2)


def test_sphericalmask_sample_large_radius_all_true() -> None:
    # A radius far larger than the volume diagonal covers every voxel, regardless of
    # the randomly sampled center.
    sm = SphericalMask(radius_range=(1000.0, 1000.0), device="cpu", seed=0)
    mask = sm.sample((2, *SMALL_SHAPE))
    assert mask.all()


def test_sphericalmask_sample_zero_radius_only_center_voxel() -> None:
    # With radius_range=(0.0, 0.0), the sampled radius is exactly 0, so only the
    # center voxel itself (squared distance 0) falls within the mask.
    sm = SphericalMask(radius_range=(0.0, 0.0), device="cpu", seed=0)
    mask = sm.sample((4, *SMALL_SHAPE))
    counts = mask.flatten(1).sum(dim=1)
    assert torch.equal(counts, torch.ones(4, dtype=counts.dtype))


def test_sphericalmask_sample_sphere_prob_zero_never_masks() -> None:
    # With sphere_prob=0, no volume is masked, regardless of radius_range.
    sm = SphericalMask(radius_range=(0.0, 0.0), sphere_prob=0.0, device="cpu", seed=0)
    mask = sm.sample((4, *SMALL_SHAPE))
    assert mask.all()


def test_sphericalmask_sample_sphere_prob_partial_skips_some_volumes() -> None:
    # With sphere_prob=0.5, some volumes keep the small spherical mask while others are
    # left fully unmasked.
    sm = SphericalMask(radius_range=(0.0, 0.0), sphere_prob=0.5, device="cpu", seed=0)
    mask = sm.sample((8, *SMALL_SHAPE))
    counts = mask.flatten(1).sum(dim=1)
    n_voxels = SMALL_SHAPE[0] * SMALL_SHAPE[1] * SMALL_SHAPE[2]
    assert (counts == 1).any()
    assert (counts == n_voxels).any()


def test_sphericalmask_apply_zeroes_outside_mask() -> None:
    x = torch.ones(2, *SMALL_SHAPE)
    mask = torch.zeros(2, *SMALL_SHAPE, dtype=torch.bool)
    mask[:, 0, 0, 0] = True
    out = SphericalMask.apply(x, mask)
    assert torch.equal(out[:, 0, 0, 0], torch.ones(2))
    assert out.sum() == 2


def test_sphericalmask_apply_preserves_shape() -> None:
    sm = SphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=0)
    x = torch.rand(3, *SMALL_SHAPE)
    mask = sm.sample(x.shape)
    out = SphericalMask.apply(x, mask)
    assert out.shape == x.shape


# --- DeformedSphericalMask ---


def test_deformedsphericalmask_init_stores_attrs() -> None:
    dm = DeformedSphericalMask(
        radius_range=(2.0, 4.0),
        deform_amplitude=0.4,
        deform_shape=(3, 3, 3),
        sphere_prob=0.5,
        device="cpu",
        seed=0,
    )
    assert dm.radius_range == (2.0, 4.0)
    assert dm.deform_amplitude == 0.4
    assert dm.deform_shape == (3, 3, 3)
    assert dm.sphere_prob == 0.5
    assert dm.device == torch.device("cpu")
    assert dm.seed == 0


def test_deformedsphericalmask_init_negative_amplitude_raises() -> None:
    with pytest.raises(ValueError):
        DeformedSphericalMask(
            radius_range=(2.0, 4.0), deform_amplitude=-0.1, device="cpu"
        )


def test_deformedsphericalmask_from_config_returns_instance() -> None:
    config = Config()
    dm = DeformedSphericalMask.from_config(config)
    assert isinstance(dm, DeformedSphericalMask)
    assert dm.radius_range == (
        config.transform.sphere_radius.min,
        config.transform.sphere_radius.max,
    )
    assert dm.deform_amplitude == config.transform.sphere_deform_amplitude
    assert dm.deform_shape == config.transform.sphere_deform_shape
    assert dm.sphere_prob == config.transform.sphere_prob


def test_deformedsphericalmask_sample_shape_batched() -> None:
    dm = DeformedSphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=0)
    mask = dm.sample((3, *SMALL_SHAPE))
    assert mask.shape == torch.Size([3, *SMALL_SHAPE])


def test_deformedsphericalmask_sample_dtype_bool() -> None:
    dm = DeformedSphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=0)
    mask = dm.sample((2, *SMALL_SHAPE))
    assert mask.dtype == torch.bool


def test_deformedsphericalmask_sample_reproducible() -> None:
    dm1 = DeformedSphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=7)
    dm2 = DeformedSphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=7)
    m1 = dm1.sample((2, *SMALL_SHAPE))
    m2 = dm2.sample((2, *SMALL_SHAPE))
    assert torch.equal(m1, m2)


def test_deformedsphericalmask_sample_different_seeds_may_differ() -> None:
    dm1 = DeformedSphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=1)
    dm2 = DeformedSphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=2)
    m1 = dm1.sample((2, *SMALL_SHAPE))
    m2 = dm2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(m1, m2)


def test_deformedsphericalmask_sample_zero_amplitude_matches_sphericalmask() -> None:
    # With deform_amplitude=0, the radius perturbation is a no-op, so the mask matches
    # SphericalMask given the same seed (both draw centers/radii in the same order).
    sm = SphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=3)
    dm = DeformedSphericalMask(
        radius_range=(2.0, 4.0), deform_amplitude=0.0, device="cpu", seed=3
    )
    m_sphere = sm.sample((2, *SMALL_SHAPE))
    m_deformed = dm.sample((2, *SMALL_SHAPE))
    assert torch.equal(m_sphere, m_deformed)


def test_deformedsphericalmask_sample_deformation_changes_mask() -> None:
    # With a non-zero deform_amplitude, the smooth noise field perturbs the radius
    # field, so the mask differs from the undeformed sphere of the same radius.
    sm = SphericalMask(radius_range=(3.0, 3.0), device="cpu", seed=3)
    dm = DeformedSphericalMask(
        radius_range=(3.0, 3.0), deform_amplitude=0.5, device="cpu", seed=3
    )
    m_sphere = sm.sample((2, *SMALL_SHAPE))
    m_deformed = dm.sample((2, *SMALL_SHAPE))
    assert not torch.equal(m_sphere, m_deformed)


def test_deformedsphericalmask_sample_large_radius_all_true() -> None:
    # A radius far larger than the volume diagonal covers every voxel, regardless of
    # the deformation noise.
    dm = DeformedSphericalMask(radius_range=(1000.0, 1000.0), device="cpu", seed=0)
    mask = dm.sample((2, *SMALL_SHAPE))
    assert mask.all()


def test_deformedsphericalmask_sample_sphere_prob_zero_never_masks() -> None:
    dm = DeformedSphericalMask(
        radius_range=(0.0, 0.0), sphere_prob=0.0, device="cpu", seed=0
    )
    mask = dm.sample((4, *SMALL_SHAPE))
    assert mask.all()


def test_deformedsphericalmask_apply_preserves_shape() -> None:
    dm = DeformedSphericalMask(radius_range=(2.0, 4.0), device="cpu", seed=0)
    x = torch.rand(3, *SMALL_SHAPE)
    mask = dm.sample(x.shape)
    out = DeformedSphericalMask.apply(x, mask)
    assert out.shape == x.shape
