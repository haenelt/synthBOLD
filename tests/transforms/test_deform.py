import torch

from synthbold.config import Config
from synthbold.transforms.deform import CaliberDeformation, ElasticDeformation

SMALL_SHAPE = (8, 8, 8)


# --- ElasticDeformation ---


def test_elasticdeformation_init_stores_attrs() -> None:
    ed = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=0)
    assert ed.sigma == 2.0
    assert ed.max_disp == 5.0
    assert ed.device == torch.device("cpu")
    assert ed.seed == 0


def test_elasticdeformation_from_config_returns_instance() -> None:
    config = Config()
    ed = ElasticDeformation.from_config(config)
    assert isinstance(ed, ElasticDeformation)
    assert ed.sigma == config.transform.elastic_sigma
    assert ed.max_disp == config.transform.elastic_max_disp


def test_elasticdeformation_sample_shape_batched() -> None:
    ed = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    disp = ed.sample(shape)
    assert disp.shape == torch.Size([3, *SMALL_SHAPE, 3])


def test_elasticdeformation_sample_shape_batch1() -> None:
    ed = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    disp = ed.sample(shape)
    assert disp.shape == torch.Size([1, *SMALL_SHAPE, 3])


def test_elasticdeformation_sample_reproducible() -> None:
    ed1 = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=7)
    ed2 = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=7)
    d1 = ed1.sample((2, *SMALL_SHAPE))
    d2 = ed2.sample((2, *SMALL_SHAPE))
    assert torch.equal(d1, d2)


def test_elasticdeformation_sample_different_seeds_differ() -> None:
    ed1 = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=1)
    ed2 = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=2)
    d1 = ed1.sample((2, *SMALL_SHAPE))
    d2 = ed2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(d1, d2)


def test_elasticdeformation_sample_max_equals_max_disp() -> None:
    # The field is normalized so its maximum value equals max_disp exactly.
    ed = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=0)
    disp = ed.sample((2, *SMALL_SHAPE))
    assert disp.max().item() == 5.0


def test_elasticdeformation_sample_isotropic_components_equal() -> None:
    # The same displacement field is applied along x, y, and z.
    ed = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=0)
    disp = ed.sample((2, *SMALL_SHAPE))
    assert torch.equal(disp[..., 0], disp[..., 1])
    assert torch.equal(disp[..., 1], disp[..., 2])


def test_elasticdeformation_apply_zero_displacement_is_identity() -> None:
    data = torch.rand(1, *SMALL_SHAPE)
    disp = torch.zeros(1, *SMALL_SHAPE, 3)
    out = ElasticDeformation.apply(data, disp)
    assert torch.equal(out, data)


def test_elasticdeformation_apply_preserves_shape() -> None:
    ed = ElasticDeformation(sigma=2.0, max_disp=5.0, device="cpu", seed=0)
    data = torch.rand(2, *SMALL_SHAPE)
    disp = ed.sample(data.shape)
    out = ElasticDeformation.apply(data, disp)
    assert out.shape == data.shape


def test_elasticdeformation_apply_uses_nearest_interpolation() -> None:
    # With nearest interpolation, every output value must come from the input
    # (no blending of values, even for non-integer displacements).
    torch.manual_seed(0)
    data = torch.rand(1, *SMALL_SHAPE)
    disp = torch.rand(1, *SMALL_SHAPE, 3) * 3 - 1.5
    out = ElasticDeformation.apply(data, disp)
    assert set(out.flatten().tolist()) <= set(data.flatten().tolist())


def test_elasticdeformation_apply_preserves_integer_dtype() -> None:
    data = torch.randint(0, 5, (1, *SMALL_SHAPE), dtype=torch.int64)
    disp = torch.rand(1, *SMALL_SHAPE, 3) * 3 - 1.5
    out = ElasticDeformation.apply(data, disp)
    assert out.dtype == data.dtype


# --- CaliberDeformation ---


def test_caliberdeformation_init_stores_attrs() -> None:
    cd = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=0)
    assert cd.sigma == 2.0
    assert cd.alpha == 4.0
    assert cd.device == torch.device("cpu")
    assert cd.seed == 0


def test_caliberdeformation_from_config_returns_instance() -> None:
    config = Config()
    cd = CaliberDeformation.from_config(config)
    assert isinstance(cd, CaliberDeformation)
    assert cd.sigma == config.transform.caliber_sigma
    assert cd.alpha == config.transform.caliber_alpha


def test_caliberdeformation_sample_shape_batched() -> None:
    cd = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=0)
    shape = (3, *SMALL_SHAPE)
    radius_field = cd.sample(shape)
    assert radius_field.shape == torch.Size(shape)


def test_caliberdeformation_sample_shape_batch1() -> None:
    cd = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=0)
    shape = (1, *SMALL_SHAPE)
    radius_field = cd.sample(shape)
    assert radius_field.shape == torch.Size(shape)


def test_caliberdeformation_sample_reproducible() -> None:
    cd1 = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=7)
    cd2 = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=7)
    r1 = cd1.sample((2, *SMALL_SHAPE))
    r2 = cd2.sample((2, *SMALL_SHAPE))
    assert torch.equal(r1, r2)


def test_caliberdeformation_sample_different_seeds_differ() -> None:
    cd1 = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=1)
    cd2 = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=2)
    r1 = cd1.sample((2, *SMALL_SHAPE))
    r2 = cd2.sample((2, *SMALL_SHAPE))
    assert not torch.equal(r1, r2)


def test_caliberdeformation_sample_nonnegative() -> None:
    cd = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=0)
    radius_field = cd.sample((2, *SMALL_SHAPE))
    assert (radius_field >= 0).all()


def test_caliberdeformation_sample_alpha_zero_gives_zero_radius() -> None:
    cd = CaliberDeformation(sigma=2.0, alpha=0.0, device="cpu", seed=0)
    radius_field = cd.sample((2, *SMALL_SHAPE))
    assert torch.equal(radius_field, torch.zeros_like(radius_field))


def test_caliberdeformation_apply_zero_radius_is_identity() -> None:
    data = torch.zeros(1, 5, 5, 5)
    data[0, 2, 2, 2] = 1.0
    radius_field = torch.zeros(1, 5, 5, 5)
    out = CaliberDeformation.apply(data, radius_field)
    assert torch.equal(out, data)


def test_caliberdeformation_apply_radius_one_dilates_to_face_neighbors() -> None:
    # A unit radius field should dilate a single labeled voxel to its 6 face-adjacent
    # neighbors (Euclidean distance 1), but not its edge/corner neighbors (> 1).
    data = torch.zeros(1, 5, 5, 5)
    data[0, 2, 2, 2] = 1.0
    radius_field = torch.ones(1, 5, 5, 5)
    out = CaliberDeformation.apply(data, radius_field)

    expected = torch.zeros(1, 5, 5, 5)
    expected[0, 2, 2, 2] = 1.0
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for dx, dy, dz in neighbors:
        expected[0, 2 + dx, 2 + dy, 2 + dz] = 1.0

    assert torch.equal(out, expected)


def test_caliberdeformation_apply_all_background_returns_zeros() -> None:
    data = torch.zeros(1, *SMALL_SHAPE)
    radius_field = torch.ones(1, *SMALL_SHAPE)
    out = CaliberDeformation.apply(data, radius_field)
    assert torch.equal(out, data)


def test_caliberdeformation_apply_preserves_shape() -> None:
    cd = CaliberDeformation(sigma=2.0, alpha=4.0, device="cpu", seed=0)
    data = torch.randint(0, 3, (2, *SMALL_SHAPE), dtype=torch.float32)
    radius_field = cd.sample(data.shape)
    out = CaliberDeformation.apply(data, radius_field)
    assert out.shape == data.shape
