import pytest
import torch

from synthbold.config import Config
from synthbold.models.background import BackgroundModel
from synthbold.transforms import RandomFlip

MU_RANGE = (10.0, 100.0)
STD_RANGE = (1.0, 2.0)
ZERO_STD = (0.0, 0.0)
SMALL_SHAPE = (4, 4, 4)


# --- BackgroundModel ---


def test_init_stores_ranges() -> None:
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=0
    )
    assert model.mu_range == MU_RANGE
    assert model.std_range == STD_RANGE


def test_init_default_pipeline_is_empty() -> None:
    model = BackgroundModel(mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu")
    assert model.pipeline.transforms == []


def test_init_pipeline_contains_given_transforms() -> None:
    flip = RandomFlip(flip_prob=1.0, device="cpu", seed=0)
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, transforms=[flip], device="cpu"
    )
    assert model.pipeline.transforms == [flip]


def test_forward_preserves_shape_and_dtype() -> None:
    data = torch.randint(0, 4, (2, *SMALL_SHAPE))
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=0
    )
    out = model(data)
    assert out.shape == data.shape
    assert out.dtype == torch.float32


def test_forward_requires_4d_input() -> None:
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=0
    )
    data = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    with pytest.raises(ValueError):
        model(data)


def test_forward_background_voxels_remain_zero() -> None:
    data = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    data[0, 0, 0, 0] = 1
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=0
    )
    out = model(data)
    assert torch.all(out[data == 0] == 0.0)
    assert out[0, 0, 0, 0] != 0.0


def test_forward_reproducible_with_same_seed() -> None:
    data = torch.randint(0, 4, (2, *SMALL_SHAPE))
    model1 = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=42
    )
    model2 = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=42
    )
    assert torch.equal(model1(data), model2(data))


def test_forward_different_seeds_produce_different_output() -> None:
    data = torch.randint(1, 4, (1, *SMALL_SHAPE))
    model1 = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=1
    )
    model2 = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=2
    )
    assert not torch.equal(model1(data), model2(data))


def test_forward_batch_elements_sampled_independently() -> None:
    # Zero std makes each label's output constant, so independent batch elements
    # must differ if they get independently sampled means.
    data = torch.full((2, *SMALL_SHAPE), 1, dtype=torch.int64)
    model = BackgroundModel(mu_range=MU_RANGE, std_range=ZERO_STD, device="cpu", seed=0)
    out = model(data)
    assert out[0].unique().item() != out[1].unique().item()


def test_forward_applies_pipeline_transforms() -> None:
    data = torch.randint(0, 4, (1, *SMALL_SHAPE))
    seed = 7

    no_pipeline = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=seed
    )
    out_unflipped = no_pipeline(data)

    flip = RandomFlip(flip_prob=1.0, device="cpu", seed=0)
    with_pipeline = BackgroundModel(
        mu_range=MU_RANGE,
        std_range=STD_RANGE,
        transforms=[flip],
        device="cpu",
        seed=seed,
    )
    out_flipped = with_pipeline(data)

    assert torch.equal(out_flipped, torch.flip(out_unflipped, dims=[1, 2, 3]))


def test_sample_labels_all_background_returns_zeros() -> None:
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=0
    )
    labels = torch.zeros(10, dtype=torch.int64)
    out = model._sample_labels(labels)
    assert out.shape == labels.shape
    assert out.dtype == torch.float32
    assert torch.all(out == 0.0)


def test_sample_labels_output_shape_matches_input() -> None:
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=STD_RANGE, device="cpu", seed=0
    )
    labels = torch.randint(0, 5, (50,))
    out = model._sample_labels(labels)
    assert out.shape == labels.shape


def test_sample_labels_constant_within_mu_range_when_std_zero() -> None:
    model = BackgroundModel(mu_range=MU_RANGE, std_range=ZERO_STD, device="cpu", seed=0)
    labels = torch.full((100,), 5, dtype=torch.int64)
    out = model._sample_labels(labels)
    assert out.unique().numel() == 1
    assert MU_RANGE[0] <= out.unique().item() <= MU_RANGE[1]


def test_sample_labels_different_labels_get_different_means() -> None:
    labels = torch.cat([torch.full((50,), 1), torch.full((50,), 2)]).to(torch.int64)
    model = BackgroundModel(mu_range=MU_RANGE, std_range=ZERO_STD, device="cpu", seed=0)
    out = model._sample_labels(labels)

    val_label1 = out[labels == 1].unique()
    val_label2 = out[labels == 2].unique()
    assert val_label1.numel() == 1
    assert val_label2.numel() == 1
    assert val_label1.item() != val_label2.item()


def test_sample_labels_no_background_label_is_nonzero() -> None:
    labels = torch.cat([torch.full((50,), 1), torch.full((50,), 2)]).to(torch.int64)
    model = BackgroundModel(mu_range=MU_RANGE, std_range=ZERO_STD, device="cpu", seed=0)
    out = model._sample_labels(labels)
    assert torch.all(out != 0.0)


def test_sample_labels_nonzero_std_produces_variation() -> None:
    labels = torch.full((1000,), 1, dtype=torch.int64)
    model = BackgroundModel(
        mu_range=MU_RANGE, std_range=(5.0, 5.0), device="cpu", seed=0
    )
    out = model._sample_labels(labels)
    assert out.unique().numel() > 1
    assert out.std().item() > 0


def test_from_config_returns_instance() -> None:
    config = Config()
    model = BackgroundModel.from_config(config)
    assert isinstance(model, BackgroundModel)


def test_from_config_ranges_match_config() -> None:
    config = Config()
    model = BackgroundModel.from_config(config)
    assert model.mu_range == (
        config.physio.background_mu.min,
        config.physio.background_mu.max,
    )
    assert model.std_range == (
        config.physio.background_std.min,
        config.physio.background_std.max,
    )


def test_from_config_device_and_seed() -> None:
    config = Config()
    model = BackgroundModel.from_config(config)
    assert model.device == torch.device(config.device)
    assert model.seed == config.seed
