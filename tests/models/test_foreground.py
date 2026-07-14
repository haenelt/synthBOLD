import torch

from synthbold.config import Config
from synthbold.models.foreground import DistractorModel, ForegroundModel
from synthbold.transforms import CaliberDeformation, ElasticDeformation, RandomFlip

SMALL_SHAPE = (4, 4, 4)


# --- ForegroundModel ---


def test_init_default_pipeline_is_empty() -> None:
    model = ForegroundModel(device="cpu")
    assert model.pipeline.transforms == []


def test_init_pipeline_contains_given_transforms() -> None:
    flip = RandomFlip(flip_prob=1.0, device="cpu", seed=0)
    model = ForegroundModel(transforms=[flip], device="cpu")
    assert model.pipeline.transforms == [flip]


def test_forward_preserves_shape_and_dtype_with_empty_pipeline() -> None:
    data = torch.randint(1, 4, (2, *SMALL_SHAPE))
    model = ForegroundModel(device="cpu", seed=0)
    out = model(data)
    assert out.shape == data.shape
    assert out.dtype == data.dtype


def test_forward_empty_pipeline_is_identity() -> None:
    data = torch.randint(1, 4, (2, *SMALL_SHAPE))
    model = ForegroundModel(device="cpu", seed=0)
    out = model(data)
    assert torch.equal(out, data)


def test_forward_applies_pipeline_transforms() -> None:
    data = torch.randint(1, 4, (1, *SMALL_SHAPE))
    flip = RandomFlip(flip_prob=1.0, device="cpu", seed=0)
    model = ForegroundModel(transforms=[flip], device="cpu", seed=0)
    out = model(data)
    assert torch.equal(out, torch.flip(data, dims=[1, 2, 3]))


def test_from_config_returns_instance() -> None:
    config = Config()
    model = ForegroundModel.from_config(config)
    assert isinstance(model, ForegroundModel)


def test_from_config_device_and_seed() -> None:
    config = Config()
    model = ForegroundModel.from_config(config)
    assert model.device == torch.device(config.device)
    assert model.seed == config.seed


def test_from_config_pipeline_contains_expected_transforms() -> None:
    config = Config()
    model = ForegroundModel.from_config(config)
    transforms = model.pipeline.transforms
    assert len(transforms) == 3
    assert isinstance(transforms[0], RandomFlip)
    assert isinstance(transforms[1], ElasticDeformation)
    assert isinstance(transforms[2], CaliberDeformation)


# --- DistractorModel ---


def test_distractor_is_foreground_model_subclass() -> None:
    assert issubclass(DistractorModel, ForegroundModel)


def test_distractor_forward_preserves_shape_and_dtype_with_empty_pipeline() -> None:
    data = torch.randint(1, 4, (2, *SMALL_SHAPE))
    model = DistractorModel(device="cpu", seed=0)
    out = model(data)
    assert out.shape == data.shape
    assert out.dtype == data.dtype


def test_distractor_from_config_returns_instance() -> None:
    config = Config()
    model = DistractorModel.from_config(config)
    assert isinstance(model, DistractorModel)


def test_distractor_from_config_device_and_seed() -> None:
    config = Config()
    model = DistractorModel.from_config(config)
    assert model.device == torch.device(config.device)
    assert model.seed == config.seed


def test_distractor_from_config_pipeline_contains_only_randomflip() -> None:
    config = Config()
    model = DistractorModel.from_config(config)
    transforms = model.pipeline.transforms
    assert len(transforms) == 1
    flip = transforms[0]
    assert isinstance(flip, RandomFlip)
    assert flip.flip_prob == config.transform.flip_prob
