from pathlib import Path
from typing import Self

import numpy as np
import pytest
import torch

from synthbold.base import (
    BaseGeometry,
    Model,
    ObjectGeometry,
    Pipeline,
    RandomGeneratorMixin,
    Transform,
)
from synthbold.config import Config

SMALL_SHAPE = (8, 8, 8)
SMALL_FOV = (2.0, 2.0, 2.0)


class DummyClass(RandomGeneratorMixin):
    """Dummy class to test the mixin."""

    pass


class DummyBaseGeometry(BaseGeometry):
    """Concrete BaseGeometry for tests: fills volume with a constant value."""

    def forward(self) -> torch.Tensor:
        return torch.ones(self.shape, device=self.device, dtype=self.dtype)


class DummyObjectGeometry(ObjectGeometry):
    """Concrete ObjectGeometry for tests: always masks the first half of X."""

    def _mask_object(self) -> torch.Tensor:
        mask = torch.zeros(self.shape, device=self.device, dtype=torch.bool)
        mask[: self.shape[0] // 2, :, :] = True
        return mask


class DummyModel(Model):
    """Dummy model for testing: doubles the input tensor."""

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2.0


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


# --- RandomGeneratorMixin ---


def test_random_generator_mixin_unseeded() -> None:
    """Test the mixin without a seed."""
    obj1 = DummyClass(seed=None, device="cpu")
    obj2 = DummyClass(seed=None, device="cpu")

    assert obj1.seed is None
    assert obj1.generator is not None
    assert obj1.np_generator is not None

    # Unseeded generators should generally produce different results
    val1 = torch.rand(1, generator=obj1.generator).item()
    val2 = torch.rand(1, generator=obj2.generator).item()
    assert val1 != val2


def test_random_generator_mixin_seeded() -> None:
    """Test the mixin with a seed for reproducibility."""
    seed = 42
    obj1 = DummyClass(seed=seed, device="cpu")
    obj2 = DummyClass(seed=seed, device="cpu")

    assert obj1.seed == seed
    assert obj2.seed == seed

    # Test PyTorch reproducibility
    val1_torch = torch.rand(10, generator=obj1.generator)
    val2_torch = torch.rand(10, generator=obj2.generator)
    assert torch.equal(val1_torch, val2_torch)

    # Test NumPy reproducibility
    val1_np = obj1.np_generator.random(10)
    val2_np = obj2.np_generator.random(10)
    np.testing.assert_array_equal(val1_np, val2_np)


def test_random_generator_mixin_different_seeds() -> None:
    """Test that different seeds produce different results."""
    obj1 = DummyClass(seed=42, device="cpu")
    obj2 = DummyClass(seed=43, device="cpu")

    val1_torch = torch.rand(10, generator=obj1.generator)
    val2_torch = torch.rand(10, generator=obj2.generator)
    assert not torch.equal(val1_torch, val2_torch)

    val1_np = obj1.np_generator.random(10)
    val2_np = obj2.np_generator.random(10)

    # NumPy arrays should not be equal with different seeds
    assert not np.array_equal(val1_np, val2_np)


# --- BaseGeometry ---


def test_base_geometry_call_batch_shape() -> None:
    geom = DummyBaseGeometry(shape=SMALL_SHAPE, device="cpu", seed=0)
    out = geom(n_sample=4)
    assert out.shape == torch.Size((4, *SMALL_SHAPE))


def test_base_geometry_call_n1() -> None:
    geom = DummyBaseGeometry(shape=SMALL_SHAPE, device="cpu", seed=0)
    out = geom(n_sample=1)
    assert out.shape == torch.Size((1, *SMALL_SHAPE))


def test_base_geometry_call_values() -> None:
    geom = DummyBaseGeometry(shape=SMALL_SHAPE, device="cpu", seed=0)
    out = geom(n_sample=2)
    assert torch.all(out == 1.0)


def test_base_geometry_attrs_values() -> None:
    geom = DummyBaseGeometry(shape=SMALL_SHAPE, device="cpu", seed=7)
    attrs = geom.attrs
    assert attrs["generator"] == "DummyBaseGeometry"
    assert attrs["shape"] == SMALL_SHAPE
    assert attrs["seed"] == 7
    assert attrs["axis_order"] == ("N", "X", "Y", "Z")


def test_base_geometry_voxel_grid_shape() -> None:
    geom = DummyBaseGeometry(shape=SMALL_SHAPE, device="cpu")
    grid = geom.voxel_grid
    assert grid.shape == torch.Size((*SMALL_SHAPE, 3))


def test_base_geometry_voxel_grid_values() -> None:
    geom = DummyBaseGeometry(shape=(3, 4, 5), device="cpu")
    grid = geom.voxel_grid
    # First coordinate axis should contain 0, 1, 2 along X
    assert grid[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert grid[1, 0, 0, 0].item() == pytest.approx(1.0)
    assert grid[2, 0, 0, 0].item() == pytest.approx(2.0)
    # Third coordinate axis should contain 0..4 along Z
    assert grid[0, 0, 0, 2].item() == pytest.approx(0.0)
    assert grid[0, 0, 4, 2].item() == pytest.approx(4.0)


def test_base_geometry_normalized_grid_shape() -> None:
    geom = DummyBaseGeometry(shape=SMALL_SHAPE, device="cpu")
    grid = geom.normalized_grid
    assert grid.shape == torch.Size((*SMALL_SHAPE, 3))


def test_base_geometry_normalized_grid_range() -> None:
    geom = DummyBaseGeometry(shape=SMALL_SHAPE, device="cpu")
    grid = geom.normalized_grid
    assert grid.min().item() == pytest.approx(-1.0)
    assert grid.max().item() == pytest.approx(1.0)


# --- ObjectGeometry ---


def test_object_geometry_both_params_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        DummyObjectGeometry(
            shape=SMALL_SHAPE,
            fov=SMALL_FOV,
            num_objects=3,
            vf_range=(0.05, 0.5),
            device="cpu",
        )


def test_object_geometry_neither_param_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        DummyObjectGeometry(
            shape=SMALL_SHAPE,
            fov=SMALL_FOV,
            num_objects=None,
            vf_range=None,
            device="cpu",
        )


def test_object_geometry_radius_range_conversion() -> None:
    # voxel_size_mean = (2/8) = 0.25 mm/vox
    # radius_range = (0.5/2/0.25, 2.0/2/0.25) = (1.0, 4.0) voxels
    obj = DummyObjectGeometry(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=1,
        vf_range=None,
        diameter_range=(0.5, 2.0),
        device="cpu",
        seed=0,
    )
    assert obj.radius_range == pytest.approx((1.0, 4.0))


def test_object_geometry_forward_shape_num_objects() -> None:
    obj = DummyObjectGeometry(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    out = obj.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_object_geometry_forward_shape_vf_range() -> None:
    obj = DummyObjectGeometry(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.1, 0.6), device="cpu", seed=0
    )
    out = obj.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_object_geometry_overlap_true_overwrites() -> None:
    # DummyObjectGeometry always masks the first-half X slice.
    # With allow_overlap=True, object 2 overwrites object 1 in that same region.
    obj = DummyObjectGeometry(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        allow_overlap=True,
        device="cpu",
        seed=0,
    )
    out = obj.forward()
    assert out.max().item() == 2
    assert 1 not in out.unique().tolist()


def test_object_geometry_overlap_false_preserves_labels() -> None:
    # With allow_overlap=False, object 2 cannot write into voxels already set.
    # Since object 1 fills the first half, object 2's mask is fully blocked.
    obj = DummyObjectGeometry(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        allow_overlap=False,
        device="cpu",
        seed=0,
    )
    out = obj.forward()
    assert out.max().item() == 1
    assert 2 not in out.unique().tolist()


def test_object_geometry_vf_range_nonzero() -> None:
    obj = DummyObjectGeometry(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.1, 0.6), device="cpu", seed=0
    )
    out = obj.forward()
    assert (out > 0).any()


def test_object_geometry_attrs_keys() -> None:
    obj = DummyObjectGeometry(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=1,
        vf_range=None,
        allow_overlap=False,
        device="cpu",
        seed=0,
    )
    attrs = obj.attrs
    for key in ("fov", "num_objects", "vf_range", "diameter_range", "allow_overlap"):
        assert key in attrs
    assert attrs["allow_overlap"] is False
    assert attrs["vf_range"] is None
    assert attrs["num_objects"] == 1


# --- Model ---


def test_model_init() -> None:
    """Test Model initialization and inherited mixin properties."""
    model = DummyModel(device="cpu", seed=42)
    assert model.seed == 42
    assert model.device == torch.device("cpu")
    assert model.generator is not None
    assert model.np_generator is not None


def test_model_call() -> None:
    """Test the Model __call__ delegates to forward."""
    model = DummyModel(device="cpu")
    input_data = torch.ones((1, 2, 2, 2))

    output = model(input_data)

    assert output.shape == (1, 2, 2, 2)
    assert torch.all(output == 2.0)


def test_model_attrs_values() -> None:
    model = DummyModel(device="cpu", seed=7)
    attrs = model.attrs
    assert attrs["generator"] == "DummyModel"
    assert attrs["seed"] == 7
    assert attrs["device"] == "cpu"
    assert attrs["axis_order"] == ("N", "X", "Y", "Z")


def test_model_call_save_zarr(tmp_path: Path) -> None:
    model = DummyModel(device="cpu")
    input_data = torch.ones((1, 2, 2, 2))
    fname = tmp_path / "out.zarr"

    output = model(input_data, fname=fname)

    assert output.shape == (1, 2, 2, 2)
    assert fname.exists()


def test_model_call_save_nifti(tmp_path: Path) -> None:
    model = DummyModel(device="cpu")
    input_data = torch.ones((1, 2, 2, 2))
    fname = tmp_path / "out.nii"

    output = model(input_data, fname=fname)

    assert output.shape == (1, 2, 2, 2)
    assert fname.exists()


# --- Transform ---


def test_transform_init() -> None:
    """Test Transform initialization and inherited mixin properties."""
    t = DummyTransform(device="cpu", seed=42)
    assert t.seed == 42
    assert t.device == torch.device("cpu")
    assert t.generator is not None
    assert t.np_generator is not None


def test_transform_call() -> None:
    """Test the Transform __call__ pipeline delegates correctly."""
    t = DummyTransform(device="cpu")
    input_data = torch.ones((2, 2, 2))

    # Output should be input (1.0s) + sample (2.0s) = 3.0s
    output = t(input_data)

    assert output.shape == (2, 2, 2)
    assert torch.all(output == 3.0)


# --- Pipeline ---


def test_pipeline() -> None:
    """Test the Pipeline execution."""
    t1 = DummyTransform(device="cpu")
    t2 = DummyTransform(device="cpu")

    pipeline = Pipeline([t1, t2])

    input_data = torch.ones((2, 2, 2))

    # t1 adds 2.0 -> 3.0
    # t2 adds 2.0 -> 5.0
    output = pipeline(input_data)

    assert output.shape == (2, 2, 2)
    assert torch.all(output == 5.0)
    assert len(pipeline.transforms) == 2
