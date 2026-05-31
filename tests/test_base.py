"""Tests for base classes and mixins."""

from typing import Self

import numpy as np
import torch

from synthbold.base import Pipeline, RandomGeneratorMixin, Transform
from synthbold.config import Config

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class DummyClass(RandomGeneratorMixin):
    """Dummy class to test the mixin."""

    pass


class DummyTransform(Transform):
    """Dummy transform for testing."""

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.ones(shape) * 2.0

    def apply(self, x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return x + transform

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(device="cpu")


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
