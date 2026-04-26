"""Base classes module

This module contains the foundational classes and mixins used to build
the transformation pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import torch

from synthbold.config import Config


class RandomGeneratorMixin:
    """Mixin class to provide reproducible PyTorch and NumPy random number generators.

    This mixin initializes:
        1. `torch.Generator` on the specified device for PyTorch-based random numbers.
        2. `numpy.random.Generator` seeded consistently with the PyTorch generator for
            NumPy-based random numbers.

    Both generators are reproducible if a seed is provided, making it suitable for
    simulations or transforms that mix PyTorch and NumPy.

    Attributes:
        seed: The seed used for reproducibility, or None if unseeded.
        device: The PyTorch device for the generator.
        generator: The initialized PyTorch random number generator.
        np_generator: The initialized NumPy random number generator.
    """

    def __init__(
        self, seed: int | None, device: str | torch.device = "cuda", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.seed = seed
        self.device = torch.device(device)
        self.generator = self._create_generator(seed, device)
        self.np_generator = self._create_numpy_generator(seed)

    @staticmethod
    def _create_generator(
        seed: int | None, device: str | torch.device
    ) -> torch.Generator:
        """Create a PyTorch generator with optional seeding.

        Args:
            seed: Seed for the generator. If None, the generator is seeded randomly.
            device: Device for the generator.

        Returns:
            Initialized PyTorch random generator.
        """
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        return gen

    @staticmethod
    def _create_numpy_generator(seed: int | None) -> np.random.Generator:
        """Create a NumPy generator with optional seeding.

        Args:
            seed: Seed for the generator. If None, the generator is not seeded.

        Returns:
            Initialized NumPy random generator.
        """
        if seed is not None:
            return np.random.default_rng(seed)
        return np.random.default_rng()


class Transform(ABC, RandomGeneratorMixin):
    """Base class for single transformations on PyTorch tensors.

    This abstract class provides the foundation for all data transformations. Subclasses
    are required to implement `sample`, `apply`, and `from_config`. The base
    implementation automatically handles moving input data to the target device and
    sequencing the sampling and application steps.

    Args:
        device: The PyTorch device (e.g., 'cpu', 'cuda') where the
            transformation will operate.
        seed: An optional seed for reproducible random number generation.
    """

    def __init__(self, device: str | torch.device, seed: int | None = None) -> None:
        super().__init__(seed=seed, device=device)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """API for data transformation."""
        # Move input tensor to the correct device
        data = data.to(self.device)
        result = self.forward(data)
        return result

    @abstractmethod
    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes transformation with shape based on input tensor shape."""
        ...

    @abstractmethod
    def apply(self, x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Applies transformation to input tensor."""
        ...

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Generates transformation and applies to input tensor."""
        transformation = self.sample(data.shape)
        return self.apply(data, transformation)

    @classmethod
    @abstractmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs a Transform instance from a configuration object."""
        ...


class Pipeline:
    """Compose multiple tensor-to-tensor transforms in sequence.

    This class chains multiple `Transform` instances together. When called, it passes
    the input tensor through each transform in the order they were provided.

    Args:
        transforms: A list of `Transform` instances to be applied sequentially.
    """

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply pipeline to input tensor."""
        for t in self.transforms:
            result = t(data)
            data = result
        return data
