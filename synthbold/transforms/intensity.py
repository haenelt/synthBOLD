"""Intensity transforms."""

from typing import Self

import torch

from synthbold.base import Transform
from synthbold.config import Config
from synthbold.sampling import sample_uniform
from synthbold.transforms.functional import sample_lowres_noise

__all__ = ["BiasField", "GammaTransform"]


class BiasField(Transform):
    """Generates a smooth multiplicative bias field.

    This transform creates a random low-resolution field, upsamples it to the target
    shape, applies an exponential to ensure positivity, and multiplies it with the input
    tensor. A randomly generated bias field is created for each 3D volume within the
    batch.

    Args:
        lowres_shape: The matrix size of the low-resolution noise map to generate a
            smooth bias field.
        amplitude_range: Minimum and maximum scaling factor for bias field variation.
        device: Target compute device, e.g. 'cuda' or 'cpu'.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If any `lowres_shape` dimension is less than 1.
    """

    def __init__(
        self,
        lowres_shape: tuple[int, int, int],
        amplitude_range: tuple[float, float] = (0.0, 1.0),
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        if any(d < 1 for d in lowres_shape):
            raise ValueError(
                f"lowres_shape dimensions must be >= 1, got {lowres_shape}."
            )

        self.lowres_shape = lowres_shape
        self.amplitude_range = amplitude_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Compute the bias field of shape ``(B, X, Y, Z)``."""
        B, X, Y, Z = shape
        # scaling factor of bias variation
        amplitude = self.amplitude_range[0] + (
            self.amplitude_range[1] - self.amplitude_range[0]
        ) * torch.rand((B,), generator=self.generator, device=self.device)

        # Smooth low-frequency noise field: (B, X, Y, Z)
        bias = sample_lowres_noise(
            B,
            self.lowres_shape,
            (X, Y, Z),
            self.device,
            self.generator,
            align_corners=True,
        )
        # Make strictly positive (multiplicative factor) and reshape amplitude for
        # broadcast
        bias = torch.exp(bias * amplitude[:, None, None, None])

        return bias

    @staticmethod
    def apply(data: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Applies the smooth bias field to the input tensor.

        Args:
            data: Input tensor of shape ``(B, X, Y, Z)``.
            bias: Bias field tensor of same shape.

        Returns:
            Biased tensor of same shape.
        """
        return data * bias

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs BiasField instance from a config object."""
        amplitude_range = (
            config.transform.bias_amplitude.min,
            config.transform.bias_amplitude.max,
        )
        return cls(
            lowres_shape=config.transform.bias_lowres_shape,
            amplitude_range=amplitude_range,
            device=config.device,
            seed=config.seed,
        )


class GammaTransform(Transform):
    """Applies a random gamma intensity transform.

    For each volume in the batch, a random `gamma` exponent is sampled from a uniform
    distribution with set boundaries. The volume is rescaled to ``[0, 1]``, raised to
    the power of `gamma`, and rescaled back to its original intensity range. Values of
    `gamma` below 1 brighten darker regions, while values above 1 darken them.

    Args:
        gamma_range: Minimum and maximum gamma exponent. Must be strictly positive.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If `gamma_range` is not strictly positive.
    """

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.5, 2.0),
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        if gamma_range[0] <= 0:
            raise ValueError("gamma_range must be strictly positive.")

        self.gamma_range = gamma_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes per-volume gamma exponents for the given output shape
        ``(B, X, Y, Z)``."""
        B, _, _, _ = shape

        gamma = sample_uniform(
            B, self.gamma_range[0], self.gamma_range[1], self.device, self.generator
        ).view(B, 1, 1, 1)
        return gamma

    @staticmethod
    def apply(data: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Applies the gamma transform to the input tensor.

        Args:
            data: Input tensor of shape ``(B, X, Y, Z)``.
            gamma: Per-volume gamma exponent tensor broadcastable to the shape of
                `data`.

        Returns:
            Tensor of the same shape with gamma-adjusted intensities.
        """
        eps = 1e-7
        data_min = data.amin(dim=(-3, -2, -1), keepdim=True)
        data_max = data.amax(dim=(-3, -2, -1), keepdim=True)
        data_range = data_max - data_min
        normalized = (data - data_min) / (data_range + eps)
        return normalized.pow(gamma) * data_range + data_min

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs GammaTransform instance from a config object."""
        gamma_range = (
            config.transform.gamma_exponent.min,
            config.transform.gamma_exponent.max,
        )
        return cls(
            gamma_range=gamma_range,
            device=config.device,
            seed=config.seed,
        )
