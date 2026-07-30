"""Generation of background data."""

from typing import Any, Self

import torch

from synthbold.base import Model, Transform
from synthbold.config import Config
from synthbold.sampling import sample_uniform
from synthbold.transforms import (
    BiasField,
    DeformedSphericalMask,
    GaussianSmoothing,
    RandomFlip,
)
from synthbold.transforms.functional import Pipeline

__all__ = ["BackgroundModel"]


class BackgroundModel(Model):
    """A model for generating synthetic background data from integer-labeled volume
    data.

    This class allows creating randomized intensity maps based on a 4D label volume
    ``(B, X, Y, Z)``. Each unique label in the input data is assigned a random mean `mu`
    and standard deviation `std` drawn from specified ranges. Voxel-wise values are
    sampled from a normal distribution. Background voxels (label 0; if exist) are
    ignored.

    Args:
        mu_range: Range of values to sample the mean intensity from.
        std_range: Range of values to sample the standard deviation from.
        transforms: List of transformations to apply.
        transform_prob: Probability of applying each transform, evaluated independently
            per transform and per batch element.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If `std_range` contains a negative value.

    Notes:
        The generated background maps can be returned as a PyTorch tensor and saved to
        disk in ZARR or NIfTI format, with metadata automatically stored when using
        ZARR.

    Examples:
        >>> data_tensor = torch.randint(1, 21, (10, 10, 10, 10))
        >>> background_model = BackgroundModel(
        >>>     mu_range=(0.0, 1.0), std_range=(0.0, 0.5)
        >>> )
        >>> sample_map = background_model(data_tensor)
    """

    def __init__(
        self,
        mu_range: tuple[float, float],
        std_range: tuple[float, float],
        transforms: list[Transform] | None = None,
        transform_prob: float = 1.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed, device=device)
        if std_range[0] < 0 or std_range[1] < 0:
            raise ValueError(
                f"Standard deviations must be non-negative, got {std_range}."
            )

        self.mu_range = mu_range
        self.std_range = std_range
        self.pipeline = Pipeline(
            transforms or [], prob=transform_prob, device=device, seed=seed
        )

    def forward(self, data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate random background map.

        Args:
            data: Integer-labeled 4D tensor of shape ``(B, X, Y, Z)``. Label `0` is
                treated as background, if present, and left at zero intensity.
            **kwargs: Unused; accepted for interface compatibility with `Model`.

        Returns:
            Synthesized background tensor of the same shape as the input tensor.
        """
        B, X, Y, Z = data.shape

        flat_labels = data.reshape(B, -1)  # [B, V]
        flat_out = torch.zeros_like(
            flat_labels, dtype=torch.float32, device=self.device
        )
        for b in range(B):
            flat_out[b] = self._sample_labels(flat_labels[b])

        out = flat_out.reshape(B, X, Y, Z)
        return self.pipeline(out)

    def _sample_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Sample intensity values for one flattened, integer-labeled volume.

        Each unique nonzero label is assigned a random mean and standard deviation, and
        its voxels are drawn from the corresponding normal distribution. Background
        voxels (label 0, if present) are left at zero.

        Args:
            labels: 1D tensor of non-negative integer labels.

        Returns:
            1D tensor of sampled intensities, same shape as `labels`.
        """
        out = torch.zeros_like(labels, dtype=torch.float32, device=self.device)

        unique_labels, inverse = torch.unique(labels, return_inverse=True)
        has_background = bool(unique_labels[0] == 0)
        n_labels = unique_labels.numel() - int(has_background)
        if n_labels == 0:
            return out

        mu_low, mu_high = self.mu_range
        std_low, std_high = self.std_range
        mus = sample_uniform(n_labels, mu_low, mu_high, self.device, self.generator)
        stds = sample_uniform(n_labels, std_low, std_high, self.device, self.generator)

        # Sorted unique labels put background (0) first, so subtracting its
        # presence turns `inverse` into a 0-based label index, with -1 for background.
        label_idx = inverse - int(has_background)
        mask = label_idx >= 0
        out[mask] = torch.normal(
            mean=mus[label_idx[mask]],
            std=stds[label_idx[mask]],
            generator=self.generator,
        )
        return out

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs BackgroundModel instance from a config object."""
        mu_range = (
            config.physio.background_mu.min,
            config.physio.background_mu.max,
        )
        std_range = (
            config.physio.background_std.min,
            config.physio.background_std.max,
        )
        transforms: list[Transform] = [
            RandomFlip.from_config(config),
            BiasField.from_config(config),
            DeformedSphericalMask.from_config(config),
            GaussianSmoothing.from_config(config),
        ]
        return cls(
            mu_range=mu_range,
            std_range=std_range,
            transforms=transforms,
            transform_prob=config.transform_prob,
            device=config.device,
            seed=config.seed,
        )
