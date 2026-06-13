"""Deformation transforms."""

from typing import Self

import torch
from monai.transforms import DistanceTransformEDT, GaussianSmooth

from synthbold.base import Transform
from synthbold.config import Config
from synthbold.transforms.functional import apply_deformation

__all__ = ["ElasticDeformation", "CaliberDeformation"]


class ElasticDeformation(Transform):
    """Applies a smooth random elastic deformation to 3D or 4D tensors)``.

    The transform generates a random noise field, smooths it with a Gaussian kernel to
    obtain a spatially smooth displacement field, and applies this displacement equally
    in the `x`, `y`, and `z` directions to warp the volume. This results in smooth
    deformation of structures (e.g., for curving vessels), without explicitly modeling
    diameter changes.

    Args:
        sigma: Standard deviation of Gaussian field to generate smooth random Gaussian
            field.
        max_disp: Maximum displacement magnitude (in voxels) applied after
            normalization of the displacement field.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sigma: float = 20.0,
        max_disp: float = 10.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.sigma = sigma
        self.max_disp = max_disp
        self.gaussian_filter = GaussianSmooth(sigma)

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Creates a random displacement field for smooth elastic deformations. Note
        that shape is expected to be a tuple of length 4, corresponding to dimension
        ``(B, X, Y, Z)``."""
        noise = (
            torch.rand(
                shape, device=self.device, dtype=torch.float32, generator=self.generator
            )
            * 2
            - 1
        )
        disp = self.gaussian_filter(noise)
        # Get the same displacmenet field in X, Y, and Z dimensions
        disp = disp.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        disp /= disp.max()
        disp *= self.max_disp
        return disp

    @staticmethod
    def apply(data: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """Randomly applies elastic transform to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            displacement: Displacement field of same shape.

        Returns:
            Tensor of the same shape with inserted elastic deformations.
        """
        return apply_deformation(data, displacement, interp="nearest")

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs ElasticDeformation instance from a config object."""
        if config.transform.elastic_sigma is None:
            raise ValueError(
                "Elastic deformation parameters are not set in the config."
            )
        return cls(
            sigma=config.transform.elastic_sigma,
            max_disp=config.transform.elastic_max_disp,
            device=config.device,
            seed=config.seed,
        )


class CaliberDeformation(Transform):
    """Applies a random caliber transform to 3D or 4D tensors.

    This transformation is designed to modify roughly cylindrical structures (e.g.
    vessels) to have smooth, spatially varying diameters. It works by generating a
    smoothed random radius field and applying a local distance-based masking ("pinch")
    to the input volume.

    Args:
        sigma: Standard deviation of Gaussian field to generate smooth random Gaussian
            field.
        alpha: Radius variation of random radius field.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sigma: float = 20.0,
        alpha: float = 4.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.sigma = sigma
        self.alpha = alpha
        self.gaussian_filter = GaussianSmooth(sigma)

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Creates a random displacement field for smooth caliber deformations. Note
        that shape is expected to be a tuple of length 4, corresponding to dimension
        ``(B, X, Y, Z)``."""
        B, _, _, _ = shape

        # Pre-generate batch noise
        noise = (
            torch.rand(
                shape, device=self.device, dtype=torch.float32, generator=self.generator
            )
            * 2
            - 1
        )
        noise_map = self.gaussian_filter(noise)

        # Normalize per batch element
        mean = noise_map.view(B, -1).mean(dim=1).view(B, 1, 1, 1)
        std = noise_map.view(B, -1).std(dim=1).view(B, 1, 1, 1) + 1e-8
        noise_map = (noise_map - mean) / std
        return torch.abs(self.alpha * noise_map)

    @staticmethod
    def apply(data: torch.Tensor, radius_field: torch.Tensor) -> torch.Tensor:
        """Randomly applies caliber transform to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            radius_field: Displacement field for caliber deformation of same shape.

        Returns:
            Tensor of the same shape with inserted caliber deformations.
        """
        edt = DistanceTransformEDT()
        max_val = int(data.max().item())
        result = torch.zeros(data.shape, dtype=data.dtype, device=data.device)

        # Loop over labels
        for i in range(1, max_val + 1):
            mask = ~(data == i)
            distance_map = edt(mask)
            # Apply radius field
            data_transformed = torch.as_tensor(
                distance_map <= radius_field, device=data.device, dtype=torch.float32
            )
            result[data_transformed > 0] = i

        return result

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs CaliberTransform instance from a config object."""
        if config.transform.caliber_sigma is None:
            raise ValueError("Caliber parameters are not set in the config.")
        return cls(
            sigma=config.transform.caliber_sigma,
            alpha=config.transform.caliber_alpha,
            device=config.device,
            seed=config.seed,
        )
