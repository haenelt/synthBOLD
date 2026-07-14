"""Generation of foreground data."""

from typing import Self

import torch

from synthbold.base import Model, Pipeline, Transform
from synthbold.config import Config
from synthbold.transforms import CaliberDeformation, ElasticDeformation, RandomFlip

__all__ = ["ForegroundModel", "DistractorModel"]


class ForegroundModel(Model):
    """A model for generating foreground data, i.e. synthetic cylinders from label
    maps, to synthesize vessel-like networks.

    This class turns an integer-labeled 4D volume ``(B, X, Y, Z)`` into randomized
    vessel-like structures by passing it through a pipeline of transforms. Unlike
    :class:`~synthbold.models.background.BackgroundModel`, no intensity sampling is
    performed here; the label map itself is spatially warped (e.g. bent, flipped,
    pinched) to look like plausible vasculature. The default alternative constructor,
    :meth:`from_config`, chains :class:`~synthbold.transforms.RandomFlip`,
    :class:`~synthbold.transforms.ElasticDeformation` (for smooth curving), and
    :class:`~synthbold.transforms.CaliberDeformation` (for varying vessel diameter).

    Args:
        transforms: List of transformations to apply, in order.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.

    Notes:
        The generated foreground maps can be returned as a PyTorch tensor and saved to
        disk in ZARR or NIfTI format, with metadata automatically stored when using
        ZARR.

    Examples:
        >>> data_tensor = torch.randint(1, 21, (10, 10, 10, 10))
        >>> foreground_model = ForegroundModel(transforms=[])
        >>> sample_map = foreground_model(data_tensor)
    """

    def __init__(
        self,
        transforms: list[Transform] | None = None,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.pipeline = Pipeline(transforms or [])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Generate random vessel map.

        Args:
            data: Integer-labeled 4D tensor of shape ``(B, X, Y, Z)`` representing
                vessels as binary long straight cylinders. Must contain labels greater
                than or equal to 1.

        Returns:
            Synthesized tensor of the same shape as the input tensor.
        """
        return self.pipeline(data)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs a ForegroundModel instance with the standard vessel-synthesis
        pipeline (random flips, elastic deformation, and caliber deformation) using
        parameters read from a config object."""
        transforms: list[Transform] = [
            RandomFlip.from_config(config),
            ElasticDeformation.from_config(config),
            CaliberDeformation.from_config(config),
        ]
        return cls(
            transforms=transforms,
            device=config.device,
            seed=config.seed,
        )


class DistractorModel(ForegroundModel):
    """A model for generating distractor data, e.g. spheres, cubes, tetrahedra, or
    toroids.

    Distractors are non-vessel objects added alongside the vasculature so that
    downstream models must learn to distinguish real vessel-like structures from
    other shapes rather than merely detecting "any foreground object". This class
    shares :class:`ForegroundModel`'s pipeline mechanics (it turns an integer-labeled
    4D volume ``(B, X, Y, Z)`` into a transformed label map) but overrides
    :meth:`from_config` to skip the elastic and caliber deformations used for
    vessels, since distractor shapes should keep their characteristic geometry and
    only be randomly flipped.

    Args:
        transforms: List of transformations to apply, in order.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.

    Notes:
        The generated distractor maps can be returned as a PyTorch tensor and saved to
        disk in ZARR or NIfTI format, with metadata automatically stored when using
        ZARR.

    Examples:
        >>> data_tensor = torch.randint(1, 21, (10, 10, 10, 10))
        >>> distractor_model = DistractorModel(transforms=[])
        >>> sample_map = distractor_model(data_tensor)
    """

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs a DistractorModel instance that only applies RandomFlip (no
        shape-deforming transforms), using parameters read from a config object."""
        transforms: list[Transform] = [RandomFlip.from_config(config)]
        return cls(
            transforms=transforms,
            device=config.device,
            seed=config.seed,
        )
