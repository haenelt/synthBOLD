"""Generation of background data."""

from typing import Self

import torch

from synthbold.base import Model
from synthbold.config import Config

__all__ = ["BackgroundModel"]


class BackgroundModel(Model):
    """A model for generating synthetic baclground data from integer-labeled volume
    data.

    This class allows creating randomized intensity maps base don a 4D label volume
    ``(B, X, Y, Z)``. Each unique label in the input data is assigned a random mean `mu`
    and standard deviation `std` drawn from specified ranges. Voxel-wise values are
    sampled from a normal distribution. Background voxels (label 0; if exist) are
    ignored.

    Args:
        config: Configuation object.

    Notes:
        The genrated background maps can be returned as a PyTorch tensor or saved and
        saved to disk in ZARR or NIfTI format, with metadata automatically stored when
        using ZARR.

    Examples:
        >>> data_tensor = torch.randint(1, 21, (10, 10, 10, 10))
        >>> config = Config()
        >>> background_model = BackgroundModel(config)
        >>> sample_map = background_model(data_tensor)
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config=config, seed=config.seed, device=config.device)
        # do it differently

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Generate random background map.

        Args:
            data: Integer-labeled 4D tensor of shape ``(B, X, Y, Z)``. Must contain
                labels greater than or equal to 1.

        Returns:
            Synthesized background tensor of the same shape as the input tensor.
        """
        B, X, Y, Z = data.shape

        out = torch.zeros_like(data, dtype=torch.float32, device=self.device)

        return out

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(config=config)
