"""Abstract base classes and mixins for the synthBOLD synthesis pipeline."""

import importlib
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from synthbold.config import Config
from synthbold.decorator import accept_unbatched, require_dim
from synthbold.io import save_nifti, save_zarr

__all__ = [
    "RandomGeneratorMixin",
    "BaseGeometry",
    "ObjectGeometry",
    "Model",
    "Transform",
    "Pipeline",
]


class RandomGeneratorMixin:
    """Mixin that provides a paired PyTorch and NumPy RNG, both seedable for
    reproducibility, for device-aware random sampling.

    Args:
        seed: Integer seed for deterministic output. Pass ``None`` for
            non-deterministic behaviour.
        device: PyTorch device on which ``generator`` is created.
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
        """Return a seeded or randomly initialised torch.Generator on the given
        device."""
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        return gen

    @staticmethod
    def _create_numpy_generator(seed: int | None) -> np.random.Generator:
        """Return a seeded or randomly initialised NumPy Generator."""
        if seed is not None:
            return np.random.default_rng(seed)
        return np.random.default_rng()


class BaseGeometry(ABC, RandomGeneratorMixin):
    """Abstract base for creating basic geometries.

    Subclasses implement ``forward()`` to produce a single sample. This class provides
    the batch loop (``__call__``), optional save-to-disk (ZARR or NIfTI), cached
    coordinate grids, and serialisation of metadata via ``attrs``.

    Args:
        shape: Spatial dimensions ``(X, Y, Z)`` of each generated volume.
        device: PyTorch device for tensor allocation and RNG.
        seed: Integer seed for reproducible output; ``None`` for random.
    """

    # data type for map elements
    dtype = torch.float32

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        device: str | torch.device = "cuda",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed, device=device)
        self.shape = tuple(shape)
        self.device = torch.device(device)

    def __call__(self, n_sample: int, fname: Path | None = None) -> torch.Tensor:
        """Generate a batch of geometries and optionally save them to disk.

        Args:
            n_sample: Number of generated geometry labels.
            fname: File name for saving data to disk in ZARR or NIfTI format.

        Returns:
            Tensor with random geometry labels of shape ``(n_sample, *self.shape)``.
        """
        data = torch.zeros(
            (n_sample, *self.shape), device=self.device, dtype=self.dtype
        )
        for i in range(n_sample):
            data[i, ...] = self.forward().to(self.dtype)

        # save to disk
        if fname is not None and fname.suffix == ".zarr":
            save_zarr(fname, data.cpu().numpy(), self.attrs)
        elif fname is not None and fname.suffix == ".nii":
            save_nifti(fname, data.cpu().numpy(), permute=True)

        return data

    @property
    def attrs(self) -> dict[str, Any]:
        """Common metadata for ZARR/NIfTI storage."""
        try:
            pkg_name = self.__class__.__module__.split(".")[0]
            version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return {
            "generator": self.__class__.__name__,
            "created_at": datetime.now(UTC).isoformat(),
            "version": version,
            "axis_order": ("N", "X", "Y", "Z"),
            "device": str(self.device),
            "shape": self.shape,
            "dtype": str(self.dtype),
            "seed": self.seed,
        }

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Generate random geometry labels."""
        ...

    @cached_property
    def voxel_grid(self) -> torch.Tensor:
        """Creates target mesh grid of shape ``(X, Y, Z, 3)``."""
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(self.shape[0], device=self.device, dtype=torch.float32),
            torch.arange(self.shape[1], device=self.device, dtype=torch.float32),
            torch.arange(self.shape[2], device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack((grid_x, grid_y, grid_z), dim=-1)

    @cached_property
    def normalized_grid(self) -> torch.Tensor:
        """Grid normalized to [-1, 1] of shape ``(X, Y, Z, 3)``."""
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.linspace(
                -1, 1, self.shape[0], device=self.device, dtype=torch.float32
            ),
            torch.linspace(
                -1, 1, self.shape[1], device=self.device, dtype=torch.float32
            ),
            torch.linspace(
                -1, 1, self.shape[2], device=self.device, dtype=torch.float32
            ),
            indexing="ij",
        )
        return torch.stack((grid_x, grid_y, grid_z), dim=-1)


class ObjectGeometry(BaseGeometry, ABC):
    """Abstract base for creating basic objects.

    Args:
        shape: Spatial dimensions ``(X, Y, Z)`` of each generated volume.
        fov: Field of view in mm, used to compute voxel size.
        num_objects: Fixed number of objects to generate. Mutually exclusive with
        `vf_range`.
        vf_range: Range of target volume fractions. Objects are added until the volume
            fraction is reached. Mutually exclusive with `num_objects`.
        diameter_range: Range of object diameters in mm.
        allow_overlap: If False, objects are only added if they contribute new voxels.
        device: PyTorch device for tensor allocation and RNG.
        seed: Integer seed for reproducible output; ``None`` for random.

    Notes:
        `num_objects` and `vf_range` are mutually exclusive. If `num_objects` is set,
        objects are added iteratively until the specified number is reached. If
        `vf_range` is set, a target volume fraction is randomly sampled from the
        specified range and objects are added iteratively until the target fraction
        of occupied voxels is reached.
    """

    def __init__(
        self,
        shape: tuple[int, int, int] = (128, 128, 128),
        fov: tuple[float, float, float] = (1.0, 1.0, 1.0),
        num_objects: int | None = None,
        vf_range: tuple[float, float] | None = (0.05, 0.1),
        diameter_range: tuple[float, float] = (0.05, 2.0),
        allow_overlap: bool = True,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(shape, seed=seed, device=device)
        self.shape = shape
        self.fov = fov
        self.num_objects = num_objects
        self.vf_range = vf_range
        self.diameter_range = diameter_range
        self.allow_overlap = allow_overlap

        # num_objects and vf_range are mutually exclusive
        if (num_objects is None) == (vf_range is None):
            raise ValueError("Specify exactly one of num_objects or vf_range.")

        # Convert diameter_range in mm to radius_range in voxel dimension.
        # We use the mean voxel size across dimensions for conversion.
        voxel_size = tuple(f / n for f, n in zip(fov, shape, strict=True))
        voxel_size_mean = sum(voxel_size) / 3
        self.radius_range = (
            diameter_range[0] / 2 / voxel_size_mean,
            diameter_range[1] / 2 / voxel_size_mean,
        )

    @property
    def attrs(self) -> dict[str, Any]:
        """Metadata for ZARR/NIfTI storage."""
        return {
            **super().attrs,
            "fov": self.fov,
            "num_objects": self.num_objects,
            "vf_range": self.vf_range,
            "diameter_range": self.diameter_range,
            "allow_overlap": self.allow_overlap,
        }

    def forward(self) -> torch.Tensor:
        """Generate a labeled 3D volume with random objects."""
        # Initialize volume
        volume = torch.zeros(self.shape, device=self.device, dtype=torch.int32)
        total_voxels = volume.numel()

        # Mode 1: fixed number of objects
        if self.num_objects is not None:
            for i in range(self.num_objects):
                volume, _ = self._add_object(i + 1, volume)

        # Mode 2: target volume fraction
        elif self.vf_range is not None:
            vf = (
                torch.rand((), device=self.device, generator=self.generator)
                * (self.vf_range[1] - self.vf_range[0])
                + self.vf_range[0]
            )
            i, occupied = 0, 0
            max_iter = total_voxels * 10
            while occupied / total_voxels < vf and i < max_iter:
                i += 1
                volume, _ = self._add_object(i, volume)
                # recompute occupied voxels
                occupied = int(torch.count_nonzero(volume).item())
        else:
            raise ValueError(
                "Either number of objects or volume fraction must be specified."
            )

        return volume

    def _add_object(self, label: int, volume: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Insert an object into a labeled 3D volume. A binary object mask is generated
        and written into the provided volume. Depending on the overlap setting, the
        object may overwrite existing labels or only fill previously empty voxels.

        Args:
            label: Integer label assigned to voxel belonging to the object.
            volume: 3D tensor representing the labeled volume. Background voxels are
                assumed to be 0.

        Returns:
            A tuple with the following elements:
                - Updated volume with the object inserted.
                - Number of voxels newly assigned to this object (i.e., voxels
                  written during this operation).
        """
        # Generate object mask. If failed, do nothing.
        mask = self._mask_object()
        if mask is None:
            return volume, 0
        if not self.allow_overlap:
            mask = mask & (volume == 0)
        new_voxels = int(torch.count_nonzero(mask).item())
        volume[mask] = label
        return volume, new_voxels

    @abstractmethod
    def _mask_object(self) -> torch.Tensor | None:
        """Return a boolean mask for one object instance."""
        ...


class Model(ABC, RandomGeneratorMixin):
    """Abstract base class for making single models callable.

    Args:
        device: PyTorch device for tensor allocation and RNG.
        seed: Integer seed for reproducible output; ``None`` for random.
    """

    def __init__(self, *, device: str = "cpu", seed: int | None = None) -> None:
        super().__init__(seed=seed, device=device)

    def __call__(self, data: torch.Tensor, fname: Path | None = None) -> torch.Tensor:
        """Generate data from input tensor and optionally save generated data to disk in
        ZARR or NIfTI format."""
        result: torch.Tensor = self.forward(data)

        # save to disk
        if fname is not None and fname.suffix == ".zarr":
            save_zarr(fname, result.cpu().numpy(), self.attrs)
        elif fname is not None and fname.suffix == ".nii":
            save_nifti(fname, result.cpu().numpy(), permute=True)

        return result

    @property
    def attrs(self) -> dict[str, Any]:
        """Common metadata for ZARR/NIfTI storage."""
        try:
            pkg_name = self.__class__.__module__.split(".")[0]
            version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return {
            "generator": self.__class__.__name__,
            "created_at": datetime.now(UTC).isoformat(),
            "version": version,
            "axis_order": ("N", "X", "Y", "Z"),
            "device": str(self.device),
            "seed": self.seed,
        }

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Generate data."""
        ...


class Transform(ABC, RandomGeneratorMixin):
    """Base class for single transformations on PyTorch tensors.

    This abstract class provides the foundation for all data transformations.
    Subclasses are required to implement `sample`, `apply`, and `from_config`. The base
    implementation automatically handles moving input data to the target device and
    sequencing the sampling and application steps.

    Args:
        device: PyTorch device for tensor allocation and RNG.
        seed: Integer seed for reproducible output; ``None`` for random.
    """

    def __init__(self, device: str | torch.device, seed: int | None = None) -> None:
        super().__init__(seed=seed, device=device)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Move data to the target device and apply the transform."""
        data = data.to(self.device)
        result = self.forward(data)
        return result

    @abstractmethod
    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Sample transform parameters for a tensor of the given shape."""
        ...

    @staticmethod
    @abstractmethod
    def apply(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Apply pre-sampled transform parameters to the input tensor."""
        ...

    @require_dim(3, 4)
    @accept_unbatched(dim=3)
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
