"""Abstract base classes and mixins for the synthBOLD synthesis pipeline."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from synthbold.config import Config
from synthbold.decorator import log_call, to_device
from synthbold.io import save_nifti, save_zarr, zarr_attributes
from synthbold.sampling import sample_uniform

__all__ = [
    "RandomGeneratorMixin",
    "BaseGeometry",
    "ObjectGeometry",
    "Model",
    "Transform",
]


logger = logging.getLogger(__name__)


class RandomGeneratorMixin:
    """Mixin that provides a paired PyTorch and NumPy RNG, both seedable for
    reproducibility, for device-aware random sampling.

    Args:
        seed: Integer seed for deterministic output. Pass ``None`` for
            non-deterministic behaviour.
        device: PyTorch device on which ``generator`` is created.
    """

    def __init__(
        self, seed: int | None, device: str | torch.device = "cpu", **kwargs: Any
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
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed, device=device)
        self.shape = tuple(shape)
        self.device = torch.device(device)

    @log_call
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
        """Metadata for ZARR/NIfTI storage."""
        return {
            **zarr_attributes(self.__class__.__name__, self.device, self.seed),
            "shape": self.shape,
            "dtype": str(self.dtype),
        }

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Generate random geometry labels."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs a geometry instance from a configuration object."""
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

    Raises:
        ValueError: If neither or both of `num_objects` and `vf_range` are given.

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

        # Convert mm units to voxel units using the mean voxel size across dimensions.
        voxel_size = tuple(f / n for f, n in zip(fov, shape, strict=True))
        self.voxel_size_mean = sum(voxel_size) / 3
        self.radius_range = (
            self._mm_to_voxels(diameter_range[0]) / 2,
            self._mm_to_voxels(diameter_range[1]) / 2,
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
                volume, new_voxels = self._add_object(i, volume)
                occupied += new_voxels
            if occupied / total_voxels < vf:
                logger.warning(
                    "%s: reached max_iter=%d without hitting target volume fraction "
                    "(%.4f<%.4f); returning under-filled volume.",
                    type(self).__name__,
                    max_iter,
                    occupied / total_voxels,
                    vf,
                )
        else:
            raise ValueError(
                "Either number of objects or volume fraction must be specified."
            )

        return volume

    def _mm_to_voxels(self, mm: float) -> float:
        """Converts a length in mm to voxel units using the mean voxel size."""
        return mm / self.voxel_size_mean

    def _segment_mask(
        self, start: torch.Tensor, end: torch.Tensor, radius: torch.Tensor
    ) -> torch.Tensor | None:
        """Generate a binary mask of a finite cylindrical segment between two points.

        Args:
            start: Segment start point of shape ``(3,)``, in voxel coordinates.
            end: Segment end point of shape ``(3,)``, in voxel coordinates.
            radius: Segment radius in voxel units.

        Returns:
            Boolean tensor of shape `shape` where True indicates voxels inside the
            segment. Returns `None` if `start` and `end` coincide.
        """
        # Vector along segment axis
        vec_v = end - start
        vec_v_norm2 = torch.dot(vec_v, vec_v)
        if vec_v_norm2 < 1e-12:
            return None

        # Vector from start point to every voxel
        vec_p = self.voxel_grid - start

        # Project each voxel onto the segment axis, clamped to the segment extent
        t = torch.tensordot(vec_p, vec_v, dims=([3], [0])) / vec_v_norm2
        t = torch.clamp(t, 0.0, 1.0)

        # Compute closest point on the segment for each voxel
        proj = start + t.unsqueeze(-1) * vec_v

        # Euclidean distance from voxel to segment
        dist = torch.norm(self.voxel_grid - proj, dim=-1)
        return dist <= radius

    def _local_segment_mask(
        self, start: torch.Tensor, end: torch.Tensor, radius: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[slice, slice, slice]] | None:
        """Generate a binary mask of a finite cylindrical segment, restricted to its
        axis-aligned bounding box (padded by `radius`) instead of the full volume. This
        is substantially cheaper than `_segment_mask` when segments are small relative
        to the volume, e.g. when growing a tree from many short segments.

        Args:
            start: Segment start point of shape ``(3,)``, in voxel coordinates.
            end: Segment end point of shape ``(3,)``, in voxel coordinates.
            radius: Segment radius in voxel units.

        Returns:
            A tuple of the boolean submask and the slices locating it within the full
            volume. Returns `None` if `start` and `end` coincide, or if the padded
            bounding box does not intersect the volume.
        """
        # Vector along segment axis
        vec_v = end - start
        vec_v_norm2 = torch.dot(vec_v, vec_v)
        if vec_v_norm2 < 1e-12:
            return None

        # Axis-aligned bounding box around the segment, padded by radius (+1 voxel
        # margin so voxels just outside the segment endpoints are still included)
        pad = radius.item() + 1.0
        lo = torch.minimum(start, end) - pad
        hi = torch.maximum(start, end) + pad

        # Clip the padded box to the volume bounds, per axis; bail out if it lies
        # entirely outside the volume
        slices = []
        for i in range(3):
            lo_i = max(int(torch.floor(lo[i]).item()), 0)
            hi_i = min(int(torch.ceil(hi[i]).item()) + 1, self.shape[i])
            if lo_i >= hi_i:
                return None
            slices.append(slice(lo_i, hi_i))
        slices_t = (slices[0], slices[1], slices[2])

        # Same projection/distance test as `_segment_mask`, but evaluated only on the
        # cropped grid instead of the full volume
        grid = self.voxel_grid[slices_t]
        vec_p = grid - start
        t = torch.tensordot(vec_p, vec_v, dims=([3], [0])) / vec_v_norm2
        t = torch.clamp(t, 0.0, 1.0)
        proj = start + t.unsqueeze(-1) * vec_v
        dist = torch.norm(grid - proj, dim=-1)
        return dist <= radius, slices_t

    def _local_bounding_box(
        self, center: torch.Tensor, radius: torch.Tensor | float
    ) -> tuple[torch.Tensor, tuple[slice, slice, slice]] | None:
        """Crop `voxel_grid` to an axis-aligned bounding box centered on `center`,
        padded by `radius` (+1 voxel margin), instead of the full volume. Valid whenever
        every point of the object lies within `radius` of `center` (e.g. a
        circumradius), and substantially cheaper than evaluating a membership test over
        the full volume when the object is small relative to it.

        Args:
            center: Box center of shape ``(3,)``, in voxel coordinates.
            radius: Bounding radius in voxel units.

        Returns:
            A tuple of the cropped ``(..., 3)`` grid and the slices locating it
            within the full volume. Returns `None` if the padded box does not
            intersect the volume.
        """
        pad = (radius.item() if isinstance(radius, torch.Tensor) else radius) + 1.0
        lo = center - pad
        hi = center + pad

        slices = []
        for i in range(3):
            lo_i = max(int(torch.floor(lo[i]).item()), 0)
            hi_i = min(int(torch.ceil(hi[i]).item()) + 1, self.shape[i])
            if lo_i >= hi_i:
                return None
            slices.append(slice(lo_i, hi_i))
        slices_t = (slices[0], slices[1], slices[2])
        return self.voxel_grid[slices_t], slices_t

    def _sample_scalar(self, low: float, high: float) -> torch.Tensor:
        """Sample a single scalar uniformly from ``[low, high)``.

        Returns:
            0-dim tensor on `self.device`.
        """
        return sample_uniform((), low, high, self.device, self.generator)

    def _random_point(self) -> torch.Tensor:
        """Sample a random point uniformly inside the volume bounds.

        Returns:
            Tensor of shape ``(3,)`` with coordinates in voxel units.
        """
        return torch.stack(
            [
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[i] - 1)
                for i in range(3)
            ]
        )

    def _random_rotation(self) -> torch.Tensor:
        """Sample a random proper rotation matrix via QR decomposition of a random
        normal matrix.

        Returns:
            Tensor of shape ``(3, 3)`` with determinant +1.
        """
        rand_mat = torch.randn(3, 3, device=self.device, generator=self.generator)
        Q, _ = torch.linalg.qr(rand_mat)
        if torch.det(Q) < 0:
            Q = -Q
        return Q

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
                - Number of voxels that transitioned from unoccupied (0) to occupied
                  as a result of this operation. When `allow_overlap` is True, an
                  object may cover voxels already claimed by an earlier object;
                  those voxels are relabeled but do not count towards this total.
        """
        # Generate object mask. If failed, do nothing.
        mask = self._mask_object()
        if mask is None:
            return volume, 0
        if not self.allow_overlap:
            mask = mask & (volume == 0)
        new_voxels = int(torch.count_nonzero(mask & (volume == 0)).item())
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

    @log_call
    @to_device
    def __call__(
        self, data: torch.Tensor, fname: Path | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """Move data to the target device, generate data via `forward`, and optionally
        save the result to disk in ZARR or NIfTI format.

        Args:
            data: Input tensor passed to `forward`.
            fname: File name for saving data to disk in ZARR or NIfTI format.
            **kwargs: Additional keyword arguments passed to `forward`.

        Returns:
            Generated tensor, of the shape produced by `forward`.
        """
        result = self.forward(data, **kwargs)

        # save to disk
        if fname is not None and fname.suffix == ".zarr":
            save_zarr(fname, result.cpu().numpy(), self.attrs)
        elif fname is not None and fname.suffix == ".nii":
            save_nifti(fname, result.cpu().numpy(), permute=True)

        return result

    @property
    def attrs(self) -> dict[str, Any]:
        """Common metadata for ZARR/NIfTI storage."""
        return zarr_attributes(self.__class__.__name__, self.device, self.seed)

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Generate data.

        Note:
            Call the instance (``model(data)``) instead of ``model.forward(data)``. In
            this implementation, ``__call__`` transfers inputs to ``self.device``
            before invoking ``forward``.
        """
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

    @log_call
    @to_device
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Move data to the target device and apply the transform.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.

        Returns:
            Transformed tensor of the same shape as `data`.
        """
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Generates transformation and applies to input tensor.

        Note:
            Call the instance (``transform(data)``) instead of
            ``transform.forward(data)``. In this implementation, ``__call__``
            transfers inputs to ``self.device`` before invoking ``forward``.
        """
        transformation = self.sample(data.shape)
        return self.apply(data, transformation)

    @classmethod
    @abstractmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs a Transform instance from a configuration object."""
        ...
