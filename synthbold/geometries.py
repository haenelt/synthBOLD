"""Definitions of raw geometry labels (e.g., Shapes, Cylinders, Squares, etc.) that are
used as inputs for data synthesis."""

from typing import Any, Self

import torch
import torch.nn.functional as F

from synthbold.base import BaseGeometry
from synthbold.config import Config

__all__ = ["Shapes", "Cylinders", "Ellipsoids", "Triangles", "Squares", "Toroids"]


class Shapes(BaseGeometry):
    """Random shape mask generator.

    Generates masks with `J` input labels of random generic shapes. Maps are created by
    first drawing `J` smoothly varying noise images by sampling voxels from a standard
    distribution at lower resolution and upsampling to full size. Second, each image is
    warped with a random smooth deformation field. Third, we create an input mask by
    assigning, for each voxel, the label corresponding to the image that has the
    highest intensity (Hoffmann et al., 2021; Hoffmann et al., 2023).

    These masks can be used to define random background tissue signal for synthetic
    BOLD data generation.

    Attributes:
        shape: Matrix size of the target shape mask ``(X, Y, Z)``.
        J: Number of distinct labels.
        displacement_shape: Matrix size of the low-resolution displacement field.
        scale: Scale factor of the low-resolution displacement field.
        device: Target compute device, e.g. 'cuda' or 'cpu'.
        seed: Random seed for reproducibility.

    Examples:
        Create 100 random maps and save to disk:

        >>> J = 26
        >>> shape = (32, 32, 32)
        >>> fname = "<fname.zarr>"
        >>> shape_mask = Shapes(shape, J)
        >>> shape_mask(100, fname)

    References:
        Hoffmann, M. et al. (2021). *Learning MRI contrast-agnostic registration*. Proc
        IEEE Int Symp Biomed Imaging.

        Hoffmann, M. et al. (2022). *SynthMorph: Learning contrast-invariant
        registration without acquired images*. IEEE Trans Med Imaging.
    """

    def __init__(
        self,
        shape: tuple[int, int, int] = (128, 128, 128),
        J: int = 10,
        displacement_shape: tuple[int, int, int] = (2, 2, 2),
        scale: float = 0.2,
        *,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(shape, seed=seed, device=device)

        self.J = J
        self.displacement_shape = displacement_shape
        self.scale = scale

    @property
    def attrs(self) -> dict[str, Any]:
        """Metadata for ZARR/NIfTI storage."""
        return {
            **super().attrs,
            "description": "Tissue labels",
            "J": self.J,
            "lowres_shape": self.displacement_shape,
            "scale": self.scale,
        }

    def forward(self) -> torch.Tensor:
        """Generate random shape mask."""
        data = torch.randn(
            self.J,
            1,
            *self.displacement_shape,
            device=self.device,
            generator=self.generator,
        )
        data = F.interpolate(
            data, size=self.shape, mode="trilinear", align_corners=False
        )
        data = data.squeeze(1)
        data = self._warp(data)
        data = torch.argmax(data, dim=0)
        data += 1

        return data

    def _warp(self, data: torch.Tensor) -> torch.Tensor:
        """Creates a low-dimensional random displacement map that is then upsampled to
        transform input noise images.

        Args:
            data: Noise iamge of shape ``(N, X, Y, Z)``.

        Returns:
            Deformed noise images.
        """
        warped_pj = []
        data = data.to(self.device)
        for j in range(data.shape[0]):
            # Random smooth displacement
            disp = torch.randn(
                1,
                3,
                *self.displacement_shape,
                device=self.device,
                generator=self.generator,
            )
            disp = F.interpolate(
                disp, size=self.shape, mode="trilinear", align_corners=True
            )
            disp = disp.squeeze(0).permute(1, 2, 3, 0)  # [X, Y, Z, 3]
            # Scale displacement
            disp = disp * self.scale  # adjust amplitude

            # Warped image
            # F.grid_sample expects grid channels in (W, H, D) order, i.e. reversed
            # relative to the (X, Y, Z) convention used by normalized_grid.
            grid = (self.normalized_grid + disp)[..., [2, 1, 0]]
            warped = F.grid_sample(
                data[j].unsqueeze(0).unsqueeze(0),  # (1, 1, X, Y, Z),
                grid.unsqueeze(0),  # (1, X, Y, Z, 3)
                mode="nearest",
                padding_mode="reflection",
                align_corners=True,
            )
            warped_pj.append(warped.squeeze(0).squeeze(0))

        return torch.stack(warped_pj, dim=0)  # (J, X, Y, Z)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs shape mask instance from configuration object."""
        return cls(
            shape=config.geom.input_shape,
            J=config.physio.label_J,
            displacement_shape=config.physio.label_displacement_shape,
            scale=config.physio.label_scale,
            device=config.device,
            seed=config.seed,
        )


class Cylinders(BaseGeometry):
    """Random cylinder mask generator.

    Generates random cylindrical label masks with set volume fraction. Volume fraction
    and diameters are randomly chosen from a uniform distribution with set boundaries.
    Optionally, instead of using volume fraction, the number of cylinders can be set
    directly.

    These masks can be used to define random vessel distributions with set blood volume
    fraction and vessel diameters.

    Attributes:
        shape: Matrix size of the cylinder label map ``(X, Y, Z)``.
        fov: Field of view in mm, used to compute voxel size.
        num_cylinders: Fixed number of cylinders to generate. Mutually exclusive with
            `vf_range`.
        vf_range: Range of target volume fractions. Cylinders are added until the
            volume fraction is reached. Mutually exclusive with `num_cylinders`.
        diameter_range: Range of cylinder diameters in mm.
        allow_overlap: If False, cylinders are only added if they contribute new voxels.
        device: Target compute device, e.g. 'cuda' or 'cpu'.
        seed: Random seed for reproducibility.

    Examples:
        Create 100 random cylinder label maps and save to disk:

        >>> shape = (128, 128, 128)
        >>> fov = (32.0, 32.0, 32.0)
        >>> num_cylinders = 10
        >>> fname = "<fname-zarr>"
        >>> cylinder_map = Cylinders(output_shape, fov, num_cylinders)
        >>> cylinder_map(100, fname)

    Notes:
        `num_cylinders` and `vf_range` are mutually exclusive. If `num_cylinders` is
        set, cylinders are added iteratively until the specified number is reached. If
        `vf_range` is set, a target volume fraction is randomly sampled from the
        specified range and cylinders are added iteratively until the target fraction
        of occupied voxels is reached.
    """

    def __init__(
        self,
        shape: tuple[int, int, int] = (128, 128, 128),
        fov: tuple[float, float, float] = (1.0, 1.0, 1.0),
        num_cylinders: int | None = None,
        vf_range: tuple[float, float] | None = (0.05, 0.1),
        diameter_range: tuple[float, float] = (0.05, 2.0),
        allow_overlap: bool = True,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        super().__init__(shape, seed=seed, device=device)
        self.shape = shape
        self.fov = fov
        self.num_cylinders = num_cylinders
        self.vf_range = vf_range
        self.diameter_range = diameter_range
        self.allow_overlap = allow_overlap

        # num_cylinders and vf_range are mutually exclusive
        if (self.num_cylinders is None) == (self.vf_range is None):
            raise ValueError("Specify exactly one of num_cylinders or vf_range.")

        # Convert diameter_range in mm to radius_range in voxel dimension.
        # We use the mean voxel size across dimensions for conversion.
        voxel_size = tuple(f / n for f, n in zip(self.fov, self.shape, strict=True))
        voxel_size_mean = sum(voxel_size) / 3
        self.radius_range = (
            self.diameter_range[0] / 2 / voxel_size_mean,
            self.diameter_range[1] / 2 / voxel_size_mean,
        )

    @property
    def attrs(self) -> dict[str, Any]:
        """Metadata for ZARR/NIfTI storage."""
        return {
            **super().attrs,
            "description": "Vessel maps",
            "fov": self.fov,
            "num_cylinders": self.num_cylinders,
            "vf_range": self.vf_range,
            "diameter_range": self.diameter_range,
            "allow_overlap": self.allow_overlap,
        }

    def forward(self) -> torch.Tensor:
        """Generate random cylinder label maps. Each cylinder is assigned a unique
        integer label (>0), while background voxels remain 0.

        Two generation modes are supported:
            1. Fixed number of cylinders (``self.num_cylinders``)
            2. Target volume fraction (``self.vf_range``)

        If `self.vf_range` is set, cylinders are added iteratively until the fraction
        of occupied voxels reaches the target volume fraction.

        Returns:
            3D tensor of shape `shape` with integer labels:
                - 0: background
                - 1..N: individual cylinders
        """
        # Initialize volume
        volume = torch.zeros(self.shape, device=self.device, dtype=torch.int32)
        total_voxels = volume.numel()

        # Mode 1: fixed number of cylinders
        if self.num_cylinders is not None:
            for i in range(self.num_cylinders):
                volume, _ = self._add_cylinder(i + 1, volume)

        # Mode 2: target volume fraction
        elif self.vf_range is not None:
            # Sample target volume fraction uniformly from range
            vf = (
                torch.rand((), device=self.device, generator=self.generator)
                * (self.vf_range[1] - self.vf_range[0])
                + self.vf_range[0]
            )
            occupied = 0
            i = 0
            max_iter = total_voxels * 10
            while occupied / total_voxels < vf and i < max_iter:
                i += 1
                volume, _ = self._add_cylinder(i, volume)
                # recompute occupied voxels
                occupied = int(torch.count_nonzero(volume).item())
        else:
            raise ValueError("Either num_cylinders or vf_range must be set.")

        return volume

    def _add_cylinder(
        self, label: int, volume: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Insert a cylinder into a labeled 3D volume. A binary cylinder mask is
        generated and written into the provided volume. Depending on the overlap
        setting, the cylinder may overwrite existing labels or only fill previously
        empty voxels.

        Args:
            label: Integer label assigned to voxel belonging to the cylinder.
            volume: 3D tensor representing the labeled volume. Background voxels are
                assumed to be 0.

        Returns:
            A tuple with the following elements:
                - Updated volume with the cylinder inserted.
                - Number of voxels newly assigned to this cylinder (i.e., voxels
                  written during this operation).
        """
        # Generate cylinder. If failed, do nothing.
        mask = self._mask_cylinder()
        if mask is None:
            return volume, 0

        if not self.allow_overlap:
            empty_mask = volume == 0
            mask = mask & empty_mask

        new_voxels = int(torch.count_nonzero(mask).item())
        volume[mask] = label
        return volume, new_voxels

    def _mask_cylinder(self) -> torch.Tensor | None:
        """Generate a binary mask of a randomly oriented cylinder. The cylinder is
        defined by (1) a random axis direction, (2) a random point inside the volume,
        and (3) a random radius. The cylider is extended along its axis to span the
        full volume by intersecting an infinite line with the axis-aligned boudig box.
        The resulting segmented defines the finite cylinder.

        Returns:
            Boolean tensor of shape `shape` where True indicates voxels inside the
            cylinder. Returns `None` if no valid intersection with the volume box is
            found.
        """
        # Random direction
        axis = torch.randn(3, device=self.device, generator=self.generator)
        axis = axis / torch.norm(axis)

        # Random point inside volume
        point = torch.stack(
            [
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[0] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[1] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[2] - 1),
            ]
        ).squeeze()

        # Intersect with box
        tmin, tmax = self._intersect_line_with_box(point, axis)
        if tmin is None:
            return None

        # Cylinder endpoints
        start = point + tmin * axis
        end = point + tmax * axis

        # Random radius
        radius = (
            torch.rand((), device=self.device, generator=self.generator)
            * (self.radius_range[1] - self.radius_range[0])
            + self.radius_range[0]
        )

        # Distance computation
        # Vector along cylinder axis
        vec_v = end - start
        vec_v_norm2 = torch.dot(vec_v, vec_v)
        if vec_v_norm2 < 1e-12:
            return None
        # Vector from start point to every voxel
        vec_p = self.voxel_grid - start
        # Project each voxel onto the cylinder axis
        t = torch.tensordot(vec_p, vec_v, dims=([3], [0])) / vec_v_norm2
        t = torch.clamp(t, 0.0, 1.0)
        # Compute closest point on axis for each voxel
        proj = start + t.unsqueeze(-1) * vec_v
        # Euclidean distance from voxel to axis
        dist = torch.norm(self.voxel_grid - proj, dim=-1)
        return dist <= radius

    def _intersect_line_with_box(
        self, point: torch.Tensor, direction: torch.Tensor, eps: float = 1e-8
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute intersection of infinite line with axis-aligned box. The line is
        parameterized as

            ``p(t) = point + t * direction``

        Using a slab-based insection algorithm, this method computes the parameter
        range `[t_min, t_max]` for which the line les inside the box defined by the
        volume bounds.

        Args:
            point: Origin of the line of shape ``(3,)``.
            direction: Line direction vector of shape ``(3,)``.
            eps: Small threshold to handle near-parallel cases.

        Returns:
            `(t_min, t_max)` defining the valid intersection interval. Returns
            `(None, None)` if not intersection exists.
        """
        dtype = point.dtype
        bounds_min = torch.zeros(3, device=self.device, dtype=dtype)
        bounds_max = torch.tensor(self.shape, device=self.device, dtype=dtype) - 1
        t_min = torch.tensor(-float("inf"), device=self.device, dtype=dtype)
        t_max = torch.tensor(float("inf"), device=self.device, dtype=dtype)

        for i in range(3):
            if torch.abs(direction[i]) < eps:
                # Line is parallel to slab
                if point[i] < bounds_min[i] or point[i] > bounds_max[i]:
                    return None, None
            else:
                t1 = (bounds_min[i] - point[i]) / direction[i]
                t2 = (bounds_max[i] - point[i]) / direction[i]
                t_low = torch.minimum(t1, t2)
                t_high = torch.maximum(t1, t2)

                t_min = torch.maximum(t_min, t_low)
                t_max = torch.minimum(t_max, t_high)

        return t_min, t_max

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs cylinder label map instance from configuration object."""
        diameter_range = (
            config.physio.cylinder_diameter.min,
            config.physio.cylinder_diameter.max,
        )
        vf_range = (
            None
            if config.physio.num_cylinders is not None
            else (config.physio.vf.min, config.physio.vf.max)
        )
        return cls(
            shape=config.geom.input_shape,
            fov=config.geom.fov,
            num_cylinders=config.physio.num_cylinders,
            vf_range=vf_range,
            diameter_range=diameter_range,
            allow_overlap=config.physio.allow_overlap,
            seed=config.seed,
            device=config.device,
        )


class Ellipsoids:
    pass


class Triangles:
    pass


class Squares:
    pass


class Toroids:
    pass
