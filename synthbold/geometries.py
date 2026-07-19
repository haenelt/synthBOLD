"""Definitions of raw geometry labels (e.g., Shapes, Cylinders, Cubes, etc.) that are
used as inputs for data synthesis."""

from typing import Any, Self

import torch
import torch.nn.functional as F

from synthbold.base import BaseGeometry, ObjectGeometry
from synthbold.config import Config
from synthbold.transforms.functional import apply_deformation

__all__ = ["Shapes", "Cylinders", "Spheres", "Tetrahedra", "Cubes", "Toroids"]


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
            data: Noise image of shape ``(N, X, Y, Z)``.

        Returns:
            Deformed noise images.
        """
        N = data.shape[0]
        data = data.to(self.device)

        # Random smooth displacement field per noise image, in normalized [-1, 1]
        # coordinate units
        disp = torch.randn(
            N,
            3,
            *self.displacement_shape,
            device=self.device,
            generator=self.generator,
        )
        disp = F.interpolate(
            disp, size=self.shape, mode="trilinear", align_corners=True
        )
        disp = disp.permute(0, 2, 3, 4, 1)  # (N, X, Y, Z, 3)

        # Scale displacement and convert from normalized coordinates to voxel units,
        # as expected by apply_deformation
        voxel_scale = torch.tensor(
            [s / 2 for s in self.shape], device=self.device, dtype=disp.dtype
        )
        disp = disp * self.scale * voxel_scale

        return apply_deformation(data, disp, interp="nearest")

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


class Cylinders(ObjectGeometry):
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
        num_objects: Fixed number of cylinders to generate. Mutually exclusive with
            `vf_range`.
        vf_range: Range of target volume fractions. Cylinders are added until the
            volume fraction is reached. Mutually exclusive with `num_objects`.
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
    """

    def _mask_object(self) -> torch.Tensor | None:
        """Generate a binary mask of a randomly oriented cylinder. The cylinder is
        defined by (1) a random axis direction, (2) a random point inside the volume,
        and (3) a random radius. The cylinder is extended along its axis to span the
        full volume by intersecting an infinite line with the axis-aligned bounding
        box. The resulting segment defines the finite cylinder.

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

        Using a slab-based intersection algorithm, this method computes the parameter
        range `[t_min, t_max]` for which the line lies inside the box defined by the
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
            config.geom.cylinder_diameter.min,
            config.geom.cylinder_diameter.max,
        )
        vf_range = (
            None
            if config.geom.cylinder_num is not None
            else (config.geom.cylinder_vf.min, config.geom.cylinder_vf.max)
        )
        return cls(
            shape=config.geom.input_shape,
            fov=config.geom.fov,
            num_objects=config.geom.cylinder_num,
            vf_range=vf_range,
            diameter_range=diameter_range,
            allow_overlap=config.geom.cylinder_allow_overlap,
            seed=config.seed,
            device=config.device,
        )


class Spheres(ObjectGeometry):
    """Random sphere mask generator.

    Generates random spherical label masks. Follows the same interface as
    :class:`Cylinders`; see that class for a full description of the shared
    parameters.
    """

    def _mask_object(self) -> torch.Tensor:
        """Generate a binary mask of a randomly placed sphere. The sphere center is
        drawn uniformly inside the volume bounds and the radius is sampled uniformly
        from ``radius_range``.

        Returns:
            Boolean tensor of shape ``shape`` where ``True`` indicates voxels
            inside the sphere.
        """
        # Random center inside volume
        center = torch.stack(
            [
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[0] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[1] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[2] - 1),
            ]
        )

        # Random radius
        radius = (
            torch.rand((), device=self.device, generator=self.generator)
            * (self.radius_range[1] - self.radius_range[0])
            + self.radius_range[0]
        )

        dist = torch.norm(self.voxel_grid - center, dim=-1)
        return dist <= radius

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs sphere label map instance from configuration object."""
        diameter_range = (
            config.geom.sphere_diameter.min,
            config.geom.sphere_diameter.max,
        )
        vf_range = (
            None
            if config.geom.sphere_num is not None
            else (config.geom.sphere_vf.min, config.geom.sphere_vf.max)
        )
        return cls(
            shape=config.geom.input_shape,
            fov=config.geom.fov,
            num_objects=config.geom.sphere_num,
            vf_range=vf_range,
            diameter_range=diameter_range,
            allow_overlap=config.geom.sphere_allow_overlap,
            seed=config.seed,
            device=config.device,
        )


class Tetrahedra(ObjectGeometry):
    """Random tetrahedron mask generator.

    Generates random tetrahedral label masks. Each tetrahedron is a regular tetrahedron
    placed at a random position in the volume with a random orientation and a random
    circumradius drawn uniformly from ``diameter_range``. Volume fraction and diameters
    are randomly chosen from a uniform distribution with set boundaries. Optionally,
    instead of using volume fraction, the number of tetrahedra can be set directly.

    Attributes:
        shape: Matrix size of the tetrahedron label map ``(X, Y, Z)``.
        fov: Field of view in mm, used to compute voxel size.
        num_objects: Fixed number of tetrahedra to generate. Mutually exclusive with
            `vf_range`.
        vf_range: Range of target volume fractions. Tetrahedra are added until the
            volume fraction is reached. Mutually exclusive with `num_objects`.
        diameter_range: Range of tetrahedron circumdiameters in mm.
        allow_overlap: If False, tetrahedra are only added if they contribute new
            voxels.
        device: Target compute device, e.g. 'cuda' or 'cpu'.
        seed: Random seed for reproducibility.

    Examples:
        Create 100 random tetrahedron label maps and save to disk:

        >>> shape = (128, 128, 128)
        >>> fov = (32.0, 32.0, 32.0)
        >>> num_tetrahedra = 10
        >>> fname = "<fname-zarr>"
        >>> tetra_map = Tetrahedra(shape, fov, num_tetrahedra)
        >>> tetra_map(100, fname)
    """

    # Canonical regular tetrahedron vertices with circumradius = 1.
    # The four vertices are (±1, ±1, ±1)/√3 with an even number of minus signs.
    _CANONICAL_VERTICES: torch.Tensor = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    ) / (3.0**0.5)

    def _mask_object(self) -> torch.Tensor:
        """Generate a binary mask of a randomly placed and oriented regular tetrahedron.

        A random circumradius is sampled from ``radius_range``, a random rotation is
        drawn via QR decomposition, and the tetrahedron is placed at a random center
        inside the volume. Voxel membership is decided by checking all four half-space
        constraints (one per face).

        Returns:
            Boolean tensor of shape ``shape`` where ``True`` marks voxels inside the
            tetrahedron.
        """
        # Random center inside volume
        center = torch.stack(
            [
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[0] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[1] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[2] - 1),
            ]
        )

        # Random circumradius
        radius = (
            torch.rand((), device=self.device, generator=self.generator)
            * (self.radius_range[1] - self.radius_range[0])
            + self.radius_range[0]
        )

        # Random rotation via QR decomposition of a random normal matrix
        rand_mat = torch.randn(3, 3, device=self.device, generator=self.generator)
        Q, _ = torch.linalg.qr(rand_mat)
        # Ensure det = +1 (proper rotation, not a reflection)
        if torch.det(Q) < 0:
            Q = -Q

        # Rotate, scale, and translate canonical vertices
        verts = self._CANONICAL_VERTICES.to(self.device)  # (4, 3), circumradius = 1
        verts = radius * (verts @ Q.T) + center  # (4, 3)

        # Half-space test: a voxel is inside the tetrahedron if it lies on the inward
        # side of all four face planes.
        pts = self.voxel_grid  # (X, Y, Z, 3)
        inside = torch.ones(self.shape, device=self.device, dtype=torch.bool)

        faces = [
            (verts[1], verts[2], verts[3], verts[0]),
            (verts[0], verts[2], verts[3], verts[1]),
            (verts[0], verts[1], verts[3], verts[2]),
            (verts[0], verts[1], verts[2], verts[3]),
        ]
        for a, b, c, opposite in faces:
            n = torch.linalg.cross(b - a, c - a)
            if torch.dot(n, opposite - a) < 0:
                n = -n
            sd = torch.tensordot(pts - a, n, dims=([3], [0]))
            inside = inside & (sd >= 0)

        return inside

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs tetrahedron label map instance from configuration object."""
        diameter_range = (
            config.geom.tetrahedron_diameter.min,
            config.geom.tetrahedron_diameter.max,
        )
        vf_range = (
            None
            if config.geom.tetrahedron_num is not None
            else (config.geom.tetrahedron_vf.min, config.geom.tetrahedron_vf.max)
        )
        return cls(
            shape=config.geom.input_shape,
            fov=config.geom.fov,
            num_objects=config.geom.tetrahedron_num,
            vf_range=vf_range,
            diameter_range=diameter_range,
            allow_overlap=config.geom.tetrahedron_allow_overlap,
            seed=config.seed,
            device=config.device,
        )


class Cubes(ObjectGeometry):
    """Random cube mask generator.

    Generates random cube label masks with set volume fraction. Each cube is a regular
    cube placed at a random position in the volume with a random orientation and a
    random circumradius drawn uniformly from ``diameter_range``. Volume fraction and
    diameters are randomly chosen from a uniform distribution with set boundaries.
    Optionally, instead of using volume fraction, the number of cubes can be set
    directly.

    Attributes:
        shape: Matrix size of the cube label map ``(X, Y, Z)``.
        fov: Field of view in mm, used to compute voxel size.
        num_objects: Fixed number of cubes to generate. Mutually exclusive with
            `vf_range`.
        vf_range: Range of target volume fractions. Cubes are added until the
            volume fraction is reached. Mutually exclusive with `num_objects`.
        diameter_range: Range of cube circumdiameters in mm.
        allow_overlap: If False, cubes are only added if they contribute new voxels.
        device: Target compute device, e.g. 'cuda' or 'cpu'.
        seed: Random seed for reproducibility.

    Notes:
        The ``diameter_range`` is interpreted as the circumdiameter (diameter of the
        circumscribed sphere), consistent with the ``Tetrahedra`` convention.

    Examples:
        Create 100 random cube label maps and save to disk:

        >>> shape = (128, 128, 128)
        >>> fov = (32.0, 32.0, 32.0)
        >>> num_cubes = 10
        >>> fname = "<fname-zarr>"
        >>> cube_map = Cubes(shape, fov, num_cubes)
        >>> cube_map(100, fname)
    """

    def _mask_object(self) -> torch.Tensor:
        """Generate a binary mask of a randomly placed and oriented cube.

        A random circumradius is sampled from ``radius_range``, a random rotation is
        drawn via QR decomposition, and the cube is placed at a random center inside
        the volume. Voxel membership is decided by transforming coordinates into the
        cube's local frame and checking the L∞ norm against the half-edge length.

        Returns:
            Boolean tensor of shape ``shape`` where ``True`` marks voxels inside the
            cube.
        """
        # Random center inside volume
        center = torch.stack(
            [
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[0] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[1] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[2] - 1),
            ]
        )

        # Random circumradius
        radius = (
            torch.rand((), device=self.device, generator=self.generator)
            * (self.radius_range[1] - self.radius_range[0])
            + self.radius_range[0]
        )

        # Random rotation via QR decomposition of a random normal matrix
        rand_mat = torch.randn(3, 3, device=self.device, generator=self.generator)
        Q, _ = torch.linalg.qr(rand_mat)
        if torch.det(Q) < 0:
            Q = -Q

        # Circumradius of a unit cube (half-edge = 1) is √3, so half-edge = r/√3
        half_edge = radius / (3.0**0.5)

        # Rotate voxel coords into cube frame: p' = Q^T @ (p - center)
        # In row-vector convention: (p - center) @ Q achieves the same result.
        pts_local = (self.voxel_grid - center) @ Q  # (X, Y, Z, 3)
        return pts_local.abs().amax(dim=-1) <= half_edge

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs cube label map instance from configuration object."""
        diameter_range = (
            config.geom.cube_diameter.min,
            config.geom.cube_diameter.max,
        )
        vf_range = (
            None
            if config.geom.cube_num is not None
            else (config.geom.cube_vf.min, config.geom.cube_vf.max)
        )
        return cls(
            shape=config.geom.input_shape,
            fov=config.geom.fov,
            num_objects=config.geom.cube_num,
            vf_range=vf_range,
            diameter_range=diameter_range,
            allow_overlap=config.geom.cube_allow_overlap,
            seed=config.seed,
            device=config.device,
        )


class Toroids(ObjectGeometry):
    """Random toroid mask generator.

    Generates random toroidal label masks with set volume fraction. Each torus is placed
    at a random position in the volume with a random orientation. The major radius (R,
    distance from the torus center to the tube center) is drawn uniformly from
    ``diameter_range`` after converting from mm to voxels. The tube radius (r) is set
    as a fraction of the major radius sampled uniformly from ``tube_ratio_range``.

    Attributes:
        shape: Matrix size of the toroid label map ``(X, Y, Z)``.
        fov: Field of view in mm, used to compute voxel size.
        num_objects: Fixed number of toroids to generate. Mutually exclusive with
            ``vf_range``.
        vf_range: Range of target volume fractions. Toroids are added until the volume
            fraction is reached. Mutually exclusive with ``num_objects``.
        diameter_range: Range of major diameters (2*R) in mm.
        tube_ratio_range: Range of r/R, where r is the tube radius. Must be in (0, 1)
            for a non-self-intersecting ring torus.
        allow_overlap: If False, toroids are only added if they contribute new voxels.
        device: Target compute device, e.g. 'cuda' or 'cpu'.
        seed: Random seed for reproducibility.

    Examples:
        Create 100 random toroid label maps and save to disk:

        >>> shape = (128, 128, 128)
        >>> fov = (32.0, 32.0, 32.0)
        >>> num_toroids = 5
        >>> fname = "<fname-zarr>"
        >>> toroid_map = Toroids(shape, fov, num_toroids)
        >>> toroid_map(100, fname)
    """

    def __init__(
        self,
        shape: tuple[int, int, int] = (128, 128, 128),
        fov: tuple[float, float, float] = (1.0, 1.0, 1.0),
        num_objects: int | None = None,
        vf_range: tuple[float, float] | None = (0.05, 0.1),
        diameter_range: tuple[float, float] = (2.0, 10.0),
        tube_ratio_range: tuple[float, float] = (0.1, 0.4),
        allow_overlap: bool = True,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            shape,
            fov,
            num_objects,
            vf_range,
            diameter_range,
            allow_overlap,
            device,
            seed,
        )
        self.tube_ratio_range = tube_ratio_range

    @property
    def attrs(self) -> dict[str, Any]:
        """Metadata for ZARR/NIfTI storage."""
        return {
            **super().attrs,
            "tube_ratio_range": self.tube_ratio_range,
        }

    def _mask_object(self) -> torch.Tensor:
        """Generate a binary mask of a randomly placed and oriented torus.

        A random major radius R is sampled from ``radius_range``, a random tube radius
        r is derived by multiplying R by a value from ``tube_ratio_range``, and a random
        rotation is drawn via QR decomposition. The torus symmetry axis is the local
        z-axis. Voxel membership uses the standard torus equation
        ``(sqrt(x'^2 + y'^2) - R)^2 + z'^2 <= r^2`` in local coordinates.

        Returns:
            Boolean tensor of shape ``shape`` where ``True`` marks voxels inside the
            torus.
        """
        # Random center inside volume
        center = torch.stack(
            [
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[0] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[1] - 1),
                torch.rand((), device=self.device, generator=self.generator)
                * (self.shape[2] - 1),
            ]
        )

        # Random major radius R (distance from torus center to tube center)
        major_radius = (
            torch.rand((), device=self.device, generator=self.generator)
            * (self.radius_range[1] - self.radius_range[0])
            + self.radius_range[0]
        )

        # Random tube radius r = tube_ratio * R
        tube_ratio = (
            torch.rand((), device=self.device, generator=self.generator)
            * (self.tube_ratio_range[1] - self.tube_ratio_range[0])
            + self.tube_ratio_range[0]
        )
        tube_radius = tube_ratio * major_radius

        # Random rotation via QR decomposition of a random normal matrix
        rand_mat = torch.randn(3, 3, device=self.device, generator=self.generator)
        Q, _ = torch.linalg.qr(rand_mat)
        if torch.det(Q) < 0:
            Q = -Q

        # Transform voxel coords into torus local frame; symmetry axis is local z
        pts_local = (self.voxel_grid - center) @ Q  # (X, Y, Z, 3)

        # Torus equation: (sqrt(x'^2 + y'^2) - R)^2 + z'^2 <= r^2
        xy_dist = torch.sqrt(pts_local[..., 0] ** 2 + pts_local[..., 1] ** 2)
        dist_sq = (xy_dist - major_radius) ** 2 + pts_local[..., 2] ** 2
        return dist_sq <= tube_radius**2

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs toroid label map instance from configuration object."""
        diameter_range = (
            config.geom.toroid_diameter.min,
            config.geom.toroid_diameter.max,
        )
        tube_ratio_range = (
            config.geom.toroid_tube_ratio.min,
            config.geom.toroid_tube_ratio.max,
        )
        vf_range = (
            None
            if config.geom.toroid_num is not None
            else (config.geom.toroid_vf.min, config.geom.toroid_vf.max)
        )
        return cls(
            shape=config.geom.input_shape,
            fov=config.geom.fov,
            num_objects=config.geom.toroid_num,
            vf_range=vf_range,
            diameter_range=diameter_range,
            tube_ratio_range=tube_ratio_range,
            allow_overlap=config.geom.toroid_allow_overlap,
            seed=config.seed,
            device=config.device,
        )
