"""Optional spline-based vessel geometry generator.

Unlinke the label maps found in `synthbold.geometies`, this module is a thin wrapper of
the spline-based vessel geometry generator `synthspline`
(https://github.com/balbasty/synthspline). The module is intentionally kept separate so
that `import synthbold` does not require `synthspline` to be installed; the dependency
is only imported lazily, at the point a `SplineVessels` instance is actually
constructed. Install `synthspline` with
``pip install git+https://github.com/haenelt/synthspline.git@main``.
"""

from typing import Any, Self

import torch

from synthbold.base import BaseGeometry
from synthbold.config import Config

__all__ = ["SplineVessels"]


class SplineVessels(BaseGeometry):
    """Random vessel-tree mask generator based on `synthspline`.

    Generates a single hierarchical, branching vessel tree per sample using
    `synthspline`'s `SynthSplineBlock`, as an alternative to the straight-cylinder and
    cylinder-tree geometries in `synthbold.geometries`. Unlike those, tree topology
    and shape parameters (density, tortuosity, radius, branching) are drawn from
    log-normal distributions defined by `synthspline` itself rather than sampled here.

    Requires the optional `synthspline` package; install with ``pip install
    git+https://github.com/haenelt/synthspline.git@main``. The dependency is only
    imported when this class is instantiated, so importing `synthbold` never requires
    it.

    Args:
        shape: Matrix size of the spline label map ``(X, Y, Z)``.
        fov: Field of view in mm, used to compute voxel size.
        nb_levels: Number of hierarchical levels in the tree.
        tree_density: ``(mean, std)`` of the log-normal distribution of trees/mm^3.
        tortuosity: ``(mean, std)`` of the log-normal distribution of tortuosity
            (cord / length).
        radius: ``(mean, std)`` of the log-normal distribution of the root radius in mm.
        radius_change: ``(mean, std)`` of the log-normal distribution of radius
            variation along a spline.
        nb_children: ``(mean, std)`` of the log-normal distribution of the number of
            branches per spline.
        radius_ratio: ``(mean, std)`` of the log-normal distribution of the child/parent
            radius ratio.
        device: Target compute device, e.g. 'cuda' or 'cpu'.
        seed: Random seed for reproducibility.

    Notes:
        `synthspline` draws from PyTorch's global RNG rather than an explicit
        `torch.Generator`, so this class seeds `torch.manual_seed` once at construction
        time (if `seed` is given) as a best effort for reproducibility. Unlike the other
        geometries in this package, samples are therefore not reproducible if other
        unseeded PyTorch random calls are interleaved on the same process between
        samples.

    Raises:
        ImportError: If `synthspline` is not installed.

    Examples:
        Create 100 random spline vessel-tree label maps and save to disk:

        >>> shape = (128, 128, 128)
        >>> fov = (32.0, 32.0, 32.0)
        >>> fname = "<fname.zarr>"
        >>> vessel_map = SplineVessels(shape, fov)
        >>> vessel_map(100, fname)
    """

    def __init__(
        self,
        shape: tuple[int, int, int] = (128, 128, 128),
        fov: tuple[float, float, float] = (32.0, 32.0, 32.0),
        nb_levels: int = 2,
        tree_density: tuple[float, float] = (0.01, 0.01),
        tortuosity: tuple[float, float] = (5.0, 3.0),
        radius: tuple[float, float] = (0.1, 0.02),
        radius_change: tuple[float, float] = (1.0, 0.1),
        nb_children: tuple[float, float] = (5.0, 5.0),
        radius_ratio: tuple[float, float] = (0.7, 0.1),
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(shape, seed=seed, device=device)
        self.fov = fov
        self.nb_levels = nb_levels
        self.tree_density = tree_density
        self.tortuosity = tortuosity
        self.radius = radius
        self.radius_change = radius_change
        self.nb_children = nb_children
        self.radius_ratio = radius_ratio

        if seed is not None:
            torch.manual_seed(seed)
        self._block = self._build_block()

    def _build_block(self) -> Any:
        """Lazily imports `synthspline` and constructs the underlying
        `SynthSplineBlock`.

        Returns:
            A `synthspline.labelsynth.SynthSplineBlock` instance configured from
            this geometry's parameters.

        Raises:
            ImportError: If `synthspline` is not installed.
        """
        try:
            from synthspline.labelsynth import SynthSplineBlock
            from synthspline.random import LogNormal
        except ImportError as exc:
            raise ImportError(
                "SplineVessels requires the optional `synthspline` package. Install "
                "it with `pip install git+https://github.com/haenelt/synthspline.git"
                "@main`."
            ) from exc

        voxel_size = sum(f / n for f, n in zip(self.fov, self.shape, strict=True)) / 3

        return SynthSplineBlock(
            shape=list(self.shape),
            voxel_size=voxel_size,
            nb_levels=self.nb_levels,
            tree_density=LogNormal(mean=self.tree_density[0], std=self.tree_density[1]),
            tortuosity=LogNormal(mean=self.tortuosity[0], std=self.tortuosity[1]),
            radius=LogNormal(mean=self.radius[0], std=self.radius[1]),
            radius_change=LogNormal(
                mean=self.radius_change[0], std=self.radius_change[1]
            ),
            nb_children=LogNormal(mean=self.nb_children[0], std=self.nb_children[1]),
            radius_ratio=LogNormal(mean=self.radius_ratio[0], std=self.radius_ratio[1]),
            device=self.device,
        )

    @property
    def attrs(self) -> dict[str, Any]:
        """Metadata for ZARR/NIfTI storage."""
        return {
            **super().attrs,
            "fov": self.fov,
            "nb_levels": self.nb_levels,
            "tree_density": self.tree_density,
            "tortuosity": self.tortuosity,
            "radius": self.radius,
            "radius_change": self.radius_change,
            "nb_children": self.nb_children,
            "radius_ratio": self.radius_ratio,
        }

    def forward(self) -> torch.Tensor:
        """Generate a single random spline vessel-tree label map."""
        sample = self._block(batch=1)
        return sample.labels[0, 0].to(self.device)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs spline vessel-tree label map instance from configuration
        object."""
        tree_density_params = (
            config.geom.spline_tree_density.mean,
            config.geom.spline_tree_density.std,
        )
        tortuosity_params = (
            config.geom.spline_tortuosity.mean,
            config.geom.spline_tortuosity.std,
        )
        radius_params = (config.geom.spline_radius.mean, config.geom.spline_radius.std)
        radius_change_params = (
            config.geom.spline_radius_change.mean,
            config.geom.spline_radius_change.std,
        )
        nb_children_params = (
            config.geom.spline_nb_children.mean,
            config.geom.spline_nb_children.std,
        )
        radius_ratio_params = (
            config.geom.spline_radius_ratio.mean,
            config.geom.spline_radius_ratio.std,
        )
        return cls(
            shape=config.geom.input_shape,
            fov=config.geom.fov,
            nb_levels=config.geom.spline_nb_levels,
            tree_density=tree_density_params,
            tortuosity=tortuosity_params,
            radius=radius_params,
            radius_change=radius_change_params,
            nb_children=nb_children_params,
            radius_ratio=radius_ratio_params,
            device=config.device,
            seed=config.seed,
        )
