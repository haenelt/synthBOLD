"""Physics-based signal generation."""

from typing import Self

import torch

from synthbold.base import Model
from synthbold.config import Config
from synthbold.decorator import accept_unbatched, require_dim
from synthbold.models.functional import dchi_to_dbz

__all__ = ["PerturbationModel", "SignalModel"]


class PerturbationModel(Model):
    """Physics-based transform that converts geometry label maps into magnetic field
    perturbation maps (ΔBz).

    Each vessel (and other geometric objects) is assigned a random susceptibility value
    and each sample is given a random B0 field orientation, from which the resulting
    magnetic field perturbation is computed using FFT convolution with a dipole kernel.

    Args:
        chi_range: Minimum and maximum susceptibility differences in ppm.
        b0: Static magnetic field strength in Tesla.
        theta_range: Minimum and maximum polar angles in radians for B0 orientations.
        phi_range: Minimum and maximum azimuthal angles in radians for B0 orientations.
        padding: Zero-padding width applied before FFT convolution. Can be either an
            integer or a tuple ``(px, py, pz)``.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        chi_range: tuple[float, float],
        b0: float,
        theta_range: tuple[float, float],
        phi_range: tuple[float, float],
        padding: int | tuple[int, int, int],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)

        self.chi_range = chi_range
        self.b0 = b0
        self.theta_range = theta_range
        self.phi_range = phi_range

        # Get padding for (x, y, z)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.padding = padding

    @require_dim(3, 4)
    @accept_unbatched(dim=3)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Compute ΔBz perturbation maps from vessel or other perturber objects.

        Args:
            data: Tensor of shape ``(B, X, Y, Z)`` containing integer vessel labels.

        Returns:
            Tensor of shape ``(B, X, Y, Z)`` containing the computed ΔBz perturbation
            maps.
        """
        data = data.to(self.device)
        B, _, _, _ = data.shape
        chi_low, chi_high = self.chi_range
        theta_low, theta_high = self.theta_range
        phi_low, phi_high = self.phi_range

        # Allocate susceptibility volume
        chi = torch.zeros_like(data, dtype=torch.float32, device=self.device)

        # Sample susceptibility values for each vessel label
        n_perturber = int(data.max().item())
        if n_perturber > 0:
            dchis = torch.empty(
                n_perturber, device=self.device, dtype=torch.float32
            ).uniform_(chi_low, chi_high, generator=self.generator)

            # Assign susceptibility values
            mask = data > 0
            chi[mask] = dchis[data[mask].long() - 1]

        # Sample random B0 orientation for each batch element
        theta = torch.empty(B, device=self.device, dtype=torch.float32).uniform_(
            theta_low, theta_high, generator=self.generator
        )
        phi = torch.empty(B, device=self.device, dtype=torch.float32).uniform_(
            phi_low, phi_high, generator=self.generator
        )

        # Compute field perturbation
        dbz = dchi_to_dbz(chi, self.padding, self.b0, theta, phi)

        return dbz

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs PerturbationModel instance from a config object."""
        chi_range = (config.physics.chi.min, config.physics.chi.max)
        theta_range = (config.physics.theta.min, config.physics.theta.max)
        phi_range = (config.physics.phi.min, config.physics.phi.max)
        return cls(
            chi_range=chi_range,
            b0=config.physics.b0,
            theta_range=theta_range,
            phi_range=phi_range,
            padding=config.geom.padding,
            device=config.device,
            seed=config.seed,
        )


class SignalModel:
    pass
