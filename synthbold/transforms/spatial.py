"""Spatial transforms."""

from typing import Self

import torch
import torch.nn.functional as F

from synthbold.base import Transform
from synthbold.config import Config
from synthbold.transforms.functional import sample_lowres_noise

__all__ = [
    "RandomFlip",
    "GaussianSmoothing",
    "GaussianSharpening",
    "GibbsRinging",
    "SphericalMask",
    "DeformedSphericalMask",
]


class RandomFlip(Transform):
    """Applies spatial random flips to 3D or 4D tensors along one or multiple axes.

    Args:
        flip_prob: Probability of flipping each axis independently.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self, flip_prob: float = 0.5, device: str = "cpu", seed: int | None = None
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.flip_prob = flip_prob

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Compute 1D random boolean tensor, denoting if single spatial dimension will
        be flipped or not."""
        offset = 1 if len(shape) == 4 else 0
        n_axes = len(shape) - offset
        return (
            torch.rand(n_axes, generator=self.generator, device=self.device)
            < self.flip_prob
        )

    @staticmethod
    def apply(data: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
        """Randomly flips the tensor along spatial axes.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            flip: 1D boolean indicating if single spatial dimensions will be flipped.

        Returns:
            Tensor of the same shape with flipped axes.
        """
        offset = 1 if len(data.shape) == 4 else 0
        dims = [i + offset for i, f in enumerate(flip) if f]
        return torch.flip(data, dims=dims)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs RandomFlip instance from a config object."""
        if config.transform.flip_prob is None:
            raise ValueError("Flip parameters are not set in the config.")
        return cls(
            flip_prob=config.transform.flip_prob,
            device=config.device,
            seed=config.seed,
        )


class GaussianSmoothing(Transform):
    """Generates an Gaussian smoothing kernel and applies it to 3D ``(X, Y, Z)`` or 4D
    ``(B, X, Y, Z)`` tensors.

    This transform creates a Gaussian kernel and applies it to the 3D volume. Standard
    deviation is randomly chosen from a uniform distribution with set boundaries.
    Parameters can be set isotropic or anisotropic, i.e., with independent smoothing
    parameters per axis. For 4D tensors, a randomly generated Gaussian kernel is created
    for each 3D volume within the batch.

    Args:
        sigma_range: Minimum and maximum standard deviation for Gaussian kernel.
        kernel_size: Isotropic or anisotropic kernel size (must be odd).
        isotropic: If True, spatially isotropic smoothing will be applied.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sigma_range: tuple[float, float],
        kernel_size: int = 3,
        isotropic: bool = True,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size
        self.isotropic = isotropic

        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size value must be odd, got {self.kernel_size}.")

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Creates a batch of 3D Gaussian kernels."""
        B, _, _, _ = shape
        k = self.kernel_size

        if self.isotropic:
            sigma = self._choose_sigma(B)
            sx, sy, sz = sigma, sigma, sigma
        else:
            sx = self._choose_sigma(B)
            sy = self._choose_sigma(B)
            sz = self._choose_sigma(B)

        # Create grid centered at 0
        x = torch.arange(k, device=self.device).float() - (k - 1) / 2
        y = torch.arange(k, device=self.device).float() - (k - 1) / 2
        z = torch.arange(k, device=self.device).float() - (k - 1) / 2
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

        # Reshape to broadcast with batch size: (1, k, k, k)
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)
        zz = zz.unsqueeze(0)

        # Sigmas shape: (B, 1, 1, 1)
        sx = sx.view(B, 1, 1, 1)
        sy = sy.view(B, 1, 1, 1)
        sz = sz.view(B, 1, 1, 1)

        # Calculate Gaussian: (B, k, k, k)
        kernel = torch.exp(-0.5 * ((xx / sx) ** 2 + (yy / sy) ** 2 + (zz / sz) ** 2))

        # Mask out values outside the kernel
        mask_x = torch.abs(xx) <= (k - 1) / 2
        mask_y = torch.abs(yy) <= (k - 1) / 2
        mask_z = torch.abs(zz) <= (k - 1) / 2
        kernel = kernel * mask_x * mask_y * mask_z

        # Normalize per batch element
        kernel_sum = kernel.sum(dim=(1, 2, 3), keepdim=True)
        kernel = kernel / (kernel_sum + 1e-8)

        # Reshape for grouped conv3d: (B, 1, k, k, k)
        return kernel.unsqueeze(1)

    @staticmethod
    def apply(data: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Applies the spatial smoothing to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            kernel: Batch of 3D Gaussian kernels of same shape.

        Returns:
            Smoothed tensor of same shape.
        """
        B, _, _, _ = data.shape
        # Prepare input for grouped conv3d: (1, B, X, Y, Z)
        input_tensor = data.unsqueeze(0)
        # Padding to preserve size
        pad = tuple(k // 2 for k in kernel.shape[2:])
        # Apply convolution with groups=b
        smoothed = F.conv3d(input_tensor, kernel, padding=tuple(pad), groups=B)
        # Restore original shape: (B, X, Y, Z)
        return smoothed.squeeze(0)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs SpatialSmoothing instance from a config object."""
        if config.transform.gsmooth_sigma is None:
            raise ValueError("Spatial smoothing parameters are not set in the config.")
        sigma_range = (
            config.transform.gsmooth_sigma.min,
            config.transform.gsmooth_sigma.max,
        )
        return cls(
            sigma_range=sigma_range,
            kernel_size=config.transform.gsmooth_kernel_size,
            isotropic=config.transform.gsmooth_isotropic,
            device=config.device,
            seed=config.seed,
        )

    def _choose_sigma(self, b: int) -> torch.Tensor:
        """Randomly select sigmas within the range."""
        min_sigma, max_sigma = self.sigma_range
        # make sure both sigmas are positive
        if min_sigma <= 0 or max_sigma <= 0:
            raise ValueError(f"sigmas must be positive, got {self.sigma_range}.")
        if min_sigma > max_sigma:
            raise ValueError(
                f"Mininimum must be smaller than maximum sigma, got {self.sigma_range}."
            )

        return torch.empty(b, device=self.device).uniform_(
            min_sigma, max_sigma, generator=self.generator
        )


class GaussianSharpening(Transform):
    """Generates an unsharp-masking kernel and applies it to 3D ``(X, Y, Z)`` or 4D
    ``(B, X, Y, Z)`` tensors.

    This transform builds a Gaussian blur kernel and combines it with an identity
    kernel into a single unsharp-masking kernel, computing
    `output = (1 + amount) * data - amount * gaussian_blur(data)`. Standard deviation of
    the underlying blur and the sharpening amount are randomly chosen from uniform
    distributions with set boundaries. For 4D tensors, an independently sampled kernel
    is created for each 3D volume within the batch.

    Args:
        sigma_range: Minimum and maximum standard deviation for the underlying
            Gaussian blur kernel.
        amount_range: Minimum and maximum sharpening strength. A value of `0` leaves
            the input unchanged, while larger values increase the sharpening effect.
        kernel_size: Isotropic kernel size (must be odd).
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sigma_range: tuple[float, float],
        amount_range: tuple[float, float],
        kernel_size: int = 3,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.sigma_range = sigma_range
        self.amount_range = amount_range
        self.kernel_size = kernel_size

        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size value must be odd, got {self.kernel_size}.")

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Creates a batch of 3D unsharp-masking kernels."""
        B, _, _, _ = shape
        k = self.kernel_size

        sigma = self._choose_sigma(B).view(B, 1, 1, 1)
        amount = self._choose_amount(B).view(B, 1, 1, 1)

        # Create grid centered at 0: (1, k, k, k)
        coord = torch.arange(k, device=self.device).float() - (k - 1) / 2
        zz, yy, xx = torch.meshgrid(coord, coord, coord, indexing="ij")
        xx, yy, zz = xx.unsqueeze(0), yy.unsqueeze(0), zz.unsqueeze(0)

        # Gaussian blur kernel, normalized per batch element: (B, k, k, k)
        gaussian = torch.exp(-0.5 * (xx**2 + yy**2 + zz**2) / sigma**2)
        gaussian = gaussian / (gaussian.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)

        # Identity kernel with a single 1 at the center: (1, k, k, k)
        center = k // 2
        identity = torch.zeros((1, k, k, k), device=self.device)
        identity[0, center, center, center] = 1.0

        # Combine into a single unsharp-masking kernel: (B, k, k, k)
        kernel = (1 + amount) * identity - amount * gaussian

        # Reshape for grouped conv3d: (B, 1, k, k, k)
        return kernel.unsqueeze(1)

    @staticmethod
    def apply(data: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Applies unsharp masking to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            kernel: Batch of 3D unsharp-masking kernels of same shape.

        Returns:
            Sharpened tensor of same shape.
        """
        B, _, _, _ = data.shape
        # Prepare input for grouped conv3d: (1, B, X, Y, Z)
        input_tensor = data.unsqueeze(0)
        # Padding to preserve size
        pad = tuple(ks // 2 for ks in kernel.shape[2:])
        # Apply convolution with groups=b
        sharpened = F.conv3d(input_tensor, kernel, padding=pad, groups=B)
        # Restore original shape: (B, X, Y, Z)
        return sharpened.squeeze(0)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs GaussianSharpening instance from a config object."""
        if (
            config.transform.sharpen_sigma is None
            or config.transform.sharpen_amount is None
        ):
            raise ValueError(
                "Gaussian sharpening parameters are not set in the config."
            )
        sigma_range = (
            config.transform.sharpen_sigma.min,
            config.transform.sharpen_sigma.max,
        )
        amount_range = (
            config.transform.sharpen_amount.min,
            config.transform.sharpen_amount.max,
        )
        return cls(
            sigma_range=sigma_range,
            amount_range=amount_range,
            kernel_size=config.transform.sharpen_kernel_size,
            device=config.device,
            seed=config.seed,
        )

    def _choose_sigma(self, b: int) -> torch.Tensor:
        """Randomly select sigma within the range."""
        min_sigma, max_sigma = self.sigma_range
        if min_sigma <= 0 or max_sigma <= 0:
            raise ValueError(f"sigmas must be positive, got {self.sigma_range}.")
        if min_sigma > max_sigma:
            raise ValueError(
                f"Mininimum must be smaller than maximum sigma, got {self.sigma_range}."
            )
        return torch.empty(b, device=self.device).uniform_(
            min_sigma, max_sigma, generator=self.generator
        )

    def _choose_amount(self, b: int) -> torch.Tensor:
        """Randomly select sharpening amount within the range."""
        min_amount, max_amount = self.amount_range
        if min_amount < 0 or max_amount < 0:
            raise ValueError(f"amount must be non-negative, got {self.amount_range}.")
        return torch.empty(b, device=self.device).uniform_(
            min_amount, max_amount, generator=self.generator
        )


class GibbsRinging(Transform):
    """Applies Gibbs ringing artifacts to 3D ``(X, Y, Z)`` or 4D ``(B, X, Y, Z)``
    tensors.

    Gibbs ringing arises from the truncation of k-space data during MRI acquisition and
    produces oscillatory artifacts near sharp intensity transitions. This transform
    simulates the effect in the frequency domain: the volume is Fourier transformed, a
    spherical low-pass mask zeroes out k-space beyond a randomly chosen normalized
    cutoff radius, and the inverse Fourier transform is taken. A cutoff of `1.0` retains
    all frequencies (no ringing), while smaller values truncate more aggressively and
    produce stronger ringing. For each volume in the batch, a random cutoff is sampled
    from a uniform distribution with set boundaries.

    Args:
        cutoff_range: Minimum and maximum normalized k-space cutoff radius, within
            `(0, 1]`.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        cutoff_range: tuple[float, float],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        if not all(0.0 < c <= 1.0 for c in cutoff_range):
            raise ValueError(f"cutoff_range must be within (0, 1], got {cutoff_range}.")
        self.cutoff_range = cutoff_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes the per-volume k-space cutoff radius for the given output shape
        ``(B, X, Y, Z)``."""
        B, _, _, _ = shape
        return self._choose_cutoff(B).view(B, 1, 1, 1)

    @staticmethod
    def apply(data: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Applies Gibbs ringing to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            cutoff: Per-volume normalized k-space cutoff radius, broadcastable to the
                shape of `data`.

        Returns:
            Tensor of the same shape with Gibbs ringing artifacts applied.
        """
        *_, X, Y, Z = data.shape
        fx = torch.fft.fftfreq(X, device=data.device)
        fy = torch.fft.fftfreq(Y, device=data.device)
        fz = torch.fft.fftfreq(Z, device=data.device)
        gx, gy, gz = torch.meshgrid(fx, fy, fz, indexing="ij")

        radius = torch.sqrt(gx**2 + gy**2 + gz**2)
        radius = radius / radius.max()

        mask = (radius <= cutoff).to(data.dtype)
        spectrum = torch.fft.fftn(data, dim=(-3, -2, -1))
        truncated = torch.fft.ifftn(spectrum * mask, dim=(-3, -2, -1))
        return truncated.real

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs GibbsRinging instance from a config object."""
        cutoff_range = (
            config.transform.gibbs_cutoff.min,
            config.transform.gibbs_cutoff.max,
        )
        return cls(
            cutoff_range=cutoff_range,
            device=config.device,
            seed=config.seed,
        )

    def _choose_cutoff(self, n_vol: int) -> torch.Tensor:
        """Randomly select cutoff radius within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.cutoff_range[0], self.cutoff_range[1], generator=self.generator
        )


class SphericalMask(Transform):
    """Generates a spherical background mask and applies it to 3D or 4D tensors.

    This transform creates a spherical mask with randomized center and radius, and
    multiples it with the input tensor. The central voxel can be any voxel within the
    input tensor array. For 4D tensors, a randomly generated mask is created for each 3D
    volume within the batch.

    Args:
        radius_range: Minimum and maximum radii for spherical mask.
        sphere_prob: Probability of applying the spherical mask to each volume. Volumes
            not selected are returned unmasked (mask is all ``True``).
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        radius_range: tuple[float, float],
        sphere_prob: float = 1.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.radius_range = radius_range
        self.sphere_prob = sphere_prob

    def _sample_sphere(
        self, shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random sphere centers and radii for each volume in the batch.

        Returns:
            A tuple with the following elements:
                - Squared distance field from the sampled centers, of shape
                  ``(B, X, Y, Z)``.
                - Sampled radii of shape ``(B, 1, 1, 1)``.
                - Boolean tensor of shape ``(B, 1, 1, 1)`` indicating whether the mask
                  should be applied to each volume, based on ``sphere_prob``.
        """
        B, X, Y, Z = shape

        # Sample random radii: (B, 1, 1, 1)
        r_min, r_max = self.radius_range
        radii = torch.empty((B, 1, 1, 1), device=self.device).uniform_(
            r_min, r_max + 1, generator=self.generator
        )

        # Sample random centers: (B, 3)
        centers = torch.stack(
            [
                torch.randint(0, X, (B,), device=self.device, generator=self.generator),
                torch.randint(0, Y, (B,), device=self.device, generator=self.generator),
                torch.randint(0, Z, (B,), device=self.device, generator=self.generator),
            ],
            dim=1,
        )

        # Create coordinate grid: (1, X, Y, Z)
        xs = torch.arange(X, device=self.device).view(1, X, 1, 1)
        ys = torch.arange(Y, device=self.device).view(1, 1, Y, 1)
        zs = torch.arange(Z, device=self.device).view(1, 1, 1, Z)

        # Broadcast centers to (B, 1, 1, 1)
        cx = centers[:, 0].view(B, 1, 1, 1)
        cy = centers[:, 1].view(B, 1, 1, 1)
        cz = centers[:, 2].view(B, 1, 1, 1)

        # Squared distance field: (B, X, Y, Z)
        dist_sq = (
            (xs - cx).float() ** 2 + (ys - cy).float() ** 2 + (zs - cz).float() ** 2
        )

        # Skip masking for volumes not selected by sphere_prob
        apply = (
            torch.rand(B, device=self.device, generator=self.generator)
            < self.sphere_prob
        ).view(B, 1, 1, 1)

        return dist_sq, radii, apply

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Compute the background mask data of shape ``(B, X, Y, Z)``."""
        dist_sq, radii, apply = self._sample_sphere(shape)
        mask = dist_sq <= radii**2
        return mask | ~apply

    @staticmethod
    def apply(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies background masks to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            mask: Mask tensor of same shape.

        Returns:
            Masked tensor of same shape.
        """
        return data * mask

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs SphericalMask instance from a config object."""
        if config.transform.sphere_radius is None:
            raise ValueError("Spherical mask parameters are not set in the config.")
        radius_range = (
            config.transform.sphere_radius.min,
            config.transform.sphere_radius.max,
        )
        return cls(
            radius_range=radius_range,
            sphere_prob=config.transform.sphere_prob,
            device=config.device,
            seed=config.seed,
        )


class DeformedSphericalMask(SphericalMask):
    """Generates a deformed spherical background mask and applies it to 3D or 4D
    tensors.

    This transform builds on :class:`SphericalMask` by perturbing the sphere radius
    with smooth, low-frequency noise, so that the resulting mask boundary is an
    irregular blob rather than a perfect sphere. The noise field is drawn at low
    resolution and trilinearly upsampled to the full volume size, then normalized per
    volume to `[-1, 1]` and scaled by `deform_amplitude` before perturbing the radius.
    For 4D tensors, an independently deformed mask is created for each 3D volume within
    the batch.

    Args:
        radius_range: Minimum and maximum radii for the underlying sphere, before
            deformation.
        deform_amplitude: Maximum relative perturbation of the radius, e.g. `0.3`
            allows the local radius to vary by up to 30% in either direction.
        deform_shape: Matrix size of the low-resolution noise field controlling the
            deformation. Smaller values yield smoother, lower-frequency deformations.
        sphere_prob: Probability of applying the mask to each volume. Volumes not
            selected are returned unmasked (mask is all ``True``).
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        radius_range: tuple[float, float],
        deform_amplitude: float = 0.3,
        deform_shape: tuple[int, int, int] = (4, 4, 4),
        sphere_prob: float = 1.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(radius_range, sphere_prob, device=device, seed=seed)
        self.deform_amplitude = deform_amplitude
        self.deform_shape = deform_shape

        if self.deform_amplitude < 0:
            raise ValueError(
                f"deform_amplitude must be non-negative, got {self.deform_amplitude}."
            )

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Compute the deformed background mask data of shape ``(B, X, Y, Z)``."""
        B, X, Y, Z = shape
        dist_sq, radii, apply = self._sample_sphere(shape)

        # Smooth low-frequency noise field, normalized to [-1, 1]: (B, X, Y, Z)
        noise = sample_lowres_noise(
            B, self.deform_shape, (X, Y, Z), self.device, self.generator
        )
        noise_max = noise.flatten(1).abs().amax(dim=1).view(B, 1, 1, 1) + 1e-8
        noise = noise / noise_max

        # Perturb the radius locally to deform the sphere into a smooth blob
        deformed_radii = (radii * (1 + self.deform_amplitude * noise)).clamp(min=0)

        mask = dist_sq <= deformed_radii**2
        return mask | ~apply

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs DeformedSphericalMask instance from a config object."""
        if config.transform.sphere_radius is None:
            raise ValueError("Spherical mask parameters are not set in the config.")
        radius_range = (
            config.transform.sphere_radius.min,
            config.transform.sphere_radius.max,
        )
        return cls(
            radius_range=radius_range,
            deform_amplitude=config.transform.sphere_deform_amplitude,
            deform_shape=config.transform.sphere_deform_shape,
            sphere_prob=config.transform.sphere_prob,
            device=config.device,
            seed=config.seed,
        )
