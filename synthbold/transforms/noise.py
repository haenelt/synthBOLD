"""Noise transforms."""

from typing import Self

import torch

from synthbold.base import Transform
from synthbold.config import Config
from synthbold.transforms.functional import sample_fractal_noise

__all__ = [
    "GaussianNoise",
    "KSpaceSpikeNoise",
    "MultiplicativeGammaNoise",
    "NoncentralChiNoise",
    "PerlinNoise",
    "PoissonNoise",
    "RicianNoise",
    "SpeckleNoise",
]


class GaussianNoise(Transform):
    """Applies random Gaussian noise to 3D ``(X, Y, Z)`` or 4D ``(B, X, Y, Z)`` tensors.
    The distribution is defined based on randomly chosen `mu` and `std` that are sampled
    from a uniform distribution with set boundaries.

    Args:
        mu_range: Minimum and maximum mean of the noise distribution.
        std_range: Minimum and maximum standard deviation of the noise distribution.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        mu_range: tuple[float, float],
        std_range: tuple[float, float],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.mu_range = mu_range
        self.std_range = std_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes gaussian noise for the given output shape ``(B, X, Y, Z)``."""
        B, _, _, _ = shape

        mu = self._choose_mu(B).view(B, 1, 1, 1)
        std = self._choose_std(B).view(B, 1, 1, 1)

        noise = torch.randn(shape, device=self.device, generator=self.generator)
        return noise * std + mu

    @staticmethod
    def apply(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Applies the noise to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            noise: Noise tensor of same shape.

        Returns:
            Tensor of the same shape with added Gaussian noise.
        """
        return data + noise

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs GaussianNoise instance from a config object."""
        if config.transform.gnoise_mu is None or config.transform.gnoise_std is None:
            raise ValueError("Gaussian noise parameters are not set in the config.")
        mu_range = (
            config.transform.gnoise_mu.min,
            config.transform.gnoise_mu.max,
        )
        std_range = (
            config.transform.gnoise_std.min,
            config.transform.gnoise_std.max,
        )
        return cls(
            mu_range=mu_range,
            std_range=std_range,
            device=config.device,
            seed=config.seed,
        )

    def _choose_mu(self, n_vol: int) -> torch.Tensor:
        """Randomly select mu within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.mu_range[0], self.mu_range[1], generator=self.generator
        )

    def _choose_std(self, n_vol: int) -> torch.Tensor:
        """Randomly select std within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.std_range[0], self.std_range[1], generator=self.generator
        )


class KSpaceSpikeNoise(Transform):
    """Applies random localized spikes in k-space to 3D ``(X, Y, Z)`` or 4D
    ``(B, X, Y, Z)`` tensors.

    K-space spike artifacts (also known as RF interference or "corduroy" artifacts)
    arise from isolated erroneous samples in the acquired frequency-domain data and
    manifest as periodic stripe patterns across the reconstructed image. This transform
    Fourier transforms the volume, overwrites the k-space coefficients at one or more
    randomly chosen locations with a value proportional to the maximum spectral
    magnitude, and takes the inverse Fourier transform. The number of spikes is randomly
    chosen once per call from `num_spikes_range`. For each spike, an independent
    intensity multiplier is sampled from `intensity_range`.

    Args:
        num_spikes_range: Minimum and maximum number of k-space spikes (inclusive).
        intensity_range: Minimum and maximum spike intensity, expressed as a multiple
            of the maximum k-space magnitude.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_spikes_range: tuple[int, int] = (1, 3),
        intensity_range: tuple[float, float] = (1.0, 3.0),
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        if num_spikes_range[0] < 0 or num_spikes_range[1] < num_spikes_range[0]:
            raise ValueError(
                f"num_spikes_range must satisfy 0 <= min <= max, got "
                f"{num_spikes_range}."
            )
        self.num_spikes_range = num_spikes_range
        self.intensity_range = intensity_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Samples per-volume spike locations and intensities for the given output
        shape ``(B, X, Y, Z)``.

        Returns:
            Tensor of shape ``(B, num_spikes, 4)``, where the last dimension holds
            the voxel coordinates ``(x, y, z)`` of each spike in k-space, followed by
            its intensity multiplier.
        """
        B, X, Y, Z = shape

        num_spikes = int(
            torch.randint(
                self.num_spikes_range[0],
                self.num_spikes_range[1] + 1,
                (1,),
                generator=self.generator,
                device=self.device,
            ).item()
        )

        x = torch.randint(
            0, X, (B, num_spikes), generator=self.generator, device=self.device
        )
        y = torch.randint(
            0, Y, (B, num_spikes), generator=self.generator, device=self.device
        )
        z = torch.randint(
            0, Z, (B, num_spikes), generator=self.generator, device=self.device
        )
        locations = torch.stack((x, y, z), dim=-1).to(torch.float32)

        intensity = self._choose_intensity((B, num_spikes))
        return torch.cat((locations, intensity.unsqueeze(-1)), dim=-1)

    @staticmethod
    def apply(data: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        """Applies localized k-space spikes to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            spikes: Spike tensor of shape ``(B, num_spikes, 4)`` as returned by
                `sample`.

        Returns:
            Tensor of the same shape with k-space spike artifacts applied.
        """
        B = data.shape[0]
        num_spikes = spikes.shape[1]

        spectrum = torch.fft.fftshift(
            torch.fft.fftn(data, dim=(-3, -2, -1)), dim=(-3, -2, -1)
        )
        max_magnitude = spectrum.abs().flatten(1).amax(dim=1)

        locations = spikes[..., :3].long()
        intensity = spikes[..., 3]
        spike_value = (intensity * max_magnitude.view(B, 1)).to(spectrum.dtype)

        batch_idx = torch.arange(B, device=data.device).view(B, 1).expand(B, num_spikes)
        spectrum[batch_idx, locations[..., 0], locations[..., 1], locations[..., 2]] = (
            spike_value
        )

        spectrum = torch.fft.ifftshift(spectrum, dim=(-3, -2, -1))
        return torch.fft.ifftn(spectrum, dim=(-3, -2, -1)).real

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs KSpaceSpikeNoise instance from a config object."""
        intensity_range = (
            config.transform.spike_intensity.min,
            config.transform.spike_intensity.max,
        )
        return cls(
            num_spikes_range=config.transform.spike_num,
            intensity_range=intensity_range,
            device=config.device,
            seed=config.seed,
        )

    def _choose_intensity(self, shape: tuple[int, int]) -> torch.Tensor:
        """Randomly select spike intensity multipliers within the range."""
        return torch.empty(shape, device=self.device).uniform_(
            self.intensity_range[0], self.intensity_range[1], generator=self.generator
        )


class MultiplicativeGammaNoise(Transform):
    """Applies multiplicative gamma noise to 3D ``(X, Y, Z)`` or 4D ``(B, X, Y, Z)``
    tensors.

    Multiplicative gamma noise models `output = data * noise`, where `noise` is drawn
    from a Gamma distribution with mean 1, i.e. `Gamma(shape, rate)` with
    `rate == shape`. The `shape` (concentration) parameter controls the noise variance,
    `var(noise) == 1 / shape`: smaller values yield stronger, more skewed noise, while
    larger values yield a multiplicative factor close to 1. For each volume in the
    batch, a random `shape` is sampled from a uniform distribution with set boundaries.

    Args:
        shape_range: Minimum and maximum shape (concentration) parameter of the
            multiplicative gamma noise distribution. Must be strictly positive.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.

    Note:
        The gamma samples drawn in `apply` rely on the global PyTorch RNG state and
        are not controlled by `seed`. Only the sampled `shape` values are reproducible.
    """

    def __init__(
        self,
        shape_range: tuple[float, float],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        if shape_range[0] <= 0:
            raise ValueError("shape_range must be strictly positive.")
        self.shape_range = shape_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes the per-volume gamma shape (concentration) parameter for the
        given output shape ``(B, X, Y, Z)``."""
        B, _, _, _ = shape

        concentration = self._choose_shape(B)
        return concentration.view(B, 1, 1, 1)

    @staticmethod
    def apply(data: torch.Tensor, concentration: torch.Tensor) -> torch.Tensor:
        """Applies multiplicative gamma noise to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            concentration: Per-volume gamma shape parameter tensor broadcastable to
                the shape of `data`.

        Returns:
            Tensor of the same shape with multiplicative gamma noise applied.
        """
        concentration = concentration.expand(data.shape)
        noise = torch._standard_gamma(concentration) / concentration
        return data * noise

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs MultiplicativeGammaNoise instance from a config object."""
        shape_range = (
            config.transform.mgamma_shape.min,
            config.transform.mgamma_shape.max,
        )
        return cls(
            shape_range=shape_range,
            device=config.device,
            seed=config.seed,
        )

    def _choose_shape(self, n_vol: int) -> torch.Tensor:
        """Randomly select the gamma shape parameter within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.shape_range[0], self.shape_range[1], generator=self.generator
        )


class NoncentralChiNoise(Transform):
    """Applies noncentral chi noise to 3D ``(X, Y, Z)`` or 4D ``(B, X, Y, Z)`` tensors.

    Noncentral chi noise generalizes Rician noise to magnitude images reconstructed
    from multiple channels via sum-of-squares, such as multi-coil MRI data. The
    magnitude is modeled as the norm of a `dof`-dimensional vector whose first
    component carries the input signal and all components are corrupted by
    independent zero-mean Gaussian noise with shared standard deviation. For
    `dof == 2` this is equivalent to Rician noise. For each volume in the batch, a
    random `std` is sampled from a uniform distribution with set boundaries and used
    as the standard deviation of all noise channels.

    Args:
        std_range: Minimum and maximum standard deviation of the underlying Gaussian
            noise channels.
        dof: Degrees of freedom, i.e. the number of underlying Gaussian noise
            channels. Must be at least 2.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        std_range: tuple[float, float],
        dof: int = 4,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        if dof < 2:
            raise ValueError("dof must be at least 2.")
        self.std_range = std_range
        self.dof = dof

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes the `dof` independent Gaussian noise channels for the given output
        shape ``(B, X, Y, Z)``, stacked along a new leading dimension."""
        B, _, _, _ = shape

        std = self._choose_std(B).view(B, 1, 1, 1)
        noise = torch.randn(
            (self.dof, *shape), device=self.device, generator=self.generator
        )
        return noise * std

    @staticmethod
    def apply(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Applies the noise to the input (magnitude) tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            noise: Noise tensor of shape ``(dof, *data.shape)`` holding the independent
                Gaussian noise channels.

        Returns:
            Tensor of the same shape with noncentral chi noise applied.
        """
        return torch.sqrt((data + noise[0]) ** 2 + noise[1:].pow(2).sum(dim=0))

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs NoncentralChiNoise instance from a config object."""
        std_range = (
            config.transform.ncchi_std.min,
            config.transform.ncchi_std.max,
        )
        return cls(
            std_range=std_range,
            dof=config.transform.ncchi_dof,
            device=config.device,
            seed=config.seed,
        )

    def _choose_std(self, n_vol: int) -> torch.Tensor:
        """Randomly select std within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.std_range[0], self.std_range[1], generator=self.generator
        )


class PerlinNoise(Transform):
    """Applies additive Perlin-like fractal noise to 3D ``(X, Y, Z)`` or 4D
    ``(B, X, Y, Z)`` tensors.

    Multiple octaves of smooth, low-frequency noise are summed with geometrically
    decaying amplitudes (persistence), yielding a self-similar fractal noise pattern
    with detail across multiple spatial scales, similar to Perlin noise. For each volume
    in the batch, a random `amplitude` is sampled from a uniform distribution with set
    boundaries and used to scale the resulting field.

    Args:
        base_shape: Matrix size of the coarsest (first) noise octave.
        octaves: Number of noise octaves to combine. Each successive octave doubles
            the resolution of the previous one, capped at the target shape. Must be
            at least 1.
        persistence: Amplitude decay factor between successive octaves. Must be in
            `(0, 1]`.
        amplitude_range: Minimum and maximum scaling factor for the noise field.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        base_shape: tuple[int, int, int],
        octaves: int = 4,
        persistence: float = 0.5,
        amplitude_range: tuple[float, float] = (0.0, 1.0),
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        if octaves < 1:
            raise ValueError("octaves must be at least 1.")
        if not 0.0 < persistence <= 1.0:
            raise ValueError("persistence must be in (0, 1].")
        self.base_shape = base_shape
        self.octaves = octaves
        self.persistence = persistence
        self.amplitude_range = amplitude_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes Perlin-like fractal noise for the given output shape
        ``(B, X, Y, Z)``."""
        B, X, Y, Z = shape

        amplitude = self._choose_amplitude(B).view(B, 1, 1, 1)
        noise = sample_fractal_noise(
            B,
            self.base_shape,
            (X, Y, Z),
            self.octaves,
            self.persistence,
            self.device,
            self.generator,
        )
        return noise * amplitude

    @staticmethod
    def apply(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Applies the noise to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            noise: Noise tensor of same shape.

        Returns:
            Tensor of the same shape with additive Perlin-like noise.
        """
        return data + noise

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs PerlinNoise instance from a config object."""
        amplitude_range = (
            config.transform.perlin_amplitude.min,
            config.transform.perlin_amplitude.max,
        )
        return cls(
            base_shape=config.transform.perlin_base_shape,
            octaves=config.transform.perlin_octaves,
            persistence=config.transform.perlin_persistence,
            amplitude_range=amplitude_range,
            device=config.device,
            seed=config.seed,
        )

    def _choose_amplitude(self, n_vol: int) -> torch.Tensor:
        """Randomly select the amplitude scaling factor within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.amplitude_range[0], self.amplitude_range[1], generator=self.generator
        )


class PoissonNoise(Transform):
    """Applies signal-dependent Poisson noise to 3D ``(X, Y, Z)`` or 4D ``(B, X, Y, Z)``
    tensors.

    For each volume in the batch, a random `peak` count is drawn from a uniform
    distribution with set boundaries. The input is scaled by `peak`, Poisson-distributed
    counts are drawn, and the result is rescaled back, so that the relative noise level
    increases as `peak` decreases.

    Args:
        peak_range: Minimum and maximum peak count used to scale the input before
            drawing Poisson counts.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.

    Note:
        The Poisson counts drawn in `apply` rely on the global PyTorch RNG state and are
        not controlled by `seed`. Only the sampled `peak` values are reproducible.
    """

    def __init__(
        self,
        peak_range: tuple[float, float],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.peak_range = peak_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes the per-volume peak count for the given output shape
        ``(B, X, Y, Z)``."""
        B, _, _, _ = shape

        peak = torch.empty(B, device=self.device).uniform_(
            self.peak_range[0], self.peak_range[1], generator=self.generator
        )
        return peak.view(B, 1, 1, 1)

    @staticmethod
    def apply(data: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
        """Applies signal-dependent Poisson noise to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            peak: Per-volume peak count tensor broadcastable to the shape of `data`.

        Returns:
            Tensor of the same shape with Poisson-distributed noise applied.
        """
        rate = data.clamp(min=0) * peak
        return torch.poisson(rate) / peak

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs PoissonNoise instance from a config object."""
        peak_range = (
            config.transform.pnoise_peak.min,
            config.transform.pnoise_peak.max,
        )
        return cls(
            peak_range=peak_range,
            device=config.device,
            seed=config.seed,
        )


class RicianNoise(Transform):
    """Applies Rician noise to 3D ``(X, Y, Z)`` or 4D ``(B, X, Y, Z)`` tensors.

    Rician noise models the magnitude of a complex signal whose real and imaginary
    parts are each corrupted by independent Gaussian noise, as is the case for MRI
    magnitude images. For each volume in the batch, a random `std` is sampled from a
    uniform distribution with set boundaries and used as the standard deviation of both
    Gaussian noise channels.

    Args:
        std_range: Minimum and maximum standard deviation of the underlying Gaussian
            noise channels.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        std_range: tuple[float, float],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.std_range = std_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes the real and imaginary Gaussian noise channels for the given
        output shape ``(B, X, Y, Z)``, stacked along a new leading dimension."""
        B, _, _, _ = shape

        std = self._choose_std(B).view(B, 1, 1, 1)
        noise = torch.randn((2, *shape), device=self.device, generator=self.generator)
        return noise * std

    @staticmethod
    def apply(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Applies the noise to the input (magnitude) tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            noise: Noise tensor of shape ``(2, *data.shape)`` holding the real and
                imaginary Gaussian noise channels.

        Returns:
            Tensor of the same shape with Rician noise applied.
        """
        return torch.sqrt((data + noise[0]) ** 2 + noise[1] ** 2)

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs RicianNoise instance from a config object."""
        std_range = (
            config.transform.rnoise_std.min,
            config.transform.rnoise_std.max,
        )
        return cls(
            std_range=std_range,
            device=config.device,
            seed=config.seed,
        )

    def _choose_std(self, n_vol: int) -> torch.Tensor:
        """Randomly select std within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.std_range[0], self.std_range[1], generator=self.generator
        )


class SpeckleNoise(Transform):
    """Applies multiplicative speckle noise to 3D ``(X, Y, Z)`` or 4D ``(B, X, Y, Z)``
    tensors.

    Speckle noise models multiplicative degradation as `output = data + data * noise`,
    where `noise` is drawn from a zero-mean Gaussian distribution. For each volume in
    the batch, a random `std` is sampled from a uniform distribution with set boundaries
    and used as the standard deviation of the noise.

    Args:
        std_range: Minimum and maximum standard deviation of the multiplicative noise
            distribution.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        std_range: tuple[float, float],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)
        self.std_range = std_range

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Computes multiplicative speckle noise for the given output shape
        ``(B, X, Y, Z)``."""
        B, _, _, _ = shape

        std = self._choose_std(B).view(B, 1, 1, 1)
        noise = torch.randn(shape, device=self.device, generator=self.generator)
        return noise * std

    @staticmethod
    def apply(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Applies the noise to the input tensor.

        Args:
            data: Input tensor of shape ``(X, Y, Z)`` or ``(B, X, Y, Z)``.
            noise: Noise tensor of same shape.

        Returns:
            Tensor of the same shape with multiplicative speckle noise applied.
        """
        return data + data * noise

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs SpeckleNoise instance from a config object."""
        std_range = (
            config.transform.speckle_std.min,
            config.transform.speckle_std.max,
        )
        return cls(
            std_range=std_range,
            device=config.device,
            seed=config.seed,
        )

    def _choose_std(self, n_vol: int) -> torch.Tensor:
        """Randomly select std within the range."""
        return torch.empty(n_vol, device=self.device).uniform_(
            self.std_range[0], self.std_range[1], generator=self.generator
        )
