"""Helper functions for transforms."""

import torch
import torch.nn.functional as F

from synthbold.base import RandomGeneratorMixin, Transform

__all__ = [
    "Pipeline",
    "apply_deformation",
    "sample_fractal_noise",
    "sample_lowres_noise",
    "zscore",
    "psc",
    "percentile_scale",
]


def sample_lowres_noise(
    batch_size: int,
    lowres_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    device: torch.device,
    generator: torch.Generator,
    align_corners: bool = False,
) -> torch.Tensor:
    """Samples a batch of smooth, low-frequency noise fields.

    Standard Gaussian noise is drawn on a coarse grid and trilinearly upsampled to the
    target resolution, yielding spatially smooth random fields.

    Args:
        batch_size: Number of independent noise fields to generate.
        lowres_shape: Matrix size ``(x, y, z)`` of the coarse noise grid.
        target_shape: Matrix size ``(X, Y, Z)`` to upsample to.
        device: Target compute device.
        generator: Random number generator for reproducibility.
        align_corners: Passed to ``torch.nn.functional.interpolate``.

    Returns:
        Noise tensor of shape ``(B, X, Y, Z)``.
    """
    noise = torch.randn(
        batch_size, 1, *lowres_shape, device=device, generator=generator
    )
    noise = F.interpolate(
        noise, size=target_shape, mode="trilinear", align_corners=align_corners
    )
    return noise.squeeze(1)


def sample_fractal_noise(
    batch_size: int,
    base_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    octaves: int,
    persistence: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Samples a batch of Perlin-like fractal noise fields.

    Multiple octaves of smooth, low-frequency noise (see `sample_lowres_noise`) are
    generated at progressively doubling resolutions, starting from `base_shape`, and
    summed with geometrically decaying amplitudes. This yields a self-similar noise
    pattern with detail across multiple scales, similar to Perlin noise.

    Args:
        batch_size: Number of independent noise fields to generate.
        base_shape: Matrix size ``(x, y, z)`` of the coarsest (first) octave.
        target_shape: Matrix size ``(X, Y, Z)`` of the output noise field.
        octaves: Number of octaves to combine. Each successive octave doubles the
            resolution of the previous one, capped at `target_shape`.
        persistence: Amplitude decay factor between successive octaves, in `(0, 1]`.
        device: Target compute device.
        generator: Random number generator for reproducibility.

    Returns:
        Noise tensor of shape ``(B, X, Y, Z)``, normalized to unit total amplitude.
    """
    noise = torch.zeros((batch_size, *target_shape), device=device)
    amplitude, total_amplitude = 1.0, 0.0
    for octave in range(octaves):
        octave_shape = tuple(
            min(b * 2**octave, t) for b, t in zip(base_shape, target_shape, strict=True)
        )
        noise += amplitude * sample_lowres_noise(
            batch_size, octave_shape, target_shape, device, generator
        )
        total_amplitude += amplitude
        amplitude *= persistence
    return noise / total_amplitude


def apply_deformation(
    data: torch.Tensor, displacement: torch.Tensor, interp: str = "bilinear"
) -> torch.Tensor:
    """Applies a 3D displacement field to a volumetric tensor. This function warps a
    batch of 3D volumes using a smooth displacement field. Each voxel in `data` is moved
    according to the corresponding vector in `displacement`. The output has the same
    shape as the input.

    Args:
        data: Input tensor of shape ``(B, X, Y, Z)``.
        displacement: Displacement field of shape ``(B, X, Y, Z, 3)``, specifying voxel
            offsets in voxel coordinates along X, Y, and Z.
        interp: Interpolation method (bilinear, nearest, ...)

    Returns:
        Deformed tensor of shape ``(B, X, Y, Z)``.
    """
    B, X, Y, Z = data.shape

    # create regular grid [-1, 1]
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(-1, 1, X, device=data.device),
        torch.linspace(-1, 1, Y, device=data.device),
        torch.linspace(-1, 1, Z, device=data.device),
        indexing="ij",
    )

    # Normalize displacement to grid_sample's [-1, 1] coordinate space. With
    # align_corners=True, a step of one voxel corresponds to 2 / (size - 1) in
    # normalized coordinates.
    disp_norm = displacement.clone()
    disp_norm[..., 0] /= (X - 1) / 2
    disp_norm[..., 1] /= (Y - 1) / 2
    disp_norm[..., 2] /= (Z - 1) / 2

    # grid_sample's last dimension is ordered (x, y, z), mapping to the (W, H, D)
    # input axes, i.e. (Z, Y, X) for data of shape (B, 1, X, Y, Z)
    grid = torch.stack(
        (
            grid_z + disp_norm[..., 2],
            grid_y + disp_norm[..., 1],
            grid_x + disp_norm[..., 0],
        ),
        dim=-1,
    )  # (B, X, Y, Z, 3)

    # grid_sample needs a channel dimension
    data_orig_dtype = data.dtype
    is_float = data.is_floating_point()
    if not is_float:
        data = data.to(torch.float32)

    data = data.unsqueeze(1)
    data_warped = F.grid_sample(
        data, grid, align_corners=True, mode=interp, padding_mode="reflection"
    )
    data_warped = data_warped.squeeze(1)

    if not is_float:
        data_warped = data_warped.round().to(data_orig_dtype)

    return data_warped


class Pipeline(RandomGeneratorMixin):
    """Compose multiple tensor-to-tensor transforms in sequence.

    This class chains multiple `Transform` instances together. When called, it passes
    the input tensor through each transform in the order they were provided. Each
    transform is independently and randomly skipped according to `prob`, evaluated
    per batch element.

    Args:
        transforms: A list of `Transform` instances.
        prob: Probability of applying each transform. Evaluated independently per
            transform and per batch element.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        transforms: list[Transform],
        prob: float = 0.5,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed, device=device)
        self.prob = prob
        self.transforms = transforms

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply pipeline to input tensor.

        Args:
            data: Input tensor of shape ``(B, X, Y, Z)``.

        Returns:
            Tensor produced by sequentially applying each transform, with each
            transform randomly skipped according to `self.prob`.
        """
        B = data.shape[0]

        for t in self.transforms:
            should_apply = (
                torch.rand(B, generator=self.generator, device=self.device) < self.prob
            )
            if not should_apply.any():
                continue

            # Transforms operate on the whole batch at once, so apply to everyone and
            # then discard the result for batch elements not selected by `should_apply`.
            result = t(data)
            mask = should_apply.view(B, 1, 1, 1)
            data = torch.where(mask, result, data)

        return data


def zscore(
    data: torch.Tensor,
    center: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalizes each batch element to zero mean and unit variance.

    Mean and standard deviation are computed per batch element over all non-batch
    dimensions, so normalization is independent across the batch.

    Args:
        data: Input tensor of shape ``(B, ...)``.
        center: Whether to subtract the mean. Set to False for quantities derived
            from squared units (e.g. a variance), which should be rescaled but not
            mean-centered.
        eps: Relative floor, as a fraction of each batch element's peak absolute
            value, added to `std` to avoid division by zero. Expressed relative to
            `data` rather than as an absolute constant so the floor stays negligible
            regardless of the physical scale/units of `data`.

    Returns:
        Tensor of the same shape as `data`, normalized.
    """
    dims = tuple(range(1, data.ndim))
    mean = data.mean(dim=dims, keepdim=True)
    std = data.std(dim=dims, keepdim=True, unbiased=False)
    floor = eps * data.detach().abs().amax(dim=dims, keepdim=True)
    scale = std + floor
    if not center:
        return data / scale
    return (data - mean) / scale


def psc(data: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Converts each batch element to percent signal change relative to its mean.

    Mean is computed per batch element over all non-batch dimensions, so the
    conversion is independent across the batch.

    Args:
        data: Input tensor of shape ``(B, ...)``.
        eps: Relative floor, as a fraction of each batch element's peak absolute
            value, added to the mean to avoid division by zero. Expressed relative
            to `data` rather than as an absolute constant so the floor stays
            negligible regardless of the physical scale/units of `data`.

    Returns:
        Tensor of the same shape as `data`, expressed as percent deviation from
        each batch element's mean.
    """
    dims = tuple(range(1, data.ndim))
    mean = data.mean(dim=dims, keepdim=True)
    floor = eps * data.detach().abs().amax(dim=dims, keepdim=True)
    return 100 * (data - mean) / (mean + floor)


def percentile_scale(
    data: torch.Tensor,
    lower: float = 0.01,
    upper: float = 0.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rescales each batch element to ``[0, 1]`` using percentile clipping.

    Per batch element, values are linearly rescaled so that the `lower` and `upper`
    quantiles map to 0 and 1 respectively, then clamped to ``[0, 1]``. This is more
    robust to outliers than min-max scaling since it doesn't rely on the tensor's
    extreme values.

    Args:
        data: Input tensor of shape ``(B, ...)``.
        lower: Lower quantile, in ``[0, 1]``, mapped to 0.
        upper: Upper quantile, in ``[0, 1]``, mapped to 1.
        eps: Relative floor, as a fraction of each batch element's peak absolute
            value, added to the denominator to avoid division by zero. Expressed
            relative to `data` rather than as an absolute constant so the floor
            stays negligible regardless of the physical scale/units of `data`.

    Returns:
        Tensor of the same shape as `data`, rescaled and clamped to ``[0, 1]``.
    """
    B = data.shape[0]
    data_flat = data.view(B, -1)

    q_low = torch.quantile(data_flat, lower, dim=1, keepdim=True)
    q_high = torch.quantile(data_flat, upper, dim=1, keepdim=True)
    floor = eps * data_flat.detach().abs().amax(dim=1, keepdim=True)

    data_flat = (data_flat - q_low) / (q_high - q_low + floor)
    data_flat = data_flat.clamp(0.0, 1.0)

    return data_flat.view_as(data)
