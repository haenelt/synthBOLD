"""Helper functions for transforms."""

import torch
import torch.nn.functional as F

__all__ = ["apply_deformation", "sample_fractal_noise", "sample_lowres_noise"]


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
