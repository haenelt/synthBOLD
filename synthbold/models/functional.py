"""Helper functions for signal modeling."""

import torch

from synthbold.sampling import sample_1d

__all__ = [
    "merge_labels",
    "labels_to_values",
    "spherical_to_cartesian",
    "dchi_to_dbz",
]


def merge_labels(*labels: torch.Tensor) -> torch.Tensor:
    """Merge two or more integer-labeled volumes into a single label map with disjoint
    indices.

    Tensors are merged in the given order: earlier tensors take precedence over later
    ones wherever they overlap. Each tensor's labels are offset by the cumulative
    maximum label of all previously merged tensors, so label indices remain unique
    across all inputs.

    Args:
        *labels: Two or more integer-labeled tensors of identical shape
            ``(B, X, Y, Z)``. Background voxels are assumed to be 0.

    Returns:
        Combined integer-labeled tensor with the same shape and dtype as the first
        input tensor.

    Raises:
        ValueError: If fewer than two tensors are given, or if their shapes differ.
    """
    if len(labels) < 2:
        raise ValueError("merge_labels requires at least two label tensors.")

    shape = labels[0].shape
    for t in labels[1:]:
        if t.shape != shape:
            raise ValueError(
                f"All label tensors must have the same shape; got {tuple(shape)} and "
                f"{tuple(t.shape)}."
            )

    combined = labels[0].clone()
    offset = combined.max()
    for t in labels[1:]:
        mask = (t > 0) & (combined == 0)
        combined[mask] = (t[mask] + offset).to(combined.dtype)
        offset = offset + t.max()

    return combined


def labels_to_values(
    labels: torch.Tensor,
    value: torch.Tensor | None,
    name: str,
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator,
    fill: float = 0.0,
) -> torch.Tensor:
    """Sample (or accept) one value per (batch, label) pair and scatter it into a
    volume matching `labels`.

    Values are drawn from one flat array shared across the batch, so each batch
    element's labels are offset by the cumulative maximum label of preceding batch
    elements (the same scheme as `merge_labels`'s offsets); this lets label ids
    repeat across batch elements without colliding.

    Args:
        labels: Integer-labeled tensor of shape ``(B, ...)``. Background voxels are
            assumed to be 0.
        value: Optional externally supplied flat values of shape ``(n_label,)``,
            where ``n_label`` is the sum, over batch elements, of each element's
            largest label value. Sampled uniformly from ``[low, high)`` if not given.
        name: Name of the parameter, used in the raised error message.
        low: Lower bound of the sampling range.
        high: Upper bound of the sampling range.
        device: Target compute device.
        generator: Random number generator for reproducibility.
        fill: Value assigned to background (unlabeled) voxels.

    Returns:
        Float32 tensor of the same shape as `labels`, with each labeled voxel set to
        its sampled/given value and background voxels set to `fill`.

    Raises:
        ValueError: If `value` is given and is not a 1D tensor of length `n_label`.
    """
    B = labels.shape[0]
    # Per-batch label count, used to offset each batch element's label ids into one
    # shared flat index space (same scheme as merge_labels).
    n_label = labels.amax(dim=tuple(range(1, labels.ndim))).long()
    offsets = torch.zeros(B, dtype=torch.long, device=device)
    offsets[1:] = n_label[:-1].cumsum(0)
    total_label = int(n_label.sum().item())

    # One value per (batch, label) pair, sampled or accepted as given.
    value = sample_1d(value, name, total_label, low, high, device, generator)

    out = torch.full_like(labels, fill, dtype=torch.float32, device=device)
    mask = labels > 0
    batch_idx = mask.nonzero(as_tuple=True)[0]
    # Map each foreground voxel's (batch, local label) to its flat index in `value`.
    flat_idx = offsets[batch_idx] + labels[mask].long() - 1
    out[mask] = value[flat_idx]
    return out


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Convert spherical coordinates (theta, phi) to Cartesian unit vectors.

    Args:
        theta: Polar angle in radians of shape ``(B)``.
        phi: Azimuthal angles in radians of shape ``(B)``.

    Returns:
        Normalized Cartesian vectors of shape ``(B, 3)``. Each row corresponds to the
        unit vector ``(x, y, z)`` for the given ``(theta, phi)``.

    Notes:
        - Assumes right-handed coordinate system:
            x = sin(theta) * cos(phi)
            y = sin(theta) * sin(phi)
            z = cos(theta)
        - The returned vectors are normalized to unit length.
    """
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    B0_vec = torch.stack(
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dim=1
    )  # shape (B, 3)
    return B0_vec / B0_vec.norm(dim=1, keepdim=True)


def dchi_to_dbz(
    chi: torch.Tensor,
    padding: int | tuple[int, ...],
    b0: float,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Compute magnetic field perturbations from a susceptibility using FFT convolution
    with a dipole kernel for a batch of volumes.

    Args:
        chi: Susceptibility distribution in ppm of shape ``(B, X, Y, Z)``.
        padding: Zero padding width applied before FFT convolution.
        b0: Static magnetic field strength in Tesla.
        theta: Polar angle of B0 direction in radians of shape ``(B)``.
        phi: Azimuthal angle of B0 direction in radians of shape ``(B)``.

    Returns:
        Magnetic field perturbation map with same shape as `chi` in Tesla.

    Notes:
        The implementation closely follows the implementation in
        `boldswimsuite.BOLDgeometry.DiscreteVoxel3D.from_vessel_index_grid_FFT`.
    """
    B, X, Y, Z = chi.shape
    device = chi.device
    dtype = chi.dtype

    # Handle padding
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    pad_t = torch.tensor(padding, device=device)
    shape = torch.tensor([X, Y, Z], device=device)
    pad = shape + 2 * pad_t

    # Convert spherical B0 -> cartesian for each batch
    b0_dir = spherical_to_cartesian(theta, phi)

    # Create coordinate grid
    axes = []
    for n in shape:
        half_n = int(torch.ceil(n.float() / 2))
        if n % 2 != 0:
            axis = torch.linspace(
                -half_n + 1, half_n - 1, n, device=device, dtype=dtype
            )
        else:
            axis = torch.linspace(-half_n + 1, half_n, n, device=device, dtype=dtype)
        axes.append(axis)

    x_grid, y_grid, z_grid = torch.meshgrid(*axes, indexing="ij")

    r2 = x_grid**2 + y_grid**2 + z_grid**2
    origin_mask = r2 == 0
    r2[origin_mask] = 1.0  # prevent divide by 0

    # Compute the projection of each voxel position on the B0 field direction.
    projection = (
        b0_dir[:, 0, None, None, None] * x_grid
        + b0_dir[:, 1, None, None, None] * y_grid
        + b0_dir[:, 2, None, None, None] * z_grid
    )

    # Compute dipole kernel for each batch (B, X, Y, Z)
    dipole_kernel = (1 / (4 * torch.pi)) * (3 * projection**2 - r2) / (r2 ** (5 / 2))
    dipole_kernel.masked_fill_(origin_mask, 0.0)

    # FFT convolution per batch
    kernel_fft = torch.fft.rfftn(dipole_kernel, s=tuple(pad.tolist()))
    chi_fft = torch.fft.rfftn(4 * torch.pi * chi, s=tuple(pad.tolist()))
    dbz_padded = torch.fft.irfftn(chi_fft * kernel_fft, s=tuple(pad.tolist()))
    dbz_padded *= b0

    # Shift and crop
    kernel_shift = (shape // 2 - 1) - pad_t
    shifts = (
        -int(kernel_shift[0].item()),
        -int(kernel_shift[1].item()),
        -int(kernel_shift[2].item()),
    )
    dbz_padded = torch.roll(dbz_padded, shifts=shifts, dims=(1, 2, 3))

    pad_x, pad_y, pad_z = pad_t.tolist()
    pad_shape = pad.tolist()
    dbz = dbz_padded[
        :,
        pad_x : pad_shape[0] - pad_x,
        pad_y : pad_shape[1] - pad_y,
        pad_z : pad_shape[2] - pad_z,
    ]

    return dbz
