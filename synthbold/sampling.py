"""Shared random-sampling helpers."""

import math

import torch

__all__ = ["sample_uniform", "sample_log_uniform", "sample_polar", "sample_1d"]


def sample_uniform(
    size: int | tuple[int, ...],
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample values uniformly from ``[low, high)``.

    Args:
        size: Number of values to sample.
        low: Lower bound of the sampling range.
        high: Upper bound of the sampling range.
        device: Target compute device.
        generator: Random number generator for reproducibility.

    Returns:
        Tensor of shape ``(size,)``.
    """
    return torch.empty(size, device=device, dtype=torch.float32).uniform_(
        low, high, generator=generator
    )


def sample_log_uniform(
    size: int | tuple[int, ...],
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample values log-uniformly from ``[low, high)``.

    Values are drawn uniformly in log-space and exponentiated back, giving even
    coverage across orders of magnitude instead of concentrating draws near `high`,
    as plain uniform sampling does when the range spans multiple decades.

    Args:
        size: Number of values to sample.
        low: Lower bound of the sampling range. Must be strictly positive.
        high: Upper bound of the sampling range. Must be strictly positive.
        device: Target compute device.
        generator: Random number generator for reproducibility.

    Returns:
        Tensor of shape ``(size,)``.

    Raises:
        ValueError: If `low` or `high` is not strictly positive.
    """
    if low <= 0 or high <= 0:
        raise ValueError(
            f"low and high must be strictly positive, got low={low}, high={high}."
        )
    log_values = sample_uniform(size, math.log(low), math.log(high), device, generator)
    return torch.exp(log_values)


def sample_polar(
    size: int | tuple[int, ...],
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample polar angles isotropically distributed on the sphere.

    Sampling `cos(theta)` uniformly, rather than `theta` itself, is what makes the
    resulting angles isotropic on the sphere: the solid angle element scales with
    `sin(theta)`, so uniform-in-angle sampling would oversample near the poles.

    Args:
        size: Number of angles to sample.
        low: Lower bound of the polar angle range, in radians.
        high: Upper bound of the polar angle range, in radians.
        device: Target compute device.
        generator: Random number generator for reproducibility.

    Returns:
        Tensor of shape ``(size,)`` containing polar angles in ``[low, high]``.
    """
    cos_values = sample_uniform(size, math.cos(high), math.cos(low), device, generator)
    return torch.acos(cos_values)


def sample_1d(
    value: torch.Tensor | None,
    name: str,
    size: int,
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator,
    polar: bool = False,
) -> torch.Tensor:
    """Return `value` moved to `device`, or sample it if `value` is None.

    Args:
        value: Optional externally supplied tensor to use instead of sampling.
        name: Name of the parameter, used in the raised error message.
        size: Expected length of `value`, or number of values to sample when
            `value` is None.
        low: Lower bound of the sampling range.
        high: Upper bound of the sampling range.
        device: Target compute device.
        generator: Random number generator for reproducibility.
        polar: If True, treat `low`/`high` as a polar angle range and sample
            `cos(theta)` uniformly instead of `theta` itself, so that the
            resulting angles are isotropic on the sphere (the solid angle
            element scales with `sin(theta)`, so uniform-in-angle sampling is
            not uniform-on-sphere).

    Returns:
        Tensor of shape ``(size,)`` moved to `device`.

    Raises:
        ValueError: If `value` is given and is not a 1D tensor of length `size`.
    """
    if value is not None:
        if value.ndim != 1 or value.shape[0] != size:
            raise ValueError(
                f"Expected {name} of shape ({size},), got shape {tuple(value.shape)}."
            )
        return value.to(device, dtype=torch.float32)
    if polar:
        return sample_polar(size, low, high, device, generator)
    return sample_uniform(size, low, high, device, generator)
