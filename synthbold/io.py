"""Reusable utilities to handle common tensor validation and manipulation tasks."""

import importlib.metadata
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import nibabel as nb
import numpy as np
import torch
import zarr
from nibabel.freesurfer.io import write_geometry
from skimage import measure

__all__ = [
    "load_nifti",
    "load_zarr",
    "save_batch",
    "save_zarr",
    "save_nifti",
    "save_mesh",
    "zarr_attributes",
]


def load_nifti(fname: str | Path) -> tuple[np.ndarray, np.ndarray, nb.Nifti1Header]:
    """Load a NifTI file and return it as a Nifti1Image.

    Args:
        fname: Path to the NifTI file.

    Returns:
        Tuple of the data array, affine matrix, and header.

    Raises:
        TypeError: If the loaded file is not a `Nifti1Image`.
    """
    img = cast(nb.Nifti1Image, nb.load(fname))
    if not isinstance(img, nb.Nifti1Image):
        raise TypeError(f"Expected Nifti1Image, got {type(img)}.")
    return img.get_fdata(), img.affine, img.header


def load_zarr(fname: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load map data from ZARR format.

    Args:
        fname: Path to the ZARR file.

    Returns:
        Tuple of the data array and attributes.

    Raises:
        ValueError: If the ZARR group contains more than one array.
    """
    z = zarr.open(fname, mode="r")
    if isinstance(z, zarr.Group):
        if len(z) != 1:
            raise ValueError(
                "ZARR group has more than one array; array name must be specified."
            )
        z = z[next(iter(z))]  # pick first array

    # load array as NumPy array
    z_array = cast(zarr.Array, z)
    data = np.asarray(z_array[:])
    attrs = dict(z_array.attrs)

    return data, attrs


def save_zarr(
    fname: str | Path, data: np.ndarray | torch.Tensor, attrs: dict[str, Any]
) -> None:
    """Save data to disk in ZARR format.

    Args:
        fname: File name for saving the ZARR file.
        data: Data array to save.
        attrs: Attributes to save with the data.

    Raises:
        TypeError: If the created ZARR object is not a `zarr.Array`.
    """
    fname = Path(fname)
    fname.parent.mkdir(exist_ok=True, parents=True)

    # Convert torch.Tensor to numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Ensure data is a NumPy array for ZARR
    data_array = np.asarray(data)

    # Create ZARR array
    z = zarr.open(
        fname,
        mode="w",
        shape=data_array.shape,
        dtype=data_array.dtype,
        chunks=(1, *data_array.shape[1:]),
    )

    if not isinstance(z, zarr.Array):
        raise TypeError(f"Expected zarr.Array, got {type(z)}.")

    # Write data
    z[:] = data_array

    # Write attributes safely
    z.attrs.update(dict(attrs))


def save_nifti(
    fname: str | Path,
    data: np.ndarray | torch.Tensor,
    affine: np.ndarray | None = None,
    header: nb.Nifti1Header | None = None,
    permute: bool = False,
    niivue: bool = False,
) -> None:
    """Save a NumPy array to a NIfTI file. If `permute` is True, the first and last
    axes of `data` are swapped. This is useful when the data has shape ``(N, X, Y, Z)``
    but needs to be saved as ``(X, Y, Z, N)`` to store 3D volumes as a time series.

    Args:
        fname: File name for saving the NIfTI file.
        data: Data array to save.
        affine: Affine matrix for the NIfTI file.
        header: Header for the NIfTI file.
        permute: Whether to permute the axes of the data.
        niivue: Whether to convert the data for fast visualization with Niivue.

    Raises:
        TypeError: If `data` is not a NumPy array (after tensor conversion).

    Notes:
        For fast visualization with Niivue, the flag converts the numpy array to
        integer format and scales the data to maximum integer range.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if affine is None:
        affine = np.eye(4)
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected NumPy array, got {type(data)}.")
    if data.dtype == np.int64:
        data = data.astype(np.int32)

    # Make output directory
    Path(fname).parent.mkdir(exist_ok=True, parents=True)

    # Optionally, permute first and last axes
    if permute:
        data = np.moveaxis(data, 0, -1)

    # Convert to integer and scale to maximum integer scale for fast visual check with
    # niivue vscode extension
    if niivue:
        int16_min = np.iinfo(np.int16).min
        int16_max = np.iinfo(np.int16).max
        data_min = np.min(data)
        data_max = np.max(data)

        if data_min == data_max:
            data = np.full(data.shape, int16_min, dtype=np.int16)
        else:
            data = np.round(
                (data - data_min) / (data_max - data_min) * (int16_max - int16_min)
                + int16_min
            ).astype(np.int16)
            data = np.clip(data, int16_min, int16_max).astype(np.int16)

    img: nb.Nifti1Image = nb.Nifti1Image(data, affine, header=header)
    nb.save(img, fname)


def save_batch(
    fields: Mapping[str, np.ndarray | torch.Tensor],
    dirname: str | Path,
    fmt: str,
    suffix: str = "",
    attrs: dict[str, Any] | None = None,
) -> None:
    """Saves each field in `fields` to `<dirname>/<field><suffix>.<fmt>`.

    Args:
        fields: Mapping of field name to data array, e.g. a `NamedTuple`'s `_asdict()`.
        dirname: Directory to save each field's file into.
        fmt: File format, either ``"zarr"`` or ``"nii"``.
        suffix: Suffix inserted between each field name and its extension.
        attrs: Attributes to save alongside ZARR arrays. Ignored for NIfTI.

    Raises:
        ValueError: If `fmt` is not ``"zarr"`` or ``"nii"``.
    """
    if fmt not in ("zarr", "nii"):
        raise ValueError(f"fmt must be 'zarr' or 'nii', got {fmt!r}.")
    dirname = Path(dirname)
    for field, data in fields.items():
        out_fname = dirname / f"{field}{suffix}.{fmt}"
        if fmt == "zarr":
            save_zarr(out_fname, data, attrs or {})
        else:
            save_nifti(out_fname, data, permute=True)


def save_mesh(fname: str | Path, data: np.ndarray | torch.Tensor) -> None:
    """Creates a mesh from a binary array using the marching cubes algorithm and saves
    the mesh to disk in FreeSurfer binary format.

    Args:
        fname: File name for saving mesh file in FreeSurfer format.
        data: Data array to save.

    Raises:
        ValueError: If `data` is not a NumPy array (after tensor conversion), or if
            it is not 3-dimensional.
    """
    fname = Path(fname)
    fname.parent.mkdir(exist_ok=True, parents=True)

    # Convert torch.Tensor to numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Check data type and array dimensions
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Input data must be a numpy array, got {type(data)}.")
    if data.ndim != 3:
        raise ValueError(f"Input data must be a 3D array, got {data.shape} dimensions.")

    data = data.astype(bool)
    verts, faces, _, _ = measure.marching_cubes(data)
    write_geometry(fname, verts, faces)


def zarr_attributes(
    cls_name: str, device: torch.device, seed: int | None
) -> dict[str, Any]:
    """Build common metadata attributes for ZARR storage.

    Args:
        cls_name: Name of the generating class, stored under ``"generator"``.
        device: PyTorch device the data was generated on.
        seed: Random seed used for generation, if any.

    Returns:
        Dict of metadata suitable for passing as `attrs` to `save_zarr`.
    """
    try:
        pkg_name = __name__.split(".")[0]
        version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "generator": cls_name,
        "created_at": datetime.now(UTC).isoformat(),
        "version": version,
        "axis_order": ("N", "X", "Y", "Z"),
        "device": str(device),
        "seed": seed,
    }
