"""Tests for utility functions."""

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
import torch
import zarr
from nibabel.freesurfer.io import read_geometry

from synthbold.io import load_nifti, load_zarr, save_mesh, save_nifti, save_zarr

SMALL_SHAPE = (4, 4, 4)


def _sphere_mask(shape: tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    center = np.array(shape) / 2
    r2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return (r2 < 16).astype(np.float32)


# --- save_nifti / load_nifti ---


def test_load_nifti(tmp_path: str) -> None:
    """Check if nifti files can be properly loaded."""
    # Create a dummy nifti image with integer data to avoid scaling issues.
    arr = np.random.randint(0, 255, (10, 10, 10), dtype=np.int16)
    fname = f"{tmp_path}/test.nii"
    save_nifti(fname, arr)

    # Test successful loading
    arr_loaded, _, _ = load_nifti(fname)
    assert isinstance(arr_loaded, np.ndarray)
    assert np.array_equal(arr, arr_loaded)

    # Test failed loading
    mgh_img = nb.MGHImage(arr.astype(np.float32), np.eye(4))
    mgh_fname = f"{tmp_path}/test.mgh"
    nb.save(mgh_img, mgh_fname)
    with pytest.raises(TypeError):
        load_nifti(mgh_fname)


def test_save_nifti_writes_file(tmp_path: Path) -> None:
    arr = np.random.randint(0, 255, (10, 10, 10), dtype=np.int16)
    fname = tmp_path / "test.nii"
    save_nifti(fname, arr)
    assert fname.exists()


def test_save_nifti_accepts_torch_tensor(tmp_path: Path) -> None:
    data = torch.ones((4, 4, 4))
    fname = tmp_path / "test.nii"
    save_nifti(fname, data)
    loaded, _, _ = load_nifti(fname)
    assert np.array_equal(loaded, data.numpy())


def test_save_nifti_raises_for_invalid_type(tmp_path: Path) -> None:
    fname = tmp_path / "test.nii"
    with pytest.raises(TypeError):
        save_nifti(fname, [1, 2, 3])  # type: ignore[arg-type]


def test_save_nifti_default_affine_is_identity(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    fname = tmp_path / "test.nii"
    save_nifti(fname, arr)
    _, affine, _ = load_nifti(fname)
    assert np.array_equal(affine, np.eye(4))


def test_save_nifti_uses_given_affine(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    fname = tmp_path / "test.nii"
    save_nifti(fname, arr, affine=affine)
    _, loaded_affine, _ = load_nifti(fname)
    assert np.allclose(loaded_affine, affine)


def test_save_nifti_uses_given_header(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    header = nb.Nifti1Header()
    header["descrip"] = b"hello"
    fname = tmp_path / "test.nii"
    save_nifti(fname, arr, header=header)
    _, _, loaded_header = load_nifti(fname)
    assert bytes(loaded_header["descrip"]).rstrip(b"\x00") == b"hello"


def test_save_nifti_downcasts_int64_to_int32(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int64)
    fname = tmp_path / "test.nii"
    save_nifti(fname, arr)
    _, _, header = load_nifti(fname)
    assert header.get_data_dtype() == np.int32


def test_save_nifti_permute_moves_first_axis_to_last(tmp_path: Path) -> None:
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    fname = tmp_path / "test.nii"
    save_nifti(fname, data, permute=True)
    loaded, _, _ = load_nifti(fname)
    assert loaded.shape == (3, 4, 5, 2)
    assert np.array_equal(loaded, np.moveaxis(data, 0, -1))


def test_save_nifti_creates_missing_parent_directories(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    fname = tmp_path / "nested" / "dir" / "test.nii"
    save_nifti(fname, arr)
    assert fname.exists()


def test_save_nifti_niivue_scales_varying_data_to_int16_range(
    tmp_path: Path,
) -> None:
    data = np.array(
        [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=np.float32
    )
    fname = tmp_path / "test.nii"
    save_nifti(fname, data, niivue=True)
    loaded, _, header = load_nifti(fname)
    assert header.get_data_dtype() == np.int16
    assert loaded.min() == np.iinfo(np.int16).min
    assert loaded.max() == np.iinfo(np.int16).max


def test_save_nifti_niivue_constant_data_becomes_int16_min(tmp_path: Path) -> None:
    data = np.full((3, 3, 3), 5.0, dtype=np.float32)
    fname = tmp_path / "test.nii"
    save_nifti(fname, data, niivue=True)
    loaded, _, header = load_nifti(fname)
    assert header.get_data_dtype() == np.int16
    assert np.all(loaded == np.iinfo(np.int16).min)


# --- save_zarr / load_zarr ---


def test_save_zarr_roundtrip(tmp_path: Path) -> None:
    arr = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float32)
    fname = tmp_path / "test.zarr"
    save_zarr(fname, arr, {"foo": "bar", "n": 1})
    data, attrs = load_zarr(fname)
    assert np.array_equal(data, arr)
    assert data.dtype == arr.dtype
    assert attrs == {"foo": "bar", "n": 1}


def test_save_zarr_accepts_torch_tensor(tmp_path: Path) -> None:
    data = torch.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).float()
    fname = tmp_path / "test.zarr"
    save_zarr(fname, data, {})
    loaded, _ = load_zarr(fname)
    assert np.array_equal(loaded, data.numpy())


def test_save_zarr_chunks_are_batch_first(tmp_path: Path) -> None:
    arr = np.zeros((2, 3, 4, 5), dtype=np.float32)
    fname = tmp_path / "test.zarr"
    save_zarr(fname, arr, {})
    z = zarr.open(fname, mode="r")
    assert isinstance(z, zarr.Array)
    assert z.chunks == (1, 3, 4, 5)


def test_save_zarr_creates_missing_parent_directories(tmp_path: Path) -> None:
    arr = np.zeros(SMALL_SHAPE, dtype=np.float32)
    fname = tmp_path / "nested" / "dir" / "test.zarr"
    save_zarr(fname, arr, {})
    assert fname.exists()


def test_save_zarr_overwrites_existing_array(tmp_path: Path) -> None:
    fname = tmp_path / "test.zarr"
    save_zarr(fname, np.ones((2, 2, 2)), {"v": 1})
    save_zarr(fname, np.zeros((3, 3, 3)), {"v": 2})
    data, attrs = load_zarr(fname)
    assert data.shape == (3, 3, 3)
    assert attrs == {"v": 2}


def test_load_zarr_group_with_single_array_returns_it(tmp_path: Path) -> None:
    fname = tmp_path / "test.zarr"
    group = zarr.open_group(fname, mode="w")
    arr = group.create_array("only", shape=(2, 2, 2), dtype="f4")
    arr[:] = 5.0
    arr.attrs["k"] = "v"
    data, attrs = load_zarr(fname)
    assert np.all(data == 5.0)
    assert attrs == {"k": "v"}


def test_load_zarr_group_with_multiple_arrays_raises(tmp_path: Path) -> None:
    fname = tmp_path / "test.zarr"
    group = zarr.open_group(fname, mode="w")
    group.create_array("x", shape=(2, 2, 2), dtype="f4")
    group.create_array("y", shape=(2, 2, 2), dtype="f4")
    with pytest.raises(ValueError):
        load_zarr(fname)


# --- save_mesh ---


def test_save_mesh_writes_file(tmp_path: Path) -> None:
    data = _sphere_mask()
    fname = tmp_path / "mesh"
    save_mesh(fname, data)
    assert fname.exists()


def test_save_mesh_writes_valid_geometry(tmp_path: Path) -> None:
    data = _sphere_mask()
    fname = tmp_path / "mesh"
    save_mesh(fname, data)
    verts, faces = read_geometry(fname)
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0


def test_save_mesh_accepts_torch_tensor(tmp_path: Path) -> None:
    data = torch.from_numpy(_sphere_mask())
    fname = tmp_path / "mesh"
    save_mesh(fname, data)
    assert fname.exists()


def test_save_mesh_raises_for_non_3d_input(tmp_path: Path) -> None:
    fname = tmp_path / "mesh"
    with pytest.raises(ValueError):
        save_mesh(fname, np.zeros((3, 3)))


def test_save_mesh_raises_for_invalid_type(tmp_path: Path) -> None:
    fname = tmp_path / "mesh"
    with pytest.raises(ValueError):
        save_mesh(fname, [1, 2, 3])  # type: ignore[arg-type]


def test_save_mesh_creates_missing_parent_directories(tmp_path: Path) -> None:
    data = _sphere_mask()
    fname = tmp_path / "nested" / "dir" / "mesh"
    save_mesh(fname, data)
    assert fname.exists()
