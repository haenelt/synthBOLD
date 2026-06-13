"""Tests for utility functions."""

import nibabel as nb
import numpy as np
import pytest

from synthbold.io import load_nifti, save_nifti


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
