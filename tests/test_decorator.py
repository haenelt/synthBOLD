"""Tests for decorators."""

import pytest
import torch

from synthbold.decorator import accept_unbatched, require_dim

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class DummyTransform:
    """Minimal class to test require_dim on methods."""

    @require_dim(3, 4)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2


@require_dim(3, 4)
def standalone(data: torch.Tensor) -> torch.Tensor:
    """Standalone function decorated with require_dim."""
    return data * 2


@require_dim(3, 4)
def keyword_only(*, data: torch.Tensor) -> torch.Tensor:
    """Function that receives the tensor as a keyword argument."""
    return data * 2


class DummyBatchTransform:
    """Minimal class to test batch_to_3d on methods."""

    @accept_unbatched(3)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Return the shape of the tensor received internally
        return data


@accept_unbatched(3)
def batch_standalone(data: torch.Tensor) -> torch.Tensor:
    """Standalone function that records the internal shape it received."""
    return data


@accept_unbatched(3)
def batch_keyword_only(data: torch.Tensor) -> torch.Tensor:
    """Function that receives the tensor passed by keyword name."""
    return data


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def test_3d_tensor_method() -> None:
    """Decorator passes through a 3D tensor on a method without error."""
    t = DummyTransform()
    x = torch.ones(2, 3, 4)
    out = t.forward(x)
    assert out.shape == (2, 3, 4)
    assert torch.all(out == 2.0)


def test_4d_tensor_method() -> None:
    """Decorator passes through a 4D tensor on a method without error."""
    t = DummyTransform()
    x = torch.ones(1, 2, 3, 4)
    out = t.forward(x)
    assert out.shape == (1, 2, 3, 4)
    assert torch.all(out == 2.0)


def test_3d_tensor_standalone() -> None:
    """Decorator passes through a 3D tensor on a standalone function."""
    x = torch.ones(2, 3, 4)
    out = standalone(x)
    assert out.shape == (2, 3, 4)


def test_4d_tensor_standalone() -> None:
    """Decorator passes through a 4D tensor on a standalone function."""
    x = torch.ones(1, 2, 3, 4)
    out = standalone(x)
    assert out.shape == (1, 2, 3, 4)


def test_tensor_as_keyword_argument() -> None:
    """Decorator finds and validates a tensor passed as a keyword argument."""
    x = torch.ones(2, 3, 4)
    out = keyword_only(data=x)
    assert out.shape == (2, 3, 4)


@pytest.mark.parametrize(
    "shape",
    [
        (4,),  # 1D
        (3, 4),  # 2D
        (1, 2, 3, 4, 5),  # 5D
    ],
)
def test_invalid_ndim_method_raises(shape: tuple[int, ...]) -> None:
    """Decorator raises ValueError for tensors that are not 3D or 4D on a method."""
    t = DummyTransform()
    x = torch.ones(*shape)
    with pytest.raises(ValueError):
        t.forward(x)


@pytest.mark.parametrize(
    "shape",
    [
        (4,),
        (3, 4),
        (1, 2, 3, 4, 5),
    ],
)
def test_invalid_ndim_standalone_raises(shape: tuple[int, ...]) -> None:
    """Decorator raises ValueError for tensors that are not 3D or 4D on a function."""
    x = torch.ones(*shape)
    with pytest.raises(ValueError):
        standalone(x)


def test_3d_input_output_shape_preserved_method() -> None:
    """3D input tensor retains its 3D shape after round-trip through the decorator."""
    t = DummyBatchTransform()
    x = torch.ones(2, 3, 4)
    out = t.forward(x)
    assert out.shape == (2, 3, 4)


def test_3d_input_output_shape_preserved_standalone() -> None:
    """3D input tensor retains its 3D shape on a standalone function."""
    x = torch.ones(2, 3, 4)
    out = batch_standalone(x)
    assert out.shape == (2, 3, 4)


def test_4d_input_unchanged_method() -> None:
    """4D input tensor is passed through untouched on a method."""
    t = DummyBatchTransform()
    x = torch.ones(1, 2, 3, 4)
    out = t.forward(x)
    assert out.shape == (1, 2, 3, 4)


def test_4d_input_unchanged_standalone() -> None:
    """4D input tensor is passed through untouched on a standalone function."""
    x = torch.ones(2, 2, 3, 4)
    out = batch_standalone(x)
    assert out.shape == (2, 2, 3, 4)


def test_3d_tensor_as_keyword_argument() -> None:
    """Decorator finds and batches a 3D tensor passed as a keyword argument."""
    x = torch.ones(2, 3, 4)
    out = batch_keyword_only(data=x)
    assert out.shape == (2, 3, 4)


def test_4d_tensor_as_keyword_argument() -> None:
    """Decorator passes through a 4D tensor given as a keyword argument unchanged."""
    x = torch.ones(1, 2, 3, 4)
    out = batch_keyword_only(data=x)
    assert out.shape == (1, 2, 3, 4)
