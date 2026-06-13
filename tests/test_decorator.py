import pytest
import torch

from synthbold.decorator import accept_unbatched, require_dim


class DummyTransform:
    @require_dim(3, 4)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2


@require_dim(3, 4)
def standalone(data: torch.Tensor) -> torch.Tensor:
    return data * 2


@require_dim(3, 4)
def keyword_only(*, data: torch.Tensor) -> torch.Tensor:
    return data * 2


class DummyBatchTransform:
    @accept_unbatched(3)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Return the shape of the tensor received internally
        return data


@accept_unbatched(3)
def batch_standalone(data: torch.Tensor) -> torch.Tensor:
    return data


@accept_unbatched(3)
def batch_keyword_only(data: torch.Tensor) -> torch.Tensor:
    return data


# --- require_dim ---


def test_3d_tensor_method() -> None:
    t = DummyTransform()
    x = torch.ones(2, 3, 4)
    out = t.forward(x)
    assert out.shape == (2, 3, 4)
    assert torch.all(out == 2.0)


def test_4d_tensor_method() -> None:
    t = DummyTransform()
    x = torch.ones(1, 2, 3, 4)
    out = t.forward(x)
    assert out.shape == (1, 2, 3, 4)
    assert torch.all(out == 2.0)


def test_3d_tensor_standalone() -> None:
    x = torch.ones(2, 3, 4)
    out = standalone(x)
    assert out.shape == (2, 3, 4)


def test_4d_tensor_standalone() -> None:
    x = torch.ones(1, 2, 3, 4)
    out = standalone(x)
    assert out.shape == (1, 2, 3, 4)


def test_tensor_as_keyword_argument() -> None:
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
    x = torch.ones(*shape)
    with pytest.raises(ValueError):
        standalone(x)


# --- accept_unbatched ---


def test_3d_input_output_shape_preserved_method() -> None:
    t = DummyBatchTransform()
    x = torch.ones(2, 3, 4)
    out = t.forward(x)
    assert out.shape == (2, 3, 4)


def test_3d_input_output_shape_preserved_standalone() -> None:
    x = torch.ones(2, 3, 4)
    out = batch_standalone(x)
    assert out.shape == (2, 3, 4)


def test_4d_input_unchanged_method() -> None:
    t = DummyBatchTransform()
    x = torch.ones(1, 2, 3, 4)
    out = t.forward(x)
    assert out.shape == (1, 2, 3, 4)


def test_4d_input_unchanged_standalone() -> None:
    x = torch.ones(2, 2, 3, 4)
    out = batch_standalone(x)
    assert out.shape == (2, 2, 3, 4)


def test_3d_tensor_as_keyword_argument() -> None:
    x = torch.ones(2, 3, 4)
    out = batch_keyword_only(data=x)
    assert out.shape == (2, 3, 4)


def test_4d_tensor_as_keyword_argument() -> None:
    x = torch.ones(1, 2, 3, 4)
    out = batch_keyword_only(data=x)
    assert out.shape == (1, 2, 3, 4)
