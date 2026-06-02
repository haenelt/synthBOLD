import torch

from synthbold.geometries import Cylinders, Shapes

SMALL_SHAPE = (16, 16, 16)
SMALL_FOV = (4.0, 4.0, 4.0)


# --- Shapes ---


def test_shapes_output_shape() -> None:
    s = Shapes(shape=SMALL_SHAPE, J=4, device="cpu", seed=0)
    out = s.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_shapes_reproducibility() -> None:
    s1 = Shapes(shape=SMALL_SHAPE, J=4, device="cpu", seed=42)
    s2 = Shapes(shape=SMALL_SHAPE, J=4, device="cpu", seed=42)
    assert torch.equal(s1.forward(), s2.forward())


def test_shapes_different_seeds() -> None:
    s1 = Shapes(shape=SMALL_SHAPE, J=4, device="cpu", seed=1)
    s2 = Shapes(shape=SMALL_SHAPE, J=4, device="cpu", seed=2)
    assert not torch.equal(s1.forward(), s2.forward())


def test_shapes_call_returns_batch() -> None:
    s = Shapes(shape=SMALL_SHAPE, J=3, device="cpu", seed=0)
    batch = s(n_sample=4)
    assert batch.shape == torch.Size((4, *SMALL_SHAPE))


def test_shapes_attrs() -> None:
    s = Shapes(shape=SMALL_SHAPE, J=5, device="cpu", seed=0)
    attrs = s.attrs
    assert attrs["generator"] == "Shapes"
    assert attrs["J"] == 5
    assert attrs["shape"] == SMALL_SHAPE


# --- Cylinders ---


def test_cylinders_num_cylinders_output_shape() -> None:
    c = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_cylinders=3,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    out = c.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_cylinders_vf_range_output_shape() -> None:
    c = Cylinders(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = c.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_cylinders_vf_range_nonzero_voxels() -> None:
    c = Cylinders(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = c.forward()
    labeled_fraction = (out > 0).float().mean().item()
    assert labeled_fraction > 0.0


def test_cylinders_reproducibility() -> None:
    c1 = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_cylinders=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    c2 = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_cylinders=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    assert torch.equal(c1.forward(), c2.forward())


def test_cylinders_different_seeds() -> None:
    c1 = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_cylinders=3,
        vf_range=None,
        device="cpu",
        seed=1,
    )
    c2 = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_cylinders=3,
        vf_range=None,
        device="cpu",
        seed=2,
    )
    assert not torch.equal(c1.forward(), c2.forward())


def test_cylinders_call_returns_batch() -> None:
    c = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_cylinders=2,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    batch = c(n_sample=3)
    assert batch.shape == torch.Size((3, *SMALL_SHAPE))


def test_cylinders_attrs() -> None:
    c = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_cylinders=2,
        vf_range=None,
        diameter_range=(0.25, 1.0),
        device="cpu",
        seed=0,
    )
    attrs = c.attrs
    assert attrs["generator"] == "Cylinders"
    assert attrs["fov"] == SMALL_FOV
    assert attrs["num_cylinders"] == 2
    assert attrs["diameter_range"] == (0.25, 1.0)
