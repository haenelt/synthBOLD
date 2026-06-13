import torch

from synthbold.geometries import Cubes, Cylinders, Shapes, Spheres, Tetrahedra, Toroids

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


def test_cylinders_num_objects_output_shape() -> None:
    c = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
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
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    c2 = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    assert torch.equal(c1.forward(), c2.forward())


def test_cylinders_different_seeds() -> None:
    c1 = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=1,
    )
    c2 = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=2,
    )
    assert not torch.equal(c1.forward(), c2.forward())


def test_cylinders_call_returns_batch() -> None:
    c = Cylinders(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
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
        num_objects=2,
        vf_range=None,
        diameter_range=(0.25, 1.0),
        device="cpu",
        seed=0,
    )
    attrs = c.attrs
    assert attrs["generator"] == "Cylinders"
    assert attrs["fov"] == SMALL_FOV
    assert attrs["num_objects"] == 2
    assert attrs["diameter_range"] == (0.25, 1.0)


# --- Spheres ---


def test_spheres_num_objects_output_shape() -> None:
    s = Spheres(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    out = s.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_spheres_vf_range_output_shape() -> None:
    s = Spheres(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = s.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_spheres_vf_range_nonzero_voxels() -> None:
    s = Spheres(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = s.forward()
    labeled_fraction = (out > 0).float().mean().item()
    assert labeled_fraction > 0.0


def test_spheres_reproducibility() -> None:
    s1 = Spheres(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    s2 = Spheres(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    assert torch.equal(s1.forward(), s2.forward())


def test_spheres_different_seeds() -> None:
    s1 = Spheres(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=1,
    )
    s2 = Spheres(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=2,
    )
    assert not torch.equal(s1.forward(), s2.forward())


def test_spheres_call_returns_batch() -> None:
    s = Spheres(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    batch = s(n_sample=3)
    assert batch.shape == torch.Size((3, *SMALL_SHAPE))


def test_spheres_attrs() -> None:
    s = Spheres(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        diameter_range=(0.25, 1.0),
        device="cpu",
        seed=0,
    )
    attrs = s.attrs
    assert attrs["generator"] == "Spheres"
    assert attrs["fov"] == SMALL_FOV
    assert attrs["num_objects"] == 2
    assert attrs["diameter_range"] == (0.25, 1.0)


# --- Tetrahedra ---


def test_tetrahedra_num_objects_output_shape() -> None:
    t = Tetrahedra(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    out = t.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_tetrahedra_vf_range_output_shape() -> None:
    t = Tetrahedra(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = t.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_tetrahedra_vf_range_nonzero_voxels() -> None:
    t = Tetrahedra(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = t.forward()
    labeled_fraction = (out > 0).float().mean().item()
    assert labeled_fraction > 0.0


def test_tetrahedra_reproducibility() -> None:
    t1 = Tetrahedra(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    t2 = Tetrahedra(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    assert torch.equal(t1.forward(), t2.forward())


def test_tetrahedra_different_seeds() -> None:
    t1 = Tetrahedra(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=1,
    )
    t2 = Tetrahedra(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=2,
    )
    assert not torch.equal(t1.forward(), t2.forward())


def test_tetrahedra_call_returns_batch() -> None:
    t = Tetrahedra(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    batch = t(n_sample=3)
    assert batch.shape == torch.Size((3, *SMALL_SHAPE))


def test_tetrahedra_attrs() -> None:
    t = Tetrahedra(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        diameter_range=(0.25, 1.0),
        device="cpu",
        seed=0,
    )
    attrs = t.attrs
    assert attrs["generator"] == "Tetrahedra"
    assert attrs["fov"] == SMALL_FOV
    assert attrs["num_objects"] == 2
    assert attrs["diameter_range"] == (0.25, 1.0)


# --- Cubes ---


def test_cubes_num_objects_output_shape() -> None:
    c = Cubes(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    out = c.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_cubes_vf_range_output_shape() -> None:
    c = Cubes(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = c.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_cubes_vf_range_nonzero_voxels() -> None:
    c = Cubes(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = c.forward()
    labeled_fraction = (out > 0).float().mean().item()
    assert labeled_fraction > 0.0


def test_cubes_reproducibility() -> None:
    c1 = Cubes(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    c2 = Cubes(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    assert torch.equal(c1.forward(), c2.forward())


def test_cubes_different_seeds() -> None:
    c1 = Cubes(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=1,
    )
    c2 = Cubes(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=2,
    )
    assert not torch.equal(c1.forward(), c2.forward())


def test_cubes_call_returns_batch() -> None:
    c = Cubes(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    batch = c(n_sample=3)
    assert batch.shape == torch.Size((3, *SMALL_SHAPE))


def test_cubes_attrs() -> None:
    c = Cubes(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        diameter_range=(0.25, 1.0),
        device="cpu",
        seed=0,
    )
    attrs = c.attrs
    assert attrs["generator"] == "Cubes"
    assert attrs["fov"] == SMALL_FOV
    assert attrs["num_objects"] == 2
    assert attrs["diameter_range"] == (0.25, 1.0)


# --- Toroids ---


def test_toroids_num_objects_output_shape() -> None:
    t = Toroids(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    out = t.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_toroids_vf_range_output_shape() -> None:
    t = Toroids(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = t.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)


def test_toroids_vf_range_nonzero_voxels() -> None:
    t = Toroids(
        shape=SMALL_SHAPE, fov=SMALL_FOV, vf_range=(0.05, 0.15), device="cpu", seed=0
    )
    out = t.forward()
    labeled_fraction = (out > 0).float().mean().item()
    assert labeled_fraction > 0.0


def test_toroids_reproducibility() -> None:
    t1 = Toroids(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    t2 = Toroids(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=99,
    )
    assert torch.equal(t1.forward(), t2.forward())


def test_toroids_different_seeds() -> None:
    t1 = Toroids(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=1,
    )
    t2 = Toroids(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=3,
        vf_range=None,
        device="cpu",
        seed=2,
    )
    assert not torch.equal(t1.forward(), t2.forward())


def test_toroids_call_returns_batch() -> None:
    t = Toroids(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        device="cpu",
        seed=0,
    )
    batch = t(n_sample=3)
    assert batch.shape == torch.Size((3, *SMALL_SHAPE))


def test_toroids_attrs() -> None:
    t = Toroids(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        num_objects=2,
        vf_range=None,
        diameter_range=(0.25, 1.0),
        tube_ratio_range=(0.1, 0.3),
        device="cpu",
        seed=0,
    )
    attrs = t.attrs
    assert attrs["generator"] == "Toroids"
    assert attrs["fov"] == SMALL_FOV
    assert attrs["num_objects"] == 2
    assert attrs["diameter_range"] == (0.25, 1.0)
    assert attrs["tube_ratio_range"] == (0.1, 0.3)
