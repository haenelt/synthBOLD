import pytest
import torch

from synthbold.models.functional import (
    dchi_to_dbz,
    merge_labels,
    spherical_to_cartesian,
)

SMALL_SHAPE = (4, 4, 4)


# --- merge_labels ---


def test_raises_with_fewer_than_two_labels() -> None:
    labels = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    with pytest.raises(ValueError):
        merge_labels(labels)


def test_raises_on_shape_mismatch() -> None:
    a = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    b = torch.zeros((1, 2, 2, 2), dtype=torch.int64)
    with pytest.raises(ValueError):
        merge_labels(a, b)


def test_output_shape_and_dtype_match_first_input() -> None:
    a = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    b = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    out = merge_labels(a, b)
    assert out.shape == a.shape
    assert out.dtype == a.dtype


def test_all_background_stays_background() -> None:
    a = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    b = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    out = merge_labels(a, b)
    assert torch.all(out == 0)


def test_non_overlapping_labels_are_combined() -> None:
    a = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    a[0, 0, 0] = 1
    b = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    b[1, 1, 1] = 1
    out = merge_labels(a, b)
    assert out[0, 0, 0] == 1
    assert out[1, 1, 1] == 2


def test_earlier_tensor_takes_precedence_on_overlap() -> None:
    a = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    a[0, 0, 0] = 1
    b = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    b[0, 0, 0] = 1
    out = merge_labels(a, b)
    assert out[0, 0, 0] == 1


def test_second_tensor_offset_by_first_tensor_max() -> None:
    a = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    a[0, 0, 0] = 3
    b = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    b[1, 1, 1] = 2
    out = merge_labels(a, b)
    assert out[0, 0, 0] == 3
    assert out[1, 1, 1] == 5


def test_labels_remain_disjoint_across_three_tensors() -> None:
    a = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    a[0, 0, 0] = 1
    b = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    b[1, 1, 1] = 1
    c = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    c[2, 2, 2] = 1
    out = merge_labels(a, b, c)
    assert out[0, 0, 0] == 1
    assert out[1, 1, 1] == 2
    assert out[2, 2, 2] == 3


def test_does_not_modify_input_tensors() -> None:
    a = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    a[0, 0, 0] = 1
    b = torch.zeros(SMALL_SHAPE, dtype=torch.int64)
    b[0, 0, 0] = 1
    a_before = a.clone()
    b_before = b.clone()
    merge_labels(a, b)
    assert torch.equal(a, a_before)
    assert torch.equal(b, b_before)


def test_batched_input_merges_independently_per_batch() -> None:
    a = torch.zeros((2, *SMALL_SHAPE), dtype=torch.int64)
    a[0, 0, 0, 0] = 1
    a[1, 1, 1, 1] = 1
    b = torch.zeros((2, *SMALL_SHAPE), dtype=torch.int64)
    b[0, 2, 2, 2] = 1
    b[1, 3, 3, 3] = 1
    out = merge_labels(a, b)
    assert out[0, 0, 0, 0] == 1
    assert out[0, 2, 2, 2] == 2
    assert out[1, 1, 1, 1] == 1
    assert out[1, 3, 3, 3] == 2


# --- spherical_to_cartesian ---


def test_output_shape() -> None:
    theta = torch.tensor([0.0, torch.pi / 2, torch.pi])
    phi = torch.tensor([0.0, torch.pi / 2, torch.pi])
    out = spherical_to_cartesian(theta, phi)
    assert out.shape == (3, 3)


def test_vectors_are_unit_norm() -> None:
    theta = torch.tensor([0.3, 1.2, 2.5])
    phi = torch.tensor([0.1, 3.0, -1.7])
    out = spherical_to_cartesian(theta, phi)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-6)


def test_north_pole_points_along_z() -> None:
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    out = spherical_to_cartesian(theta, phi)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


def test_south_pole_points_along_negative_z() -> None:
    theta = torch.tensor([torch.pi])
    phi = torch.tensor([0.0])
    out = spherical_to_cartesian(theta, phi)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6)


def test_equator_phi_zero_points_along_x() -> None:
    theta = torch.tensor([torch.pi / 2])
    phi = torch.tensor([0.0])
    out = spherical_to_cartesian(theta, phi)
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_equator_phi_quarter_turn_points_along_y() -> None:
    theta = torch.tensor([torch.pi / 2])
    phi = torch.tensor([torch.pi / 2])
    out = spherical_to_cartesian(theta, phi)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_batch_elements_computed_independently() -> None:
    theta = torch.tensor([0.0, torch.pi / 2])
    phi = torch.tensor([0.0, 0.0])
    out = spherical_to_cartesian(theta, phi)
    assert torch.allclose(out[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


# --- dchi_to_dbz ---

DIPOLE_SHAPE = (12, 12, 12)
DIPOLE_PADDING = 12


def _point_source(amplitude: float = 1.0) -> tuple[torch.Tensor, tuple[int, int, int]]:
    x, y, z = DIPOLE_SHAPE
    center = (x // 2, y // 2, z // 2)
    chi = torch.zeros(1, x, y, z)
    chi[0, *center] = amplitude
    return chi, center


def test_output_shape_and_dtype_match_chi() -> None:
    chi, _ = _point_source()
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    assert dbz.shape == chi.shape
    assert dbz.dtype == chi.dtype


def test_zero_susceptibility_gives_zero_field() -> None:
    chi = torch.zeros(1, *DIPOLE_SHAPE)
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    assert torch.allclose(dbz, torch.zeros_like(dbz), atol=1e-5)


def test_field_at_source_voxel_is_zero() -> None:
    # The dipole kernel is defined to be zero at its own origin.
    chi, center = _point_source()
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    assert dbz[0, *center].item() == pytest.approx(0.0, abs=1e-5)


def test_field_along_b0_axis_matches_dipole_formula() -> None:
    # For a unit point source with B0 along z, dbz(0,0,1) = b0*(3*1-1)/1**2.5 = 2*b0.
    chi, (cx, cy, cz) = _point_source()
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    assert dbz[0, cx, cy, cz + 1].item() == pytest.approx(2.0, abs=1e-4)
    assert dbz[0, cx, cy, cz - 1].item() == pytest.approx(2.0, abs=1e-4)


def test_field_perpendicular_to_b0_axis_matches_dipole_formula() -> None:
    # For a unit point source with B0 along z, dbz(1,0,0) = b0*(3*0-1)/1**2.5 = -b0.
    chi, (cx, cy, cz) = _point_source()
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    assert dbz[0, cx + 1, cy, cz].item() == pytest.approx(-1.0, abs=1e-4)
    assert dbz[0, cx, cy + 1, cz].item() == pytest.approx(-1.0, abs=1e-4)


def test_field_pattern_rotates_with_b0_direction() -> None:
    # With B0 along x (theta=pi/2, phi=0), the axis/perpendicular roles swap: the
    # field is positive along x and negative along y and z.
    chi, (cx, cy, cz) = _point_source()
    theta = torch.tensor([torch.pi / 2])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    assert dbz[0, cx + 1, cy, cz].item() == pytest.approx(2.0, abs=1e-4)
    assert dbz[0, cx, cy + 1, cz].item() == pytest.approx(-1.0, abs=1e-4)
    assert dbz[0, cx, cy, cz + 1].item() == pytest.approx(-1.0, abs=1e-4)


def test_linear_in_chi() -> None:
    chi, (cx, cy, cz) = _point_source()
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    dbz_scaled = dchi_to_dbz(chi * 3.0, DIPOLE_PADDING, 1.0, theta, phi)
    assert torch.allclose(dbz_scaled, dbz * 3.0, atol=1e-4)


def test_linear_in_b0() -> None:
    chi, _ = _point_source()
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    dbz_scaled = dchi_to_dbz(chi, DIPOLE_PADDING, 2.5, theta, phi)
    assert torch.allclose(dbz_scaled, dbz * 2.5, atol=1e-4)


def test_int_padding_matches_equivalent_tuple_padding() -> None:
    chi, _ = _point_source()
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    dbz_int = dchi_to_dbz(chi, DIPOLE_PADDING, 1.0, theta, phi)
    dbz_tuple = dchi_to_dbz(
        chi, (DIPOLE_PADDING, DIPOLE_PADDING, DIPOLE_PADDING), 1.0, theta, phi
    )
    assert torch.allclose(dbz_int, dbz_tuple)


def test_dbz_batch_elements_computed_independently() -> None:
    chi_a, _ = _point_source(amplitude=1.0)
    chi_b, _ = _point_source(amplitude=2.0)
    theta_a, phi_a = torch.tensor([0.0]), torch.tensor([0.0])
    theta_b, phi_b = torch.tensor([torch.pi / 2]), torch.tensor([0.0])

    out_a = dchi_to_dbz(chi_a, DIPOLE_PADDING, 1.0, theta_a, phi_a)
    out_b = dchi_to_dbz(chi_b, DIPOLE_PADDING, 1.0, theta_b, phi_b)

    chi_batch = torch.cat([chi_a, chi_b], dim=0)
    theta_batch = torch.cat([theta_a, theta_b])
    phi_batch = torch.cat([phi_a, phi_b])
    out_batch = dchi_to_dbz(chi_batch, DIPOLE_PADDING, 1.0, theta_batch, phi_batch)

    assert torch.allclose(out_batch[0], out_a[0], atol=1e-5)
    assert torch.allclose(out_batch[1], out_b[0], atol=1e-5)
