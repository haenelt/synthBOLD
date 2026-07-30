import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from synthbold.config import Config
from synthbold.models.signal import PerturbationModel, SignalModel

CHI_RANGE = (-2.0, 2.0)
B0 = 7.0
THETA_RANGE = (0.0, math.pi)
PHI_RANGE = (0.0, 2 * math.pi)
PADDING = 2
SMALL_SHAPE = (4, 4, 4)

OUTPUT_SHAPE = (2, 2, 2)
T2_RANGE = (20.0, 80.0)
TE_RANGE = (5.0, 30.0)
SPIN_ECHO = None
GAMMA = 2.675221e5


# --- PerturbationModel ---


def test_init_stores_params() -> None:
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=0,
    )
    assert model.chi_range == CHI_RANGE
    assert model.b0 == B0
    assert model.theta_range == THETA_RANGE
    assert model.phi_range == PHI_RANGE


def test_init_padding_int_broadcasts_to_tuple() -> None:
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=3,
        device="cpu",
        seed=0,
    )
    assert model.padding == (3, 3, 3)


def test_init_padding_tuple_passthrough() -> None:
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=(1, 2, 3),
        device="cpu",
        seed=0,
    )
    assert model.padding == (1, 2, 3)


def test_forward_output_shape_batched() -> None:
    data = torch.randint(0, 3, (2, *SMALL_SHAPE), dtype=torch.int64)
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=0,
    )
    out = model(data)
    assert out.shape == data.shape


def test_forward_dchi_length_mismatch_raises() -> None:
    data = torch.ones((1, *SMALL_SHAPE), dtype=torch.int64)
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=0,
    )
    with pytest.raises(ValueError, match="dchi"):
        model(data, dchi=torch.zeros(5))


def test_forward_theta_length_mismatch_raises() -> None:
    data = torch.ones((2, *SMALL_SHAPE), dtype=torch.int64)
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=0,
    )
    with pytest.raises(ValueError, match="theta"):
        model(data, theta=torch.zeros(1))


def test_forward_phi_length_mismatch_raises() -> None:
    data = torch.ones((2, *SMALL_SHAPE), dtype=torch.int64)
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=0,
    )
    with pytest.raises(ValueError, match="phi"):
        model(data, phi=torch.zeros(1))


def test_forward_scalar_theta_raises_value_error() -> None:
    data = torch.ones((2, *SMALL_SHAPE), dtype=torch.int64)
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=0,
    )
    with pytest.raises(ValueError, match="theta"):
        model(data, theta=torch.tensor(0.5))


def test_forward_all_background_gives_zero_output() -> None:
    data = torch.zeros((1, *SMALL_SHAPE), dtype=torch.int64)
    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=0,
    )
    out = model(data)
    assert torch.all(out == 0.0)


def test_forward_reproducible_with_same_seed() -> None:
    data = torch.randint(0, 3, (2, *SMALL_SHAPE), dtype=torch.int64)
    model1 = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=42,
    )
    model2 = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=42,
    )
    assert torch.equal(model1(data), model2(data))


def test_forward_different_seeds_produce_different_output() -> None:
    data = torch.ones((1, *SMALL_SHAPE), dtype=torch.int64)
    model1 = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=1,
    )
    model2 = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=2,
    )
    assert not torch.equal(model1(data), model2(data))


def test_forward_explicit_params_override_randomness() -> None:
    # With dchi/theta/phi all supplied, output must be fully determined by the
    # inputs, regardless of the model's own seed.
    data = torch.ones((1, *SMALL_SHAPE), dtype=torch.int64)
    dchi = torch.tensor([1.0])
    theta = torch.tensor([math.pi / 3])
    phi = torch.tensor([math.pi / 4])

    model1 = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=1,
    )
    model2 = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=PADDING,
        device="cpu",
        seed=2,
    )

    out1 = model1(data, dchi=dchi, theta=theta, phi=phi)
    out2 = model2(data, dchi=dchi, theta=theta, phi=phi)
    assert torch.equal(out1, out2)


def test_forward_assigns_dchi_per_label_with_batch_offsets() -> None:
    # batch 0 has labels {1, 2}; batch 1 has label {1}. dchi is laid out flat as
    # [batch0-label1, batch0-label2, batch1-label1]; the offset logic in forward
    # must route each voxel to its correct entry despite the shared label ids.
    # dchi_to_dbz is patched to the identity so the assigned susceptibility volume
    # can be checked directly, without going through the FFT-based physics.
    data = torch.zeros((2, 2, 2, 2), dtype=torch.int64)
    data[0, 0, 0, 0] = 1
    data[0, 1, 1, 1] = 2
    data[1, 0, 0, 0] = 1

    dchi = torch.tensor([10.0, 20.0, 30.0])
    theta = torch.tensor([0.0, 0.0])
    phi = torch.tensor([0.0, 0.0])

    model = PerturbationModel(
        chi_range=CHI_RANGE,
        b0=B0,
        theta_range=THETA_RANGE,
        phi_range=PHI_RANGE,
        padding=1,
        device="cpu",
        seed=0,
    )
    with patch("synthbold.models.signal.dchi_to_dbz") as mock_dbz:
        mock_dbz.side_effect = lambda chi, *args, **kwargs: chi
        out = model(data, dchi=dchi, theta=theta, phi=phi)

    assert out[0, 0, 0, 0].item() == pytest.approx(10.0)
    assert out[0, 1, 1, 1].item() == pytest.approx(20.0)
    assert out[1, 0, 0, 0].item() == pytest.approx(30.0)
    assert torch.all(out[data == 0] == 0.0)


def test_from_config_returns_instance() -> None:
    config = Config()
    model = PerturbationModel.from_config(config)
    assert isinstance(model, PerturbationModel)


def test_from_config_params_match_config() -> None:
    config = Config()
    model = PerturbationModel.from_config(config)
    assert model.chi_range == (config.physics.chi.min, config.physics.chi.max)
    assert model.b0 == config.physics.b0
    assert model.theta_range == (config.physics.theta.min, config.physics.theta.max)
    assert model.phi_range == (config.physics.phi.min, config.physics.phi.max)
    assert model.padding == (
        config.geom.padding,
        config.geom.padding,
        config.geom.padding,
    )
    assert model.device == torch.device(config.device)
    assert model.seed == config.seed


# --- SignalModel ---


def _make_vessel_tissue_dbz(
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tissue = torch.rand((batch, *SMALL_SHAPE))
    vessel = torch.zeros((batch, *SMALL_SHAPE), dtype=torch.int64)
    vessel[:, 0, 0, 0] = 1
    dbz = torch.rand((batch, *SMALL_SHAPE)) * 1e-6
    return tissue, vessel, dbz


def test_signal_init_stores_params() -> None:
    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    assert model.shape == OUTPUT_SHAPE
    assert model.t2_range == T2_RANGE
    assert model.te_range == TE_RANGE
    assert model.spin_echo == SPIN_ECHO
    assert model.b0 == B0
    assert model.gamma == GAMMA


def test_signal_call_matches_forward() -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz()
    t2 = torch.tensor([40.0])
    te = torch.tensor([10.0])

    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    out_call = model(tissue.clone(), vessel.clone(), dbz.clone(), t2=t2, te=te)
    out_forward = model.forward(
        tissue.clone(), vessel.clone(), dbz.clone(), t2=t2, te=te
    )

    for a, b in zip(out_call, out_forward, strict=True):
        assert torch.equal(a, b)


def test_signal_forward_output_shapes() -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz(batch=2)
    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    out = model(tissue, vessel, dbz)

    expected_shape = (2, *OUTPUT_SHAPE)
    assert out.magnitude.shape == expected_shape
    assert out.phase.shape == expected_shape
    assert out.tissue.shape == expected_shape
    assert out.vessel.shape == expected_shape
    assert out.mean.shape == expected_shape
    assert out.variance.shape == expected_shape
    assert out.skewness.shape == expected_shape
    assert out.kurtosis.shape == expected_shape
    assert out.te.shape == (2, 1, 1, 1)


def test_signal_forward_magnitude_is_nonnegative() -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz()
    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    out = model(tissue, vessel, dbz)
    assert torch.all(out.magnitude >= 0.0)


def test_signal_forward_reproducible_with_same_seed() -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz()

    model1 = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=42,
    )
    model2 = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=42,
    )

    out1 = model1(tissue.clone(), vessel.clone(), dbz.clone())
    out2 = model2(tissue.clone(), vessel.clone(), dbz.clone())
    for a, b in zip(out1, out2, strict=True):
        assert torch.equal(a, b)


def test_signal_forward_different_seeds_produce_different_output() -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz()

    model1 = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=1,
    )
    model2 = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=2,
    )

    out1 = model1(tissue.clone(), vessel.clone(), dbz.clone())
    out2 = model2(tissue.clone(), vessel.clone(), dbz.clone())
    assert not torch.equal(out1.magnitude, out2.magnitude)


def test_signal_forward_explicit_t2_te_override_randomness() -> None:
    # With t2/te both supplied, output must be fully determined by the inputs,
    # regardless of the model's own seed.
    tissue, vessel, dbz = _make_vessel_tissue_dbz()
    t2 = torch.tensor([40.0])
    te = torch.tensor([10.0])

    model1 = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=1,
    )
    model2 = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=2,
    )

    out1 = model1(tissue.clone(), vessel.clone(), dbz.clone(), t2=t2, te=te)
    out2 = model2(tissue.clone(), vessel.clone(), dbz.clone(), t2=t2, te=te)
    for a, b in zip(out1, out2, strict=True):
        assert torch.equal(a, b)


def test_signal_forward_raises_on_non_divisible_shape() -> None:
    tissue = torch.rand((1, 3, 3, 3))
    vessel = torch.zeros((1, 3, 3, 3), dtype=torch.int64)
    dbz = torch.zeros((1, 3, 3, 3))

    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    with pytest.raises(ValueError, match="not evenly divisible"):
        model(tissue, vessel, dbz)


def test_signal_from_config_returns_instance() -> None:
    config = Config()
    model = SignalModel.from_config(config)
    assert isinstance(model, SignalModel)


def test_signal_from_config_params_match_config() -> None:
    config = Config()
    model = SignalModel.from_config(config)
    assert model.shape == config.geom.output_shape
    assert model.t2_range == (config.physics.t2.min, config.physics.t2.max)
    assert model.te_range == (config.physics.te.min, config.physics.te.max)
    assert model.spin_echo == config.physics.spin_echo
    assert model.b0 == config.physics.b0
    assert model.gamma == config.physics.gamma
    assert model.device == torch.device(config.device)
    assert model.seed == config.seed


def test_signal_call_save_zarr_creates_one_file_per_field(tmp_path: Path) -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz()
    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    out = model(tissue, vessel, dbz, dirname=tmp_path, fmt="zarr")

    for field in out._fields:
        assert (tmp_path / f"{field}.zarr").exists()


def test_signal_call_save_nifti_creates_one_file_per_field(tmp_path: Path) -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz()
    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    out = model(tissue, vessel, dbz, dirname=tmp_path, fmt="nii")

    for field in out._fields:
        assert (tmp_path / f"{field}.nii").exists()


def test_signal_call_invalid_fmt_raises(tmp_path: Path) -> None:
    tissue, vessel, dbz = _make_vessel_tissue_dbz()
    model = SignalModel(
        shape=OUTPUT_SHAPE,
        t2_range=T2_RANGE,
        te_range=TE_RANGE,
        spin_echo=SPIN_ECHO,
        b0=B0,
        gamma=GAMMA,
        device="cpu",
        seed=0,
    )
    with pytest.raises(ValueError, match="fmt must be"):
        model(tissue, vessel, dbz, dirname=tmp_path, fmt="mgz")
