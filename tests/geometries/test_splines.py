import sys
import types
from typing import Any

import pytest
import torch

from synthbold.config import Config, GeometryParams
from synthbold.geometries import SplineVessels

SMALL_SHAPE = (8, 8, 8)
SMALL_FOV = (4.0, 4.0, 4.0)


class _FakeSample:
    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels


class _FakeSynthSplineBlock:
    """Stand-in for `synthspline.labelsynth.SynthSplineBlock` that records the kwargs
    it was constructed with and returns deterministic, correctly-shaped labels."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, batch: int = 1) -> _FakeSample:
        shape = self.kwargs["shape"]
        n = batch * 1
        for s in shape:
            n *= s
        labels = torch.arange(n, dtype=torch.int64).reshape(batch, 1, *shape)
        return _FakeSample(labels=labels)


class _FakeLogNormal:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std


def _install_fake_synthspline(monkeypatch: pytest.MonkeyPatch) -> None:
    labelsynth_mod = types.ModuleType("synthspline.labelsynth")
    labelsynth_mod.SynthSplineBlock = _FakeSynthSplineBlock  # type: ignore[attr-defined]

    random_mod = types.ModuleType("synthspline.random")
    random_mod.LogNormal = _FakeLogNormal  # type: ignore[attr-defined]

    synthspline_mod = types.ModuleType("synthspline")
    synthspline_mod.labelsynth = labelsynth_mod  # type: ignore[attr-defined]
    synthspline_mod.random = random_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "synthspline", synthspline_mod)
    monkeypatch.setitem(sys.modules, "synthspline.labelsynth", labelsynth_mod)
    monkeypatch.setitem(sys.modules, "synthspline.random", random_mod)


# --- SplineVessels ---


def test_forward_output_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_synthspline(monkeypatch)
    sv = SplineVessels(shape=SMALL_SHAPE, fov=SMALL_FOV, device="cpu", seed=0)
    out = sv.forward()
    assert out.shape == torch.Size(SMALL_SHAPE)
    assert out.device == torch.device("cpu")


def test_call_returns_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_synthspline(monkeypatch)
    sv = SplineVessels(shape=SMALL_SHAPE, fov=SMALL_FOV, device="cpu", seed=0)
    batch = sv(n_sample=3)
    assert batch.shape == torch.Size((3, *SMALL_SHAPE))


def test_block_receives_voxel_size_and_distribution_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_synthspline(monkeypatch)
    sv = SplineVessels(
        shape=SMALL_SHAPE,
        fov=SMALL_FOV,
        nb_levels=3,
        tree_density=(0.02, 0.01),
        radius=(0.2, 0.05),
        device="cpu",
        seed=0,
    )
    kwargs = sv._block.kwargs
    assert kwargs["shape"] == list(SMALL_SHAPE)
    assert kwargs["voxel_size"] == pytest.approx(SMALL_FOV[0] / SMALL_SHAPE[0])
    assert kwargs["nb_levels"] == 3
    assert isinstance(kwargs["tree_density"], _FakeLogNormal)
    assert kwargs["tree_density"].mean == 0.02
    assert kwargs["tree_density"].std == 0.01
    assert kwargs["radius"].mean == 0.2
    assert kwargs["radius"].std == 0.05


def test_from_config_maps_geometry_params(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_synthspline(monkeypatch)
    config = Config(
        seed=1,
        device="cpu",
        geom=GeometryParams(input_shape=SMALL_SHAPE, fov=SMALL_FOV),
    )
    sv = SplineVessels.from_config(config)
    assert sv.shape == SMALL_SHAPE
    assert sv.nb_levels == config.geom.spline_nb_levels
    assert sv.tree_density == (
        config.geom.spline_tree_density.mean,
        config.geom.spline_tree_density.std,
    )
    assert sv.seed == 1


def test_attrs_contains_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_synthspline(monkeypatch)
    sv = SplineVessels(shape=SMALL_SHAPE, fov=SMALL_FOV, device="cpu", seed=0)
    attrs = sv.attrs
    assert attrs["generator"] == "SplineVessels"
    assert attrs["fov"] == SMALL_FOV
    assert attrs["nb_levels"] == sv.nb_levels
