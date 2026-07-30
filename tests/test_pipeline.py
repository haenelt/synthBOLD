from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from synthbold.config import Config, GeometryParams, PhysioParams
from synthbold.geometries import Cylinders, Shapes
from synthbold.models.signal import SignalOutput
from synthbold.pipeline import SynthPipeline, SynthSample

INPUT_SHAPE = (8, 8, 8)
OUTPUT_SHAPE = (4, 4, 4)
FOV = (4.0, 4.0, 4.0)
N_SAMPLES = 3
BATCH_SIZE = 2


def _make_config(seed: int = 0, **geom_overrides: Any) -> Config:
    geom_kwargs: dict[str, Any] = dict(
        input_shape=INPUT_SHAPE,
        output_shape=OUTPUT_SHAPE,
        fov=FOV,
        padding=1,
        cylinder_num=2,
        sphere_num=1,
        tetrahedron_num=1,
        cube_num=1,
        toroid_num=1,
    )
    geom_kwargs.update(geom_overrides)
    return Config(
        seed=seed,
        device="cpu",
        geom=GeometryParams(**geom_kwargs),
        physio=PhysioParams(label_displacement_shape=(2, 2, 2), label_J=2),
    )


def _make_pipeline(
    dirname: Path, config: Config | None = None, **kwargs: Any
) -> SynthPipeline:
    return SynthPipeline(
        dirname=dirname,
        n_samples=N_SAMPLES,
        config=config or _make_config(),
        **kwargs,
    )


def _make_distractor_pipeline(
    dirname: Path, config: Config | None = None, **kwargs: Any
) -> SynthPipeline:
    return _make_pipeline(
        dirname,
        config=config,
        include_spheres=True,
        include_tetrahedra=True,
        include_cubes=True,
        include_toroids=True,
        **kwargs,
    )


@pytest.fixture
def pipeline(tmp_path: Path) -> SynthPipeline:
    return _make_pipeline(tmp_path)


# --- SynthPipeline ---


def test_init_creates_shapes_and_cylinders_labels(tmp_path: Path) -> None:
    _make_pipeline(tmp_path)
    labels_dir = tmp_path / "labels"
    assert (labels_dir / "shapes.zarr").exists()
    assert (labels_dir / "cylinders.zarr").exists()
    assert not (labels_dir / "spheres.zarr").exists()
    assert not (labels_dir / "tetrahedra.zarr").exists()
    assert not (labels_dir / "cubes.zarr").exists()
    assert not (labels_dir / "toroids.zarr").exists()


def test_init_creates_only_requested_distractor_labels(tmp_path: Path) -> None:
    _make_pipeline(tmp_path, include_spheres=True, include_cubes=True)
    labels_dir = tmp_path / "labels"
    assert (labels_dir / "spheres.zarr").exists()
    assert (labels_dir / "cubes.zarr").exists()
    assert not (labels_dir / "tetrahedra.zarr").exists()
    assert not (labels_dir / "toroids.zarr").exists()


def test_second_init_does_not_regenerate_existing_labels(tmp_path: Path) -> None:
    _make_pipeline(tmp_path)
    with (
        patch.object(Shapes, "from_config") as shapes_from_config,
        patch.object(Cylinders, "from_config") as cylinders_from_config,
    ):
        _make_pipeline(tmp_path)
    assert not shapes_from_config.called
    assert not cylinders_from_config.called


def test_cylinders_zarr_matches_generated_batch_size(pipeline: SynthPipeline) -> None:
    assert pipeline.cylinders_zarr.shape == (N_SAMPLES, *INPUT_SHAPE)


def test_shapes_zarr_matches_generated_batch_size(pipeline: SynthPipeline) -> None:
    assert pipeline.shapes_zarr.shape == (N_SAMPLES, *INPUT_SHAPE)


def test_spheres_zarr_raises_when_not_generated(pipeline: SynthPipeline) -> None:
    with pytest.raises(FileNotFoundError):
        _ = pipeline.spheres_zarr


def test_sample_labels_output_shape_and_dtype(pipeline: SynthPipeline) -> None:
    out = pipeline._sample_labels(pipeline.cylinders_zarr, batch_size=5)
    assert out.shape == (5, *INPUT_SHAPE)
    assert out.dtype == torch.int64
    assert out.device == pipeline.device


def test_sample_labels_restricted_to_given_indices(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path, indices=[1])
    expected = torch.from_numpy(pipeline.cylinders_zarr[1]).long()
    out = pipeline._sample_labels(pipeline.cylinders_zarr, batch_size=4)
    for b in range(4):
        assert torch.equal(out[b], expected)


def test_to_synth_sample_maps_fields_correctly(pipeline: SynthPipeline) -> None:
    fake_output = SignalOutput(
        magnitude=torch.tensor([1.0]),
        phase=torch.tensor([2.0]),
        tissue=torch.tensor([3.0]),
        vessel=torch.tensor([4.0]),
        mean=torch.tensor([5.0]),
        variance=torch.tensor([6.0]),
        skewness=torch.tensor([7.0]),
        kurtosis=torch.tensor([8.0]),
        te=torch.tensor([9.0]),
    )
    sample = pipeline._to_synth_sample(fake_output)
    assert isinstance(sample, SynthSample)
    assert sample.magnitude.item() == 1.0
    assert sample.phase.item() == 2.0
    assert sample.tissue.item() == 3.0
    assert sample.compartment.item() == 4.0
    assert sample.dbz_mean.item() == 5.0
    assert sample.dbz_variance.item() == 6.0
    assert sample.dbz_skewness.item() == 7.0
    assert sample.dbz_kurtosis.item() == 8.0


def test_call_sample_distractor_is_none_without_distractors(
    pipeline: SynthPipeline,
) -> None:
    sample, sample_distractor, params = pipeline(batch_size=BATCH_SIZE)
    assert isinstance(sample, SynthSample)
    assert sample_distractor is None


def test_call_sample_distractor_is_synth_sample_with_distractors(
    tmp_path: Path,
) -> None:
    pipeline = _make_distractor_pipeline(tmp_path)
    sample, sample_distractor, params = pipeline(batch_size=BATCH_SIZE)
    assert isinstance(sample, SynthSample)
    assert isinstance(sample_distractor, SynthSample)


def test_call_sample_field_shapes(pipeline: SynthPipeline) -> None:
    sample, sample_distractor, params = pipeline(batch_size=BATCH_SIZE)
    expected_shape = (BATCH_SIZE, *OUTPUT_SHAPE)
    for field in sample._fields:
        assert getattr(sample, field).shape == expected_shape


def test_call_param_shapes(pipeline: SynthPipeline) -> None:
    sample, sample_distractor, params = pipeline(batch_size=BATCH_SIZE)
    assert params.theta.shape == (BATCH_SIZE,)
    assert params.phi.shape == (BATCH_SIZE,)
    assert params.te.shape == (BATCH_SIZE,)
    assert params.chi.shape == params.t2.shape
    assert params.chi.numel() > 0


def test_call_distractor_compartment_covers_foreground_compartment(
    tmp_path: Path,
) -> None:
    # `distractor_data` is `merge_labels(foreground_data, distractor_shapes_data)`,
    # which only fills in voxels where the foreground branch is background (0), so
    # every voxel occupied in the foreground branch must remain occupied in the
    # distractor branch.
    pipeline = _make_distractor_pipeline(tmp_path)
    sample, sample_distractor, params = pipeline(batch_size=BATCH_SIZE)
    assert sample_distractor is not None
    foreground_occupied = sample.compartment > 0
    distractor_occupied = sample_distractor.compartment > 0
    assert torch.all(distractor_occupied[foreground_occupied])


def test_call_reproducible_with_same_seed(tmp_path: Path) -> None:
    config = _make_config(seed=7)
    pipeline1 = _make_pipeline(tmp_path / "a", config=config)
    pipeline2 = _make_pipeline(tmp_path / "b", config=config)

    sample1, _, params1 = pipeline1(batch_size=BATCH_SIZE)
    sample2, _, params2 = pipeline2(batch_size=BATCH_SIZE)

    assert torch.equal(sample1.magnitude, sample2.magnitude)
    assert torch.equal(sample1.compartment, sample2.compartment)
    assert torch.equal(params1.chi, params2.chi)


def test_call_different_seeds_produce_different_output(tmp_path: Path) -> None:
    pipeline1 = _make_pipeline(tmp_path / "a", config=_make_config(seed=1))
    pipeline2 = _make_pipeline(tmp_path / "b", config=_make_config(seed=2))

    sample1, _, params1 = pipeline1(batch_size=BATCH_SIZE)
    sample2, _, params2 = pipeline2(batch_size=BATCH_SIZE)

    assert not torch.equal(sample1.magnitude, sample2.magnitude)


def test_call_saves_zarr_file_per_field(
    tmp_path: Path, pipeline: SynthPipeline
) -> None:
    out_dir = tmp_path / "out"
    sample, _, params = pipeline(batch_size=BATCH_SIZE, dirname=out_dir, fmt="zarr")
    for field in sample._fields:
        assert (out_dir / f"{field}.zarr").exists()
    assert (out_dir / "params.pt").exists()


def test_call_saves_nifti_file_per_field(
    tmp_path: Path, pipeline: SynthPipeline
) -> None:
    out_dir = tmp_path / "out"
    sample, _, params = pipeline(batch_size=BATCH_SIZE, dirname=out_dir, fmt="nii")
    for field in sample._fields:
        assert (out_dir / f"{field}.nii").exists()


def test_call_saves_distractor_suffixed_files(tmp_path: Path) -> None:
    pipeline = _make_distractor_pipeline(tmp_path / "pipeline")
    out_dir = tmp_path / "out"
    sample, sample_distractor, params = pipeline(
        batch_size=BATCH_SIZE, dirname=out_dir, fmt="zarr"
    )
    assert sample_distractor is not None
    for field in sample_distractor._fields:
        assert (out_dir / f"{field}_distractor.zarr").exists()


def test_call_saved_params_roundtrip(tmp_path: Path, pipeline: SynthPipeline) -> None:
    out_dir = tmp_path / "out"
    sample, _, params = pipeline(batch_size=BATCH_SIZE, dirname=out_dir, fmt="zarr")
    loaded = torch.load(out_dir / "params.pt")
    assert torch.equal(loaded["theta"], params.theta)
    assert torch.equal(loaded["chi"], params.chi)


def test_call_invalid_fmt_raises(tmp_path: Path, pipeline: SynthPipeline) -> None:
    with pytest.raises(ValueError, match="fmt must be"):
        pipeline(batch_size=BATCH_SIZE, dirname=tmp_path / "out", fmt="mgz")


def test_attrs_contains_expected_keys(pipeline: SynthPipeline) -> None:
    attrs = pipeline.attrs
    assert attrs["generator"] == "SynthPipeline"
    assert attrs["device"] == str(pipeline.device)
    assert attrs["seed"] == pipeline.seed


def test_benchmark_returns_nonnegative_float(pipeline: SynthPipeline) -> None:
    avg = pipeline.benchmark(batch_size=1, n=1, warmup=0)
    assert isinstance(avg, float)
    assert avg >= 0.0
