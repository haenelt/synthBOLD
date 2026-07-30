import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from synthbold import __main__ as m
from synthbold.config import Config, GeometryParams, PhysioParams
from synthbold.io import load_nifti
from synthbold.pipeline import SynthSample

INPUT_SHAPE = [8, 8, 8]
OUTPUT_SHAPE = [4, 4, 4]
FOV = [4.0, 4.0, 4.0]


def _fast_config(seed: int = 0, **geom_overrides: Any) -> Config:
    """Builds a small, fast-to-synthesize Config for exercising `main()`."""
    geom_kwargs: dict[str, Any] = dict(
        input_shape=tuple(INPUT_SHAPE),
        output_shape=tuple(OUTPUT_SHAPE),
        fov=tuple(FOV),
        padding=1,
        cylinder_num=2,
    )
    geom_kwargs.update(geom_overrides)
    return Config(
        seed=seed,
        device="cpu",
        geom=GeometryParams(**geom_kwargs),
        physio=PhysioParams(label_displacement_shape=(2, 2, 2), label_J=2),
    )


def _write_config_yaml(path: Path, seed: int = 0, **geom_overrides: Any) -> Path:
    """Writes a small, fast-to-synthesize config YAML for exercising `main()` via
    ``--config``."""
    geom: dict[str, Any] = dict(
        input_shape=INPUT_SHAPE,
        output_shape=OUTPUT_SHAPE,
        fov=FOV,
        padding=1,
        cylinder_num=2,
    )
    geom.update(geom_overrides)
    data = {
        "seed": seed,
        "device": "cpu",
        "geom": geom,
        "physio": {"label_displacement_shape": [2, 2, 2], "label_J": 2},
    }
    path.write_text(yaml.safe_dump(data))
    return path


def _make_sample(shape: tuple[int, ...] = (2, 3, 3, 3)) -> SynthSample:
    generator = torch.Generator().manual_seed(0)
    fields = {
        name: torch.randn(shape, generator=generator) * 5 + 3
        for name in SynthSample._fields
    }
    return SynthSample(**fields)


# --- get_parser ---


def test_get_parser_requires_output() -> None:
    parser = m.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_get_parser_defaults() -> None:
    parser = m.get_parser()
    args = parser.parse_args(["--output", "out"])
    assert args.output == Path("out")
    assert args.n_sample == 10
    assert args.batch_size == 4
    assert args.config is None
    assert args.include_spheres is False
    assert args.include_tetrahedra is False
    assert args.include_cubes is False
    assert args.include_toroids is False
    assert args.seed is None
    assert args.log_level == "INFO"


def test_get_parser_custom_values() -> None:
    parser = m.get_parser()
    args = parser.parse_args(
        [
            "--output",
            "out",
            "--n-sample",
            "5",
            "--batch-size",
            "2",
            "--config",
            "cfg.yaml",
            "--include-spheres",
            "--include-cubes",
            "--seed",
            "3",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.n_sample == 5
    assert args.batch_size == 2
    assert args.config == Path("cfg.yaml")
    assert args.include_spheres is True
    assert args.include_cubes is True
    assert args.include_tetrahedra is False
    assert args.include_toroids is False
    assert args.seed == 3
    assert args.log_level == "DEBUG"


def test_get_parser_invalid_log_level_raises() -> None:
    parser = m.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "out", "--log-level", "TRACE"])


# --- setup_logging ---


def test_setup_logging_sets_level() -> None:
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers.clear()
    try:
        m.setup_logging(logging.DEBUG)
        assert root.level == logging.DEBUG
    finally:
        root.handlers.clear()
        root.handlers.extend(saved_handlers)
        root.setLevel(saved_level)


# --- standardize ---


def test_standardize_zscores_mean_fields() -> None:
    sample = _make_sample()
    result = m.standardize(sample)
    for field in ("dbz_mean", "dbz_skewness", "dbz_kurtosis"):
        value = getattr(result, field)
        dims = tuple(range(1, value.ndim))
        assert torch.allclose(
            value.mean(dim=dims), torch.zeros(value.shape[0]), atol=1e-5
        )
        assert torch.allclose(
            value.std(dim=dims, unbiased=False), torch.ones(value.shape[0]), atol=1e-4
        )


def test_standardize_variance_field_is_not_mean_centered() -> None:
    sample = _make_sample()
    result = m.standardize(sample)
    # center=False only rescales, it does not subtract the mean, so a field whose
    # original mean is nonzero stays nonzero after standardization.
    mean_after = result.dbz_variance.mean(dim=(1, 2, 3))
    assert not torch.allclose(mean_after, torch.zeros_like(mean_after))


def test_standardize_leaves_other_fields_untouched() -> None:
    sample = _make_sample()
    result = m.standardize(sample)
    assert torch.equal(result.magnitude, sample.magnitude)
    assert torch.equal(result.phase, sample.phase)
    assert torch.equal(result.tissue, sample.tissue)
    assert torch.equal(result.vf, sample.vf)


# --- main ---


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["synthbold", *argv])
    m.main()


def test_main_default_run_writes_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config_yaml(tmp_path / "config.yaml")
    output = tmp_path / "out"

    _run_main(
        monkeypatch,
        [
            "--output",
            str(output),
            "--config",
            str(config_path),
            "--n-sample",
            "2",
            "--batch-size",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    for field in SynthSample._fields:
        assert (output / f"{field}.nii").exists()
        assert not (output / f"{field}_distractor.nii").exists()


def test_main_with_distractor_flag_writes_distractor_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config_yaml(tmp_path / "config.yaml", sphere_num=1)
    output = tmp_path / "out"

    _run_main(
        monkeypatch,
        [
            "--output",
            str(output),
            "--config",
            str(config_path),
            "--include-spheres",
            "--n-sample",
            "2",
            "--batch-size",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    for field in SynthSample._fields:
        assert (output / f"{field}.nii").exists()
        assert (output / f"{field}_distractor.nii").exists()


def test_main_without_config_uses_default_config_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_cls = Mock(return_value=_fast_config(seed=0))
    config_cls.from_yaml = Mock(
        side_effect=AssertionError("from_yaml should not be called")
    )
    monkeypatch.setattr(m, "Config", config_cls)
    output = tmp_path / "out"

    _run_main(
        monkeypatch,
        [
            "--output",
            str(output),
            "--n-sample",
            "2",
            "--batch-size",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    config_cls.assert_called_once_with()
    config_cls.from_yaml.assert_not_called()
    assert (output / "magnitude.nii").exists()


def test_main_seed_cli_override_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The YAML's own seed (999) must be overridden by --seed for both runs to match.
    config_path = _write_config_yaml(tmp_path / "config.yaml", seed=999)

    def run(output: Path) -> None:
        _run_main(
            monkeypatch,
            [
                "--output",
                str(output),
                "--config",
                str(config_path),
                "--seed",
                "7",
                "--n-sample",
                "2",
                "--batch-size",
                "1",
                "--log-level",
                "WARNING",
            ],
        )

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run(out_a)
    run(out_b)

    mag_a, _, _ = load_nifti(out_a / "magnitude.nii")
    mag_b, _, _ = load_nifti(out_b / "magnitude.nii")
    assert np.allclose(mag_a, mag_b)


def test_main_seed_cli_override_changes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config_yaml(tmp_path / "config.yaml")

    def run(output: Path, seed: str) -> None:
        _run_main(
            monkeypatch,
            [
                "--output",
                str(output),
                "--config",
                str(config_path),
                "--seed",
                seed,
                "--n-sample",
                "2",
                "--batch-size",
                "1",
                "--log-level",
                "WARNING",
            ],
        )

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run(out_a, "1")
    run(out_b, "2")

    mag_a, _, _ = load_nifti(out_a / "magnitude.nii")
    mag_b, _, _ = load_nifti(out_b / "magnitude.nii")
    assert not np.allclose(mag_a, mag_b)
