"""Tests for the CLI entry point."""

import sys
from pathlib import Path

import pytest

from synthbold.__main__ import get_parser, main

# --------------------------------------------------------------------------------------
# get_parser
# --------------------------------------------------------------------------------------


def test_parser_minimal(tmp_path: Path) -> None:
    """Required flags only; optional args use their defaults."""
    args = get_parser().parse_args(["-o", str(tmp_path / "out.zarr"), "-l", "tissue"])
    assert args.output == tmp_path / "out.zarr"
    assert args.label == "tissue"
    assert args.n_sample == 1
    assert args.seed is None
    assert args.config is None


def test_parser_all_args(tmp_path: Path) -> None:
    """All flags are parsed and converted to the correct types."""
    cfg = tmp_path / "cfg.yaml"
    args = get_parser().parse_args(
        [
            "-o",
            str(tmp_path / "out.nii"),
            "-l",
            "vessel",
            "--n-sample",
            "5",
            "--seed",
            "42",
            "--config",
            str(cfg),
        ]
    )
    assert args.output == tmp_path / "out.nii"
    assert args.label == "vessel"
    assert args.n_sample == 5
    assert args.seed == 42
    assert args.config == cfg


def test_parser_output_is_path(tmp_path: Path) -> None:
    """--output is converted to a Path object."""
    args = get_parser().parse_args(["-o", str(tmp_path / "out.zarr"), "-l", "tissue"])
    assert isinstance(args.output, Path)


def test_parser_missing_output() -> None:
    """Missing --output raises SystemExit."""
    with pytest.raises(SystemExit):
        get_parser().parse_args(["-l", "tissue"])


def test_parser_missing_label(tmp_path: Path) -> None:
    """Missing --label raises SystemExit."""
    with pytest.raises(SystemExit):
        get_parser().parse_args(["-o", str(tmp_path / "out.zarr")])


def test_parser_invalid_label(tmp_path: Path) -> None:
    """An unrecognised --label choice raises SystemExit."""
    with pytest.raises(SystemExit):
        get_parser().parse_args(["-o", str(tmp_path / "out.zarr"), "-l", "brain"])


# --------------------------------------------------------------------------------------
# output suffix validation
# --------------------------------------------------------------------------------------


def test_main_invalid_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """main() exits when --output has an unsupported extension."""
    monkeypatch.setattr(
        sys, "argv", ["synthbold", "-o", str(tmp_path / "out.txt"), "-l", "tissue"]
    )
    with pytest.raises(SystemExit):
        main()
