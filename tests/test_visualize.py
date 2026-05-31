from pathlib import Path

import pytest
import torch

from synthbold.visualize import make_gif, render_3d


@pytest.fixture
def sphere_mask() -> torch.Tensor:
    size = 16
    x = torch.linspace(-1, 1, size)
    grid = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1)
    return (grid.norm(dim=-1) < 0.7).float()


# --- make_gif ---


def test_make_gif_creates_file(tmp_path: Path) -> None:
    fname = tmp_path / "out.gif"
    data = torch.rand(3, 8, 8, 8)
    make_gif(fname, data)
    assert fname.exists()
    assert fname.stat().st_size > 0


def test_make_gif_single_frame(tmp_path: Path) -> None:
    fname = tmp_path / "single.gif"
    data = torch.rand(1, 8, 8, 8)
    make_gif(fname, data)
    assert fname.exists()


def test_make_gif_custom_params(tmp_path: Path) -> None:
    fname = tmp_path / "custom.gif"
    data = torch.rand(2, 8, 8, 8)
    make_gif(fname, data, cmap="gray", fps=5)
    assert fname.exists()


# --- render_3d ---


def test_render_3d_creates_file(tmp_path: Path, sphere_mask: torch.Tensor) -> None:
    fname = tmp_path / "out.png"
    render_3d(fname, sphere_mask)
    assert fname.exists()
    assert fname.stat().st_size > 0


def test_render_3d_output_is_png(tmp_path: Path, sphere_mask: torch.Tensor) -> None:
    fname = tmp_path / "out.png"
    render_3d(fname, sphere_mask)
    assert fname.read_bytes()[:4] == b"\x89PNG"


def test_render_3d_creates_parent_dirs(
    tmp_path: Path, sphere_mask: torch.Tensor
) -> None:
    fname = tmp_path / "nested" / "dir" / "out.png"
    render_3d(fname, sphere_mask)
    assert fname.exists()


def test_render_3d_custom_params(tmp_path: Path, sphere_mask: torch.Tensor) -> None:
    fname = tmp_path / "custom.png"
    render_3d(fname, sphere_mask, color="tomato", alpha=0.5, elev=45.0, azim=90.0)
    assert fname.exists()


def test_render_3d_wrong_ndim(tmp_path: Path) -> None:
    fname = tmp_path / "out.png"
    with pytest.raises(ValueError, match="3D"):
        render_3d(fname, torch.zeros(4, 8, 8, 8))
