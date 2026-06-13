"""Visualization utilities for synthBOLD outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

__all__ = ["make_gif", "render_3d"]


def make_gif(
    fname: Path,
    data: torch.Tensor,
    cmap: str = "tab20",
    mip: bool = False,
    fps: int = 2,
) -> None:
    """Make a GIF animation from the middle axial slice of a batch of 3D volumes.

    Each batch element becomes one frame. The slice is taken at the midpoint of the
    last (z) axis.

    Args:
        fname: Output file path for the saved GIF.
        data: Batch of 3D volumes with shape ``(N, X, Y, Z)``.
        cmap: Matplotlib colormap name.
        mip: Whether to display a maximum intensity projection.
        fps: Frames per second for the output GIF.
    """
    fig, ax = plt.subplots()
    ax.axis("off")

    # Start with first slice of first volume
    volume = data[0]
    if mip:
        slice_0 = torch.max(volume, dim=-1).values
    else:
        slice_0 = volume[:, :, volume.shape[-1] // 2]
    im = ax.imshow(slice_0.detach().cpu().numpy(), cmap=cmap, animated=True)

    # Get frames (one frame per batch element)
    frames = []
    for volume in data:
        if mip:
            frames.append(torch.max(volume, dim=-1).values)
        else:
            frames.append(volume[:, :, volume.shape[-1] // 2])

    # Animation function
    def update(frame_index: int) -> list:
        im.set_array(frames[frame_index].detach().cpu().numpy())
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)

    # Save as GIF
    anim.save(fname, writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_3d(
    fname: Path,
    mask: torch.Tensor,
    color: str = "steelblue",
    alpha: float = 0.7,
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    """Render a 3D binary mask as a surface mesh and save as PNG.

    Uses marching cubes to extract an isosurface from the binary mask, then renders it
    as a shaded polygon mesh from a configurable viewpoint.

    Args:
        fname: Output file path for the saved PNG.
        mask: 3D binary mask tensor of shape ``(X, Y, Z)``.
        color: Face color for the surface mesh.
        alpha: Opacity of the surface mesh.
        elev: Elevation angle of the 3D view in degrees.
        azim: Azimuth angle of the 3D view in degrees.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(mask.shape)}.")

    mask_np = mask.cpu().numpy().astype(bool)
    verts, faces, _, _ = measure.marching_cubes(mask_np)

    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(mesh)

    # Equal aspect ratio: expand each axis to the same half-range around its centre
    half_range = max(verts[:, i].max() - verts[:, i].min() for i in range(3)) / 2
    mid = [float(verts[:, i].mean()) for i in range(3)]
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    Path(fname).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(fname, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
