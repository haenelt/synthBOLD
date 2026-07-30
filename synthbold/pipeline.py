"""Orchestration of individual synthesis models into an end-to-end pipeline."""

import logging
import os
import time
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import zarr

from synthbold.base import BaseGeometry, RandomGeneratorMixin
from synthbold.config import Config
from synthbold.decorator import log_call
from synthbold.geometries import Cubes, Cylinders, Shapes, Spheres, Tetrahedra, Toroids
from synthbold.io import save_batch, zarr_attributes
from synthbold.models import (
    BackgroundModel,
    DistractorModel,
    ForegroundModel,
    PerturbationModel,
    SignalModel,
)
from synthbold.models.functional import merge_labels
from synthbold.sampling import sample_polar, sample_uniform

__all__ = ["SynthSample", "SynthParams", "SynthPipeline"]


logger = logging.getLogger(__name__)


class SynthSample(NamedTuple):
    """A batch of synthesized samples.

    Attributes:
        magnitude: Synthesized magnitude image of shape ``(B, x, y, z)``.
        phase: Synthesized phase image of shape ``(B, x, y, z)``.
        tissue: Ground-truth tissue map of shape ``(B, x, y, z)``.
        vf: Ground-truth vf map of shape ``(B, x, y, z)``.
        dbz_mean: Mean of the ΔBz distribution of shape ``(B, x, y, z)``.
        dbz_variance: Variance of the ΔBz distribution of shape ``(B, x, y, z)``.
        dbz_skewness: Skewness of the ΔBz distribution of shape ``(B, x, y, z)``.
        dbz_kurtosis: Kurtosis of the ΔBz distribution of shape ``(B, x, y, z)``.
    """

    magnitude: torch.Tensor
    phase: torch.Tensor
    tissue: torch.Tensor
    vf: torch.Tensor
    dbz_mean: torch.Tensor
    dbz_variance: torch.Tensor
    dbz_skewness: torch.Tensor
    dbz_kurtosis: torch.Tensor


class SynthParams(NamedTuple):
    """Shared parameters used to generate a `SynthSample`.

    Attributes:
        theta: B0 polar angle, shape ``(B,)``.
        phi: B0 azimuthal angle, shape ``(B,)``.
        chi: Susceptibility values of shape ``(n_total,)``, where ``n_total`` is the
            sum, over batch elements, of each element's largest label id (cylinder plus
            distractor labels).
        t2: T2 values, shape ``(n_total,)``. Same layout as `chi`.
        te: Echo time of shape ``(B,)``.
    """

    theta: torch.Tensor
    phi: torch.Tensor
    chi: torch.Tensor
    t2: torch.Tensor
    te: torch.Tensor


class SynthPipeline(RandomGeneratorMixin):
    """End-to-end orchestration of geometry, model, and transform components into a
    single callable that synthesizes batches of BOLD samples.

    On construction, label maps for cylinders, background shapes, and any requested
    distractor geometries (spheres, tetrahedra, cubes, toroids) are generated once and
    cached to ZARR under ``<dirname>/labels`` so that `__call__` only ever samples from
    precomputed data. Each call draws a random batch of these label maps, runs them
    through the background, foreground, distractor, and perturbation models, and applies
    the signal model to produce magnitude/phase images alongside ground-truth tissue and
    perturber volume fraction maps.

    Args:
        dirname: Root directory for cached label maps.
        n_samples: Number of label maps to pre-generate for each geometry type.
        config: Configuration object shared by all sub-models and geometry generators.
        indices: Subset of label-map indices, into the range ``[0, n_samples)``, to
            sample from in `__call__`. Defaults to all indices.
        include_spheres: Whether to use sphere distractor labels.
        include_tetrahedra: Whether to use tetrahedron distractor labels.
        include_cubes: Whether to use cube distractor labels.
        include_toroids: Whether to use toroid distractor labels.
    """

    def __init__(
        self,
        dirname: Path,
        n_samples: int,
        config: Config,
        indices: list[int] | None = None,
        include_spheres: bool = False,
        include_tetrahedra: bool = False,
        include_cubes: bool = False,
        include_toroids: bool = False,
    ) -> None:
        super().__init__(device=config.device, seed=config.seed)
        self.dirname = Path(dirname)
        self.n_samples = n_samples
        self.config = config
        self.indices = list(range(self.n_samples)) if indices is None else indices

        # Set boolean flags for including different types of distractor shapes
        self.include_spheres = include_spheres
        self.include_tetrahedra = include_tetrahedra
        self.include_cubes = include_cubes
        self.include_toroids = include_toroids

        # Location of label maps saved to disk
        self.labels_dir = self.dirname / "labels"
        self.shapes_fname = self.labels_dir / "shapes.zarr"
        self.cylinders_fname = self.labels_dir / "cylinders.zarr"
        self.spheres_fname = self.labels_dir / "spheres.zarr"
        self.tetrahedra_fname = self.labels_dir / "tetrahedra.zarr"
        self.cubes_fname = self.labels_dir / "cubes.zarr"
        self.toroids_fname = self.labels_dir / "toroids.zarr"

        # Generate any missing label maps in __init__, so __call__ never has to generate
        # data on the fly
        self._generate_labels(self.shapes_fname, Shapes)
        self._generate_labels(self.cylinders_fname, Cylinders)
        if self.include_spheres:
            self._generate_labels(self.spheres_fname, Spheres)
        if self.include_tetrahedra:
            self._generate_labels(self.tetrahedra_fname, Tetrahedra)
        if self.include_cubes:
            self._generate_labels(self.cubes_fname, Cubes)
        if self.include_toroids:
            self._generate_labels(self.toroids_fname, Toroids)

        # Instantiate models
        self.bm = BackgroundModel.from_config(config)
        self.fm = ForegroundModel.from_config(config)
        self.dm = DistractorModel.from_config(config)
        self.pm = PerturbationModel.from_config(config)
        self.sm = SignalModel.from_config(config)

    @log_call
    def __call__(
        self,
        batch_size: int,
        dirname: Path | None = None,
        fmt: str = "zarr",
    ) -> tuple[SynthSample, SynthSample | None, SynthParams]:
        """Synthesize a batch of `batch_size` samples.

        Args:
            batch_size: Number of samples to synthesize.
            dirname: Directory for saving the outputs to disk. If given, each field
                of the returned `SynthSample`(s) is saved as its own file, named
                ``<dirname>/<field>.<fmt>`` (e.g. ``magnitude.zarr``), or
                ``<dirname>/<field>_distractor.<fmt>`` for a distractor sample if
                returned. `SynthParams` is saved as ``<dirname>/params.pt``.
            fmt: File format for saving `SynthSample` fields, either ``"zarr"`` or
                ``"nii"``. Ignored if `dirname` is not given.

        Returns:
            A tuple containing sampled data and parameters.

        Raises:
            ValueError: If `dirname` is given and `fmt` is not ``"zarr"`` or ``"nii"``.
        """
        # Sample random label maps independently to increase combinations.
        shapes_raw = self._sample_labels(self.shapes_zarr, batch_size)
        cylinders_raw = self._sample_labels(self.cylinders_zarr, batch_size)

        # Optionally, sample random label maps for distractor labels.
        spheres_raw = None
        tetrahedra_raw = None
        cubes_raw = None
        toroids_raw = None
        if self.include_spheres:
            spheres_raw = self._sample_labels(self.spheres_zarr, batch_size)
        if self.include_tetrahedra:
            tetrahedra_raw = self._sample_labels(self.tetrahedra_zarr, batch_size)
        if self.include_cubes:
            cubes_raw = self._sample_labels(self.cubes_zarr, batch_size)
        if self.include_toroids:
            toroids_raw = self._sample_labels(self.toroids_zarr, batch_size)

        # Pool distractor shapes
        distractor_shape_labels = [
            t
            for t in (spheres_raw, tetrahedra_raw, cubes_raw, toroids_raw)
            if t is not None
        ]

        # Apply models to foreground, background and optionally distractor
        background_data = self.bm(shapes_raw)
        foreground_data = self.fm(cylinders_raw)

        # Pipe distractor labels through distractor model since these shapes should be
        # left undeformed. Then, add transformed foreground data so that both branches
        # (foreground and distractor) share the same cylinder geometry. `merge_labels`
        # gives precedence to its first argument and offsets the others by the first's
        # max label, so `distractor_data`'s labels come out as cylinder first, then
        # distractor shapes.
        distractor_data = None
        if distractor_shape_labels:
            distractor_shapes_data = self.dm(
                merge_labels(*distractor_shape_labels)
                if len(distractor_shape_labels) > 1
                else distractor_shape_labels[0]
            )
            distractor_data = merge_labels(foreground_data, distractor_shapes_data)

        # Sample parameters that are shared across branches (foreground and distractor).
        # Additional distractor labels (offset by cylinder labels) get their own
        # independently sampled susceptibilities and t2 values.
        n_cylinder = foreground_data.amax(dim=(1, 2, 3))
        n_branch_labels = (
            distractor_data.amax(dim=(1, 2, 3))
            if distractor_data is not None
            else n_cylinder
        )
        n_total = int(n_branch_labels.sum().item())

        theta = sample_polar(
            batch_size, *self.pm.theta_range, self.device, self.generator
        )
        phi = sample_uniform(
            batch_size, *self.pm.phi_range, self.device, self.generator
        )
        te = sample_uniform(batch_size, *self.sm.te_range, self.device, self.generator)
        chi = sample_uniform(n_total, *self.pm.chi_range, self.device, self.generator)
        t2 = sample_uniform(n_total, *self.sm.t2_range, self.device, self.generator)

        # Flat-index offset of each batch element's own label range within `chi`/`t2`,
        # which is the same cumulative scheme `labels_to_values` uses internally.
        distractor_offsets = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )
        distractor_offsets[1:] = n_branch_labels[:-1].cumsum(0)

        # chi and t2 are sampled for both cylinder and distractor labels in a 1D array.
        # To extract only the cylinder chi / t2 values, we have to consider the offset
        # from the distractor labels per batch.
        cylinders_chi = torch.cat(
            [
                chi[distractor_offsets[b] : distractor_offsets[b] + n_cylinder[b]]
                for b in range(batch_size)
            ]
        )
        cylinders_t2 = torch.cat(
            [
                t2[distractor_offsets[b] : distractor_offsets[b] + n_cylinder[b]]
                for b in range(batch_size)
            ]
        )

        # Apply perturbation and signal model to cylinder data.
        foreground_dbz = self.pm(
            foreground_data, dchi=cylinders_chi, theta=theta, phi=phi
        )
        foreground_signal = self.sm(
            background_data, foreground_data, foreground_dbz, cylinders_t2, te
        )
        sample = self._to_synth_sample(foreground_signal)

        # Optionally, apply perturbation and signal model to distractor data if present.
        sample_distractor = None
        if distractor_data is not None:
            distractor_dbz = self.pm(distractor_data, dchi=chi, theta=theta, phi=phi)
            distractor_signal = self.sm(
                background_data, distractor_data, distractor_dbz, t2, te
            )
            sample_distractor = self._to_synth_sample(distractor_signal)

        params = SynthParams(theta=theta, phi=phi, chi=chi, t2=t2, te=te)

        # Save batch to disk.
        if dirname is not None:
            dirname = Path(dirname)
            save_batch(sample._asdict(), dirname, fmt, "", self.attrs)
            if sample_distractor is not None:
                save_batch(
                    sample_distractor._asdict(), dirname, fmt, "_distractor", self.attrs
                )
            torch.save(params._asdict(), dirname / "params.pt")

        return sample, sample_distractor, params

    @property
    def attrs(self) -> dict[str, Any]:
        """Common metadata for ZARR/NIfTI storage."""
        return zarr_attributes(self.__class__.__name__, self.device, self.seed)

    @cached_property
    def shapes_zarr(self) -> Any:
        """Lazily opened shape label maps."""
        return zarr.open(self.shapes_fname, mode="r")

    @cached_property
    def cylinders_zarr(self) -> Any:
        """Lazily opened cylinder label maps."""
        return zarr.open(self.cylinders_fname, mode="r")

    @cached_property
    def spheres_zarr(self) -> Any:
        """Lazily opened sphere label maps."""
        return zarr.open(self.spheres_fname, mode="r")

    @cached_property
    def tetrahedra_zarr(self) -> Any:
        """Lazily opened tetrahedron label maps."""
        return zarr.open(self.tetrahedra_fname, mode="r")

    @cached_property
    def cubes_zarr(self) -> Any:
        """Lazily opened cube label maps."""
        return zarr.open(self.cubes_fname, mode="r")

    @cached_property
    def toroids_zarr(self) -> Any:
        """Lazily opened toroid label maps."""
        return zarr.open(self.toroids_fname, mode="r")

    def benchmark(self, batch_size: int = 4, n: int = 10, warmup: int = 10) -> float:
        """Benchmark the average runtime of ``__call__``."""
        logger.info("Start benchmark...")
        for _ in range(warmup):
            _ = self(batch_size=batch_size)

        if self.config.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n):
            _ = self(batch_size=batch_size)

        if self.config.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg = (t1 - t0) / n
        logger.info(f"Average __call__: {avg:.6f}s.")

        return avg

    def _generate_labels(
        self,
        fname: Path,
        geometry_cls: type[BaseGeometry],
        poll_interval: float = 0.5,
        timeout: float = 3600.0,
    ) -> None:
        """Generate a stored geometry label map if it does not already exist.

        Guards against concurrent writers sharing `dirname` (e.g. multiple `DataLoader`
        workers or distributed-training ranks each instantiating their own
        `SynthPipeline` instance) with a lock file plus an atomic rename: only the first
        caller to create the lock file generates data, writing to a temporary path
        and renaming it into place once complete. Any other caller waits for `fname`
        to appear instead of reading a partially written cache.

        Args:
            fname: Target path of the cached label map.
            geometry_cls: Geometry class used to generate the label map.
            poll_interval: Seconds to sleep between checks while waiting for another
                process to finish generating `fname`.
            timeout: Maximum seconds to wait for another process to finish generating
                `fname` before giving up.

        Raises:
            TimeoutError: If `fname` does not appear within `timeout` seconds while
                another process holds the lock (e.g. because that process crashed
                mid-generation, leaving a stale lock file).
        """
        if fname.exists():
            return

        fname.parent.mkdir(parents=True, exist_ok=True)
        lock_path = fname.with_suffix(fname.suffix + ".lock")
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            # Another process is already generating this file; wait for it to
            # finish instead of reading a partially written cache.
            start = time.perf_counter()
            while not fname.exists():
                if time.perf_counter() - start > timeout:
                    raise TimeoutError(
                        f"Timed out waiting for {fname} to be generated by "
                        "another process."
                    ) from None
                time.sleep(poll_interval)
            return

        try:
            if fname.exists():  # generated by another process between the checks
                return
            logger.info("Creating %s label maps...", geometry_cls.__name__.lower())
            # `BaseGeometry.__call__` dispatches on `fname.suffix` (".zarr"/".nii") to
            # decide how to save, so the temp path must keep that suffix; the ".tmp"
            # marker goes into the stem instead.
            tmp = fname.with_name(f"{fname.stem}.tmp{fname.suffix}")
            geometry_cls.from_config(self.config)(self.n_samples, tmp)
            os.replace(tmp, fname)  # atomic on the same filesystem
        finally:
            lock_path.unlink(missing_ok=True)

    def _sample_labels(self, label_zarr: Any, batch_size: int) -> torch.Tensor:
        """Draw `batch_size` random label maps from `<label>_zarr` (with replacement)
        in a single fancy-indexed read, as an int64 tensor of shape
        ``(batch_size, X, Y, Z)`` on `self.device`."""
        label_idx = self.np_generator.choice(self.indices, size=batch_size)
        arr = np.asarray(label_zarr[label_idx])
        return torch.from_numpy(arr).long().to(self.device)

    def _to_synth_sample(self, signal_output: Any) -> SynthSample:
        """Maps a `signal_output` onto the corresponding `SynthSample` fields."""
        return SynthSample(
            magnitude=signal_output.magnitude,
            phase=signal_output.phase,
            tissue=signal_output.tissue,
            vf=signal_output.vf,
            dbz_mean=signal_output.mean,
            dbz_variance=signal_output.variance,
            dbz_skewness=signal_output.skewness,
            dbz_kurtosis=signal_output.kurtosis,
        )
