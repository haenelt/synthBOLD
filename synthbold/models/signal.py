"""Physics-based signal generation."""

from pathlib import Path
from typing import Any, NamedTuple, Self

import torch
import torch.nn.functional as F

from synthbold.base import Model, RandomGeneratorMixin
from synthbold.config import Config
from synthbold.decorator import log_call, to_device
from synthbold.io import save_nifti, save_zarr, zarr_attributes
from synthbold.models.functional import dchi_to_dbz, labels_to_values
from synthbold.sampling import sample_1d

__all__ = ["PerturbationModel", "SignalOutput", "SignalModel"]


class PerturbationModel(Model):
    """Physics-based transform that converts geometry label maps into magnetic field
    perturbation maps (ΔBz).

    Each vessel (or other geometric objects) is assigned a random susceptibility value
    and each sample is given a random B0 field orientation, from which the resulting
    magnetic field perturbation is computed using FFT convolution with a dipole kernel.

    Args:
        chi_range: Minimum and maximum susceptibility differences in ppm.
        b0: Static magnetic field strength in Tesla.
        theta_range: Minimum and maximum polar angles in radians for B0 orientations.
        phi_range: Minimum and maximum azimuthal angles in radians for B0 orientations.
        padding: Zero-padding width applied before FFT convolution. Can be either an
            integer or a tuple ``(px, py, pz)``.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        chi_range: tuple[float, float],
        b0: float,
        theta_range: tuple[float, float],
        phi_range: tuple[float, float],
        padding: int | tuple[int, int, int],
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)

        self.chi_range = chi_range
        self.b0 = b0
        self.theta_range = theta_range
        self.phi_range = phi_range

        # Get padding for (x, y, z)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.padding = padding

    def forward(
        self,
        data: torch.Tensor,
        dchi: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ΔBz perturbation maps from vessel or other perturber objects.

        Args:
            data: Tensor of shape ``(B, X, Y, Z)`` containing integer vessel labels or
                labels from other perturber objects.
            dchi: Optional per-batch, per-label susceptibility differences of shape
                ``(n_perturber,)``, where ``n_perturber`` is the sum, over batch
                elements, of each element's largest label value. Values are laid out
                flat, ordered by batch element and then by label id within each
                element (the same scheme as `merge_labels`'s offsets). Sampled
                uniformly from ``chi_range`` if not given.
            theta: Optional per-batch B0 polar angles of shape ``(B,)``. Sampled
                uniformly from ``theta_range`` if not given.
            phi: Optional per-batch B0 azimuthal angles of shape ``(B,)``. Sampled
                uniformly from ``phi_range`` if not given.

        Returns:
            Tensor of shape ``(B, X, Y, Z)`` containing the computed ΔBz perturbation
            maps.

        Raises:
            ValueError: If ``dchi``, ``theta``, or ``phi`` is given and is not a 1D
                tensor of the expected length (the summed per-batch perturber counts
                in ``data`` for ``dchi``, or ``B`` for ``theta``/``phi``).
        """
        B, _, _, _ = data.shape
        chi_low, chi_high = self.chi_range
        theta_low, theta_high = self.theta_range
        phi_low, phi_high = self.phi_range

        # Sample a random susceptibility difference for each (batch, label) pair
        chi = labels_to_values(
            data, dchi, "dchi", chi_low, chi_high, self.device, self.generator
        )

        # Sample random B0 orientation for each batch element
        theta = sample_1d(
            theta,
            "theta",
            B,
            theta_low,
            theta_high,
            self.device,
            self.generator,
            polar=True,
        )
        phi = sample_1d(phi, "phi", B, phi_low, phi_high, self.device, self.generator)

        # Compute field perturbation
        dbz = dchi_to_dbz(chi, self.padding, self.b0, theta, phi)

        return dbz

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs PerturbationModel instance from a config object."""
        chi_range = (config.physics.chi.min, config.physics.chi.max)
        theta_range = (config.physics.theta.min, config.physics.theta.max)
        phi_range = (config.physics.phi.min, config.physics.phi.max)
        return cls(
            chi_range=chi_range,
            b0=config.physics.b0,
            theta_range=theta_range,
            phi_range=phi_range,
            padding=config.geom.padding,
            device=config.device,
            seed=config.seed,
        )


class SignalOutput(NamedTuple):
    """Outputs of `SignalModel` at the target resolution, i.e., after downsampling done
    to estimate the static dephasing contribution.

    Attributes:
        magnitude: Magnitude image of shape ``(B, x, y, z)``.
        phase: Phase image of shape ``(B, x, y, z)``.
        tissue: Ground-truth tissue data of shape ``(B, x, y, z)``.
        vessel: Vessel (or other perturber) labels of shape ``(B, x, y, z)``.
        mean: Mean of the ΔBz distribution of shape ``(B, x, y, z)``.
        variance: Variance of the ΔBz distribution of shape ``(B, x, y, z)``.
        skewness: Skewness of the ΔBz distribution of shape ``(B, x, y, z)``.
        kurtosis: Kurtosis of the ΔBz distribution of shape ``(B, x, y, z)``.
        te: Sampled echo times in ms of shape ``(B,)``.

    Notes:
        - The application of static dephasing is essentially a downsampling operation of
          the sampled signal tensor, so the output tensor has different spatial
          dimensions than the input: lowercase letters (``x``, ``y``, ``z``) denote the
          downsampled output resolution, in contrast to uppercase letters (``X``, ``Y``,
          ``Z``), which denote the finer input resolution.
    """

    magnitude: torch.Tensor
    phase: torch.Tensor
    tissue: torch.Tensor
    vessel: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    skewness: torch.Tensor
    kurtosis: torch.Tensor
    te: torch.Tensor


class SignalModel(RandomGeneratorMixin):
    """Signal estimation based on static dephasing.

    Based on tissue, label (vessel) and estimated field perturbation maps, the resulting
    complex-valued MRI signal is generated, under the assumption of static dephasing.
    More precisely, each voxel is assigned a complex signal value, based on tissue data
    (S0), T2, and field offset. The signal is computed at a finer resolution than the
    target output, so that each output voxel is composed of multiple sub-voxels with
    distinct field offsets; downsampling then coherently sums these sub-voxel signals,
    thereby simulating the intra-voxel dephasing caused by field inhomogeneities (e.g.,
    from nearby vessels) that a real, coarser-resolution voxel would experience. The
    resulting magnitude image is obtained by downsampling to the target spatial
    dimension.

    This model only considers static dephasing and susceptibility-induced field
    perturbations from macrovascular compartments, e.g., no gradient fields, etc. No
    additional tissue relaxation effects are applied. No chemical shift is considered.
    Constant T2 values are assigned per vessel, sampled from a uniform distribution
    within the specified range. Variations within the same vessel should be small.

    Args:
        shape: Target shape of the output signal after downsampling.
        t2_range: Minimum and maximum T2s values of vascular compartments in ms.
        te_range: Minimum and maximum echo times in ms.
        spin_echo: Spin echo time in ms. If set to `None`, no spin echo is applied.
        b0: Static magnetic field strength in Tesla.
        gamma: Gyromagnetic ratio of the proton in ms-1T-1.
        device: Target compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        t2_range: tuple[float, float],
        te_range: tuple[float, float],
        spin_echo: float | None,
        b0: float,
        gamma: float,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device, seed=seed)

        self.shape = shape
        self.t2_range = t2_range
        self.te_range = te_range
        self.spin_echo = spin_echo
        self.b0 = b0
        self.gamma = gamma

    @property
    def attrs(self) -> dict[str, Any]:
        """Common metadata for ZARR/NIfTI storage."""
        return {
            **zarr_attributes(self.__class__.__name__, self.device, self.seed),
            "shape": self.shape,
        }

    @log_call
    @to_device
    def __call__(
        self,
        tissue: torch.Tensor,
        vessel: torch.Tensor,
        dbz: torch.Tensor,
        t2: torch.Tensor | None = None,
        te: torch.Tensor | None = None,
        dirname: Path | None = None,
        fmt: str = "zarr",
    ) -> SignalOutput:
        """Move inputs to the target device, compute the MRI signal, and optionally
        save each output tensor to disk.

        Args:
            tissue: Tissue data of shape ``(B, X, Y, Z)``.
            vessel: Integer vessel labels of shape ``(B, X, Y, Z)``.
            dbz: ΔBz maps in Tesla of shape ``(B, X, Y, Z)``.
            t2: Optional per-batch, per-label T2 values. Sampled uniformly from
                ``t2_range`` if not given.
            te: Optional per-batch echo times in ms. Sampled uniformly from ``te_range``
                if not given.
            dirname: Directory for saving the outputs to disk. If given, each field
                of `SignalOutput` is saved as its own file, named
                ``<dirname>/<field>.<fmt>`` (e.g. ``magnitude.zarr``, ``phase.zarr``).
            fmt: File format for saving, either ``"zarr"`` or ``"nii"``. Ignored if
                `dirname` is not given.

        Returns:
            SignalOutput: The computed signal output.

        Raises:
            ValueError: If `dirname` is given and `fmt` is not ``"zarr"`` or
                ``"nii"``.
        """
        result = self.forward(tissue, vessel, dbz, t2=t2, te=te)

        if dirname is not None:
            if fmt not in ("zarr", "nii"):
                raise ValueError(f"fmt must be 'zarr' or 'nii', got {fmt!r}.")
            dirname.mkdir(parents=True, exist_ok=True)
            for field, data in result._asdict().items():
                out_fname = dirname / f"{field}.{fmt}"
                if fmt == "zarr":
                    save_zarr(out_fname, data.cpu().numpy(), self.attrs)
                else:
                    save_nifti(out_fname, data.cpu().numpy(), permute=True)

        return result

    def forward(
        self,
        tissue: torch.Tensor,
        vessel: torch.Tensor,
        dbz: torch.Tensor,
        t2: torch.Tensor | None = None,
        te: torch.Tensor | None = None,
    ) -> SignalOutput:
        """Computation of MRI signal after static dephasing.

        Args:
            tissue: Tissue data of shape ``(B, X, Y, Z)``.
            vessel: Integer vessel labels of shape ``(B, X, Y, Z)``.
            dbz: ΔBz maps in Tesla of shape ``(B, X, Y, Z)``.
            t2: Optional per-batch, per-label T2 values of shape ``(n_perturber,)``,
                where ``n_perturber`` is the sum, over batch elements, of each element's
                largest label value. Values are laid out flat, ordered by batch element
                and then by label id within each element (the same scheme as
                `merge_labels`'s offsets). Sampled uniformly from ``t2_range`` if not
                given.
            te: Optional per-batch echo times in ms of shape ``(B,)``. Sampled uniformly
                from ``te_range`` if not given.

        Returns:
            SignalOutput: The computed signal output.

        Notes:
            - Note that we are estimating an integer downsampling factor here. If input
              and output shapes are not divisible, this can lead to undesired behavior.
            - Call the instance (``model(tissue, vessel, dbz)``) instead of
              ``model.forward(tissue, vessel, dbz)``. In this implementation,
              ``__call__`` transfers inputs to ``self.device`` before invoking
              ``forward``.
        """
        tissue = tissue.clone()
        vessel = vessel.clone()

        # Vessel label maps are probably of type integer. Here, we convert the tensor to
        # float to be able to generate volume fraction maps after downsampling.
        vessel = vessel.float()

        B, X, Y, Z = tissue.shape
        t2_low, t2_high = self.t2_range
        te_low, te_high = self.te_range

        # Sample a random T2 value for each (batch, label) pair
        t2 = labels_to_values(
            vessel,
            t2,
            "t2",
            t2_low,
            t2_high,
            self.device,
            self.generator,
            fill=torch.inf,
        )

        # Sample echo times
        te = sample_1d(te, "te", B, te_low, te_high, self.device, self.generator)
        te = te[:, None, None, None]

        # Compute complex signal
        spin_echo = 0.0 if self.spin_echo is None else self.spin_echo
        signal = self._complex_signal(tissue, dbz, t2, te, spin_echo)

        # Compute static dephasing
        factor = self._downsample_factor((X, Y, Z))
        magnitude, phase = self._static_dephasing(signal, factor)
        mean, variance, skewness, kurtosis = self._moments(dbz, factor)

        # Include intravascular decay into tissue data
        tissue[vessel > 0.5] = torch.abs(signal)[vessel > 0.5]

        # Downsample real-valued tensors to the same target resolution as the signal,
        # so each output voxel aggregates the same sub-voxels that were coherently
        # summed into it.
        tissue_map = F.avg_pool3d(
            tissue.unsqueeze(1), kernel_size=factor, stride=factor
        ).squeeze(1)
        vessel[vessel > 1.0] = 1.0
        vessel_map = F.avg_pool3d(
            vessel.unsqueeze(1), kernel_size=factor, stride=factor
        ).squeeze(1)

        return SignalOutput(
            magnitude=magnitude,
            phase=phase,
            tissue=tissue_map,
            vessel=vessel_map,
            mean=mean,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            te=te,
        )

    def _downsample_factor(self, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Compute the per-axis integer downsampling factor from input to output shape.

        Args:
            shape: Input shape ``(X, Y, Z)`` to downsample from.

        Returns:
            Per-axis downsampling factor ``(fx, fy, fz)`` such that
            ``shape[i] == self.shape[i] * factor[i]`` for each axis.

        Raises:
            ValueError: If any dimension of `shape` is not evenly divisible by the
                corresponding dimension of `self.shape`.
        """
        X, Y, Z = shape
        ox, oy, oz = self.shape

        # Get downsampling factor from input and output shapes. If input shape is not
        # exactly divisible by output shape, this may cause unintended cropping or shape
        # mismatch.
        if X % ox or Y % oy or Z % oz:
            raise ValueError(
                f"Input shape {(X, Y, Z)} is not evenly divisible by output shape "
                f"{(ox, oy, oz)}"
            )
        return (X // ox, Y // oy, Z // oz)

    def _complex_signal(
        self,
        tissue: torch.Tensor,
        dbz: torch.Tensor,
        t2: torch.Tensor,
        te: torch.Tensor,
        offset: float,
    ) -> torch.Tensor:
        """Compute the per-voxel complex MRI signal under static dephasing.

        Combines T2 decay with phase accrual from the field perturbation, relative to
        the spin-echo refocusing time.

        Args:
            tissue: Tissue data (S0) of shape ``(B, X, Y, Z)``.
            dbz: ΔBz maps in Tesla of shape ``(B, X, Y, Z)``.
            t2: Per-voxel T2 values in ms of shape ``(B, X, Y, Z)``.
            te: Echo times in ms, broadcastable to ``(B, X, Y, Z)``.
            offset: Spin echo time in ms at which the ΔBz-induced phase is refocused
                (``0.0`` if no spin echo is applied).

        Returns:
            Complex64 tensor of the same shape as `tissue`.
        """
        signal = tissue * torch.exp(1j * self.gamma * dbz * (te - offset) - te / t2)
        return signal.to(torch.complex64)

    def _static_dephasing(
        self, signal: torch.Tensor, factor: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Downsample the sub-voxel complex signal by coherent summation.

        Each output voxel's signal is the average of its constituent sub-voxel
        signals, so dephasing between sub-voxels (e.g. from nearby vessels) reduces
        the resulting magnitude, simulating intra-voxel static dephasing.

        Args:
            signal: Complex per-voxel signal of shape ``(B, X, Y, Z)``, at the finer
                input resolution.
            factor: Downsampling factor ``(fx, fy, fz)`` giving the number of
                sub-voxels aggregated into each output voxel along each axis.

        Returns:
            Tuple ``(magnitude, phase)`` of the downsampled complex signal, each of
            shape ``(B, x, y, z)``.
        """
        # avg_pool3d does not support complex tensors, so real and imaginary parts are
        # pooled separately and then recombined to coherently sum sub-voxel signals.
        signal_stack = torch.stack([signal.real, signal.imag], dim=1)
        signal_ds = F.avg_pool3d(signal_stack, kernel_size=factor, stride=factor)
        signal_complex = signal_ds[:, 0] + 1j * signal_ds[:, 1]

        return torch.abs(signal_complex), torch.angle(signal_complex)

    def _moments(
        self, dbz: torch.Tensor, factor: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes per-output-voxel moments of the sub-voxel ΔBz distribution.

        Args:
            dbz: ΔBz maps in Tesla of shape ``(B, X, Y, Z)``.
            factor: Downsampling factor ``(fx, fy, fz)`` giving the number of
                sub-voxels aggregated into each output voxel along each axis.

        Returns:
            Tuple ``(mean, variance, skewness, kurtosis)``, each of shape
            ``(B, x, y, z)``, describing the distribution of ΔBz values within each
            output voxel. Kurtosis is excess kurtosis (0 for a normal distribution).
        """

        # Raw moments E[x^k] of each output voxel's sub-voxel ΔBz values are just the
        # average-pooled powers of dbz, since avg_pool3d computes a per-block mean.
        # Central moments are then recovered from the raw moments algebraically, so no
        # sub-voxel values need to be gathered explicitly.
        def pooled_power(power: int) -> torch.Tensor:
            return F.avg_pool3d(
                (dbz**power).unsqueeze(1), kernel_size=factor, stride=factor
            ).squeeze(1)

        m1, m2, m3, m4 = (pooled_power(k) for k in (1, 2, 3, 4))

        mean = m1
        # Clamp against tiny negative rounding error from the m2 - m1**2 formulation.
        variance = (m2 - m1**2).clamp_min(0.0)
        std = torch.sqrt(variance)

        # A block whose sub-voxel ΔBz values are all identical (e.g. far from any
        # perturber, where ΔBz is exactly 0 throughout) has exactly zero variance, so
        # the standardized third/fourth moments below would divide 0/0 into NaN.
        # Skewness/kurtosis are undefined for a degenerate (constant) distribution;
        # define them as 0 there instead of propagating NaN.
        zero_variance = variance == 0.0
        safe_std = torch.where(zero_variance, torch.ones_like(std), std)

        skewness = torch.where(
            zero_variance,
            torch.zeros_like(std),
            (m3 - 3 * m1 * m2 + 2 * m1**3) / safe_std**3,
        )
        kurtosis = torch.where(
            zero_variance,
            torch.zeros_like(std),
            (m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4) / safe_std**4 - 3,
        )

        return mean, variance, skewness, kurtosis

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Constructs SignalModel instance from a config object."""
        t2_range = (config.physics.t2.min, config.physics.t2.max)
        te_range = (config.physics.te.min, config.physics.te.max)
        return cls(
            config.geom.output_shape,
            t2_range,
            te_range,
            config.physics.spin_echo,
            config.physics.b0,
            config.physics.gamma,
            device=config.device,
            seed=config.seed,
        )
