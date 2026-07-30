"""Validated configuration structures and custom validators used throughout the
synthBOLD package to parse and ensure data synthesis parameters are correct."""

from pathlib import Path
from typing import Self

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

__all__ = ["Range", "GeometryParams", "PhysioParams", "Config"]


class Range(BaseModel):
    """Defines a numerical range."""

    min: float = Field(description="The minimum value of the range.")
    max: float = Field(description="The maximum value of the range.")

    @field_validator("max")
    @classmethod
    def check_range(cls, v: float, info: ValidationInfo) -> float:
        """Validates that the maximum value is greater than or equal to the minimum.

        Args:
            v: The value of max.
            info: Validation context containing other fields.

        Returns:
            The validated max value.

        Raises:
            ValueError: If max is less than min.
        """
        if "min" in info.data and v < info.data["min"]:
            raise ValueError(
                f"max ({v}) must be greater than or equal to min ({info.data['min']})."
            )
        return v


class GeometryParams(BaseModel):
    """Spatial parameters for input and output tensors.

    Args:
        input_shape: Shape of source volume.
        output_shape: Shape of target volume (after static dephasing downsampling).
        fov: Field-of-view of source volume in mm.
        padding: Voxel padding for calculating field perturbation.
        cylinder_num: Number of generated cylinders.
        cylinder_vf: Volume fraction of cylinders.
        cylinder_diameter: Range of cylinder diameters in mm.
        cylinder_allow_overlap: Allow cylinders to overlap.
        sphere_num: Number of generated spheres.
        sphere_vf: Volume fraction of spheres.
        sphere_diameter: Range of sphere diameters in mm.
        sphere_allow_overlap: Allow spheres to overlap.
        tetrahedron_num: Number of generated tetrahedra.
        tetrahedron_vf: Volume fraction of tetrahedra.
        tetrahedron_diameter: Range of tetrahedron circumdiameters in mm.
        tetrahedron_allow_overlap: Allow tetrahedra to overlap.
        cube_num: Number of generated cubes.
        cube_vf: Volume fraction of cubes.
        cube_diameter: Range of cube circumdiameters in mm.
        cube_allow_overlap: Allow cubes to overlap.
        toroid_num: Number of generated toroids.
        toroid_vf: Volume fraction of toroids.
        toroid_diameter: Range of toroid major diameters in mm.
        toroid_tube_ratio: Range of toroid tube-to-major radius ratios.
        toroid_allow_overlap: Allow toroids to overlap.
    """

    # general geometry
    input_shape: tuple[int, int, int] = (128, 128, 128)
    output_shape: tuple[int, int, int] = (32, 32, 32)
    fov: tuple[float, float, float] = (32.0, 32.0, 32.0)
    padding: int = 64

    # object geometry: cylinders
    cylinder_num: int | None = None
    cylinder_vf: Range = Range(min=0.005, max=0.01)
    cylinder_diameter: Range = Range(min=0.25, max=2.0)
    cylinder_allow_overlap: bool = True

    # object geometry: spheres
    sphere_num: int | None = None
    sphere_vf: Range = Range(min=0.005, max=0.01)
    sphere_diameter: Range = Range(min=0.2, max=2.0)
    sphere_allow_overlap: bool = True

    # object geometry: tetrahedra
    tetrahedron_num: int | None = None
    tetrahedron_vf: Range = Range(min=0.005, max=0.01)
    tetrahedron_diameter: Range = Range(min=0.4, max=4.0)
    tetrahedron_allow_overlap: bool = True

    # object geometry: cubes
    cube_num: int | None = None
    cube_vf: Range = Range(min=0.005, max=0.01)
    cube_diameter: Range = Range(min=2.0, max=10.0)
    cube_allow_overlap: bool = True

    # object geometry: toroids
    toroid_num: int | None = None
    toroid_vf: Range = Range(min=0.005, max=0.01)
    toroid_diameter: Range = Range(min=2.0, max=10.0)
    toroid_tube_ratio: Range = Range(min=0.1, max=0.4)
    toroid_allow_overlap: bool = True

    @field_validator("output_shape")
    @classmethod
    def check_divisible_output(
        cls, v: tuple[int, ...], info: ValidationInfo
    ) -> tuple[int, ...]:
        """Validates that input_shape is divisible by output_shape in every dim.

        Args:
            v: The value of output_shape.
            info: Validation context containing other fields.

        Returns:
            The validated output_shape value.

        Raises:
            ValueError: If any `input_shape` dimension is not evenly divisible by the
                corresponding `output_shape` dimension.
        """
        input_shape = info.data.get("input_shape")
        if input_shape is None:
            return v
        for i_dim, o_dim in zip(input_shape, v, strict=True):
            if i_dim % o_dim != 0:
                raise ValueError(
                    f"Input shape dimension {i_dim} is not divisible by "
                    f"output_shape dimension {o_dim}."
                )
        return v


class PhysioParams(BaseModel):
    """Parameters for shape and cylinder maps.

    Args:
        label_displacement_shape: Shape of noise map for smooth displacement field.
        label_J: Number of tissue classes.
        label_scale: Scale factor of the noise map.
        background_mu: Mean intensity range for different tissue compartments.
        background_std: Standard deviation range for intensities within tissue types.
    """

    label_displacement_shape: tuple[int, int, int] = (4, 4, 4)
    label_J: int = 8
    label_scale: float = 0.2
    background_mu: Range = Range(min=10.0, max=100.0)
    background_std: Range = Range(min=5.0, max=10.0)


class PhysicsParams(BaseModel):
    """Parameters governing the MRI signal generation.

    Args:
    b0: Static magnetic field strength in Tesla.
    gamma: Gyromagnetic ratio in ms^-1 T^-1
    chi: Vessel susceptibilities (difference between IV and EV) in ppm.
    t2: Range of T2 relaxation times in ms.
    te: Range of echo times in ms.
    theta: Range of polar angles in radians for B0 orientation.
    phi: Range of azimuthal angles in radians for B0 orientation.
    spin_echo: Time of spin echo in ms if exists.
    """

    b0: float = 7.0
    gamma: float = 2.675221e5
    chi: Range = Range(min=-0.01e-6, max=0.1e-6)
    t2: Range = Range(min=5.0, max=50.0)
    te: Range = Range(min=5.0, max=100.0)
    theta: Range = Range(min=0.0, max=np.pi)
    phi: Range = Range(min=0.0, max=2 * np.pi)
    spin_echo: float | None = 64.0


class TransformParams(BaseModel):
    """Parameters for data augmentation transforms.

    Args:
        bias_lowres_shape: Shape of noise map to generate a smooth bias field.
        bias_amplitude: Range of scaling factors for bias field variation.
        gamma_exponent: Range of gamma exponents for gamma intensity transform.
        gnoise_mu: Range of the gaussian noise distribution means.
        gnoise_std: Range of the gaussian noise distribution standard deviations.
        spike_num: Minimum and maximum number of localized k-space spikes.
        spike_intensity: Range of spike intensity multipliers.
        mgamma_shape: Range of shape parameters for multiplicative gamma noise.
        perlin_base_shape: Matrix size of the coarsest octave for Perlin noise.
        perlin_octaves: Number of octaves to combine for Perlin noise.
        perlin_persistence: Amplitude decay factor between octaves for Perlin noise.
        perlin_amplitude: Range of scaling factors for Perlin-like noise.
        pnoise_peak: Range of peak counts for signal-dependent Poisson noise.
        rnoise_std: Range of standard deviations for Rician noise.
        ncchi_std: Range of standard deviations for noncentral chi noise.
        ncchi_dof: DOF (number of Gaussian noise channels) for noncentral chi noise.
        speckle_std: Range of standard deviations for multiplicative speckle noise.
        flip_prob: Probability of performing random axis flip.
        sphere_radius: Range of radii for spherical background mask.
        sphere_prob: Probability of applying the spherical background mask.
        sphere_deform_amplitude: Maximum relative radius for perturbation.
        sphere_deform_shape: Shape of the low-resolution noise field.
        gsmooth_sigma: Range of the gaussian smoothing kernel standard deviations.
        gsmooth_kernel_size: Kernel size for spatial smoothing.
        gsmooth_isotropic: If True, isotropic spatial smoothing is applied.
        sharpen_sigma: Range of the gaussian sharpening kernel standard deviations.
        sharpen_amount: Range of unsharp masking sharpening strengths.
        sharpen_kernel_size: Kernel size for gaussian sharpening.
        gibbs_cutoff: Range of normalized k-space cutoff radii for Gibbs ringing.
        elastic_sigma: Standard deviation to generate smooth random Gaussian field.
        elastic_max_disp: Maximum displacement magnitude in voxels.
        caliber_sigma: Standard deviation to generate smooth random Gaussian field.
        caliber_alpha: Radius variation of generated random radius field.
    """

    # transform: BiasField
    bias_lowres_shape: tuple[int, int, int] = (4, 4, 4)
    bias_amplitude: Range = Range(min=0.0, max=1.0)

    # transform: GammaTransform
    gamma_exponent: Range = Range(min=0.5, max=2.0)

    # transform: GaussianNoise
    gnoise_mu: Range = Range(min=-0.2, max=0.2)
    gnoise_std: Range = Range(min=0.0, max=0.1)

    # transform: KSpaceSpikeNoise
    spike_num: tuple[int, int] = (1, 3)
    spike_intensity: Range = Range(min=1.0, max=3.0)

    # transform: MultiplicativeGammaNoise
    mgamma_shape: Range = Range(min=2.0, max=50.0)

    # transform: PerlinNoise
    perlin_base_shape: tuple[int, int, int] = (4, 4, 4)
    perlin_octaves: int = 4
    perlin_persistence: float = 0.5
    perlin_amplitude: Range = Range(min=0.0, max=0.1)

    # transform: PoissonNoise
    pnoise_peak: Range = Range(min=0.0001, max=0.001)

    # transform: RicianNoise
    rnoise_std: Range = Range(min=0.0, max=2.0)

    # transform: NoncentralChiNoise
    ncchi_std: Range = Range(min=0.0, max=0.000000001)
    ncchi_dof: int = 4

    # transform: SpeckleNoise
    speckle_std: Range = Range(min=0.0, max=0.5)

    # transform: RandomFLip
    flip_prob: float = 1.0

    # transform: SphericalMask / DeformedSphericalMask
    sphere_radius: Range = Range(min=80.0, max=100.0)
    sphere_prob: float = 0.5
    sphere_deform_amplitude: float = 0.3
    sphere_deform_shape: tuple[int, int, int] = (4, 4, 4)

    # GaussianSmoothing
    gsmooth_sigma: Range = Range(min=0.001, max=1.0)
    gsmooth_kernel_size: int = 3
    gsmooth_isotropic: bool = True

    # transform: GaussianSharpening
    sharpen_sigma: Range = Range(min=0.001, max=1.0)
    sharpen_amount: Range = Range(min=0.5, max=2.0)
    sharpen_kernel_size: int = 3

    # transform: GibbsRinging
    gibbs_cutoff: Range = Range(min=0.5, max=1.0)

    # transform: ElasticDeformation
    elastic_sigma: float = 20.0
    elastic_max_disp: float = 20.0

    # transform: CaliberDeformation
    caliber_sigma: float = 10.0
    caliber_alpha: float = 2.0


class Config(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42  # Random seed for reproducibility.
    device: str = "cpu"  # Computation device to use.
    transform_prob: float = 1.0  # Probability for each transform in pipeline.
    geom: GeometryParams = GeometryParams()
    physio: PhysioParams = PhysioParams()
    physics: PhysicsParams = PhysicsParams()
    transform: TransformParams = TransformParams()

    def save(self, path: str | Path) -> None:
        """Saves the configuration to a JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def _deep_merge(cls, base: dict, update: dict) -> dict:
        """Deeply merges two dictionaries."""
        result = dict(base)
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load configuration parameters from dictionary."""
        default = cls().model_dump()
        merged = cls._deep_merge(default, data)
        return cls.model_validate(merged)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load configuration parameters from yaml file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
