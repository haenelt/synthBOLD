"""Configuration.

This module provides the validated configuration structures and custom validators
used throughout the synthBOLD package to parse and ensure data synthesis parameters
are correct.
"""

from pathlib import Path
from typing import Self

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
    """

    input_shape: tuple[int, int, int] = (128, 128, 128)
    output_shape: tuple[int, int, int] = (32, 32, 32)
    fov: tuple[float, float, float] = (32.0, 32.0, 32.0)

    @field_validator("output_shape")
    @classmethod
    def check_divisible_output(
        cls, v: tuple[int, ...], info: ValidationInfo
    ) -> tuple[int, ...]:
        """Validates that input_shape is divisible by output_shape in every dim."""
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
    """Parameters for generating tissue and vessel maps.

    Args:
        label_displacement_shape: Shape of noise map for smooth displacement field.
        label_J: Number of tissue classes.
        label_scale: Scale factor of the noise map.
        num_cylinders: Number of generated cylinders.
        vf: Volume fraction.
        cylinder_diameter: Range of cylinder diameters in mm.
        allow_overlap: Allow cylinders to overlap.
        tissue_mu: Mean intensity range for different tissue compartments.
        tissue_std: Standard deviation range for intensities within tissue types.

    Note:
        num_vessels and cbv are mutually exclusive.
    """

    label_displacement_shape: tuple[int, int, int] = (4, 4, 4)
    label_J: int = 8
    label_scale: float = 0.2
    num_cylinders: int | None = None
    vf: Range = Range(min=0.005, max=0.01)
    cylinder_diameter: Range = Range(min=0.25, max=2.0)
    allow_overlap: bool = True
    tissue_mu: Range = Range(min=10.0, max=100.0)
    tissue_std: Range = Range(min=5.0, max=10.0)


class Config(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42  # Random seed for reproducibility.
    device: str = "cuda"  # Computation device to use.
    geom: GeometryParams = GeometryParams()
    physio: PhysioParams = PhysioParams()

    def save(self, path: str | Path) -> None:
        """Saves the configuration to a JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def deep_merge(cls, base: dict, update: dict) -> dict:
        result = dict(base)
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls.deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load configuration parameters from dictionary."""
        default = cls().model_dump()
        merged = cls.deep_merge(default, data)
        return cls.model_validate(merged)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load configuration parameters from yaml file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
