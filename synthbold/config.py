"""Configuration.

This module provides the validated configuration structures and custom validators
used throughout the synthBOLD package to parse and ensure data synthesis parameters
are correct.
"""

from pydantic import BaseModel, Field, ValidationInfo, field_validator

__all__ = ["Range", "Config"]


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


class Config(BaseModel):
    pass
