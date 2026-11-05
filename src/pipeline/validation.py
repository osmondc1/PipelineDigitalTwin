"""Shared input checks used across the pipeline package.

Each function raises ValueError if its input is not acceptable, and does
nothing otherwise.
"""

import math


def require_positive(field_name: str, value: float, unit: str) -> None:
    """Raise ValueError unless value is a finite number greater than zero."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be > 0 {unit}, got {value}")


def require_non_negative(field_name: str, value: float, unit: str) -> None:
    """Raise ValueError unless value is a finite number of zero or more."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be >= 0 {unit}, got {value}")


def require_finite(field_name: str, value: float, unit: str) -> None:
    """Raise ValueError unless value is a finite number (it may be negative)."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number of {unit}, got {value}")


def require_name(field_name: str, value: str) -> None:
    """Raise ValueError if a name is empty or only spaces."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
