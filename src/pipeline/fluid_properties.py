"""Fluid properties used by the hydraulic model.

All values are in SI units.

Assumptions:
    - Properties are constant along the pipeline (no temperature effects yet).
    - The fluid is an incompressible Newtonian liquid.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Fluid:
    """A liquid with constant properties.

    Attributes:
        density: Mass per unit volume, rho [kg/m^3]. Must be greater than zero.
        dynamic_viscosity: Resistance to flow, mu [Pa·s]. Must be greater than
            zero, because the Reynolds number divides by it.
        vapor_pressure: Absolute pressure below which the liquid starts to
            boil, Pv [Pa]. Must be zero or greater.
    """

    density: float
    dynamic_viscosity: float
    vapor_pressure: float

    def __post_init__(self) -> None:
        """Reject physically impossible values as soon as a Fluid is created."""
        # isfinite comes first because NaN ("not a number") is neither greater
        # nor less than anything, so "value <= 0" alone would let NaN through.
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError(f"density must be > 0 kg/m^3, got {self.density}")

        if not math.isfinite(self.dynamic_viscosity) or self.dynamic_viscosity <= 0:
            raise ValueError(
                f"dynamic_viscosity must be > 0 Pa·s, got {self.dynamic_viscosity}"
            )

        if not math.isfinite(self.vapor_pressure) or self.vapor_pressure < 0:
            raise ValueError(
                f"vapor_pressure must be >= 0 Pa, got {self.vapor_pressure}"
            )

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity, nu = mu / rho [m^2/s]."""
        return self.dynamic_viscosity / self.density
