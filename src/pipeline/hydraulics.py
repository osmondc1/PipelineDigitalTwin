"""Single-phase liquid hydraulics calculations.

All inputs and outputs are in SI units. Each function takes plain numbers, so
it can be tested and checked by hand without building a whole pipeline.

Assumptions:
    - The pipe is circular and runs completely full of liquid.
    - Flow rate is the same at every point in a segment (steady flow of an
      incompressible liquid).
    - Flow travels in one direction only, from inlet to outlet.
"""

import math

from pipeline.validation import require_non_negative, require_positive


def pipe_area(internal_diameter: float) -> float:
    """Return the internal cross-sectional area of a circular pipe.

    Equation: A = pi * D^2 / 4

    Args:
        internal_diameter: Inside diameter, D [m].

    Returns:
        Cross-sectional area, A [m^2].
    """
    require_positive("internal_diameter", internal_diameter, "m")
    return math.pi * internal_diameter**2 / 4


def fluid_velocity(flow_rate: float, area: float) -> float:
    """Return the average fluid velocity across the pipe.

    Equation: v = Q / A

    Args:
        flow_rate: Volumetric flow rate, Q [m^3/s]. Zero means no flow.
        area: Internal cross-sectional area, A [m^2].

    Returns:
        Average velocity, v [m/s].
    """
    require_non_negative("flow_rate", flow_rate, "m^3/s")
    require_positive("area", area, "m^2")
    return flow_rate / area


def reynolds_number(
    density: float,
    velocity: float,
    internal_diameter: float,
    dynamic_viscosity: float,
) -> float:
    """Return the Reynolds number, the ratio of inertial to viscous forces.

    Equation: Re = rho * v * D / mu

    Args:
        density: Fluid density, rho [kg/m^3].
        velocity: Average fluid velocity, v [m/s]. Zero means no flow.
        internal_diameter: Inside diameter, D [m].
        dynamic_viscosity: Fluid dynamic viscosity, mu [Pa·s].

    Returns:
        Reynolds number, Re [dimensionless].
    """
    require_positive("density", density, "kg/m^3")
    require_non_negative("velocity", velocity, "m/s")
    require_positive("internal_diameter", internal_diameter, "m")
    require_positive("dynamic_viscosity", dynamic_viscosity, "Pa·s")
    return density * velocity * internal_diameter / dynamic_viscosity
