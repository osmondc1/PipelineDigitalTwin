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

from pipeline.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)

# Standard acceleration due to gravity [m/s^2].
GRAVITY = 9.80665

# Flow regime limits [dimensionless Reynolds number]. These are conventional
# engineering values, not sharp physical boundaries.
LAMINAR_REYNOLDS_LIMIT = 2300.0
TURBULENT_REYNOLDS_LIMIT = 4000.0

# Top of the standard Moody chart [dimensionless]. Anything above this is
# treated as an input mistake, usually a roughness in the wrong units.
MAX_RELATIVE_ROUGHNESS = 0.05


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


def _laminar_friction_factor(reynolds: float) -> float:
    """Return the laminar Darcy friction factor, f = 64 / Re."""
    return 64.0 / reynolds


def _turbulent_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Return the turbulent Darcy friction factor from Swamee-Jain.

    Equation: f = 0.25 / [log10(rr / 3.7 + 5.74 / Re^0.9)]^2

    Swamee-Jain is an explicit approximation of the Colebrook equation, rated
    for 5000 <= Re <= 1e8 and 1e-6 <= rr <= 0.01. This model also uses it
    slightly outside that range, from Re = 4000 and up to rr = 0.05.

    Measured difference from Colebrook: within 1% for 1e4 <= Re <= 1e7 with
    rr <= 1e-3, about 2.8% at worst in the rated range (Re = 5000, rr = 0.01),
    and about 3.1% at worst outside it (Re = 4000).
    """
    logarithm = math.log10(relative_roughness / 3.7 + 5.74 / reynolds**0.9)
    return 0.25 / logarithm**2


def friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Return the Darcy friction factor for any flow regime.

    - Laminar (Re <= 2300): f = 64 / Re
    - Turbulent (Re >= 4000): Swamee-Jain approximation
    - Transitional (2300 < Re < 4000): straight-line interpolation between the
      two limits. This is a modelling assumption that keeps the result
      continuous. Real transitional flow is unstable and not predictable.

    Zero flow is rejected because f is undefined at Re = 0. The pressure loss
    calculation handles zero flow itself.

    Args:
        reynolds: Reynolds number, Re [dimensionless]. Must be greater than zero.
        relative_roughness: Roughness divided by internal diameter, epsilon / D
            [dimensionless]. Must be between 0 and 0.05.

    Returns:
        Darcy friction factor, f [dimensionless].
    """
    require_positive("reynolds", reynolds, "(dimensionless)")
    require_non_negative("relative_roughness", relative_roughness, "(dimensionless)")
    if relative_roughness > MAX_RELATIVE_ROUGHNESS:
        raise ValueError(
            f"relative_roughness must be <= {MAX_RELATIVE_ROUGHNESS}, got "
            f"{relative_roughness}. Check that roughness is in metres."
        )

    if reynolds <= LAMINAR_REYNOLDS_LIMIT:
        return _laminar_friction_factor(reynolds)

    if reynolds >= TURBULENT_REYNOLDS_LIMIT:
        return _turbulent_friction_factor(reynolds, relative_roughness)

    # Transitional: move in a straight line between the two regime limits.
    laminar_f = _laminar_friction_factor(LAMINAR_REYNOLDS_LIMIT)
    turbulent_f = _turbulent_friction_factor(
        TURBULENT_REYNOLDS_LIMIT, relative_roughness
    )
    fraction = (reynolds - LAMINAR_REYNOLDS_LIMIT) / (
        TURBULENT_REYNOLDS_LIMIT - LAMINAR_REYNOLDS_LIMIT
    )
    return laminar_f + fraction * (turbulent_f - laminar_f)


def friction_pressure_loss(
    density: float,
    velocity: float,
    length: float,
    internal_diameter: float,
    dynamic_viscosity: float,
    roughness: float,
) -> float:
    """Return the pressure lost to wall friction, using Darcy-Weisbach.

    Equation: delta_p = f * (L / D) * (rho * v^2 / 2)

    Fittings, bends and valves are not included.

    Args:
        density: Fluid density, rho [kg/m^3].
        velocity: Average fluid velocity, v [m/s]. Zero gives zero loss.
        length: Pipe length, L [m].
        internal_diameter: Inside diameter, D [m].
        dynamic_viscosity: Fluid dynamic viscosity, mu [Pa·s].
        roughness: Absolute wall roughness, epsilon [m].

    Returns:
        Friction pressure loss, delta_p [Pa]. Always zero or positive.
    """
    require_positive("length", length, "m")
    require_non_negative("roughness", roughness, "m")

    reynolds = reynolds_number(density, velocity, internal_diameter, dynamic_viscosity)
    if reynolds == 0:
        return 0.0  # No flow means no friction, and f is undefined at Re = 0.

    darcy_f = friction_factor(reynolds, roughness / internal_diameter)
    dynamic_pressure = density * velocity**2 / 2
    return darcy_f * (length / internal_diameter) * dynamic_pressure


def elevation_pressure_change(density: float, elevation_change: float) -> float:
    """Return the pressure needed to lift the fluid through a height change.

    Equation: delta_p = rho * g * delta_z

    Args:
        density: Fluid density, rho [kg/m^3].
        elevation_change: Outlet minus inlet elevation, delta_z [m].

    Returns:
        Pressure change, delta_p [Pa]. Positive uphill (pressure is lost) and
        negative downhill (pressure is gained).
    """
    require_positive("density", density, "kg/m^3")
    require_finite("elevation_change", elevation_change, "m")
    return density * GRAVITY * elevation_change


def segment_outlet_pressure(
    inlet_pressure: float,
    friction_loss: float,
    elevation_change_pressure: float,
    pump_boost: float = 0.0,
) -> float:
    """Return the pressure at the outlet of a segment.

    Equation: p_out = p_in + pump_boost - friction_loss - elevation_change_pressure

    This is the only place the sign convention is applied. The result may be
    low or negative, which means the operating point is not physically
    achievable. Checking for that is the job of the pipeline model.

    Args:
        inlet_pressure: Pressure just upstream of any pump, p_in [Pa].
        friction_loss: Friction pressure loss in the segment [Pa].
        elevation_change_pressure: Result of elevation_pressure_change [Pa].
        pump_boost: Pressure added at the segment inlet [Pa]. Defaults to zero.

    Returns:
        Outlet pressure, p_out [Pa].
    """
    require_finite("inlet_pressure", inlet_pressure, "Pa")
    require_non_negative("friction_loss", friction_loss, "Pa")
    require_finite("elevation_change_pressure", elevation_change_pressure, "Pa")
    require_non_negative("pump_boost", pump_boost, "Pa")
    return inlet_pressure + pump_boost - friction_loss - elevation_change_pressure
