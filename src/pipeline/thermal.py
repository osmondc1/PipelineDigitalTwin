"""Steady-state temperature along the pipeline, and its effect on viscosity.

All temperatures are in kelvin. All other values are in SI units.

Assumptions:
    - Heat only moves between the oil and the surrounding ground, which stays
      at one constant temperature.
    - The heat transfer coefficient and specific heat are constant.
    - Heating of the oil by friction is ignored.
    - Temperature responds instantly to changes. Real oil takes hours to travel
      100 km, so a change at the inlet would reach the outlet much later.
    - Viscosity is evaluated at each segment's average temperature.
    - Density does not change with temperature.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.parameters import PipeSegment
from pipeline.validation import require_non_negative, require_positive


@dataclass(frozen=True)
class ThermalProperties:
    """Thermal behaviour of the oil and the ground around the pipe.

    Attributes:
        specific_heat: Energy to warm 1 kg of oil by 1 K, cp [J/(kg·K)].
        heat_transfer_coefficient: Heat flow through the pipe wall and ground
            per m^2 of wall per K, U [W/(m^2·K)]. Zero means perfect insulation.
        ground_temperature: Temperature of the surrounding ground, Tg [K].
        reference_temperature: Temperature at which the fluid has its stated
            dynamic viscosity, Tref [K].
        viscosity_temperature_coefficient: How quickly viscosity falls as
            temperature rises, beta [1/K]. Zero means no temperature effect.
    """

    specific_heat: float
    heat_transfer_coefficient: float
    ground_temperature: float
    reference_temperature: float
    viscosity_temperature_coefficient: float

    def __post_init__(self) -> None:
        """Reject impossible values as soon as the object is created."""
        require_positive("specific_heat", self.specific_heat, "J/(kg·K)")
        require_non_negative(
            "heat_transfer_coefficient", self.heat_transfer_coefficient, "W/(m^2·K)"
        )
        require_positive("ground_temperature", self.ground_temperature, "K")
        require_positive("reference_temperature", self.reference_temperature, "K")
        require_non_negative(
            "viscosity_temperature_coefficient",
            self.viscosity_temperature_coefficient,
            "1/K",
        )


@dataclass(frozen=True)
class SegmentTemperature:
    """Temperatures for one segment.

    Attributes:
        segment_name: Name of the segment.
        inlet_temperature: Oil temperature entering the segment [K].
        outlet_temperature: Oil temperature leaving the segment [K].
    """

    segment_name: str
    inlet_temperature: float
    outlet_temperature: float

    @property
    def average_temperature(self) -> float:
        """Mean of the inlet and outlet temperatures [K]."""
        return (self.inlet_temperature + self.outlet_temperature) / 2


def segment_outlet_temperature(
    inlet_temperature: float,
    thermal: ThermalProperties,
    internal_diameter: float,
    length: float,
    mass_flow_rate: float,
) -> float:
    """Return the oil temperature at the end of a segment.

    Equation: T_out = Tg + (T_in - Tg) * exp(-U * pi * D * L / (m_dot * cp))

    Oil approaches ground temperature with distance. The exponent compares the
    heat that can escape through the pipe wall with the heat the moving oil
    carries.

    Args:
        inlet_temperature: Oil temperature entering the segment, T_in [K].
        thermal: Thermal properties.
        internal_diameter: Inside diameter, D [m].
        length: Segment length, L [m].
        mass_flow_rate: Mass of oil passing per second, m_dot [kg/s]. At zero
            flow the oil has settled to ground temperature.

    Returns:
        Outlet temperature, T_out [K].
    """
    require_positive("inlet_temperature", inlet_temperature, "K")
    require_positive("internal_diameter", internal_diameter, "m")
    require_positive("length", length, "m")
    require_non_negative("mass_flow_rate", mass_flow_rate, "kg/s")

    # Perfect insulation comes first, so no heat is lost even at zero flow.
    if thermal.heat_transfer_coefficient == 0:
        return inlet_temperature
    if mass_flow_rate == 0:
        return thermal.ground_temperature

    wall_area = math.pi * internal_diameter * length
    exponent = (thermal.heat_transfer_coefficient * wall_area) / (
        mass_flow_rate * thermal.specific_heat
    )
    ground = thermal.ground_temperature
    return ground + (inlet_temperature - ground) * math.exp(-exponent)


def viscosity_at_temperature(
    reference_viscosity: float, temperature: float, thermal: ThermalProperties
) -> float:
    """Return the dynamic viscosity at a temperature.

    Equation: mu = mu_ref * exp(-beta * (T - Tref))

    This simple exponential fit only suits a moderate range around Tref.

    Args:
        reference_viscosity: Viscosity at the reference temperature, mu_ref [Pa·s].
        temperature: Oil temperature, T [K].
        thermal: Thermal properties.

    Returns:
        Dynamic viscosity, mu [Pa·s].
    """
    require_positive("reference_viscosity", reference_viscosity, "Pa·s")
    require_positive("temperature", temperature, "K")
    temperature_difference = temperature - thermal.reference_temperature
    return reference_viscosity * math.exp(
        -thermal.viscosity_temperature_coefficient * temperature_difference
    )


def calculate_temperature_profile(
    segments: Sequence[PipeSegment],
    thermal: ThermalProperties,
    density: float,
    flow_rate: float,
    inlet_temperature: float,
) -> tuple[SegmentTemperature, ...]:
    """Return the temperatures of every segment, marching downstream.

    Args:
        segments: Pipe segments ordered from upstream to downstream.
        thermal: Thermal properties.
        density: Fluid density [kg/m^3].
        flow_rate: Volumetric flow rate [m^3/s].
        inlet_temperature: Oil temperature at the start of the line [K].

    Returns:
        One SegmentTemperature per segment.
    """
    require_positive("density", density, "kg/m^3")
    require_non_negative("flow_rate", flow_rate, "m^3/s")
    mass_flow_rate = density * flow_rate

    results = []
    temperature = inlet_temperature
    for segment in segments:
        outlet = segment_outlet_temperature(
            temperature,
            thermal,
            segment.internal_diameter,
            segment.length,
            mass_flow_rate,
        )
        results.append(SegmentTemperature(segment.name, temperature, outlet))
        temperature = outlet  # Becomes the next segment's inlet.

    return tuple(results)


def segment_viscosities(
    temperatures: Sequence[SegmentTemperature],
    reference_viscosity: float,
    thermal: ThermalProperties,
) -> tuple[float, ...]:
    """Return each segment's viscosity at its average temperature [Pa·s].

    The result can be passed directly to calculate_pipeline.
    """
    return tuple(
        viscosity_at_temperature(
            reference_viscosity, segment.average_temperature, thermal
        )
        for segment in temperatures
    )
