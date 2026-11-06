"""Steady-state pressure model for a pipeline made of several segments.

The model starts from a known inlet pressure and marches downstream, using the
outlet pressure of each segment as the inlet pressure of the next.

All pressures are ABSOLUTE, in pascals. All other values are in SI units.

Assumptions:
    - Steady flow: the same flow rate passes through every segment.
    - Flow rate and inlet pressure are inputs. The model does not solve for
      them from pump curves or delivery conditions.
    - Density is constant along the line. Viscosity is too, unless a value is
      supplied for each segment (see thermal.py).
    - Each pump adds a fixed pressure at the inlet of its segment.
    - Pressure varies linearly along a segment, so the lowest pressure in a
      segment is always at one of its ends.

Limitations:
    - If pressure falls below vapour pressure, the real line would stop running
      full. The calculation continues so the problem can be reported, but
      results downstream of a flagged point are not physically meaningful.
    - Maximum allowable operating pressure is not checked.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.fluid_properties import Fluid
from pipeline.hydraulics import (
    elevation_pressure_change,
    fluid_velocity,
    friction_factor,
    friction_pressure_loss,
    pipe_area,
    reynolds_number,
    segment_outlet_pressure,
)
from pipeline.parameters import PipeSegment, PumpStation
from pipeline.validation import require_non_negative, require_positive

# Largest elevation mismatch allowed where two segments join [m].
ELEVATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SegmentResult:
    """Calculated conditions for one segment.

    Attributes:
        segment_name: Name of the segment.
        velocity: Average fluid velocity [m/s].
        reynolds: Reynolds number [dimensionless].
        darcy_friction_factor: Friction factor [dimensionless], or None at zero
            flow, where it is undefined.
        friction_loss: Pressure lost to friction [Pa].
        elevation_pressure: Pressure change from height, positive uphill [Pa].
        pump_boost: Pressure added at the segment inlet [Pa].
        inlet_pressure: Absolute pressure arriving at the segment, upstream of
            any pump [Pa].
        pressure_after_pump: Absolute pressure entering the pipe [Pa].
        outlet_pressure: Absolute pressure leaving the segment [Pa].
        below_vapor_pressure: True if the pressure arriving at the segment or
            leaving it is below the fluid's vapour pressure.
    """

    segment_name: str
    velocity: float
    reynolds: float
    darcy_friction_factor: float | None
    friction_loss: float
    elevation_pressure: float
    pump_boost: float
    inlet_pressure: float
    pressure_after_pump: float
    outlet_pressure: float
    below_vapor_pressure: bool


@dataclass(frozen=True)
class PipelineResult:
    """Calculated conditions for the whole pipeline.

    Attributes:
        flow_rate: Volumetric flow rate [m^3/s].
        segments: One result per segment, ordered from upstream to downstream.
    """

    flow_rate: float
    segments: tuple[SegmentResult, ...]

    @property
    def inlet_pressure(self) -> float:
        """Absolute pressure at the start of the pipeline [Pa]."""
        return self.segments[0].inlet_pressure

    @property
    def outlet_pressure(self) -> float:
        """Absolute pressure at the end of the pipeline [Pa]."""
        return self.segments[-1].outlet_pressure

    @property
    def minimum_pressure(self) -> float:
        """Lowest absolute pressure anywhere in the pipeline [Pa]."""
        return min(
            min(segment.inlet_pressure, segment.outlet_pressure)
            for segment in self.segments
        )

    @property
    def has_low_pressure(self) -> bool:
        """True if any segment fell below the fluid's vapour pressure."""
        return any(segment.below_vapor_pressure for segment in self.segments)


def _validate_layout(
    segments: Sequence[PipeSegment], pumps: Sequence[PumpStation]
) -> None:
    """Raise ValueError unless the segments and pumps form a connected line."""
    if not segments:
        raise ValueError("a pipeline needs at least one segment")

    segment_names = [segment.name for segment in segments]
    if len(set(segment_names)) != len(segment_names):
        raise ValueError(f"segment names must be unique, got {segment_names}")

    # zip pairs each segment with the one after it.
    for upstream, downstream in zip(segments, segments[1:]):
        if not math.isclose(
            upstream.outlet_elevation,
            downstream.inlet_elevation,
            abs_tol=ELEVATION_TOLERANCE,
        ):
            raise ValueError(
                f"elevation gap between {upstream.name} "
                f"({upstream.outlet_elevation} m) and {downstream.name} "
                f"({downstream.inlet_elevation} m)"
            )

    pumped_segments = [pump.segment_name for pump in pumps]
    if len(set(pumped_segments)) != len(pumped_segments):
        raise ValueError(f"only one pump is allowed per segment, got {pumped_segments}")

    for pump in pumps:
        if pump.segment_name not in segment_names:
            raise ValueError(
                f"pump {pump.name} refers to unknown segment {pump.segment_name}"
            )


def _calculate_segment(
    segment: PipeSegment,
    fluid: Fluid,
    dynamic_viscosity: float,
    flow_rate: float,
    inlet_pressure: float,
    pump_boost: float,
) -> SegmentResult:
    """Return the calculated conditions for one segment."""
    velocity = fluid_velocity(flow_rate, pipe_area(segment.internal_diameter))
    reynolds = reynolds_number(
        fluid.density, velocity, segment.internal_diameter, dynamic_viscosity
    )
    darcy_f = None
    if reynolds > 0:
        darcy_f = friction_factor(
            reynolds, segment.roughness / segment.internal_diameter
        )

    friction_loss = friction_pressure_loss(
        fluid.density,
        velocity,
        segment.length,
        segment.internal_diameter,
        dynamic_viscosity,
        segment.roughness,
    )
    elevation_pressure = elevation_pressure_change(
        fluid.density, segment.elevation_change
    )
    outlet_pressure = segment_outlet_pressure(
        inlet_pressure, friction_loss, elevation_pressure, pump_boost
    )

    # The pressure arriving at a pump can be the lowest point on the line, so
    # check it as well as the outlet. A pump only raises pressure afterwards.
    lowest_pressure = min(inlet_pressure, outlet_pressure)

    return SegmentResult(
        segment_name=segment.name,
        velocity=velocity,
        reynolds=reynolds,
        darcy_friction_factor=darcy_f,
        friction_loss=friction_loss,
        elevation_pressure=elevation_pressure,
        pump_boost=pump_boost,
        inlet_pressure=inlet_pressure,
        pressure_after_pump=inlet_pressure + pump_boost,
        outlet_pressure=outlet_pressure,
        below_vapor_pressure=lowest_pressure < fluid.vapor_pressure,
    )


def calculate_pipeline(
    fluid: Fluid,
    flow_rate: float,
    inlet_pressure: float,
    segments: Sequence[PipeSegment],
    pumps: Sequence[PumpStation] = (),
    segment_viscosities: Sequence[float] | None = None,
) -> PipelineResult:
    """Calculate the steady-state pressure through a pipeline.

    Args:
        fluid: The liquid in the pipeline.
        flow_rate: Volumetric flow rate, Q [m^3/s]. Zero means a static line.
        inlet_pressure: Absolute pressure at the start of the line [Pa].
        segments: Pipe segments ordered from upstream to downstream.
        pumps: Pump stations, each at the inlet of a named segment.
        segment_viscosities: Optional dynamic viscosity for each segment, in the
            same order as segments [Pa·s]. If left out, every segment uses
            fluid.dynamic_viscosity.

    Returns:
        The calculated conditions for every segment.

    Raises:
        ValueError: If an input is impossible or the layout is not connected.
    """
    require_non_negative("flow_rate", flow_rate, "m^3/s")
    require_non_negative("inlet_pressure", inlet_pressure, "Pa")
    _validate_layout(segments, pumps)

    if segment_viscosities is None:
        segment_viscosities = [fluid.dynamic_viscosity] * len(segments)
    if len(segment_viscosities) != len(segments):
        raise ValueError(
            f"segment_viscosities needs one value per segment: got "
            f"{len(segment_viscosities)} for {len(segments)} segments"
        )
    for viscosity in segment_viscosities:
        require_positive("segment_viscosities", viscosity, "Pa·s")

    # A dictionary lets us look up each segment's pump boost by name.
    pump_boosts = {pump.segment_name: pump.pressure_boost for pump in pumps}

    results = []
    pressure = inlet_pressure
    for segment, viscosity in zip(segments, segment_viscosities):
        boost = pump_boosts.get(segment.name, 0.0)
        result = _calculate_segment(
            segment, fluid, viscosity, flow_rate, pressure, boost
        )
        results.append(result)
        pressure = result.outlet_pressure  # Becomes the next segment's inlet.

    return PipelineResult(flow_rate=flow_rate, segments=tuple(results))
