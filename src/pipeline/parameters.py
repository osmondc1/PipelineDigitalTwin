"""Physical description of the pipeline: segments and pump stations.

All values are in SI units: lengths and elevations in metres, pressures in
pascals. These classes only describe the pipeline. The hydraulic calculations
live in hydraulics.py and pipeline_model.py.

Assumptions:
    - Each segment has one constant internal diameter and roughness.
    - Elevation changes linearly along a segment, so any high point between
      its two ends is not represented.
    - A pump station gives a fixed pressure boost, independent of flow rate.
"""

from dataclasses import dataclass

from pipeline.validation import (
    require_finite,
    require_name,
    require_non_negative,
    require_positive,
)


@dataclass(frozen=True)
class PipeSegment:
    """One straight run of pipe with constant properties.

    Attributes:
        name: Unique label for the segment, for example "S1".
        length: Length measured along the pipe, L [m].
        internal_diameter: Inside diameter, D [m].
        roughness: Absolute roughness of the pipe wall, epsilon [m]. Zero means
            a perfectly smooth wall. Commercial steel is about 45e-6 m.
        inlet_elevation: Height of the upstream end above datum, z1 [m].
        outlet_elevation: Height of the downstream end above datum, z2 [m].
    """

    name: str
    length: float
    internal_diameter: float
    roughness: float
    inlet_elevation: float
    outlet_elevation: float

    def __post_init__(self) -> None:
        """Reject impossible segments as soon as one is created."""
        require_name("name", self.name)
        require_positive("length", self.length, "m")
        require_positive("internal_diameter", self.internal_diameter, "m")
        require_non_negative("roughness", self.roughness, "m")
        require_finite("inlet_elevation", self.inlet_elevation, "m")
        require_finite("outlet_elevation", self.outlet_elevation, "m")

        if self.roughness >= self.internal_diameter:
            raise ValueError(
                f"roughness ({self.roughness} m) must be smaller than "
                f"internal_diameter ({self.internal_diameter} m)"
            )

        # A pipe cannot climb further than its own length. This usually means
        # a length was entered in kilometres instead of metres.
        if abs(self.elevation_change) > self.length:
            raise ValueError(
                f"elevation change ({self.elevation_change} m) cannot exceed "
                f"length ({self.length} m)"
            )

    @property
    def elevation_change(self) -> float:
        """Outlet minus inlet elevation, z2 - z1 [m]. Positive means uphill."""
        return self.outlet_elevation - self.inlet_elevation


@dataclass(frozen=True)
class PumpStation:
    """A pump station located at the inlet of a pipe segment.

    Attributes:
        name: Label for the station, for example "PS1".
        segment_name: Name of the segment whose inlet the station feeds.
        pressure_boost: Pressure added to the flow, delta_p [Pa]. Zero is
            allowed and represents a station that is switched off.
    """

    name: str
    segment_name: str
    pressure_boost: float

    def __post_init__(self) -> None:
        """Reject impossible pump stations as soon as one is created."""
        require_name("name", self.name)
        require_name("segment_name", self.segment_name)
        require_non_negative("pressure_boost", self.pressure_boost, "Pa")
