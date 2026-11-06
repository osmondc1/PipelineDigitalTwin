"""Load a pipeline case from a TOML configuration file."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pipeline.fluid_properties import Fluid
from pipeline.parameters import PipeSegment, PumpStation
from pipeline.thermal import ThermalProperties


@dataclass(frozen=True)
class PipelineCase:
    """Everything the pipeline model needs for one run.

    Attributes:
        name: Description of the case.
        fluid: The liquid in the pipeline.
        flow_rate: Volumetric flow rate [m^3/s].
        inlet_pressure: Absolute pressure at the start of the line [Pa].
        segments: Pipe segments ordered from upstream to downstream.
        pumps: Pump stations.
        thermal: Thermal properties, or None if the case does not model
            temperature.
    """

    name: str
    fluid: Fluid
    flow_rate: float
    inlet_pressure: float
    segments: tuple[PipeSegment, ...]
    pumps: tuple[PumpStation, ...]
    thermal: ThermalProperties | None = None


def load_case(path: Path) -> PipelineCase:
    """Read a TOML file and build a PipelineCase from it.

    The Fluid, PipeSegment and PumpStation classes check their own values, so
    an impossible number is rejected here too. Whether the segments connect
    properly is checked when the case is run.

    Args:
        path: Location of the TOML file.

    Returns:
        The loaded case.

    Raises:
        ValueError: If a setting is missing or impossible, or the file is not
            valid TOML.
        FileNotFoundError: If the file does not exist.
    """
    with path.open("rb") as file:  # tomllib requires binary mode ("rb").
        data = tomllib.load(file)

    try:
        return PipelineCase(
            name=data["name"],
            fluid=_build_fluid(data["fluid"]),
            flow_rate=data["flow_rate_m3_per_s"],
            inlet_pressure=data["inlet_pressure_pa_abs"],
            segments=tuple(_build_segment(row) for row in data["segments"]),
            pumps=tuple(_build_pump(row) for row in data.get("pumps", [])),
            thermal=_build_thermal(data["thermal"]) if "thermal" in data else None,
        )
    except KeyError as error:
        raise ValueError(f"{path} is missing the setting {error}") from error


def _build_fluid(settings: dict) -> Fluid:
    """Build a Fluid from the [fluid] table."""
    return Fluid(
        density=settings["density_kg_per_m3"],
        dynamic_viscosity=settings["dynamic_viscosity_pa_s"],
        vapor_pressure=settings["vapor_pressure_pa_abs"],
    )


def _build_thermal(settings: dict) -> ThermalProperties:
    """Build ThermalProperties from the [thermal] table."""
    return ThermalProperties(
        specific_heat=settings["specific_heat_j_per_kg_k"],
        heat_transfer_coefficient=settings["heat_transfer_coefficient_w_per_m2_k"],
        ground_temperature=settings["ground_temperature_k"],
        reference_temperature=settings["reference_temperature_k"],
        viscosity_temperature_coefficient=settings[
            "viscosity_temperature_coefficient_per_k"
        ],
    )


def _build_segment(settings: dict) -> PipeSegment:
    """Build a PipeSegment from one [[segments]] entry."""
    return PipeSegment(
        name=settings["name"],
        length=settings["length_m"],
        internal_diameter=settings["internal_diameter_m"],
        roughness=settings["roughness_m"],
        inlet_elevation=settings["inlet_elevation_m"],
        outlet_elevation=settings["outlet_elevation_m"],
    )


def _build_pump(settings: dict) -> PumpStation:
    """Build a PumpStation from one [[pumps]] entry."""
    return PumpStation(
        name=settings["name"],
        segment_name=settings["segment_name"],
        pressure_boost=settings["pressure_boost_pa"],
    )
