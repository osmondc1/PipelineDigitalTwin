"""Simulated pipeline sensors.

A reading is built from three parts:

    reading = true value + fixed bias + random noise

The true value comes from the engineering model. Sensors report in the units a
monitoring system would use (barg, m3/h, degC), while the model stays in SI.
All readings here are SIMULATED, and accuracy values are assumptions.
"""

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.parameters import PipeSegment
from pipeline.pipeline_model import PipelineResult
from pipeline.report import to_bar_gauge
from pipeline.thermal import SegmentTemperature
from pipeline.validation import require_name, require_non_negative, require_positive

SECONDS_PER_HOUR = 3600.0
KELVIN_OFFSET = 273.15
M_PER_KM = 1000.0

# The unit each measured quantity is reported in.
UNITS = {"pressure": "barg", "flow": "m3/h", "temperature": "degC"}


@dataclass(frozen=True)
class Sensor:
    """One simulated instrument.

    Attributes:
        sensor_id: Unique tag, for example "PT-001".
        quantity: What it measures: "pressure", "flow" or "temperature".
        location: Plain-language description of where it is.
        location_km: Distance from the start of the line [km].
        sampling_interval: Time between readings [s].
        noise_std: Standard deviation of random noise, in the sensor's unit.
        bias_std: Standard deviation of the fixed calibration error across
            sensors, in the sensor's unit.
        missing_probability: Chance that a sample is not recorded, from 0 to 1.
        upstream_of_pump: For a pressure sensor at a pump, True if it sits on
            the pump's suction side and False if on its discharge side.
    """

    sensor_id: str
    quantity: str
    location: str
    location_km: float
    sampling_interval: float
    noise_std: float
    bias_std: float
    missing_probability: float
    upstream_of_pump: bool = False

    def __post_init__(self) -> None:
        """Reject impossible sensors as soon as one is created."""
        require_name("sensor_id", self.sensor_id)
        require_name("location", self.location)
        if self.quantity not in UNITS:
            raise ValueError(
                f"quantity must be one of {sorted(UNITS)}, got {self.quantity!r}"
            )
        require_non_negative("location_km", self.location_km, "km")
        require_positive("sampling_interval", self.sampling_interval, "s")
        require_non_negative("noise_std", self.noise_std, self.unit)
        require_non_negative("bias_std", self.bias_std, self.unit)
        require_non_negative("missing_probability", self.missing_probability, "")
        if self.missing_probability > 1:
            raise ValueError(
                f"missing_probability must be <= 1, got {self.missing_probability}"
            )

    @property
    def unit(self) -> str:
        """The unit this sensor reports in."""
        return UNITS[self.quantity]

    @property
    def uncertainty(self) -> float:
        """Combined standard uncertainty, in the sensor's unit.

        Independent errors combine as the root of the sum of their squares.
        """
        return math.sqrt(self.noise_std**2 + self.bias_std**2)


def load_sensors(path: Path) -> tuple[Sensor, ...]:
    """Read the sensor network from a TOML file.

    Args:
        path: Location of the TOML file.

    Returns:
        The sensors, in file order.

    Raises:
        ValueError: If a setting is missing or impossible, or an ID is repeated.
        FileNotFoundError: If the file does not exist.
    """
    with path.open("rb") as file:
        data = tomllib.load(file)

    try:
        sensors = tuple(_build_sensor(row) for row in data["sensors"])
    except KeyError as error:
        raise ValueError(f"{path} is missing the setting {error}") from error

    sensor_ids = [sensor.sensor_id for sensor in sensors]
    if len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError(f"sensor IDs must be unique, got {sensor_ids}")
    return sensors


def _build_sensor(settings: dict) -> Sensor:
    """Build a Sensor from one [[sensors]] entry."""
    return Sensor(
        sensor_id=settings["sensor_id"],
        quantity=settings["quantity"],
        location=settings["location"],
        location_km=settings["location_km"],
        sampling_interval=settings["sampling_interval_s"],
        noise_std=settings["noise_std"],
        bias_std=settings["bias_std"],
        missing_probability=settings["missing_probability"],
        upstream_of_pump=settings.get("upstream_of_pump", False),
    )


def _segment_starting_at(
    location_km: float, segments: Sequence[PipeSegment]
) -> int | None:
    """Return the index of the segment starting here, or None at the line's end.

    Raises:
        ValueError: If the location is not at a segment boundary, because the
            model only calculates conditions at those points.
    """
    location = location_km * M_PER_KM
    start = 0.0
    for index, segment in enumerate(segments):
        if math.isclose(start, location, abs_tol=1e-6):
            return index
        start += segment.length

    if math.isclose(start, location, abs_tol=1e-6):
        return None
    raise ValueError(f"no segment boundary at {location_km} km")


def true_value(
    sensor: Sensor,
    segments: Sequence[PipeSegment],
    result: PipelineResult,
    temperatures: Sequence[SegmentTemperature],
) -> float:
    """Return the model's exact value at a sensor, in the sensor's unit.

    Args:
        sensor: The sensor.
        segments: Pipe segments ordered from upstream to downstream.
        result: Pressure model results.
        temperatures: Temperature model results.

    Returns:
        The true value, before any measurement error.
    """
    index = _segment_starting_at(sensor.location_km, segments)

    if sensor.quantity == "flow":
        return result.flow_rate * SECONDS_PER_HOUR

    if sensor.quantity == "temperature":
        kelvin = (
            temperatures[-1].outlet_temperature
            if index is None
            else temperatures[index].inlet_temperature
        )
        return kelvin - KELVIN_OFFSET

    if index is None:
        return to_bar_gauge(result.outlet_pressure)
    conditions = result.segments[index]
    if sensor.upstream_of_pump:
        return to_bar_gauge(conditions.inlet_pressure)
    return to_bar_gauge(conditions.pressure_after_pump)


def draw_biases(
    sensors: Sequence[Sensor], rng: np.random.Generator
) -> dict[str, float]:
    """Draw one fixed calibration error for each sensor.

    Args:
        sensors: The sensors.
        rng: Random number generator. Using a seeded one makes results repeat.

    Returns:
        Each sensor's bias, keyed by sensor ID.
    """
    return {
        sensor.sensor_id: float(rng.normal(0.0, sensor.bias_std)) for sensor in sensors
    }


def take_reading(
    sensor: Sensor, value: float, bias: float, rng: np.random.Generator
) -> float | None:
    """Return one simulated reading, or None if the sample is missing.

    Both random numbers are always drawn, so changing the missing rate does not
    change the noise on the readings that remain.

    Args:
        sensor: The sensor.
        value: The true value, in the sensor's unit.
        bias: This sensor's fixed calibration error.
        rng: Random number generator.

    Returns:
        The reading, or None if it was not recorded.
    """
    noise = float(rng.normal(0.0, sensor.noise_std))
    is_missing = rng.random() < sensor.missing_probability
    return None if is_missing else value + bias + noise
