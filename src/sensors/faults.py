"""Sensor malfunctions.

A fault changes what an instrument reports, not what the pipeline is doing. The
true values are never altered, so a detection method can later be scored against
what really happened.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.validation import require_name, require_non_negative

FAULT_KINDS = ("stuck", "drift", "spike", "bias_shift", "dropout")

# These kinds have no effect unless they have a size.
KINDS_NEEDING_MAGNITUDE = ("drift", "spike", "bias_shift")


@dataclass(frozen=True)
class SensorFault:
    """A scripted malfunction of one sensor.

    Attributes:
        sensor_id: The affected sensor.
        kind: One of FAULT_KINDS.
        start_day: When the fault begins [days from the start].
        end_day: When the fault ends [days from the start].
        magnitude: Size of the error, in the sensor's unit. Not used by stuck or
            dropout faults.
        spike_probability: Chance of a spike at each reading, for spike faults.
    """

    sensor_id: str
    kind: str
    start_day: float
    end_day: float
    magnitude: float = 0.0
    spike_probability: float = 0.0

    def __post_init__(self) -> None:
        """Reject impossible faults as soon as one is created."""
        require_name("sensor_id", self.sensor_id)
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"kind must be one of {FAULT_KINDS}, got {self.kind!r}")
        require_non_negative("start_day", self.start_day, "days")
        if self.end_day <= self.start_day:
            raise ValueError(f"{self.sensor_id}: end_day must be after start_day")
        if self.kind in KINDS_NEEDING_MAGNITUDE and self.magnitude == 0:
            raise ValueError(f"{self.sensor_id}: {self.kind} needs a magnitude")
        if self.kind == "spike" and not 0 < self.spike_probability <= 1:
            raise ValueError(
                f"{self.sensor_id}: spike_probability must be above 0 and at most 1"
            )


def load_faults(path: Path) -> tuple[SensorFault, ...]:
    """Read the [[sensor_faults]] entries from a TOML file.

    Args:
        path: Location of the TOML file.

    Returns:
        The faults, or an empty tuple if the file lists none.
    """
    with path.open("rb") as file:
        data = tomllib.load(file)
    return tuple(SensorFault(**row) for row in data.get("sensor_faults", []))


def check_faults(faults: tuple[SensorFault, ...], sensor_ids: list[str]) -> None:
    """Raise ValueError if a fault names an unknown sensor or overlaps another."""
    for fault in faults:
        if fault.sensor_id not in sensor_ids:
            raise ValueError(f"fault refers to unknown sensor {fault.sensor_id}")

    ordered = sorted(faults, key=lambda fault: (fault.sensor_id, fault.start_day))
    for earlier, later in zip(ordered, ordered[1:]):
        same_sensor = earlier.sensor_id == later.sensor_id
        if same_sensor and later.start_day < earlier.end_day:
            raise ValueError(f"faults on {later.sensor_id} overlap")


def fault_window(fault: SensorFault, days: np.ndarray) -> np.ndarray:
    """Return True for each time that falls inside the fault.

    Args:
        fault: The fault.
        days: Time of each reading [days from the start].

    Returns:
        A True or False value per reading.
    """
    return (days >= fault.start_day) & (days < fault.end_day)


def apply_fault(
    readings: np.ndarray,
    fault: SensorFault,
    days: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of the readings with a fault applied.

    Missing readings stay missing, except that a dropout removes all readings in
    its window.

    Args:
        readings: Readings, with NaN for missing values.
        fault: The fault.
        days: Time of each reading [days from the start].
        rng: Random number generator.

    Returns:
        The faulty readings.
    """
    faulty = readings.copy()
    window = fault_window(fault, days)

    if fault.kind == "dropout":
        faulty[window] = np.nan

    elif fault.kind == "bias_shift":
        faulty[window] += fault.magnitude

    elif fault.kind == "drift":
        duration = fault.end_day - fault.start_day
        progress = (days[window] - fault.start_day) / duration
        faulty[window] += fault.magnitude * progress

    elif fault.kind == "spike":
        spikes = rng.random(int(window.sum())) < fault.spike_probability
        signs = rng.choice([-1.0, 1.0], size=spikes.size)
        faulty[window] += fault.magnitude * signs * spikes

    elif fault.kind == "stuck":
        recorded = ~np.isnan(faulty)
        faulty[window & recorded] = _last_recorded_reading(readings, window)

    return faulty


def _last_recorded_reading(readings: np.ndarray, window: np.ndarray) -> float:
    """Return the most recent reading up to the start of the window.

    If nothing has been recorded by then, use the first reading after it.
    """
    start = int(np.argmax(window))
    earlier = readings[: start + 1]
    earlier = earlier[~np.isnan(earlier)]
    if earlier.size > 0:
        return float(earlier[-1])
    later = readings[start:]
    return float(later[~np.isnan(later)][0])
