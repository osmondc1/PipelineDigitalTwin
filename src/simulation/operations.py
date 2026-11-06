"""Generate synthetic pipeline operating data over time.

Each time step gets its own flow, inlet pressure, inlet temperature and pump
boosts. These are built from a daily cycle, slow random wander and scripted
events, then passed through the temperature and pressure models.

All output is SYNTHETIC. It contains the true values at each sensor, before any
measurement error is added.

Assumptions:
    - Quasi-steady flow: every time step is treated as fully settled, so pressure
      responds instantly. Slow changes suit this well, but fast pressure waves
      from events such as a pump trip are not represented.
    - Oil temperature travels with the oil. Each location uses the inlet
      temperature from when that oil entered the line, and the average flow
      during that oil's journey.
    - The first day has no earlier history, so the oil already in the line is
      assumed to have entered at the first inlet temperature.
"""

import math
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config_loader import PipelineCase
from pipeline.hydraulics import pipe_area
from pipeline.pipeline_model import calculate_pipeline
from pipeline.thermal import (
    SegmentTemperature,
    calculate_temperature_profile,
    segment_viscosities,
)
from pipeline.validation import require_name, require_non_negative, require_positive
from sensors.sensor import Sensor, true_value

HOURS_PER_DAY = 24.0
SECONDS_PER_HOUR = 3600.0
NORMAL_LABEL = "normal"


@dataclass(frozen=True)
class Event:
    """A scripted change in operating conditions.

    Attributes:
        label: Name recorded against every affected time step.
        start_day: When the ramp up begins [days from the start].
        end_day: When the ramp down begins [days from the start].
        ramp_up_hours: Time to reach full effect [h].
        ramp_down_hours: Time to return to normal after end_day [h].
        flow_fraction: Flow multiplier at full effect.
        pump: Name of the affected pump, or None.
        boost_fraction: Pump boost multiplier at full effect. Zero means stopped.
    """

    label: str
    start_day: float
    end_day: float
    ramp_up_hours: float
    ramp_down_hours: float
    flow_fraction: float = 1.0
    pump: str | None = None
    boost_fraction: float = 1.0

    def __post_init__(self) -> None:
        """Reject impossible events as soon as one is created."""
        require_name("label", self.label)
        require_non_negative("start_day", self.start_day, "days")
        require_non_negative("ramp_up_hours", self.ramp_up_hours, "h")
        require_non_negative("ramp_down_hours", self.ramp_down_hours, "h")
        require_positive("flow_fraction", self.flow_fraction, "")
        require_non_negative("boost_fraction", self.boost_fraction, "")
        if self.end_day <= self.start_day:
            raise ValueError(f"event {self.label}: end_day must be after start_day")
        if self.ramp_up_hours > self.hold_hours:
            raise ValueError(f"event {self.label}: ramp_up_hours exceeds its duration")

    @property
    def hold_hours(self) -> float:
        """Hours from start_day to end_day."""
        return (self.end_day - self.start_day) * HOURS_PER_DAY

    @property
    def finish_day(self) -> float:
        """Day on which the event has fully ended."""
        return self.end_day + self.ramp_down_hours / HOURS_PER_DAY


@dataclass(frozen=True)
class SimulationPlan:
    """Settings for one run of the operating data generator.

    Attributes:
        seed: Random seed. The same seed gives the same data.
        start_time: Timestamp of the first step, including its time zone.
        duration_days: Length of the run [days].
        time_step: Time between steps [s].
        noise_correlation_hours: How long random wander stays correlated [h].
        flow_daily_amplitude: Daily flow swing, as a fraction of base flow.
        flow_peak_hour: Hour of the day when flow peaks [h].
        flow_noise_std: Flow wander, as a fraction of base flow.
        pump_noise_std: Pump boost wander, as a fraction of base boost.
        inlet_pressure_noise_std: Inlet pressure wander [Pa].
        inlet_temperature_mean: Average inlet temperature [K].
        inlet_temperature_amplitude: Daily inlet temperature swing [K].
        inlet_temperature_peak_hour: Hour of the day of the warmest oil [h].
        events: Scripted events, in time order and not overlapping.
    """

    seed: int
    start_time: datetime
    duration_days: float
    time_step: float
    noise_correlation_hours: float
    flow_daily_amplitude: float
    flow_peak_hour: float
    flow_noise_std: float
    pump_noise_std: float
    inlet_pressure_noise_std: float
    inlet_temperature_mean: float
    inlet_temperature_amplitude: float
    inlet_temperature_peak_hour: float
    events: tuple[Event, ...] = ()

    def __post_init__(self) -> None:
        """Reject impossible plans as soon as one is created."""
        require_positive("duration_days", self.duration_days, "days")
        require_positive("time_step", self.time_step, "s")
        require_positive("noise_correlation_hours", self.noise_correlation_hours, "h")
        require_non_negative("flow_daily_amplitude", self.flow_daily_amplitude, "")
        require_non_negative("flow_noise_std", self.flow_noise_std, "")
        require_non_negative("pump_noise_std", self.pump_noise_std, "")
        require_non_negative(
            "inlet_pressure_noise_std", self.inlet_pressure_noise_std, "Pa"
        )
        require_positive("inlet_temperature_mean", self.inlet_temperature_mean, "K")
        require_non_negative(
            "inlet_temperature_amplitude", self.inlet_temperature_amplitude, "K"
        )
        if self.start_time.tzinfo is None:
            raise ValueError("start_time must include a time zone")
        for earlier, later in zip(self.events, self.events[1:]):
            if later.start_day < earlier.finish_day:
                raise ValueError(f"events {earlier.label} and {later.label} overlap")

    @property
    def steps(self) -> int:
        """Number of time steps in the run."""
        return round(self.duration_days * HOURS_PER_DAY * SECONDS_PER_HOUR / self.time_step)


def load_plan(path: Path) -> SimulationPlan:
    """Read a simulation plan from a TOML file.

    Args:
        path: Location of the TOML file.

    Returns:
        The plan.

    Raises:
        ValueError: If a setting is missing or impossible.
        FileNotFoundError: If the file does not exist.
    """
    with path.open("rb") as file:
        data = tomllib.load(file)

    try:
        return SimulationPlan(
            seed=data["seed"],
            start_time=data["start_time"],
            duration_days=data["duration_days"],
            time_step=data["time_step_s"],
            noise_correlation_hours=data["noise_correlation_hours"],
            flow_daily_amplitude=data["flow"]["daily_amplitude_fraction"],
            flow_peak_hour=data["flow"]["peak_hour"],
            flow_noise_std=data["flow"]["noise_std_fraction"],
            pump_noise_std=data["pumps"]["noise_std_fraction"],
            inlet_pressure_noise_std=data["inlet_pressure"]["noise_std_pa"],
            inlet_temperature_mean=data["inlet_temperature"]["mean_k"],
            inlet_temperature_amplitude=data["inlet_temperature"]["daily_amplitude_k"],
            inlet_temperature_peak_hour=data["inlet_temperature"]["peak_hour"],
            events=tuple(Event(**row) for row in data.get("events", [])),
        )
    except KeyError as error:
        raise ValueError(f"{path} is missing the setting {error}") from error


def correlated_noise(
    rng: np.random.Generator, steps: int, std: float, correlation_steps: float
) -> np.ndarray:
    """Return slowly wandering random values with a mean of zero.

    Each value is mostly the previous value plus a small random change:

        x[i] = a * x[i-1] + sqrt(1 - a^2) * std * random
        a = exp(-1 / correlation_steps)

    Args:
        rng: Random number generator.
        steps: Number of values.
        std: Standard deviation of the values.
        correlation_steps: Number of steps over which values stay related.

    Returns:
        The noise values.
    """
    memory = math.exp(-1 / correlation_steps)
    shocks = rng.normal(0.0, std * math.sqrt(1 - memory**2), steps)
    noise = np.empty(steps)
    noise[0] = rng.normal(0.0, std)
    for step in range(1, steps):
        noise[step] = memory * noise[step - 1] + shocks[step]
    return noise


def daily_cycle(hours: np.ndarray, peak_hour: float) -> np.ndarray:
    """Return a wave between -1 and 1 that peaks at peak_hour each day."""
    return np.cos(2 * np.pi * (hours - peak_hour) / HOURS_PER_DAY)


def event_strength(event: Event, hours: np.ndarray) -> np.ndarray:
    """Return how strongly an event applies, from 0 (none) to 1 (full).

    Args:
        event: The event.
        hours: Time of each step [hours from the start].

    Returns:
        The strength at each step.
    """
    start = event.start_day * HOURS_PER_DAY
    end = event.end_day * HOURS_PER_DAY

    if event.ramp_up_hours > 0:
        rising = np.clip((hours - start) / event.ramp_up_hours, 0.0, 1.0)
    else:
        rising = (hours >= start).astype(float)

    if event.ramp_down_hours > 0:
        falling = np.clip((end + event.ramp_down_hours - hours) / event.ramp_down_hours, 0.0, 1.0)
    else:
        falling = (hours < end).astype(float)

    return np.minimum(rising, falling)


def departure_steps(
    flow_rates: np.ndarray, time_step: float, upstream_volume: float
) -> np.ndarray:
    """Return the step at which the oil now at a location entered the line.

    Oil at a location entered when exactly upstream_volume has since been pumped
    in. Using volume rather than a fixed delay means it works when flow changes.

    Args:
        flow_rates: Flow rate at each step [m^3/s].
        time_step: Time between steps [s].
        upstream_volume: Volume of pipe between the inlet and the location [m^3].

    Returns:
        The entry step for each time step. Oil that entered before the run
        started is given step 0.
    """
    pumped_volume = np.cumsum(flow_rates * time_step)
    return np.searchsorted(pumped_volume, pumped_volume - upstream_volume)


def transit_flow_rates(
    flow_rates: np.ndarray, time_step: float, upstream_volume: float
) -> np.ndarray:
    """Return the average flow during the journey of the oil now at a location.

    Cooling depends on how long oil has spent in the pipe. Using the journey's
    average flow means that a sudden flow change affects outlet temperature
    gradually, as the oil that experienced it arrives.

    Args:
        flow_rates: Flow rate at each step [m^3/s].
        time_step: Time between steps [s].
        upstream_volume: Volume of pipe between the inlet and the location [m^3].

    Returns:
        The average flow at each step [m^3/s]. Until the line has been filled
        there is no complete journey, so the current flow is used.
    """
    times = np.arange(len(flow_rates)) * time_step
    pumped_volume = np.cumsum(flow_rates * time_step)
    entry_volume = pumped_volume - upstream_volume

    entry_times = np.interp(entry_volume, pumped_volume, times)
    journey_times = times - entry_times
    has_full_journey = (entry_volume >= pumped_volume[0]) & (journey_times > 0)

    transit_flows = flow_rates.copy()
    transit_flows[has_full_journey] = upstream_volume / journey_times[has_full_journey]
    return transit_flows


def _upstream_volumes(case: PipelineCase) -> list[float]:
    """Return the pipe volume upstream of each segment boundary [m^3]."""
    volumes = [0.0]
    for segment in case.segments:
        volume = pipe_area(segment.internal_diameter) * segment.length
        volumes.append(volumes[-1] + volume)
    return volumes


def _segment_temperatures(
    case: PipelineCase,
    boundary_flow_rates: list[float],
    boundary_inlet_temperatures: list[float],
) -> tuple[SegmentTemperature, ...]:
    """Return one step's segment temperatures, allowing for travel time.

    Each boundary is calculated by cooling the oil that is there now, starting
    from the inlet temperature it entered at, along all the pipe upstream of it,
    at the average flow of its journey.
    """
    boundaries = [boundary_inlet_temperatures[0]]
    for index in range(1, len(case.segments) + 1):
        profile = calculate_temperature_profile(
            case.segments[:index],
            case.thermal,
            case.fluid.density,
            boundary_flow_rates[index],
            boundary_inlet_temperatures[index],
        )
        boundaries.append(profile[-1].outlet_temperature)

    return tuple(
        SegmentTemperature(segment.name, boundaries[index], boundaries[index + 1])
        for index, segment in enumerate(case.segments)
    )


def _build_inputs(
    case: PipelineCase, plan: SimulationPlan, rng: np.random.Generator
) -> dict:
    """Return the input values for every time step."""
    steps = plan.steps
    hours = np.arange(steps) * plan.time_step / SECONDS_PER_HOUR
    correlation_steps = (
        plan.noise_correlation_hours * SECONDS_PER_HOUR / plan.time_step
    )

    flow_factor = 1 + plan.flow_daily_amplitude * daily_cycle(hours, plan.flow_peak_hour)
    flow_factor += correlated_noise(rng, steps, plan.flow_noise_std, correlation_steps)

    boost_factors = {
        pump.name: 1 + correlated_noise(
            rng, steps, plan.pump_noise_std, correlation_steps
        )
        for pump in case.pumps
    }

    labels = np.full(steps, NORMAL_LABEL, dtype=object)
    for event in plan.events:
        strength = event_strength(event, hours)
        flow_factor *= 1 + (event.flow_fraction - 1) * strength
        if event.pump is not None:
            if event.pump not in boost_factors:
                raise ValueError(f"event {event.label} refers to unknown pump {event.pump}")
            boost_factors[event.pump] *= 1 + (event.boost_fraction - 1) * strength
        labels[strength > 0] = event.label

    inlet_pressure = case.inlet_pressure + correlated_noise(
        rng, steps, plan.inlet_pressure_noise_std, correlation_steps
    )
    inlet_temperature = (
        plan.inlet_temperature_mean
        + plan.inlet_temperature_amplitude
        * daily_cycle(hours, plan.inlet_temperature_peak_hour)
    )

    return {
        "flow_rate": case.flow_rate * flow_factor,
        "boost_factors": boost_factors,
        "labels": labels,
        "inlet_pressure": inlet_pressure,
        "inlet_temperature": inlet_temperature,
    }


def generate_operating_data(
    case: PipelineCase, sensors: tuple[Sensor, ...], plan: SimulationPlan
) -> pd.DataFrame:
    """Generate the true operating conditions for every time step.

    Args:
        case: The pipeline. It must include thermal properties.
        sensors: The sensor network.
        plan: The simulation plan.

    Returns:
        One row per time step. Columns: timestamp, flow_rate_m3_s,
        inlet_pressure_pa, inlet_temperature_k, one boost_<pump>_pa per pump,
        event, low_pressure, and one column of true values per sensor.

    Raises:
        ValueError: If the case has no thermal properties or an event refers to
            an unknown pump.
    """
    if case.thermal is None:
        raise ValueError("the case needs thermal properties to generate data")

    rng = np.random.default_rng(plan.seed)
    inputs = _build_inputs(case, plan, rng)
    flows = inputs["flow_rate"]
    inlet_temperatures = inputs["inlet_temperature"]

    volumes = _upstream_volumes(case)
    entry_steps = [departure_steps(flows, plan.time_step, v) for v in volumes]
    transit_flows = [transit_flow_rates(flows, plan.time_step, v) for v in volumes]
    boosts = {
        pump.name: pump.pressure_boost * inputs["boost_factors"][pump.name]
        for pump in case.pumps
    }

    low_pressure = np.zeros(plan.steps, dtype=bool)
    readings = {sensor.sensor_id: np.empty(plan.steps) for sensor in sensors}

    for step in range(plan.steps):
        pumps = tuple(
            replace(pump, pressure_boost=float(boosts[pump.name][step]))
            for pump in case.pumps
        )
        temperatures = _segment_temperatures(
            case,
            [float(flow[step]) for flow in transit_flows],
            [float(inlet_temperatures[entry[step]]) for entry in entry_steps],
        )
        viscosities = segment_viscosities(
            temperatures, case.fluid.dynamic_viscosity, case.thermal
        )
        result = calculate_pipeline(
            case.fluid,
            float(flows[step]),
            float(inputs["inlet_pressure"][step]),
            case.segments,
            pumps,
            viscosities,
        )
        low_pressure[step] = result.has_low_pressure
        for sensor in sensors:
            readings[sensor.sensor_id][step] = true_value(
                sensor, case.segments, result, temperatures
            )

    columns = {
        "timestamp": pd.date_range(
            plan.start_time, periods=plan.steps, freq=pd.Timedelta(seconds=plan.time_step)
        ),
        "flow_rate_m3_s": flows,
        "inlet_pressure_pa": inputs["inlet_pressure"],
        "inlet_temperature_k": inlet_temperatures,
    }
    columns.update({f"boost_{name}_pa": boost for name, boost in boosts.items()})
    columns.update({"event": inputs["labels"], "low_pressure": low_pressure})
    columns.update(readings)
    return pd.DataFrame(columns)
