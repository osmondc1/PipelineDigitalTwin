"""Tests for the simulated sensors."""

import math
from pathlib import Path

import numpy as np
import pytest

from pipeline.config_loader import load_case
from pipeline.pipeline_model import calculate_pipeline
from pipeline.thermal import ThermalProperties, calculate_temperature_profile
from sensors.sensor import (
    Sensor,
    draw_biases,
    load_sensors,
    take_reading,
    true_value,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CASE_PATH = PROJECT_ROOT / "config" / "base_case.toml"
SENSORS_PATH = PROJECT_ROOT / "config" / "sensors.toml"
SEED = 12345


def make_sensor(**changes: float | str | bool) -> Sensor:
    """Build a valid pressure sensor, replacing any values a test changes."""
    values = {
        "sensor_id": "PT-001",
        "quantity": "pressure",
        "location": "Inlet",
        "location_km": 0.0,
        "sampling_interval": 60.0,
        "noise_std": 0.03,
        "bias_std": 0.04,
        "missing_probability": 0.002,
    }
    values.update(changes)
    return Sensor(**values)


def many_readings(sensor: Sensor, count: int = 100_000) -> list[float | None]:
    """Take many readings of a true value of 50 with a bias of 1."""
    rng = np.random.default_rng(SEED)
    return [take_reading(sensor, 50.0, 1.0, rng) for _ in range(count)]


@pytest.fixture
def base_case_values() -> dict[str, float]:
    """True values at every sensor for the base case, with 40 °C oil entering."""
    case = load_case(BASE_CASE_PATH)
    thermal = ThermalProperties(2000.0, 2.0, 288.15, 303.15, 0.03)
    result = calculate_pipeline(
        case.fluid, case.flow_rate, case.inlet_pressure, case.segments, case.pumps
    )
    temperatures = calculate_temperature_profile(
        case.segments, thermal, case.fluid.density, case.flow_rate, 313.15
    )
    return {
        sensor.sensor_id: true_value(sensor, case.segments, result, temperatures)
        for sensor in load_sensors(SENSORS_PATH)
    }


# --- Sensor definition ---


@pytest.mark.parametrize(
    "quantity, unit",
    [("pressure", "barg"), ("flow", "m3/h"), ("temperature", "degC")],
)
def test_each_quantity_has_its_unit(quantity: str, unit: str) -> None:
    assert make_sensor(quantity=quantity).unit == unit


def test_uncertainty_combines_noise_and_bias() -> None:
    # Hand check: sqrt(0.03^2 + 0.04^2) = 0.05
    assert make_sensor().uncertainty == pytest.approx(0.05)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("sensor_id", ""),
        ("location", "   "),
        ("quantity", "humidity"),
        ("location_km", -1.0),
        ("sampling_interval", 0.0),
        ("noise_std", -0.1),
        ("noise_std", math.nan),
        ("bias_std", -0.1),
        ("missing_probability", -0.1),
        ("missing_probability", 1.1),
    ],
)
def test_impossible_sensor_is_rejected(field: str, bad_value: float | str) -> None:
    with pytest.raises(ValueError, match=field):
        make_sensor(**{field: bad_value})


# --- Loading the network ---


def test_network_has_the_eight_planned_sensors() -> None:
    sensor_ids = [sensor.sensor_id for sensor in load_sensors(SENSORS_PATH)]

    assert sensor_ids == [
        "PT-001", "PT-002", "PT-003",
        "FT-001", "FT-002",
        "TT-001", "TT-002", "TT-003",
    ]


def test_pressure_sensors_are_at_the_agreed_points() -> None:
    pt_001, pt_002, pt_003 = load_sensors(SENSORS_PATH)[:3]

    assert (pt_001.location_km, pt_001.upstream_of_pump) == (0.0, False)
    assert (pt_002.location_km, pt_002.upstream_of_pump) == (55.0, True)
    assert pt_003.location_km == 100.0


def test_every_sensor_samples_once_a_minute() -> None:
    assert all(s.sampling_interval == 60.0 for s in load_sensors(SENSORS_PATH))


def write_sensors(tmp_path: Path, text: str) -> Path:
    """Write text to a temporary TOML file and return its path."""
    path = tmp_path / "sensors.toml"
    path.write_text(text)
    return path


ONE_SENSOR = """
[[sensors]]
sensor_id = "PT-001"
quantity = "pressure"
location = "Inlet"
location_km = 0.0
sampling_interval_s = 60.0
noise_std = 0.03
bias_std = 0.04
missing_probability = 0.002
"""


def test_repeated_sensor_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        load_sensors(write_sensors(tmp_path, ONE_SENSOR + ONE_SENSOR))


def test_missing_setting_is_named_in_the_error(tmp_path: Path) -> None:
    lines = [line for line in ONE_SENSOR.splitlines() if "noise_std" not in line]

    with pytest.raises(ValueError, match="noise_std"):
        load_sensors(write_sensors(tmp_path, "\n".join(lines)))


def test_upstream_of_pump_defaults_to_false(tmp_path: Path) -> None:
    sensors = load_sensors(write_sensors(tmp_path, ONE_SENSOR))

    assert sensors[0].upstream_of_pump is False


# --- True values from the model ---


def test_pt_001_sees_ps1_discharge(base_case_values: dict[str, float]) -> None:
    # Hand check: (38 bar abs - 1.01325) = 36.99 barg, after the 35 bar boost.
    assert base_case_values["PT-001"] == pytest.approx(36.99, abs=0.01)


def test_pt_002_sees_arrival_pressure_before_ps2(
    base_case_values: dict[str, float],
) -> None:
    # Matches the 15.41 barg "Arrive" value for S3 in the Week 2 report.
    assert base_case_values["PT-002"] == pytest.approx(15.41, abs=0.01)


def test_pt_003_sees_outlet_pressure(base_case_values: dict[str, float]) -> None:
    assert base_case_values["PT-003"] == pytest.approx(20.86, abs=0.01)


@pytest.mark.parametrize("sensor_id", ["FT-001", "FT-002"])
def test_flow_sensors_read_base_flow_in_m3_per_hour(
    sensor_id: str, base_case_values: dict[str, float]
) -> None:
    # Hand check: 0.25 m^3/s * 3600 s/h = 900 m^3/h
    assert base_case_values[sensor_id] == pytest.approx(900.0)


def test_temperature_sensors_follow_the_cooling_oil(
    base_case_values: dict[str, float],
) -> None:
    # Hand check: 40 °C in, 26.94 °C out, as in test_thermal.py.
    assert base_case_values["TT-001"] == pytest.approx(40.0)
    assert base_case_values["TT-003"] == pytest.approx(26.94, abs=0.01)
    assert 26.94 < base_case_values["TT-002"] < 40.0


def test_sensor_between_segment_joins_is_rejected() -> None:
    case = load_case(BASE_CASE_PATH)
    result = calculate_pipeline(
        case.fluid, case.flow_rate, case.inlet_pressure, case.segments, case.pumps
    )

    with pytest.raises(ValueError, match="no segment boundary"):
        true_value(make_sensor(location_km=42.0), case.segments, result, ())


# --- Readings ---


def test_readings_average_to_true_value_plus_bias() -> None:
    readings = [r for r in many_readings(make_sensor()) if r is not None]

    assert np.mean(readings) == pytest.approx(51.0, abs=0.001)


def test_reading_scatter_matches_noise_setting() -> None:
    readings = [r for r in many_readings(make_sensor()) if r is not None]

    assert np.std(readings) == pytest.approx(0.03, rel=0.02)


def test_missing_rate_matches_its_setting() -> None:
    readings = many_readings(make_sensor(missing_probability=0.1))

    assert sum(r is None for r in readings) / len(readings) == pytest.approx(
        0.1, rel=0.03
    )


def test_perfect_sensor_reads_true_value_plus_bias() -> None:
    sensor = make_sensor(noise_std=0.0, missing_probability=0.0)

    assert all(r == 51.0 for r in many_readings(sensor, count=100))


def test_zero_missing_probability_never_drops_a_reading() -> None:
    readings = many_readings(make_sensor(missing_probability=0.0), count=10_000)

    assert None not in readings


def test_full_missing_probability_drops_every_reading() -> None:
    readings = many_readings(make_sensor(missing_probability=1.0), count=100)

    assert all(r is None for r in readings)


def test_same_seed_gives_the_same_readings() -> None:
    assert many_readings(make_sensor(), 100) == many_readings(make_sensor(), 100)


def test_different_seed_gives_different_readings() -> None:
    rng = np.random.default_rng(SEED + 1)
    other = [take_reading(make_sensor(), 50.0, 1.0, rng) for _ in range(100)]

    assert other != many_readings(make_sensor(), 100)


def test_missing_rate_does_not_change_other_readings() -> None:
    none_missing = many_readings(make_sensor(missing_probability=0.0), 1000)
    some_missing = many_readings(make_sensor(missing_probability=0.1), 1000)

    for complete, partial in zip(none_missing, some_missing):
        assert partial is None or partial == complete


# --- Biases ---


def test_bias_stays_fixed_for_each_sensor() -> None:
    sensor = make_sensor(noise_std=0.0, missing_probability=0.0)
    rng = np.random.default_rng(SEED)
    bias = draw_biases([sensor], rng)["PT-001"]

    readings = {take_reading(sensor, 50.0, bias, rng) for _ in range(100)}

    assert readings == {50.0 + bias}


def test_sensors_get_different_biases() -> None:
    sensors = [make_sensor(sensor_id=f"PT-{i}") for i in range(3)]

    biases = draw_biases(sensors, np.random.default_rng(SEED))

    assert len(set(biases.values())) == 3


def test_zero_bias_setting_gives_zero_bias() -> None:
    biases = draw_biases([make_sensor(bias_std=0.0)], np.random.default_rng(SEED))

    assert biases["PT-001"] == 0.0


def test_bias_spread_matches_its_setting() -> None:
    sensors = [make_sensor(sensor_id=str(i)) for i in range(50_000)]

    biases = draw_biases(sensors, np.random.default_rng(SEED))

    assert np.std(list(biases.values())) == pytest.approx(0.04, rel=0.02)
