"""Tests for the SQLite storage layer."""

import runpy
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config_loader import load_case
from sensors.faults import SensorFault, load_faults
from sensors.readings import SensorData, generate_sensor_readings
from sensors.sensor import Sensor, load_sensors
from sensors.storage import (
    list_sensors,
    load_ground_truth,
    load_operating_states,
    load_readings,
    save_simulation,
    _read_query,
)
from simulation.operations import generate_operating_data, load_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FOLDER = PROJECT_ROOT / "config"
SEED = 12345
STEPS = 120  # Two hours, one row per minute.


def make_sensor(sensor_id: str = "PT-001", **changes: float) -> Sensor:
    """Build a sensor that has noise, a bias and missing readings."""
    values = {
        "sensor_id": sensor_id,
        "quantity": "pressure",
        "location": "Inlet",
        "location_km": 0.0,
        "sampling_interval": 60.0,
        "noise_std": 0.5,
        "bias_std": 1.0,
        "missing_probability": 0.2,
    }
    values.update(changes)
    return Sensor(**values)


def make_truth(*sensor_ids: str) -> pd.DataFrame:
    """Build two hours of true data, with every sensor truly reading 50."""
    timestamps = pd.date_range("2026-01-01", periods=STEPS, freq="60s", tz="UTC")
    data = {
        "timestamp": timestamps,
        "event": ["normal"] * 60 + ["ps2_trip"] * 60,
        "low_pressure": [False] * 119 + [True],
        "flow_rate_m3_s": 0.25,
        "inlet_pressure_pa": 300_000.0,
        "inlet_temperature_k": 313.15,
    }
    return pd.DataFrame(data | {sensor_id: 50.0 for sensor_id in sensor_ids})


def simulate(*sensor_ids: str) -> tuple:
    """Return sensors, truth and sensor data for a small simulation."""
    sensors = tuple(make_sensor(sensor_id) for sensor_id in sensor_ids)
    truth = make_truth(*sensor_ids)
    fault = SensorFault("PT-001", "stuck", 0.0, 1 / 48)  # The first 30 minutes.
    return sensors, truth, generate_sensor_readings(truth, sensors, (fault,), SEED)


@pytest.fixture
def database(tmp_path: Path) -> tuple[Path, SensorData]:
    """A small saved simulation with two sensors."""
    path = tmp_path / "test.db"
    sensors, truth, sensor_data = simulate("PT-001", "PT-002")
    save_simulation(path, sensors, truth, sensor_data)
    return path, sensor_data


def count_rows(path: Path, table: str) -> int:
    """Return the number of rows in a table."""
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def column_names(path: Path, table: str) -> set[str]:
    """Return a table's column names."""
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


# --- Saving ---


def test_saving_creates_missing_folders(tmp_path: Path) -> None:
    path = tmp_path / "new" / "folder" / "test.db"
    sensors, truth, sensor_data = simulate("PT-001")

    save_simulation(path, sensors, truth, sensor_data)

    assert path.exists()


def test_each_table_has_the_expected_number_of_rows(database) -> None:
    # Hand check: 2 sensors * 120 minutes = 240 rows per sensor table.
    path, _ = database

    assert count_rows(path, "sensors") == 2
    assert count_rows(path, "sensor_readings") == 240
    assert count_rows(path, "ground_truth") == 240
    assert count_rows(path, "sensor_biases") == 2
    assert count_rows(path, "operating_states") == 120


def test_saving_again_replaces_the_old_data(database) -> None:
    path, _ = database
    sensors, truth, sensor_data = simulate("PT-001", "PT-002")

    save_simulation(path, sensors, truth, sensor_data)

    assert count_rows(path, "sensor_readings") == 240


def test_failed_save_keeps_the_previous_data(database) -> None:
    path, _ = database
    sensors, truth, sensor_data = simulate("PT-001", "PT-002")
    broken_truth = truth.drop(columns="flow_rate_m3_s")

    with pytest.raises(KeyError):
        save_simulation(path, sensors, broken_truth, sensor_data)

    assert count_rows(path, "sensor_readings") == 240
    assert count_rows(path, "operating_states") == 120


def test_save_with_an_undefined_sensor_is_refused(database) -> None:
    # The save switches on foreign keys, so bad data cannot be stored.
    path, _ = database
    sensors, truth, sensor_data = simulate("PT-001", "PT-002")
    bad_data = SensorData(
        sensor_data.readings, sensor_data.fault_labels, {"PT-999": 1.0}
    )

    with pytest.raises(sqlite3.IntegrityError):
        save_simulation(path, sensors, truth, bad_data)

    assert count_rows(path, "sensor_biases") == 2


def test_readings_table_contains_no_answers(database) -> None:
    # Answers in this table would leak into anything trained on the readings.
    path, _ = database

    assert column_names(path, "sensor_readings") == {
        "sensor_id",
        "timestamp",
        "value",
        "unit",
    }


def test_reading_for_an_unknown_sensor_is_refused(database) -> None:
    # The database itself protects against readings from undefined sensors.
    path, _ = database
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO sensor_readings VALUES ('PT-999', 'x', 1.0, 'barg')"
            )


def test_same_sensor_and_time_cannot_be_stored_twice(database) -> None:
    path, _ = database
    with closing(sqlite3.connect(path)) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO sensor_readings VALUES "
                "('PT-001', '2026-01-01T00:00:00Z', 1.0, 'barg')"
            )


# --- Reading back ---


def test_values_come_back_exactly_as_saved(database) -> None:
    path, sensor_data = database

    stored = load_readings(path, "PT-002")

    np.testing.assert_array_equal(
        stored["value"].to_numpy(), sensor_data.readings["PT-002"].to_numpy()
    )


def test_timestamps_come_back_as_utc_datetimes(database) -> None:
    path, sensor_data = database

    stored = load_readings(path, "PT-001")

    pd.testing.assert_series_equal(
        stored["timestamp"],
        sensor_data.readings["timestamp"],
        check_dtype=False,
        check_freq=False,
    )


def test_missing_readings_are_stored_as_empty_values(database) -> None:
    # A gap stays visible as a row. It is not silently left out.
    path, sensor_data = database

    stored = load_readings(path, "PT-002")

    expected = sensor_data.readings["PT-002"].isna().sum()
    assert expected > 0
    assert stored["value"].isna().sum() == expected
    assert len(stored) == STEPS


def test_units_are_stored_with_readings(database) -> None:
    path, _ = database

    assert set(load_readings(path, "PT-001")["unit"]) == {"barg"}


def test_readings_only_return_the_chosen_sensor(database) -> None:
    path, sensor_data = database

    stored = load_readings(path, "PT-001")

    np.testing.assert_array_equal(
        stored["value"].to_numpy(), sensor_data.readings["PT-001"].to_numpy()
    )


def test_list_sensors_returns_the_network(database) -> None:
    path, _ = database

    sensors = list_sensors(path)

    assert sensors["sensor_id"].tolist() == ["PT-001", "PT-002"]
    assert sensors["uncertainty"].iloc[0] == pytest.approx(np.sqrt(0.5**2 + 1.0**2))


def test_ground_truth_holds_true_values_and_fault_labels(database) -> None:
    path, _ = database

    stored = load_ground_truth(path, "PT-001")

    assert (stored["true_value"] == 50.0).all()
    assert (stored["fault"].iloc[:30] == "stuck").all()
    assert (stored["fault"].iloc[30:] == "none").all()


def test_operating_states_hold_events_and_flags(database) -> None:
    path, _ = database

    states = load_operating_states(path)

    assert states["event"].iloc[0] == "normal"
    assert states["event"].iloc[-1] == "ps2_trip"
    assert states["low_pressure"].sum() == 1
    assert states["low_pressure"].dtype == bool


# --- Time ranges ---


def test_time_range_includes_start_and_excludes_end(database) -> None:
    path, _ = database

    stored = load_readings(
        path, "PT-001", "2026-01-01 00:10", "2026-01-01 00:20"
    )

    assert len(stored) == 10
    assert stored["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:10", tz="UTC")
    assert stored["timestamp"].iloc[-1] == pd.Timestamp("2026-01-01 00:19", tz="UTC")


def test_start_alone_reads_to_the_end(database) -> None:
    path, _ = database

    assert len(load_readings(path, "PT-001", start="2026-01-01 01:50")) == 10


def test_end_alone_reads_from_the_beginning(database) -> None:
    path, _ = database

    assert len(load_readings(path, "PT-001", end="2026-01-01 00:05")) == 5


def test_times_in_other_zones_are_converted_to_utc(database) -> None:
    # 02:10 at UTC+2 is 00:10 UTC.
    path, _ = database

    stored = load_readings(path, "PT-001", start="2026-01-01 02:10+02:00")

    assert stored["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:10", tz="UTC")


def test_range_with_no_data_returns_an_empty_table(database) -> None:
    path, _ = database

    assert load_readings(path, "PT-001", start="2030-01-01").empty


def test_operating_states_accept_a_time_range(database) -> None:
    path, _ = database

    states = load_operating_states(path, "2026-01-01 01:00", "2026-01-01 01:30")

    assert len(states) == 30
    assert set(states["event"]) == {"ps2_trip"}


def test_ground_truth_accepts_a_time_range(database) -> None:
    path, _ = database

    assert len(load_ground_truth(path, "PT-001", end="2026-01-01 00:30")) == 30


# --- Invalid requests ---


def test_unknown_sensor_is_rejected(database) -> None:
    path, _ = database

    with pytest.raises(ValueError, match="unknown sensor"):
        load_readings(path, "PT-999")


def test_text_that_looks_like_sql_is_treated_as_a_name(database) -> None:
    # The ? placeholder passes this as plain text, so nothing is executed.
    path, _ = database

    with pytest.raises(ValueError, match="unknown sensor"):
        load_readings(path, "PT-001' OR '1'='1")

    assert count_rows(path, "sensor_readings") == 240


def test_missing_database_is_reported_and_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError):
        load_readings(path, "PT-001")

    assert not path.exists()


def test_reading_cannot_change_the_database(database) -> None:
    path, _ = database

    with pytest.raises(pd.errors.DatabaseError, match="readonly"):
        _read_query(path, "DELETE FROM sensors", [])

    assert count_rows(path, "sensors") == 2


# --- The real 30-day plan ---


@pytest.fixture(scope="module")
def full_database(tmp_path_factory) -> tuple[Path, SensorData]:
    """The real plan, saved to a database."""
    path = tmp_path_factory.mktemp("full") / "pipeline.db"
    case = load_case(CONFIG_FOLDER / "base_case.toml")
    sensors = load_sensors(CONFIG_FOLDER / "sensors.toml")
    plan = load_plan(CONFIG_FOLDER / "simulation.toml")
    faults = load_faults(CONFIG_FOLDER / "simulation.toml")
    truth = generate_operating_data(case, sensors, plan)
    sensor_data = generate_sensor_readings(truth, sensors, faults, plan.seed)
    save_simulation(path, sensors, truth, sensor_data)
    return path, sensor_data


def test_full_database_holds_every_reading(full_database) -> None:
    # Hand check: 43,200 minutes * 8 sensors = 345,600 rows, gaps included.
    path, _ = full_database

    assert count_rows(path, "sensor_readings") == 345_600
    assert count_rows(path, "ground_truth") == 345_600
    assert count_rows(path, "operating_states") == 43_200


def test_full_database_recorded_count_matches_the_simulation(full_database) -> None:
    path, sensor_data = full_database

    with closing(sqlite3.connect(path)) as connection:
        recorded = connection.execute(
            "SELECT COUNT(value) FROM sensor_readings"
        ).fetchone()[0]

    assert recorded == sensor_data.readings.drop(columns="timestamp").count().sum()


def test_stored_dropout_appears_as_gaps(full_database) -> None:
    path, _ = full_database

    stored = load_readings(path, "FT-001", "2026-01-27 00:00", "2026-01-27 12:00")

    assert len(stored) == 720
    assert stored["value"].isna().all()


def test_ground_truth_shows_the_dropout_that_readings_cannot(full_database) -> None:
    path, _ = full_database

    truth = load_ground_truth(path, "FT-001", "2026-01-27 00:00", "2026-01-27 12:00")

    assert (truth["fault"] == "dropout").all()
    assert truth["true_value"].notna().all()


# --- The script ---


def test_generate_data_script_runs_and_saves(tmp_path: Path, capsys) -> None:
    # Runs the real script, but saves to a temporary database.
    path = tmp_path / "pipeline.db"
    script = runpy.run_path(str(PROJECT_ROOT / "scripts" / "generate_data.py"))

    script["main"](path)

    assert count_rows(path, "sensor_readings") == 345_600
    assert "Saved to" in capsys.readouterr().out
