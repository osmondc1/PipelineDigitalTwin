"""Store and retrieve simulated data in a SQLite database.

All stored data is SYNTHETIC. The tables are split by who would have the data:

    Monitoring data, which a real control room would receive:
        sensors          One row per sensor.
        sensor_readings  One row per sensor per minute. value is NULL if no
                         reading was recorded.

    Simulation answers, which a real control room would NOT have. Keep these
    away from anything that learns from the readings:
        ground_truth     True value and active fault per sensor per minute.
        sensor_biases    Each sensor's true calibration error.
        operating_states True inputs and scripted event per minute.

Timestamps are stored as UTC text such as "2026-01-01T00:00:00Z". This format
sorts correctly as plain text, so SQL can compare it directly.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from sensors.readings import SensorData
from sensors.sensor import Sensor

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

SCHEMA = (
    "DROP TABLE IF EXISTS sensor_readings",
    "DROP TABLE IF EXISTS ground_truth",
    "DROP TABLE IF EXISTS sensor_biases",
    "DROP TABLE IF EXISTS operating_states",
    "DROP TABLE IF EXISTS sensors",
    """
    CREATE TABLE sensors (
        sensor_id TEXT PRIMARY KEY,
        quantity TEXT NOT NULL,
        location TEXT NOT NULL,
        location_km REAL NOT NULL,
        unit TEXT NOT NULL,
        sampling_interval_s REAL NOT NULL,
        uncertainty REAL NOT NULL
    )
    """,
    """
    CREATE TABLE sensor_readings (
        sensor_id TEXT NOT NULL REFERENCES sensors (sensor_id),
        timestamp TEXT NOT NULL,
        value REAL,
        unit TEXT NOT NULL,
        PRIMARY KEY (sensor_id, timestamp)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE ground_truth (
        sensor_id TEXT NOT NULL REFERENCES sensors (sensor_id),
        timestamp TEXT NOT NULL,
        true_value REAL NOT NULL,
        fault TEXT NOT NULL,
        PRIMARY KEY (sensor_id, timestamp)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE sensor_biases (
        sensor_id TEXT PRIMARY KEY REFERENCES sensors (sensor_id),
        bias REAL NOT NULL
    )
    """,
    """
    CREATE TABLE operating_states (
        timestamp TEXT PRIMARY KEY,
        event TEXT NOT NULL,
        low_pressure INTEGER NOT NULL,
        flow_rate_m3_s REAL NOT NULL,
        inlet_pressure_pa REAL NOT NULL,
        inlet_temperature_k REAL NOT NULL
    ) WITHOUT ROWID
    """,
)


def _timestamp_text(timestamps: pd.Series) -> list[str]:
    """Convert timestamps to the UTC text stored in the database."""
    return timestamps.dt.tz_convert("UTC").dt.strftime(TIMESTAMP_FORMAT).tolist()


def _reading_rows(
    sensors: tuple[Sensor, ...], sensor_data: SensorData
) -> Iterator[tuple]:
    """Yield one sensor_readings row per sensor per minute."""
    timestamps = _timestamp_text(sensor_data.readings["timestamp"])
    for sensor in sensors:
        values = sensor_data.readings[sensor.sensor_id]
        for timestamp, value in zip(timestamps, values.tolist()):
            # sqlite3 stores a NaN value as NULL, so gaps stay as empty values.
            yield sensor.sensor_id, timestamp, value, sensor.unit


def _truth_rows(
    sensors: tuple[Sensor, ...], truth: pd.DataFrame, sensor_data: SensorData
) -> Iterator[tuple]:
    """Yield one ground_truth row per sensor per minute."""
    timestamps = _timestamp_text(truth["timestamp"])
    for sensor in sensors:
        true_values = truth[sensor.sensor_id].tolist()
        faults = sensor_data.fault_labels[sensor.sensor_id].tolist()
        for row in zip(timestamps, true_values, faults):
            yield (sensor.sensor_id, *row)


def save_simulation(
    path: Path,
    sensors: tuple[Sensor, ...],
    truth: pd.DataFrame,
    sensor_data: SensorData,
) -> None:
    """Save a simulation, replacing anything already in the database.

    The whole save is one transaction. If anything fails, the database keeps
    its previous contents.

    Args:
        path: Location of the database file. It is created if needed.
        sensors: The sensor network.
        truth: True operating data.
        sensor_data: The simulated readings, fault labels and biases.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # closing() releases the file afterwards, which "with connection" does not.
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        # "with connection" commits at the end, or undoes everything on an error.
        with connection:
            connection.execute("BEGIN")
            for statement in SCHEMA:
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO sensors VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        sensor.sensor_id,
                        sensor.quantity,
                        sensor.location,
                        sensor.location_km,
                        sensor.unit,
                        sensor.sampling_interval,
                        sensor.uncertainty,
                    )
                    for sensor in sensors
                ),
            )
            connection.executemany(
                "INSERT INTO sensor_readings VALUES (?, ?, ?, ?)",
                _reading_rows(sensors, sensor_data),
            )
            connection.executemany(
                "INSERT INTO ground_truth VALUES (?, ?, ?, ?)",
                _truth_rows(sensors, truth, sensor_data),
            )
            connection.executemany(
                "INSERT INTO sensor_biases VALUES (?, ?)",
                sensor_data.biases.items(),
            )
            connection.executemany(
                "INSERT INTO operating_states VALUES (?, ?, ?, ?, ?, ?)",
                zip(
                    _timestamp_text(truth["timestamp"]),
                    truth["event"].tolist(),
                    truth["low_pressure"].astype(int).tolist(),
                    truth["flow_rate_m3_s"].tolist(),
                    truth["inlet_pressure_pa"].tolist(),
                    truth["inlet_temperature_k"].tolist(),
                ),
            )


def _read_query(path: Path, query: str, parameters: list) -> pd.DataFrame:
    """Run a query on a read-only connection and return the rows.

    Args:
        path: Location of the database file.
        query: SQL, using ? for each value.
        parameters: The values for the ? placeholders.

    Returns:
        The rows. Any timestamp column is converted to UTC datetimes.

    Raises:
        FileNotFoundError: If the database does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"no database at {path}")

    # mode=ro stops SQLite creating an empty file or changing anything.
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = pd.read_sql_query(query, connection, params=parameters)

    if "timestamp" in rows:
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    return rows


def _time_filter(
    start: str | datetime | None, end: str | datetime | None
) -> tuple[str, list[str]]:
    """Return the SQL and values for an optional time range.

    Only fixed SQL text is added here. The times themselves are always passed
    as ? values.
    """
    clauses = ""
    values = []
    if start is not None:
        clauses += " AND timestamp >= ?"
        values.append(_to_database_time(start))
    if end is not None:
        clauses += " AND timestamp < ?"
        values.append(_to_database_time(end))
    return clauses, values


def _to_database_time(moment: str | datetime) -> str:
    """Convert a time to stored UTC text. A time with no zone is taken as UTC."""
    timestamp = pd.Timestamp(moment)
    if timestamp.tzinfo is None:
        return timestamp.strftime(TIMESTAMP_FORMAT)
    return timestamp.tz_convert("UTC").strftime(TIMESTAMP_FORMAT)


def _require_known_sensor(path: Path, sensor_id: str) -> None:
    """Raise ValueError unless the sensor is in the database."""
    query = "SELECT sensor_id FROM sensors WHERE sensor_id = ?"
    if _read_query(path, query, [sensor_id]).empty:
        raise ValueError(f"unknown sensor {sensor_id}")


def list_sensors(path: Path) -> pd.DataFrame:
    """Return the sensor network as stored in the database."""
    return _read_query(path, "SELECT * FROM sensors ORDER BY sensor_id", [])


def load_readings(
    path: Path,
    sensor_id: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pd.DataFrame:
    """Return one sensor's readings, oldest first.

    Args:
        path: Location of the database file.
        sensor_id: The sensor.
        start: Earliest time to include. Leave out for no limit.
        end: Time to stop before. This moment itself is not included.

    Returns:
        Columns timestamp, value and unit. value is empty for missing readings.

    Raises:
        ValueError: If the sensor is not in the database.
    """
    _require_known_sensor(path, sensor_id)
    clauses, values = _time_filter(start, end)
    query = (
        "SELECT timestamp, value, unit FROM sensor_readings "
        f"WHERE sensor_id = ?{clauses} ORDER BY timestamp"
    )
    return _read_query(path, query, [sensor_id, *values])


def load_ground_truth(
    path: Path,
    sensor_id: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pd.DataFrame:
    """Return one sensor's true values and fault labels, oldest first.

    These are simulation answers. See the note at the top of this file.

    Args:
        path: Location of the database file.
        sensor_id: The sensor.
        start: Earliest time to include. Leave out for no limit.
        end: Time to stop before. This moment itself is not included.

    Returns:
        Columns timestamp, true_value and fault.

    Raises:
        ValueError: If the sensor is not in the database.
    """
    _require_known_sensor(path, sensor_id)
    clauses, values = _time_filter(start, end)
    query = (
        "SELECT timestamp, true_value, fault FROM ground_truth "
        f"WHERE sensor_id = ?{clauses} ORDER BY timestamp"
    )
    return _read_query(path, query, [sensor_id, *values])


def load_operating_states(
    path: Path,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pd.DataFrame:
    """Return the true operating conditions, oldest first.

    These are simulation answers. See the note at the top of this file.

    Args:
        path: Location of the database file.
        start: Earliest time to include. Leave out for no limit.
        end: Time to stop before. This moment itself is not included.

    Returns:
        One row per minute.
    """
    clauses, values = _time_filter(start, end)
    query = f"SELECT * FROM operating_states WHERE 1 = 1{clauses} ORDER BY timestamp"
    states = _read_query(path, query, values)
    states["low_pressure"] = states["low_pressure"].astype(bool)
    return states
