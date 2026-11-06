"""Tests for the history chart and CSV export."""

import math
import runpy
from pathlib import Path

import pandas as pd
import pytest

from sensors.history import event_periods, export_readings_csv, plot_history
from sensors.readings import generate_sensor_readings
from sensors.sensor import Sensor
from sensors.storage import save_simulation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 12345
STEPS = 120  # Two hours, one row per minute.

# Sensor ID, quantity and location: two at the inlet and one at the outlet.
NETWORK = (
    ("PT-001", "pressure", 0.0),
    ("PT-003", "pressure", 100.0),
    ("FT-001", "flow", 0.0),
    ("TT-001", "temperature", 0.0),
)


def make_sensor(sensor_id: str, quantity: str, location_km: float) -> Sensor:
    """Build a sensor with noise and some missing readings."""
    return Sensor(
        sensor_id=sensor_id,
        quantity=quantity,
        location="Somewhere",
        location_km=location_km,
        sampling_interval=60.0,
        noise_std=0.5,
        bias_std=0.0,
        missing_probability=0.1,
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """A small saved simulation with one event, in the middle hour's first half."""
    path = tmp_path / "test.db"
    sensors = tuple(make_sensor(*row) for row in NETWORK)
    columns = {
        "timestamp": pd.date_range("2026-01-01", periods=STEPS, freq="60s", tz="UTC"),
        "event": ["normal"] * 30 + ["ps2_trip"] * 30 + ["normal"] * 60,
        "low_pressure": False,
        "flow_rate_m3_s": 0.25,
        "inlet_pressure_pa": 300_000.0,
        "inlet_temperature_k": 313.15,
    }
    truth = pd.DataFrame(columns | {sensor.sensor_id: 50.0 for sensor in sensors})
    sensor_data = generate_sensor_readings(truth, sensors, (), SEED)
    save_simulation(path, sensors, truth, sensor_data)
    return path


def make_states(*labels: str) -> pd.DataFrame:
    """Build operating states with one row per label, a minute apart."""
    timestamps = pd.date_range("2026-01-01", periods=len(labels), freq="60s", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "event": labels})


# --- event_periods ---


def test_normal_operation_has_no_events() -> None:
    assert event_periods(make_states("normal", "normal")) == []


def test_event_period_has_its_first_and_last_times() -> None:
    states = make_states("normal", "trip", "trip", "trip", "normal")

    [(label, first, last)] = event_periods(states)

    assert label == "trip"
    assert first == states["timestamp"].iloc[1]
    assert last == states["timestamp"].iloc[3]


def test_two_separate_events_stay_separate() -> None:
    states = make_states("trip", "normal", "trip")

    assert len(event_periods(states)) == 2


def test_neighbouring_events_are_told_apart() -> None:
    states = make_states("demand", "trip")

    assert [period[0] for period in event_periods(states)] == ["demand", "trip"]


def test_event_at_the_end_of_the_data_is_included() -> None:
    assert len(event_periods(make_states("normal", "trip"))) == 1


# --- export_readings_csv ---


def test_export_contains_every_reading(database: Path, tmp_path: Path) -> None:
    # Hand check: 4 sensors * 120 minutes = 480 rows.
    csv_path = tmp_path / "out.csv"

    rows = export_readings_csv(database, csv_path)

    assert rows == 480
    assert len(pd.read_csv(csv_path)) == 480


def test_export_contains_no_simulation_answers(
    database: Path, tmp_path: Path
) -> None:
    csv_path = tmp_path / "out.csv"

    export_readings_csv(database, csv_path)

    assert list(pd.read_csv(csv_path).columns) == [
        "timestamp",
        "sensor_id",
        "value",
        "unit",
    ]


def test_export_keeps_gaps_as_empty_values(database: Path, tmp_path: Path) -> None:
    csv_path = tmp_path / "out.csv"

    export_readings_csv(database, csv_path)

    exported = pd.read_csv(csv_path)
    assert exported["value"].isna().mean() == pytest.approx(0.1, abs=0.05)


def test_export_uses_utc_timestamps(database: Path, tmp_path: Path) -> None:
    csv_path = tmp_path / "out.csv"

    export_readings_csv(database, csv_path)

    assert pd.read_csv(csv_path)["timestamp"].iloc[0] == "2026-01-01T00:00:00Z"


def test_export_can_choose_sensors(database: Path, tmp_path: Path) -> None:
    csv_path = tmp_path / "out.csv"

    rows = export_readings_csv(database, csv_path, ["PT-001", "FT-001"])

    assert rows == 240
    assert set(pd.read_csv(csv_path)["sensor_id"]) == {"PT-001", "FT-001"}


def test_export_can_choose_a_time_range(database: Path, tmp_path: Path) -> None:
    rows = export_readings_csv(
        database,
        tmp_path / "out.csv",
        ["PT-001"],
        "2026-01-01 00:10",
        "2026-01-01 00:20",
    )

    assert rows == 10


def test_export_creates_missing_folders(database: Path, tmp_path: Path) -> None:
    csv_path = tmp_path / "new" / "folder" / "out.csv"

    export_readings_csv(database, csv_path)

    assert csv_path.exists()


def test_export_rejects_an_unknown_sensor(database: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown sensor"):
        export_readings_csv(database, tmp_path / "out.csv", ["PT-999"])


# --- plot_history ---


def test_chart_has_a_panel_per_quantity(database: Path) -> None:
    labels = [axes.get_ylabel() for axes in plot_history(database).axes]

    assert labels == ["Pressure (barg)", "Flow (m3/h)", "Temperature (degC)"]


def test_chart_draws_every_sensor_with_a_legend(database: Path) -> None:
    pressure, flow, temperature = plot_history(database).axes

    assert len(pressure.lines) == 2
    assert len(flow.lines) == 1
    assert len(temperature.lines) == 1
    assert [t.get_text() for t in pressure.get_legend().get_texts()] == [
        "PT-001 (0 km)",
        "PT-003 (100 km)",
    ]


def test_chart_says_the_data_is_synthetic(database: Path) -> None:
    title = plot_history(database).axes[0].get_title(loc="left")

    assert "synthetic" in title


def test_chart_leaves_gaps_empty(database: Path) -> None:
    # Filling a gap would draw a line through a period with no data.
    values = plot_history(database).axes[0].lines[0].get_ydata()

    assert any(math.isnan(value) for value in values)


def test_a_location_keeps_its_colour_across_panels(database: Path) -> None:
    pressure, flow, _ = plot_history(database).axes
    pt_001, pt_003 = pressure.lines

    assert pt_001.get_color() == flow.lines[0].get_color()
    assert pt_001.get_color() != pt_003.get_color()


def test_chart_shades_and_names_events_on_every_panel(database: Path) -> None:
    figure = plot_history(database)

    assert all(len(axes.patches) == 1 for axes in figure.axes)
    assert [text.get_text() for text in figure.axes[0].texts] == ["ps2 trip"]


def test_chart_can_show_a_time_range(database: Path) -> None:
    figure = plot_history(database, "2026-01-01 01:00", "2026-01-01 01:30")

    assert len(figure.axes[0].lines[0].get_ydata()) == 30
    assert len(figure.axes[0].patches) == 0


def test_chart_of_a_missing_database_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        plot_history(tmp_path / "missing.db")


# --- The script ---


def test_review_script_saves_chart_and_csv(database: Path, tmp_path: Path) -> None:
    script = runpy.run_path(str(PROJECT_ROOT / "scripts" / "review_history.py"))
    output_folder = tmp_path / "outputs"

    script["main"](database, output_folder)

    assert (output_folder / "sensor_history.png").stat().st_size > 0
    assert len(pd.read_csv(output_folder / "synthetic_sensor_readings.csv")) == 480
