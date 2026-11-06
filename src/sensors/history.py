"""Review stored sensor data: a history chart and a CSV export.

All data is SYNTHETIC. This module only reads the database.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure

from sensors.sensor import UNITS
from sensors.storage import list_sensors, load_operating_states, load_readings

NORMAL_LABEL = "normal"
CSV_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Panels, top to bottom.
QUANTITIES = ("pressure", "flow", "temperature")

SURFACE_COLOR = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
EVENT_COLOR = "#ecebe6"

# In fixed order: inlet, intermediate, outlet. A location keeps its colour in
# every panel, so a colour always means the same place on the line.
LOCATION_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def export_readings_csv(
    database_path: Path,
    csv_path: Path,
    sensor_ids: list[str] | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> int:
    """Save sensor readings to a CSV file.

    The file contains monitoring data only, never the simulation's answers.
    Missing readings keep their row, with an empty value.

    Args:
        database_path: Location of the database file.
        csv_path: Where to write the CSV file.
        sensor_ids: Sensors to include. Leave out for all of them.
        start: Earliest time to include. Leave out for no limit.
        end: Time to stop before. This moment itself is not included.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If a sensor is not in the database.
    """
    if sensor_ids is None:
        sensor_ids = list_sensors(database_path)["sensor_id"].tolist()

    tables = [
        load_readings(database_path, sensor_id, start, end).assign(sensor_id=sensor_id)
        for sensor_id in sensor_ids
    ]
    readings = pd.concat(tables)[["timestamp", "sensor_id", "value", "unit"]]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    readings.to_csv(csv_path, index=False, date_format=CSV_TIME_FORMAT)
    return len(readings)


def event_periods(states: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Return each run of non-normal operation.

    Args:
        states: Operating states, with timestamp and event columns.

    Returns:
        A (label, first timestamp, last timestamp) entry per event, in order.
    """
    # Each time the label changes, the running total moves on to a new group.
    run_numbers = states["event"].ne(states["event"].shift()).cumsum()

    periods = []
    for _, run in states.groupby(run_numbers):
        label = run["event"].iloc[0]
        if label != NORMAL_LABEL:
            periods.append((label, run["timestamp"].iloc[0], run["timestamp"].iloc[-1]))
    return periods


def plot_history(
    database_path: Path,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> Figure:
    """Draw every sensor's readings over time.

    There is one panel per measured quantity, because they have different
    units. Gaps stay as gaps. Scripted events are shaded, but sensor faults are
    not marked, so this is close to what a control room would see.

    Args:
        database_path: Location of the database file.
        start: Earliest time to include. Leave out for no limit.
        end: Time to stop before. This moment itself is not included.

    Returns:
        The finished figure. Save it with figure.savefig(path).
    """
    sensors = list_sensors(database_path)
    locations = sorted(sensors["location_km"].unique())
    colors = dict(zip(locations, LOCATION_COLORS))
    events = event_periods(load_operating_states(database_path, start, end))

    figure = Figure(figsize=(12, 8.5), layout="constrained", facecolor=SURFACE_COLOR)
    panels = figure.subplots(len(QUANTITIES), 1, sharex=True)

    for axes, quantity in zip(panels, QUANTITIES):
        for label, first, last in events:
            axes.axvspan(first, last, color=EVENT_COLOR, linewidth=0)

        chosen = sensors[sensors["quantity"] == quantity]
        for sensor in chosen.itertuples():
            readings = load_readings(database_path, sensor.sensor_id, start, end)
            axes.plot(
                readings["timestamp"],
                readings["value"],
                color=colors[sensor.location_km],
                linewidth=0.8,
                label=f"{sensor.sensor_id} ({sensor.location_km:.0f} km)",
            )

        axes.set_ylabel(f"{quantity.capitalize()} ({UNITS[quantity]})", color=MUTED_TEXT_COLOR)
        legend = axes.legend(
            loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9
        )
        for text in legend.get_texts():
            text.set_color(MUTED_TEXT_COLOR)
        _apply_style(axes)

    _label_events(panels[0], events)
    panels[0].set_title(
        "Simulated sensor readings (synthetic data)",
        loc="left", color=TEXT_COLOR, fontsize=13, pad=24,
    )
    panels[-1].margins(x=0)  # Start and end the time axis at the data.
    panels[-1].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    panels[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    panels[-1].set_xlabel("Date (UTC)", color=MUTED_TEXT_COLOR)
    return figure


def _label_events(axes, events: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> None:
    """Name each event above the top panel, centred on its shaded band."""
    for label, first, last in events:
        middle = first + (last - first) / 2
        axes.text(
            middle, 1.01, label.replace("_", " "),
            transform=axes.get_xaxis_transform(),
            color=MUTED_TEXT_COLOR, fontsize=9, ha="center", va="bottom",
        )


def _apply_style(axes) -> None:
    """Keep the grid and axes quiet so that the data stands out."""
    axes.set_facecolor(SURFACE_COLOR)
    axes.grid(color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED_TEXT_COLOR, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID_COLOR)
