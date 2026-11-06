"""Generate the simulated operating data and print a summary.

The data is kept in memory for now. All of it is SYNTHETIC. Run it from the
project folder with:
    .venv\\Scripts\\python scripts\\generate_data.py
"""

from pathlib import Path

from pipeline.config_loader import load_case
from sensors.sensor import load_sensors
from simulation.operations import generate_operating_data, load_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FOLDER = PROJECT_ROOT / "config"


def main() -> None:
    """Generate the data and print a summary for each operating state."""
    case = load_case(CONFIG_FOLDER / "base_case.toml")
    sensors = load_sensors(CONFIG_FOLDER / "sensors.toml")
    plan = load_plan(CONFIG_FOLDER / "simulation.toml")

    data = generate_operating_data(case, sensors, plan)

    print(f"Generated {len(data):,} time steps of synthetic data")
    print(f"From {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
    print(f"Steps with low pressure: {data['low_pressure'].sum()}\n")

    sensor_ids = [sensor.sensor_id for sensor in sensors]
    summary = data.groupby("event")[sensor_ids].mean().round(2)
    summary.insert(0, "steps", data.groupby("event").size())
    print("Average true value by operating state:")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
