"""Generate the simulated data and print a summary.

The data is kept in memory for now. All of it is SYNTHETIC. Run it from the
project folder with:
    .venv\\Scripts\\python scripts\\generate_data.py
"""

from pathlib import Path

from pipeline.config_loader import load_case
from sensors.faults import load_faults
from sensors.readings import NO_FAULT, generate_sensor_readings
from sensors.sensor import load_sensors
from simulation.operations import generate_operating_data, load_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FOLDER = PROJECT_ROOT / "config"


def main() -> None:
    """Generate the data and print a summary."""
    case = load_case(CONFIG_FOLDER / "base_case.toml")
    sensors = load_sensors(CONFIG_FOLDER / "sensors.toml")
    plan = load_plan(CONFIG_FOLDER / "simulation.toml")
    faults = load_faults(CONFIG_FOLDER / "simulation.toml")

    truth = generate_operating_data(case, sensors, plan)
    sensor_data = generate_sensor_readings(truth, sensors, faults, plan.seed)

    print(f"Generated {len(truth):,} time steps of synthetic data")
    print(f"From {truth['timestamp'].iloc[0]} to {truth['timestamp'].iloc[-1]}")
    print(f"Steps with low pressure: {truth['low_pressure'].sum()}\n")

    sensor_ids = [sensor.sensor_id for sensor in sensors]
    summary = truth.groupby("event")[sensor_ids].mean().round(2)
    summary.insert(0, "steps", truth.groupby("event").size())
    print("Average true value by operating state:")
    print(summary.T.to_string())

    readings = sensor_data.readings[sensor_ids]
    labels = sensor_data.fault_labels[sensor_ids]
    print(f"\nExpected readings: {readings.size:,}")
    print(f"Recorded readings: {readings.count().sum():,}")
    print(f"Readings with no value: {readings.isna().sum().sum():,}\n")

    print("Sensor faults (minutes affected):")
    for sensor_id in sensor_ids:
        for kind, minutes in labels[sensor_id].value_counts().items():
            if kind != NO_FAULT:
                print(f"  {sensor_id}: {kind}, {minutes:,}")


if __name__ == "__main__":
    main()
