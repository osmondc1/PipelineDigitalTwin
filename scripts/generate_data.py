"""Generate the simulated data, save it to SQLite and print a summary.

Running this replaces data/simulated/pipeline.db. The file is not kept in Git,
because this script recreates it exactly. All of the data is SYNTHETIC. Run it
from the project folder with:
    .venv\\Scripts\\python scripts\\generate_data.py
"""

from pathlib import Path

from pipeline.config_loader import load_case
from sensors.faults import load_faults
from sensors.readings import NO_FAULT, generate_sensor_readings
from sensors.sensor import load_sensors
from sensors.storage import save_simulation
from simulation.operations import generate_operating_data, load_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FOLDER = PROJECT_ROOT / "config"
DATABASE_PATH = PROJECT_ROOT / "data" / "simulated" / "pipeline.db"


def main(database_path: Path = DATABASE_PATH) -> None:
    """Generate the data, save it and print a summary."""
    case = load_case(CONFIG_FOLDER / "base_case.toml")
    sensors = load_sensors(CONFIG_FOLDER / "sensors.toml")
    plan = load_plan(CONFIG_FOLDER / "simulation.toml")
    faults = load_faults(CONFIG_FOLDER / "simulation.toml")

    truth = generate_operating_data(case, sensors, plan)
    sensor_data = generate_sensor_readings(truth, sensors, faults, plan.seed)
    save_simulation(database_path, sensors, truth, sensor_data)

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

    size_mb = database_path.stat().st_size / 1_000_000
    print(f"\nSaved to {database_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
