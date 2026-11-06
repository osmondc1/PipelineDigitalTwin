"""Chart the stored sensor history and export the readings to CSV.

Run scripts\\generate_data.py first to create the database. All of the data is
SYNTHETIC. Run this from the project folder with:
    .venv\\Scripts\\python scripts\\review_history.py
"""

from pathlib import Path

from sensors.history import export_readings_csv, plot_history

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "simulated" / "pipeline.db"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"


def main(
    database_path: Path = DATABASE_PATH, output_folder: Path = OUTPUT_FOLDER
) -> None:
    """Save the history chart and the CSV export."""
    output_folder.mkdir(parents=True, exist_ok=True)
    chart_path = output_folder / "sensor_history.png"
    csv_path = output_folder / "synthetic_sensor_readings.csv"

    plot_history(database_path).savefig(chart_path, dpi=150)
    rows = export_readings_csv(database_path, csv_path)

    print(f"Chart saved to {chart_path}")
    print(f"{rows:,} rows exported to {csv_path}")


if __name__ == "__main__":
    main()
