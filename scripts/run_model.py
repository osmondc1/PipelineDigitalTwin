"""Run the steady-state pipeline model on the base case and print the results.

Run it from the project folder with:
    .venv\\Scripts\\python scripts\\run_model.py
"""

from pathlib import Path

from pipeline.config_loader import load_case
from pipeline.pipeline_model import calculate_pipeline
from pipeline.report import format_results

# Build the path from this file's location, so the script works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CASE_PATH = PROJECT_ROOT / "config" / "base_case.toml"


def main() -> None:
    """Load the base case, run the model and print the report."""
    case = load_case(BASE_CASE_PATH)
    result = calculate_pipeline(
        case.fluid, case.flow_rate, case.inlet_pressure, case.segments, case.pumps
    )
    print(format_results(case, result))


# Only run main() when this file is run directly, not when it is imported.
if __name__ == "__main__":
    main()
