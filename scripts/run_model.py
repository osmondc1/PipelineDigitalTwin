"""Run the steady-state pipeline model on the base case.

This prints a report and saves a pressure profile chart. Run it from the
project folder with:
    .venv\\Scripts\\python scripts\\run_model.py
"""

from pathlib import Path

from pipeline.config_loader import load_case
from pipeline.pipeline_model import calculate_pipeline
from pipeline.plotting import plot_pressure_profile
from pipeline.report import format_results

# Build paths from this file's location, so the script works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CASE_PATH = PROJECT_ROOT / "config" / "base_case.toml"
CHART_PATH = PROJECT_ROOT / "outputs" / "pressure_profile.png"


def main() -> None:
    """Load the base case, run the model, then print and plot the results."""
    case = load_case(BASE_CASE_PATH)
    result = calculate_pipeline(
        case.fluid, case.flow_rate, case.inlet_pressure, case.segments, case.pumps
    )
    print(format_results(case, result))

    CHART_PATH.parent.mkdir(exist_ok=True)
    plot_pressure_profile(case, result).savefig(CHART_PATH, dpi=150)
    print(f"\nChart saved to {CHART_PATH}")


# Only run main() when this file is run directly, not when it is imported.
if __name__ == "__main__":
    main()
